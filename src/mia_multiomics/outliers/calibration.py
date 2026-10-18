"""Simulation calibration for outlier scores.

An outlier score (a distance, an isolation-forest score, ...) has no natural
cut-off, and with 20 samples in ~100 dimensions any fixed rule flags far too
many or too few.  So every detector setting is calibrated on data that contain
no outliers by construction:

1. estimate the mean and covariance of the real samples;
2. simulate many multivariate-normal data sets of the same size from them;
3. score every simulated sample - these scores are the null distribution;
4. the 95 % / 99 % quantiles of the null are the limits; a real sample above
   the 95 % limit is flagged;
5. check the limit on a *second*, independent set of simulations: the share of
   null samples above it is the achieved false-positive rate (target 5 %).

The stage runs this twice ("two-stage calibration", ``stages/outliers.py``):
first with all samples, then with the flagged samples left out of the
covariance estimate, so that an outlier cannot widen its own null.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from .detectors import (
    isolation_forest_null_scores,
    ledoit_wolf_mahalanobis_sweep_array,
    local_outlier_factor_sweep_array,
)

ScoreFunction = Callable[[pd.DataFrame], pd.Series]


def simulate_null_limits(
    values: pd.DataFrame,
    score_function: ScoreFunction,
    *,
    calibration_values: pd.DataFrame | None = None,
    n_replicates: int = 2_000,
    random_state: int = 20_260_826,
) -> tuple[pd.Series, pd.Series]:
    """Simulate null scores using covariance from the selected calibration rows.

    Generic version for any ``score_function``: each replicate is a fresh
    normal data set of the same shape as ``values``.  Returns the 95/99 % limits
    and all null scores.
    """
    matrix, covariance, mean = _calibration_inputs(values, calibration_values)
    rng = np.random.default_rng(random_state)
    draws: list[np.ndarray] = []
    for _ in range(n_replicates):
        simulated = rng.multivariate_normal(mean, covariance, size=len(matrix), method="eigh")
        scores = score_function(pd.DataFrame(simulated, index=values.index, columns=values.columns))
        draws.append(scores.to_numpy(dtype=float))
    null_scores = pd.Series(np.concatenate(draws), name="null_score")
    limits = pd.Series({"limit_95": null_scores.quantile(0.95), "limit_99": null_scores.quantile(0.99)})
    return limits, null_scores


def score_to_percentile(scores: pd.Series, null_scores: pd.Series) -> pd.Series:
    """Convert increasing anomaly scores to their empirical null percentiles (0.97 = above 97 % of null scores)."""
    reference = np.sort(pd.to_numeric(null_scores, errors="raise").to_numpy(dtype=float))
    if len(reference) == 0:
        raise ValueError("Null score reference cannot be empty.")
    positions = np.searchsorted(reference, scores.to_numpy(dtype=float), side="right")
    return pd.Series(positions / len(reference), index=scores.index, name="null_percentile")


def calibrate_ensemble(
    values: pd.DataFrame,
    score_functions: dict[str, ScoreFunction],
    *,
    calibration_values: pd.DataFrame | None = None,
    n_replicates: int = 2_000,
    random_state: int = 20_260_828,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate correlated percentile combinations on independent null draws.

    Each simulated matrix is scored by every detector before its detector-wise
    scores are converted to null percentiles.  This preserves their dependence
    when calibrating the trinity and five-detector combinations.

    Not used by the pipeline stages (the ensemble there uses the per-detector
    stability rule); kept, with its test, as the alternative rule compared in the thesis.
    """
    matrix, covariance, mean = _calibration_inputs(values, calibration_values)

    def draw_scores(seed: int) -> dict[str, list[np.ndarray]]:
        """Score n_replicates simulated data sets with every detector."""
        rng = np.random.default_rng(seed)
        draws = {name: [] for name in score_functions}
        for _ in range(n_replicates):
            simulated = rng.multivariate_normal(mean, covariance, size=len(matrix), method="eigh")
            frame = pd.DataFrame(simulated, index=values.index, columns=values.columns)
            for name, scorer in score_functions.items():
                draws[name].append(scorer(frame).to_numpy(dtype=float))
        return draws

    calibration = draw_scores(random_state)
    references = {name: np.sort(np.concatenate(draws)) for name, draws in calibration.items()}

    def combinations(draws: dict[str, list[np.ndarray]]) -> pd.DataFrame:
        """Turn each detector's scores into null percentiles, then combine them."""
        components = {
            name: np.concatenate([np.searchsorted(references[name], score, side="right") / len(references[name]) for score in scores])
            for name, scores in draws.items()
        }
        table = pd.DataFrame(components)
        return pd.DataFrame({
            "trinity_mean": table.loc[:, ["mahalanobis", "knn", "isolation_forest"]].mean(axis=1),
            "trinity_max": table.loc[:, ["mahalanobis", "knn", "isolation_forest"]].max(axis=1),
            "five_detector_mean": table.mean(axis=1),
        })

    calibration_combinations = combinations(calibration)
    limits = calibration_combinations.quantile([0.95, 0.99]).T.rename(columns={0.95: "limit_95", 0.99: "limit_99"})
    verification = combinations(draw_scores(random_state + 1))
    rates = pd.DataFrame({
        "false_positive_rate": {name: verification[name].gt(limits.loc[name, "limit_95"]).mean() for name in limits.index}
    })
    return limits, rates


def calibrate_knn_sweep(
    values: pd.DataFrame,
    *,
    k_values: list[int],
    calibration_values: pd.DataFrame | None = None,
    n_replicates: int = 2_000,
    random_state: int = 20_260_826,
    batch_size: int = 100,
) -> tuple[pd.DataFrame, dict[int, pd.Series]]:
    """Calibrate all k-NN neighbourhood sizes from shared null simulations.

    The neighbour ordering is calculated once per simulated matrix, then its
    cumulative mean provides every requested ``k``.  This is mathematically
    identical to fitting each k separately but avoids repeating the expensive
    pairwise-distance calculation for every parameter value.
    """
    matrix, covariance, mean = _calibration_inputs(values, calibration_values)
    n_samples = len(matrix)
    if not k_values or min(k_values) < 1 or max(k_values) >= n_samples:
        raise ValueError("k values must be between 1 and the number of samples minus one.")
    k_values = sorted(set(k_values))
    # Square-root factor of the covariance: standard-normal draws @ factor.T have that covariance.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    factor = eigenvectors * np.sqrt(np.clip(eigenvalues, 0, None))

    def scores_from_draws(seed: int) -> dict[int, list[np.ndarray]]:
        """kNN scores (mean distance to the k nearest samples) for every k, in batches of simulations."""
        rng = np.random.default_rng(seed)
        collected = {k: [] for k in k_values}
        for start in range(0, n_replicates, batch_size):
            size = min(batch_size, n_replicates - start)
            draws = rng.standard_normal((size, n_samples, matrix.shape[1])) @ factor.T + mean
            distances = np.sqrt(((draws[:, :, None, :] - draws[:, None, :, :]) ** 2).sum(axis=3))
            indices = np.arange(n_samples)
            distances[:, indices, indices] = np.inf
            ordered = np.sort(distances, axis=2)
            cumulative = np.cumsum(ordered, axis=2)
            for k in k_values:
                collected[k].append((cumulative[:, :, k - 1] / k).ravel())
        return collected

    # Limits from one set of simulations, false-positive rate checked on a second set.
    calibration = scores_from_draws(random_state)
    null_scores = {k: pd.Series(np.concatenate(draws), name="null_score") for k, draws in calibration.items()}
    limits = pd.DataFrame({
        "limit_95": {k: scores.quantile(0.95) for k, scores in null_scores.items()},
        "limit_99": {k: scores.quantile(0.99) for k, scores in null_scores.items()},
    })
    limits.index.name = "k"
    verification = scores_from_draws(random_state + 1)
    limits["false_positive_rate"] = {
        k: float(np.mean(np.concatenate(scores).astype(float) > limits.loc[k, "limit_95"]))
        for k, scores in verification.items()
    }
    return limits, null_scores


def calibrate_isolation_forest(
    values: pd.DataFrame,
    *,
    seeds: Sequence[int],
    calibration_values: pd.DataFrame | None = None,
    n_estimators: int = 500,
    n_replicates: int = 2_000,
    random_state: int = 20_260_826,
    n_jobs: int = -1,
) -> tuple[pd.Series, pd.Series, float]:
    """Calibrate the seed-averaged Isolation Forest using batched simulations.

    Returns the 95/99 % limits, the null scores and the false-positive rate on
    an independent second batch of simulations.
    """
    matrix, covariance, mean = _calibration_inputs(values, calibration_values)
    # Null scores: isolation-forest scores of n_replicates simulated data sets at once.
    rng = np.random.default_rng(random_state)
    simulations = rng.multivariate_normal(
        mean, covariance, size=(n_replicates, len(matrix)), method="eigh"
    )
    null_matrix_scores = isolation_forest_null_scores(
        simulations,
        seeds=seeds,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
    )
    null_scores = pd.Series(null_matrix_scores.ravel(), name="null_score")
    limits = pd.Series({
        "limit_95": null_scores.quantile(0.95),
        "limit_99": null_scores.quantile(0.99),
    })
    # Independent second batch: the share above the 95 % limit is the false-positive rate.
    verification_simulations = rng.multivariate_normal(
        mean, covariance, size=(n_replicates, len(matrix)), method="eigh"
    )
    verification_scores = isolation_forest_null_scores(
        verification_simulations,
        seeds=seeds,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
    )
    false_positive_rate = float(np.mean(verification_scores > limits["limit_95"]))
    return limits, null_scores, false_positive_rate


def calibrate_lof_sweep(
    values: pd.DataFrame,
    *,
    k_values: Sequence[int],
    calibration_values: pd.DataFrame | None = None,
    n_replicates: int = 2_000,
    random_state: int = 20_260_826,
    batch_size: int = 100,
) -> tuple[pd.DataFrame, dict[int, pd.Series]]:
    """Calibrate every LOF k from shared, batched null simulations."""
    k_values = tuple(sorted(set(k_values)))
    if not k_values or min(k_values) < 1 or max(k_values) >= len(values):
        raise ValueError("LOF neighbours must be between 1 and n_samples minus one.")
    return _calibrate_sweep(
        values, calibration_values, k_values,
        lambda simulations: local_outlier_factor_sweep_array(simulations, k_values=k_values),
        index_name="k", n_replicates=n_replicates, random_state=random_state, batch_size=batch_size)


def calibrate_ledoit_wolf_sweep(
    values: pd.DataFrame,
    *,
    shrinkages: Sequence[float],
    calibration_values: pd.DataFrame | None = None,
    n_replicates: int = 2_000,
    random_state: int = 20_260_826,
    batch_size: int = 16,
) -> tuple[pd.DataFrame, dict[float, pd.Series]]:
    """Calibrate all shrinkage intensities without p-by-p matrix inverses."""
    shrinkages = tuple(float(alpha) for alpha in shrinkages)
    if not shrinkages or min(shrinkages) < 0 or max(shrinkages) > 1:
        raise ValueError("Ledoit-Wolf shrinkage intensities must be in [0, 1].")
    return _calibrate_sweep(
        values, calibration_values, shrinkages,
        lambda simulations: ledoit_wolf_mahalanobis_sweep_array(simulations, shrinkages=shrinkages),
        index_name="shrinkage", n_replicates=n_replicates, random_state=random_state, batch_size=batch_size)


def _calibrate_sweep(
    values: pd.DataFrame,
    calibration_values: pd.DataFrame | None,
    parameters: Sequence,
    scorer: Callable[[np.ndarray], np.ndarray],
    *,
    index_name: str,
    n_replicates: int,
    random_state: int,
    batch_size: int,
) -> tuple[pd.DataFrame, dict]:
    """Shared recipe for a detector with several parameter values (one score column each).

    Null scores from one batch of simulations give the 95/99 % limit of every
    parameter value; an independent second batch gives its false-positive rate.
    """
    matrix, covariance, mean = _calibration_inputs(values, calibration_values)
    options = dict(n_replicates=n_replicates, batch_size=batch_size)
    null_scores = dict(zip(parameters, _batched_null_scores(matrix, mean, covariance, scorer,
                                                           random_state=random_state, **options)))
    limits = pd.DataFrame({
        "limit_95": {value: null_scores[value].quantile(0.95) for value in parameters},
        "limit_99": {value: null_scores[value].quantile(0.99) for value in parameters},
    })
    limits.index.name = index_name
    verification = _batched_null_scores(matrix, mean, covariance, scorer, random_state=random_state + 1, **options)
    limits["false_positive_rate"] = {
        value: float(np.mean(scores.to_numpy() > limits.loc[value, "limit_95"]))
        for value, scores in zip(parameters, verification)
    }
    return limits, null_scores


def _calibration_inputs(
    values: pd.DataFrame,
    calibration_values: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return target shape plus Gaussian parameters from calibration rows.

    ``values`` sets the size of each simulated data set; the mean and covariance
    come from ``calibration_values`` (default: the same rows), which lets the
    second calibration stage leave flagged samples out of the covariance.
    """
    target = values.apply(pd.to_numeric, errors="coerce")
    calibration = target if calibration_values is None else calibration_values.apply(pd.to_numeric, errors="coerce")
    if target.empty or target.isna().to_numpy().any() or not np.isfinite(target.to_numpy()).all():
        raise ValueError("Calibration requires a complete finite target matrix.")
    if calibration.empty or calibration.isna().to_numpy().any() or not np.isfinite(calibration.to_numpy()).all():
        raise ValueError("Calibration requires a complete finite calibration matrix.")
    if target.shape[1] != calibration.shape[1]:
        raise ValueError("Target and calibration matrices must have the same feature columns.")
    if len(calibration) < 2:
        raise ValueError("At least two retained samples are required for covariance calibration.")
    matrix = target.to_numpy(dtype=float)
    calibration_matrix = calibration.to_numpy(dtype=float)
    return matrix, np.cov(calibration_matrix, rowvar=False), calibration_matrix.mean(axis=0)


def _batched_null_scores(
    matrix: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    scorer: Callable[[np.ndarray], np.ndarray],
    *,
    n_replicates: int,
    random_state: int,
    batch_size: int,
) -> list[pd.Series]:
    """Generate Gaussian null matrices in batches and collect score columns.

    ``scorer`` takes a (replicates x samples x features) array and returns
    (replicates x samples x parameters) scores; one Series per parameter is returned.
    """
    if n_replicates < 1 or batch_size < 1:
        raise ValueError("n_replicates and batch_size must be positive.")
    rng = np.random.default_rng(random_state)
    collected: list[np.ndarray] | None = None
    # Simulate in batches to bound memory: each batch is (size x samples x features).
    for start in range(0, n_replicates, batch_size):
        size = min(batch_size, n_replicates - start)
        simulations = rng.multivariate_normal(
            mean,
            covariance,
            size=(size, len(matrix)),
            method="eigh",
        )
        batch_scores = scorer(simulations)
        if batch_scores.ndim != 3:
            raise ValueError("Batched detector scores must have shape (replicate, sample, parameter).")
        # Collect the scores of every parameter value separately, flattened over replicates and samples.
        if collected is None:
            collected = [[] for _ in range(batch_scores.shape[2])]
        for column in range(batch_scores.shape[2]):
            collected[column].append(batch_scores[:, :, column].ravel())
    if collected is None:
        raise ValueError("No null scores were generated.")
    return [pd.Series(np.concatenate(scores), name="null_score") for scores in collected]
