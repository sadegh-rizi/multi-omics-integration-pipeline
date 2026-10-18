"""Simulation calibration for exploratory outlier scores."""

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
    """Simulate null scores using covariance from the selected calibration rows."""
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
    """Convert increasing anomaly scores to their empirical null percentiles."""
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
    """
    matrix, covariance, mean = _calibration_inputs(values, calibration_values)

    def draw_scores(seed: int) -> dict[str, list[np.ndarray]]:
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
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    factor = eigenvectors * np.sqrt(np.clip(eigenvalues, 0, None))

    def scores_from_draws(seed: int) -> dict[int, list[np.ndarray]]:
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
    """Calibrate the seed-averaged Isolation Forest using batched simulations."""
    matrix, covariance, mean = _calibration_inputs(values, calibration_values)
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
    matrix, covariance, mean = _calibration_inputs(values, calibration_values)
    k_values = tuple(sorted(set(k_values)))
    if not k_values or min(k_values) < 1 or max(k_values) >= len(matrix):
        raise ValueError("LOF neighbours must be between 1 and n_samples minus one.")
    null_score_columns = _batched_null_scores(
        matrix,
        mean,
        covariance,
        lambda simulations: local_outlier_factor_sweep_array(simulations, k_values=k_values),
        n_replicates=n_replicates,
        random_state=random_state,
        batch_size=batch_size,
    )
    null_scores = dict(zip(k_values, null_score_columns))
    limits = pd.DataFrame({
        "limit_95": {k: null_scores[k].quantile(0.95) for k in k_values},
        "limit_99": {k: null_scores[k].quantile(0.99) for k in k_values},
    })
    limits.index.name = "k"
    verification_columns = _batched_null_scores(
        matrix,
        mean,
        covariance,
        lambda simulations: local_outlier_factor_sweep_array(simulations, k_values=k_values),
        n_replicates=n_replicates,
        random_state=random_state + 1,
        batch_size=batch_size,
    )
    limits["false_positive_rate"] = {
        k: float(np.mean(scores.to_numpy() > limits.loc[k, "limit_95"]))
        for k, scores in zip(k_values, verification_columns)
    }
    return limits, null_scores


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
    matrix, covariance, mean = _calibration_inputs(values, calibration_values)
    shrinkages = tuple(float(alpha) for alpha in shrinkages)
    if not shrinkages or min(shrinkages) < 0 or max(shrinkages) > 1:
        raise ValueError("Ledoit-Wolf shrinkage intensities must be in [0, 1].")
    null_score_columns = _batched_null_scores(
        matrix,
        mean,
        covariance,
        lambda simulations: ledoit_wolf_mahalanobis_sweep_array(simulations, shrinkages=shrinkages),
        n_replicates=n_replicates,
        random_state=random_state,
        batch_size=batch_size,
    )
    null_scores = dict(zip(shrinkages, null_score_columns))
    limits = pd.DataFrame({
        "limit_95": {alpha: null_scores[alpha].quantile(0.95) for alpha in shrinkages},
        "limit_99": {alpha: null_scores[alpha].quantile(0.99) for alpha in shrinkages},
    })
    limits.index.name = "shrinkage"
    verification_columns = _batched_null_scores(
        matrix,
        mean,
        covariance,
        lambda simulations: ledoit_wolf_mahalanobis_sweep_array(simulations, shrinkages=shrinkages),
        n_replicates=n_replicates,
        random_state=random_state + 1,
        batch_size=batch_size,
    )
    limits["false_positive_rate"] = {
        alpha: float(np.mean(scores.to_numpy() > limits.loc[alpha, "limit_95"]))
        for alpha, scores in zip(shrinkages, verification_columns)
    }
    return limits, null_scores


def _calibration_inputs(
    values: pd.DataFrame,
    calibration_values: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return target shape plus Gaussian parameters from calibration rows."""
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
    """Generate Gaussian null matrices in batches and collect score columns."""
    if n_replicates < 1 or batch_size < 1:
        raise ValueError("n_replicates and batch_size must be positive.")
    rng = np.random.default_rng(random_state)
    collected: list[np.ndarray] | None = None
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
        if collected is None:
            collected = [[] for _ in range(batch_scores.shape[2])]
        for column in range(batch_scores.shape[2]):
            collected[column].append(batch_scores[:, :, column].ravel())
    if collected is None:
        raise ValueError("No null scores were generated.")
    return [pd.Series(np.concatenate(scores), name="null_score") for scores in collected]
