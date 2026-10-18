"""Exploratory PLS-DA scores and their permutation test.

PLS-DA finds the directions that best *separate the given labels*, so its score
plot shows separated groups even when the groups do not differ.  The only valid
reading is through a permutation test: refit on shuffled labels many times and
ask whether the real separation is larger than the shuffled ones
(:func:`permutation_test_plsda`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.cross_decomposition import PLSRegression


def _align_series(values: pd.DataFrame, series: pd.Series, name: str) -> pd.Series:
    """``series`` (indexed by sample) reordered to the value columns; no sample may be missing."""
    if values.columns.has_duplicates:
        raise ValueError("PLS-DA sample identifiers must be unique.")
    if series.index.has_duplicates:
        raise ValueError(f"{name} sample identifiers must be unique.")
    missing = values.columns.difference(series.index)
    if len(missing):
        raise ValueError(f"{name} is missing samples: {missing.tolist()}")
    aligned = series.reindex(values.columns)
    if aligned.isna().any():
        raise ValueError(f"{name} contains missing values for plotted samples.")
    return aligned


def _validate_values(
    values: pd.DataFrame,
    labels: pd.Series,
    n_components: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Complete numeric values, a feasible number of components and at least two label groups."""
    numeric = values.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().to_numpy().any():
        raise ValueError("PLS-DA requires complete values; do not impute missing values.")
    if n_components < 1 or n_components > min(numeric.shape[0], numeric.shape[1] - 1):
        raise ValueError("n_components exceeds the available feature or sample rank.")
    aligned_labels = _align_series(numeric, labels, "Labels")
    if aligned_labels.nunique() < 2:
        raise ValueError("PLS-DA requires at least two groups.")
    return numeric, aligned_labels


def _cumulative_x_variance(model: PLSRegression, values: pd.DataFrame) -> np.ndarray:
    """Cumulative share of the X variance explained by the PLS components (removed one at a time)."""
    centered = values.T.to_numpy(dtype=float)
    centered = centered - centered.mean(axis=0, keepdims=True)
    total_sum_squares = float(np.square(centered).sum())
    if total_sum_squares == 0:
        raise ValueError("PLS-DA requires non-zero feature variation.")

    residual = centered.copy()
    cumulative: list[float] = []
    explained = 0.0
    for component in range(model.x_scores_.shape[1]):
        before = float(np.square(residual).sum())
        residual = residual - np.outer(
            model.x_scores_[:, component], model.x_loadings_[:, component]
        )
        explained += max(before - float(np.square(residual).sum()), 0.0) / total_sum_squares
        cumulative.append(min(explained, 1.0))
    return np.asarray(cumulative)


def fit_plsda(
    values: pd.DataFrame,
    labels: pd.Series,
    *,
    n_components: int = 2,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Fit exploratory PLS-DA and return sample scores and cumulative X variance."""
    numeric, aligned_labels = _validate_values(values, labels, n_components)
    # PLS regression of the 0/1 group indicators on the (samples x features) matrix.
    response = pd.get_dummies(aligned_labels, dtype=float)
    model = PLSRegression(n_components=n_components, scale=False)
    model.fit(numeric.T.to_numpy(dtype=float), response.to_numpy(dtype=float))
    scores = pd.DataFrame(
        model.x_scores_,
        index=numeric.columns,
        columns=[f"PLS {component}" for component in range(1, n_components + 1)],
    )
    return scores, _cumulative_x_variance(model, numeric)


def _centroid_separation(scores: pd.DataFrame, labels: pd.Series) -> float:
    """Mean distance between the group centroids in the PLS score space (the test statistic)."""
    centroids = scores.assign(_group=labels.to_numpy()).groupby(
        "_group", observed=True
    ).mean()
    return float(pdist(centroids.to_numpy(dtype=float), metric="euclidean").mean())


def permutation_test_plsda(
    values: pd.DataFrame,
    labels: pd.Series,
    strata: pd.Series,
    *,
    n_components: int = 2,
    n_permutations: int = 1_000,
    random_state: int = 20_260_721,
    exchangeability_unit: str = "strata",
) -> tuple[pd.Series, np.ndarray]:
    """Test PLS-DA separation with declared sample or cage permutations.

    For ``exchangeability_unit='cage'``, ``strata`` must contain one cage label
    per sample and Bio-group labels must be constant within each cage.
    ``'within_cage'`` permutes labels only among samples in the same cage.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive.")
    if exchangeability_unit not in {"strata", "within_cage", "cage"}:
        raise ValueError("exchangeability_unit must be 'strata', 'within_cage', or 'cage'.")
    numeric, aligned_labels = _validate_values(values, labels, n_components)
    aligned_strata = _align_series(numeric, strata, "Permutation strata")

    # Observed separation with the true labels.
    observed_scores, _ = fit_plsda(numeric, aligned_labels, n_components=n_components)
    observed_separation = _centroid_separation(observed_scores, aligned_labels)
    rng = np.random.default_rng(random_state)
    permuted_separations = np.empty(n_permutations)

    # Null: refit PLS-DA on relabelled data and record the separation each time.
    for permutation_index in range(n_permutations):
        permuted_labels = aligned_labels.copy()
        if exchangeability_unit == "cage":
            # Whole cages keep one label; the cage labels are shuffled.
            cluster_frame = pd.DataFrame({"cluster": aligned_strata, "label": aligned_labels})
            cluster_labels = cluster_frame.groupby("cluster", observed=True)["label"].nunique()
            if cluster_labels.ne(1).any():
                raise ValueError("Cage-level PLS-DA permutations require one label per cage.")
            cluster_names = cluster_labels.index.to_numpy()
            shuffled = rng.permutation(
                cluster_frame.drop_duplicates("cluster").set_index("cluster")["label"].to_numpy()
            )
            mapping = dict(zip(cluster_names, shuffled))
            permuted_labels = aligned_strata.map(mapping)
        else:
            # Shuffle labels within each stratum (e.g. within sex, or within cage).
            for _, sample_index in aligned_strata.groupby(
                aligned_strata, observed=True
            ).groups.items():
                permuted_labels.loc[sample_index] = rng.permutation(
                    aligned_labels.loc[sample_index].to_numpy()
                )
        permuted_scores, _ = fit_plsda(
            numeric, permuted_labels, n_components=n_components
        )
        permuted_separations[permutation_index] = _centroid_separation(
            permuted_scores, permuted_labels
        )

    # p = (1 + permutations at least as separated) / (1 + permutations).
    result = pd.Series({
        "observed_centroid_separation": observed_separation,
        "permutation_p_value": (
            1 + np.count_nonzero(permuted_separations >= observed_separation)
        )
        / (n_permutations + 1),
        "n_permutations": n_permutations,
        "permutation_strata": (
            "across cages" if exchangeability_unit == "cage"
            else "within cages" if exchangeability_unit == "within_cage"
            else "within supplied strata"
        ),
    })
    return result, permuted_separations
