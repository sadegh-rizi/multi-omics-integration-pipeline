"""Structure among *features* (rather than samples) within one region.

Two complementary views of a complete features x samples matrix:

* **feature-space PCA** - an SVD with features as observations
  (:func:`fit_feature_margin_svd`), with bootstrap and split-half checks of
  whether the feature loadings are stable enough to interpret;
* **co-abundance modules** - features clustered on their signed Spearman
  correlation (:func:`assign_hierarchical_modules`), with bootstrap
  co-membership (:func:`bootstrap_module_comembership`) and a permuted null for
  module size (:func:`permuted_null_module_sizes`).

With 20 samples per region both are descriptive: they say whether structure
exists, not which features drive it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .pca import align_loading_signs, class_composition
from .pca import complete_numeric as _complete_numeric


def _component_names(n_components: int) -> list[str]:
    """['PC 1', 'PC 2', ...]."""
    return [f"PC {number}" for number in range(1, n_components + 1)]


def _orient_svd(
    left_vectors: pd.DataFrame,
    right_vectors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Orient left and right SVD vectors using the shared loading convention."""
    aligned_left = align_loading_signs(left_vectors)
    aligned_right = right_vectors.copy()
    for component in left_vectors:
        feature = left_vectors[component].abs().idxmax()
        sign = np.sign(aligned_left.loc[feature, component] / left_vectors.loc[feature, component])
        aligned_right[component] *= sign
    return aligned_left, aligned_right


def _svd_tables(
    matrix: pd.DataFrame,
    *,
    n_components: int | None,
    scale_rows: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit an SVD to a feature-by-sample matrix after the requested margin transform.

    Each feature (row) is centred across samples, and divided by its SD if
    ``scale_rows``.  Left singular vectors = feature coordinates, right singular
    vectors = sample coordinates; returns both plus the scree table.
    """
    numeric = _complete_numeric(matrix)
    if len(numeric.columns) < 3:
        raise ValueError("At least three samples are required for two-component stability.")

    centered = numeric.subtract(numeric.mean(axis=1), axis=0)
    row_sd = centered.std(axis=1, ddof=1)
    if row_sd.le(0).any() or row_sd.isna().any():
        raise ValueError("Every feature must vary across samples after centering.")
    transformed = centered.divide(row_sd, axis=0) if scale_rows else centered

    left, singular_values, right_transposed = np.linalg.svd(
        transformed.to_numpy(dtype=float),
        full_matrices=False,
    )
    maximum = min(transformed.shape[0], transformed.shape[1] - 1)
    if n_components is None:
        n_components = maximum
    if not 1 <= n_components <= maximum:
        raise ValueError(f"n_components must be between 1 and {maximum}.")

    components = _component_names(n_components)
    feature_vectors = pd.DataFrame(
        left[:, :n_components],
        index=numeric.index,
        columns=components,
    )
    sample_vectors = pd.DataFrame(
        right_transposed[:n_components].T,
        index=numeric.columns,
        columns=components,
    )
    # Same sign convention as the sample PCA; variance explained = squared singular values.
    feature_vectors, sample_vectors = _orient_svd(feature_vectors, sample_vectors)
    variance = singular_values[:n_components] ** 2
    variance /= np.square(singular_values).sum()
    scree = pd.DataFrame({
        "Component": components,
        "singular_value": singular_values[:n_components],
        "variance_explained": variance,
        "cumulative_variance_explained": np.cumsum(variance),
    })
    return feature_vectors, sample_vectors, scree


def fit_feature_margin_svd(
    values: pd.DataFrame,
    *,
    n_components: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit the row-centred feature-margin SVD.

    Features are the observations and samples are the variables. Each feature
    is centred across its regional samples, but is not variance-scaled. The
    returned feature table contains normalized left singular vectors ``U``;
    this is the feature coordinate used for loading stability and class
    composition. The sample table contains the corresponding right vectors.
    """
    return _svd_tables(values, n_components=n_components, scale_rows=False)


def fit_sample_space_loadings(
    values: pd.DataFrame,
    *,
    n_components: int | None = None,
    scale_features: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return feature loadings and scree values for the sample-space PCA view.

    This is used only to reproduce the direct feature view for a population
    that has no persisted loading table. Feature rows are centred and, by
    default, standardized across samples before the sample PCA SVD.
    """
    feature_vectors, _, scree = _svd_tables(
        values,
        n_components=n_components,
        scale_rows=scale_features,
    )
    return feature_vectors, scree


def spearman_correlation(values: pd.DataFrame) -> pd.DataFrame:
    """Return the complete feature-by-feature Spearman correlation matrix."""
    numeric = _complete_numeric(values)
    if len(numeric.columns) < 3:
        raise ValueError("At least three samples are required for correlations.")
    correlation = numeric.T.corr(method="spearman")
    if correlation.isna().to_numpy().any():
        raise ValueError("Spearman correlation is undefined for at least one feature pair.")
    return correlation


def assign_hierarchical_modules(
    correlation: pd.DataFrame,
    *,
    distance_cut: float = 0.30,
    signed: bool = True,
) -> pd.Series:
    """Assign deterministic average-linkage modules from a correlation matrix."""
    if correlation.empty or correlation.shape[0] != correlation.shape[1]:
        raise ValueError("Correlation matrix must be non-empty and square.")
    if not correlation.index.equals(correlation.columns):
        raise ValueError("Correlation matrix row and column identifiers must match.")
    if not 0 < distance_cut < 2:
        raise ValueError("distance_cut must be between zero and two.")
    matrix = correlation.to_numpy(dtype=float)
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T):
        raise ValueError("Correlation matrix must be finite and symmetric.")
    if not np.allclose(np.diag(matrix), 1.0):
        raise ValueError("Correlation matrix diagonal must equal one.")

    # Signed distance 1 - rho: positively correlated features are close, anti-correlated
    # ones far apart (unsigned would put them together).  Average linkage, cut at distance_cut.
    distance = 1.0 - (matrix if signed else np.abs(matrix))
    np.fill_diagonal(distance, 0.0)
    distance = np.clip(distance, 0.0, 2.0)
    if len(correlation) == 1:
        return pd.Series("module_1", index=correlation.index, name="module")

    tree = linkage(squareform(distance, checks=False), method="average")
    raw_labels = fcluster(tree, t=distance_cut, criterion="distance")
    # Rename clusters module_1, module_2, ... in a reproducible order.
    groups = pd.Series(raw_labels, index=correlation.index)
    ordering = sorted(
        groups.groupby(groups, sort=False).groups.items(),
        key=lambda item: min(str(feature) for feature in item[1]),
    )
    label_map = {raw: f"module_{number}" for number, (raw, _) in enumerate(ordering, start=1)}
    return groups.map(label_map).rename("module")


def loading_stability_summary(
    bootstrap_loadings: np.ndarray,
    feature_index: pd.Index,
    *,
    component_names: Sequence[str] = ("PC 1", "PC 2"),
) -> pd.DataFrame:
    """Summarize sign-aligned bootstrap feature loadings."""
    loadings = np.asarray(bootstrap_loadings, dtype=float)
    if loadings.ndim != 3:
        raise ValueError("bootstrap_loadings must have shape (resamples, features, components).")
    if loadings.shape[1] != len(feature_index) or loadings.shape[2] != len(component_names):
        raise ValueError("Bootstrap loading dimensions do not match the feature index.")
    if not np.isfinite(loadings).all():
        raise ValueError("Bootstrap loadings must be finite.")

    # How often is each feature among the top 10 % |loading| of a component across bootstraps?
    absolute = np.abs(loadings)
    top_count = max(1, int(np.ceil(len(feature_index) * 0.10)))
    top_indices = np.argsort(-absolute, axis=1, kind="stable")[:, :top_count, :]
    top_mask = np.zeros_like(absolute, dtype=bool)
    for component in range(loadings.shape[2]):
        for resample in range(loadings.shape[0]):
            top_mask[resample, top_indices[resample, :, component], component] = True

    # Per feature and component: centre, spread and 95 % interval of the bootstrap loadings.
    rows: list[dict[str, object]] = []
    for component_number, component in enumerate(component_names):
        for feature_number, feature in enumerate(feature_index):
            distribution = loadings[:, feature_number, component_number]
            rows.append({
                "Feature": feature,
                "Component": component,
                "loading_mean": distribution.mean(),
                "loading_median": np.median(distribution),
                "loading_sd": distribution.std(ddof=1),
                "loading_q025": np.quantile(distribution, 0.025),
                "loading_q975": np.quantile(distribution, 0.975),
                "top_decile_fraction": top_mask[:, feature_number, component_number].mean(),
                "n_bootstraps": loadings.shape[0],
                "sign_alignment": "largest_absolute_loading_positive",
            })
    return pd.DataFrame(rows)


def bootstrap_feature_loadings(
    values: pd.DataFrame,
    *,
    n_bootstraps: int,
    random_state: int,
    n_components: int = 2,
) -> np.ndarray:
    """Fit sign-aligned feature-margin loadings on sample bootstraps."""
    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be positive.")
    numeric = _complete_numeric(values)
    if n_components < 1:
        raise ValueError("n_components must be positive.")
    generator = np.random.default_rng(random_state)
    result = np.empty((n_bootstraps, len(numeric), n_components), dtype=float)
    for number in range(n_bootstraps):
        while True:
            sample_indices = generator.integers(0, len(numeric.columns), size=len(numeric.columns))
            if len(np.unique(sample_indices)) >= 3:
                break
        resampled = numeric.iloc[:, sample_indices].copy()
        resampled.columns = [f"bootstrap_{position}" for position in range(len(sample_indices))]
        feature_loadings, _, _ = fit_feature_margin_svd(resampled, n_components=n_components)
        result[number] = feature_loadings.to_numpy(dtype=float)
    return result


def split_half_loading_stability(
    values: pd.DataFrame,
    *,
    random_state: int,
    n_components: int = 2,
) -> pd.DataFrame:
    """Compare feature-margin loadings from one deterministic sample split.

    Fit the feature SVD on two random halves of the samples and correlate the
    loadings: a component is interpretable only if both halves agree (r >= 0.5).
    """
    numeric = _complete_numeric(values)
    if len(numeric.columns) < 6:
        raise ValueError("At least six samples are required for split-half stability.")
    generator = np.random.default_rng(random_state)
    order = generator.permutation(len(numeric.columns))
    midpoint = len(order) // 2
    first, second = order[:midpoint], order[midpoint:]
    first_loadings, _, _ = fit_feature_margin_svd(numeric.iloc[:, first], n_components=n_components)
    second_loadings, _, _ = fit_feature_margin_svd(numeric.iloc[:, second], n_components=n_components)
    rows = []
    for component in first_loadings:
        correlation = first_loadings[component].corr(second_loadings[component])
        rows.append({
            "Component": component,
            "split_half_loading_correlation": correlation,
            "split_seed": random_state,
            "n_samples_half_1": len(first),
            "n_samples_half_2": len(second),
            "n_features": len(numeric),
        })
    return pd.DataFrame(rows)


def bootstrap_module_comembership(
    values: pd.DataFrame,
    *,
    n_bootstraps: int,
    random_state: int,
    distance_cut: float,
    signed: bool,
) -> pd.DataFrame:
    """Estimate feature-pair co-membership across sample bootstraps."""
    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be positive.")
    numeric = _complete_numeric(values)
    generator = np.random.default_rng(random_state)
    counts = np.zeros((len(numeric), len(numeric)), dtype=np.int32)
    for _ in range(n_bootstraps):
        while True:
            sample_indices = generator.integers(0, len(numeric.columns), size=len(numeric.columns))
            if len(np.unique(sample_indices)) >= 3:
                break
        resampled = numeric.iloc[:, sample_indices].copy()
        resampled.columns = [f"bootstrap_{position}" for position in range(len(sample_indices))]
        correlation = spearman_correlation(resampled)
        modules = assign_hierarchical_modules(
            correlation,
            distance_cut=distance_cut,
            signed=signed,
        )
        labels = modules.to_numpy()
        counts += labels[:, np.newaxis] == labels[np.newaxis, :]
    return pd.DataFrame(
        counts / n_bootstraps,
        index=numeric.index,
        columns=numeric.index,
    )


def module_stability_summary(
    modules: pd.Series,
    comembership: pd.DataFrame,
    *,
    stability_floor: float,
    minimum_size: int,
) -> pd.DataFrame:
    """Summarize observed-module pairwise bootstrap co-membership."""
    if not 0 <= stability_floor <= 1:
        raise ValueError("stability_floor must be between zero and one.")
    if minimum_size < 2:
        raise ValueError("minimum_size must be at least two.")
    if not modules.index.equals(comembership.index) or not comembership.index.equals(comembership.columns):
        raise ValueError("Module labels and co-membership matrix must share the same index.")
    rows = []
    for module, feature_index in modules.groupby(modules, sort=False).groups.items():
        positions = comembership.index.get_indexer(feature_index)
        pair_values = comembership.to_numpy()[np.ix_(positions, positions)]
        upper = pair_values[np.triu_indices(len(positions), k=1)]
        mean = float(np.mean(upper)) if len(upper) else np.nan
        n_features = len(feature_index)
        rows.append({
            "module": module,
            "n_features": n_features,
            "n_feature_pairs": len(upper),
            "mean_co_membership": mean,
            "stability_floor": stability_floor,
            "minimum_module_size": minimum_size,
            "reportable": bool(
                n_features >= minimum_size
                and pd.notna(mean)
                and mean >= stability_floor
            ),
        })
    return pd.DataFrame(rows)


def loading_annotation_composition(
    loadings: pd.DataFrame,
    annotations: pd.Series,
    *,
    annotation_name: str = "Annotation",
    n_largest: int = 6,
) -> pd.DataFrame:
    """Report squared-loading shares by a feature annotation (see :func:`.pca.class_composition`)."""
    return class_composition(loadings, annotations, n_largest=n_largest, annotation_name=annotation_name)


def module_annotation_composition(
    modules: pd.Series,
    annotations: pd.Series,
    *,
    annotation_name: str = "Annotation",
) -> pd.DataFrame:
    """Report module annotation shares and enrichment against the panel."""
    labels = annotations.reindex(modules.index).fillna("Unannotated").astype(str)
    panel_share = labels.value_counts(normalize=True)
    rows: list[dict[str, object]] = []
    for module, feature_index in modules.groupby(modules, sort=False).groups.items():
        module_labels = labels.reindex(feature_index)
        counts = module_labels.value_counts()
        for annotation, count in counts.items():
            module_share = count / len(feature_index)
            rows.append({
                "module": module,
                "n_features": len(feature_index),
                annotation_name: annotation,
                "n_features_in_annotation": count,
                "module_feature_share": module_share,
                "panel_feature_share": panel_share[annotation],
                "enrichment_vs_panel": module_share / panel_share[annotation],
            })
    return pd.DataFrame(rows)


def permuted_null_module_sizes(
    values: pd.DataFrame,
    *,
    n_permutations: int,
    random_state: int,
    distance_cut: float,
    signed: bool,
    minimum_size: int,
) -> pd.DataFrame:
    """Run the module pipeline on independently sample-permuted features."""
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive.")
    numeric = _complete_numeric(values)
    generator = np.random.default_rng(random_state)
    array = numeric.to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    # Shuffle each feature's values across samples independently: this destroys all
    # correlation between features but keeps every feature's distribution.  Modules
    # found in such data show how large modules get by chance.
    for permutation in range(1, n_permutations + 1):
        permuted = np.empty_like(array)
        for feature_number in range(array.shape[0]):
            permuted[feature_number] = array[feature_number, generator.permutation(array.shape[1])]
        permuted_values = pd.DataFrame(permuted, index=numeric.index, columns=numeric.columns)
        correlation = spearman_correlation(permuted_values)
        modules = assign_hierarchical_modules(
            correlation,
            distance_cut=distance_cut,
            signed=signed,
        )
        sizes = modules.value_counts(sort=False)
        for module, size in sizes.items():
            rows.append({
                "null_permutation": permutation,
                "null_module": module,
                "n_features": int(size),
                "reportable_by_size": bool(size >= minimum_size),
            })
    return pd.DataFrame(rows)
