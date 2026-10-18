"""ASCA+ decomposition for the repeated region and treatment design.

The design uses sum-to-zero effect coding for Region, Group, Sex, and their
interactions.  Mouse is nested within Group x Sex and is represented by
within-cell mouse contrasts.  This is the fixed-effect analogue of the
random-intercept term used for the repeated regional measurements.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REGION_LEVELS = ("Cerebellum", "Cortex", "Hippocampus")
GROUP_LEVELS = ("Saline Control", "MIA")
SEX_LEVELS = ("Male", "Female")
SCALING_MODES = ("autoscaled", "unscaled", "pareto")
EFFECT_ORDER = (
    "Region",
    "Group",
    "Sex",
    "Group:Sex",
    "Mouse",
    "Region:Group",
    "Region:Sex",
    "Region:Group:Sex",
)


@dataclass(frozen=True)
class AscaFit:
    """Fitted ASCA decomposition for one feature matrix and scaling arm."""

    feature_names: tuple[str, ...]
    metadata: pd.DataFrame
    scaling: str
    design: np.ndarray
    term_columns: dict[str, tuple[int, ...]]
    coefficients: np.ndarray
    prepared_data: np.ndarray
    centered_data: np.ndarray
    effects: dict[str, np.ndarray]
    residual: np.ndarray
    raw_residual: np.ndarray
    feature_means: np.ndarray
    feature_scales: np.ndarray

    @property
    def n_samples(self) -> int:
        return self.prepared_data.shape[0]

    @property
    def n_features(self) -> int:
        return self.prepared_data.shape[1]

    @property
    def model_rank(self) -> int:
        return int(np.linalg.matrix_rank(self.design))

    @property
    def total_ssq(self) -> float:
        return float(np.square(self.centered_data).sum())


def _effect_code(labels: pd.Series, levels: tuple[str, ...]) -> np.ndarray:
    """Return Patsy-style sum coding with the last level as reference."""
    values = labels.astype(str).to_numpy()
    if set(values) != set(levels):
        missing = sorted(set(levels).difference(values), key=str)
        extra = sorted(set(values).difference(levels), key=str)
        raise ValueError(f"Factor levels do not match; missing={missing}, extra={extra}.")
    coded = np.zeros((len(values), len(levels) - 1), dtype=float)
    for column, level in enumerate(levels[:-1]):
        coded[values == level, column] = 1.0
        coded[values == levels[-1], column] = -1.0
    return coded


def _validate_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    required = {"Sample type", "Bio group", "Sex", "Mouse number"}
    missing = required.difference(metadata.columns)
    if missing:
        raise KeyError(f"ASCA metadata is missing columns: {sorted(missing)}")
    result = metadata.reset_index(drop=True).copy()
    for column in required:
        if result[column].isna().any():
            raise ValueError(f"ASCA metadata contains missing values in {column!r}.")
    for mouse, rows in result.groupby("Mouse number", sort=False, observed=True):
        for column in ("Bio group", "Sex"):
            if rows[column].nunique(dropna=False) != 1:
                raise ValueError(f"Mouse {mouse!r} has inconsistent {column} labels.")
    return result


def _nested_mouse_code(metadata: pd.DataFrame) -> np.ndarray:
    """Encode Mouse within each observed Group x Sex cell."""
    rows: list[np.ndarray] = []
    for (group, sex), cell in metadata.groupby(
        ["Bio group", "Sex"], sort=True, observed=True
    ):
        mice = sorted(cell["Mouse number"].astype(str).unique(), key=str)
        if len(mice) < 2:
            continue
        reference = mice[-1]
        for mouse in mice[:-1]:
            column = np.zeros(len(metadata), dtype=float)
            column[(metadata["Mouse number"].astype(str) == mouse).to_numpy()] = 1.0
            column[
                (metadata["Bio group"].astype(str) == str(group))
                & (metadata["Sex"].astype(str) == str(sex))
                & (metadata["Mouse number"].astype(str) == reference)
            ] = -1.0
            rows.append(column)
    return np.column_stack(rows) if rows else np.empty((len(metadata), 0), dtype=float)


def build_effect_coded_design(
    metadata: pd.DataFrame,
    *,
    allow_rank_deficient: bool = False,
) -> tuple[np.ndarray, dict[str, tuple[int, ...]]]:
    """Build the full-rank effect-coded ASCA+ design matrix."""
    metadata = _validate_metadata(metadata)
    region = _effect_code(metadata["Sample type"], REGION_LEVELS)
    group = _effect_code(metadata["Bio group"], GROUP_LEVELS)
    sex = _effect_code(metadata["Sex"], SEX_LEVELS)
    group_sex = group * sex
    mouse = _nested_mouse_code(metadata)

    parts: list[np.ndarray] = [np.ones((len(metadata), 1), dtype=float)]
    names: list[str] = ["Intercept"]
    term_parts = {
        "Region": (region, ["Region:Cerebellum", "Region:Cortex"]),
        "Group": (group, ["Group:Saline Control"]),
        "Sex": (sex, ["Sex:Male"]),
        "Group:Sex": (group_sex, ["Group:Sex"]),
        "Mouse": (mouse, [f"Mouse:{i}" for i in range(mouse.shape[1])]),
        "Region:Group": (
            region * group[:, [0]],
            ["Region:Cerebellum:Group", "Region:Cortex:Group"],
        ),
        "Region:Sex": (
            region * sex[:, [0]],
            ["Region:Cerebellum:Sex", "Region:Cortex:Sex"],
        ),
        "Region:Group:Sex": (
            region * group[:, [0]] * sex[:, [0]],
            ["Region:Cerebellum:Group:Sex", "Region:Cortex:Group:Sex"],
        ),
    }
    term_columns: dict[str, tuple[int, ...]] = {}
    next_column = 1
    for term in EFFECT_ORDER:
        values, column_names = term_parts[term]
        if values.shape[1] != len(column_names):
            raise AssertionError(f"Column-name mismatch for {term}.")
        parts.append(values)
        names.extend(column_names)
        term_columns[term] = tuple(range(next_column, next_column + values.shape[1]))
        next_column += values.shape[1]

    design = np.column_stack(parts)
    if not allow_rank_deficient:
        rank = np.linalg.matrix_rank(design)
        if rank != design.shape[1]:
            raise ValueError(
                f"ASCA design is rank deficient: rank {rank}, columns {design.shape[1]}."
            )
    if len(names) != design.shape[1]:
        raise AssertionError("ASCA design names do not match matrix columns.")
    return design, term_columns


def _scale_sample_matrix(
    sample_by_feature: np.ndarray,
    scaling: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scaling not in SCALING_MODES:
        raise ValueError(f"Unknown ASCA scaling {scaling!r}; choose from {SCALING_MODES}.")
    data = np.asarray(sample_by_feature, dtype=float)
    if data.ndim != 2 or data.size == 0:
        raise ValueError("ASCA requires a non-empty two-dimensional matrix.")
    if not np.isfinite(data).all():
        raise ValueError("ASCA requires a complete finite feature matrix.")
    means = data.mean(axis=0)
    centered = data - means
    standard_deviation = data.std(axis=0, ddof=1)
    if np.any(standard_deviation <= 0) or not np.isfinite(standard_deviation).all():
        raise ValueError("ASCA cannot scale a feature with zero or non-finite variance.")
    if scaling == "autoscaled":
        divisors = standard_deviation
    elif scaling == "pareto":
        divisors = np.sqrt(standard_deviation)
    else:
        divisors = np.ones(data.shape[1], dtype=float)
    return centered / divisors, means, divisors


def fit_asca_prepared(
    prepared_data: np.ndarray,
    feature_names: list[str] | tuple[str, ...],
    metadata: pd.DataFrame,
    *,
    scaling: str,
    feature_means: np.ndarray | None = None,
    feature_scales: np.ndarray | None = None,
    allow_rank_deficient: bool = False,
) -> AscaFit:
    """Fit ASCA to a finite, already-scaled sample-by-feature matrix."""
    data = np.asarray(prepared_data, dtype=float)
    if data.ndim != 2 or data.size == 0:
        raise ValueError("ASCA requires a non-empty two-dimensional matrix.")
    if len(feature_names) != data.shape[1] or len(metadata) != data.shape[0]:
        raise ValueError("ASCA matrix, feature names, and metadata are misaligned.")
    if not np.isfinite(data).all():
        raise ValueError("Prepared ASCA data must be finite.")
    checked_metadata = _validate_metadata(metadata)
    design, term_columns = build_effect_coded_design(
        checked_metadata, allow_rank_deficient=allow_rank_deficient
    )
    coefficients = np.linalg.pinv(design) @ data
    fitted = design @ coefficients
    centered_data = data - data.mean(axis=0)
    effects = {
        term: design[:, columns] @ coefficients[list(columns)]
        for term, columns in term_columns.items()
    }
    residual = centered_data - sum(effects.values(), np.zeros_like(centered_data))
    raw_residual = data - fitted
    return AscaFit(
        feature_names=tuple(str(feature) for feature in feature_names),
        metadata=checked_metadata,
        scaling=scaling,
        design=design,
        term_columns=term_columns,
        coefficients=coefficients,
        prepared_data=data,
        centered_data=centered_data,
        effects=effects,
        residual=residual,
        raw_residual=raw_residual,
        feature_means=(
            np.zeros(data.shape[1], dtype=float)
            if feature_means is None
            else np.asarray(feature_means, dtype=float)
        ),
        feature_scales=(
            np.ones(data.shape[1], dtype=float)
            if feature_scales is None
            else np.asarray(feature_scales, dtype=float)
        ),
    )


def fit_asca(
    values: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    scaling: str = "autoscaled",
    sample_column: str = "Sample",
) -> AscaFit:
    """Fit ASCA to a complete feature-by-sample matrix."""
    if values.columns.has_duplicates:
        raise ValueError("ASCA sample columns must be unique.")
    numeric = values.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().to_numpy().any():
        raise ValueError("ASCA requires complete numeric feature values.")
    if sample_column in metadata.columns:
        indexed = metadata.set_index(sample_column, drop=False)
        missing = pd.Index(values.columns).difference(indexed.index)
        if len(missing):
            raise ValueError(f"ASCA metadata is missing samples: {missing.tolist()}")
        aligned = indexed.loc[list(values.columns)].reset_index(drop=True)
    else:
        if len(metadata) != values.shape[1]:
            raise ValueError("ASCA metadata rows do not match sample columns.")
        aligned = metadata.reset_index(drop=True)
    return fit_asca_sample_matrix(
        numeric.T.to_numpy(),
        [str(feature) for feature in numeric.index],
        aligned,
        scaling=scaling,
    )


def fit_asca_sample_matrix(
    sample_by_feature: np.ndarray,
    feature_names: list[str] | tuple[str, ...],
    metadata: pd.DataFrame,
    *,
    scaling: str = "autoscaled",
    allow_rank_deficient: bool = False,
) -> AscaFit:
    """Fit ASCA from a sample-by-feature array, including bootstrap duplicates."""
    prepared, means, scales = _scale_sample_matrix(sample_by_feature, scaling)
    return fit_asca_prepared(
        prepared,
        feature_names,
        metadata,
        scaling=scaling,
        feature_means=means,
        feature_scales=scales,
        allow_rank_deficient=allow_rank_deficient,
    )


def effect_variance_table(fit: AscaFit) -> pd.DataFrame:
    """Return sums of squares and variance shares for every model term."""
    total = fit.total_ssq
    rows = []
    for term in EFFECT_ORDER:
        effect = fit.effects[term]
        rows.append({
            "Effect": term,
            "degrees_of_freedom": len(fit.term_columns[term]),
            "ssq": float(np.square(effect).sum()),
            "variance_share": float(np.square(effect).sum() / total) if total else np.nan,
        })
    residual_ssq = float(np.square(fit.residual).sum())
    rows.append({
        "Effect": "Residual",
        "degrees_of_freedom": max(fit.n_samples - fit.model_rank, 0),
        "ssq": residual_ssq,
        "variance_share": residual_ssq / total if total else np.nan,
    })
    result = pd.DataFrame(rows)
    result["variance_share_sum"] = result["variance_share"].sum()
    return result


def effect_svd(
    fit: AscaFit,
    effect: str,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Return sample scores, feature loadings, and singular values for an effect."""
    if effect not in fit.effects:
        raise KeyError(f"Unknown ASCA effect: {effect}")
    effect_matrix = fit.effects[effect]
    gram = effect_matrix @ effect_matrix.T
    eigenvalues, left = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    left = left[:, order]
    n_components = min(len(fit.term_columns[effect]), len(eigenvalues))
    singular_values = np.sqrt(eigenvalues[:n_components])
    scores_array = left[:, :n_components] * singular_values
    loadings_array = effect_matrix.T @ left[:, :n_components]
    nonzero = singular_values > np.finfo(float).eps
    loadings_array[:, nonzero] /= singular_values[nonzero]
    loadings_array[:, ~nonzero] = 0.0
    for component in range(n_components):
        largest = int(np.argmax(np.abs(loadings_array[:, component])))
        if loadings_array[largest, component] < 0:
            scores_array[:, component] *= -1
            loadings_array[:, component] *= -1
    components = [f"PC{number}" for number in range(1, n_components + 1)]
    scores = pd.DataFrame(scores_array, columns=components)
    scores.insert(0, "Sample", fit.metadata.get("Sample", pd.Series(range(fit.n_samples))))
    loadings = pd.DataFrame(loadings_array, index=fit.feature_names, columns=components)
    return scores, loadings, singular_values


def procrustes_align(loadings: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Align bootstrap loading columns to a reference by an orthogonal rotation."""
    values = np.asarray(loadings, dtype=float)
    target = np.asarray(reference, dtype=float)
    if values.shape != target.shape or values.ndim != 2:
        raise ValueError("Bootstrap and reference loadings must have equal 2-D shapes.")
    left, _, right = np.linalg.svd(values.T @ target, full_matrices=False)
    rotation = left @ right
    aligned = values @ rotation
    for component in range(aligned.shape[1]):
        if np.dot(aligned[:, component], target[:, component]) < 0:
            aligned[:, component] *= -1
    return aligned



def effect_residual_projection(
    fit: AscaFit,
    effect: str,
    loadings: pd.DataFrame,
) -> pd.DataFrame:
    """Project effect plus residual onto the effect's loading directions."""
    directions = loadings.loc[list(fit.feature_names)].to_numpy(dtype=float)
    projected = (fit.effects[effect] + fit.residual) @ directions
    result = pd.DataFrame(projected, columns=loadings.columns)
    result.insert(0, "Sample", fit.metadata.get("Sample", pd.Series(range(fit.n_samples))))
    result.insert(1, "Effect", effect)
    return result
