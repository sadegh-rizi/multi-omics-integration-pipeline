"""Abundance-dependent sample-similarity diagnostics for lipid peak areas.

The functions operate on feature-by-sample matrices already transformed to
log2 positive peak areas. They use the noisyR count-matrix ordering: each
reference sample ranks features by its own abundance before comparison with
the other samples. Missing values remain missing and are excluded only from
the reference-comparator correlation in which they occur.
"""

from __future__ import annotations

from itertools import combinations
from math import ceil

import numpy as np
import pandas as pd
from scipy.stats import rankdata

DEFAULT_THRESHOLD_RULES = (
    "pearson_gt_0_25",
    "pearson_gt_0_50",
    "pearson_90pct_plateau",
    "bio_tech_gap_le_0_05",
    "pearson_sustained_gt_0_25",
)

NOISYR_COUNT_ORDERING = "noisyr_sample_specific_abundance"
REFERENCE_WINDOW_ABUNDANCE = "arithmetic_mean"
CORRELATION_AGGREGATION = "mean_across_comparators_then_median_across_references"

_DIRECTED_METADATA_COLUMNS = [
    "sample_i",
    "sample_j",
    "window_index",
    "window_start",
    "window_stop",
    "mean_log2_abundance",
    "correlation",
    "n_lipids_in_window",
    "n_lipids_in_correlation",
    "ordering_method",
]

_PAIRWISE_COLUMNS = [
    *_DIRECTED_METADATA_COLUMNS,
    "correlation",
    "method",
]

_DIRECTED_COLUMNS = [
    *_DIRECTED_METADATA_COLUMNS,
    "pearson",
    "spearman",
]

_CURVE_COLUMNS = [
    "window_index",
    "median_log2_abundance",
    "median_pearson",
    "q25_pearson",
    "q75_pearson",
    "median_spearman",
    "n_reference_samples",
    "n_pairwise_correlations",
    "n_lipids_in_window",
    "median_lipids_in_correlation",
    "window_size",
    "ordering_method",
    "reference_window_abundance",
    "correlation_aggregation",
    "min_comparators_per_reference",
    "min_pairwise_correlations_per_point",
    "min_pairwise_observations",
]


def _validated_values(values: pd.DataFrame, *, window_size: int, step: int) -> pd.DataFrame:
    if not isinstance(values, pd.DataFrame):
        raise TypeError("values must be a pandas DataFrame.")
    if values.index.has_duplicates:
        raise ValueError("Feature indexes must be unique.")
    if values.columns.has_duplicates:
        raise ValueError("Sample columns must be unique.")
    if values.shape[1] < 2:
        raise ValueError("At least two samples are required.")
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer.")
    if not isinstance(step, int) or step <= 0:
        raise ValueError("step must be a positive integer.")
    if window_size > len(values.index):
        raise ValueError("window_size cannot exceed the number of lipids.")

    numeric = values.apply(pd.to_numeric, errors="raise")
    if not numeric.apply(lambda column: np.isfinite(column.dropna()).all()).all():
        raise ValueError("values may contain finite numbers or missing values only.")
    return numeric


def _window_starts(n_lipids: int, window_size: int, step: int) -> list[int]:
    starts = list(range(0, n_lipids - window_size + 1, step))
    final_start = n_lipids - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _correlation(left: np.ndarray, right: np.ndarray, *, method: str) -> float:
    """Calculate Pearson or Spearman correlation for one complete feature set."""
    if method == "spearman":
        left = rankdata(left, method="average")
        right = rankdata(right, method="average")

    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = np.sqrt(np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered))
    if denominator == 0:
        return np.nan
    return float(np.clip(np.dot(left_centered, right_centered) / denominator, -1.0, 1.0))


def _directed_window_correlations(
    values: pd.DataFrame,
    *,
    window_size: int,
    step: int,
    methods: tuple[str, ...],
    min_pairwise_observations: int = 2,
) -> pd.DataFrame:
    """Calculate directed noisyR correlations for one or more methods.

    Each reference sample ranks its own observed features in descending
    abundance, as in the noisyR count-matrix approach. The returned window
    indexes run from low to high abundance so threshold rules remain ordered in
    the plotted direction. A comparator can be missing a feature in a window;
    that feature is omitted from that correlation only.
    """
    if not methods or set(methods).difference({"pearson", "spearman"}):
        raise ValueError("methods must contain 'pearson', 'spearman', or both.")
    if not isinstance(min_pairwise_observations, int) or min_pairwise_observations < 2:
        raise ValueError("min_pairwise_observations must be an integer of at least 2.")

    numeric = _validated_values(values, window_size=window_size, step=step)
    matrix = numeric.astype(float).to_numpy()
    sample_names = numeric.columns.tolist()
    rows: list[dict[str, object]] = []

    for reference_position, sample_i in enumerate(sample_names):
        reference_values = matrix[:, reference_position]
        observed_positions = np.flatnonzero(np.isfinite(reference_values))
        if len(observed_positions) < window_size:
            continue
        descending_order = observed_positions[
            np.argsort(-reference_values[observed_positions], kind="stable")
        ]
        starts = _window_starts(len(descending_order), window_size, step)
        for descending_index, start in enumerate(starts):
            stop = start + window_size
            window_positions = descending_order[start:stop]
            reference_window = reference_values[window_positions]
            window_index = len(starts) - descending_index - 1
            for comparator_position, sample_j in enumerate(sample_names):
                if reference_position == comparator_position:
                    continue
                comparator_window = matrix[window_positions, comparator_position]
                complete = np.isfinite(reference_window) & np.isfinite(comparator_window)
                n_lipids_in_correlation = int(complete.sum())
                if n_lipids_in_correlation < min_pairwise_observations:
                    continue
                correlations = {
                    method: _correlation(
                        reference_window[complete], comparator_window[complete], method=method
                    )
                    for method in methods
                }
                if not any(np.isfinite(correlation) for correlation in correlations.values()):
                    continue
                rows.append({
                    "sample_i": sample_i,
                    "sample_j": sample_j,
                    "window_index": window_index,
                    "window_start": start,
                    "window_stop": stop,
                    "mean_log2_abundance": reference_window.mean(),
                    "n_lipids_in_window": len(window_positions),
                    "n_lipids_in_correlation": n_lipids_in_correlation,
                    "ordering_method": NOISYR_COUNT_ORDERING,
                    **correlations,
                })

    return pd.DataFrame(rows, columns=_DIRECTED_COLUMNS)


def pairwise_window_correlation(
    values: pd.DataFrame,
    *,
    window_size: int,
    step: int,
    method: str,
    min_pairwise_observations: int = 2,
) -> pd.DataFrame:
    """Calculate directed reference-comparator correlations in noisyR windows."""
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'.")

    directed = _directed_window_correlations(
        values,
        window_size=window_size,
        step=step,
        methods=(method,),
        min_pairwise_observations=min_pairwise_observations,
    )
    mask = directed[method].notna()
    result = directed.loc[mask, _DIRECTED_METADATA_COLUMNS].copy()
    result["correlation"] = directed.loc[mask, method]
    result["method"] = method
    return result.loc[:, _PAIRWISE_COLUMNS]


def abundance_similarity_curve(
    values: pd.DataFrame,
    *,
    window_size: int,
    step: int,
    min_pairwise_correlations_per_point: int,
    min_comparators_per_reference: int = 2,
    min_pairwise_observations: int = 2,
) -> pd.DataFrame:
    """Aggregate noisyR-style sample-specific correlations by abundance rank.

    For every reference sample and window, noisyR uses the arithmetic mean
    reference abundance for the x-coordinate and averages correlations to
    comparator samples. Regional curves then report the median and IQR of
    those reference-sample values as a descriptive project-level summary.
    """
    if (
        not isinstance(min_pairwise_correlations_per_point, int)
        or min_pairwise_correlations_per_point <= 0
    ):
        raise ValueError("min_pairwise_correlations_per_point must be a positive integer.")
    if not isinstance(min_comparators_per_reference, int) or min_comparators_per_reference <= 0:
        raise ValueError("min_comparators_per_reference must be a positive integer.")
    if not isinstance(min_pairwise_observations, int) or min_pairwise_observations < 2:
        raise ValueError("min_pairwise_observations must be an integer of at least 2.")

    paired = _directed_window_correlations(
        values,
        window_size=window_size,
        step=step,
        methods=("pearson", "spearman"),
        min_pairwise_observations=min_pairwise_observations,
    )

    reference_rows: list[dict[str, object]] = []
    for (sample_i, window_index), window_rows in paired.groupby(["sample_i", "window_index"], sort=True):
        pearson_rows = window_rows.dropna(subset=["pearson"])
        if len(pearson_rows) < min_comparators_per_reference:
            continue
        reference_rows.append({
            "sample_i": sample_i,
            "window_index": int(window_index),
            "mean_log2_abundance": pearson_rows["mean_log2_abundance"].iloc[0],
            "mean_pearson": pearson_rows["pearson"].mean(),
            "mean_spearman": pearson_rows["spearman"].mean(),
            "n_pairwise_correlations": len(pearson_rows),
            "n_lipids_in_window": int(pearson_rows["n_lipids_in_window"].median()),
            "median_lipids_in_correlation": pearson_rows["n_lipids_in_correlation"].median(),
            "ordering_method": pearson_rows["ordering_method"].iloc[0],
        })

    if not reference_rows:
        return pd.DataFrame(columns=_CURVE_COLUMNS)

    reference_curve = pd.DataFrame(reference_rows)
    rows: list[dict[str, object]] = []
    for window_index, window_rows in reference_curve.groupby("window_index", sort=True):
        n_pairwise_correlations = int(window_rows["n_pairwise_correlations"].sum())
        if n_pairwise_correlations < min_pairwise_correlations_per_point:
            continue
        rows.append({
            "window_index": int(window_index),
            "median_log2_abundance": window_rows["mean_log2_abundance"].median(),
            "median_pearson": window_rows["mean_pearson"].median(),
            "q25_pearson": window_rows["mean_pearson"].quantile(0.25),
            "q75_pearson": window_rows["mean_pearson"].quantile(0.75),
            "median_spearman": window_rows["mean_spearman"].median(),
            "n_reference_samples": len(window_rows),
            "n_pairwise_correlations": n_pairwise_correlations,
            "n_lipids_in_window": int(window_rows["n_lipids_in_window"].median()),
            "median_lipids_in_correlation": window_rows["median_lipids_in_correlation"].median(),
            "window_size": window_size,
            "ordering_method": window_rows["ordering_method"].iloc[0],
            "reference_window_abundance": REFERENCE_WINDOW_ABUNDANCE,
            "correlation_aggregation": CORRELATION_AGGREGATION,
            "min_comparators_per_reference": min_comparators_per_reference,
            "min_pairwise_correlations_per_point": min_pairwise_correlations_per_point,
            "min_pairwise_observations": min_pairwise_observations,
        })

    return pd.DataFrame(rows, columns=_CURVE_COLUMNS)


def candidate_thresholds(
    curve: pd.DataFrame,
    *,
    technical_curve: pd.DataFrame | None = None,
    rules: tuple[str, ...] = DEFAULT_THRESHOLD_RULES,
    plateau_fraction: float = 0.20,
    convergence_tolerance: float = 0.05,
    sustained_windows: int = 5,
) -> pd.DataFrame:
    """Report candidate abundance floors without selecting one for filtering."""
    required = {"window_index", "median_log2_abundance", "median_pearson", "median_spearman"}
    if missing := required.difference(curve.columns):
        raise KeyError(f"curve is missing columns: {sorted(missing)}")
    unknown_rules = set(rules).difference(DEFAULT_THRESHOLD_RULES)
    if unknown_rules:
        raise ValueError(f"Unknown threshold rule(s): {sorted(unknown_rules)}")
    if not 0 < plateau_fraction <= 1:
        raise ValueError("plateau_fraction must be in (0, 1].")
    if convergence_tolerance < 0:
        raise ValueError("convergence_tolerance must be non-negative.")
    if not isinstance(sustained_windows, int) or sustained_windows <= 0:
        raise ValueError("sustained_windows must be a positive integer.")

    biological = curve.dropna(
        subset=["median_log2_abundance", "median_pearson"]
    ).sort_values("window_index")
    technical = None
    if technical_curve is not None:
        if missing := required.difference(technical_curve.columns):
            raise KeyError(f"technical_curve is missing columns: {sorted(missing)}")
        technical = technical_curve.dropna(
            subset=["window_index", "median_log2_abundance", "median_pearson"]
        ).sort_values("window_index")

    plateau = np.nan
    if not biological.empty:
        n_plateau_windows = max(1, ceil(len(biological) * plateau_fraction))
        plateau = biological.tail(n_plateau_windows)["median_pearson"].median()

    rows: list[dict[str, object]] = []

    for rule in rules:
        rule_target = np.nan
        candidates = pd.DataFrame()
        comparison_points = pd.DataFrame()
        status = "found"
        n_biological_points_outside_technical_range: int | float = np.nan

        if biological.empty:
            status = "insufficient_curve_points"
        elif rule == "pearson_gt_0_25":
            rule_target = 0.25
            comparison_points = biological
            candidates = comparison_points.loc[comparison_points["median_pearson"].gt(rule_target)]
        elif rule == "pearson_gt_0_50":
            rule_target = 0.50
            comparison_points = biological
            candidates = comparison_points.loc[comparison_points["median_pearson"].gt(rule_target)]
        elif rule == "pearson_sustained_gt_0_25":
            rule_target = 0.25
            comparison_points = biological
            run_length = 0
            for position in range(len(comparison_points)):
                window_is_consecutive = (
                    position == 0
                    or comparison_points.iloc[position]["window_index"]
                    == comparison_points.iloc[position - 1]["window_index"] + 1
                )
                if window_is_consecutive and comparison_points.iloc[position]["median_pearson"] > rule_target:
                    run_length += 1
                else:
                    run_length = 1 if comparison_points.iloc[position]["median_pearson"] > rule_target else 0
                if run_length >= sustained_windows:
                    start = position - sustained_windows + 1
                    candidates = comparison_points.iloc[[start]]
                    break
        elif rule == "pearson_90pct_plateau":
            rule_target = 0.90 * plateau
            comparison_points = biological
            candidates = comparison_points.loc[comparison_points["median_pearson"].ge(rule_target)]
        else:
            rule_target = convergence_tolerance
            if technical is None:
                status = "technical_curve_missing"
            else:
                technical_by_abundance = (
                    technical.loc[:, ["median_log2_abundance", "median_pearson"]]
                    .groupby("median_log2_abundance", as_index=False, sort=True)
                    .median()
                )
                if len(technical_by_abundance) < 2:
                    status = "insufficient_technical_abundance_range"
                else:
                    technical_minimum = technical_by_abundance["median_log2_abundance"].min()
                    technical_maximum = technical_by_abundance["median_log2_abundance"].max()
                    comparison_points = biological.loc[
                        biological["median_log2_abundance"].between(
                            technical_minimum, technical_maximum, inclusive="both"
                        )
                    ].copy()
                    n_biological_points_outside_technical_range = len(biological) - len(comparison_points)
                    if comparison_points.empty:
                        status = "no_shared_technical_abundance_range"
                    else:
                        comparison_points["technical_median_pearson"] = np.interp(
                            comparison_points["median_log2_abundance"],
                            technical_by_abundance["median_log2_abundance"],
                            technical_by_abundance["median_pearson"],
                        )
                        comparison_points["technical_log2_abundance"] = comparison_points[
                            "median_log2_abundance"
                        ]
                        comparison_points["absolute_pearson_gap"] = (
                            comparison_points["median_pearson"]
                            - comparison_points["technical_median_pearson"]
                        ).abs()
                        candidates = comparison_points.loc[
                            comparison_points["absolute_pearson_gap"].le(convergence_tolerance)
                        ]

        if status == "found" and candidates.empty:
            status = "not_reached"

        row: dict[str, object] = {
            "threshold_rule": rule,
            "threshold_log2_abundance": np.nan,
            "threshold_window_index": np.nan,
            "median_pearson_at_threshold": np.nan,
            "median_spearman_at_threshold": np.nan,
            "technical_median_pearson_at_threshold": np.nan,
            "technical_log2_abundance_at_threshold": np.nan,
            "absolute_pearson_gap_at_threshold": np.nan,
            "comparison_alignment": (
                "interpolated_technical_at_biological_abundance"
                if rule == "bio_tech_gap_le_0_05"
                else "not_applicable"
            ),
            "n_later_windows_failing_rule": np.nan,
            "threshold_persists_through_high_abundance": pd.NA,
            "rule_target": rule_target,
            "sustained_windows_required": (
                sustained_windows if rule == "pearson_sustained_gt_0_25" else np.nan
            ),
            "plateau_median_pearson": plateau,
            "n_curve_points": len(biological),
            "n_technical_curve_points": len(technical) if technical is not None else 0,
            "n_abundance_aligned_curve_points": (
                len(comparison_points) if rule == "bio_tech_gap_le_0_05" else 0
            ),
            "n_biological_points_outside_technical_abundance_range": (
                n_biological_points_outside_technical_range
                if rule == "bio_tech_gap_le_0_05"
                else np.nan
            ),
            "status": status,
        }
        if status == "found":
            point = candidates.iloc[0]
            window_index = point["window_index"]
            if rule == "bio_tech_gap_le_0_05":
                technical_pearson = point["technical_median_pearson"]
                technical_abundance = point["technical_log2_abundance"]
            else:
                technical_pearson = np.nan
                technical_abundance = np.nan

            later_points = comparison_points.loc[
                comparison_points["window_index"].ge(window_index)
            ]
            if rule in {"pearson_gt_0_25", "pearson_gt_0_50", "pearson_sustained_gt_0_25"}:
                later_failures = later_points["median_pearson"].le(rule_target)
            elif rule == "pearson_90pct_plateau":
                later_failures = later_points["median_pearson"].lt(rule_target)
            else:
                later_failures = later_points["absolute_pearson_gap"].gt(rule_target)
            row.update({
                "threshold_log2_abundance": point["median_log2_abundance"],
                "threshold_window_index": int(window_index),
                "median_pearson_at_threshold": point["median_pearson"],
                "median_spearman_at_threshold": point["median_spearman"],
                "technical_median_pearson_at_threshold": technical_pearson,
                "technical_log2_abundance_at_threshold": technical_abundance,
                "absolute_pearson_gap_at_threshold": abs(
                    point["median_pearson"] - technical_pearson
                ) if np.isfinite(technical_pearson) else np.nan,
                "n_later_windows_failing_rule": int(later_failures.sum()),
                "threshold_persists_through_high_abundance": not later_failures.any(),
            })
        rows.append(row)

    return pd.DataFrame(rows)


def compare_noise_criteria(
    abundance_flags: pd.Series,
    blank_flags: pd.Series,
    cv_flags: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a complete 2x2x2 contingency table and pairwise Jaccards."""
    flags = {
        "abundance_below_floor": abundance_flags,
        "blank_dominated": blank_flags,
        "qc_cv_above_30": cv_flags,
    }
    first_index: pd.Index | None = None
    for name, values in flags.items():
        if not isinstance(values, pd.Series):
            raise TypeError(f"{name} must be a pandas Series.")
        if values.index.has_duplicates:
            raise ValueError(f"{name} index must be unique.")
        if values.isna().any() or not values.isin([True, False]).all():
            raise ValueError(f"{name} must contain only True and False values.")
        if first_index is None:
            first_index = values.index
        elif not values.index.equals(first_index):
            raise ValueError("All noise-criterion indexes must align exactly.")

    table = pd.DataFrame({name: values.astype(bool) for name, values in flags.items()})
    criteria = list(flags)
    all_combinations = pd.MultiIndex.from_product(
        [[False, True]] * len(criteria), names=criteria
    )
    contingency = (
        table.groupby(criteria, observed=False)
        .size()
        .reindex(all_combinations, fill_value=0)
        .rename("n_features")
        .reset_index()
    )

    jaccard_rows: list[dict[str, object]] = []
    for criterion_a, criterion_b in combinations(criteria, 2):
        intersection = int((table[criterion_a] & table[criterion_b]).sum())
        union = int((table[criterion_a] | table[criterion_b]).sum())
        jaccard_rows.append({
            "criterion_a": criterion_a,
            "criterion_b": criterion_b,
            "intersection": intersection,
            "union": union,
            "jaccard": intersection / union if union else np.nan,
        })
    return contingency, pd.DataFrame(jaccard_rows)
