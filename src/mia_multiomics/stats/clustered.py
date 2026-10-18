"""Cage (litter) statistics.

Pups of one litter share a mother, a cage and a collection date, so they are
not independent; the litter is the experimental unit for MIA.  This module:

* collapses samples to one mean per cage (:func:`collapse_to_cage_means`);
* estimates how similar littermates are (intraclass correlation, :func:`intraclass_correlation`);
* enumerates every possible assignment of whole cages to MIA / control
  (:func:`cage_group_assignments`) for the exact cage permutation test, and
  reports its smallest attainable p-value (:func:`cage_permutation_ceiling`).
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd


def _aligned_metadata(
    values: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    sample_column: str,
) -> pd.DataFrame:
    """Metadata rows in the order of the value columns, after basic checks."""
    if values.columns.has_duplicates:
        raise ValueError("Values must have unique sample columns.")
    required = {sample_column, "Cage", "Bio group"}
    missing = required.difference(metadata.columns)
    if missing:
        raise KeyError(f"Metadata is missing columns: {sorted(missing)}")
    if metadata[sample_column].duplicated().any():
        raise ValueError("Metadata sample identifiers must be unique.")
    missing_samples = values.columns.difference(metadata[sample_column])
    if len(missing_samples):
        raise ValueError(f"Metadata is missing samples: {missing_samples.tolist()}")
    return metadata.set_index(sample_column, drop=False).loc[values.columns].copy()


def _one_value_per_cage(metadata: pd.DataFrame, column: str) -> None:
    """Raise if a cage has more than one value of ``column`` (e.g. two treatment groups)."""
    if column not in metadata:
        return
    counts = metadata.groupby("Cage", observed=True)[column].nunique(dropna=False)
    if counts.ne(1).any():
        raise ValueError(f"Each cage must have one {column} value.")


def collapse_to_cage_means(
    values: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    sample_column: str = "Sample",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse a feature-by-sample matrix to one mean per cage."""
    aligned = _aligned_metadata(values, metadata, sample_column=sample_column)
    _one_value_per_cage(aligned, "Bio group")
    _one_value_per_cage(aligned, "Date collected")
    _one_value_per_cage(aligned, "Collection cohort")

    # Mean of each feature over the samples of a cage (missing values ignored).
    numeric = values.loc[:, aligned[sample_column]].apply(pd.to_numeric, errors="coerce")
    cage_values = numeric.T.assign(Cage=aligned["Cage"].to_numpy())
    cage_values = cage_values.groupby("Cage", sort=False, observed=True).mean(numeric_only=True).T

    # One metadata row per cage; a cage with pups of both sexes is labelled "Mixed".
    rows: list[dict[str, object]] = []
    for cage, cage_rows in aligned.groupby("Cage", sort=False, observed=True):
        row: dict[str, object] = {
            "Sample": cage,
            "Cage": cage,
            "n_samples": len(cage_rows),
            "Sex": (
                cage_rows["Sex"].iat[0]
                if "Sex" in cage_rows and cage_rows["Sex"].nunique(dropna=False) == 1
                else "Mixed"
            ),
        }
        for column in ("Bio group", "Date collected", "Collection cohort"):
            if column in cage_rows:
                row[column] = cage_rows[column].iat[0]
        if "Sex" in cage_rows:
            row["Sex composition"] = ";".join(sorted(cage_rows["Sex"].astype(str).unique()))
        rows.append(row)
    cage_metadata = pd.DataFrame(rows)
    cage_metadata["Sample"] = cage_metadata["Cage"]
    return cage_values, cage_metadata


def _icc_one_feature(groups: list[np.ndarray]) -> float:
    """One-way ANOVA ICC from a list of per-cage value arrays.

    ICC = (MS_between - MS_within) / (MS_between + (k - 1) MS_within), with k the
    mean cage size: the share of variance due to differences between cages.
    """
    groups = [group[np.isfinite(group)] for group in groups]
    groups = [group for group in groups if len(group) >= 2]
    if len(groups) < 2:
        return np.nan
    n_total = sum(len(group) for group in groups)
    n_cages = len(groups)
    means = np.asarray([group.mean() for group in groups])
    grand_mean = np.concatenate(groups).mean()
    between_ms = sum(len(group) * (mean - grand_mean) ** 2 for group, mean in zip(groups, means)) / (n_cages - 1)
    within_ss = sum(np.square(group - mean).sum() for group, mean in zip(groups, means))
    within_df = n_total - n_cages
    within_ms = within_ss / within_df if within_df else 0.0
    mean_size = n_total / n_cages
    denominator = between_ms + (mean_size - 1) * within_ms
    if denominator <= 0:
        return 0.0
    return float(np.clip((between_ms - within_ms) / denominator, 0.0, 1.0))


def intraclass_correlation(
    values: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    sample_column: str = "Sample",
    n_bootstraps: int = 1_000,
    random_state: int = 20_260_917,
) -> pd.DataFrame:
    """Estimate cage ICCs with cluster bootstrap intervals.

    Only cages with at least two usable samples contribute. The ICC is the
    one-way random-intercept ANOVA estimator, clipped at zero because a
    negative variance component is not interpretable as a design parameter.
    """
    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be positive.")
    aligned = _aligned_metadata(values, metadata, sample_column=sample_column)
    numeric = values.loc[:, aligned[sample_column]].apply(pd.to_numeric, errors="coerce")
    cage_groups = aligned.groupby("Cage", sort=False, observed=True).groups
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []

    for feature, feature_values in numeric.iterrows():
        groups = [feature_values.loc[samples].to_numpy(dtype=float) for samples in cage_groups.values()]
        groups = [group[np.isfinite(group)] for group in groups]
        groups = [group for group in groups if len(group) >= 2]
        observed = _icc_one_feature(groups)
        # Cluster bootstrap: resample whole cages with replacement.
        bootstrap = np.full(n_bootstraps, np.nan, dtype=float)
        if len(groups) >= 2:
            for number in range(n_bootstraps):
                selected = rng.integers(0, len(groups), size=len(groups))
                bootstrap[number] = _icc_one_feature([groups[index] for index in selected])
        finite = bootstrap[np.isfinite(bootstrap)]
        rows.append({
            "Feature": feature,
            "icc": observed,
            "icc_q025": np.quantile(finite, 0.025) if len(finite) else np.nan,
            "icc_q975": np.quantile(finite, 0.975) if len(finite) else np.nan,
            "n_multi_pup_cages": len(groups),
            "n_usable_samples": sum(len(group) for group in groups),
            "n_bootstraps": n_bootstraps,
            "interval_method": "cage bootstrap percentile interval",
            "design_parameter_caveat": f"{len(groups)} multi-pup cages provide a weak, interval-only design estimate.",
        })
    return pd.DataFrame(rows)


def cage_group_assignments(
    metadata: pd.DataFrame,
    *,
    cage_column: str = "Cage",
    group_column: str = "Bio group",
    target_group: str = "MIA",
) -> list[pd.Series]:
    """Enumerate every whole-cage assignment preserving group counts.

    With 8 cages of which 6 are MIA there are C(8, 6) = 28 ways to choose the MIA
    cages; each returned Series gives one such relabelling for every sample.
    """
    required = {cage_column, group_column}
    missing = required.difference(metadata.columns)
    if missing:
        raise KeyError(f"Metadata is missing columns: {sorted(missing)}")
    cage_info = metadata.loc[:, [cage_column, group_column]].drop_duplicates(cage_column)
    if len(cage_info) != metadata[cage_column].nunique(dropna=False):
        raise ValueError("Each cage must have exactly one group label.")
    cages = cage_info[cage_column].tolist()
    n_target = int(cage_info[group_column].eq(target_group).sum())
    if n_target == 0 or n_target == len(cages):
        raise ValueError("The target group must occupy some but not all cages.")

    assignments: list[pd.Series] = []
    for target_cages in combinations(cages, n_target):
        target_set = set(target_cages)
        labels = {
            cage: (target_group if cage in target_set else "Saline Control")
            for cage in cages
        }
        assignments.append(
            metadata[cage_column].map(labels).rename(group_column)
        )
    return assignments


def cage_permutation_ceiling(
    metadata: pd.DataFrame,
    *,
    cage_column: str = "Cage",
    group_column: str = "Bio group",
    target_group: str = "MIA",
    fdr_denominator: int | None = None,
) -> dict[str, object]:
    """Return the exact assignment count and empirical-p ceiling.

    The smallest p-value an exact cage permutation test can give is
    1 / (number of assignments), e.g. 1/28 = 0.036; with BH over many features
    the smallest attainable q is (number of features) / (number of assignments).
    """
    cage_info = metadata.loc[:, [cage_column, group_column]].drop_duplicates(cage_column)
    n_cages = len(cage_info)
    n_target = int(cage_info[group_column].eq(target_group).sum())
    n_assignments = comb(n_cages, n_target)
    result: dict[str, object] = {
        "exchangeability_unit": "cage",
        "n_cages": n_cages,
        "n_target_group_cages": n_target,
        "n_distinct_assignments": n_assignments,
        "minimum_empirical_p_value": 1 / n_assignments,
        "empirical_p_value_floor": "1 / n_distinct_assignments",
    }
    if fdr_denominator is not None:
        result["minimum_bh_q_value"] = fdr_denominator / n_assignments
        result["fdr_denominator"] = fdr_denominator
    return result


def sex_exchangeability_summary(
    metadata: pd.DataFrame,
    *,
    cage_column: str = "Cage",
    sex_column: str = "Sex",
) -> dict[str, object]:
    """Describe which sex permutation units are available in the design."""
    cage_sexes = metadata.groupby(cage_column, observed=True)[sex_column].nunique(dropna=False)
    return {
        "within_cage": "available",
        "cage_level": "available" if cage_sexes.eq(1).all() else "not_identifiable_mixed_sex_cages",
        "n_mixed_sex_cages": int(cage_sexes.gt(1).sum()),
        "n_single_sex_cages": int(cage_sexes.eq(1).sum()),
    }
