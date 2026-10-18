"""Paired regional contrasts and regional-signature persistence statistics.

Every mouse gives all three regions, so a region difference (e.g. cortex minus
cerebellum) can be computed *within* each mouse: litter, sex and handling
effects cancel.  The question is then whether MIA changes these within-mouse
differences.

Main steps (used by ``stages/region_persistence.py``):

1. :func:`build_region_contrasts` - one mouse x feature matrix of differences per region pair;
2. :func:`concordance_summary` - correlation of the control and MIA mean difference profiles;
3. :func:`estimate_split_half_reliability` - how reproducible each group's profile is
   (a profile measured with noise cannot correlate perfectly even with itself);
4. :func:`disattenuated_concordance` - the correlation divided by that noise ceiling,
   with a bootstrap interval (1.0 = profile fully preserved);
5. :func:`one_sample_region_tests` and :func:`interaction_tests` - per-feature tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, t

from .common import bh_adjust, safe_correlation

REGION_CONTRASTS = (
    ("Cortex - Cerebellum", "Cortex", "Cerebellum"),
    ("Cortex - Hippocampus", "Cortex", "Hippocampus"),
    ("Hippocampus - Cerebellum", "Hippocampus", "Cerebellum"),
)
GROUPS = ("Saline Control", "MIA")


@dataclass(frozen=True)
class RegionContrast:
    """One pairwise region-difference matrix and its mouse-level metadata."""

    name: str
    region_a: str
    region_b: str
    values: pd.DataFrame
    metadata: pd.DataFrame
    dropped_mice: tuple[str, ...] = ()

    @property
    def n_mice(self) -> int:
        """Number of mice with both regions of this contrast."""
        return len(self.metadata)


def _validate_matrix_and_metadata(
    values: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    sample_column: str,
    mouse_column: str,
    region_column: str,
    group_column: str,
) -> None:
    """Unique feature and sample IDs, metadata for every sample, and at most one sample per mouse and region."""
    required = {sample_column, mouse_column, region_column, group_column}
    missing = required.difference(metadata.columns)
    if missing:
        raise KeyError(f"Metadata is missing columns: {sorted(missing)}")
    if values.index.has_duplicates:
        raise ValueError("Feature identifiers must be unique.")
    if values.columns.has_duplicates:
        raise ValueError("Sample identifiers in values must be unique.")
    if metadata[sample_column].duplicated().any():
        raise ValueError("Metadata sample identifiers must be unique.")
    missing_samples = values.columns.difference(metadata[sample_column])
    if len(missing_samples):
        raise ValueError(f"Metadata is missing samples: {missing_samples.tolist()}")
    if metadata[[mouse_column, region_column]].duplicated().any():
        raise ValueError("Each mouse must have at most one sample per region.")


def build_region_contrasts(
    values: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    sample_column: str = "Sample",
    mouse_column: str = "Mouse number",
    region_column: str = "Sample type",
    group_column: str = "Bio group",
    regions: tuple[str, ...] = ("Cerebellum", "Cortex", "Hippocampus"),
    contrast_definitions: tuple[tuple[str, str, str], ...] = REGION_CONTRASTS,
) -> dict[str, RegionContrast]:
    """Pair mice separately for each contrast's two regions.

    A missing regional sample removes only pairs involving that region.
    Missing feature values remain missing and are handled feature-wise by the
    downstream tests.
    """
    _validate_matrix_and_metadata(
        values,
        metadata,
        sample_column=sample_column,
        mouse_column=mouse_column,
        region_column=region_column,
        group_column=group_column,
    )
    if len(set(regions)) != len(regions):
        raise ValueError("regions must contain unique region names.")
    missing_regions = set(regions).difference(metadata[region_column].dropna().unique())
    if missing_regions:
        raise ValueError(f"Metadata is missing regions: {sorted(missing_regions)}")

    # A mouse must have the same group and cage in all of its regions.
    for mouse, rows in metadata.groupby(mouse_column, observed=True):
        for column in (group_column, "Cage"):
            if column in rows and rows[column].nunique(dropna=False) != 1:
                raise ValueError(f"Mouse {mouse!r} has inconsistent {column} labels across regions.")
    available = metadata.loc[
        metadata[sample_column].isin(values.columns) & metadata[region_column].isin(regions)
    ]
    region_rows = {
        region: available.loc[available[region_column].eq(region)].set_index(mouse_column)
        for region in regions
    }

    result: dict[str, RegionContrast] = {}
    for name, region_a, region_b in contrast_definitions:
        # Mice that have both regions of this pair; others are left out of this pair only.
        if region_a not in region_rows or region_b not in region_rows:
            raise ValueError(f"Contrast {name!r} refers to an unknown region.")
        mouse_order = sorted(region_rows[region_a].index.intersection(region_rows[region_b].index), key=str)
        dropped_mice = tuple(sorted(set(metadata[mouse_column]).difference(mouse_order), key=str))
        left = region_rows[region_a].loc[mouse_order]
        right = region_rows[region_b].loc[mouse_order]
        sample_a = left[sample_column].tolist()
        sample_b = right[sample_column].tolist()
        # Region A minus region B, column by column for the same mouse (log2 scale = log ratio).
        difference = values.loc[:, sample_a].copy()
        difference.columns = [str(mouse) for mouse in mouse_order]
        difference = difference.subtract(values.loc[:, sample_b].set_axis(difference.columns, axis=1))

        # One metadata row per mouse (group, sex, cage, date, tissue masses of both samples).
        pair_metadata = pd.DataFrame(index=difference.columns)
        pair_metadata.index.name = "Mouse"
        pair_metadata["Sample"] = difference.columns
        pair_metadata["Mouse number"] = mouse_order
        pair_metadata["Bio group"] = left[group_column].to_numpy()
        pair_metadata["Sample A"] = sample_a
        pair_metadata["Sample B"] = sample_b
        if "Sex" in left.columns:
            pair_metadata["Sex"] = left["Sex"].to_numpy()
        if "Cage" in left.columns:
            pair_metadata["Cage"] = left["Cage"].to_numpy()
        if "Date collected" in left.columns:
            pair_metadata["Date collected"] = left["Date collected"].to_numpy()
        if "Collection cohort" in left.columns:
            pair_metadata["Collection cohort"] = left["Collection cohort"].to_numpy()
        mass_column = "Tissue mass (mg)"
        if mass_column in left.columns and mass_column in right.columns:
            pair_metadata["Tissue mass A (mg)"] = pd.to_numeric(left[mass_column], errors="coerce").to_numpy()
            pair_metadata["Tissue mass B (mg)"] = pd.to_numeric(right[mass_column], errors="coerce").to_numpy()
            pair_metadata["Tissue mass difference (mg)"] = (
                pair_metadata["Tissue mass A (mg)"] - pair_metadata["Tissue mass B (mg)"]
            )
        result[name] = RegionContrast(
            name=name,
            region_a=region_a,
            region_b=region_b,
            values=difference,
            metadata=pair_metadata.reset_index(drop=True),
            dropped_mice=dropped_mice,
        )
    return result


def _profile_summaries(left: pd.Series, right: pd.Series) -> dict[str, float]:
    """Pearson and Spearman r and the regression slope of the MIA profile on the control profile."""
    valid = pd.concat([left.rename("control"), right.rename("mia")], axis=1).dropna()
    result = {
        "n_features": len(valid),
        "pearson_r": np.nan,
        "spearman_r": np.nan,
        "slope_mia_on_control": np.nan,
    }
    if len(valid) < 3:
        return result
    control = valid["control"].to_numpy(dtype=float)
    mia = valid["mia"].to_numpy(dtype=float)
    if np.std(control, ddof=1) > 0 and np.std(mia, ddof=1) > 0:
        result["pearson_r"] = float(pearsonr(control, mia).statistic)
        result["spearman_r"] = float(spearmanr(control, mia).statistic)
    variance = np.var(control, ddof=1)
    if variance > 0:
        result["slope_mia_on_control"] = float(np.cov(control, mia, ddof=1)[0, 1] / variance)
    return result


def concordance_summary(
    contrast: RegionContrast,
    *,
    group_column: str = "Bio group",
) -> pd.DataFrame:
    """Summarize agreement between the two group mean difference profiles.

    Each profile is the mean over the group's mice of the per-feature region
    difference; the correlation over features says how similar the two are.
    """
    metadata = contrast.metadata
    groups = metadata[group_column]
    control = contrast.values.loc[:, metadata.loc[groups.eq(GROUPS[0]), "Sample"]].mean(axis=1)
    mia = contrast.values.loc[:, metadata.loc[groups.eq(GROUPS[1]), "Sample"]].mean(axis=1)
    profile = _profile_summaries(control, mia)
    return pd.DataFrame([{
        "Contrast": contrast.name,
        "Region A": contrast.region_a,
        "Region B": contrast.region_b,
        "n_mice_control": int(groups.eq(GROUPS[0]).sum()),
        "n_mice_mia": int(groups.eq(GROUPS[1]).sum()),
        "median_abs_effect_control": float(control.abs().median(skipna=True)),
        "median_abs_effect_mia": float(mia.abs().median(skipna=True)),
        **profile,
    }])


def one_sample_region_tests(
    contrast: RegionContrast,
    *,
    group_column: str = "Bio group",
    fdr_denominator: int | None = None,
) -> pd.DataFrame:
    """Run one-sample t tests on each group's paired region differences.

    Tests, per group and feature, whether the mean within-mouse difference is
    zero (= does the feature differ between the two regions in this group?).
    """
    if fdr_denominator is None:
        fdr_denominator = len(contrast.values.index)
    rows: list[dict[str, object]] = []
    for group in GROUPS:
        samples = contrast.metadata.loc[contrast.metadata[group_column].eq(group), "Sample"]
        for feature, row in contrast.values.loc[:, samples].iterrows():
            observed = pd.to_numeric(row, errors="coerce").dropna().to_numpy(dtype=float)
            n = len(observed)
            mean = float(np.mean(observed)) if n else np.nan
            standard_deviation = float(np.std(observed, ddof=1)) if n >= 2 else np.nan
            standard_error = standard_deviation / np.sqrt(n) if n >= 2 else np.nan
            # Edge cases: too few mice (no test) or zero variance (t is 0 or infinite).
            if n < 2 or not np.isfinite(standard_error):
                statistic = p_value = ci_lower = ci_upper = np.nan
                degrees_of_freedom = n - 1 if n else np.nan
            elif standard_error == 0:
                statistic = 0.0 if mean == 0 else np.sign(mean) * np.inf
                p_value = 1.0 if mean == 0 else 0.0
                degrees_of_freedom = n - 1
                ci_lower = ci_upper = mean
            else:
                degrees_of_freedom = n - 1
                statistic = mean / standard_error
                p_value = float(2 * t.sf(abs(statistic), degrees_of_freedom))
                critical = float(t.ppf(0.975, degrees_of_freedom))
                ci_lower = mean - critical * standard_error
                ci_upper = mean + critical * standard_error
            rows.append({
                "Contrast": contrast.name,
                "Region A": contrast.region_a,
                "Region B": contrast.region_b,
                "Group": group,
                "Feature": feature,
                "n_mice": n,
                "estimate_log2": mean,
                "standard_deviation": standard_deviation,
                "standard_error": standard_error,
                "t_statistic": statistic,
                "degrees_of_freedom": degrees_of_freedom,
                "ci_lower_log2": ci_lower,
                "ci_upper_log2": ci_upper,
                "p_value": p_value,
            })
    results = pd.DataFrame(rows)
    results["p_value_fdr"] = np.nan
    for _, indices in results.groupby(["Contrast", "Group"], observed=True).groups.items():
        results.loc[indices, "p_value_fdr"] = bh_adjust(results.loc[indices, "p_value"], fdr_denominator)
    results["fdr_denominator"] = fdr_denominator
    return results


def _welch_statistics(
    control: np.ndarray,
    mia: np.ndarray,
    *,
    control_design_effect: float = 1.0,
    mia_design_effect: float = 1.0,
) -> dict[str, float]:
    """Welch's t test of MIA minus control, with optional variance inflation per group.

    A design effect > 1 inflates a group's variance to account for mice of one
    litter being correlated (design effect = 1 + (litter size - 1) x ICC).
    Degrees of freedom by the Welch-Satterthwaite formula.
    """
    control = control[np.isfinite(control)]
    mia = mia[np.isfinite(mia)]
    n_control, n_mia = len(control), len(mia)
    result = {
        "n_control": n_control,
        "n_mia": n_mia,
        "mean_control": float(np.mean(control)) if n_control else np.nan,
        "mean_mia": float(np.mean(mia)) if n_mia else np.nan,
        "estimate_log2": np.nan,
        "standard_error": np.nan,
        "t_statistic": np.nan,
        "degrees_of_freedom": np.nan,
        "ci_lower_log2": np.nan,
        "ci_upper_log2": np.nan,
        "p_value": np.nan,
    }
    if n_control < 2 or n_mia < 2:
        return result
    estimate = result["mean_mia"] - result["mean_control"]
    control_variance = float(np.var(control, ddof=1))
    mia_variance = float(np.var(mia, ddof=1))
    control_component = control_variance * control_design_effect / n_control
    mia_component = mia_variance * mia_design_effect / n_mia
    variance = control_component + mia_component
    result["estimate_log2"] = estimate
    if variance == 0:
        result.update({
            "standard_error": 0.0,
            "t_statistic": 0.0 if estimate == 0 else np.sign(estimate) * np.inf,
            "degrees_of_freedom": n_control + n_mia - 2,
            "p_value": 1.0 if estimate == 0 else 0.0,
            "ci_lower_log2": estimate,
            "ci_upper_log2": estimate,
        })
        return result
    degrees_of_freedom = variance**2 / (
        control_component**2 / (n_control - 1) + mia_component**2 / (n_mia - 1)
    )
    standard_error = np.sqrt(variance)
    statistic = estimate / standard_error
    critical = float(t.ppf(0.975, degrees_of_freedom))
    result.update({
        "standard_error": standard_error,
        "t_statistic": statistic,
        "degrees_of_freedom": degrees_of_freedom,
        "ci_lower_log2": estimate - critical * standard_error,
        "ci_upper_log2": estimate + critical * standard_error,
        "p_value": float(2 * t.sf(abs(statistic), degrees_of_freedom)),
    })
    return result


def interaction_tests(
    contrast: RegionContrast,
    icc: pd.DataFrame | None = None,
    *,
    group_column: str = "Bio group",
    cluster_column: str = "Cage",
    fdr_denominator: int | None = None,
) -> pd.DataFrame:
    """Test one group contrast of paired differences, with ICC sensitivities.

    The region x MIA interaction for each feature: does the within-mouse region
    difference differ between MIA and control mice?  Reported four times: as a
    plain Welch test ('raw') and with the variance inflated by the litter ICC
    (point estimate and both ends of its interval).
    """
    if fdr_denominator is None:
        fdr_denominator = len(contrast.values.index) * len(REGION_CONTRASTS)
    metadata = contrast.metadata
    control_samples = metadata.loc[metadata[group_column].eq(GROUPS[0]), "Sample"]
    mia_samples = metadata.loc[metadata[group_column].eq(GROUPS[1]), "Sample"]
    # Mean litter size per group, needed for the design effect.
    mean_cluster_size: dict[str, float] = {group: 1.0 for group in GROUPS}
    if cluster_column in metadata.columns:
        cluster_sizes = metadata.groupby([group_column, cluster_column], observed=True).size()
        for group in GROUPS:
            sizes = cluster_sizes.loc[group] if group in cluster_sizes.index.get_level_values(0) else pd.Series(dtype=float)
            if len(sizes):
                mean_cluster_size[group] = float(sizes.mean())

    icc_indexed = None
    if icc is not None:
        if "Feature" not in icc.columns:
            raise KeyError("ICC table must contain a Feature column.")
        icc_indexed = icc.set_index("Feature")

    scenario_names = ("raw", "icc_point", "icc_lower", "icc_upper")
    rows: list[dict[str, object]] = []
    for feature in contrast.values.index:
        control = pd.to_numeric(contrast.values.loc[feature, control_samples], errors="coerce").to_numpy(dtype=float)
        mia = pd.to_numeric(contrast.values.loc[feature, mia_samples], errors="coerce").to_numpy(dtype=float)
        row: dict[str, object] = {
            "Contrast": contrast.name,
            "Region A": contrast.region_a,
            "Region B": contrast.region_b,
            "Feature": feature,
            "interaction_definition": "MIA minus Saline Control mean of paired region differences",
        }
        raw = _welch_statistics(control, mia)
        for key, value in raw.items():
            row[f"{key}_raw"] = value
        point_icc = lower_icc = upper_icc = np.nan
        if icc_indexed is not None and feature in icc_indexed.index:
            point_icc = float(icc_indexed.loc[feature, "icc"])
            lower_icc = float(icc_indexed.loc[feature, "icc_q025"])
            upper_icc = float(icc_indexed.loc[feature, "icc_q975"])
        row.update({"icc": point_icc, "icc_q025": lower_icc, "icc_q975": upper_icc})
        # Repeat the test with the variance inflated by each ICC scenario.
        for scenario, current_icc in zip(scenario_names[1:], (point_icc, lower_icc, upper_icc)):
            if np.isfinite(current_icc):
                control_de = 1 + (mean_cluster_size[GROUPS[0]] - 1) * current_icc
                mia_de = 1 + (mean_cluster_size[GROUPS[1]] - 1) * current_icc
            else:
                control_de = mia_de = np.nan
            adjusted = _welch_statistics(
                control,
                mia,
                control_design_effect=control_de,
                mia_design_effect=mia_de,
            ) if np.isfinite(control_de) else {key: np.nan for key in raw}
            row[f"design_effect_control_{scenario}"] = control_de
            row[f"design_effect_mia_{scenario}"] = mia_de
            for key, value in adjusted.items():
                row[f"{key}_{scenario}"] = value
        rows.append(row)
    results = pd.DataFrame(rows)
    p_columns = ["p_value_raw", "p_value_icc_point", "p_value_icc_lower", "p_value_icc_upper"]
    for p_column in p_columns:
        q_column = p_column.replace("p_value", "q_value")
        results[q_column] = bh_adjust(results[p_column], fdr_denominator)
    results["fdr_denominator"] = fdr_denominator
    return results


def apply_interaction_fdr(
    results: pd.DataFrame,
    *,
    n_features: int,
    n_contrasts: int = len(REGION_CONTRASTS),
) -> pd.DataFrame:
    """Apply the declared shared FDR family across all region contrasts."""
    denominator = n_features * n_contrasts
    results = results.copy()
    for p_column in ("p_value_raw", "p_value_icc_point", "p_value_icc_lower", "p_value_icc_upper"):
        q_column = p_column.replace("p_value", "q_value")
        results[q_column] = bh_adjust(results[p_column], denominator)
    results["fdr_denominator"] = denominator
    results["fdr_family"] = f"all {n_contrasts} region contrasts"
    return results


def estimate_split_half_reliability(
    contrast: RegionContrast,
    *,
    n_splits: int = 1_000,
    random_state: int = 20_260_919,
    group_column: str = "Bio group",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate full-profile reliability by repeated mouse-level split halves.

    The Spearman-Brown correction converts each split-half correlation into an
    estimate for the full group mean profile used by the concordance statistic.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be positive.")
    rng = np.random.default_rng(random_state)
    summary_rows: list[dict[str, object]] = []
    draw_rows: list[dict[str, object]] = []
    for group in GROUPS:
        samples = contrast.metadata.loc[contrast.metadata[group_column].eq(group), "Sample"].tolist()
        if len(samples) < 4:
            valid = np.array([], dtype=float)
        else:
            correlations: list[float] = []
            # Split the group's mice randomly in two halves, correlate the two half-profiles,
            # and step the correlation up to full size with Spearman-Brown: 2r / (1 + r).
            for split in range(n_splits):
                shuffled = rng.permutation(samples)
                midpoint = len(shuffled) // 2
                left = contrast.values.loc[:, shuffled[:midpoint]].mean(axis=1)
                right = contrast.values.loc[:, shuffled[midpoint:]].mean(axis=1)
                correlation = safe_correlation(left, right)
                correlations.append(correlation)
                draw_rows.append({
                    "Contrast": contrast.name,
                    "Group": group,
                    "split": split,
                    "correlation": correlation,
                    "reliability": 2 * correlation / (1 + correlation) if np.isfinite(correlation) and correlation > -1 else np.nan,
                    "n_mice": len(samples),
                })
            valid = np.asarray(correlations, dtype=float)
            valid = valid[np.isfinite(valid)]
            reliability_values = 2 * valid / (1 + valid)
            reliability_values = reliability_values[np.isfinite(reliability_values)]
        summary_rows.append({
            "Contrast": contrast.name,
            "Group": group,
            "n_mice": len(samples),
            "n_splits": n_splits,
            "n_valid_splits": len(valid),
            "split_half_correlation_mean": float(np.mean(valid)) if len(valid) else np.nan,
            "reliability": float(np.mean(reliability_values)) if len(valid) else np.nan,
            "reliability_mean": float(np.mean(reliability_values)) if len(valid) else np.nan,
            "reliability_median": float(np.median(reliability_values)) if len(valid) else np.nan,
            "reliability_q025": float(np.quantile(reliability_values, 0.025)) if len(valid) else np.nan,
            "reliability_q975": float(np.quantile(reliability_values, 0.975)) if len(valid) else np.nan,
            "random_state": random_state,
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(draw_rows)


def _array_profile_mean(values: np.ndarray) -> np.ndarray:
    """Feature-wise observed means, including NaN for an unobserved feature."""
    counts = np.isfinite(values).sum(axis=1)
    return np.divide(np.nansum(values, axis=1), counts,
                     out=np.full(values.shape[0], np.nan), where=counts > 0)


def _array_profile_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Pearson correlation over the features observed in both profiles (NaN if fewer than 3)."""
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return np.nan
    left = left[valid] - left[valid].mean()
    right = right[valid] - right[valid].mean()
    denominator = np.sqrt(np.dot(left, left) * np.dot(right, right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else np.nan


def _split_half_reliability(
    values: np.ndarray, mouse_ids: np.ndarray, n_splits: int, rng: np.random.Generator,
) -> float:
    """Mean Spearman–Brown reliability, splitting unique IDs, never copies.

    Bootstrap multiplicities remain as weights within a half. They must not
    create shared mice between halves, which would inflate reliability.
    """
    mouse_ids = np.asarray(mouse_ids)
    if values.shape[1] != len(mouse_ids) or n_splits < 1:
        raise ValueError("Mouse IDs must align with columns and n_splits must be positive.")
    unique = np.unique(mouse_ids)
    if len(unique) < 2:
        return np.nan
    reliabilities = []
    for _ in range(n_splits):
        shuffled = rng.permutation(unique)
        in_left = np.isin(mouse_ids, shuffled[:len(unique) // 2])
        correlation = _array_profile_correlation(
            _array_profile_mean(values[:, in_left]),
            _array_profile_mean(values[:, ~in_left]),
        )
        if np.isfinite(correlation) and correlation > -1:
            reliabilities.append(2 * correlation / (1 + correlation))
    return float(np.mean(reliabilities)) if reliabilities else np.nan


def disattenuated_concordance(
    contrast: RegionContrast,
    reliability_summary: pd.DataFrame,
    reliability_draws: pd.DataFrame,
    *,
    n_bootstraps: int = 1_000,
    n_inner_splits: int = 100,
    random_state: int = 20_260_919,
    group_column: str = "Bio group",
) -> pd.DataFrame:
    """Bootstrap mice and recompute ID-aware reliabilities inside each draw.

    reliability_draws remains in the signature for compatibility, but the
    interval no longer samples original-data reliability draws.
    """
    if n_bootstraps < 1 or n_inner_splits < 1:
        raise ValueError("Bootstrap and inner split counts must be positive.")
    # Point estimate: observed r divided by the attenuation ceiling sqrt(rel_control x rel_mia),
    # the largest correlation two noisy profiles could reach if the true profiles were identical.
    observed = concordance_summary(contrast, group_column=group_column).iloc[0]
    relevant = reliability_summary.loc[reliability_summary["Contrast"].eq(contrast.name)].set_index("Group")
    point = {group: float(relevant.loc[group, "reliability_mean"]) for group in GROUPS}
    ceiling = np.sqrt(np.prod([max(point[group], 0.0) for group in GROUPS]))
    estimate = float(observed["pearson_r"] / ceiling) if ceiling > 0 else np.nan

    rng = np.random.default_rng(random_state)
    group_samples = {
        group: contrast.metadata.loc[contrast.metadata[group_column].eq(group), "Sample"].tolist()
        for group in GROUPS
    }
    group_values = {
        group: contrast.values.loc[:, group_samples[group]].to_numpy(dtype=float)
        for group in GROUPS
    }
    # Bootstrap: resample mice within each group, and recompute both the correlation and the
    # reliabilities inside every draw (reusing the original reliabilities would understate
    # the uncertainty).
    bootstrap_estimates: list[float] = []
    for _ in range(n_bootstraps):
        profiles = {}
        reliability_values = {}
        for group in GROUPS:
            n_mice = len(group_samples[group])
            selected = rng.integers(0, n_mice, size=n_mice)
            resampled = group_values[group][:, selected]
            profiles[group] = _array_profile_mean(resampled)
            reliability_values[group] = _split_half_reliability(
                resampled, selected, n_inner_splits, rng,
            )
        correlation = _array_profile_correlation(profiles[GROUPS[0]], profiles[GROUPS[1]])
        bootstrap_ceiling = np.sqrt(np.prod([max(reliability_values[group], 0.0) for group in GROUPS]))
        if bootstrap_ceiling > 0 and np.isfinite(correlation):
            bootstrap_estimates.append(correlation / bootstrap_ceiling)
    bootstrap = np.asarray(bootstrap_estimates, dtype=float)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975]) if len(bootstrap) else (np.nan, np.nan)
    return pd.DataFrame([{
        "Contrast": contrast.name,
        "Region A": contrast.region_a,
        "Region B": contrast.region_b,
        "r_observed": observed["pearson_r"],
        "reliability_control": point[GROUPS[0]],
        "reliability_mia": point[GROUPS[1]],
        "attenuation_ceiling": ceiling,
        "disattenuated_concordance": estimate,
        "bootstrap_q025": lower,
        "bootstrap_q975": upper,
        "point_inside_interval": bool(lower <= estimate <= upper),
        "n_inner_splits": n_inner_splits,
        "n_bootstraps": n_bootstraps,
        "n_valid_bootstraps": len(bootstrap),
        "bootstrap_random_state": random_state,
        "bootstrap_method": "within-group mouse resampling; ID-aware reliability recomputed per draw; mean Spearman-Brown",
        "reliability_negative_values_clipped_at_zero": True,
    }])
