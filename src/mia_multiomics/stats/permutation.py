"""Permutation nulls for the region-wise OLS model (group x sex per feature).

Why: with ~300 correlated features and 20 samples per region, the parametric
p-values and their count of "hits" need a reference.  Shuffling the labels
destroys any real group effect but keeps the correlation between features, so
refitting the model on many shuffles shows how many small p-values appear by
chance alone.

The labels are shuffled at a chosen *exchangeability unit*:

* ``'sample'``      - group labels shuffled between samples, within sex
                      (ignores litters; the classical, too-optimistic null);
* ``'cage'``        - whole litters keep one label; every possible assignment
                      of litters to MIA/control is enumerated (exact test);
* ``'within_cage'`` - labels shuffled inside each litter;
* ``'block'``       - whole blocks (e.g. mice) shuffled within sex.

:func:`run_permutation_null` is the entry point; it returns per-feature
empirical p-values and FDR, per-permutation summaries, and an estimate of the
effective number of independent tests.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .clustered import cage_group_assignments, cage_permutation_ceiling
from .common import bh_adjust, exact_empirical_p

# Column of each effect in the design matrix built by _design_matrix
# (column 0 is the intercept).
EFFECTS = {
    "MIA vs Saline Control": 1,
    "Female vs Male": 2,
    "MIA x Female interaction": 3,
}
# Effects tested by shuffling the group label; the sex effect shuffles sex instead.
_GROUP_PERMUTATION_EFFECTS = {"MIA vs Saline Control", "MIA x Female interaction"}


def _empirical_null_fdr(observed_t: np.ndarray, null_statistics: np.ndarray) -> np.ndarray:
    """Estimate FDR(t) from expected null discoveries at observed thresholds.

    For each observed |t| used as a threshold: FDR = (expected number of null
    features at least this extreme) / (observed number at least this extreme).
    ``null_statistics`` is features x permutations.
    """
    observed_absolute = np.abs(observed_t)
    result = np.full(len(observed_absolute), np.nan, dtype=float)
    finite = np.isfinite(observed_absolute)
    if not finite.any():
        return result
    finite_null = np.isfinite(null_statistics)
    if not finite_null.any():
        return result
    null_absolute = np.abs(null_statistics)
    valid_counts = finite_null.sum(axis=1)
    thresholds = np.sort(np.unique(observed_absolute[finite]))[::-1]
    raw_fdr: list[float] = []
    for threshold in thresholds:
        # Each feature's chance of exceeding the threshold under the null, summed
        # over features = expected number of null discoveries.
        observed_count = np.sum(observed_absolute >= threshold)
        hit_counts = np.sum(finite_null & (null_absolute >= threshold), axis=1)
        per_feature_rates = np.divide(
            hit_counts,
            valid_counts,
            out=np.full(len(valid_counts), np.nan, dtype=float),
            where=valid_counts > 0,
        )
        expected_null_count = np.nansum(per_feature_rates)
        raw_fdr.append(min(1.0, expected_null_count / observed_count) if observed_count else np.nan)
    # q(t) is the minimum FDR over the less-stringent thresholds t' <= t.
    # Thresholds are sorted high-to-low, so this is a suffix minimum.
    monotone_fdr = np.minimum.accumulate(np.asarray(raw_fdr)[::-1])[::-1]
    for threshold, fdr in zip(thresholds, monotone_fdr):
        result[finite & (observed_absolute == threshold)] = fdr
    return result


def permute_group_labels(
    metadata: pd.DataFrame,
    rng: np.random.Generator,
    *,
    group_column: str = "Bio group",
    sex_column: str = "Sex",
    exchangeability_unit: str = "sample",
    block_column: str | None = None,
) -> pd.Series:
    """Permute Bio-group labels within sex at the selected exchangeability unit.

    ``exchangeability_unit='cage'`` reassigns the complete Bio-group labels
    across cages and does not stratify by sex. This is the valid exchangeability
    unit for the cage-level MIA contrast.
    """
    if exchangeability_unit == "cage":
        if block_column is None:
            raise ValueError("block_column is required for cage exchangeability.")
        assignments = cage_group_assignments(
            metadata,
            cage_column=block_column,
            group_column=group_column,
        )
        return assignments[rng.integers(len(assignments))]
    return _permute_column(
        metadata,
        rng,
        target_column=group_column,
        stratify_column=sex_column,
        exchangeability_unit=exchangeability_unit,
        block_column=block_column,
    )


def _permute_column(
    metadata: pd.DataFrame,
    rng: np.random.Generator,
    *,
    target_column: str,
    stratify_column: str,
    exchangeability_unit: str,
    block_column: str | None,
) -> pd.Series:
    """Shuffle ``target_column`` labels at the chosen unit, keeping ``stratify_column`` fixed.

    sample      : shuffle within each stratum (e.g. group within sex)
    within_cage : shuffle within each cage
    cage        : draw one complete, valid assignment of labels to whole cages
    block       : shuffle whole blocks (one label each) within strata
    """
    required = {target_column, stratify_column}
    missing = required.difference(metadata.columns)
    if missing:
        raise KeyError(f"Metadata is missing columns: {sorted(missing)}")
    if exchangeability_unit not in {"sample", "block", "within_cage", "cage"}:
        raise ValueError("exchangeability_unit must be 'sample', 'block', 'within_cage', or 'cage'.")
    if exchangeability_unit in {"block", "within_cage", "cage"} and block_column is None:
        raise ValueError("block_column is required for block, within_cage, or cage exchangeability.")

    labels = metadata[target_column].copy()
    if exchangeability_unit == "sample":
        for _, positions in metadata.groupby(stratify_column, sort=False, observed=True).groups.items():
            labels.loc[positions] = rng.permutation(labels.loc[positions].to_numpy())
        return labels

    assert block_column is not None
    # Whole cages: pick one of the enumerated cage -> group assignments at random.
    if exchangeability_unit == "cage":
        block_info = metadata.loc[:, [block_column, target_column]].drop_duplicates(block_column)
        if len(block_info) != metadata[block_column].nunique(dropna=False):
            raise ValueError("Each cage must have one target label for cage exchangeability.")
        assignments = cage_group_assignments(
            metadata,
            cage_column=block_column,
            group_column=target_column,
        )
        return assignments[rng.integers(len(assignments))]

    if exchangeability_unit == "within_cage":
        for _, positions in metadata.groupby(block_column, sort=False, observed=True).groups.items():
            labels.loc[positions] = rng.permutation(labels.loc[positions].to_numpy())
        return labels

    # Whole blocks: every block must carry a single label and stratum, then the
    # block labels are shuffled within each stratum and copied to their samples.
    block_info = metadata.loc[:, [block_column, stratify_column, target_column]].drop_duplicates(block_column)
    if len(block_info) != metadata[block_column].nunique(dropna=False):
        raise ValueError("Each block must have one stratum and target label.")
    for block, rows in metadata.groupby(block_column, sort=False, observed=True).groups.items():
        block_rows = metadata.loc[rows]
        if block_rows[stratify_column].nunique(dropna=False) != 1 or block_rows[target_column].nunique(dropna=False) != 1:
            raise ValueError("Each block must have one stratum and target label.")
    for _, block_rows in block_info.groupby(stratify_column, sort=False, observed=True):
        block_ids = block_rows[block_column].to_numpy()
        shuffled_targets = rng.permutation(block_rows[target_column].to_numpy())
        for block, target in zip(block_ids, shuffled_targets):
            labels.loc[metadata[block_column].eq(block)] = target
    return labels


def permute_labels_for_effect(
    metadata: pd.DataFrame,
    rng: np.random.Generator,
    effect: str,
    *,
    exchangeability_unit: str = "sample",
    block_column: str | None = None,
) -> pd.DataFrame:
    """Return metadata with the label permutation appropriate to one effect.

    Bio-group and interaction effects preserve sex strata.  The sex effect
    instead permutes sex within Bio-group strata; otherwise its reported null
    would leave the tested label unchanged.
    """
    if effect not in EFFECTS:
        raise ValueError(f"Unknown effect {effect!r}.")
    permuted = metadata.copy()
    if effect in _GROUP_PERMUTATION_EFFECTS:
        permuted["BioGroup"] = permute_group_labels(
            permuted,
            rng,
            group_column="BioGroup",
            sex_column="Sex",
            exchangeability_unit=exchangeability_unit,
            block_column=block_column,
        )
    else:
        permuted["Sex"] = _permute_column(
            permuted,
            rng,
            target_column="Sex",
            stratify_column="BioGroup",
            exchangeability_unit=exchangeability_unit,
            block_column=block_column,
        )
    return permuted


def _design_matrix(metadata: pd.DataFrame, group_values: Iterable[object] | None = None) -> np.ndarray:
    """Columns: intercept, MIA (0/1), female (0/1), MIA x female - the same coding as the OLS formula."""
    if group_values is None:
        group_values = metadata["BioGroup"]
    group = pd.Series(group_values, index=metadata.index)
    sex = metadata["Sex"]
    return np.column_stack((
        np.ones(len(metadata)),
        group.eq("MIA").to_numpy(dtype=float),
        sex.eq("Female").to_numpy(dtype=float),
        (group.eq("MIA") & sex.eq("Female")).to_numpy(dtype=float),
    ))


def _region_statistics(
    values: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    min_per_cell: int,
) -> pd.DataFrame:
    """Calculate OLS statistics for one region with feature-wise missingness.

    The batched calculation is algebraically the same as fitting one OLS model
    per feature, but avoids millions of repeated pandas model setup operations
    during the permutation loop.
    """
    design = _design_matrix(metadata)                     # samples x 4
    numeric = values.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)   # features x samples
    observed = np.isfinite(numeric)

    # Which features have enough values in every group x sex cell to be fitted?
    groups = metadata["BioGroup"].to_numpy()
    sexes = metadata["Sex"].to_numpy()
    cell_masks = np.column_stack([
        (groups == group) & (sexes == sex)
        for group in ("Saline Control", "MIA")
        for sex in ("Male", "Female")
    ])
    cell_counts = observed.astype(int) @ cell_masks.astype(int)
    eligible = cell_counts.min(axis=1) >= min_per_cell
    if not eligible.any():
        return pd.DataFrame(columns=[
            "Feature", "Effect", "estimate_log2", "standard_error",
            "t_statistic", "p_value", "residual_df", "n_usable",
        ])

    # Weighted least squares with 0/1 weights: a missing value gets weight 0,
    # which is the same as dropping that sample for that feature only.
    # Solved for all features at once: beta_f = (X' W_f X)^-1 X' W_f y_f.
    weights = observed[eligible].astype(float)
    response = np.nan_to_num(numeric[eligible], nan=0.0)
    weighted_x = np.einsum("fi,ij,ik->fjk", weights, design, design)
    inverse = np.linalg.pinv(weighted_x)
    weighted_y = np.einsum("fi,ij->fj", weights * response, design)
    coefficients = np.einsum("fij,fj->fi", inverse, weighted_y)
    # Residual variance and standard errors, feature by feature.
    fitted = coefficients @ design.T
    residuals = (response - fitted) * weights
    n_usable = observed[eligible].sum(axis=1).astype(int)
    residual_df = n_usable - design.shape[1]
    residual_variance = np.sum(residuals * residuals, axis=1) / residual_df
    standard_errors = np.sqrt(np.maximum(
        np.diagonal(inverse, axis1=1, axis2=2) * residual_variance[:, None],
        0,
    ))
    # Keep the three effects of interest; t = estimate / SE, two-sided p from Student's t.
    term_indices = list(EFFECTS.values())
    effect_names = list(EFFECTS)
    estimates = coefficients[:, term_indices]
    errors = standard_errors[:, term_indices]
    t_values = np.divide(estimates, errors, out=np.full_like(estimates, np.nan), where=errors > 0)
    p_values = 2 * student_t.sf(np.abs(t_values), residual_df[:, None])
    features = values.index[eligible].to_numpy()
    return pd.DataFrame({
        "Feature": np.repeat(features, len(effect_names)),
        "Effect": np.tile(effect_names, len(features)),
        "estimate_log2": estimates.reshape(-1),
        "standard_error": errors.reshape(-1),
        "t_statistic": t_values.reshape(-1),
        "p_value": p_values.reshape(-1),
        "residual_df": np.repeat(residual_df, len(effect_names)),
        "n_usable": np.repeat(n_usable, len(effect_names)),
    })


def ols_t_statistics(
    values: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    sample_column: str = "Sample",
    region_column: str = "Sample type",
    group_column: str = "Bio group",
    sex_column: str = "Sex",
    min_per_cell: int = 3,
) -> pd.DataFrame:
    """Return OLS test statistics using the same design as ``diffexp.py``."""
    required = {sample_column, region_column, group_column, sex_column}
    missing = required.difference(metadata.columns)
    if missing:
        raise KeyError(f"Metadata is missing columns: {sorted(missing)}")
    if min_per_cell < 1:
        raise ValueError("min_per_cell must be at least 1.")
    ordered = metadata.set_index(sample_column, drop=False).loc[values.columns].copy()
    ordered = ordered.rename(columns={region_column: "Region", group_column: "BioGroup"})
    results: list[pd.DataFrame] = []
    for region, region_metadata in ordered.groupby("Region", sort=False, observed=True):
        samples = region_metadata[sample_column].tolist()
        current = _region_statistics(
            values.loc[:, samples],
            region_metadata.loc[samples, ["BioGroup", sex_column]].rename(columns={sex_column: "Sex"}),
            min_per_cell=min_per_cell,
        )
        if not current.empty:
            current.insert(1, "Region", region)
            results.append(current)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def _null_summary(
    null_statistics: np.ndarray,
    *,
    region: object,
    effect: str,
    permutation_scheme: str,
    permutation_caveat: str,
) -> pd.DataFrame:
    """One row per permutation: quantiles of |t| over all features (how extreme chance gets)."""
    rows: list[dict[str, object]] = []
    for permutation in range(null_statistics.shape[1]):
        current = null_statistics[:, permutation]
        current = current[np.isfinite(current)]
        rows.append({
            "Region": region,
            "Effect": effect,
            "permutation_scheme": permutation_scheme,
            "permutation_caveat": permutation_caveat,
            "Permutation": permutation + 1,
            "n_statistics": len(current),
            "mean_abs_t": np.mean(np.abs(current)) if len(current) else np.nan,
            "p50_abs_t": np.quantile(np.abs(current), 0.50) if len(current) else np.nan,
            "p90_abs_t": np.quantile(np.abs(current), 0.90) if len(current) else np.nan,
            "p95_abs_t": np.quantile(np.abs(current), 0.95) if len(current) else np.nan,
            "p99_abs_t": np.quantile(np.abs(current), 0.99) if len(current) else np.nan,
            "max_abs_t": np.max(np.abs(current)) if len(current) else np.nan,
        })
    return pd.DataFrame(rows)


def estimate_effective_tests(
    observed_effect: pd.DataFrame,
    null_statistics: np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Estimate effective tests from the null discovery-count variance.

    For a family of ``M`` features, ``K`` is the number with nominal null
    p-value below ``alpha`` in each permutation.  The effective-test estimate
    is the effective sample size of the discovery proportion,
    ``M**2 * p * (1-p) / Var(K)``.  It is descriptive and is not used to
    replace the planned BH correction.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one.")
    required = {"Feature", "residual_df"}
    missing = required.difference(observed_effect.columns)
    if missing:
        raise KeyError(f"Observed statistics are missing columns: {sorted(missing)}")
    if null_statistics.ndim != 2 or len(observed_effect) != null_statistics.shape[0]:
        raise ValueError("Observed statistics and null statistics must have matching feature rows.")

    residual_df = pd.to_numeric(observed_effect["residual_df"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(null_statistics) & np.isfinite(residual_df[:, None])
    complete = finite.all(axis=1)
    n_features = int(complete.sum())
    if n_features < 2:
        return {
            "n_features_complete_null": n_features,
            "n_permutations": null_statistics.shape[1],
            "nominal_alpha": alpha,
            "mean_nominal_discoveries": np.nan,
            "variance_nominal_discoveries": np.nan,
            "effective_number_of_tests": np.nan,
            "effective_tests_status": "insufficient complete null features",
        }

    # In every permutation, count the features with nominal p < alpha.  If the
    # features were independent these counts would be binomial(M, alpha); their
    # extra variance measures how correlated the features are.
    complete_null = null_statistics[complete]
    complete_df = residual_df[complete]
    null_p_values = 2 * student_t.sf(np.abs(complete_null), complete_df[:, None])
    discovery_counts = np.sum(null_p_values < alpha, axis=0)
    mean_discoveries = float(np.mean(discovery_counts))
    variance_discoveries = float(np.var(discovery_counts, ddof=1)) if len(discovery_counts) > 1 else np.nan
    nominal_fraction = mean_discoveries / n_features
    # Solve Var(K) = M_eff * p * (1 - p) * (M / M_eff)^2 for M_eff, clipped to [1, M].
    if not np.isfinite(variance_discoveries) or variance_discoveries <= 0:
        effective = float(n_features)
        status = "variance-degenerate; capped at complete null feature count"
    elif nominal_fraction <= 0 or nominal_fraction >= 1:
        effective = np.nan
        status = "nominal null discovery fraction outside estimable range"
    else:
        effective = float(np.clip(
            n_features**2 * nominal_fraction * (1 - nominal_fraction) / variance_discoveries,
            1,
            n_features,
        ))
        status = "estimated from permutation discovery-count variance"
    return {
        "n_features_complete_null": n_features,
        "n_permutations": null_statistics.shape[1],
        "nominal_alpha": alpha,
        "mean_nominal_discoveries": mean_discoveries,
        "variance_nominal_discoveries": variance_discoveries,
        "effective_number_of_tests": effective,
        "effective_tests_status": status,
    }


def run_permutation_null(
    values: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    n_permutations: int = 999,
    random_state: int = 20_260_903,
    min_per_cell: int = 3,
    exchangeability_unit: str = "sample",
    block_column: str | None = None,
    effects: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build an empirical null and empirical p/FDR values per feature.

    The empirical p value for each feature is the two-sided tail probability
    of its absolute OLS t statistic under the matched permuted feature.  BH is
    then applied to those empirical p values within each region/effect family.
    The permutation table stores per-permutation quantiles. Feature rows retain
    null quantiles, valid-draw counts, and exceedance counts so empirical p
    values remain auditable without an opaque multi-million-row dump.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive.")
    if exchangeability_unit not in {"sample", "block", "within_cage", "cage"}:
        raise ValueError("exchangeability_unit must be 'sample', 'block', 'within_cage', or 'cage'.")
    effects_to_run = tuple(EFFECTS) if effects is None else effects
    unknown_effects = set(effects_to_run).difference(EFFECTS)
    if unknown_effects:
        raise ValueError(f"Unknown effects: {sorted(unknown_effects)}")
    required = {"Sample", "Sample type", "Bio group", "Sex"}
    if missing := required.difference(metadata.columns):
        raise KeyError(f"Metadata is missing columns: {sorted(missing)}")

    if exchangeability_unit in {"block", "within_cage", "cage"} and block_column is None:
        raise ValueError("block_column is required for block, within_cage, or cage exchangeability.")

    # Metadata in value-column order with the short names used by the batched OLS.
    ordered = metadata.set_index("Sample", drop=False).loc[values.columns].copy()
    ordered = ordered.rename(columns={"Sample type": "Region", "Bio group": "BioGroup"})
    rng = np.random.default_rng(random_state)
    feature_rows: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    effective_test_rows: list[dict[str, object]] = []

    for region, region_metadata in ordered.groupby("Region", sort=False, observed=True):
        samples = region_metadata["Sample"].tolist()
        region_values = values.loc[:, samples]

        # 1. Observed t statistics with the true labels.
        observed = _region_statistics(
            region_values,
            region_metadata.loc[samples, ["BioGroup", "Sex", *( [block_column] if block_column else [])]],
            min_per_cell=min_per_cell,
        )
        if observed.empty:
            continue
        observed = observed.set_index(["Feature", "Effect"])

        # 2. How many permutations per effect: all possible cage assignments for the
        #    exact cage test (e.g. 28), otherwise n_permutations random shuffles.
        permutation_counts = {
            effect: (
                len(cage_group_assignments(
                    region_metadata,
                    cage_column=block_column,
                    group_column="BioGroup",
                ))
                if exchangeability_unit == "cage" and effect in _GROUP_PERMUTATION_EFFECTS
                else n_permutations
            )
            for effect in effects_to_run
        }
        null_by_effect = {
            effect: np.full((len(values.index), permutation_counts[effect]), np.nan, dtype=float)
            for effect in effects_to_run
        }
        feature_positions = {feature: position for position, feature in enumerate(values.index)}

        # 3. Refit the model on every relabelled copy and store each feature's null t.
        for effect in effects_to_run:
            if exchangeability_unit == "cage" and effect in _GROUP_PERMUTATION_EFFECTS:
                assignments = cage_group_assignments(
                    region_metadata,
                    cage_column=block_column,
                    group_column="BioGroup",
                )
                effect_permutations = [
                    region_metadata.assign(BioGroup=assignment)
                    for assignment in assignments
                ]
            elif exchangeability_unit == "cage":
                raise ValueError(
                    "Cage-level sex permutation is not identifiable when any cage contains multiple sexes."
                )
            else:
                effect_permutations = [
                    permute_labels_for_effect(
                        region_metadata,
                        rng,
                        effect,
                        exchangeability_unit=exchangeability_unit,
                        block_column=block_column,
                    )
                    for _ in range(n_permutations)
                ]
            for permutation, permuted in enumerate(effect_permutations):
                current = _region_statistics(region_values, permuted, min_per_cell=min_per_cell)
                current = current.loc[current["Effect"].eq(effect)]
                if current.empty:
                    continue
                for row in current.itertuples(index=False):
                    null_by_effect[effect][feature_positions[row.Feature], permutation] = row.t_statistic
        # 4. Summarise each effect's null: per-permutation quantiles, effective number
        #    of tests, and per-feature empirical p-values and FDR.
        for effect, null_statistics in null_by_effect.items():
            permutation_scheme = "BioGroup within Sex" if effect in _GROUP_PERMUTATION_EFFECTS else "Sex within BioGroup"
            if exchangeability_unit == "cage":
                permutation_scheme = "BioGroup across cages"
                permutation_caveat = (
                    "Exhaustive cage assignments are used; the minimum empirical p-value is "
                    "the reciprocal of the distinct cage-assignment count."
                )
                ceiling = cage_permutation_ceiling(
                    region_metadata,
                    cage_column=block_column,
                    group_column="BioGroup",
                    fdr_denominator=len(values.index),
                )
            elif exchangeability_unit == "within_cage":
                permutation_scheme = "labels within cage"
                permutation_caveat = "Within-cage permutation preserves cage membership but does not test the cage-level MIA contrast."
                ceiling = {}
            else:
                permutation_caveat = "Sample-level permutation does not preserve cage clustering and is not the cage-aware sensitivity."
                ceiling = {}
            summary_frames.append(_null_summary(
                null_statistics,
                region=region,
                effect=effect,
                permutation_scheme=permutation_scheme,
                permutation_caveat=permutation_caveat,
            ))
            observed_effect = observed.loc[
                observed.index.get_level_values("Effect") == effect
            ].copy()
            observed_effect = observed_effect.reset_index()
            observed_effect.insert(1, "Region", region)
            selected_positions = [feature_positions[feature] for feature in observed_effect["Feature"]]
            effective_test_rows.append({
                "Region": region,
                "Effect": effect,
                "permutation_scheme": permutation_scheme,
                **ceiling,
                **estimate_effective_tests(
                    observed_effect.loc[:, ["Feature", "residual_df"]],
                    null_statistics[selected_positions],
                ),
            })
            # Empirical p = share of null |t| at least as large as the observed |t|
            # ((1 + exceedances) / (1 + permutations) for random shuffles; exact
            # enumeration for the cage test, whose smallest p is 1 / #assignments).
            empirical_p: list[float] = []
            valid_statistics: list[int] = []
            exceedance_counts: list[float] = []
            for feature, observed_t in zip(observed_effect["Feature"], observed_effect["t_statistic"]):
                observed_t = float(observed_t)
                null_t = null_statistics[feature_positions[feature]]
                null_t = null_t[np.isfinite(null_t)]
                exceedance_count = int(np.sum(np.abs(null_t) >= abs(observed_t))) if len(null_t) else np.nan
                empirical_p.append(
                    exact_empirical_p(observed_t, null_t)
                    if exchangeability_unit == "cage" and effect in _GROUP_PERMUTATION_EFFECTS
                    else ((1 + exceedance_count) / (len(null_t) + 1) if len(null_t) else np.nan)
                )
                valid_statistics.append(len(null_t))
                exceedance_counts.append(exceedance_count)
            null_quantiles = []
            for feature in observed_effect["Feature"]:
                null_t = null_statistics[feature_positions[feature]]
                null_t = null_t[np.isfinite(null_t)]
                null_quantiles.append(
                    np.quantile(null_t, [0.025, 0.50, 0.975]) if len(null_t) else [np.nan, np.nan, np.nan]
                )
            null_quantiles = np.asarray(null_quantiles)
            observed_effect["null_t_p02_5"] = null_quantiles[:, 0]
            observed_effect["null_t_p50"] = null_quantiles[:, 1]
            observed_effect["null_t_p97_5"] = null_quantiles[:, 2]
            observed_effect["p_value_empirical"] = empirical_p
            observed_effect["null_exceedance_count"] = exceedance_counts
            observed_effect["fdr_denominator"] = len(values.index)
            observed_effect["p_value_empirical_fdr"] = bh_adjust(
                observed_effect["p_value_empirical"],
                denominator=len(values.index),
            ).to_numpy()
            observed_effect["empirical_null_fdr"] = _empirical_null_fdr(
                observed_effect["t_statistic"].to_numpy(dtype=float),
                null_statistics,
            )
            observed_effect["n_permutations"] = permutation_counts[effect]
            observed_effect["empirical_p_floor"] = (
                1 / permutation_counts[effect] if exchangeability_unit == "cage"
                else 1 / (permutation_counts[effect] + 1)
            )
            observed_effect["n_valid_null_statistics"] = valid_statistics
            observed_effect["exchangeability_unit"] = exchangeability_unit
            observed_effect["permutation_scheme"] = permutation_scheme
            observed_effect["permutation_caveat"] = permutation_caveat
            for key, value in ceiling.items():
                observed_effect[key] = value
            feature_rows.append(observed_effect)

    features = pd.concat(feature_rows, ignore_index=True) if feature_rows else pd.DataFrame()
    if not features.empty and features.duplicated(["Feature", "Effect", "Region"]).any():
        raise AssertionError("Permutation feature keys must be unique within Region x Effect.")
    summaries = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    effective_tests = pd.DataFrame(effective_test_rows)
    return features, summaries, effective_tests
