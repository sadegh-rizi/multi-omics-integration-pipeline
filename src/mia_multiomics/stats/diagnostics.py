"""P-value calibration and residual diagnostics for differential results."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import levene, shapiro

from .common import safe_correlation
from .permutation import ols_t_statistics

P_VALUE_THRESHOLDS = (0.05, 0.01, 0.001)
_FORMULA = "value ~ C(BioGroup, Treatment(reference='Saline Control')) * C(Sex, Treatment(reference='Male'))"


def summarize_p_values(
    effects: pd.DataFrame,
    *,
    source: str,
    p_column: str = "p_value",
) -> pd.DataFrame:
    """Count observed and expected small p values by region and effect."""
    required = {"Region", "Effect", p_column}
    if missing := required.difference(effects.columns):
        raise KeyError(f"Differential results are missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for (region, effect), group in effects.groupby(["Region", "Effect"], sort=False, observed=True):
        p_values = pd.to_numeric(group[p_column], errors="coerce").dropna()
        for threshold in P_VALUE_THRESHOLDS:
            observed = int(p_values.lt(threshold).sum())
            expected = len(p_values) * threshold
            rows.append({
                "Source": source,
                "Region": region,
                "Effect": effect,
                "p_threshold": threshold,
                "n_features": len(p_values),
                "observed_count": observed,
                "expected_uniform_count": expected,
                "observed_expected_ratio": observed / expected if expected else np.nan,
            })
    return pd.DataFrame(rows)




def _diagnostic_features(values: pd.DataFrame, count: int) -> list[object]:
    if count < 1:
        raise ValueError("count must be positive.")
    if values.empty:
        return []
    positions = np.linspace(0, len(values.index) - 1, min(count, len(values.index)), dtype=int)
    return values.index[positions].tolist()


def residual_diagnostics(
    values: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    count: int = 12,
    min_per_cell: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assess residual normality and four-cell variance homogeneity.

    Twelve features are selected at deterministic evenly spaced positions in
    the filtered feature panel.  This is a diagnostic sample, not a selected
    subset for inference.
    """
    selected = _diagnostic_features(values, count)
    selected_table = pd.DataFrame({"Feature": selected, "diagnostic_selection": "evenly spaced feature-panel positions"})
    rows: list[dict[str, object]] = []
    metadata = metadata.set_index("Sample", drop=False).loc[values.columns]
    for region, region_metadata in metadata.groupby("Sample type", sort=False, observed=True):
        samples = region_metadata["Sample"].tolist()
        for feature in selected:
            analysis = region_metadata.loc[:, ["Sample", "Bio group", "Sex"]].copy().rename(
                columns={"Bio group": "BioGroup"}
            )
            analysis["value"] = pd.to_numeric(values.loc[feature, samples], errors="coerce").to_numpy()
            cell_values = [
                analysis.loc[analysis["BioGroup"].eq(group) & analysis["Sex"].eq(sex), "value"].dropna().to_numpy()
                for group in ("Saline Control", "MIA")
                for sex in ("Male", "Female")
            ]
            cell_counts = [len(current) for current in cell_values]
            row: dict[str, object] = {
                "Feature": feature,
                "Region": region,
                "n_usable": int(analysis["value"].notna().sum()),
                "min_cell_n": min(cell_counts),
                "shapiro_p_value": np.nan,
                "levene_p_value": np.nan,
                "residual_sd": np.nan,
                "max_abs_studentized_residual": np.nan,
                "fit_status": "excluded",
            }
            if min(cell_counts) < min_per_cell:
                rows.append(row)
                continue
            model = smf.ols(_FORMULA, data=analysis.dropna(subset=["value"])).fit()
            residuals = model.resid.to_numpy(dtype=float)
            row.update({
                "shapiro_p_value": shapiro(residuals).pvalue if 3 <= len(residuals) <= 5000 else np.nan,
                "levene_p_value": levene(*cell_values, center="median").pvalue,
                "residual_sd": np.std(residuals, ddof=1),
                "max_abs_studentized_residual": np.max(np.abs(model.get_influence().resid_studentized_internal)),
                "fit_status": "fitted",
                "residual_df": model.df_resid,
            })
            rows.append(row)
    return pd.DataFrame(rows), selected_table


def sample_influence_summary(
    values: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    min_per_cell: int = 3,
) -> pd.DataFrame:
    """Summarize leave-one-sample-out changes in OLS standard errors."""
    metadata = metadata.set_index("Sample", drop=False).loc[values.columns]
    rows: list[dict[str, object]] = []
    for region, region_metadata in metadata.groupby("Sample type", sort=False, observed=True):
        samples = region_metadata["Sample"].tolist()
        region_values = values.loc[:, samples]
        full = ols_t_statistics(region_values, region_metadata.reset_index(drop=True), min_per_cell=min_per_cell)
        if full.empty:
            continue
        full = full.set_index(["Feature", "Effect"])
        for sample in samples:
            remaining_metadata = region_metadata.loc[region_metadata["Sample"].ne(sample)].reset_index(drop=True)
            remaining_values = region_values.drop(columns=sample)
            leave_one_out = ols_t_statistics(remaining_values, remaining_metadata, min_per_cell=min_per_cell)
            if leave_one_out.empty:
                rows.extend({
                    "Sample": sample,
                    "Region": region,
                    "Effect": effect,
                    "n_features": 0,
                    "median_relative_se_change": np.nan,
                    "p95_relative_se_change": np.nan,
                    "max_relative_se_change": np.nan,
                    "fraction_se_increased": np.nan,
                    "fraction_se_increased_10pct": np.nan,
                    "fit_status": "not_estimable_after_sample_removal",
                } for effect in full.index.get_level_values("Effect").unique())
                continue
            leave_one_out = leave_one_out.set_index(["Feature", "Effect"])
            comparable = full[["standard_error"]].join(
                leave_one_out[["standard_error"]].rename(columns={"standard_error": "loo_standard_error"}),
                how="inner",
            )
            comparable = comparable.loc[comparable["standard_error"].gt(0)]
            if comparable.empty:
                rows.extend({
                    "Sample": sample,
                    "Region": region,
                    "Effect": effect,
                    "n_features": 0,
                    "median_relative_se_change": np.nan,
                    "p95_relative_se_change": np.nan,
                    "max_relative_se_change": np.nan,
                    "fraction_se_increased": np.nan,
                    "fraction_se_increased_10pct": np.nan,
                    "fit_status": "not_comparable_after_sample_removal",
                } for effect in full.index.get_level_values("Effect").unique())
                continue
            relative_change = comparable["loo_standard_error"].div(comparable["standard_error"]).sub(1)
            for effect in relative_change.index.get_level_values("Effect").unique():
                current = relative_change.xs(effect, level="Effect")
                rows.append({
                    "Sample": sample,
                    "Region": region,
                    "Effect": effect,
                    "n_features": len(current),
                    "median_relative_se_change": current.median(),
                    "p95_relative_se_change": current.quantile(0.95),
                    "max_relative_se_change": current.max(),
                    "fraction_se_increased": np.mean(current.gt(0)),
                    "fraction_se_increased_10pct": np.mean(current.gt(0.10)),
                    "fit_status": "fitted",
                })
    return pd.DataFrame(rows)


def cross_check_qc_tables(variance: pd.DataFrame | None, effects: pd.DataFrame) -> pd.DataFrame:
    """Do features with noisier measurement (higher technical/biological variance ratio) get smaller |t|?"""
    if variance is None or effects.empty:
        return pd.DataFrame([{"metric": "abs_t_vs_variance_ratio_spearman_r", "value": np.nan,
                              "note": "No technical-variance table available."}])
    differential = effects.assign(abs_t_statistic=effects["estimate_log2"].abs() / effects["standard_error"])
    joined = differential.merge(variance.loc[:, ["Feature", "Region", "variance_ratio"]], on=["Feature", "Region"], how="inner")
    rows = []
    for (region, effect), group in joined.groupby(["Region", "Effect"], sort=False, observed=True):
        valid = group.dropna(subset=["abs_t_statistic", "variance_ratio"])
        rows.append({
            "metric": "abs_t_vs_variance_ratio_spearman_r", "Region": region, "Effect": effect,
            "n_features_joined": len(valid),
            "value": safe_correlation(valid["abs_t_statistic"], valid["variance_ratio"], method="spearman"),
            "note": "Diagnostic association, not an inferential test.",
        })
    return pd.DataFrame(rows)


def missingness_se_crosscheck(
    feature_qc: pd.DataFrame,
    effects: pd.DataFrame,
    detection_table: pd.DataFrame,
) -> pd.DataFrame:
    """Check whether missing or detection-masked cells track model SEs."""
    required_qc = {"Feature", "Region", "n_missing_after_preprocessing"}
    required_effects = {"Feature", "Region", "Effect", "standard_error"}
    if required_qc.difference(feature_qc.columns) or required_effects.difference(effects.columns):
        return pd.DataFrame([{
            "metric": "feature_join",
            "value": np.nan,
            "note": "Required differential QC/effect columns were unavailable.",
        }])
    qc = feature_qc.loc[:, ["Feature", "Region", "n_missing_after_preprocessing"]].copy()
    effect_table = effects.loc[:, ["Feature", "Region", "Effect", "standard_error"]].copy()
    qc["Feature_key"] = qc["Feature"].astype(str)
    effect_table["Feature_key"] = effect_table["Feature"].astype(str)
    merged = effect_table.merge(qc.loc[:, ["Feature_key", "Region", "n_missing_after_preprocessing"]], on=["Feature_key", "Region"], how="left")
    if not detection_table.empty and {"Feature", "Sample type"}.issubset(detection_table.columns):
        detection_counts = detection_table.groupby(["Feature", "Sample type"], observed=True).size().rename("n_detection_masked").reset_index()
        detection_counts["Feature_key"] = detection_counts["Feature"].astype(str)
        merged = merged.merge(
            detection_counts.loc[:, ["Feature_key", "Sample type", "n_detection_masked"]].rename(columns={"Sample type": "Region"}),
            on=["Feature_key", "Region"],
            how="left",
        )
    else:
        merged["n_detection_masked"] = np.nan
    rows: list[dict[str, object]] = []
    for (region, effect), group in merged.groupby(["Region", "Effect"], sort=False, observed=True):
        for metric in ("n_missing_after_preprocessing", "n_detection_masked"):
            valid = group.dropna(subset=["standard_error", metric])
            correlation = safe_correlation(valid["standard_error"], valid[metric], method='spearman')
            rows.append({
                "Region": region,
                "Effect": effect,
                "metric": f"standard_error_vs_{metric}_spearman_r",
                "n_features": len(valid),
                "value": correlation,
                "note": "Diagnostic association; missingness and detection masking remain distinct from imputation.",
            })
    return pd.DataFrame(rows)
