"""Region-wise differential lipidomics models and pooled mixed-model sensitivity checks."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .common import bh_adjust

FdrDenominator = Literal["fitted", "all_features"]

# The planned model, fitted per feature within one region.  Control and male are
# the reference levels, so the three coefficients below read as
#   MIA vs control (in males), female vs male (in controls), and the interaction
#   (how much the MIA effect differs between females and males).
_FORMULA = "value ~ C(BioGroup, Treatment(reference='Saline Control')) * C(Sex, Treatment(reference='Male'))"
_EFFECTS = {
    "MIA vs Saline Control": "C(BioGroup, Treatment(reference='Saline Control'))[T.MIA]",
    "Female vs Male": "C(Sex, Treatment(reference='Male'))[T.Female]",
    "MIA x Female interaction": "C(BioGroup, Treatment(reference='Saline Control'))[T.MIA]:C(Sex, Treatment(reference='Male'))[T.Female]",
}


@dataclass(frozen=True)
class DifferentialResults:
    """Fitted effects and per-feature eligibility diagnostics."""

    effects: pd.DataFrame
    feature_qc: pd.DataFrame


def _validate_inputs(values: pd.DataFrame, metadata: pd.DataFrame, sample_column: str) -> None:
    """Every sample column of ``values`` needs exactly one metadata row."""
    if sample_column not in metadata:
        raise KeyError(f"Metadata is missing {sample_column!r}.")
    if metadata[sample_column].duplicated().any():
        raise ValueError("Metadata sample identifiers must be unique.")
    missing = set(values.columns).difference(metadata[sample_column])
    if missing:
        raise ValueError(f"Metadata is missing intensity samples: {sorted(missing)}")


def fit_regionwise_differential(
    values: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    sample_column: str = "Sample",
    region_column: str = "Sample type",
    group_column: str = "Bio group",
    sex_column: str = "Sex",
    annotations: pd.DataFrame | None = None,
    min_per_cell: int = 3,
    fdr_denominator: FdrDenominator = "fitted",
    effect_scale: str = "normalized_log2",
) -> DifferentialResults:
    """Fit the planned OLS model independently for each lipid within region.

    ``values`` must be the log2, median-normalized feature-by-sample matrix.
    Missing values remain missing: a feature is fitted only when each of the
    four BioGroup x Sex cells has at least ``min_per_cell`` usable samples.
    """
    _validate_inputs(values, metadata, sample_column)
    required = {region_column, group_column, sex_column}
    missing_columns = required.difference(metadata)
    if missing_columns:
        raise KeyError(f"Metadata is missing columns: {sorted(missing_columns)}")
    if min_per_cell < 1:
        raise ValueError("min_per_cell must be at least 1.")
    if fdr_denominator not in {"fitted", "all_features"}:
        raise ValueError("fdr_denominator must be 'fitted' or 'all_features'.")
    if effect_scale not in {"normalized_log2", "pareto_scaled_log2"}:
        raise ValueError("effect_scale must be 'normalized_log2' or 'pareto_scaled_log2'.")

    # Metadata in the same sample order as the value columns, with short column names for the formula.
    ordered_metadata = metadata.set_index(sample_column, drop=False).loc[values.columns].copy()
    ordered_metadata = ordered_metadata.rename(
        columns={region_column: "Region", group_column: "BioGroup", sex_column: "Sex"}
    )
    groups = ("Saline Control", "MIA")
    sexes = ("Male", "Female")
    feature_qc: list[dict[str, object]] = []
    fitted_effects: list[dict[str, object]] = []

    # One model per (region, feature).
    for region, region_metadata in ordered_metadata.groupby("Region", sort=False, observed=True):
        samples = region_metadata[sample_column].tolist()
        for feature, row in values.loc[:, samples].iterrows():
            # Long table for this feature: one row per sample with group, sex and value.
            analysis = region_metadata.loc[:, [sample_column, "BioGroup", "Sex"]].copy()
            analysis["value"] = pd.to_numeric(row.reindex(samples), errors="coerce").to_numpy()
            # How many non-missing values each group x sex cell has (missing values are never imputed).
            cell_counts = {
                f"n_{group.replace(' ', '_').lower()}_{sex.lower()}": int(
                    analysis.loc[
                        analysis["BioGroup"].eq(group) & analysis["Sex"].eq(sex), "value"
                    ].notna().sum()
                )
                for group in groups
                for sex in sexes
            }
            n_observed = int(analysis["value"].notna().sum())
            qc: dict[str, object] = {
                "Feature": feature,
                "Region": region,
                "n_region_samples": len(samples),
                "n_usable": n_observed,
                "n_missing_after_preprocessing": len(samples) - n_observed,
                **cell_counts,
            }
            # Too few values in any cell: the interaction model is not estimable, so skip and record why.
            if min(cell_counts.values()) < min_per_cell:
                qc["fit_status"] = "excluded"
                qc["exclusion_reason"] = f"fewer than {min_per_cell} usable observations in at least one BioGroup x Sex cell"
                feature_qc.append(qc)
                continue

            # Ordinary least squares on the observed values only.
            model = smf.ols(_FORMULA, data=analysis.dropna(subset=["value"])).fit()
            qc["fit_status"] = "fitted"
            qc["exclusion_reason"] = ""
            feature_qc.append(qc)
            for effect, term in _EFFECTS.items():
                fitted_effects.append({
                    "Feature": feature,
                    "Region": region,
                    "Effect": effect,
                    "estimate_log2": model.params[term],
                    "effect_scale": effect_scale,
                    "standard_error": model.bse[term],
                    "ci_lower_log2": model.conf_int().loc[term, 0],
                    "ci_upper_log2": model.conf_int().loc[term, 1],
                    "fold_change": 2 ** model.params[term] if effect_scale == "normalized_log2" else np.nan,
                    "p_value": model.pvalues[term],
                    "residual_df": model.df_resid,
                    "n_usable": n_observed,
                    **cell_counts,
                })

    qc_frame = pd.DataFrame(feature_qc)
    effects = pd.DataFrame(fitted_effects)
    if effects.empty:
        return DifferentialResults(effects=effects, feature_qc=qc_frame)
    if annotations is not None:
        if annotations.index.has_duplicates:
            raise ValueError("Annotation index must uniquely identify features.")
        effects = effects.join(annotations, on="Feature", how="left")

    # Benjamini-Hochberg FDR, separately for every region x effect family.  With
    # 'all_features' the denominator is the whole panel (features that could not
    # be fitted count as tests that were not significant), which is conservative.
    effects["p_value_fdr"] = np.nan
    effects["fdr_denominator"] = np.nan
    for _, index in effects.groupby(["Region", "Effect"], sort=False, observed=True).groups.items():
        subset = effects.loc[index, "p_value"]
        denominator = len(subset) if fdr_denominator == "fitted" else len(values.index)
        effects.loc[index, "p_value_fdr"] = bh_adjust(subset, denominator)
        effects.loc[index, "fdr_denominator"] = denominator
    return DifferentialResults(effects=effects, feature_qc=qc_frame)


def fit_pooled_mixed_sensitivity(
    values: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    sample_column: str = "Sample",
    mouse_column: str = "Mouse number",
    region_column: str = "Sample type",
    group_column: str = "Bio group",
    sex_column: str = "Sex",
) -> pd.DataFrame:
    """Fit exploratory pooled random-intercept models, one lipid at a time.

    Failures are returned as status rows rather than silently omitted. These
    models are directional sensitivity checks, not primary inferential tests.
    """
    _validate_inputs(values, metadata, sample_column)
    required = {mouse_column, region_column, group_column, sex_column}
    missing_columns = required.difference(metadata)
    if missing_columns:
        raise KeyError(f"Metadata is missing columns: {sorted(missing_columns)}")
    design = metadata.set_index(sample_column, drop=False).loc[values.columns].copy().rename(
        columns={mouse_column: "Mouse", region_column: "Region", group_column: "BioGroup", sex_column: "Sex"}
    )
    # All three regions in one model: group x region and group x sex fixed effects,
    # and a random intercept per mouse because each mouse contributes three samples.
    group, sex = "C(BioGroup, Treatment('Saline Control'))", "C(Sex, Treatment('Male'))"
    formula = f"value ~ {group} * C(Region) + {group} * {sex}"
    rows: list[dict[str, object]] = []
    for feature, values_row in values.iterrows():
        analysis = design.copy()
        analysis["value"] = pd.to_numeric(values_row, errors="coerce").to_numpy()
        analysis = analysis.dropna(subset=["value"])
        model, failure = None, ""
        for method in ("lbfgs", "powell"):        # L-BFGS fails at a zero mouse variance; Powell copes
            try:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always")
                    model = smf.mixedlm(formula, analysis, groups=analysis["Mouse"]).fit(reml=False, method=method, disp=False)
                    _ = model.bse
                break
            except (ValueError, np.linalg.LinAlgError) as error:
                model, failure = None, str(error)
        if model is None:
            rows.append({"Feature": feature, "Term": "", "estimate_log2": np.nan, "standard_error": np.nan, "p_value": np.nan,
                         "n_usable": len(analysis), "fit_status": "failed", "failure_reason": failure})
            continue
        # Keep convergence warnings with the result instead of printing them.
        warning_text = " | ".join(str(warning.message) for warning in caught_warnings)
        fit_status = "fitted" if model.converged else "not_converged"
        for term, estimate in model.params.items():
            if term == "Group Var":
                continue
            rows.append({"Feature": feature, "Term": term, "estimate_log2": estimate, "standard_error": model.bse[term],
                         "p_value": model.pvalues[term], "n_usable": len(analysis), "fit_status": fit_status,
                         "warning_message": warning_text})
    return pd.DataFrame(rows)
