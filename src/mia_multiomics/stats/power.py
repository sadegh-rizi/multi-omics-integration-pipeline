"""Detectable-effect-size summaries and honest volcano plots."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

MIA_EFFECT = "MIA vs Saline Control"


def fdr_hit_table(effects: pd.DataFrame, *, top_n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return all FDR hits and the top labels for each Region x Effect family."""
    required = {"Feature", "Region", "Effect", "p_value", "p_value_fdr"}
    if missing := required.difference(effects.columns):
        raise KeyError(f"Differential results are missing columns: {sorted(missing)}")
    if top_n < 1:
        raise ValueError("top_n must be positive.")
    hits = effects.loc[pd.to_numeric(effects["p_value_fdr"], errors="coerce").lt(0.05)].copy()
    if hits.empty:
        hits["hit_rank"] = pd.Series(dtype="Int64")
        hits["annotate"] = pd.Series(dtype=bool)
        hits["label"] = pd.Series(dtype=str)
        return hits, hits.copy()
    hits["_feature_sort_key"] = hits["Feature"].astype(str)
    hits = hits.sort_values(
        ["Region", "Effect", "p_value_fdr", "p_value", "_feature_sort_key"],
        kind="mergesort",
    )
    hits["hit_rank"] = hits.groupby(["Region", "Effect"], sort=False, observed=True).cumcount() + 1
    name = hits["Lipid Name"] if "Lipid Name" in hits.columns else hits["Feature"].astype(str)
    hits["label"] = name.where(name.notna() & name.astype(str).ne(""), hits["Feature"].astype(str))
    hits["annotate"] = hits["hit_rank"].le(top_n)
    hits = hits.drop(columns="_feature_sort_key")
    return hits, hits.loc[hits["annotate"]].copy()


def detectable_effect_sizes(
    effects: pd.DataFrame,
    *,
    alpha: float = 0.05,
    target_power: float = 0.80,
    illustrative_litter_n: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate feature-level and summary nominal 80%-power MDEs.

    The observed standard error is treated as fixed and the two-sided normal
    approximation is used: ``(z_(1-alpha/2) + z_power) * SE``.  The litter
    row is only an illustrative scaling of the observed SE to n=8; it is not a
    litter-aware model or a replacement for the deferred analysis.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one.")
    if not 0 < target_power < 1:
        raise ValueError("target_power must be between zero and one.")
    if illustrative_litter_n < 1:
        raise ValueError("illustrative_litter_n must be positive.")
    required = {"Feature", "Region", "Effect", "standard_error"}
    if missing := required.difference(effects.columns):
        raise KeyError(f"Differential results are missing columns: {sorted(missing)}")
    selected = effects.loc[effects["Effect"].eq(MIA_EFFECT)].copy()
    selected["standard_error"] = pd.to_numeric(selected["standard_error"], errors="coerce")
    selected = selected.loc[selected["standard_error"].gt(0)].copy()
    z_multiplier = norm.ppf(1 - alpha / 2) + norm.ppf(target_power)
    selected["mde_log2"] = z_multiplier * selected["standard_error"]
    if "effect_scale" not in selected.columns:
        selected["effect_scale"] = "normalized_log2"
    selected["power_scale"] = "observed_sample_level"
    selected["alpha"] = alpha
    selected["target_power"] = target_power

    summary_rows: list[dict[str, object]] = []
    for region, region_effects in selected.groupby("Region", sort=False, observed=True):
        median_se = region_effects["standard_error"].median()
        if "n_usable" in region_effects:
            observed_n = pd.to_numeric(region_effects["n_usable"], errors="coerce").median()
        else:
            observed_n = np.nan
        summary_rows.extend([
            {
                "Region": region,
                "power_scale": "observed_sample_level",
                "n_features": len(region_effects),
                "median_standard_error": median_se,
                "mde_p10_log2": region_effects["mde_log2"].quantile(0.10),
                "mde_median_log2": region_effects["mde_log2"].median(),
                "mde_p90_log2": region_effects["mde_log2"].quantile(0.90),
                "observed_n_median": observed_n,
                "alpha": alpha,
                "target_power": target_power,
            },
            {
                "Region": region,
                "power_scale": "illustrative_litter_n",
                "n_features": len(region_effects),
                "median_standard_error": median_se * np.sqrt(observed_n / illustrative_litter_n) if observed_n > 0 else np.nan,
                "mde_p10_log2": region_effects["mde_log2"].quantile(0.10) * np.sqrt(observed_n / illustrative_litter_n) if observed_n > 0 else np.nan,
                "mde_median_log2": region_effects["mde_log2"].median() * np.sqrt(observed_n / illustrative_litter_n) if observed_n > 0 else np.nan,
                "mde_p90_log2": region_effects["mde_log2"].quantile(0.90) * np.sqrt(observed_n / illustrative_litter_n) if observed_n > 0 else np.nan,
                "observed_n_median": observed_n,
                "illustrative_litter_n": illustrative_litter_n,
                "alpha": alpha,
                "target_power": target_power,
                "note": "Illustration only; litter-aware model is deferred and litter identity is unresolved.",
            },
        ])
    return selected, pd.DataFrame(summary_rows)
