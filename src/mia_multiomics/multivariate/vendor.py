"""Analysis helpers for reproducing the vendor's exploratory lipidomics plots."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage


def select_top_n_by_nominal_p(
    differential_effects: pd.DataFrame,
    *,
    region: str | None,
    eligible_features: Iterable[object],
    n_features: int = 50,
    effect: str = "MIA vs Saline Control",
) -> pd.Index:
    """Return the top complete features ranked by nominal MIA/control p-value.

    A region-specific heatmap uses that region's p-values. The all-groups
    heatmap uses each feature's smallest p-value across regions as a documented
    proxy because no pooled all-groups differential model is available.
    """
    required = {"Feature", "Region", "Effect", "p_value"}
    missing = required.difference(differential_effects.columns)
    if missing:
        raise KeyError(f"Differential effects are missing columns: {sorted(missing)}")
    if n_features < 1:
        raise ValueError("n_features must be at least one.")

    ranked = differential_effects.loc[
        differential_effects["Effect"].eq(effect), ["Feature", "Region", "p_value"]
    ].copy()
    if region is not None:
        ranked = ranked.loc[ranked["Region"].eq(region)]
    ranked = ranked.loc[ranked["Feature"].isin(eligible_features)]
    ranked["p_value"] = pd.to_numeric(ranked["p_value"], errors="coerce")
    ranked = ranked.dropna(subset=["p_value"])
    ranked = ranked.sort_values(["p_value", "Feature"], kind="stable").drop_duplicates("Feature")
    if len(ranked) < n_features:
        scope = "all groups" if region is None else region
        raise ValueError(f"{scope} has only {len(ranked)} complete features; need {n_features}.")
    return pd.Index(ranked["Feature"].iloc[:n_features], name="Feature")


def cluster_heatmap(values: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Ward-cluster complete feature-by-sample values on both axes.

    Input rows are features and columns are samples. Values must already be
    auto-scaled and complete because no missing values are imputed for plotting.
    """
    numeric = values.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().to_numpy().any():
        raise ValueError("Heatmap clustering requires complete values; do not impute missing values.")
    if numeric.shape[0] < 2 or numeric.shape[1] < 2:
        raise ValueError("Heatmap clustering requires at least two features and two samples.")

    row_linkage = linkage(numeric.to_numpy(dtype=float), method="ward", metric="euclidean")
    column_linkage = linkage(numeric.T.to_numpy(dtype=float), method="ward", metric="euclidean")
    return (
        numeric.iloc[leaves_list(row_linkage), leaves_list(column_linkage)],
        row_linkage,
        column_linkage,
    )


def vendor_column_order(metadata: pd.DataFrame) -> pd.Index:
    """Return the vendor heatmap's imposed BioGroup/sex/sample-number order."""
    required = {"Sample", "Bio group", "Sex", "Sample number"}
    missing = required.difference(metadata.columns)
    if missing:
        raise KeyError(f"Metadata is missing columns: {sorted(missing)}")
    group = pd.Categorical(metadata["Bio group"], ["Saline Control", "MIA"], ordered=True)
    sex = pd.Categorical(metadata["Sex"], ["Male", "Female"], ordered=True)
    ordered = metadata.assign(_group=group, _sex=sex).sort_values(
        ["_group", "_sex", "Sample number"], kind="stable"
    )
    return pd.Index(ordered["Sample"], name="Sample")
