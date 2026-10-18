"""Sample-space PCA: scree variants, score plots coloured by any design
factor, the pooled region/group composite, and the loading-composition plots.

``scope`` is ``'pooled'`` (all kept samples) or a region name.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from .. import config, plotting
from ..multivariate import pca as pca_tools
from ._common import augmented_metadata, class_colors, finish
from .view import SCOPES, View, resolve_scope

FACTOR_COLORS = dict(zip(["Region", "Bio group", "Sex", "Cage", "Collection date", "Tissue mass"], plotting.COLORBLIND_COLORS))


# ---------------------------------------------------------------- scree
def scree(view: View, scope: str = "pooled", n_components: int = 10) -> plt.Figure:
    model, _, _ = view.pca(scope)
    fig = plotting.scree_plot(model.explained_variance_ratio_[:n_components], f"{view.omics}: PCA scree ({resolve_scope(scope)})")
    return finish(fig, view)


def scree_grouped(view: View, n_components: int = 10) -> plt.Figure:
    """Variance explained per component for the pooled PCA and each region, in one plot."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for scope in SCOPES:
        ratio = view.pca(scope)[0].explained_variance_ratio_[:n_components] * 100
        color = plotting.BRAIN_REGION_COLORS.get(scope, "#444444")
        ax.plot(np.arange(1, len(ratio) + 1), ratio, "o-", color=color, label=scope, linewidth=2 if scope == "pooled" else 1.2)
    ax.set(xlabel="component", ylabel="variance explained (%)", title=f"{view.omics}: scree, pooled and per region")
    ax.legend(frameon=False)
    return finish(fig, view)


def _attribution(view: View, scope: str) -> pd.DataFrame:
    """Marginal R² of each design factor for PC1-PC5 of one scope."""
    model, scores, _ = view.pca(scope)
    meta = view.metadata.copy()
    meta["Date collected"] = pd.to_datetime(meta["Date collected"]).dt.strftime("%Y-%m-%d")
    return pca_tools.score_attribution(scores, model.explained_variance_ratio_, meta, "pooled" if resolve_scope(scope) == "pooled" else scope)


def scree_attribution(view: View, scope: str = "pooled") -> plt.Figure:
    """Each PC's variance (grey) and the part of it associated with each design factor (PC variance x R²)."""
    table = _attribution(view, scope)
    total = table.loc[table["Factor"].eq("Total")].set_index("Component")["pc_variance_explained"]
    factors = [f for f in FACTOR_COLORS if f in set(table["Factor"])]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    positions = np.arange(len(total))
    ax.bar(positions, total * 100, 0.85, color="#E5E5E5", label="PC variance")
    width = 0.8 / len(factors)
    for i, factor in enumerate(factors):
        share = table.loc[table["Factor"].eq(factor)].set_index("Component")["factor_share_of_total_variance"].reindex(total.index)
        ax.bar(positions - 0.4 + (i + 0.5) * width, share * 100, width, color=FACTOR_COLORS[factor], label=factor)
    ax.set_xticks(positions, total.index)
    ax.set(ylabel="variance (%)", title=f"{view.omics}: what each component tracks ({resolve_scope(scope)})")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    return finish(fig, view, "Factor bars = PC variance x marginal R²; factors overlap, so bars are not additive.")


# ---------------------------------------------------------------- scores
def pca_scores(view: View, scope: str = "pooled", color_by: str = config.REGION_COL, *, x: str = "PC 1", y: str = "PC 2",
               label: str | None = None, ellipses: bool | None = None) -> plt.Figure:
    """Score plot of any two components, coloured by any metadata column.

    Extra colour options: ``'Tissue mass (binned)'`` (tertiles), ``'Group x sex'``,
    ``'Date collected'``.  ``label='Sample number'`` writes sample numbers next to the points.
    """
    model, scores, _ = view.pca(scope)
    meta = augmented_metadata(view)
    if color_by not in meta:
        raise ValueError(f"color_by must be a metadata column, e.g. {[c for c in meta.columns if c != 'Sample']}.")
    if ellipses is None:
        ellipses = color_by in (config.REGION_COL, config.GROUP_COL, config.SEX_COL, "Group x sex")
    fig = plotting.scores_plot(scores, meta, color_by, variance=model.explained_variance_ratio_, x=x, y=y, label=label,
                               ellipses=ellipses, title=f"{resolve_scope(scope)}: PCA scores by {color_by}")
    return finish(fig, view)


def region_group_composite(view: View) -> plt.Figure:
    """Pooled PCA: region x treatment, cage and collection date on PC1/2 and PC3/4, plus factor R²."""
    model, scores, _ = view.pca("pooled")
    fig = pca_tools.pooled_region_group_composite(scores, model.explained_variance_ratio_, view.metadata, _attribution(view, "pooled"))
    return finish(fig, view)


def pca_hotelling(view: View, region: str = "Cortex") -> plt.Figure:
    """Regional PC1/PC2 scores with 2-D Hotelling T2 limits at 95 % and 99 %, labelled by sample number."""
    model, scores, _ = view.pca(region)
    plane = scores[["PC 1", "PC 2"]]
    curves = [pca_tools.hotelling_ellipse(plane, alpha=a)[0] for a in (0.95, 0.99)]
    fig = plotting.scores_plot(plane, view.metadata, config.GROUP_COL, variance=model.explained_variance_ratio_,
                               title=f"{resolve_scope(region)}: PCA with Hotelling T2 95/99 %", label="Sample number",
                               ellipses=False, outline=curves)
    return finish(fig, view)


# ---------------------------------------------------------------- loadings
def loadings_lollipop(view: View, scope: str = "pooled", component: str = "PC 1", n_features: int | None = None) -> plt.Figure:
    """The strongest loadings (by absolute value) of one component."""
    _, _, loadings = view.pca(scope)
    n_features = n_features or view.options.top_n
    top = loadings[component].abs().nlargest(n_features).index
    fig = plotting.lollipop_plot(loadings.loc[top, component], view.labels,
                                 title=f"{resolve_scope(scope)}: strongest {component} loadings", xlabel=f"{component} loading")
    return finish(fig, view)


def loadings_scatter(view: View, scope: str = "pooled", x: str = "PC 1", y: str = "PC 2", n_labels: int = 10) -> plt.Figure:
    """Loadings of two components, coloured by class; the strongest features are named."""
    model, _, loadings = view.pca(scope)
    lumped, colors = class_colors(view.classes.reindex(loadings.index).fillna("unannotated"))
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, color in colors.items():
        members = lumped.index[lumped.eq(name)]
        ax.scatter(loadings.loc[members, x], loadings.loc[members, y], s=18, color=color, label=name, alpha=0.8)
    strongest = np.hypot(loadings[x], loadings[y]).nlargest(n_labels).index
    for feature in strongest:
        ax.annotate(view.labels.get(feature, feature), (loadings.loc[feature, x], loadings.loc[feature, y]), fontsize=7)
    ax.axhline(0, color="#BBBBBB", linewidth=0.8)
    ax.axvline(0, color="#BBBBBB", linewidth=0.8)
    ratio = model.explained_variance_ratio_
    ax.set(xlabel=f"{x} loading ({ratio[int(x.split()[1]) - 1] * 100:.1f}%)", ylabel=f"{y} loading ({ratio[int(y.split()[1]) - 1] * 100:.1f}%)",
           title=f"{resolve_scope(scope)}: loadings by class")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    return finish(fig, view)


def loadings_heatmap(view: View, scope: str = "pooled", n_features: int | None = None, n_components: int = 4) -> plt.Figure:
    """Loadings of the features that matter most for PC1-PC4 (ranked by summed squared loading)."""
    _, _, loadings = view.pca(scope)
    n_features = n_features or view.options.top_n
    block = loadings.iloc[:, :n_components]
    top = block.pow(2).sum(axis=1).nlargest(n_features).index
    shown = block.loc[top]
    limit = float(np.abs(shown.to_numpy()).max())
    fig, ax = plt.subplots(figsize=(5 + 0.5 * n_components, 0.28 * n_features + 1.8))
    image = ax.imshow(shown.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    names = [f"{view.labels.get(f, f)}  [{view.classes.get(f, '')}]" for f in top]
    ax.set_yticks(range(len(top)), names, fontsize=7)
    ax.set_xticks(range(n_components), shown.columns)
    ax.grid(False)
    ax.set_title(f"{resolve_scope(scope)}: top {n_features} loadings")
    fig.colorbar(image, ax=ax, label="loading", fraction=0.05)
    return finish(fig, view)


def loadings_class_composition(view: View, scope: str = "pooled", n_largest: int = 6) -> plt.Figure:
    """Share of each component's squared loading carried by each lipid class / pathway."""
    _, _, loadings = view.pca(scope)
    composition = pca_tools.class_composition(loadings.iloc[:, :4], view.classes.reindex(loadings.index), n_largest=n_largest)
    components = list(loadings.columns[:4])
    names = [n for n in composition["Class"].drop_duplicates() if n != "Other"] + ["Other"]
    palette = {n: plt.get_cmap("tab20")(i % 20) for i, n in enumerate(names)} | {"Other": "#BBBBBB"}
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = pd.Series(0.0, index=components)
    for name in names:
        share = composition.loc[composition["Class"].eq(name)].set_index("Component")["squared_loading_share"].reindex(components, fill_value=0)
        color = palette[name]
        ax.bar(components, share * 100, bottom=bottom * 100, color=color, label=name)
        bottom += share
    ax.set(ylabel="share of component variance (%)", ylim=(0, 100), title=f"{resolve_scope(scope)}: loading composition by class")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    return finish(fig, view, "Compare with the panel: a class carrying more than its share of features is enriched in the component.")


def loadings_cumulative_share(view: View, scope: str = "pooled") -> plt.Figure:
    """How concentrated each component is: cumulative squared loading of its strongest features."""
    _, _, loadings = view.pca(scope)
    curve = pca_tools.cumulative_share_curve(loadings.iloc[:, :4])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, (component, rows) in enumerate(curve.groupby("Component", sort=False)):
        ax.plot(rows["n_features"], rows["cumulative_squared_loading_share"] * 100, color=plotting.COLORBLIND_COLORS[i], label=component)
    first = curve.loc[curve["Component"].eq(curve["Component"].iloc[0])]
    ax.plot(first["n_features"], first["uniform_share"] * 100, "--", color="#777777", label="all features equal")
    ax.set(xscale="log", xlabel="number of strongest features", ylabel="cumulative component variance (%)", ylim=(0, 100),
           title=f"{resolve_scope(scope)}: loading concentration")
    ax.legend(frameon=False)
    return finish(fig, view)


def loadings_top_features(view: View, scope: str = "pooled", n_features: int | None = None) -> plt.Figure:
    """Contribution of one shared set of top features to PC1-PC4."""
    _, _, loadings = view.pca(scope)
    n_features = min(n_features or view.options.top_n, 20)
    components = list(loadings.columns[:4])
    squared = loadings[components].pow(2)
    selected = squared.sum(axis=1).nlargest(n_features).index
    colors = [plt.get_cmap("tab20")(i) for i in range(n_features)]
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = pd.Series(0.0, index=components)
    for feature, color in zip(selected, colors):
        share = squared.loc[feature] / squared.sum()
        ax.bar(components, share * 100, bottom=bottom * 100, color=color)
        bottom += share
    ax.bar(components, (1 - bottom) * 100, bottom=bottom * 100, color="#DDDDDD")
    handles = [Patch(color=c, label=view.labels.get(f, f)) for f, c in zip(selected, colors)] + [Patch(color="#DDDDDD", label="all other features")]
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left", ncol=2, fontsize=7, frameon=False)
    ax.set(ylabel="share of component variance (%)", ylim=(0, 100), title=f"{resolve_scope(scope)}: top {n_features} features across PC1-PC4")
    return finish(fig, view)
