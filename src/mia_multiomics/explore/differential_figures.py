"""Per-feature models: volcano plots, p-value calibration, top hits, single-feature
profiles, forest plots and lipid-class effects.

All use :meth:`View.differential` - ``log2 value ~ group * sex`` fitted per
feature within each region, BH-FDR over the whole panel - on the view's
samples and features, so changing the exclusions or thresholds changes them.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config, plotting
from ..lipids import aggregate_class_profiles
from ..stats.differential import fit_regionwise_differential
from ._common import EFFECTS, MIA_EFFECT, finish
from .view import View, resolve_scope

P_THRESHOLDS = (0.001, 0.01, 0.05)


def top_hits(view: View, effect: str | None = MIA_EFFECT, n: int | None = None) -> pd.DataFrame:
    """The features with the smallest p-values (per region) for one effect (or all effects)."""
    effects = view.differential()
    if effect is not None:
        effects = effects.loc[effects["Effect"].eq(effect)]
    columns = ["Region", "Effect", "Feature", "Label", "Class", "estimate_log2", "ci_lower_log2", "ci_upper_log2",
               "p_value", "p_value_fdr", "n_usable"]
    return (effects.sort_values("p_value").groupby(["Region", "Effect"]).head(n or view.options.top_n)[columns]
            .sort_values(["Effect", "Region", "p_value"]).reset_index(drop=True))


def hit_counts(view: View) -> pd.DataFrame:
    """FDR hits per region and effect at the chosen alpha, with the small-p excess."""
    effects = view.differential()
    alpha = view.options.fdr_alpha
    return (effects.groupby(["Region", "Effect"]).agg(
        n_tested=("p_value", "size"), n_fdr_hits=("p_value_fdr", lambda q: int((q <= alpha).sum())),
        smallest_q=("p_value_fdr", "min"), n_p_below_0_01=("p_value", lambda p: int((p < 0.01).sum())))
        .assign(expected_p_below_0_01=lambda t: 0.01 * t["n_tested"]).reset_index())


def _volcano_panel(ax, block: pd.DataFrame, alpha: float, n_labels: int) -> None:
    """Draw one volcano (effect vs -log10 p) into ``ax``; FDR hits in orange, smallest p labelled."""
    hit = block["p_value_fdr"].le(alpha)
    log_p = -np.log10(block["p_value"])
    ax.scatter(block["estimate_log2"], log_p, s=9, color="#999999", alpha=0.6)
    ax.scatter(block.loc[hit, "estimate_log2"], log_p[hit], s=16, color=plotting.COLORBLIND_COLORS[1], label=f"q <= {alpha:g}")
    if hit.any():
        ax.axhline(-np.log10(block.loc[hit, "p_value"].max()), color=plotting.COLORBLIND_COLORS[1], linestyle=":", linewidth=0.8)
    for _, row in block.nsmallest(n_labels, "p_value").iterrows():
        ax.annotate(row["Label"], (row["estimate_log2"], -np.log10(row["p_value"])), fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax.axvline(0, color="#BBBBBB", linewidth=0.8)


def volcano(view: View, region: str = "Cortex", effect: str = MIA_EFFECT, n_labels: int = 5) -> plt.Figure:
    """Effect size against -log10 p for one region and effect; FDR hits in orange."""
    effects = view.differential()
    block = effects.loc[effects["Region"].eq(resolve_scope(region)) & effects["Effect"].eq(effect)]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    _volcano_panel(ax, block, view.options.fdr_alpha, n_labels)
    n_hits = int(block["p_value_fdr"].le(view.options.fdr_alpha).sum())
    ax.set(xlabel="log2 effect", ylabel="-log10 p", title=f"{resolve_scope(region)}: {effect} ({n_hits} FDR hits)")
    return finish(fig, view, "Dotted line: p-value of the weakest FDR hit. Labels: smallest p-values.")


def volcano_grid(view: View, n_labels: int = 3) -> plt.Figure:
    """Volcano plots for every region (columns) and effect (rows)."""
    effects = view.differential()
    fig, axes = plt.subplots(len(EFFECTS), len(config.REGIONS), figsize=(13, 10), squeeze=False)
    for i, effect in enumerate(EFFECTS):
        for j, region in enumerate(config.REGIONS):
            block = effects.loc[effects["Region"].eq(region) & effects["Effect"].eq(effect)]
            _volcano_panel(axes[i, j], block, view.options.fdr_alpha, n_labels)
            axes[i, j].set_title(f"{region}\n{effect}", fontsize=9)
            axes[i, j].set(xlabel="log2 effect", ylabel="-log10 p")
    fig.tight_layout()
    return finish(fig, view)


def p_value_histograms(view: View) -> plt.Figure:
    """p-value distributions; under no effect they are flat (dashed line)."""
    effects = view.differential()
    fig, axes = plt.subplots(len(config.REGIONS), len(EFFECTS), figsize=(12, 8.5), squeeze=False, layout="constrained")
    bins = np.linspace(0, 1, 21)
    for i, region in enumerate(config.REGIONS):
        for j, effect in enumerate(EFFECTS):
            p = effects.loc[effects["Region"].eq(region) & effects["Effect"].eq(effect), "p_value"].dropna()
            ax = axes[i, j]
            ax.hist(p, bins=bins, density=True, color=plotting.COLORBLIND_COLORS[0], alpha=0.8)
            ax.axhline(1, color="#333333", linestyle="--", linewidth=0.8)
            ax.set(xlim=(0, 1), title=f"{region}: {effect}", xlabel="p-value", ylabel="density")
            ax.title.set_fontsize(9)
    return finish(fig, view, "A peak near 0 means more small p-values than chance; a flat histogram means none.")


def small_p_counts(view: View, thresholds: tuple[float, ...] = P_THRESHOLDS) -> plt.Figure:
    """Observed number of p-values below each threshold against the number expected by chance."""
    effects = view.differential()
    groups = effects.groupby(["Region", "Effect"], sort=False)
    labels = [f"{r}\n{e}" for (r, e) in groups.groups]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, len(thresholds), figsize=(4.5 * len(thresholds), 4.8), layout="constrained")
    for ax, threshold in zip(np.atleast_1d(axes), thresholds):
        observed = [int((rows["p_value"] < threshold).sum()) for _, rows in groups]
        expected = [threshold * len(rows) for _, rows in groups]
        ax.bar(x - 0.2, observed, 0.4, color=plotting.COLORBLIND_COLORS[1], label="observed")
        ax.bar(x + 0.2, expected, 0.4, color="white", edgecolor="#333333", label="expected by chance")
        ax.set_xticks(x, labels, rotation=90, fontsize=7)
        ax.set(title=f"p < {threshold:g}", ylabel="features")
    np.atleast_1d(axes)[0].legend(frameon=False)
    return finish(fig, view)


def _find_feature(view: View, feature) -> object:
    """Accept a feature ID or a readable name (e.g. "Cer(d18:0_14:0)"); it must survive the filters."""
    if feature in view.values.index:
        return feature
    matches = view.labels.index[view.labels.eq(str(feature))]
    matches = [m for m in matches if m in view.values.index]
    if not matches:
        raise ValueError(f"{feature!r} is not among the {len(view.values)} features kept with these settings.")
    return matches[0]


SHORT_EFFECT = {MIA_EFFECT: "MIA", "Female vs Male": "F vs M", "MIA x Female interaction": "MIA x F"}


def feature_profile(view: View, feature, effect: str = MIA_EFFECT) -> plt.Figure:
    """Every value of one feature by region, group and sex, with the estimate of ``effect`` per region."""
    feature = _find_feature(view, feature)
    data = view.values.loc[feature].rename("value").to_frame().join(view.metadata.set_index("Sample"))
    effects = view.differential()
    fitted = effects.loc[effects["Feature"].eq(feature) & effects["Effect"].eq(effect)].set_index("Region")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    cells = [(g, s) for g in config.GROUPS for s in config.SEXES]
    rng = np.random.default_rng(0)
    for r, region in enumerate(config.REGIONS):
        for c, (group, sex) in enumerate(cells):
            values = data.loc[data[config.REGION_COL].eq(region) & data[config.GROUP_COL].eq(group) & data[config.SEX_COL].eq(sex), "value"].dropna()
            position = r + (c - 1.5) * 0.18
            ax.scatter(position + rng.uniform(-0.04, 0.04, len(values)), values, s=22, marker=plotting.SEX_MARKERS[sex],
                       color=plotting.BIO_GROUP_COLORS[group], edgecolor="black", linewidth=0.3)
            if len(values):
                ax.hlines(values.mean(), position - 0.07, position + 0.07, color="black", linewidth=1.5)
        if region in fitted.index:
            row = fitted.loc[region]
            ax.annotate(f"{SHORT_EFFECT.get(effect, effect)}: {row['estimate_log2']:+.2f} log2\nq = {row['p_value_fdr']:.2f}", (r, 1.0), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=8)
    ax.set_xticks(range(len(config.REGIONS)), config.REGIONS)
    handles = [plt.Line2D([], [], marker=plotting.SEX_MARKERS[s], linestyle="", color=plotting.BIO_GROUP_COLORS[g], markeredgecolor="black",
                          label=f"{g}, {s}") for g, s in cells]
    ax.legend(handles=handles, fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    ax.set(ylabel="normalised log2 abundance", title=f"{view.labels.get(feature, feature)}  [{view.classes.get(feature, '')}]")
    ax.margins(y=0.25)
    return finish(fig, view, "Bars: cell means. Missing values are not shown (never imputed).")


def effect_forest(view: View, region: str = "Cortex", effect: str = MIA_EFFECT, n_features: int | None = None) -> plt.Figure:
    """Estimates and 95 % CIs of the features with the smallest p-values."""
    effects = view.differential()
    block = effects.loc[effects["Region"].eq(resolve_scope(region)) & effects["Effect"].eq(effect)]
    block = block.nsmallest(n_features or view.options.top_n, "p_value").iloc[::-1]
    hit = block["p_value_fdr"].le(view.options.fdr_alpha)
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(block) + 1.6))
    y = np.arange(len(block))
    ax.hlines(y, block["ci_lower_log2"], block["ci_upper_log2"], color=np.where(hit, plotting.COLORBLIND_COLORS[1], "#777777"))
    ax.scatter(block["estimate_log2"], y, color=np.where(hit, plotting.COLORBLIND_COLORS[1], "#333333"), zorder=3)
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_yticks(y, [f"{lab}  (q={q:.2f})" for lab, q in zip(block["Label"], block["p_value_fdr"])], fontsize=7)
    ax.set(xlabel="log2 effect, 95 % CI", title=f"{resolve_scope(region)}: {effect}, top {len(block)} by p")
    return finish(fig, view, "Orange: FDR hit at the chosen alpha.")


def class_effects(view: View, effect: str = MIA_EFFECT, statistic: str = "mean") -> plt.Figure:
    """Effect on lipid classes (mean of member species) per region; * marks FDR hits."""
    if "Lipid Class" not in view.features:
        raise ValueError("class_effects needs lipid classes; use omics='lipidomics'.")
    profiles, annotations = aggregate_class_profiles(view.values, view.features[["Lipid Name", "Lipid Class"]], statistic=statistic)
    fitted = fit_regionwise_differential(profiles, view.metadata, annotations=annotations, min_per_cell=view.options.min_per_cell,
                                         fdr_denominator="all_features").effects
    table = fitted.loc[fitted["Effect"].eq(effect)]
    estimate = table.pivot(index="Feature", columns="Region", values="estimate_log2").reindex(columns=config.REGIONS)
    q = table.pivot(index="Feature", columns="Region", values="p_value_fdr").reindex(columns=config.REGIONS)
    limit = float(np.nanmax(np.abs(estimate.to_numpy()))) or 1.0
    fig, ax = plt.subplots(figsize=(5.5, 0.32 * len(estimate) + 1.8))
    image = ax.imshow(estimate.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for (i, j), value in np.ndenumerate(q.to_numpy()):
        if value <= view.options.fdr_alpha:
            ax.text(j, i, "*", ha="center", va="center", fontsize=12)
    ax.set_yticks(range(len(estimate)), [str(n).split("::")[-1] for n in estimate.index], fontsize=8)
    ax.set_xticks(range(len(config.REGIONS)), config.REGIONS)
    ax.grid(False)
    ax.set_title(f"Lipid classes ({statistic} of species): {effect}", fontsize=10)
    fig.colorbar(image, ax=ax, label="log2 effect", fraction=0.06)
    return finish(fig, view)
