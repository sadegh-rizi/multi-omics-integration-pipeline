"""Structure beyond single features: co-abundance modules, litters, regional
profiles, ASCA+, the facility's figure style, and the lipid-protein myelin check.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage

from .. import config, plotting
from ..multivariate.asca import EFFECT_ORDER, effect_svd, effect_variance_table, fit_asca
from ..multivariate.feature_modules import (
    assign_hierarchical_modules,
    bootstrap_feature_loadings,
    bootstrap_module_comembership,
    fit_sample_space_loadings,
    loading_stability_summary,
    module_stability_summary,
    permuted_null_module_sizes,
    spearman_correlation,
    split_half_loading_stability,
)
from ..multivariate.plsda import fit_plsda, permutation_test_plsda
from ..multivariate.vendor import vendor_column_order
from ..preprocessing import complete_features
from ..stages.effect_sizes import annotate_lipids, composite_members, composite_scores
from ..stages.feature_modules import MODULE_DISTANCE_CUT, MODULE_MIN_SIZE, MODULE_STABILITY_FLOOR
from ..stages.vendor_figures import VENDOR_COLORS, f_data_ellipse
from ._common import MIA_EFFECT, class_colors, finish
from .view import View, resolve_scope

REGION_PAIRS = (("Cortex", "Cerebellum"), ("Hippocampus", "Cerebellum"), ("Cortex", "Hippocampus"))


# ---------------------------------------------------------------- feature space (modules)
def _region_complete(view: View, region: str) -> pd.DataFrame:
    return complete_features(view.values.loc[:, view.samples(region)])


def _modules(view: View, region: str) -> dict:
    """Correlation, modules, bootstrap co-membership and the permuted null for one region (cached)."""
    key = ("modules", resolve_scope(region))
    if key not in view._cache:
        complete = _region_complete(view, region)
        seed = view.options.seed
        correlation = spearman_correlation(complete)
        modules = assign_hierarchical_modules(correlation, distance_cut=MODULE_DISTANCE_CUT, signed=True)
        n_boot = max(10, view.options.n_bootstraps // 4)
        co_membership = bootstrap_module_comembership(complete, n_bootstraps=n_boot, random_state=seed,
                                                      distance_cut=MODULE_DISTANCE_CUT, signed=True)
        summary = module_stability_summary(modules, co_membership, stability_floor=MODULE_STABILITY_FLOOR, minimum_size=MODULE_MIN_SIZE)
        null = permuted_null_module_sizes(complete, n_permutations=20, random_state=seed + 1, distance_cut=MODULE_DISTANCE_CUT,
                                          signed=True, minimum_size=MODULE_MIN_SIZE)
        view._cache[key] = {"complete": complete, "correlation": correlation, "modules": modules, "summary": summary,
                            "null": null, "n_bootstraps": n_boot}
    return view._cache[key]


def feature_correlation_heatmap(view: View, region: str = "Cortex", max_features: int = 150) -> plt.Figure:
    """Signed Spearman correlation between features, ordered by co-abundance module."""
    result = _modules(view, region)
    modules, correlation = result["modules"], result["correlation"]
    order = modules.sort_values(kind="stable").index
    if len(order) > max_features:
        largest = modules.value_counts().index[:10]
        order = modules.loc[modules.isin(largest)].sort_values(kind="stable").index[:max_features]
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    image = ax.imshow(correlation.loc[order, order], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set(xticks=[], yticks=[], title=f"{resolve_scope(region)}: feature correlation by module")
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="Spearman rho")
    n_reportable = int(result["summary"]["reportable"].sum())
    return finish(fig, view, f"{modules.nunique()} modules, {n_reportable} stable in >= {MODULE_STABILITY_FLOOR:.0%} of bootstraps.")


def feature_loadings_view(view: View, region: str = "Cortex") -> plt.Figure:
    """The sample PCA seen from the features: PC1 vs PC2 loading of every complete feature."""
    loadings, scree = fit_sample_space_loadings(_region_complete(view, region), scale_features=True)
    lumped, colors = class_colors(view.classes.reindex(loadings.index).fillna("unannotated"))
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, color in colors.items():
        members = lumped.index[lumped.eq(name)]
        ax.scatter(loadings.loc[members, "PC 1"], loadings.loc[members, "PC 2"], s=20, color=color, label=name,
                   edgecolor="black", linewidth=0.2)
    ax.axhline(0, color="#BBBBBB", linewidth=0.7)
    ax.axvline(0, color="#BBBBBB", linewidth=0.7)
    variance = scree["variance_explained"].to_numpy()
    ax.set(xlabel=f"PC 1 feature loading ({variance[0] * 100:.1f}%)", ylabel=f"PC 2 feature loading ({variance[1] * 100:.1f}%)",
           title=f"{resolve_scope(region)}: feature view of the PCA")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    return finish(fig, view)


def feature_loading_stability(view: View, region: str = "Cortex") -> plt.Figure:
    """Bootstrap median and 95 % interval of each feature's PC1/PC2 loading."""
    complete = _region_complete(view, region)
    boot = bootstrap_feature_loadings(complete, n_bootstraps=view.options.n_bootstraps, random_state=view.options.seed, n_components=2)
    stability = loading_stability_summary(boot, complete.index)
    split = split_half_loading_stability(complete, random_state=view.options.seed + 1, n_components=2)
    order = stability.groupby("Feature", sort=False)["top_decile_fraction"].max().sort_values(ascending=False, kind="stable").index
    positions = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, component in enumerate(("PC 1", "PC 2")):
        rows = stability.loc[stability["Component"].eq(component)].set_index("Feature").loc[order]
        ax.vlines(positions, rows["loading_q025"], rows["loading_q975"], color=plotting.COLORBLIND_COLORS[i], linewidth=0.5, alpha=0.45)
        ax.scatter(positions, rows["loading_median"], color=plotting.COLORBLIND_COLORS[i], s=6, label=component)
    split_text = ", ".join(f"{c}: r = {r:.2f}" for c, r in zip(split["Component"], split["split_half_loading_correlation"]))
    ax.set(xlabel="features, ordered by how often they are in the top 10 % of loadings",
           ylabel="sign-aligned loading", title=f"{resolve_scope(region)}: loading stability (split-half {split_text})")
    ax.legend(frameon=False)
    return finish(fig, view, f"{view.options.n_bootstraps} sample bootstraps. Loadings are interpretable only if split-half r >= 0.5.")


def module_stability(view: View, region: str = "Cortex") -> plt.Figure:
    """Module size against bootstrap co-membership, and the largest module against the permuted null."""
    result = _modules(view, region)
    summary = result["summary"]
    largest_null = result["null"].groupby("null_permutation")["n_features"].max()
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    multi = summary.loc[summary["n_features"].ge(2)]
    left.scatter(multi["n_features"], multi["mean_co_membership"], color=np.where(multi["reportable"], plotting.COLORBLIND_COLORS[1], "#777777"))
    left.axhline(MODULE_STABILITY_FLOOR, color="#444444", linestyle="--", label=f"stability floor {MODULE_STABILITY_FLOOR:.0%}")
    left.set(xlabel="module size (features)", ylabel="mean bootstrap co-membership", ylim=(0, 1.02),
             title=f"{resolve_scope(region)}: module stability")
    left.legend(frameon=False)
    right.boxplot(largest_null, positions=[0], widths=0.4)
    right.scatter([1], [summary["n_features"].max()], color=plotting.COLORBLIND_COLORS[0], s=50, zorder=3)
    right.set_xticks([0, 1], ["permuted\nnull", "observed"])
    right.set(ylabel="largest module (features)", title="larger than chance?")
    fig.tight_layout()
    return finish(fig, view, f"{result['n_bootstraps']} bootstraps; null: each feature shuffled across samples (20 times).")


# ---------------------------------------------------------------- litters
def pca_by_cage(view: View, region: str = "Cortex") -> plt.Figure:
    """Regional PC1/PC2 scores coloured by cage (litter)."""
    model, scores, _ = view.pca(region)
    fig = plotting.scores_plot(scores, view.metadata, config.CAGE_COL, variance=model.explained_variance_ratio_,
                               title=f"{resolve_scope(region)}: PCA by cage (litter)", ellipses=False)
    return finish(fig, view)


def cage_panels(view: View, region: str = "Cortex") -> plt.Figure:
    """Control and MIA side by side, coloured by cage, marker = sex: do litters cluster?"""
    model, scores, _ = view.pca(region)
    joined = scores[["PC 1", "PC 2"]].join(view.metadata.set_index("Sample")[[config.CAGE_COL, config.GROUP_COL, config.SEX_COL]])
    colors = plotting.palette_for(joined[config.CAGE_COL], config.CAGE_COL)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for ax, group in zip(axes, config.GROUPS):
        rows = joined.loc[joined[config.GROUP_COL].eq(group)]
        for (cage, sex), cell in rows.groupby([config.CAGE_COL, config.SEX_COL]):
            ax.scatter(cell["PC 1"], cell["PC 2"], color=colors[cage], marker=plotting.SEX_MARKERS[sex], s=55, edgecolor="black", linewidth=0.3)
        for cage, cell in rows.groupby(config.CAGE_COL):
            ax.annotate(cage, cell[["PC 1", "PC 2"]].mean(), fontsize=7, color=colors[cage])
        ax.set(title=group, xlabel=f"PC 1 ({model.explained_variance_ratio_[0] * 100:.1f}%)")
        ax.axhline(0, color="#BBBBBB", linewidth=0.8)
        ax.axvline(0, color="#BBBBBB", linewidth=0.8)
    axes[0].set_ylabel(f"PC 2 ({model.explained_variance_ratio_[1] * 100:.1f}%)")
    fig.suptitle(f"{resolve_scope(region)}: treatment, sex (marker) and cage (colour)")
    return finish(fig, view, "Cage labels sit at the cage mean.")


def within_cage_distance(view: View, region: str = "Cortex") -> plt.Figure:
    """Distance of each mouse to its cage centroid in PC1/PC2."""
    _, scores, _ = view.pca(region)
    joined = scores[["PC 1", "PC 2"]].join(view.metadata.set_index("Sample")[[config.CAGE_COL, config.GROUP_COL]])
    centroids = joined.groupby(config.CAGE_COL)[["PC 1", "PC 2"]].transform("mean")
    joined["distance"] = np.hypot(joined["PC 1"] - centroids["PC 1"], joined["PC 2"] - centroids["PC 2"])
    cages = sorted(joined[config.CAGE_COL].unique())
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, cage in enumerate(cages):
        rows = joined.loc[joined[config.CAGE_COL].eq(cage)]
        ax.scatter(np.full(len(rows), i), rows["distance"], color=rows[config.GROUP_COL].map(plotting.BIO_GROUP_COLORS), edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(len(cages)), cages, rotation=30)
    ax.set(ylabel="distance to cage centroid (PC1-PC2)", title=f"{resolve_scope(region)}: within-litter spread")
    ax.legend(handles=[Patch(color=c, label=g) for g, c in plotting.BIO_GROUP_COLORS.items()], frameon=False)
    return finish(fig, view, "Cages with one mouse have distance 0.")


# ---------------------------------------------------------------- regional profiles
def region_differences(view: View) -> plt.Figure:
    """For each region pair: mean within-mouse difference in controls (x) against MIA (y), per feature."""
    meta = view.metadata.set_index("Sample")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (first, second) in zip(axes, REGION_PAIRS):
        a = view.values.loc[:, view.samples(first)].T.join(meta[[config.MOUSE_COL]]).set_index(config.MOUSE_COL)
        b = view.values.loc[:, view.samples(second)].T.join(meta[[config.MOUSE_COL]]).set_index(config.MOUSE_COL)
        mice = a.index.intersection(b.index)
        difference = a.loc[mice] - b.loc[mice]
        group = meta.drop_duplicates(config.MOUSE_COL).set_index(config.MOUSE_COL).loc[mice, config.GROUP_COL]
        control = difference.loc[group.eq(config.CONTROL)].mean()
        mia = difference.loc[group.eq(config.MIA)].mean()
        ok = control.notna() & mia.notna()
        ax.scatter(control[ok], mia[ok], s=8, alpha=0.6, color=plotting.COLORBLIND_COLORS[0])
        limit = float(np.nanmax(np.abs(np.r_[control[ok], mia[ok]]))) * 1.05
        ax.plot([-limit, limit], [-limit, limit], "--", color="#777777", linewidth=0.8)
        r = np.corrcoef(control[ok], mia[ok])[0, 1]
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="control: mean difference (log2)", ylabel="MIA: mean difference (log2)",
               title=f"{first} - {second}: r = {r:.2f}  ({len(mice)} mice)")
        ax.set_aspect("equal")
    fig.tight_layout()
    return finish(fig, view, "Each point is a feature. On the diagonal = the regional difference is the same in both groups. "
                             "Mice missing either region are left out of that pair.")


# ---------------------------------------------------------------- ASCA+
def asca_overview(view: View, effects: tuple[str, ...] = ("Region", "Group", "Sex", "Region:Group")) -> plt.Figure:
    """ASCA+ variance shares of every model term, and the first score of selected effects."""
    values = complete_features(view.values)
    meta = view.metadata.set_index("Sample").loc[list(values.columns)].reset_index()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_asca(values, meta, scaling=view.options.scaling)
    variance = effect_variance_table(fit).set_index("Effect")["variance_share"]
    fig = plt.figure(figsize=(14, 4.8))
    grid = fig.add_gridspec(1, 1 + len(effects), width_ratios=[1.6] + [1] * len(effects))
    bar = fig.add_subplot(grid[0])
    order = [*EFFECT_ORDER, "Residual"]
    bar.barh(order[::-1], variance.reindex(order[::-1]) * 100, color=plotting.COLORBLIND_COLORS[0])
    bar.set(xlabel="variance share (%)", title="ASCA+ variance partition")
    colour_by = {"Region": config.REGION_COL, "Group": config.GROUP_COL, "Sex": config.SEX_COL, "Region:Group": config.GROUP_COL}
    for i, effect in enumerate(effects):
        ax = fig.add_subplot(grid[i + 1])
        scores, _, _ = effect_svd(fit, effect)
        column = colour_by.get(effect, config.GROUP_COL)
        groups = meta[column].astype(str)
        colors = plotting.palette_for(groups, column)
        rng = np.random.default_rng(0)
        for level in sorted(groups.unique()):
            mask = groups.eq(level).to_numpy()
            ax.scatter(rng.uniform(-0.3, 0.3, mask.sum()) + sorted(groups.unique()).index(level), scores.loc[mask, "PC1"],
                       s=14, color=colors.get(level, "#777777"))
        ax.set_xticks(range(groups.nunique()), sorted(groups.unique()), rotation=30, fontsize=7)
        ax.set(title=f"{effect} ({variance[effect] * 100:.1f}%)", ylabel="effect score 1" if i == 0 else "")
    fig.tight_layout()
    return finish(fig, view, "Variance shares only; the walkthrough's permutation tests say which effects exceed chance.")


# ---------------------------------------------------------------- facility style
def vendor_heatmap(view: View, region: str = "Cortex", n_features: int = 25) -> plt.Figure:
    """The facility's heatmap: top features by nominal MIA p, row z-scores, samples ordered by group and sex."""
    region = resolve_scope(region)
    matrix = view.matrix(region)
    effects = view.differential()
    ranked = effects.loc[effects["Region"].eq(region) & effects["Effect"].eq(MIA_EFFECT) & effects["Feature"].isin(matrix.index)]
    top = ranked.nsmallest(n_features, "p_value")["Feature"]
    block = matrix.loc[top]
    meta = view.metadata.loc[view.metadata["Sample"].isin(block.columns)]
    row_linkage = linkage(block.to_numpy(dtype=float), method="ward")
    ordered = block.iloc[leaves_list(row_linkage)].loc[:, vendor_column_order(meta)]
    ordered_meta = meta.set_index("Sample").loc[ordered.columns]
    fig = plt.figure(figsize=(12, 0.3 * len(ordered) + 2))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.2, 8), height_ratios=(0.3, 8))
    strip, tree, heat = fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    strip.imshow(np.array([[plt.matplotlib.colors.to_rgba(VENDOR_COLORS[g]) for g in ordered_meta[config.GROUP_COL]]]), aspect="auto")
    strip.set_axis_off()
    dendrogram(row_linkage, ax=tree, orientation="left", no_labels=True, color_threshold=0, above_threshold_color="#555555")
    tree.set_axis_off()
    image = heat.imshow(ordered.to_numpy(), cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    heat.set_yticks(range(len(ordered)), view.labels.reindex(ordered.index).fillna("").tolist(), fontsize=7)
    heat.yaxis.tick_right()
    heat.set_xticks(range(ordered.shape[1]), ordered_meta["Sample number"].astype(str), fontsize=7)
    heat.set(xlabel="sample number", title=f"{region}: top {len(ordered)} by nominal MIA p (facility style)")
    heat.grid(False)
    fig.colorbar(image, ax=heat, fraction=0.03, pad=0.12, label="z-score")
    fig.legend(handles=[Patch(facecolor=c, label=g) for g, c in VENDOR_COLORS.items()], loc="upper right", frameon=False)
    return finish(fig, view, "Choosing features by their group p-value and then clustering them always 'separates' the groups.")


def plsda_scores(view: View, region: str = "Cortex") -> plt.Figure:
    """PLS-DA scores (facility style) next to the permutation null of the group separation."""
    region = resolve_scope(region)
    matrix = view.matrix(region)
    meta = view.metadata.set_index("Sample").loc[matrix.columns]
    scores, variance = fit_plsda(matrix, meta[config.GROUP_COL])
    test, null = permutation_test_plsda(matrix, meta[config.GROUP_COL], meta[config.SEX_COL],
                                        n_permutations=view.options.n_permutations, random_state=view.options.seed)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.3, 1]})
    joined = scores.join(meta[[config.GROUP_COL]])
    for group, color in VENDOR_COLORS.items():
        subset = joined.loc[joined[config.GROUP_COL].eq(group)]
        ellipse = f_data_ellipse(subset[["PLS 1", "PLS 2"]])
        left.fill(ellipse[:, 0], ellipse[:, 1], color=color, alpha=0.2)
        left.scatter(subset["PLS 1"], subset["PLS 2"], facecolors="none", edgecolors=color, s=60, linewidth=1.2, label=group)
    components = np.diff(np.r_[0, variance])
    left.set(xlabel=f"component 1 ({components[0] * 100:.0f} %)", ylabel=f"component 2 ({components[1] * 100:.0f} %)",
             title=f"{region}: PLS-DA scores")
    left.legend(frameon=False)
    observed = float(test["observed_centroid_separation"])
    right.hist(null, bins=30, color="#BBBBBB")
    right.axvline(observed, color=plotting.COLORBLIND_COLORS[1], linewidth=2, label="observed")
    right.set(xlabel="group-centroid separation", ylabel="permutations",
              title=f"permutation p = {float(test['permutation_p_value']):.3f}")
    right.legend(frameon=False)
    fig.tight_layout()
    return finish(fig, view, f"{view.options.n_permutations} label permutations within sex. A supervised score plot "
                             "separates groups even on noise; only the permutation test says whether it is more than chance.")


# ---------------------------------------------------------------- lipid vs protein
def myelin_lipid_vs_protein(view: View, region: str = "Hippocampus") -> plt.Figure:
    """Mouse by mouse: the lipid myelin index (C1) against a myelin protein score."""
    region = resolve_scope(region)
    lipid, protein = view.sibling("lipidomics"), view.sibling("proteomics")
    samples = [s for s in lipid.samples(region) if s in set(protein.samples(region))]
    scores = composite_scores(lipid.values.loc[:, samples], composite_members(annotate_lipids(lipid.features)))
    if "C1 myelin lipid index" not in scores.index:
        raise ValueError("The myelin lipid index cannot be computed with these settings (no complete member lipids).")
    lipid_index = scores.loc["C1 myelin lipid index"]
    values = protein.values.loc[:, samples]
    genes = protein.features["Gene"]
    members = genes.index[genes.isin(config.OLIGODENDROCYTE_GENES)]
    z = values.sub(values.mean(axis=1), axis=0).div(values.std(axis=1, ddof=1), axis=0)
    protein_score = z.loc[members].mean()
    meta = lipid.metadata.set_index("Sample").loc[samples]
    both = lipid.metadata.groupby(config.COHORT_COL)[config.GROUP_COL].nunique()
    matched = meta[config.COHORT_COL].isin(both.index[both.eq(2)])
    fig, ax = plt.subplots(figsize=(6, 5))
    for is_matched, marker in ((True, "o"), (False, "^")):
        rows = meta.index[matched.eq(is_matched)]
        ax.scatter(lipid_index[rows], protein_score[rows], marker=marker, s=55, edgecolor="black", linewidth=0.5,
                   color=meta.loc[rows, config.GROUP_COL].map(plotting.BIO_GROUP_COLORS))
    for cage, rows in meta.groupby(config.CAGE_COL):
        ax.annotate(cage, (lipid_index[rows.index].mean(), protein_score[rows.index].mean()), fontsize=6, alpha=0.7)
    r = lipid_index.corr(protein_score)
    handles = [Patch(color=c, label=g) for g, c in plotting.BIO_GROUP_COLORS.items()]
    handles += [plt.Line2D([], [], marker="o", linestyle="", color="white", markeredgecolor="black", label="date with both groups"),
                plt.Line2D([], [], marker="^", linestyle="", color="white", markeredgecolor="black", label="other dates")]
    ax.legend(handles=handles, fontsize=7, frameon=False)
    ax.set(xlabel="lipid myelin index C1 (log2 ratio)", ylabel=f"myelin protein score (mean z, {len(members)} proteins)",
           title=f"{region}: lipid vs protein myelin, r = {r:.2f}")
    return finish(fig, view, "Each point is a mouse. Agreement between two assays of the same tissue supports the signal being biological.")
