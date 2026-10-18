"""Stage 9 - reproduce the core facility's report figures (heatmap + PLS-DA per region).

The facility's report showed a heatmap of the top-25 lipids and PLS-DA score
plots with clean-looking MIA/control separation.  Reproducing them with the
same inputs, and adding the permutation test the report did not show, makes
the point that a supervised score plot separates groups even on noise.

Output: ``results/lipidomics/09_vendor_figures/``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.stats import f

from .. import config, plotting
from ..config import Settings
from ..multivariate.plsda import fit_plsda, permutation_test_plsda
from ..multivariate.vendor import select_top_n_by_nominal_p, vendor_column_order
from ..preprocessing import for_multivariate
from ..stats.differential import fit_regionwise_differential
from .common import feature_label, prepare, region_samples, write_csv

VENDOR_COLORS = {config.CONTROL: "#91c96b", config.MIA: "#176b32"}
EXCLUDED_SAMPLES: tuple[str, ...] = ()   # samples the facility left out of its figures (none in the toy data)


def f_data_ellipse(scores: pd.DataFrame, level: float = 0.95, n_points: int = 100) -> np.ndarray:
    """Hotelling/F *data* ellipse (where 95 % of points fall), not a CI for the mean."""
    covariance = np.cov(scores.iloc[:, :2].to_numpy(dtype=float), rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    n = len(scores)
    radius = np.sqrt(2 * (n - 1) / (n - 2) * f.ppf(level, 2, n - 2))
    angles = np.linspace(0, 2 * np.pi, n_points)
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    return circle @ (vectors * (np.sqrt(values) * radius)).T + scores.iloc[:, :2].mean(axis=0).to_numpy()


def run(settings: Settings) -> pd.DataFrame:
    """Heatmap and PLS-DA per region in the facility's style, plus the missing permutation test."""
    plotting.apply_plot_style()
    out = settings.out("lipidomics", "09_vendor_figures")
    prepared = prepare(settings, "lipidomics", max_missing_fraction=None)   # the facility did not filter by prevalence
    meta = prepared.metadata.loc[~prepared.metadata["Sample"].isin(EXCLUDED_SAMPLES)]
    values = prepared.values.loc[:, meta["Sample"]]
    # The facility picked its heatmap lipids by nominal p of the MIA effect; reproduce that choice.
    effects = fit_regionwise_differential(values, meta, annotations=prepared.features).effects
    scaled = for_multivariate(values)
    labels = feature_label(prepared.features)
    rows = []
    for region in config.REGIONS:
        region_meta = meta.loc[meta[config.REGION_COL].eq(region)]
        samples = region_samples(meta, region)
        complete = scaled.loc[:, samples]
        top = select_top_n_by_nominal_p(effects, region=region, eligible_features=complete.index)
        plotting.save_figure(_heatmap(complete.loc[top], region_meta, labels, region), out / f"heatmap_{region.lower()}")
        group = region_meta.set_index("Sample")[config.GROUP_COL]
        sex = region_meta.set_index("Sample")[config.SEX_COL]
        scores, variance = fit_plsda(complete, group)
        test, _ = permutation_test_plsda(complete, group, sex, n_permutations=settings.n_permutations, random_state=settings.seed)
        plotting.save_figure(_plsda(scores, region_meta, variance, float(test["permutation_p_value"])), out / f"plsda_{region.lower()}")
        rows.append({"Region": region, "n_samples": len(samples), "n_features": len(complete), **test.to_dict()})
    table = pd.DataFrame(rows)
    write_csv(table, out / "plsda_permutation_tests.csv")
    return table


def _heatmap(values: pd.DataFrame, meta: pd.DataFrame, labels: pd.Series, region: str) -> plt.Figure:
    """Facility-style heatmap: Ward-clustered rows, samples ordered by group, sex and number."""
    row_linkage = linkage(values.to_numpy(dtype=float), method="ward")
    ordered = values.iloc[leaves_list(row_linkage)].loc[:, vendor_column_order(meta)]
    ordered_meta = meta.set_index("Sample").loc[ordered.columns]
    fig = plt.figure(figsize=(12, 9))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.2, 8), height_ratios=(0.3, 8))
    strip, tree, matrix = fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    strip.imshow(np.array([[plt.matplotlib.colors.to_rgba(VENDOR_COLORS[g]) for g in ordered_meta[config.GROUP_COL]]]), aspect="auto")
    strip.set_axis_off()
    dendrogram(row_linkage, ax=tree, orientation="left", no_labels=True, color_threshold=0, above_threshold_color="#555555")
    tree.set_axis_off()
    image = matrix.imshow(ordered.to_numpy(), cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    matrix.set_yticks(range(len(ordered)), labels.reindex(ordered.index).fillna("").tolist(), fontsize=6)
    matrix.yaxis.tick_right()
    matrix.set_xticks(range(ordered.shape[1]), ordered_meta["Sample number"].astype(str), fontsize=7)
    matrix.set(xlabel="Sample number", title=f"{region}: top 25 lipids by nominal p (facility style)")
    matrix.grid(False)
    fig.colorbar(image, ax=matrix, fraction=0.03, pad=0.12, label="row z-score")
    fig.legend(handles=[Patch(facecolor=c, label=g) for g, c in VENDOR_COLORS.items()], loc="upper right", frameon=False)
    return fig


def _plsda(scores: pd.DataFrame, meta: pd.DataFrame, variance: np.ndarray, p_value: float) -> plt.Figure:
    """Facility-style PLS-DA score plot with 95 % data ellipses and the permutation p in the title."""
    joined = scores.join(meta.set_index("Sample")[[config.GROUP_COL, "Sample number"]])
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    for group, color in VENDOR_COLORS.items():
        subset = joined.loc[joined[config.GROUP_COL].eq(group)]
        ellipse = f_data_ellipse(subset[["PLS 1", "PLS 2"]])
        ax.fill(ellipse[:, 0], ellipse[:, 1], color=color, alpha=0.2)
        ax.scatter(subset["PLS 1"], subset["PLS 2"], facecolors="none", edgecolors=color, s=60, linewidth=1.2, label=group)
    components = np.diff(np.r_[0, variance])
    ax.set(xlabel=f"Component 1 ({components[0] * 100:.0f} %)", ylabel=f"Component 2 ({components[1] * 100:.0f} %)",
           title=f"PLS-DA scores - permutation p = {p_value:.3f}")
    ax.legend()
    fig.text(0.5, 0.0, "Separation in a supervised score plot is expected even without any real group difference.",
             ha="center", fontsize=8)
    return fig
