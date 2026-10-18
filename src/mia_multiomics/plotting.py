"""Shared figure style, colours and the handful of plot types used throughout.

Every stage draws its figures with these functions, so all figures look the
same and a colour or style is changed in one place.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are written to files; no display needed

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap, to_rgba  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import chi2  # noqa: E402

from . import config  # noqa: E402

SEX_COLORS = {"Male": "#0072B2", "Female": "#CC79A7"}
BIO_GROUP_COLORS = {config.CONTROL: "#009E73", config.MIA: "#D55E00"}
BRAIN_REGION_COLORS = {"Cortex": "#56B4E9", "Cerebellum": "#E69F00", "Hippocampus": "#0072B2"}
SEX_MARKERS = {"Male": "o", "Female": "s"}
MARKER_CYCLE = ("o", "s", "^", "D", "P", "X", "v", "<", ">")
COLORBLIND_COLORS = ("#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#999999", "#000000")
PALETTES = {config.REGION_COL: BRAIN_REGION_COLORS, config.GROUP_COL: BIO_GROUP_COLORS, config.SEX_COL: SEX_COLORS}

STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#D9D9D9",
    "grid.alpha": 0.7,
    "legend.fontsize": 9,
    "figure.dpi": 100,
    "savefig.bbox": "tight",
}
FIGURE_DPI = 150


def apply_plot_style() -> None:
    """Apply the project-wide Matplotlib defaults."""
    plt.rcParams.update(STYLE)


def save_figure(fig: plt.Figure, path: Path) -> Path:
    """Save ``fig`` as ``<path>.png`` (creating folders) and close it."""
    path = Path(path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def palette_for(values: Iterable, column: str) -> dict:
    """Fixed colours for region / group / sex, a colour-blind cycle otherwise."""
    if column in PALETTES:
        return PALETTES[column]
    categories = sorted(pd.unique(pd.Series(list(values)).astype(str)))
    return {c: COLORBLIND_COLORS[i % len(COLORBLIND_COLORS)] for i, c in enumerate(categories)}


# ---------------------------------------------------------------- PCA-type plots
def confidence_ellipse(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    level: float = 0.95,
    n_points: int = 100,
) -> np.ndarray:
    """Return normal-theory data-ellipse coordinates for two score vectors."""
    coordinates = np.column_stack((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
    if coordinates.shape[0] < 3:
        raise ValueError("A confidence ellipse requires at least three observations.")
    if not 0 < level < 1:
        raise ValueError("level must be between zero and one.")
    if not np.isfinite(coordinates).all():
        raise ValueError("Ellipse coordinates must be finite.")
    covariance = np.cov(coordinates, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.any(eigenvalues <= 0):
        raise ValueError("Ellipse coordinates must have non-singular covariance.")
    angles = np.linspace(0, 2 * np.pi, n_points)
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    scale = np.sqrt(chi2.ppf(level, df=2))
    return circle @ (eigenvectors * (np.sqrt(eigenvalues) * scale)).T + coordinates.mean(axis=0)


def scores_plot(scores: pd.DataFrame, metadata: pd.DataFrame, color_by: str, *, title: str,
                variance: Iterable[float] | None = None, label: str | None = None,
                ellipses: bool = True, x: str = "PC 1", y: str = "PC 2",
                outline: list[np.ndarray] | None = None) -> plt.Figure:
    """Score scatter plot (any two components) coloured by one metadata column.

    Numeric columns get a colour bar; categorical ones a legend and, if
    ``ellipses``, a 95 % data ellipse per category.  ``label`` writes that
    metadata column next to each point (e.g. sample number).  ``outline`` adds
    extra closed curves such as Hotelling T2 limits.
    """
    meta = metadata.set_index("Sample").loc[scores.index]
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    values = meta[color_by]
    if pd.api.types.is_numeric_dtype(values) and values.nunique() > 8:
        points = ax.scatter(scores[x], scores[y], c=values, cmap="viridis", edgecolor="black", linewidth=0.3)
        fig.colorbar(points, ax=ax, label=color_by)
    else:
        values = values.astype(str)
        colors = palette_for(values, color_by)
        markers = meta[config.SEX_COL].map(SEX_MARKERS) if color_by != config.SEX_COL and config.SEX_COL in meta else None
        for category in sorted(values.unique(), key=str):
            mask = values.eq(category)
            subset = scores.loc[mask]
            for marker in (markers[mask].unique() if markers is not None else ["o"]):
                keep = (markers[mask] == marker).to_numpy() if markers is not None else slice(None)
                ax.scatter(subset[x][keep], subset[y][keep], color=colors.get(category, "#777777"), marker=marker,
                           edgecolor="black", linewidth=0.3, s=45)
            if ellipses and mask.sum() >= 3:
                e = confidence_ellipse(subset[x], subset[y])
                ax.plot(e[:, 0], e[:, 1], color=colors.get(category, "#777777"), linewidth=0.8, linestyle="--")
        handles = [Patch(color=colors.get(c, "#777777"), label=c) for c in sorted(values.unique(), key=str)]
        if markers is not None:
            handles += [plt.Line2D([], [], marker=m, linestyle="", color="#555555", label=s) for s, m in SEX_MARKERS.items()]
        ax.legend(handles=handles, title=color_by, fontsize=8, loc="best")
    for curve in outline or []:
        ax.plot(curve[:, 0], curve[:, 1], color="#555555", linewidth=1.0)
    if label is not None:
        for sample, row in meta.iterrows():
            ax.annotate(str(row[label]), (scores.loc[sample, x], scores.loc[sample, y]), fontsize=7,
                        xytext=(3, 2), textcoords="offset points")
    variance = list(variance) if variance is not None else None
    xi, yi = int(x.split()[-1]) - 1, int(y.split()[-1]) - 1
    ax.set(xlabel=f"{x} ({variance[xi] * 100:.1f}%)" if variance else x,
           ylabel=f"{y} ({variance[yi] * 100:.1f}%)" if variance else y, title=title)
    ax.axhline(0, color="#BBBBBB", linewidth=0.8)
    ax.axvline(0, color="#BBBBBB", linewidth=0.8)
    return fig


def scree_plot(variance_ratio: Iterable[float], title: str) -> plt.Figure:
    """Bars of variance explained per component plus the cumulative line."""
    ratio = np.asarray(list(variance_ratio)) * 100
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    positions = np.arange(1, len(ratio) + 1)
    ax.bar(positions, ratio, color=COLORBLIND_COLORS[0], label="per component")
    ax.plot(positions, np.cumsum(ratio), "o-", color=COLORBLIND_COLORS[1], label="cumulative")
    ax.set(xlabel="Component", ylabel="Variance explained (%)", title=title)
    ax.legend(frameon=False)
    return fig


def lollipop_plot(values: pd.Series, labels: pd.Series | None = None, *, title: str, xlabel: str) -> plt.Figure:
    """Horizontal lollipop chart, e.g. the strongest loadings of one component."""
    values = values.sort_values()
    names = labels.reindex(values.index).fillna(pd.Series(values.index, index=values.index)) if labels is not None else values.index.to_series()
    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.25 * len(values))))
    ax.hlines(names.astype(str), 0, values, color="#777777", linewidth=0.8)
    ax.scatter(values, names.astype(str), color=COLORBLIND_COLORS[0], zorder=3)
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set(xlabel=xlabel, title=title)
    return fig


def index_plot(values: pd.Series, labels: pd.Series, limits: Mapping[str, float], *, title: str, ylabel: str) -> plt.Figure:
    """Statistic per sample (x = sample number) with horizontal limit lines."""
    order = labels.sort_values().index
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.scatter(labels.loc[order], values.loc[order], color=COLORBLIND_COLORS[0])
    for sample in order:
        if any(values.loc[sample] > limit for limit in limits.values()):
            ax.annotate(str(labels.loc[sample]), (labels.loc[sample], values.loc[sample]), fontsize=8)
    for i, (name, limit) in enumerate(limits.items()):
        ax.axhline(limit, color=COLORBLIND_COLORS[1], linestyle=["--", ":"][i % 2], label=name)
    ax.set(xlabel="Sample number", ylabel=ylabel, title=title)
    ax.legend(frameon=False)
    return fig


def presence_heatmap(present: pd.DataFrame, metadata: pd.DataFrame, annotate: list[str], *, title: str) -> plt.Figure:
    """Observed (dark) / missing (light) heatmap with colour strips for sample groups."""
    meta = metadata.set_index("Sample").loc[present.columns]
    heights = [0.03] * len(annotate) + [1]
    fig, axes = plt.subplots(len(annotate) + 1, 1, figsize=(11, 7), gridspec_kw={"height_ratios": heights, "hspace": 0.05})
    handles = []
    for ax, column in zip(axes[:-1], annotate):
        colors = palette_for(meta[column], column)
        ax.imshow(np.array([[to_rgba(colors[str(v)] if str(v) in colors else colors.get(v, "#777777")) for v in meta[column]]]), aspect="auto")
        ax.set(xticks=[], yticks=[])
        ax.set_ylabel(column, rotation=0, ha="right", va="center", fontsize=8)
        handles += [Patch(color=c, label=str(k)) for k, c in colors.items() if k in set(meta[column].astype(str)) | set(meta[column])]
    axes[-1].imshow(present.astype(int).to_numpy(), cmap=ListedColormap(["#f0f0f0", COLORBLIND_COLORS[0]]),
                    aspect="auto", interpolation="none")
    axes[-1].set(xticks=[], yticks=[], xlabel=f"Samples (n={present.shape[1]})", ylabel=f"Features (n={present.shape[0]})")
    axes[-1].legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, fontsize=8)
    axes[0].set_title(title)
    return fig
