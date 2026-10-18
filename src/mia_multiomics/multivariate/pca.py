"""PCA for complete feature-by-sample matrices: fit, scree, loadings,
Hotelling T2 ellipses, and how much of each PC is explained by design factors."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import beta
from sklearn.decomposition import PCA

from ..plotting import BRAIN_REGION_COLORS, COLORBLIND_COLORS, MARKER_CYCLE, save_figure
from ..stats.common import r_squared


def complete_numeric(values: pd.DataFrame) -> pd.DataFrame:
    numeric = values.apply(pd.to_numeric, errors="coerce")
    if numeric.empty or numeric.isna().to_numpy().any():
        raise ValueError("PCA requires a non-empty complete feature-by-sample matrix.")
    return numeric


def fit_pca(
    values: pd.DataFrame,
    *,
    n_components: int | None = None,
) -> tuple[PCA, pd.DataFrame, pd.DataFrame]:
    """Fit PCA to a complete feature-by-sample matrix.

    Returns the fitted model, sample scores, and feature loadings. Features
    are expected to have been auto-scaled before this call.
    """
    numeric = complete_numeric(values)
    maximum = min(numeric.shape)
    if n_components is None:
        n_components = maximum
    if not 1 <= n_components <= maximum:
        raise ValueError(f"n_components must be between 1 and {maximum}.")

    model = PCA(n_components=n_components)
    scores = model.fit_transform(numeric.T.to_numpy(dtype=float))
    components = [f"PC {number}" for number in range(1, n_components + 1)]
    return (
        model,
        pd.DataFrame(scores, index=numeric.columns, columns=components),
        pd.DataFrame(model.components_.T, index=numeric.index, columns=components),
    )


def scree_table(model: PCA) -> pd.DataFrame:
    """Return explained and cumulative variance for a fitted PCA model."""
    ratios = model.explained_variance_ratio_
    return pd.DataFrame({
        "Component": [f"PC {number}" for number in range(1, len(ratios) + 1)],
        "variance_explained": ratios,
        "cumulative_variance_explained": np.cumsum(ratios),
    })


def loadings_table(
    loadings: pd.DataFrame,
    annotations: pd.DataFrame | None = None,
    *,
    n_features: int = 20,
) -> pd.DataFrame:
    """Rank the strongest positive and negative feature loadings per PC."""
    if n_features < 1:
        raise ValueError("n_features must be at least one.")
    rows: list[pd.DataFrame] = []
    for component in loadings:
        ranking = pd.DataFrame({"Feature": loadings.index, "loading": loadings[component]})
        ranking["absolute_loading"] = ranking["loading"].abs()
        ranking = ranking.nlargest(n_features, "absolute_loading", keep="all")
        ranking.insert(0, "Component", component)
        rows.append(ranking)
    result = pd.concat(rows, ignore_index=True)
    if annotations is not None:
        result = result.merge(
            annotations,
            left_on="Feature",
            right_index=True,
            how="left",
            validate="many_to_one",
        )
    return result.sort_values(["Component", "absolute_loading"], ascending=[True, False], kind="stable")


def _two_dimensional_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Validate the PC1/PC2 score space used for a plotted T-squared boundary."""
    numeric = scores.apply(pd.to_numeric, errors="coerce")
    if numeric.shape[1] != 2:
        raise ValueError("Hotelling T-squared ellipse requires exactly two score columns.")
    if len(numeric) <= 2:
        raise ValueError("Hotelling T-squared ellipse requires more than two samples.")
    if numeric.isna().to_numpy().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Hotelling T-squared scores must be finite.")
    score_sd = numeric.std(axis=0, ddof=1)
    if score_sd.le(0).any():
        raise ValueError("Hotelling T-squared scores must vary on both displayed components.")
    return numeric


def hotelling_t2_2d(scores: pd.DataFrame) -> pd.Series:
    """Return per-sample T-squared in the displayed PC1/PC2 score space."""
    numeric = _two_dimensional_scores(scores)
    centered = numeric.subtract(numeric.mean(axis=0), axis=1)
    standardized = centered.divide(numeric.std(axis=0, ddof=1), axis=1)
    return standardized.pow(2).sum(axis=1).rename("hotelling_t2")


def hotelling_ellipse(
    scores: pd.DataFrame,
    *,
    alpha: float = 0.95,
    n_points: int = 100,
) -> tuple[np.ndarray, float]:
    """Return an axis-aligned 2-D Hotelling T-squared boundary and limit.

    The displayed plane fixes p=2; using the outlier screen's higher-dimensional
    PC count here would draw a boundary that does not match this plot.
    """
    numeric = _two_dimensional_scores(scores)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one.")
    if n_points < 3:
        raise ValueError("n_points must be at least three.")
    n_samples = len(numeric)
    p = numeric.shape[1]
    # The Beta quantile calibrates in-sample observations, unlike the F form
    # used for a future observation from the fitted score distribution.
    limit = (n_samples - 1) ** 2 / n_samples * beta.ppf(
        alpha,
        p / 2,
        (n_samples - p - 1) / 2,
    )
    angles = np.linspace(0, 2 * np.pi, n_points)
    semi_axes = numeric.std(axis=0, ddof=1).to_numpy() * np.sqrt(limit)
    center = numeric.mean(axis=0).to_numpy()
    boundary = np.column_stack((np.cos(angles), np.sin(angles))) * semi_axes + center
    return boundary, float(limit)


# ---------------------------------------------------------------- loadings
def align_loading_signs(loadings: pd.DataFrame) -> pd.DataFrame:
    """Orient each PC so its first largest-magnitude loading is positive."""
    aligned = loadings.copy()
    for component in aligned:
        largest = aligned[component].abs().idxmax()
        if aligned.loc[largest, component] < 0:
            aligned[component] = -aligned[component]
    return aligned


def top_loadings(loadings: pd.DataFrame, n_features: int) -> pd.DataFrame:
    """Return the top features per PC, ranked by absolute loading."""
    if n_features < 1:
        raise ValueError("n_features must be at least one.")
    rows = []
    for component in loadings:
        ranked = pd.DataFrame({"Feature": loadings.index, "loading": loadings[component]})
        ranked["absolute_loading"] = ranked["loading"].abs()
        ranked = ranked.nlargest(n_features, "absolute_loading", keep="all")
        ranked.insert(0, "Component", component)
        rows.append(ranked)
    return pd.concat(rows, ignore_index=True)


def loading_concentration(loadings: pd.DataFrame, n_features: tuple[int, ...] = (10, 25, 50)) -> pd.DataFrame:
    """Return each PC's share of squared loading carried by its strongest features."""
    if not n_features or any(count < 1 for count in n_features):
        raise ValueError("n_features must contain positive counts.")
    rows = []
    for component in loadings:
        squared = loadings[component].pow(2).sort_values(ascending=False, kind="stable")
        total = squared.sum()
        for count in n_features:
            rows.append({"Component": component, "n_features": count, "squared_loading_share": squared.iloc[:count].sum() / total})
    return pd.DataFrame(rows)


def class_composition(loadings: pd.DataFrame, classes: pd.Series, n_largest: int = 6) -> pd.DataFrame:
    """Summarise each component's squared-loading share by lipid class."""
    if n_largest < 1:
        raise ValueError("n_largest must be positive.")
    classes = classes.reindex(loadings.index).fillna("Unannotated").astype(str)
    panel_share = classes.value_counts(normalize=True)
    rows = []
    for component in loadings:
        shares = loadings[component].pow(2).groupby(classes, observed=True).sum()
        shares = shares / shares.sum()
        largest = shares.nlargest(n_largest, keep="all").index
        kept = shares.loc[largest]
        if len(largest) < len(shares):
            kept.loc["Other"] = shares.drop(largest).sum()
        for lipid_class, share in kept.items():
            rows.append({
                "Component": component,
                "Class": lipid_class,
                "squared_loading_share": share,
                "panel_feature_share": panel_share.get(lipid_class, np.nan),
                "enrichment_vs_panel": share / panel_share[lipid_class] if lipid_class in panel_share else np.nan,
            })
    return pd.DataFrame(rows)


def cumulative_share_curve(loadings: pd.DataFrame) -> pd.DataFrame:
    """Return cumulative squared-loading shares ordered by absolute loading."""
    rows = []
    for component in loadings:
        squared = loadings[component].pow(2).sort_values(ascending=False, kind="stable")
        total = squared.sum()
        for count, share in enumerate((squared / total).cumsum(), start=1):
            rows.append({"Component": component, "n_features": count, "cumulative_squared_loading_share": share, "uniform_share": count / len(squared)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- design attribution
def score_attribution(
    scores: pd.DataFrame, variance: np.ndarray, metadata: pd.DataFrame, scope: str,
) -> pd.DataFrame:
    """Marginal R², not additive variance partitioning or an inferential test."""
    metadata = metadata.set_index("Sample").loc[scores.index]
    factors = [("Region", "Sample type")] if scope == "pooled" else []
    factors += [("Bio group", "Bio group"), ("Sex", "Sex"), ("Tissue mass", "Tissue mass (mg)"),
                ("Cage", "Cage"), ("Collection date", "Date collected")]
    rows = []
    for component, pc_share in zip(scores.columns[:5], variance[:5]):
        for factor, column in [("Total", None), *factors]:
            if factor == "Total":
                r2 = 1.0
            elif factor == "Tissue mass":
                r2 = r_squared(scores[component], pd.to_numeric(metadata[column], errors="raise"))
            else:
                r2 = r_squared(scores[component], metadata[column].astype(str))
            rows.append({"scope": scope, "Component": component, "Factor": factor,
                         "pc_variance_explained": pc_share, "factor_score_r2": r2,
                         "factor_share_of_total_variance": pc_share * r2})
    return pd.DataFrame(rows)


def pooled_region_group_composite(
    scores: pd.DataFrame, variance: np.ndarray, metadata: pd.DataFrame,
    attribution: pd.DataFrame, output: Path | None = None,
) -> plt.Figure:
    """Show both PC pairs with region/group, cage/date and marginal attribution.

    Returns the figure; if ``output`` is given it is also saved there (and closed).
    """
    joined = metadata.set_index("Sample").loc[scores.index].copy()
    joined["Date collected"] = pd.to_datetime(joined["Date collected"]).dt.strftime("%Y-%m-%d")
    figure = plt.figure(figsize=(17, 11))
    grid = figure.add_gridspec(2, 3, width_ratios=[1, 1, 1.25], height_ratios=[1, 1.8])
    lower = grid[1, :2].subgridspec(2, 2, hspace=0.48)
    pairs = [("PC 1", "PC 2"), ("PC 3", "PC 4")]
    for column, (x, y) in enumerate(pairs):
        axis = figure.add_subplot(grid[0, column])
        for region, color in BRAIN_REGION_COLORS.items():
            for group, marker in [("Saline Control", "o"), ("MIA", "^")]:
                selected = joined["Sample type"].eq(region) & joined["Bio group"].eq(group)
                axis.scatter(scores.loc[selected, x], scores.loc[selected, y], marker=marker,
                             facecolors=color if group == "MIA" else "none", edgecolors=color, s=40)
        axis.set_title("Region (colour) and treatment (marker)")
        axes = [axis]
        for row, variable in enumerate(("Cage", "Date collected")):
            current = figure.add_subplot(lower[row, column])
            categories = sorted(joined[variable].astype(str).unique())
            for number, category in enumerate(categories):
                selected = joined[variable].astype(str).eq(category)
                current.scatter(scores.loc[selected, x], scores.loc[selected, y],
                                color=(*COLORBLIND_COLORS, "#000000")[number],
                                marker=MARKER_CYCLE[number], s=32, label=category)
            current.set_title(variable)
            if column == 0:
                current.legend(fontsize=7, ncol=2, loc="best")
            axes.append(current)
        for current in axes:
            current.axhline(0, color="0.8", linewidth=0.6)
            current.axvline(0, color="0.8", linewidth=0.6)
            current.set_xlabel(f"{x} ({variance[int(x.split()[1]) - 1] * 100:.1f}%)")
            current.set_ylabel(f"{y} ({variance[int(y.split()[1]) - 1] * 100:.1f}%)")

    heatmap = figure.add_subplot(grid[:, 2])
    factors = ["Region", "Bio group", "Sex", "Cage", "Collection date", "Tissue mass"]
    table = attribution.pivot(index="Component", columns="Factor", values="factor_score_r2").loc[scores.columns[:5], factors]
    image = heatmap.imshow(table, vmin=0, vmax=1, cmap="cividis", aspect="auto")
    heatmap.grid(False)
    for row in range(len(table)):
        for column in range(len(factors)):
            value = table.iloc[row, column]
            heatmap.text(column, row, f"{value:.2f}", ha="center", va="center",
                         color="white" if value < 0.5 else "black", fontsize=9)
    heatmap.set(xticks=range(len(factors)), xticklabels=factors,
                yticks=range(len(table)), yticklabels=table.index,
                title="Marginal score attribution (R²)")
    heatmap.tick_params(axis="x", labelrotation=60)
    figure.colorbar(image, ax=heatmap, shrink=0.5, label="R² (overlapping factors)")
    handles = [Line2D([], [], color=color, marker="s", linestyle="none", label=region)
               for region, color in BRAIN_REGION_COLORS.items()]
    handles += [Line2D([], [], color="black", marker=marker, markerfacecolor=fill,
                       linestyle="none", label=label)
                for marker, fill, label in [("o", "none", "Saline Control"), ("^", "black", "MIA")]]
    figure.legend(handles=handles, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.98))
    figure.suptitle(f"Pooled PCA: {len(joined)} regional samples, {joined['Mouse number'].nunique()} mice", y=1.00)
    figure.text(0.5, 0.01,
                "Log₂-normalized intensities; complete features; autoscaled. Descriptive only; regional samples repeat within mouse.\n"
                "Group-associated variance is cage/date-aliased wherever cage/date R² exceeds group R²; treatment is confounded with these factors.\n"
                "Marginal R² values overlap and do not establish an independent treatment effect.", ha="center", fontsize=9)
    figure.subplots_adjust(top=0.92, bottom=0.16, left=0.05, right=0.97, hspace=0.32, wspace=0.38)
    if output is not None:
        save_figure(figure, output)
    return figure
