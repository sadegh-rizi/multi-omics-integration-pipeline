"""Publication-ready diagnostic plots for the outlier ensemble evidence."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.stats import beta

from ..plotting import COLORBLIND_COLORS, apply_plot_style, save_figure


SCOPE_ORDER = ("pooled", "cerebellum", "cortex", "hippocampus")
SCOPE_LABELS = {
    "pooled": "Pooled",
    "cerebellum": "Cerebellum",
    "cortex": "Cortex",
    "hippocampus": "Hippocampus",
}
SCOPE_COLORS = {
    "pooled": "#666666",
    "cerebellum": "#E69F00",
    "cortex": "#56B4E9",
    "hippocampus": "#0072B2",
}
DETECTOR_ORDER = (
    "knn_isotropic",
    "knn_soft",
    "lof",
    "mahalanobis",
    "isolation_forest",
    "kernel_pca",
)
DETECTOR_LABELS = {
    "knn_isotropic": "k-NN\nisotropic",
    "knn_soft": "k-NN\nsoft",
    "lof": "LOF",
    "mahalanobis": "Mahalanobis",
    "isolation_forest": "Isolation\nForest",
    "kernel_pca": "Kernel\nPCA",
}
DETECTOR_COLORS = dict(zip(DETECTOR_ORDER, COLORBLIND_COLORS[: len(DETECTOR_ORDER)], strict=True))
REPRESENTATION_MARKERS = {"raw": "o", "pc3": "s", "pc5": "^", "pc10": "D"}


def _normalise_scores(scores: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Sample",
        "score",
        "null_percentile",
        "flag_95",
        "detector",
        "parameter_name",
        "parameter_value",
        "representation",
        "scope",
        "calibration_stage",
    }
    if missing := required.difference(scores.columns):
        raise ValueError(f"Missing plotting columns: {sorted(missing)}")
    result = scores.copy()
    result["parameter_value"] = result["parameter_value"].astype(str)
    result["display_detector"] = result["detector"].replace({"lof_supporting": "lof"})
    return result


def sample_label_mapping(metadata: pd.DataFrame) -> dict[str, str]:
    """Create compact y-axis labels from sample number, sex, and bio group."""
    required = {"Sample", "Sample number", "Bio group", "Sex"}
    if missing := required.difference(metadata.columns):
        raise ValueError(f"Missing sample metadata columns: {sorted(missing)}")
    sex_codes = {"Male": "M", "Female": "F"}
    group_codes = {"MIA": "T", "Saline Control": "C"}
    unknown_sex = sorted(set(metadata["Sex"].dropna()).difference(sex_codes))
    unknown_group = sorted(set(metadata["Bio group"].dropna()).difference(group_codes))
    if unknown_sex or unknown_group:
        raise ValueError(f"Unknown sample metadata: sex={unknown_sex}, bio_group={unknown_group}")
    sample_numbers = pd.to_numeric(metadata["Sample number"], errors="raise").astype(int)
    labels = (
        metadata["Sex"].map(sex_codes)
        + "_"
        + metadata["Bio group"].map(group_codes)
        + "_"
        + sample_numbers.map(lambda number: f"{number:02d}")
    )
    mapping = dict(zip(metadata["Sample"].astype(str), labels, strict=True))
    if len(mapping) != len(metadata):
        raise ValueError("Sample metadata contains duplicate Sample identifiers.")
    return mapping


def _scope_order(table: pd.DataFrame) -> list[str]:
    present = set(table["scope"].dropna().astype(str))
    return [scope for scope in SCOPE_ORDER if scope in present] + sorted(present.difference(SCOPE_ORDER))


def _parameter_sort_key(value: str) -> tuple[int, float, str]:
    text = str(value)
    if text.lower() == "auto":
        return (1, 0.0, text)
    try:
        return (0, float(text), text)
    except ValueError:
        return (0, float("inf"), text)


def _parameter_values(table: pd.DataFrame) -> list[str]:
    return sorted(table["parameter_value"].dropna().astype(str).unique(), key=_parameter_sort_key)


def _format_parameter(value: str) -> str:
    text = str(value)
    if text.lower() == "auto":
        return "auto"
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return f"{number:.3g}"


def _parameter_mask(values: pd.Series, target: str | float | int) -> pd.Series:
    target_text = str(target)
    mask = values.eq(target_text)
    if target_text.lower() != "auto":
        numeric = pd.to_numeric(values, errors="coerce")
        mask = mask | np.isclose(numeric, float(target), equal_nan=False)
    return mask


def _representative_percentiles(scores: pd.DataFrame) -> pd.DataFrame:
    """Select the same representative settings used by the ensemble verdict."""
    rows: list[pd.DataFrame] = []
    for scope in _scope_order(scores):
        scope_scores = scores.loc[scores["scope"].eq(scope) & scores["calibration_stage"].eq("trimmed")]
        samples = sorted(scope_scores["Sample"].dropna().unique())
        n_samples = len(samples)
        lof_detector = "lof" if scope == "pooled" else "lof_supporting"
        specifications = (
            ("knn_isotropic", "knn_isotropic", round(np.sqrt(n_samples)), "pc3"),
            ("knn_soft", "knn_soft", round(np.sqrt(n_samples)), "pc3"),
            ("lof", lof_detector, 10 if scope == "pooled" else 8, "pc3"),
            ("mahalanobis", "mahalanobis", "auto", "raw"),
            ("isolation_forest", "isolation_forest", "50", "pc3"),
            ("kernel_pca", "kernel_pca", np.logspace(-2, 0, 9)[4], "pc3"),
        )
        for display_detector, detector, parameter, representation in specifications:
            selected = scope_scores.loc[
                scope_scores["detector"].eq(detector)
                & _parameter_mask(scope_scores["parameter_value"], parameter)
                & scope_scores["representation"].eq(representation)
            ].set_index("Sample")["null_percentile"].reindex(samples)
            if selected.empty or selected.isna().any():
                continue
            rows.append(pd.DataFrame({
                "Sample": samples,
                "scope": scope,
                "detector": display_detector,
                "null_percentile": selected.to_numpy(),
                "parameter_name": scope_scores.loc[
                    scope_scores["detector"].eq(detector)
                    & _parameter_mask(scope_scores["parameter_value"], parameter)
                    & scope_scores["representation"].eq(representation),
                    "parameter_name",
                ].iloc[0],
                "parameter_value": _format_parameter(str(parameter)),
            }))
    if not rows:
        return pd.DataFrame(columns=["Sample", "scope", "detector", "null_percentile", "parameter_name", "parameter_value"])
    return pd.concat(rows, ignore_index=True)


def _stability_matrix(
    scores: pd.DataFrame,
    scope: str,
) -> pd.DataFrame:
    scope_scores = scores.loc[scores["scope"].eq(scope) & scores["calibration_stage"].eq("trimmed")]
    counted = scope_scores.loc[scope_scores["counts_for_stability"].astype(str).str.lower().eq("true")]
    if scope != "pooled":
        supporting_lof = scope_scores.loc[scope_scores["detector"].eq("lof_supporting")]
        counted = pd.concat([counted, supporting_lof], ignore_index=True)
    counted = counted.assign(display_detector=counted["detector"].replace({"lof_supporting": "lof"}))
    stability = counted.groupby(["Sample", "display_detector"], observed=True)["flag_95"].mean()
    samples = sorted(scope_scores["Sample"].dropna().unique())
    matrix = stability.unstack("display_detector").reindex(index=samples, columns=DETECTOR_ORDER)
    if matrix.empty:
        return matrix
    order = (matrix.fillna(0).max(axis=1) + matrix.fillna(0).mean(axis=1) / 1_000).sort_values(ascending=False).index
    return matrix.loc[order]


def _draw_heatmap(
    axis: plt.Axes,
    matrix: pd.DataFrame,
    scope: str,
    n_total: int | None = None,
    sample_labels: Mapping[str, str] | None = None,
) -> mpl.image.AxesImage:
    cmap = plt.get_cmap("Blues").with_extremes(bad="#E6E6E6")
    detectors = list(matrix.columns) if len(matrix.columns) else list(DETECTOR_ORDER)
    values = matrix.to_numpy(dtype=float) if not matrix.empty else np.full((1, len(detectors)), np.nan)
    total = len(matrix) if n_total is None else n_total
    sample_count = f"n={total}" if len(matrix) == total else f"n={total}; {len(matrix)} shown"
    yticklabels = [
        sample_labels.get(str(sample), str(sample)) if sample_labels is not None else str(sample)
        for sample in matrix.index
    ]
    image = axis.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, norm=mpl.colors.Normalize(0, 1))
    axis.set(
        xticks=np.arange(len(detectors)),
        xticklabels=[DETECTOR_LABELS.get(name, str(name)) for name in detectors],
        yticks=np.arange(len(matrix)),
        yticklabels=yticklabels if not matrix.empty else [],
        title=f"{SCOPE_LABELS.get(scope, scope.title())} ({sample_count})",
    )
    axis.tick_params(axis="x", labelrotation=0)
    axis.tick_params(axis="y", labelsize=6)
    if matrix.empty:
        axis.text(0.5, 0.5, "No nonzero stability", ha="center", va="center", transform=axis.transAxes)
    return image


def _hotelling_t2_flags(verdict: pd.DataFrame, scope: str, samples: pd.Index) -> pd.Series:
    """Return 3-PC in-sample Hotelling T2 95% flags in heatmap row order."""
    required = {"Sample", "scope", "hotelling_t2"}
    if missing := required.difference(verdict.columns):
        raise ValueError(f"Missing Hotelling T2 columns: {sorted(missing)}")
    current = verdict.loc[verdict["scope"].eq(scope), ["Sample", "hotelling_t2"]].set_index("Sample")
    n_samples = current.index.nunique()
    n_components = 3
    if n_samples <= n_components + 1:
        raise ValueError("Hotelling T2 requires more samples than retained PCs plus one.")
    limit = (n_samples - 1) ** 2 / n_samples * beta.ppf(
        0.95,
        n_components / 2,
        (n_samples - n_components - 1) / 2,
    )
    return current["hotelling_t2"].gt(limit).reindex(samples).fillna(False)


def _draw_heatmap_panel(
    figure: plt.Figure,
    heatmap_slot: mpl.gridspec.SubplotSpec,
    strip_slot: mpl.gridspec.SubplotSpec,
    matrix: pd.DataFrame,
    scope: str,
    t2_flags: pd.Series,
    n_total: int,
    sample_labels: Mapping[str, str] | None = None,
) -> tuple[plt.Axes, plt.Axes, mpl.image.AxesImage]:
    heatmap_axis = figure.add_subplot(heatmap_slot)
    strip_axis = figure.add_subplot(strip_slot)
    image = _draw_heatmap(heatmap_axis, matrix, scope, n_total=n_total, sample_labels=sample_labels)
    flags = t2_flags.to_numpy(dtype=float)[:, None]
    if flags.size == 0:
        flags = np.full((1, 1), np.nan)
    strip_axis.imshow(
        np.ma.masked_invalid(flags),
        aspect="auto",
        cmap=mpl.colors.ListedColormap(["#F0F0F0", "#D55E00"]),
        norm=mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5], 2),
    )
    strip_axis.set(
        xticks=[0],
        xticklabels=["T2\n>95%"],
        yticks=np.arange(len(matrix)),
        yticklabels=[],
    )
    strip_axis.set_ylim(heatmap_axis.get_ylim())
    strip_axis.tick_params(axis="x", length=0, labelsize=8)
    strip_axis.tick_params(axis="y", left=False, labelleft=False)
    return heatmap_axis, strip_axis, image


def plot_stability_heatmap(
    scores: pd.DataFrame,
    output_dir: Path,
    sample_labels: Mapping[str, str] | None = None,
) -> None:
    """Save a continuous stability heatmap, faceted across analysis scopes."""
    scores = _normalise_scores(scores)
    scopes = list(SCOPE_ORDER)
    fig, axes = plt.subplots(2, 2, figsize=(12, 15), constrained_layout=True, squeeze=False)
    images = []
    for axis, scope in zip(axes.flat, scopes, strict=True):
        n_total = scores.loc[
            scores["scope"].eq(scope) & scores["calibration_stage"].eq("trimmed"), "Sample"
        ].nunique()
        image = _draw_heatmap(
            axis,
            _stability_matrix(scores, scope),
            scope,
            n_total=n_total,
            sample_labels=sample_labels,
        )
        images.append(image)
    axes[0, 0].set_ylabel("Sample")
    axes[1, 0].set_ylabel("Sample")
    fig.colorbar(images[-1], ax=axes.ravel().tolist(), label="Stability fraction (flagged settings)", shrink=0.7)
    fig.suptitle("Outlier stability across detector settings", fontweight="bold")
    save_figure(fig, output_dir / "stability_heatmap_all_scopes")
    plt.close(fig)

    # Keep the established per-scope filenames, but replace their old binary
    # content with the same continuous encoding for easy comparison.
    for scope in scopes:
        matrix = _stability_matrix(scores, scope)
        n_total = scores.loc[
            scores["scope"].eq(scope) & scores["calibration_stage"].eq("trimmed"), "Sample"
        ].nunique()
        fig, axis = plt.subplots(figsize=(8, max(4, len(matrix) * 0.25)), constrained_layout=True)
        image = _draw_heatmap(axis, matrix, scope, n_total=n_total, sample_labels=sample_labels)
        axis.set_ylabel("Sample")
        fig.colorbar(image, ax=axis, label="Stability fraction (flagged settings)")
        save_figure(fig, output_dir / f"stability_heatmap_{scope}")
        plt.close(fig)


def plot_stability_heatmap_without_mahalanobis(
    scores: pd.DataFrame,
    verdict: pd.DataFrame,
    output_dir: Path,
    sample_labels: Mapping[str, str] | None = None,
) -> None:
    """Save an additional stability heatmap with Mahalanobis excluded."""
    scores = _normalise_scores(scores)
    scopes = list(SCOPE_ORDER)
    fig = plt.figure(figsize=(13, 15))
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=(20, 1, 20, 1),
        left=0.06,
        right=0.90,
        top=0.93,
        bottom=0.09,
        hspace=0.22,
        wspace=0.05,
    )
    axes = np.empty((2, 2), dtype=object)
    images = []
    for index, scope in enumerate(scopes):
        row, column = divmod(index, 2)
        n_total = scores.loc[
            scores["scope"].eq(scope) & scores["calibration_stage"].eq("trimmed"), "Sample"
        ].nunique()
        matrix = _stability_matrix(scores, scope).drop(columns=["mahalanobis"], errors="ignore")
        axis, _, image = _draw_heatmap_panel(
            fig,
            grid[row, column * 2],
            grid[row, column * 2 + 1],
            matrix,
            scope,
            _hotelling_t2_flags(verdict, scope, matrix.index),
            n_total,
            sample_labels=sample_labels,
        )
        axes[row, column] = axis
        images.append(image)
    axes[0, 0].set_ylabel("Sample")
    axes[1, 0].set_ylabel("Sample")
    colorbar_axis = fig.add_axes([0.92, 0.25, 0.018, 0.48])
    fig.colorbar(images[-1], cax=colorbar_axis, label="Stability fraction (flagged settings)")
    fig.legend(
        handles=[
            Patch(facecolor="#D55E00", label="Hotelling T2 > 95% limit"),
            Patch(facecolor="#F0F0F0", edgecolor="#999999", label="Hotelling T2 <= 95% limit"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "Outlier stability across detector settings "
        "(Mahalanobis excluded; Hotelling T2 side-strip)",
        fontweight="bold",
    )
    save_figure(fig, output_dir / "stability_heatmap_without_mahalanobis_all_scopes")
    plt.close(fig)

    for scope in scopes[1:]:
        matrix = _stability_matrix(scores, scope).drop(columns=["mahalanobis"], errors="ignore")
        n_total = scores.loc[
            scores["scope"].eq(scope) & scores["calibration_stage"].eq("trimmed"), "Sample"
        ].nunique()
        fig = plt.figure(figsize=(8.5, max(4, len(matrix) * 0.25)))
        grid = fig.add_gridspec(
            1,
            2,
            width_ratios=(20, 1),
            left=0.08,
            right=0.82,
            top=0.88,
            bottom=0.16,
            wspace=0.05,
        )
        axis, _, image = _draw_heatmap_panel(
            fig,
            grid[0, 0],
            grid[0, 1],
            matrix,
            scope,
            _hotelling_t2_flags(verdict, scope, matrix.index),
            n_total,
            sample_labels=sample_labels,
        )
        axis.set_ylabel("Sample")
        colorbar_axis = fig.add_axes([0.85, 0.24, 0.025, 0.52])
        fig.colorbar(image, cax=colorbar_axis, label="Stability fraction (flagged settings)")
        fig.legend(
            handles=[
                Patch(facecolor="#D55E00", label="Hotelling T2 > 95% limit"),
                Patch(facecolor="#F0F0F0", edgecolor="#999999", label="Hotelling T2 <= 95% limit"),
            ],
            loc="lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
        )
        fig.suptitle(
            f"Outlier stability: {SCOPE_LABELS.get(scope, scope.title())} "
            "(Mahalanobis excluded; Hotelling T2 side-strip)",
            fontweight="bold",
        )
        save_figure(fig, output_dir / f"stability_heatmap_without_mahalanobis_{scope}")
        plt.close(fig)


def plot_corroboration_scatter(verdict: pd.DataFrame, output_dir: Path) -> None:
    """Show the detector-corroboration rule and label every sample."""
    required = {"Sample", "scope", "stable_detector_count", "stable_non_covariance_count"}
    if missing := required.difference(verdict.columns):
        raise ValueError(f"Missing corroboration columns: {sorted(missing)}")
    scopes = list(SCOPE_ORDER)
    max_x = max(2, int(verdict["stable_detector_count"].max()))
    max_y = max(1, int(verdict["stable_non_covariance_count"].max()))
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True, squeeze=False)
    for axis, scope in zip(axes.flat, scopes, strict=True):
        current = verdict.loc[verdict["scope"].eq(scope)].copy()
        x_limit = max_x + 0.5
        y_limit = max_y + 0.5
        axis.add_patch(Rectangle((1.5, 0.5), x_limit - 1.5, y_limit - 0.5, facecolor="#D55E00", alpha=0.12, zorder=0))
        axis.axvline(1.5, color="#777777", linestyle="--", linewidth=0.8)
        axis.axhline(0.5, color="#777777", linestyle="--", linewidth=0.8)
        axis.scatter(
            current["stable_detector_count"],
            current["stable_non_covariance_count"],
            s=30,
            color="#0072B2",
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )
        for (x_value, y_value), group in current.groupby(["stable_detector_count", "stable_non_covariance_count"], observed=True):
            labels = ", ".join(group["Sample"].astype(str))
            axis.annotate(labels, (x_value, y_value), xytext=(3, 3), textcoords="offset points", fontsize=7)
        axis.set(
            xlim=(0.5, x_limit),
            ylim=(0.5, y_limit),
            xticks=np.arange(1, max_x + 1),
            yticks=np.arange(1, max_y + 1),
            xlabel="Stable detector count",
            ylabel="Stable non-covariance count",
            title=f"{SCOPE_LABELS.get(scope, scope.title())} (n={len(current)})",
        )
        axis.text(0.04, 0.94, "Shaded: count >= 2 and non-covariance >= 1", transform=axis.transAxes, fontsize=8, va="top")
    fig.suptitle("Detector corroboration and the candidate rule", fontweight="bold")
    save_figure(fig, output_dir / "corroboration_scatter_all_scopes")
    plt.close(fig)


def plot_percentile_strip(scores: pd.DataFrame, output_dir: Path) -> None:
    """Plot one null-percentile point per sample at each representative setting."""
    representative = _representative_percentiles(_normalise_scores(scores))
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True, squeeze=False)
    for axis, scope in zip(axes.flat, SCOPE_ORDER, strict=True):
        current = representative.loc[representative["scope"].eq(scope)]
        matrix = current.pivot(index="Sample", columns="detector", values="null_percentile").reindex(columns=DETECTOR_ORDER)
        y_positions = {detector: index for index, detector in enumerate(DETECTOR_ORDER)}
        for sample, sample_values in matrix.iterrows():
            present = sample_values.dropna()
            if len(present) > 1:
                axis.plot(present.to_numpy(), [y_positions[name] for name in present.index], color="#BDBDBD", linewidth=0.5, alpha=0.35, zorder=1)
        for detector in DETECTOR_ORDER:
            values = matrix[detector].dropna() if detector in matrix else pd.Series(dtype=float)
            if values.empty:
                continue
            axis.scatter(
                values.to_numpy(),
                np.full(len(values), y_positions[detector]),
                s=24,
                color=DETECTOR_COLORS[detector],
                edgecolor="white",
                linewidth=0.4,
                label=DETECTOR_LABELS[detector].replace("\n", " "),
                zorder=2,
            )
        axis.axvline(0.95, color="#D55E00", linestyle="--", linewidth=1.0, label="0.95 reference")
        axis.set(
            xlim=(0, 1),
            xticks=np.linspace(0, 1, 11),
            yticks=np.arange(len(DETECTOR_ORDER)),
            yticklabels=[DETECTOR_LABELS[name].replace("\n", " ") for name in DETECTOR_ORDER],
            xlabel="Null percentile (representative setting)",
            ylabel="Detector",
            title=f"{SCOPE_LABELS.get(scope, scope.title())} (n={matrix.index.size})",
        )
    handles = [Line2D([0], [0], marker="o", linestyle="", color=DETECTOR_COLORS[name], label=DETECTOR_LABELS[name].replace("\n", " ")) for name in DETECTOR_ORDER]
    handles.append(Line2D([0], [0], color="#D55E00", linestyle="--", label="0.95 reference"))
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1, frameon=False)
    fig.suptitle("Representative detector null percentiles", fontweight="bold")
    save_figure(fig, output_dir / "percentile_strip_all_scopes")
    plt.close(fig)


def _display_limits(limits: pd.DataFrame) -> pd.DataFrame:
    required = {"scope", "detector", "parameter_name", "parameter_value", "representation", "calibration_stage", "false_positive_rate"}
    if missing := required.difference(limits.columns):
        raise ValueError(f"Missing calibration columns: {sorted(missing)}")
    result = limits.copy()
    result["parameter_value"] = result["parameter_value"].astype(str)
    result["display_detector"] = result["detector"].replace({"lof_supporting": "lof"})
    return result.loc[result["false_positive_rate"].notna()]


def _scope_handles() -> list[Line2D]:
    return [Line2D([0], [0], marker="o", linestyle="", color=SCOPE_COLORS.get(scope, COLORBLIND_COLORS[0]), label=SCOPE_LABELS.get(scope, scope.title())) for scope in SCOPE_ORDER]


def _representation_handles() -> list[Line2D]:
    return [Line2D([0], [0], marker=marker, linestyle="", color="#555555", label=representation) for representation, marker in REPRESENTATION_MARKERS.items()]


def _stage_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", linestyle="", color="#555555", markerfacecolor="black", label="contaminated"),
        Line2D([0], [0], marker="o", linestyle="", color="#555555", markerfacecolor="white", label="trimmed"),
    ]


def _draw_calibration_panel(axis: plt.Axes, panel: pd.DataFrame, display_detector: str) -> None:
    if panel.empty:
        axis.set_title(DETECTOR_LABELS[display_detector].replace("\n", " "))
        axis.text(0.5, 0.5, "No calibration rows", ha="center", va="center", transform=axis.transAxes)
        return
    parameters = _parameter_values(panel)
    positions = {parameter: index for index, parameter in enumerate(parameters)}
    groups = [
        (scope, representation, stage)
        for scope in _scope_order(panel)
        for representation in REPRESENTATION_MARKERS
        for stage in ("contaminated", "trimmed")
        if not panel.loc[
            panel["scope"].eq(scope)
            & panel["representation"].eq(representation)
            & panel["calibration_stage"].eq(stage)
        ].empty
    ]
    offsets = np.linspace(-0.3, 0.3, max(1, len(groups)))
    for offset, (scope, representation, stage) in zip(offsets, groups, strict=True):
        current = panel.loc[
            panel["scope"].eq(scope)
            & panel["representation"].eq(representation)
            & panel["calibration_stage"].eq(stage)
        ]
        x_values = []
        y_values = []
        for parameter in parameters:
            row = current.loc[current["parameter_value"].eq(parameter)]
            if row.empty:
                continue
            x_values.append(positions[parameter] + offset)
            y_values.append(float(row["false_positive_rate"].iloc[0]))
        color = SCOPE_COLORS.get(scope, COLORBLIND_COLORS[0])
        facecolor = color if stage == "contaminated" else "white"
        axis.scatter(x_values, y_values, color=facecolor, marker=REPRESENTATION_MARKERS[representation], s=28, edgecolor=color, linewidth=0.8)
    axis.axhline(0.05, color="#D55E00", linestyle="--", linewidth=1.0)
    axis.set(
        xticks=np.arange(len(parameters)),
        xticklabels=[_format_parameter(parameter) for parameter in parameters],
        xlabel=str(panel["parameter_name"].iloc[0]),
        title=DETECTOR_LABELS[display_detector].replace("\n", " "),
    )
    axis.tick_params(axis="x", labelrotation=90)


def plot_calibration_diagnostic(limits: pd.DataFrame, output_dir: Path) -> None:
    """Plot held-out null FPR for every detector setting."""
    table = _display_limits(limits)
    maximum = float(table["false_positive_rate"].max()) if not table.empty else 0.05
    upper = min(1.0, max(0.1, maximum * 1.15))
    fig, axes = plt.subplots(2, 3, figsize=(14, 10), constrained_layout=True, squeeze=False)
    for axis, detector in zip(axes.flat, DETECTOR_ORDER, strict=True):
        _draw_calibration_panel(axis, table.loc[table["display_detector"].eq(detector)], detector)
        axis.set_ylim(0, upper)
    fig.supxlabel("Parameter setting; labels show the numeric setting", y=0.02)
    fig.supylabel("Held-out null false-positive rate", x=0.02)
    handles = _scope_handles() + _representation_handles() + _stage_handles() + [Line2D([0], [0], color="#D55E00", linestyle="--", label="0.05 reference")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Calibration diagnostic across detector settings", fontweight="bold")
    save_figure(fig, output_dir / "calibration_fpr_all_detectors")
    plt.close(fig)


def _draw_sweep_panel(axis: plt.Axes, panel: pd.DataFrame, display_detector: str) -> None:
    if panel.empty:
        axis.set_title(DETECTOR_LABELS[display_detector].replace("\n", " "))
        axis.text(0.5, 0.5, "No detector rows", ha="center", va="center", transform=axis.transAxes)
        return
    parameters = _parameter_values(panel)
    positions = {parameter: index for index, parameter in enumerate(parameters)}
    parameter_name = str(panel["parameter_name"].iloc[0])
    for representation, current in panel.groupby("representation", observed=True, sort=False):
        fraction = current.groupby("parameter_value", observed=True)["flag_95"].mean()
        ordered = [parameter for parameter in parameters if parameter in fraction.index]
        axis.plot(
            [positions[parameter] for parameter in ordered],
            [fraction[parameter] for parameter in ordered],
            marker=REPRESENTATION_MARKERS.get(representation, "o"),
            label=representation,
            color=DETECTOR_COLORS[display_detector],
            alpha=0.85,
        )
    axis.axhline(0.05, color="#D55E00", linestyle="--", linewidth=1.0, label="0.05 reference")
    axis.set(
        xlim=(-0.5, len(parameters) - 0.5),
        ylim=(0, 1),
        xticks=np.arange(len(parameters)),
        xticklabels=[f"{parameter_name}={_format_parameter(parameter)}" for parameter in parameters],
        xlabel="Detector setting",
        ylabel="Proportion flagged at 95%",
        title=DETECTOR_LABELS[display_detector].replace("\n", " "),
    )
    axis.tick_params(axis="x", labelrotation=90)
    axis.legend(fontsize=8)


def plot_detector_sweeps(scores: pd.DataFrame, output_dir: Path) -> None:
    """Plot observed flagged proportions across every detector sweep."""
    scores = _normalise_scores(scores)
    for scope in SCOPE_ORDER:
        scope_scores = scores.loc[
            scores["scope"].eq(scope) & scores["calibration_stage"].eq("trimmed")
        ].copy()
        fig, axes = plt.subplots(2, 3, figsize=(15, 11), constrained_layout=True, squeeze=False)
        for axis, detector in zip(axes.flat, DETECTOR_ORDER, strict=True):
            panel = scope_scores.loc[scope_scores["display_detector"].eq(detector)]
            _draw_sweep_panel(axis, panel, detector)
        fig.suptitle(f"Detector sweep curves: {SCOPE_LABELS.get(scope, scope.title())}", fontweight="bold")
        save_figure(fig, output_dir / f"sweep_curves_all_detectors_{scope}")
        plt.close(fig)

        for detector in DETECTOR_ORDER:
            panel = scope_scores.loc[scope_scores["display_detector"].eq(detector)]
            if panel.empty:
                continue
            source_detector = "lof_supporting" if detector == "lof" and scope != "pooled" else detector
            fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
            _draw_sweep_panel(axis, panel, detector)
            axis.set_title(f"{DETECTOR_LABELS[detector].replace(chr(10), ' ')}: {SCOPE_LABELS.get(scope, scope.title())}")
            save_figure(fig, output_dir / f"sweep_curves_{source_detector}_{scope}")
            plt.close(fig)


def plot_outlier_diagnostics(
    scores: pd.DataFrame,
    limits: pd.DataFrame,
    verdict: pd.DataFrame,
    output_dir: Path,
    sample_labels: Mapping[str, str] | None = None,
) -> None:
    """Generate the complete requested diagnostic figure set."""
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_plot_style()
    plot_stability_heatmap(scores, output_dir, sample_labels=sample_labels)
    plot_stability_heatmap_without_mahalanobis(scores, verdict, output_dir, sample_labels=sample_labels)
    plot_corroboration_scatter(verdict, output_dir)
    plot_percentile_strip(scores, output_dir)
    plot_calibration_diagnostic(limits, output_dir)
    plot_detector_sweeps(scores, output_dir)
