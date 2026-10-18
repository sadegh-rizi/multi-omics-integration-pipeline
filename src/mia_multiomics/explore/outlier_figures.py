"""Sample outliers: the regional T2 / DModX screen recomputed with your settings,
suggested exclusions, assay concordance, and the calibrated ensemble.

Two kinds of input:

* :func:`outlier_table` and the plots built on it use the *view* (your
  exclusions and thresholds) - useful to see whether removing a sample
  makes another one stand out;
* :func:`suggest_exclusions` and the ensemble always use **all 60** samples
  with the default preprocessing: outliers are found before anything is
  removed.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.stats import f as f_dist

from .. import config, plotting, qc
from ..config import Settings
from ..multivariate.pca import fit_pca
from ..outliers.screen import sample_outlier_table
from ..stages import outliers as outlier_stage
from ._common import finish
from .view import Options, View, build_view, resolve_scope

DETECTOR_ORDER = ("mahalanobis", "knn_isotropic", "knn_soft", "lof", "lof_supporting", "isolation_forest", "kernel_pca")


# ---------------------------------------------------------------- regional screen
def outlier_table(view: View) -> pd.DataFrame:
    """Hotelling T2 (robust Mahalanobis too) and normalised DModX per region, with 95/99 % flags."""
    if "outliers" in view._cache:
        return view._cache["outliers"]
    k = view.options.outlier_pcs
    tables = []
    for region in config.REGIONS:
        matrix = view.matrix(region)
        model, scores, _ = fit_pca(matrix, n_components=k)
        table = sample_outlier_table(matrix, model, scores, random_state=view.options.seed)
        n_features, n_samples = matrix.shape
        table["normalized_dmodx"] = table["dmodx"] / np.sqrt(np.mean(table["dmodx"] ** 2))
        table["dmodx_95_limit"] = float(np.sqrt(f_dist.ppf(0.95, n_features - k, (n_features - k) * (n_samples - k - 1))))
        table["normalized_dmodx_95_flag"] = table["normalized_dmodx"].gt(table["dmodx_95_limit"])
        tables.append(table.rename_axis("Sample").reset_index().assign(Region=region))
    meta = view.metadata[["Sample", "Sample number", config.MOUSE_COL, config.GROUP_COL, config.SEX_COL]]
    result = pd.concat(tables, ignore_index=True).merge(meta, on="Sample")
    view._cache["outliers"] = result
    return result


def suggest_exclusions(settings: Settings, omics: str = "lipidomics", level: float = 0.95) -> pd.DataFrame:
    """Samples flagged by the regional Hotelling T2 screen on all 60 samples (default preprocessing).

    These are the samples the thesis allows you to remove (``t2_95`` or
    ``t2_99``); copy their sample numbers into ``EXCLUDE``.
    """
    if level not in (0.95, 0.99):
        raise ValueError("level must be 0.95 or 0.99.")
    table = outlier_table(build_view(settings, Options(omics=omics)))
    flag = f"hotelling_t2_{int(level * 100)}_flag"
    columns = ["Sample number", "Sample", "Region", config.MOUSE_COL, config.GROUP_COL, config.SEX_COL,
               "hotelling_t2", f"hotelling_t2_{int(level * 100)}_limit", "normalized_dmodx", "normalized_dmodx_95_flag"]
    return table.loc[table[flag], columns].sort_values("Sample number").reset_index(drop=True)


def _region_rows(view: View, region: str) -> pd.DataFrame:
    table = outlier_table(view)
    return table.loc[table["Region"].eq(resolve_scope(region))].set_index("Sample")


def t2_plot(view: View, region: str = "Cortex") -> plt.Figure:
    """Hotelling T² of every sample of one region, with the 95 % and 99 % limits."""
    rows = _region_rows(view, region)
    fig = plotting.index_plot(rows["hotelling_t2"], rows["Sample number"],
                              {"95 %": rows["hotelling_t2_95_limit"].iloc[0], "99 %": rows["hotelling_t2_99_limit"].iloc[0]},
                              title=f"{resolve_scope(region)}: Hotelling T2 ({view.options.outlier_pcs} PCs)", ylabel="T2")
    return finish(fig, view)


def dmodx_plot(view: View, region: str = "Cortex") -> plt.Figure:
    """Normalised distance to the PCA model of every sample of one region, with the 95 % limit."""
    rows = _region_rows(view, region)
    fig = plotting.index_plot(rows["normalized_dmodx"], rows["Sample number"], {"95 %": rows["dmodx_95_limit"].iloc[0]},
                              title=f"{resolve_scope(region)}: distance to the PCA model (DModX)", ylabel="normalised DModX")
    return finish(fig, view)


def influence_plot(view: View, region: str = "Cortex") -> plt.Figure:
    """T2 (unusual *within* the model) against DModX (badly *described* by the model)."""
    rows = _region_rows(view, region)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    colors = rows[config.GROUP_COL].map(plotting.BIO_GROUP_COLORS)
    ax.scatter(rows["hotelling_t2"], rows["normalized_dmodx"], color=colors, edgecolor="black", linewidth=0.3)
    for _, row in rows.iterrows():
        ax.annotate(str(row["Sample number"]), (row["hotelling_t2"], row["normalized_dmodx"]), fontsize=7, xytext=(3, 2), textcoords="offset points")
    ax.axvline(rows["hotelling_t2_95_limit"].iloc[0], color=plotting.COLORBLIND_COLORS[1], linestyle="--", label="T2 95 %")
    ax.axhline(rows["dmodx_95_limit"].iloc[0], color=plotting.COLORBLIND_COLORS[1], linestyle=":", label="DModX 95 %")
    ax.set(xlabel=f"Hotelling T2 ({view.options.outlier_pcs} PCs)", ylabel="normalised DModX",
           title=f"{resolve_scope(region)}: PCA influence plot")
    ax.legend(frameon=False)
    return finish(fig, view, "Top right: unusual in both ways. Colour = treatment group.")


def outlier_concordance(view: View, statistic: str = "hotelling_t2") -> plt.Figure:
    """Do lipidomics and metabolomics flag the same samples (95 % flags, all regions)?"""
    if statistic not in ("hotelling_t2", "normalized_dmodx"):
        raise ValueError("statistic must be 'hotelling_t2' or 'normalized_dmodx'.")
    table = qc.outlier_flag_concordance(outlier_table(view.sibling("lipidomics")), outlier_table(view.sibling("metabolomics")), statistic)
    counts = table.groupby(["lipidomics_flag", "metabolomics_flag"])["n_samples"].sum()
    matrix = np.array([[counts.get((False, False), 0), counts.get((False, True), 0)],
                       [counts.get((True, False), 0), counts.get((True, True), 0)]])
    fig, ax = plt.subplots(figsize=(5.6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    for (row, column), count in np.ndenumerate(matrix):
        ax.text(column, row, str(count), ha="center", va="center", fontsize=14, color="white" if count > matrix.max() / 2 else "black")
    ax.set(xticks=[0, 1], xticklabels=["not flagged", "flagged"], yticks=[0, 1], yticklabels=["not flagged", "flagged"],
           xlabel="metabolomics 95 % flag", ylabel="lipidomics 95 % flag", title=f"{statistic}: do the assays agree?")
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="samples")
    expected = 3 * float(table["expected_both_under_independence"].iloc[0])
    return finish(fig, view, f"Flagged by both if independent: about {expected:.2f} samples. "
                             "Both assays share one tissue extract, so agreement is not independent evidence.")


# ---------------------------------------------------------------- calibrated ensemble (all 60)
@dataclass
class EnsembleResults:
    scores: pd.DataFrame      # one row per sample x detector setting
    stable: pd.DataFrame      # share of each detector's settings that flag each sample
    verdict: pd.DataFrame     # per sample and scope: detector counts and candidate status


def ensemble_results(settings: Settings, omics: str = "lipidomics", rerun: bool = False) -> EnsembleResults:
    """Read the pipeline's ensemble tables, running the screen + ensemble first if they are missing.

    The ensemble always uses all 60 samples and the default preprocessing
    (about 2 minutes with ``Settings.fast``).
    """
    folder = settings.results_dir / omics / "02_outliers"
    files = [folder / "ensemble" / name for name in ("detector_scores_long.csv", "stability_fractions.csv", "ensemble_verdict.csv")]
    if rerun or not all(f.exists() for f in files):
        outlier_stage.run_screen(settings, omics)
        outlier_stage.run_ensemble(settings, omics)
    scores, stable, verdict = (pd.read_csv(f) for f in files)
    return EnsembleResults(scores, stable, verdict)


def _scopes() -> list[str]:
    return ["pooled", *(r.lower() for r in config.REGIONS)]


def ensemble_stability(results: EnsembleResults, scope: str = "cortex") -> plt.Figure:
    """Share of each detector's settings that flag each sample (trimmed calibration)."""
    block = results.stable.loc[results.stable["calibration_stage"].eq("trimmed") & results.stable["scope"].eq(scope.lower())]
    labels = results.verdict.drop_duplicates("Sample").set_index("Sample")["Sample number"]
    matrix = block.pivot(index="detector", columns="Sample", values="stability_fraction")
    matrix = matrix.loc[:, sorted(matrix.columns, key=lambda s: labels[s])]
    fig, ax = plt.subplots(figsize=(max(6, 0.3 * matrix.shape[1]), 3))
    image = ax.imshow(matrix, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set(xticks=range(matrix.shape[1]), yticks=range(matrix.shape[0]), yticklabels=matrix.index,
           title=f"{scope}: share of settings flagging each sample")
    ax.set_xticklabels([labels[s] for s in matrix.columns], fontsize=7, rotation=90)
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="stability")
    return finish(fig, None, "All 60 samples, default preprocessing. A detector 'stably' flags a sample at >= 60 %.")


def ensemble_corroboration(results: EnsembleResults) -> plt.Figure:
    """How many detectors stably flag each sample, and how many of them are not covariance-based."""
    verdict = results.verdict
    max_x = max(2, int(verdict["stable_detector_count"].max()))
    max_y = max(1, int(verdict["stable_non_covariance_count"].max()))
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for ax, scope in zip(axes.flat, _scopes()):
        rows = verdict.loc[verdict["scope"].eq(scope)]
        ax.add_patch(Rectangle((1.5, 0.5), max_x - 1, max_y, facecolor="#D55E00", alpha=0.12, zorder=0))
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, (len(rows), 2))
        ax.scatter(rows["stable_detector_count"] + jitter[:, 0], rows["stable_non_covariance_count"] + jitter[:, 1], s=26, color="#0072B2")
        for (x, y), group in rows.loc[rows["stable_detector_count"].gt(0)].groupby(["stable_detector_count", "stable_non_covariance_count"]):
            ax.annotate(", ".join(group["Sample number"].astype(str)), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.set(xlim=(-0.5, max_x + 0.5), ylim=(-0.5, max_y + 0.5), xticks=range(max_x + 1), yticks=range(max_y + 1),
               xlabel="detectors stably flagging", ylabel="... of which not covariance-based", title=f"{scope} (n={len(rows)})")
    fig.suptitle("Detector corroboration (shaded: candidate rule, >= 2 detectors incl. >= 1 non-covariance)", fontweight="bold")
    return finish(fig, None, "All 60 samples, default preprocessing. Labels are sample numbers.")


def ensemble_percentiles(results: EnsembleResults) -> plt.Figure:
    """Each sample's null percentile per detector (median over that detector's settings)."""
    scores = results.scores.loc[results.scores["calibration_stage"].eq("trimmed")]
    medians = scores.groupby(["scope", "detector", "Sample"])["null_percentile"].median().reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, scope in zip(axes.flat, _scopes()):
        block = medians.loc[medians["scope"].eq(scope)].pivot(index="Sample", columns="detector", values="null_percentile")
        detectors = [d for d in DETECTOR_ORDER if d in block] + sorted(set(block.columns) - set(DETECTOR_ORDER))
        for sample, row in block.iterrows():
            present = row.reindex(detectors).dropna()
            ax.plot(present.to_numpy(), [detectors.index(d) for d in present.index], color="#BDBDBD", linewidth=0.5, alpha=0.4)
        for i, detector in enumerate(detectors):
            if detector in block:
                ax.scatter(block[detector], np.full(len(block), i), s=18, color=plotting.COLORBLIND_COLORS[i % 8], zorder=2)
        ax.axvline(0.95, color="#D55E00", linestyle="--", linewidth=1)
        ax.set(xlim=(0, 1.01), yticks=range(len(detectors)), yticklabels=detectors, xlabel="null percentile", title=f"{scope} (n={len(block)})")
    fig.suptitle("Where each sample sits in each detector's null distribution", fontweight="bold")
    return finish(fig, None, "All 60 samples, default preprocessing. Grey lines connect one sample across detectors.")
