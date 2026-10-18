"""Preprocessing, pooled-QC, missingness and noise-floor figures.

The prevalence and blank filters are computed on **all 60** raw samples (so
every population sees the same feature panel); sample exclusions affect
everything downstream of them.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config, plotting, qc
from ..noise_floor import DEFAULT_THRESHOLD_RULES, abundance_similarity_curve, candidate_thresholds
from ..preprocessing import blank_dominated, exclude_nonpositive, log2_transform, missing_fraction
from ..stages.noise_floor import CURVE_OPTIONS, WINDOW_SIZE, WINDOW_STEP
from ._common import finish, require_blanks
from .view import View


# ---------------------------------------------------------------- preprocessing
def feature_funnel(view: View) -> plt.Figure:
    """How many features survive each preprocessing step with the current thresholds."""
    require_blanks(view, "The preprocessing funnel")
    s = view.prepared.summary.iloc[0]
    after_blank = s["n_after_prevalence"] - s["n_blank_dominated"]
    steps = {"raw\npanel": s["n_features_input"], "after\nprevalence": s["n_after_prevalence"],
             "after blank\nfilter": after_blank, "after detection\nlimit": after_blank - s["n_no_detectable_value"],
             "final": s["n_features_output"]}
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.bar(list(steps), list(steps.values()), color=plotting.COLORBLIND_COLORS[0])
    ax.bar_label(bars, fmt="%d")
    ax.set(ylabel="features", title=f"{view.omics}: features kept after each step")
    return finish(fig, view, f"{int(s['n_values_below_detection'])} single values set to missing below the detection limit "
                             f"(floor {s['detection_floor']:.3g}).")


def prevalence_curve(view: View) -> plt.Figure:
    """Features kept as a function of the prevalence threshold (current threshold marked)."""
    require_blanks(view, "The prevalence curve")
    fraction = missing_fraction(view.raw.values)
    dominated = view.raw.values.index.isin(blank_dominated(view.raw.values, view.raw.blanks, view.options.blank_multiplier))
    grid = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid * 100, [(fraction <= t).sum() for t in grid], color=plotting.COLORBLIND_COLORS[0], label="prevalence filter only")
    ax.plot(grid * 100, [((fraction <= t) & ~dominated).sum() for t in grid], color=plotting.COLORBLIND_COLORS[1],
            label=f"+ blank filter (x{view.options.blank_multiplier:g})")
    if view.options.max_missing_fraction is not None:
        ax.axvline(view.options.max_missing_fraction * 100, color="#444444", linestyle="--", label="current threshold")
    ax.set(xlabel="maximum share of samples missing (%)", ylabel="features kept", title=f"{view.omics}: choosing the prevalence threshold")
    ax.legend(frameon=False)
    return finish(fig, view, "Missingness is counted over all 60 raw samples.")


def signal_to_blank(view: View) -> plt.Figure:
    """Distribution of median sample signal / mean blank, with the blank-filter cut."""
    require_blanks(view, "The signal-to-blank plot")
    blank = view.raw.blanks.fillna(0).mean(axis=1)
    median = exclude_nonpositive(view.raw.values).median(axis=1)
    ratio = np.log2(median / blank.where(blank > 0))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ratio.dropna(), bins=40, color=plotting.COLORBLIND_COLORS[0])
    ax.axvline(np.log2(view.options.blank_multiplier), color=plotting.COLORBLIND_COLORS[1], linestyle="--",
               label=f"cut: {view.options.blank_multiplier:g} x blank")
    n_removed = int((ratio < np.log2(view.options.blank_multiplier)).sum())
    ax.set(xlabel="log2(median sample signal / mean blank)", ylabel="features",
           title=f"{view.omics}: signal above blank ({n_removed} features below the cut)")
    ax.legend(frameon=False)
    return finish(fig, view, f"{int(ratio.isna().sum())} features have no blank signal and are not shown.")


def techmix_cv(view: View) -> plt.Figure:
    """Pooled-QC (Tech-Mix) CV per feature, split into kept and removed features."""
    require_blanks(view, "The Tech-Mix CV")
    cv = qc.qc_injection_cv(view.raw.techmix)["normalized_cv_percent"]
    kept = cv.index.isin(view.values.index)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, max(60, float(cv.max(skipna=True) or 0)), 31)
    ax.hist([cv[kept].dropna(), cv[~kept].dropna()], bins=bins, stacked=True,
            color=[plotting.COLORBLIND_COLORS[0], "#BBBBBB"], label=["kept", "removed by the filters"])
    ax.axvline(config.QC_CV_THRESHOLD, color=plotting.COLORBLIND_COLORS[1], linestyle="--", label=f"{config.QC_CV_THRESHOLD:g} % (reported, not a filter)")
    ax.set(xlabel="Tech-Mix CV (%)", ylabel="features", title=f"{view.omics}: pooled-QC reproducibility")
    ax.legend(frameon=False)
    return finish(fig, view, f"median CV of kept features: {cv[kept].median():.1f} %")


def techmix_drift(view: View) -> plt.Figure:
    """Median peak area of each Tech-Mix injection in injection order."""
    require_blanks(view, "The Tech-Mix drift plot")
    techmix = view.raw.techmix
    median = techmix.where(techmix > 0).median()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(range(1, len(median) + 1), median.to_numpy(), "o-", color=plotting.COLORBLIND_COLORS[0])
    ax.set(xlabel="Tech-Mix injection (order)", ylabel="median peak area", title=f"{view.omics}: pooled-QC signal drift")
    return finish(fig, view)


# ---------------------------------------------------------------- missingness
def missingness_per_feature(view: View) -> plt.Figure:
    """Share of samples missing per feature, with the prevalence threshold."""
    fraction = missing_fraction(view.raw.values) * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(fraction, bins=np.linspace(0, 100, 41), color=plotting.COLORBLIND_COLORS[0])
    if view.options.max_missing_fraction is not None:
        ax.axvline(view.options.max_missing_fraction * 100, color=plotting.COLORBLIND_COLORS[1], linestyle="--", label="prevalence threshold")
        ax.legend(frameon=False)
    ax.set(xlabel="samples missing (%)", ylabel="features", yscale="log",
           title=f"{view.omics}: missingness per feature ({int((fraction == 0).sum())} complete)")
    return finish(fig, view, "Raw panel, all 60 samples (what the prevalence filter sees).")


def _raw_kept(view: View) -> pd.DataFrame:
    return exclude_nonpositive(view.raw.values.loc[:, view.kept_samples])


def missingness_per_sample(view: View) -> plt.Figure:
    """Missing values per kept sample, by treatment group."""
    by_sample = _raw_kept(view).isna().mean() * 100
    meta = view.metadata.set_index("Sample")
    fig, ax = plt.subplots(figsize=(7, 4))
    for group, color in plotting.BIO_GROUP_COLORS.items():
        ax.hist(by_sample[meta[config.GROUP_COL].eq(group)], bins=15, alpha=0.6, color=color, label=group)
    ax.set(xlabel="features missing in the sample (%)", ylabel="samples", title=f"{view.omics}: missingness per sample")
    ax.legend(frameon=False)
    return finish(fig, view, "Raw panel, kept samples.")


def missingness_by_region_and_group(view: View) -> plt.Figure:
    """Mean (+/- SD) share of missing values per region and treatment group."""
    by_sample = (_raw_kept(view).isna().mean() * 100).rename("pct").to_frame().join(view.metadata.set_index("Sample"))
    table = by_sample.groupby([config.REGION_COL, config.GROUP_COL], observed=True)["pct"].agg(["mean", "std"]).unstack()
    fig, ax = plt.subplots(figsize=(7, 4))
    positions = np.arange(len(table))
    for i, (group, color) in enumerate(plotting.BIO_GROUP_COLORS.items()):
        ax.bar(positions + (i - 0.5) * 0.38, table[("mean", group)], 0.38, yerr=table[("std", group)],
               color=color, label=group, capsize=3)
    ax.set_xticks(positions, table.index)
    ax.set(ylabel="features missing per sample (%)", title=f"{view.omics}: missingness by region and group")
    ax.legend(frameon=False)
    return finish(fig, view)


def missingness_heatmap(view: View, stage: str = "raw", sort_by: tuple[str, ...] = (config.REGION_COL, config.GROUP_COL)) -> plt.Figure:
    """Observed / missing cells for every feature and kept sample.

    ``stage='raw'`` shows the delivered panel; ``stage='filtered'`` shows the
    data after preprocessing (prevalence, blank filter, detection limit).
    """
    if stage not in {"raw", "filtered"}:
        raise ValueError("stage must be 'raw' or 'filtered'.")
    values = _raw_kept(view) if stage == "raw" else view.values
    classes = view.classes.reindex(values.index)
    order = classes.sort_values(kind="stable").index
    samples = view.metadata.sort_values(list(sort_by))["Sample"]
    fig = plotting.presence_heatmap(values.loc[order, samples].notna(), view.metadata, list(sort_by)[:2],
                                    title=f"{view.omics}: observed values ({stage}; features ordered by class)")
    return finish(fig, view)


# ---------------------------------------------------------------- noise floor
def abundance_similarity(view: View, window_size: int = WINDOW_SIZE) -> plt.Figure:
    """noisyR-style curve: how well samples agree as a function of abundance, per region and for Tech-Mix."""
    require_blanks(view, "The abundance-similarity curve")
    samples_log2 = log2_transform(exclude_nonpositive(view.raw.values.loc[:, view.kept_samples]))
    techmix_log2 = log2_transform(exclude_nonpositive(view.raw.techmix))
    size = min(window_size, len(samples_log2) // 3)
    step = max(1, min(WINDOW_STEP, size // 5))
    technical = abundance_similarity_curve(techmix_log2, window_size=size, step=step, **CURVE_OPTIONS)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(technical["median_log2_abundance"], technical["median_pearson"], "--", color="#444444", label="Tech-Mix")
    for region in config.REGIONS:
        curve = abundance_similarity_curve(samples_log2.loc[:, view.samples(region)], window_size=size, step=step, **CURVE_OPTIONS)
        color = plotting.BRAIN_REGION_COLORS[region]
        ax.plot(curve["median_log2_abundance"], curve["median_pearson"], marker="o", markersize=3, color=color, label=region)
        found = candidate_thresholds(curve, technical_curve=technical, rules=DEFAULT_THRESHOLD_RULES)
        for value in found["threshold_log2_abundance"].dropna():
            ax.axvline(value, color=color, alpha=0.3, linewidth=0.8)
    ax.set(xlabel="median log2 abundance of the window", ylabel="median sample-sample Pearson r",
           title=f"{view.omics}: abundance-similarity (window {size})")
    ax.legend(fontsize=8, frameon=False)
    return finish(fig, view, "Vertical lines: candidate noise floors (reported, never applied).")
