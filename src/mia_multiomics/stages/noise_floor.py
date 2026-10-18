"""Stage 1b - is there an abundance below which lipids are just noise?

noisyR-style abundance-similarity curve: features are ordered by abundance
and cut into sliding windows; inside each window we correlate every pair of
samples.  At low abundance (noise) samples agree poorly; as abundance rises
the correlation climbs to a plateau.  The Tech-Mix injections (the same pooled
sample injected six times) give the purely technical reference curve.

Candidate thresholds are reported, never applied.  Variants: window size 50 /
100 / 200 and median-normalised input.  With ``settings.use_r`` the R package
noisyR itself is run on the same input for comparison (``r/noisyr_crosscheck.R``).

Output: ``results/<omics>/01_qc/noise_floor/``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .. import config, plotting, qc
from ..config import Settings
from ..noise_floor import DEFAULT_THRESHOLD_RULES, abundance_similarity_curve, candidate_thresholds
from ..preprocessing import blank_dominated, exclude_nonpositive, log2_transform, median_normalize
from .common import load, region_samples, write_csv

WINDOW_SIZE = 100
WINDOW_STEP = 20
CURVE_OPTIONS = dict(min_pairwise_correlations_per_point=10, min_comparators_per_reference=3, min_pairwise_observations=10)


def run(settings: Settings, omics: str = "lipidomics") -> pd.DataFrame:
    """Abundance-similarity curves per region and for Tech-Mix; candidate noise thresholds."""
    plotting.apply_plot_style()
    out = settings.out(omics, "01_qc", "noise_floor")
    data = load(settings, omics)
    window = min(WINDOW_SIZE, len(data.values) // 3)
    samples_log2 = log2_transform(exclude_nonpositive(data.values))
    techmix_log2 = log2_transform(exclude_nonpositive(data.techmix))
    variants = [("primary", samples_log2, techmix_log2, window),
                ("window_half", samples_log2, techmix_log2, window // 2),
                ("window_double", samples_log2, techmix_log2, min(2 * window, len(data.values) // 2)),
                ("median_normalized", median_normalize(samples_log2), median_normalize(techmix_log2), window)]
    thresholds, curves = [], []
    for name, biological, technical, size in variants:
        step = max(1, min(WINDOW_STEP, size // 5))
        techmix_curve = abundance_similarity_curve(technical, window_size=size, step=step, **CURVE_OPTIONS)
        if name == "primary":
            curves.append(techmix_curve.assign(source="Tech-Mix"))
        for region in config.REGIONS:
            curve = abundance_similarity_curve(biological.loc[:, region_samples(data.metadata, region)],
                                               window_size=size, step=step, **CURVE_OPTIONS)
            found = candidate_thresholds(curve, technical_curve=techmix_curve, rules=DEFAULT_THRESHOLD_RULES)
            thresholds.append(found.assign(Region=region, variant=name, window_size=size))
            if name == "primary":
                curves.append(curve.assign(source=region))
    thresholds = pd.concat(thresholds, ignore_index=True)
    curves = pd.concat(curves, ignore_index=True)
    write_csv(thresholds, out / "candidate_thresholds.csv")
    write_csv(curves, out / "abundance_similarity_curves.csv")
    write_csv(_overlap(data, samples_log2, thresholds.loc[thresholds["variant"].eq("primary")]), out / "overlap_with_blank_and_cv_filters.csv")
    _plot(curves, thresholds.loc[thresholds["variant"].eq("primary")], out / "figures")
    if settings.use_r:
        _noisyr_crosscheck(settings, samples_log2, techmix_log2, data.metadata, out)
    return thresholds


def _overlap(data, samples_log2: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    """Would a noise threshold remove the same lipids as the blank or Tech-Mix CV filters?"""
    blank = data.values.index.isin(blank_dominated(data.values, data.blanks))
    high_cv = qc.qc_injection_cv(data.techmix)["normalized_cv_percent"].gt(config.QC_CV_THRESHOLD).to_numpy()
    rows = []
    for _, row in thresholds.dropna(subset=["threshold_log2_abundance"]).iterrows():
        median = samples_log2.loc[:, region_samples(data.metadata, row["Region"])].median(axis=1)
        below = median.lt(row["threshold_log2_abundance"]).to_numpy()
        rows.append({"Region": row["Region"], "rule": row["threshold_rule"], "threshold_log2": row["threshold_log2_abundance"],
                     "n_below_threshold": int(below.sum()), "n_blank_dominated": int(blank.sum()),
                     "n_both_below_and_blank": int((below & blank).sum()), "n_high_cv": int(high_cv.sum()),
                     "n_both_below_and_high_cv": int((below & high_cv).sum())})
    return pd.DataFrame(rows)


def _plot(curves: pd.DataFrame, thresholds: pd.DataFrame, out) -> None:
    """Curves per region and for Tech-Mix, with the candidate noise floors as vertical lines."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = {**plotting.BRAIN_REGION_COLORS, "Tech-Mix": "#444444"}
    for source, curve in curves.groupby("source"):
        ax.plot(curve["median_log2_abundance"], curve["median_pearson"], marker="o", markersize=3,
                color=colors.get(source, "#999999"), linestyle="--" if source == "Tech-Mix" else "-", label=source)
    for _, row in thresholds.dropna(subset=["threshold_log2_abundance"]).iterrows():
        ax.axvline(row["threshold_log2_abundance"], color=colors.get(row["Region"], "#999999"), alpha=0.3, linewidth=0.8)
    ax.set(xlabel="median log2 abundance of window", ylabel="median sample-sample Pearson r",
           title="Abundance-similarity curve (vertical lines: candidate noise floors)")
    ax.legend(fontsize=8)
    plotting.save_figure(fig, out / "abundance_similarity")


def _noisyr_crosscheck(settings: Settings, samples_log2: pd.DataFrame, techmix_log2: pd.DataFrame,
                       metadata: pd.DataFrame, out) -> None:
    """Run the R package noisyR on the same matrices (optional; needs R + noisyR)."""
    from ..noisyr import panel_scaled_window_parameters, run_noisyr, threshold_median
    work = out / "noisyr_work"
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    scopes = [(r.lower(), samples_log2.loc[:, region_samples(metadata, r)]) for r in config.REGIONS] + [("techmix", techmix_log2)]
    for label, values in scopes:
        window, step = panel_scaled_window_parameters(len(values), 0.10, minimum_features=25)
        curve, found, status = run_noisyr(values, work, label, settings.project_root, window=window, step=step)
        rows.append({"scope": label, "status": status, "window": window, "step": step,
                     "noisyr_threshold_log2": threshold_median(found, status)})
    write_csv(pd.DataFrame(rows), out / "noisyr_crosscheck.csv")
