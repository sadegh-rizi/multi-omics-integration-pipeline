"""Stage 1 - data quality: preprocessing log, pooled-QC reproducibility,
technical vs biological variance, missingness, and descriptive PCA.

Output: ``results/<omics>/01_qc/``.  Nothing here changes the data used by
later stages; it documents what preprocessing did and what structure the
data contain (region dominates, MIA does not separate).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config, plotting, qc
from ..config import Settings
from ..multivariate import pca
from ..preprocessing import (
    below_detection_reasons,
    blank_dominated,
    exclude_nonpositive,
    for_multivariate,
    log2_transform,
    missing_fraction,
)
from .common import feature_class, feature_label, load, prepare, region_samples, write_csv, write_json

PCA_COLOR_BY = (config.REGION_COL, config.GROUP_COL, config.SEX_COL, config.CAGE_COL,
                config.COHORT_COL, "Sample number", "Tissue mass (mg)")


def run(settings: Settings, omics: str = "lipidomics") -> dict:
    """Write all QC tables and figures for one omics; return a few headline numbers."""
    plotting.apply_plot_style()
    out = settings.out(omics, "01_qc")
    data = load(settings, omics)
    prepared = prepare(settings, omics)
    classes = feature_class(data.features)

    write_csv(prepared.summary, out / "preprocessing_summary.csv")
    write_csv(prepared.exclusions.merge(data.features, left_on="Feature", right_index=True, how="left"),
              out / "feature_exclusions.csv")
    write_csv(prepared.detection, out / "detection_limit_values.csv")

    headline = {"omics": omics, **prepared.summary.iloc[0].to_dict()}
    if data.blanks is not None:
        headline |= _pooled_qc(data, out)
        _missingness(data, classes, out)
    headline |= _pca_overview(prepared.values, prepared.metadata, data.features, classes, out)
    write_json(headline, out / "headline.json")
    return headline


# ---------------------------------------------------------------- pooled QC and blanks
def _pooled_qc(data, out) -> dict:
    raw, blanks, techmix = data.values, data.blanks, data.techmix

    # Signal-to-blank ratio for every feature (the blank filter's evidence).
    mean_blank = blanks.fillna(0).mean(axis=1)
    median_signal = exclude_nonpositive(raw).median(axis=1)
    blank_table = pd.DataFrame({
        "median_sample_signal": median_signal,
        "mean_blank_signal": mean_blank,
        "signal_to_blank_ratio": median_signal / mean_blank.where(mean_blank > 0),
        "blank_dominated": raw.index.isin(blank_dominated(raw, blanks)),
    }).join(data.features)
    write_csv(blank_table.rename_axis("Feature").reset_index(), out / "signal_to_blank.csv")

    # Tech-Mix (pooled QC) CV and injection drift.
    cv = qc.qc_injection_cv(techmix).join(data.features)
    cv["would_fail_cv_filter"] = cv["normalized_cv_percent"].gt(config.QC_CV_THRESHOLD)
    write_csv(cv.rename_axis("Feature").reset_index(), out / "techmix_cv.csv")
    drift = pd.DataFrame({"injection": techmix.columns, "order": range(1, techmix.shape[1] + 1),
                          "median_positive_signal": techmix.where(techmix > 0).median().to_numpy()})
    write_csv(drift, out / "techmix_drift.csv")
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(drift["order"], drift["median_positive_signal"], "o-", color=plotting.COLORBLIND_COLORS[0])
    ax.set(xlabel="Tech-Mix injection", ylabel="Median peak area", title=f"{data.name}: pooled-QC drift")
    plotting.save_figure(fig, out / "figures" / "techmix_drift")

    # Technical (Tech-Mix) vs biological (per region) variance on log2 detectable values.
    reasons = below_detection_reasons(raw, blanks)
    log_samples = log2_transform(exclude_nonpositive(raw).mask(reasons.notna()))
    log_techmix = log2_transform(exclude_nonpositive(techmix))
    ratios = qc.variance_ratio_table(log_techmix, log_samples, data.metadata).merge(
        data.features, left_on="Feature", right_index=True, how="left")
    write_csv(ratios, out / "technical_vs_biological_variance.csv")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(cv["normalized_cv_percent"].dropna(), bins=30, color=plotting.COLORBLIND_COLORS[0])
    ax.axvline(config.QC_CV_THRESHOLD, color=plotting.COLORBLIND_COLORS[1], linestyle="--")
    ax.set(xlabel="Tech-Mix CV (%)", ylabel="Features", title=f"{data.name}: pooled-QC reproducibility")
    plotting.save_figure(fig, out / "figures" / "techmix_cv_histogram")
    return {
        "median_techmix_cv_percent": float(cv["normalized_cv_percent"].median()),
        "n_features_cv_above_threshold": int(cv["would_fail_cv_filter"].sum()),
        "share_technical_dominated": float(ratios["technical_dominated"].mean()),
    }


# ---------------------------------------------------------------- missingness
def _missingness(data, classes, out) -> None:
    """Where are values missing, and does missingness differ between groups?"""
    out = out / "missingness"
    raw = exclude_nonpositive(data.values)
    meta = data.metadata
    by_feature = qc.missingness_by_feature(raw, data.features)
    by_sample = qc.missingness_by_sample(raw, meta)
    write_csv(by_feature, out / "missingness_by_feature.csv")
    write_csv(by_sample, out / "missingness_by_sample.csv")
    write_csv(pd.DataFrame([{
        "pct_cells_missing": 100 * raw.isna().to_numpy().mean(),
        "pct_features_complete": 100 * by_feature["n_missing"].eq(0).mean(),
        "pct_features_above_prevalence_limit": 100 * missing_fraction(data.values).gt(config.MAX_MISSING_FRACTION).mean(),
    }]), out / "missingness_summary.csv")

    tests = []
    for column, first, second in ((config.GROUP_COL, config.MIA, config.CONTROL), (config.SEX_COL, "Female", "Male")):
        write_csv(qc.missingness_by_group(by_sample, column), out / f"missingness_by_{column.lower().replace(' ', '_')}.csv")
        tests.append(qc.test_missingness_by_group(by_sample, column, first, second))
    write_csv(qc.missingness_by_group(by_sample, config.REGION_COL), out / "missingness_by_region.csv")
    write_csv(pd.DataFrame(tests), out / "missingness_group_tests.csv")
    write_csv(qc.missingness_by_class(raw, classes, meta), out / "missingness_by_class_fisher.csv")

    # How missingness changes along the preprocessing chain.
    after_prevalence = raw.loc[missing_fraction(data.values).le(config.MAX_MISSING_FRACTION)]
    blanks = data.blanks.loc[after_prevalence.index]
    after_blank = after_prevalence.drop(index=blank_dominated(after_prevalence, blanks))
    after_detection = after_blank.mask(below_detection_reasons(after_blank, blanks.loc[after_blank.index]).notna())
    write_csv(qc.filtering_stage_missingness(
        [("raw", raw), ("prevalence", after_prevalence), ("blank_filter", after_blank),
         ("detection_limit", after_detection)], meta), out / "missingness_by_stage.csv")

    with_missing = by_feature.loc[by_feature["n_missing"].gt(0), "Feature"]
    if len(with_missing):
        order_features = classes.loc[with_missing].sort_values().index
        for name, sort_by in (("by_group", [config.GROUP_COL, config.SEX_COL, config.REGION_COL]),
                              ("by_region", [config.REGION_COL, config.GROUP_COL])):
            samples = meta.sort_values(sort_by)["Sample"]
            fig = plotting.presence_heatmap(raw.loc[order_features, samples].notna(), meta, sort_by[:2],
                                            title=f"{data.name}: observed values (features with any missing)")
            plotting.save_figure(fig, out / "figures" / f"missingness_heatmap_{name}")
    fig, ax = plt.subplots(figsize=(6, 4))
    for group, color in plotting.BIO_GROUP_COLORS.items():
        ax.hist(by_sample.loc[by_sample[config.GROUP_COL].eq(group), "pct_missing"], bins=15, alpha=0.6, color=color, label=group)
    ax.set(xlabel="Missing values per sample (%)", ylabel="Samples", title=f"{data.name}: missingness per sample")
    ax.legend(frameon=False)
    plotting.save_figure(fig, out / "figures" / "missingness_per_sample")


# ---------------------------------------------------------------- descriptive PCA
def _pca_overview(values: pd.DataFrame, metadata: pd.DataFrame, features: pd.DataFrame,
                  classes: pd.Series, out) -> dict:
    """Pooled and regional PCA: scree, scores coloured by design factors, loadings, attribution."""
    out = out / "pca"
    labels = feature_label(features)
    scopes = [("pooled", metadata["Sample"].tolist())] + [(r.lower(), region_samples(metadata, r)) for r in config.REGIONS]
    attribution, concentration, composition, headline = [], [], [], {}
    for scope, samples in scopes:
        matrix = for_multivariate(values.loc[:, samples])
        model, scores, loadings = pca.fit_pca(matrix)
        loadings = pca.align_loading_signs(loadings)
        write_csv(pca.scree_table(model), out / f"scree_{scope}.csv")
        write_csv(scores.rename_axis("Sample").reset_index(), out / f"scores_{scope}.csv")
        write_csv(pca.loadings_table(loadings, features), out / f"top_loadings_{scope}.csv")
        attribution.append(pca.score_attribution(scores, model.explained_variance_ratio_, metadata, scope))
        concentration.append(pca.loading_concentration(loadings).assign(scope=scope))
        composition.append(pca.class_composition(loadings.iloc[:, :4], classes.loc[loadings.index]).assign(scope=scope))

        plotting.save_figure(plotting.scree_plot(model.explained_variance_ratio_[:10], f"PCA scree: {scope}"),
                             out / "figures" / f"scree_{scope}")
        for column in PCA_COLOR_BY if scope == "pooled" else (config.GROUP_COL, config.SEX_COL, config.CAGE_COL):
            fig = plotting.scores_plot(scores, metadata, column, variance=model.explained_variance_ratio_,
                                       title=f"{scope}: PCA scores by {column}",
                                       ellipses=column in (config.REGION_COL, config.GROUP_COL, config.SEX_COL))
            plotting.save_figure(fig, out / "figures" / f"scores_{scope}_{column.lower().replace(' ', '_').replace('(', '').replace(')', '')}")
        for component in ("PC 1", "PC 2"):
            top = loadings[component].abs().nlargest(20).index
            fig = plotting.lollipop_plot(loadings.loc[top, component], labels, title=f"{scope}: strongest {component} loadings",
                                         xlabel=f"{component} loading")
            plotting.save_figure(fig, out / "figures" / f"loadings_{scope}_{component.replace(' ', '').lower()}")
        headline[f"pc1_variance_{scope}"] = float(model.explained_variance_ratio_[0])

    attribution = pd.concat(attribution, ignore_index=True)
    write_csv(attribution, out / "score_attribution.csv")
    write_csv(pd.concat(concentration, ignore_index=True), out / "loading_concentration.csv")
    write_csv(pd.concat(composition, ignore_index=True), out / "loading_class_composition.csv")

    # Bar chart: how much of PC1..PC5 is associated with each design factor (pooled).
    pooled = attribution.loc[attribution["scope"].eq("pooled") & attribution["Factor"].ne("Total")]
    table = pooled.pivot(index="Component", columns="Factor", values="factor_score_r2")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    table.plot.bar(ax=ax, color=plotting.COLORBLIND_COLORS[:table.shape[1]], width=0.8)
    ax.set(ylabel="R² of PC scores", ylim=(0, 1), title="Pooled PCA: what explains each component?")
    ax.legend(title="Factor", bbox_to_anchor=(1.01, 1), loc="upper left")
    plotting.save_figure(fig, out / "figures" / "score_attribution_pooled")
    headline["pc1_region_r2"] = float(table.loc["PC 1", "Region"]) if "Region" in table else np.nan
    headline["max_group_r2_pc1_5"] = float(table["Bio group"].max())
    return headline
