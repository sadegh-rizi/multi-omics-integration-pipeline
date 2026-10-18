"""Stage 3 - MIA vs control, feature by feature, within each brain region.

For every feature and region: ``log2 value ~ group * sex`` (OLS), giving the
MIA effect, the sex effect and their interaction, with Benjamini-Hochberg FDR
over the whole panel.  Around that core model the stage writes everything
needed to judge whether a "hit" is believable:

* label-permutation null -> empirical p-values and the *effective* number of
  independent tests (much smaller than the panel size: features are correlated);
* residual and leave-one-sample-out influence diagnostics;
* minimum detectable effect at 80 % power (sample level and litter level);
* multivariate comparability checks (PERMANOVA, PLS-DA permutation test);
* lipid class and structural-index summaries (lipidomics only);
* optional limma (R) on the same inputs.

:func:`compare_populations` then puts all60 / t2_95 / t2_99 side by side -
the analysis that showed sample exclusion manufactures "significant" hits by
shrinking the standard error, not by changing the effect.

Output: ``results/<omics>/03_differential/<population>/``.
"""

from __future__ import annotations

import shutil
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config, plotting
from ..config import Settings
from ..imputation import censoring_mask, impute_rf_all, impute_rf_mar_only
from ..lipids import aggregate_class_profiles, structural_index_profiles
from ..multivariate.pca import fit_pca, scree_table
from ..multivariate.plsda import permutation_test_plsda
from ..preprocessing import below_detection_reasons, drop_zero_variance, filter_low_iqr, pareto_scale, scale
from ..stats import diagnostics
from ..stats.differential import fit_regionwise_differential
from ..stats.permanova import permanova_by_region
from ..stats.permutation import run_permutation_null
from ..stats.power import detectable_effect_sizes, fdr_hit_table
from .common import feature_label, load, prepare, region_samples, write_csv, write_json

MIA_EFFECT = "MIA vs Saline Control"


def run(settings: Settings, omics: str = "lipidomics", population: str = config.PRIMARY_POPULATION, *,
        imputation: str | None = None, scaling: str = "none", iqr_filter: bool = False,
        normalization: str = "median", name: str | None = None, light: bool = False) -> pd.DataFrame:
    """Fit the region-wise model for one omics / population; return the effect table.

    The keyword options create the sensitivity variants: RF ``imputation``
    ('rf_mar_only' or 'rf_all'), ``scaling='pareto'``, a label-free
    ``iqr_filter`` and ``normalization='pqn'``.  ``light=True`` skips the
    slow diagnostics (used inside the specification curve).
    """
    plotting.apply_plot_style()
    name = name or population
    out = settings.out(omics, "03_differential", name)
    # 1. Data of this population (the preprocessing itself is in preprocessing.preprocess).
    prepared = prepare(settings, omics, population, normalization=normalization)
    values, meta, features = prepared.values, prepared.metadata, prepared.features

    # 2. Optional sensitivity variants: imputation, a label-free variance filter, Pareto scaling.
    if imputation is not None:
        values, routes = impute_by_region(settings, omics, values, meta, imputation)
        write_csv(routes.stack().value_counts().rename_axis("route").rename("n_cells").reset_index(), out / "imputation_routes.csv")
    if iqr_filter:
        values, ranking = filter_low_iqr(values)
        write_csv(ranking, out / "iqr_filter.csv")
    if scaling == "pareto":
        values = pareto_scale(drop_zero_variance(values))
    effect_scale = "pareto_scaled_log2" if scaling == "pareto" else "normalized_log2"

    # 3. The model: log2 value ~ group * sex per feature and region, BH-FDR over the whole panel.
    fitted = fit_regionwise_differential(values, meta, annotations=features, min_per_cell=config.MIN_PER_CELL,
                                         fdr_denominator="all_features", effect_scale=effect_scale)
    effects = fitted.effects
    write_csv(effects, out / "differential_effects.csv")
    write_csv(fitted.feature_qc, out / "differential_feature_qc.csv")
    write_csv(values.rename_axis("Feature").reset_index(), out / "input_matrix.csv")
    hits, _ = fdr_hit_table(effects, top_n=config.TOP_FDR_HITS)
    write_csv(hits, out / "fdr_hits.csv")
    summary = hit_summary(effects).assign(population=population, variant=name)
    write_csv(summary, out / "hit_summary.csv")
    if light:
        return effects

    # 4. Everything needed to read the hits honestly: detectable effect sizes and volcano plots,
    #    the permutation null and diagnostics, multivariate checks, and lipid-class summaries.
    _power_and_volcanoes(effects, features, out)
    _null_and_diagnostics(settings, omics, values, meta, fitted, prepared.detection, out)
    _multivariate_checks(settings, values, meta, out)
    if omics == "lipidomics":
        _lipid_summaries(values, meta, features, out)
    if settings.use_r:
        run_limma(settings, values, meta, out, effect_scale)
    write_json({"omics": omics, "population": population, "variant": name, "n_samples": values.shape[1],
                "n_features": values.shape[0], "imputation": imputation or "none", "scaling": scaling,
                "iqr_filter": iqr_filter, "normalization": normalization,
                "n_permutations": settings.n_permutations, "model": "log2 value ~ group * sex, per region (OLS)"},
               out / "run_manifest.json")
    return effects


def hit_summary(effects: pd.DataFrame) -> pd.DataFrame:
    """Per region and effect: features tested, FDR hits, smallest q, small-p excess."""
    rows = []
    for (region, effect), rows_ in effects.groupby(["Region", "Effect"], sort=False):
        rows.append({"Region": region, "Effect": effect, "n_tested": len(rows_),
                     "n_fdr_hits": int(rows_["p_value_fdr"].le(config.FDR_ALPHA).sum()),
                     "min_q": rows_["p_value_fdr"].min(),
                     "n_p_below_0.01": int(rows_["p_value"].lt(0.01).sum()),
                     "expected_p_below_0.01": 0.01 * len(rows_)})
    return pd.DataFrame(rows)


def impute_by_region(settings: Settings, omics: str, values: pd.DataFrame, meta: pd.DataFrame,
                     method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label-free RF imputation, run separately inside each region (sensitivity only)."""
    data = load(settings, omics)
    raw = data.values.loc[values.index, values.columns]
    mask = censoring_mask(raw, below_detection_reasons(raw, data.blanks.loc[values.index]))
    imputer = {"rf_mar_only": impute_rf_mar_only, "rf_all": impute_rf_all}[method]
    completed, routes = values.copy(), pd.DataFrame("none", index=values.index, columns=values.columns)
    for region in config.REGIONS:
        samples = region_samples(meta, region)
        completed.loc[:, samples], routes.loc[:, samples] = imputer(values.loc[:, samples], mask.loc[:, samples],
                                                                    random_state=settings.seed)
    return completed, routes


# ---------------------------------------------------------------- power, volcanoes
def _power_and_volcanoes(effects: pd.DataFrame, features: pd.DataFrame, out) -> None:
    """Minimum detectable effects, and a volcano grid (rows = effects, columns = regions)."""
    mde, mde_summary = detectable_effect_sizes(effects, alpha=config.POWER_ALPHA, target_power=config.POWER_TARGET,
                                               illustrative_litter_n=config.ILLUSTRATIVE_LITTER_N)
    write_csv(mde, out / "detectable_effect_sizes.csv")
    write_csv(mde_summary, out / "detectable_effect_size_summary.csv")
    labelled = effects.assign(Label=effects["Feature"].map(feature_label(features)))
    regions = list(dict.fromkeys(effects["Region"]))
    effect_names = list(dict.fromkeys(effects["Effect"]))
    fig, axes = plt.subplots(len(effect_names), len(regions), figsize=(4 * len(regions), 3.4 * len(effect_names)), squeeze=False)
    for i, effect in enumerate(effect_names):
        for j, region in enumerate(regions):
            block = labelled.loc[labelled["Effect"].eq(effect) & labelled["Region"].eq(region)]
            ax = axes[i, j]
            hit = block["p_value_fdr"].le(config.FDR_ALPHA)
            ax.scatter(block["estimate_log2"], -np.log10(block["p_value"]), s=8, color="#999999", alpha=0.6)
            ax.scatter(block.loc[hit, "estimate_log2"], -np.log10(block.loc[hit, "p_value"]), s=12, color=plotting.COLORBLIND_COLORS[1])
            for _, row in block.loc[hit].nsmallest(3, "p_value").iterrows():
                ax.annotate(row["Label"], (row["estimate_log2"], -np.log10(row["p_value"])), fontsize=6)
            ax.set_title(f"{region}\n{effect}", fontsize=9)
            ax.set_xlabel("log2 effect")
            ax.set_ylabel("-log10 p")
    fig.tight_layout()
    plotting.save_figure(fig, out / "figures" / "volcano_grid")


# ---------------------------------------------------------------- permutation null, diagnostics
def _null_and_diagnostics(settings, omics, values, meta, fitted, detection, out) -> None:
    """Sample-level permutation null with effective tests, then residual, influence and QC cross-checks."""
    features, null_summary, effective = run_permutation_null(
        values, meta, n_permutations=settings.n_permutations, random_state=settings.seed,
        min_per_cell=config.MIN_PER_CELL, exchangeability_unit="sample")
    effective.insert(1, "n_features_in_panel", len(values))
    write_csv(features, out / "permutation_feature_statistics.csv")
    write_csv(null_summary, out / "permutation_null_distribution.csv")
    write_csv(effective, out / "effective_number_of_tests.csv")

    residuals, selected = diagnostics.residual_diagnostics(values, meta, count=config.DIAGNOSTIC_FEATURE_COUNT,
                                                           min_per_cell=config.MIN_PER_CELL)
    write_csv(residuals, out / "diagnostics" / "residual_diagnostics.csv")
    write_csv(selected, out / "diagnostics" / "diagnostic_features.csv")
    write_csv(diagnostics.sample_influence_summary(values, meta, min_per_cell=config.MIN_PER_CELL),
              out / "diagnostics" / "sample_influence_leave_one_out.csv")
    variance_path = settings.results_dir / omics / "01_qc" / "technical_vs_biological_variance.csv"
    variance = pd.read_csv(variance_path) if variance_path.exists() else None
    write_csv(diagnostics.cross_check_qc_tables(variance, fitted.effects), out / "diagnostics" / "qc_crosscheck.csv")
    detection = detection.merge(meta[["Sample", config.REGION_COL]], on="Sample") if len(detection) else detection
    write_csv(diagnostics.missingness_se_crosscheck(fitted.feature_qc, fitted.effects, detection),
              out / "diagnostics" / "missingness_se_crosscheck.csv")
    write_csv(diagnostics.summarize_p_values(fitted.effects, source="observed"), out / "p_value_calibration.csv")


def _multivariate_checks(settings: Settings, values: pd.DataFrame, meta: pd.DataFrame, out) -> None:
    """PERMANOVA and PLS-DA on the same samples - is there *any* multivariate MIA signal?"""
    scaled = scale(drop_zero_variance(values), "autoscaled")
    permanova, permanova_diagnostics = permanova_by_region(scaled, meta, n_permutations=settings.n_permutations,
                                                           random_state=settings.seed)
    write_csv(permanova.drop(columns=["p_value_fdr"], errors="ignore"), out / "multivariate" / "permanova_by_region.csv")
    write_csv(permanova_diagnostics, out / "multivariate" / "permanova_diagnostics.csv")
    scree, plsda = [], []
    for region in config.REGIONS:
        samples = region_samples(meta, region)
        complete = scaled.loc[:, samples].dropna()
        model, _, _ = fit_pca(complete)
        scree.append(scree_table(model).assign(Region=region))
        labels = meta.set_index("Sample").loc[samples, config.GROUP_COL]
        sexes = meta.set_index("Sample").loc[samples, config.SEX_COL]
        result, _ = permutation_test_plsda(complete, labels, sexes, n_permutations=settings.n_permutations,
                                           random_state=settings.seed)
        plsda.append({"Region": region, "n_samples": len(samples), "n_features": len(complete), **result.to_dict()})
    write_csv(pd.concat(scree, ignore_index=True), out / "multivariate" / "pca_scree_by_region.csv")
    write_csv(pd.DataFrame(plsda), out / "multivariate" / "plsda_permutation_test.csv")


def _lipid_summaries(values: pd.DataFrame, meta: pd.DataFrame, features: pd.DataFrame, out) -> None:
    """Test lipid classes (mean / median of species) and structural indices instead of species."""
    annotations = features[["Lipid Name", "Lipid Class"]]
    tables = []
    for statistic in ("mean", "median"):
        profiles, profile_annotations = aggregate_class_profiles(values, annotations, statistic=statistic)
        fitted = fit_regionwise_differential(profiles, meta, annotations=profile_annotations,
                                             min_per_cell=config.MIN_PER_CELL, fdr_denominator="all_features")
        tables.append(fitted.effects.assign(summary=f"class_{statistic}"))
    structural, structural_annotations, _ = structural_index_profiles(values, annotations)
    fitted = fit_regionwise_differential(structural, meta, annotations=structural_annotations,
                                         min_per_cell=config.MIN_PER_CELL, fdr_denominator="all_features")
    tables.append(fitted.effects.assign(summary="structural_index"))
    write_csv(pd.concat(tables, ignore_index=True), out / "lipid_class_and_structure_effects.csv")


# ---------------------------------------------------------------- optional limma (R)
def run_limma(settings: Settings, values: pd.DataFrame, meta: pd.DataFrame, out, effect_scale: str) -> None:
    """Same model with limma's moderated variances (needs R + limma; see r/README.md)."""
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Rscript not found: install R and limma, or run with settings.use_r=False.")
    inputs = out / "limma_inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    values.rename_axis("Feature").to_csv(inputs / "matrix.csv")
    meta.rename(columns={config.REGION_COL: "Region", config.GROUP_COL: "BioGroup"})[
        ["Sample", "Region", "BioGroup", config.SEX_COL, config.MOUSE_COL]].to_csv(inputs / "metadata.csv", index=False)
    subprocess.run([rscript, str(settings.project_root / "r" / "differential_limma.R"), str(inputs / "matrix.csv"),
                    str(inputs / "metadata.csv"), str(out / "differential_effects_limma.csv"), "all_features",
                    effect_scale], check=True)


# ---------------------------------------------------------------- populations side by side
def compare_populations(settings: Settings, omics: str = "lipidomics") -> pd.DataFrame:
    """How do hits, effects and standard errors change when outlier samples are removed?"""
    plotting.apply_plot_style()
    out = settings.out(omics, "03_differential", "population_comparison")
    effects = {p: pd.read_csv(settings.results_dir / omics / "03_differential" / p / "differential_effects.csv")
               for p in config.POPULATIONS}
    summary = pd.concat([hit_summary(e).assign(population=p) for p, e in effects.items()], ignore_index=True)
    write_csv(summary, out / "hits_by_population.csv")
    calibration = pd.concat([diagnostics.summarize_p_values(e, source=p) for p, e in effects.items()], ignore_index=True)
    write_csv(calibration, out / "p_value_calibration_by_population.csv")

    # Same feature, same effect: does the estimate move, or only its standard error?
    keys = ["Feature", "Region", "Effect"]
    base = effects["all60"][keys + ["estimate_log2", "standard_error", "p_value_fdr"]]
    paired = []
    for population in [p for p in config.POPULATIONS if p != "all60"]:
        other = effects[population][keys + ["estimate_log2", "standard_error", "p_value_fdr"]]
        joined = base.merge(other, on=keys, suffixes=("_all60", "_excluded"))
        joined["population"] = population
        joined["se_ratio"] = joined["standard_error_excluded"] / joined["standard_error_all60"]
        joined["estimate_change"] = joined["estimate_log2_excluded"] - joined["estimate_log2_all60"]
        joined["hit_only_after_exclusion"] = joined["p_value_fdr_excluded"].le(config.FDR_ALPHA) & joined["p_value_fdr_all60"].gt(config.FDR_ALPHA)
        paired.append(joined)
    paired = pd.concat(paired, ignore_index=True)
    write_csv(paired, out / "estimate_vs_standard_error_change.csv")
    write_csv(paired.loc[paired["hit_only_after_exclusion"]], out / "exclusion_dependent_hits.csv")

    fig, axes = plt.subplots(1, len(effects), figsize=(4 * len(effects), 3.4), sharey=True)
    for ax, (population, table) in zip(axes, effects.items()):
        ax.hist(table.loc[table["Effect"].eq(MIA_EFFECT), "p_value"].dropna(), bins=20, color=plotting.COLORBLIND_COLORS[0])
        ax.axhline(table["Effect"].eq(MIA_EFFECT).sum() / 20, color="#444444", linestyle="--", linewidth=0.8)
        ax.set(title=f"{population}: MIA p-values", xlabel="p-value")
    axes[0].set_ylabel("features x regions")
    plotting.save_figure(fig, out / "figures" / "mia_p_value_histograms")
    return summary
