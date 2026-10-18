"""Stage 4 - does the (null) MIA answer depend on analysis choices?

* :func:`specification_curve` - prevalence filter x Tech-Mix CV filter x
  imputation method (16 specifications), each refitted and summarised;
* :func:`rf_variants` - RF imputation x scaling x population with a label-free
  IQR filter (the "pipeline_test_rf" grid);
* :func:`normalization_comparison` - median vs PQN normalisation, effect by effect;
* :func:`cage_structure` - litter-aware re-analyses: cage means (A), cage
  random intercept (B), collection-date fixed effect (C), the date-matched
  subset (D), whole-cage permutations and cage ICCs;
* :func:`pooled_mixed_model` - all regions in one model per feature with a
  random intercept per mouse (directional check, not a primary test).

Output: ``results/<omics>/04_sensitivity/``.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

from .. import config, plotting, qc
from ..config import Settings
from ..imputation import censoring_mask, impute_mnar_halfmin, impute_none, impute_rf_all, impute_rf_mar_only
from ..multivariate.pca import fit_pca
from ..multivariate.plsda import permutation_test_plsda
from ..preprocessing import below_detection_reasons, for_multivariate, preprocess
from ..stats.clustered import (
    cage_permutation_ceiling,
    collapse_to_cage_means,
    intraclass_correlation,
    sex_exchangeability_summary,
)
from ..stats.common import bh_adjust, r_squared
from ..stats.differential import fit_pooled_mixed_sensitivity, fit_regionwise_differential
from ..stats.permutation import run_permutation_null
from . import differential
from .common import load, prepare, region_samples, write_csv, write_json

IMPUTERS = {"none": impute_none, "mnar_halfmin": impute_mnar_halfmin,
            "rf_mar_only": impute_rf_mar_only, "rf_all": impute_rf_all}


# ---------------------------------------------------------------- specification curve
def specification_curve(settings: Settings, omics: str = "lipidomics") -> pd.DataFrame:
    """Refit the model under every combination of filtering and imputation choices.

    With ``settings.quick`` the slow naive RF arm ('rf_all') and the RF
    recovery-error check are skipped.
    """
    plotting.apply_plot_style()
    out = settings.out(omics, "04_sensitivity", "specification_curve")
    data = load(settings, omics)
    cv = qc.qc_injection_cv(data.techmix)
    low_cv = cv.index[cv["normalized_cv_percent"].le(config.QC_CV_THRESHOLD)]
    rows, recovery = [], []
    for prevalence in (False, True):
        prepared = preprocess(data, max_missing_fraction=config.MAX_MISSING_FRACTION if prevalence else None)
        values, meta = prepared.values, prepared.metadata
        raw = data.values.loc[values.index, values.columns]
        mask = censoring_mask(raw, below_detection_reasons(raw, data.blanks.loc[values.index]))
        for cv_filter in (False, True):
            panel = values.index.intersection(low_cv) if cv_filter else values.index
            for method, imputer in IMPUTERS.items():
                if settings.quick and method == "rf_all":
                    continue
                spec = f"prevalence_{'on' if prevalence else 'off'}__cv_{'on' if cv_filter else 'off'}__{method}"
                completed = _impute(values.loc[panel], mask.loc[panel], meta, imputer, settings.seed)
                effects = fit_regionwise_differential(completed, meta, annotations=data.features,
                                                      min_per_cell=config.MIN_PER_CELL).effects
                plsda = _plsda_p(completed, meta, settings)
                for (region, effect), block in effects.groupby(["Region", "Effect"], sort=False):
                    rows.append({"specification": spec, "prevalence_filter": prevalence, "cv_filter": cv_filter,
                                 "imputation": method, "Region": region, "Effect": effect,
                                 "n_features": len(block), "n_fdr_hits": int(block["p_value_fdr"].le(config.FDR_ALPHA).sum()),
                                 "min_q": block["p_value_fdr"].min(), "n_p_below_0.01": int(block["p_value"].lt(0.01).sum()),
                                 "expected_p_below_0.01": 0.01 * len(block), "plsda_p": plsda[region]})
                if method != "none" and not (settings.quick and method.startswith("rf")):
                    recovery.append({"specification": spec, "imputation": method,
                                     "rmse": _recovery_rmse(values.loc[panel], mask.loc[panel], meta, imputer, settings.seed)})
    results = pd.DataFrame(rows)
    write_csv(results, out / "specification_results.csv")
    write_csv(pd.DataFrame(recovery), out / "imputation_recovery_rmse.csv")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (effect, rows_) in enumerate(results.groupby("Effect", sort=False)):
        ordered = rows_.groupby("specification")["min_q"].min().sort_values()
        ax.plot(range(len(ordered)), ordered.to_numpy(), "o-", color=plotting.COLORBLIND_COLORS[i], label=effect, markersize=4)
    ax.axhline(config.FDR_ALPHA, color="#666666", linestyle="--")
    ax.set(xlabel="specification (sorted within each effect)", ylabel="smallest FDR q over regions",
           title="Specification curve: smallest q-value per analysis choice")
    ax.legend(fontsize=8)
    plotting.save_figure(fig, out / "figures" / "specification_curve")
    return results


def _impute(values, mask, meta, imputer, seed) -> pd.DataFrame:
    completed = values.copy()
    for region in config.REGIONS:
        samples = region_samples(meta, region)
        kwargs = {"random_state": seed} if imputer in (impute_rf_mar_only, impute_rf_all) else {}
        completed.loc[:, samples] = imputer(values.loc[:, samples], mask.loc[:, samples], **kwargs)[0]
    return completed


def _recovery_rmse(values, mask, meta, imputer, seed) -> float:
    """Hide 1 % of observed cells, impute them, and measure the error."""
    hidden = values.stack().sample(frac=0.01, random_state=seed)
    masked, mask = values.copy(), mask.copy()
    for feature, sample in hidden.index:
        masked.loc[feature, sample] = np.nan
        mask.loc[feature, sample] = "mar"
    completed = _impute(masked, mask, meta, imputer, seed)
    return float(np.sqrt(np.mean((completed.stack().reindex(hidden.index) - hidden) ** 2)))


def _plsda_p(values, meta, settings) -> dict:
    result = {}
    for region in config.REGIONS:
        samples = region_samples(meta, region)
        complete = values.loc[:, samples].dropna()
        labels = meta.set_index("Sample").loc[samples, config.GROUP_COL]
        sexes = meta.set_index("Sample").loc[samples, config.SEX_COL]
        test, _ = permutation_test_plsda(complete, labels, sexes, n_permutations=settings.n_permutations,
                                         random_state=settings.seed)
        result[region] = float(test["permutation_p_value"])
    return result


# ---------------------------------------------------------------- RF / scaling grid
def rf_variants(settings: Settings, omics: str = "lipidomics") -> pd.DataFrame:
    """Population x RF imputation x scaling, all with the IQR filter (summary table only)."""
    out = settings.out(omics, "04_sensitivity", "rf_variants")
    summaries = []
    populations = [config.PRIMARY_POPULATION] if settings.quick else list(config.POPULATIONS)
    for population in populations:
        for imputation in ("rf_mar_only", "rf_all"):
            for scaling in ("none", "pareto"):
                name = f"rf/{imputation}__{population}__{scaling}"
                effects = differential.run(settings, omics, population, imputation=imputation, scaling=scaling,
                                           iqr_filter=True, name=name, light=True)
                summaries.append(differential.hit_summary(effects).assign(
                    population=population, imputation=imputation, scaling=scaling))
    table = pd.concat(summaries, ignore_index=True)
    write_csv(table, out / "rf_variant_hits.csv")
    return table


# ---------------------------------------------------------------- median vs PQN
def normalization_comparison(settings: Settings, omics: str = "metabolomics") -> pd.DataFrame:
    """Refit with PQN instead of median normalisation and compare effect by effect."""
    out = settings.out(omics, "04_sensitivity", "normalization")
    arms = {norm: differential.run(settings, omics, normalization=norm, name=f"normalization_{norm}", light=True)
            for norm in ("median", "pqn")}
    keys = ["Feature", "Region", "Effect"]
    columns = [*keys, "estimate_log2", "standard_error", "p_value_fdr"]
    paired = arms["median"][columns].merge(arms["pqn"][columns], on=keys, suffixes=("_median", "_pqn"))
    for arm in ("median", "pqn"):
        paired[f"t_{arm}"] = paired[f"estimate_log2_{arm}"] / paired[f"standard_error_{arm}"]
    paired["q05_status_changed"] = paired["p_value_fdr_median"].lt(config.FDR_ALPHA) != paired["p_value_fdr_pqn"].lt(config.FDR_ALPHA)
    write_csv(paired, out / "normalization_effect_comparison.csv")
    summary = paired.groupby(["Region", "Effect"]).apply(lambda g: pd.Series({
        "spearman_t": spearmanr(g["t_median"], g["t_pqn"]).statistic,
        "n_changed_q05": int(g["q05_status_changed"].sum()),
        "min_q_median": g["p_value_fdr_median"].min(), "min_q_pqn": g["p_value_fdr_pqn"].min(),
    }), include_groups=False).reset_index()
    write_csv(summary, out / "normalization_comparison_summary.csv")

    # Do the extra PQN shifts track treatment?  (They should not.)
    median_values = prepare(settings, omics).values
    pqn_values = prepare(settings, omics, normalization="pqn").values
    shift = (median_values - pqn_values).median(axis=0)
    meta = prepare(settings, omics).metadata.set_index("Sample").loc[shift.index]
    write_csv(pd.DataFrame([{"shift_sd_log2": shift.std(ddof=1),
                             "r2_group": r_squared(shift, meta[config.GROUP_COL]),
                             "r2_region": r_squared(shift, meta[config.REGION_COL])}]),
              out / "normalization_shift_summary.csv")
    return summary


# ---------------------------------------------------------------- litter (cage) structure
PRIMARY = "value ~ C(BioGroup, Treatment('Saline Control')) * C(Sex, Treatment('Male'))"
GROUP_TERM = "C(BioGroup, Treatment('Saline Control'))[T.MIA]"
ARMS = {
    "A": "cage means ~ group (the litter as the unit)",
    "B": "primary model + random intercept per cage",
    "C": "primary model + collection-date fixed effect",
    "D": "group + sex, restricted to the collection date with both groups",
}


def _fit(frame: pd.DataFrame, formula: str, groups: str | None = None) -> dict:
    """One feature, one model; return the MIA estimate or a fit status."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = None
        # L-BFGS often fails when the litter variance sits at zero; Powell then still converges.
        for method in (["ols"] if groups is None else ["lbfgs", "powell"]):
            try:
                if groups is None:
                    model = smf.ols(formula, data=frame).fit()
                else:
                    model = smf.mixedlm(formula, frame, groups=frame[groups]).fit(reml=False, method=method, disp=False)
                    _ = model.bse                      # fails here if the Hessian is singular
                break
            except Exception as error:  # statsmodels raises several numerical error types
                model, failure = None, str(error)[:80]
        if model is None:
            return {"fit_status": "failed", "failure_reason": failure}
    if GROUP_TERM not in model.params or not np.isfinite(model.pvalues.get(GROUP_TERM, np.nan)):
        return {"fit_status": "not_estimable"}
    status = "fitted"
    if groups is not None and (not model.converged or float(model.cov_re.iloc[0, 0]) <= 1e-10):
        status = "singular"
    return {"fit_status": status, "estimate_log2": model.params[GROUP_TERM],
            "standard_error": model.bse[GROUP_TERM], "p_value": model.pvalues[GROUP_TERM]}


def _arm(values: pd.DataFrame, meta: pd.DataFrame, arm: str, region: str, formula: str,
         groups: str | None = None) -> pd.DataFrame:
    frame = meta.rename(columns={config.GROUP_COL: "BioGroup", config.COHORT_COL: "Cohort"}).set_index("Sample")
    rows = []
    for feature, row in values.iterrows():
        data = frame.assign(value=row.reindex(frame.index).to_numpy()).dropna(subset=["value"])
        rows.append({"Arm": arm, "Region": region, "Feature": feature, "n_used": len(data), **_fit(data, formula, groups)})
    table = pd.DataFrame(rows)
    table["p_value_fdr"] = bh_adjust(table["p_value"]) if "p_value" in table else np.nan
    return table


def cage_structure(settings: Settings, omics: str = "lipidomics") -> pd.DataFrame:
    """Litter-aware sensitivity analyses of the MIA effect."""
    plotting.apply_plot_style()
    out = settings.out(omics, "04_sensitivity", "cage_structure")
    prepared = prepare(settings, omics)
    values, meta = prepared.values, prepared.metadata
    both_groups = meta.groupby(config.COHORT_COL, observed=True)[config.GROUP_COL].nunique()
    matched_date = both_groups.index[both_groups.eq(2)]

    mice = meta.drop_duplicates(config.MOUSE_COL)
    cages = mice.groupby(config.CAGE_COL).agg(n_mice=(config.MOUSE_COL, "nunique"), group=(config.GROUP_COL, "first"),
                                              sexes=(config.SEX_COL, lambda s: ";".join(sorted(s.unique()))),
                                              collected=(config.COHORT_COL, "first")).reset_index()
    write_csv(cages, out / "cage_summary.csv")

    tables, iccs = [], []
    for region in config.REGIONS:
        samples = region_samples(meta, region)
        block, block_meta = values.loc[:, samples], meta.set_index("Sample").loc[samples].reset_index()
        cage_values, cage_meta = collapse_to_cage_means(block, block_meta)
        tables.append(_arm(cage_values, cage_meta, "A", region, "value ~ C(BioGroup, Treatment('Saline Control'))"))
        tables.append(_arm(block, block_meta, "B", region, PRIMARY, groups=config.CAGE_COL))
        tables.append(_arm(block, block_meta, "C", region, PRIMARY + " + C(Cohort)"))
        matched = block_meta.loc[block_meta[config.COHORT_COL].isin(matched_date)]
        tables.append(_arm(block.loc[:, matched["Sample"]], matched, "D", region,
                           "value ~ C(BioGroup, Treatment('Saline Control')) + C(Sex, Treatment('Male'))"))
        # Whole cages are the exchangeable unit for the MIA label.
        features, null_summary, _ = run_permutation_null(block, block_meta, min_per_cell=config.MIN_PER_CELL,
                                                         exchangeability_unit="cage", block_column=config.CAGE_COL,
                                                         effects=("MIA vs Saline Control",))
        write_csv(features, out / f"cage_permutation_{region.lower()}.csv")
        iccs.append(intraclass_correlation(block, block_meta, n_bootstraps=settings.n_bootstraps,
                                           random_state=settings.seed).assign(Region=region))
    effects = pd.concat(tables, ignore_index=True)
    write_csv(effects, out / "cage_aware_effects.csv")
    write_csv(effects.groupby(["Arm", "Region", "fit_status"]).size().rename("n_features").reset_index(), out / "fit_status_counts.csv")
    icc = pd.concat(iccs, ignore_index=True)
    write_csv(icc, out / "cage_icc.csv")
    ceiling = cage_permutation_ceiling(meta)
    write_json({"arms": ARMS, "date_matched_collection": [str(d) for d in matched_date],
                "permutation_ceiling": ceiling, "sex_exchangeability": sex_exchangeability_summary(meta),
                "median_icc": float(icc["icc"].median()) if "icc" in icc else None},
               out / "cage_structure_manifest.json")
    _cage_figures(values, meta, out / "figures")
    return effects


def _cage_figures(values: pd.DataFrame, meta: pd.DataFrame, out) -> None:
    """PCA coloured by cage, and how far each mouse sits from its cage centroid."""
    for region in config.REGIONS:
        samples = region_samples(meta, region)
        model, scores, _ = fit_pca(for_multivariate(values.loc[:, samples]), n_components=2)
        fig = plotting.scores_plot(scores, meta, config.CAGE_COL, variance=model.explained_variance_ratio_,
                                   title=f"{region}: PCA by cage (litter)", ellipses=False)
        plotting.save_figure(fig, out / f"pca_by_cage_{region.lower()}")
        joined = scores.join(meta.set_index("Sample")[[config.CAGE_COL, config.GROUP_COL]])
        centroids = joined.groupby(config.CAGE_COL)[["PC 1", "PC 2"]].transform("mean")
        joined["distance"] = np.hypot(joined["PC 1"] - centroids["PC 1"], joined["PC 2"] - centroids["PC 2"])
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, (cage, rows) in enumerate(joined.groupby(config.CAGE_COL)):
            ax.scatter(np.full(len(rows), i), rows["distance"], color=rows[config.GROUP_COL].map(plotting.BIO_GROUP_COLORS))
        ax.set_xticks(range(joined[config.CAGE_COL].nunique()), sorted(joined[config.CAGE_COL].unique()), rotation=30)
        ax.set(ylabel="distance to cage centroid (PC1-PC2)", title=f"{region}: within-litter spread")
        plotting.save_figure(fig, out / f"within_cage_distance_{region.lower()}")


# ---------------------------------------------------------------- pooled mixed model
def pooled_mixed_model(settings: Settings, omics: str = "lipidomics") -> pd.DataFrame:
    """One model per feature over all 60 samples: group x region + group x sex, random intercept per mouse."""
    out = settings.out(omics, "04_sensitivity", "pooled_mixed_model")
    prepared = prepare(settings, omics)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = fit_pooled_mixed_sensitivity(prepared.values, prepared.metadata)
    mia = table.loc[table["Term"].eq("C(BioGroup, Treatment('Saline Control'))[T.MIA]")].copy()
    mia["p_value_fdr"] = bh_adjust(mia["p_value"], denominator=len(prepared.values))
    write_csv(table, out / "pooled_mixed_effects.csv")
    write_csv(mia, out / "pooled_mixed_mia_main_effect.csv")
    return mia
