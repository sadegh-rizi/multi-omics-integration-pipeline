"""Stage 6 - ASCA+: one multivariate ANOVA instead of hundreds of tests.

The sample x feature matrix (complete features only) is split into effect
matrices - Region, Group (MIA), Sex, their interactions, Mouse (nested in
group x sex) and residual - with an effect-coded linear model (ASCA+,
which handles the unbalanced design).  Each effect matrix is summarised by
its share of the variance and by a PCA (scores + loadings).  Significance comes
from permuting labels at the right unit:

* Region     - shuffle region labels *within* each mouse;
* Group      - shuffle whole cages (exact; floor = 1 / number of assignments)
               and, as a liberal comparison, mice within sex;
* Sex        - shuffle sex within mixed-sex control cages.

A null/power simulation (primary population only) checks that the mouse-level
tests are calibrated and how large a Region x Group effect would be detectable.

Output: ``results/<omics>/06_asca/<population>/``.
"""

from __future__ import annotations

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kstest

from .. import config, plotting
from ..config import Settings
from ..multivariate.asca import (
    EFFECT_ORDER,
    SCALING_MODES,
    effect_residual_projection,
    effect_svd,
    effect_variance_table,
    fit_asca,
    fit_asca_sample_matrix,
    procrustes_align,
)
from ..preprocessing import complete_features
from ..stats.asca_permutation import (
    cage_group_permutations,
    control_cage_sex_permutations,
    effect_ssq_by_scaling,
    make_spike_profile,
    mouse_group_permutations,
    region_label_permutations,
    stratified_mouse_bootstrap_indices,
)
from ..stats.common import exact_empirical_p
from .common import feature_class, prepare, write_csv, write_json

PRIMARY_SCALING = "autoscaled"
BOOTSTRAP_EFFECTS = ("Region", "Group", "Sex", "Region:Group")   # loading intervals only for the interpretable effects


def run(settings: Settings, omics: str = "lipidomics", population: str = config.PRIMARY_POPULATION,
        *, power: bool | None = None) -> pd.DataFrame:
    """Fit ASCA+ for every scaling, test effects by permutation; return the variance table."""
    plotting.apply_plot_style()
    out = settings.out(omics, "06_asca", population)
    prepared = prepare(settings, omics, population)
    values = complete_features(prepared.values)
    meta = prepared.metadata.set_index("Sample").loc[list(values.columns)].reset_index()
    classes = feature_class(prepared.features)
    write_csv(values.rename_axis("Feature").reset_index(), out / "inputs" / "complete_feature_matrix.csv")   # also read by r/asca_crosscheck.R
    write_csv(meta, out / "inputs" / "metadata.csv")

    fits, variances, loadings, projections = {}, [], [], []
    for scaling in SCALING_MODES:
        fit = fit_asca(values, meta, scaling=scaling)
        fits[scaling] = fit
        variances.append(effect_variance_table(fit).assign(Scaling=scaling))
        intervals = _bootstrap_intervals(values, meta, fit, settings) if scaling == PRIMARY_SCALING else {}
        for effect in EFFECT_ORDER:
            _, effect_loadings, singular = effect_svd(fit, effect)
            table = effect_loadings.rename_axis("Feature").reset_index().melt(
                id_vars="Feature", var_name="Component", value_name="Loading")
            table = table.assign(Effect=effect, Scaling=scaling, absolute_loading=table["Loading"].abs())
            if effect in intervals:
                table = table.merge(intervals[effect], on=["Feature", "Component"], how="left")
            loadings.append(table.merge(classes.rename_axis("Feature").reset_index(), on="Feature", how="left"))
            projections.append(effect_residual_projection(fit, effect, effect_loadings).assign(Scaling=scaling))
    variance = pd.concat(variances, ignore_index=True)
    loading_table = pd.concat(loadings, ignore_index=True)
    write_csv(variance, out / "effect_variance.csv")
    write_csv(loading_table, out / "loadings.csv")
    write_csv(pd.concat(projections, ignore_index=True), out / "effect_residual_projection.csv")
    write_csv(_scores(fits[PRIMARY_SCALING], meta), out / "scores_autoscaled.csv")

    tests, null = permutation_tests(fits, meta, settings)
    write_csv(tests, out / "permutation_tests.csv")
    write_csv(null, out / "permutation_null_ssq.csv")
    _composite_figure(fits[PRIMARY_SCALING], variance.loc[variance["Scaling"].eq(PRIMARY_SCALING)], tests,
                      loading_table.loc[loading_table["Scaling"].eq(PRIMARY_SCALING)], omics, out / "figures")

    if power if power is not None else population == config.PRIMARY_POPULATION:
        calibration, spikes = power_simulation(fits[PRIMARY_SCALING], meta, settings)
        write_csv(calibration, out / "power_null_calibration.csv")
        write_csv(spikes, out / "power_simulation.csv")
    write_json({"omics": omics, "population": population, "n_samples": len(meta), "n_complete_features": len(values),
                "n_permutations": settings.n_asca_permutations, "n_bootstraps": settings.n_bootstraps,
                "scalings": list(SCALING_MODES)}, out / "manifest.json")
    return variance


def _scores(fit, meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for effect in EFFECT_ORDER:
        scores, _, _ = effect_svd(fit, effect)
        block = scores.iloc[:, :2].copy()
        block.columns = ["PC1", "PC2"][:block.shape[1]]
        rows.append(pd.concat([meta[["Sample", config.REGION_COL, config.GROUP_COL, config.SEX_COL]].reset_index(drop=True),
                               block.reset_index(drop=True)], axis=1).assign(Effect=effect))
    return pd.concat(rows, ignore_index=True)


def _bootstrap_intervals(values: pd.DataFrame, meta: pd.DataFrame, fit, settings: Settings) -> dict[str, pd.DataFrame]:
    """95 % loading intervals from a mouse bootstrap (stratified by group x sex), Procrustes-aligned."""
    references = {effect: effect_svd(fit, effect)[1] for effect in BOOTSTRAP_EFFECTS}
    draws = {effect: [] for effect in BOOTSTRAP_EFFECTS}
    source = values.to_numpy(dtype=float).T
    for b in range(settings.n_bootstraps):
        index = stratified_mouse_bootstrap_indices(meta, random_state=settings.seed + b)
        boot = fit_asca_sample_matrix(source[index], list(fit.feature_names), meta.iloc[index].reset_index(drop=True),
                                      scaling=fit.scaling, allow_rank_deficient=True)
        for effect, reference in references.items():
            loading = effect_svd(boot, effect)[1].to_numpy(dtype=float)
            padded = np.zeros(reference.shape)
            k = min(loading.shape[1], reference.shape[1])
            padded[:, :k] = loading[:, :k]
            draws[effect].append(procrustes_align(padded, reference.to_numpy(dtype=float)))
    result = {}
    for effect, reference in references.items():
        stack = np.stack(draws[effect])
        result[effect] = pd.concat([pd.DataFrame({
            "Feature": reference.index, "Component": component,
            "loading_q025": np.quantile(stack[:, :, i], 0.025, axis=0),
            "loading_q975": np.quantile(stack[:, :, i], 0.975, axis=0)}) for i, component in enumerate(reference.columns)])
    return result


def permutation_tests(fits: dict, meta: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Effect sum of squares vs its permutation null, for every scaling."""
    prepared = {scaling: fit.prepared_data for scaling, fit in fits.items()}
    n, seed = settings.n_asca_permutations, settings.seed
    engines = [
        ("Region", ("Region",), "region within mouse", region_label_permutations(meta, n_random=n, random_state=seed)),
        ("Group (cage)", ("Group", "Region:Group"), "whole cage", cage_group_permutations(meta)),
        ("Group (mouse)", ("Group", "Region:Group"), "mouse within sex", mouse_group_permutations(meta, n_random=n, random_state=seed + 1)),
        ("Sex", ("Sex", "Region:Sex"), "sex within control cage", control_cage_sex_permutations(meta, n_random=n, random_state=seed + 2)),
    ]
    null_rows = []
    for engine, effects, unit, permutations in engines:
        for permutation_id, permuted in permutations:
            for scaling, statistics in effect_ssq_by_scaling(prepared, permuted, effects).items():
                for effect, ssq in statistics.items():
                    null_rows.append({"Engine": engine, "Unit": unit, "Permutation": permutation_id,
                                      "Scaling": scaling, "Effect": effect, "SSQ": ssq})
    null = pd.DataFrame(null_rows)
    tests = []
    for (engine, unit, scaling, effect), group in null.groupby(["Engine", "Unit", "Scaling", "Effect"], sort=False):
        observed = float(np.square(fits[scaling].effects[effect]).sum())
        statistics = group["SSQ"].to_numpy(dtype=float)
        tests.append({"Engine": engine, "Unit": unit, "Scaling": scaling, "Effect": effect,
                      "observed_ssq": observed, "variance_share": observed / fits[scaling].total_ssq,
                      "null_q025": np.quantile(statistics, 0.025), "null_q975": np.quantile(statistics, 0.975),
                      "p_value": exact_empirical_p(observed, statistics), "n_permutations": len(statistics)})
    return pd.DataFrame(tests), null


def _residual_factor(fit) -> np.ndarray:
    """Low-rank factor that reproduces the residual covariance (for simulating null data)."""
    residual = fit.residual - fit.residual.mean(axis=0)
    _, singular, right = np.linalg.svd(residual, full_matrices=False)
    rank = max(1, min(np.linalg.matrix_rank(residual), len(singular)))
    return right[:rank].T * (singular[:rank] / np.sqrt(max(fit.n_samples - 1, 1)))


def power_simulation(fit, meta: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(1) p-values on simulated null data should be uniform; (2) detection rate of planted effects."""
    rng = np.random.default_rng(settings.seed)
    factor = _residual_factor(fit)
    n_perm = settings.n_power_permutations
    ssq = lambda data, m, effect: effect_ssq_by_scaling({fit.scaling: data}, m, (effect,))[fit.scaling][effect]  # noqa: E731
    calibration = []
    for dataset in range(settings.n_power_datasets):
        data = rng.normal(size=(fit.n_samples, factor.shape[1])) @ factor.T
        for effect, permutations in (("Region", region_label_permutations(meta, n_random=n_perm, random_state=settings.seed + dataset)),
                                     ("Region:Group", mouse_group_permutations(meta, n_random=n_perm, random_state=settings.seed + 10_000 + dataset))):
            null = np.array([ssq(data, m, effect) for _, m in permutations])
            calibration.append({"dataset": dataset, "Effect": effect, "p_value": exact_empirical_p(ssq(data, meta, effect), null)})
    calibration = pd.DataFrame(calibration)
    summary = calibration.groupby("Effect")["p_value"].agg(
        rejection_rate_at_0_05=lambda p: float(np.mean(p < 0.05)),
        ks_p_value=lambda p: float(kstest(p, "uniform").pvalue)).reset_index()
    calibration = pd.concat([calibration, summary.assign(dataset="summary")], ignore_index=True)

    profile = make_spike_profile(meta)
    mia = meta[config.GROUP_COL].eq(config.MIA).to_numpy(dtype=float)
    cages = list(cage_group_permutations(meta))
    mice = list(mouse_group_permutations(meta, n_random=n_perm, random_state=settings.seed + 30_000))
    rows = []
    for shift, n_features in product((0.25, 0.5, 1.0), (10, 50, 200)):
        for dataset in range(max(2, settings.n_power_datasets // 4)):
            data = rng.normal(size=(fit.n_samples, factor.shape[1])) @ factor.T
            data[:, :min(n_features, fit.n_features)] += (shift * profile * mia)[:, None]
            observed = ssq(data, meta, "Region:Group")
            for engine, assignments in (("cage", cages), ("mouse", mice)):
                null = np.array([ssq(data, m, "Region:Group") for _, m in assignments])
                rows.append({"shift_sd": shift, "n_spiked_features": n_features, "engine": engine,
                             "detected": exact_empirical_p(observed, null) < 0.05})
    spikes = pd.DataFrame(rows).groupby(["shift_sd", "n_spiked_features", "engine"])["detected"].mean().rename("detection_rate").reset_index()
    return calibration, spikes


def _composite_figure(fit, variance: pd.DataFrame, tests: pd.DataFrame, loadings: pd.DataFrame, omics: str, out) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    shown = variance.loc[~variance["Effect"].isin(["Residual"])]
    axes[0, 0].bar(range(len(shown)), shown["variance_share"], color=plotting.COLORBLIND_COLORS[0])
    axes[0, 0].set_xticks(range(len(shown)), shown["Effect"], rotation=40, ha="right")
    for i, effect in enumerate(shown["Effect"]):
        row = tests.loc[tests["Effect"].eq(effect) & tests["Scaling"].eq(PRIMARY_SCALING)]
        if len(row):
            axes[0, 0].vlines(i, row["null_q025"].iloc[0] / fit.total_ssq, row["null_q975"].iloc[0] / fit.total_ssq, color="#333333", linewidth=3)
            axes[0, 0].annotate(f"p={row['p_value'].iloc[0]:.3f}", (i, shown["variance_share"].iloc[i]), fontsize=7, ha="center", va="bottom")
    axes[0, 0].set(ylabel="share of variance", title="Effect size (bars) vs permutation null (black)")
    for ax, effect, color_by in ((axes[0, 1], "Region", config.REGION_COL), (axes[1, 0], "Region:Group", config.GROUP_COL)):
        scores, _, _ = effect_svd(fit, effect)
        colors = plotting.PALETTES[color_by]
        labels = fit.metadata[color_by].astype(str).to_numpy()
        ax.scatter(scores.iloc[:, 0], scores.iloc[:, 1] if scores.shape[1] > 1 else np.zeros(len(scores)),
                   c=[colors[label] for label in labels], edgecolor="black", linewidth=0.3)
        ax.set(xlabel="PC1", ylabel="PC2", title=f"{effect} effect + residual, coloured by {color_by}")
    top = loadings.loc[loadings["Effect"].eq("Region") & loadings["Component"].eq("PC1")].nlargest(15, "absolute_loading").sort_values("Loading")
    axes[1, 1].barh(top["Feature"].astype(str), top["Loading"], color=plotting.COLORBLIND_COLORS[2])
    axes[1, 1].set(xlabel="Region PC1 loading", title="Strongest region loadings")
    fig.suptitle(f"ASCA+ summary: {omics}", fontweight="bold")
    plotting.save_figure(fig, out / "asca_composite")
