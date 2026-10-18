"""Stage 2 - sample outliers.

Two complementary screens, both evidence only (they never delete data):

* :func:`run_screen` - the classical regional PCA screen: Hotelling T2 and a
  robust (MCD) Mahalanobis distance in the first 3 PCs, plus DModX (distance
  to the PCA model).  Its T2 95 % / 99 % flags define the two permitted
  sensitivity populations ``t2_95`` and ``t2_99``.
* :func:`run_ensemble` - six detector families, every parameter swept, each
  setting calibrated against simulated outlier-free data in two stages
  (contaminated, then trimmed).  A detector "stably" flags a sample when at
  least 60 % of its settings flag it; a sample is a candidate outlier when at
  least two detectors (one of them not covariance-based) stably flag it.

:func:`run_concordance` asks whether lipidomics and metabolomics flag the same
samples (they share an extraction, so agreement is not independent).

Output: ``results/<omics>/02_outliers/``.
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

from .. import config, plotting, qc
from ..config import Settings
from ..multivariate.pca import fit_pca, hotelling_ellipse
from ..outliers import calibration, detectors
from ..outliers.ensemble import apply_stability_rule, stability_fraction
from ..outliers.screen import sample_outlier_table
from ..preprocessing import for_multivariate
from .common import prepare, region_samples, write_csv


# ---------------------------------------------------------------- classical screen
def run_screen(settings: Settings, omics: str = "lipidomics") -> pd.DataFrame:
    """Regional PCA outlier screen on all 60 samples; writes ``outlier_screen.csv``."""
    plotting.apply_plot_style()
    out = settings.out(omics, "02_outliers")
    prepared = prepare(settings, omics)          # always all 60 samples
    meta = prepared.metadata
    tables = []
    for region in config.REGIONS:
        samples = region_samples(meta, region)
        matrix = for_multivariate(prepared.values.loc[:, samples])
        model, scores, _ = fit_pca(matrix, n_components=config.OUTLIER_SCREEN_PCS)
        table = sample_outlier_table(matrix, model, scores, random_state=settings.seed)

        # DModX normalised to its RMS and compared with an F-based 95 % limit.
        n_features, n_samples, k = matrix.shape[0], matrix.shape[1], config.OUTLIER_SCREEN_PCS
        table["normalized_dmodx"] = table["dmodx"] / np.sqrt(np.mean(table["dmodx"] ** 2))
        table["dmodx_95_limit"] = float(np.sqrt(f_dist.ppf(0.95, n_features - k, (n_features - k) * (n_samples - k - 1))))
        table["normalized_dmodx_95_flag"] = table["normalized_dmodx"].gt(table["dmodx_95_limit"])
        table = table.rename_axis("Sample").reset_index()
        table.insert(1, "Region", region)
        tables.append(table.merge(meta[["Sample", "Sample number", config.MOUSE_COL, config.GROUP_COL, config.SEX_COL]], on="Sample"))
        _screen_figures(prepared.values.loc[:, samples], table.merge(meta[["Sample", "Sample number"]], on="Sample"),
                        meta, region, out / "figures")
    screen = pd.concat(tables, ignore_index=True)
    write_csv(screen, out / "outlier_screen.csv")
    write_csv(screen.groupby("Region")[[c for c in screen if c.endswith("_flag")]].sum().reset_index(),
              out / "outlier_flag_counts.csv")
    return screen


def _screen_figures(values: pd.DataFrame, table: pd.DataFrame, meta: pd.DataFrame, region: str, out) -> None:
    """T² and DModX index plots, and the PC1/PC2 plot with 2-D Hotelling ellipses, for one region."""
    labels = table.set_index("Sample")["Sample number"]
    t2 = table.set_index("Sample")
    fig = plotting.index_plot(t2["hotelling_t2"], labels,
                              {"95 %": t2["hotelling_t2_95_limit"].iloc[0], "99 %": t2["hotelling_t2_99_limit"].iloc[0]},
                              title=f"{region}: Hotelling T2 ({config.OUTLIER_SCREEN_PCS} PCs)", ylabel="T2")
    plotting.save_figure(fig, out / f"t2_{region.lower()}")
    fig = plotting.index_plot(t2["normalized_dmodx"], labels, {"95 %": t2["dmodx_95_limit"].iloc[0]},
                              title=f"{region}: distance to PCA model (DModX)", ylabel="normalised DModX")
    plotting.save_figure(fig, out / f"dmodx_{region.lower()}")

    # PC1/PC2 score plot with the 2-D Hotelling T2 ellipses at 95 % and 99 %.
    model, scores, _ = fit_pca(for_multivariate(values), n_components=2)
    curves = [hotelling_ellipse(scores, alpha=a)[0] for a in (0.95, 0.99)]
    fig = plotting.scores_plot(scores, meta, config.GROUP_COL, variance=model.explained_variance_ratio_,
                               title=f"{region}: PCA with Hotelling T2 95/99 %", label="Sample number",
                               ellipses=False, outline=curves)
    plotting.save_figure(fig, out / f"pca_hotelling_{region.lower()}")


# ---------------------------------------------------------------- calibrated ensemble
GAMMAS = tuple(np.logspace(-2, 0, 9))          # kernel-PCA RBF widths
SHRINKAGES = (0.875, 1.0)                       # Ledoit-Wolf shrinkage values swept (plus the data-chosen one)
NON_COVARIANCE = ("knn_isotropic", "lof", "isolation_forest", "kernel_pca")


def _two_stage(values: pd.DataFrame, observed: pd.DataFrame,
               calibrate: Callable[[pd.DataFrame], tuple[pd.DataFrame, dict]]) -> pd.DataFrame:
    """Calibrate every column (= parameter setting) of ``observed`` in two stages.

    Stage "contaminated": simulate null data with the covariance of all samples.
    Stage "trimmed": re-simulate with the covariance of the samples *not*
    flagged in stage 1, so a real outlier cannot inflate its own threshold.
    ``calibrate(calibration_rows)`` returns (limits per parameter, null scores per parameter).
    """
    rows = []
    # Stage 1: limits from all samples.
    limits1, nulls1 = calibrate(values)
    # Parameter settings that flag the same samples share one stage-2 calibration.
    masks: dict[tuple, list] = {}
    for parameter in observed:
        flags = observed[parameter].gt(limits1.loc[parameter, "limit_95"])
        masks.setdefault(tuple(flags), []).append(parameter)
    # Stage 2: recalibrate without the samples flagged in stage 1 (if enough samples remain).
    limits2, nulls2 = {}, {}
    for mask, parameters in masks.items():
        kept = values.loc[~np.array(mask)]
        limits, nulls = calibrate(kept) if len(kept) > values.shape[1] + 1 or len(kept) >= 3 else (limits1, nulls1)
        for parameter in parameters:
            limits2[parameter], nulls2[parameter] = limits.loc[parameter], nulls[parameter]
    # One long table: every sample x setting x stage, with its score, null percentile and flags.
    for stage, limits, nulls in (("contaminated", limits1, nulls1), ("trimmed", pd.DataFrame(limits2).T, nulls2)):
        for parameter in observed:
            score = observed[parameter]
            rows.append(pd.DataFrame({
                "Sample": score.index, "parameter_value": str(parameter), "score": score.to_numpy(),
                "null_percentile": calibration.score_to_percentile(score, nulls[parameter]).to_numpy(),
                "flag_95": score.gt(limits.loc[parameter, "limit_95"]).to_numpy(),
                "flag_99": score.gt(limits.loc[parameter, "limit_99"]).to_numpy(),
                "limit_95": limits.loc[parameter, "limit_95"],
                "false_positive_rate": limits.loc[parameter].get("false_positive_rate", np.nan),
                "calibration_stage": stage,
            }))
    return pd.concat(rows, ignore_index=True)


def _generic_calibrator(values: pd.DataFrame, scorers: dict, n: int, seed: int):
    """Simulation calibration for detectors without a batched calibrator (kernel PCA)."""
    def calibrate(rows: pd.DataFrame):
        """Limits and null scores for every parameter, with covariance from ``rows``."""
        limits, nulls = {}, {}
        for parameter, scorer in scorers.items():
            limit, null = calibration.simulate_null_limits(values, scorer, calibration_values=rows,
                                                           n_replicates=n, random_state=seed)
            limits[parameter], nulls[parameter] = limit, null
        return pd.DataFrame(limits).T, nulls
    return calibrate


def _detector_settings(scope: str, raw: pd.DataFrame, pcs: dict, settings: Settings) -> list[pd.DataFrame]:
    """Score and calibrate every detector setting for one scope; return long tables."""
    n = settings.n_null_replicates
    seed = settings.seed
    n_samples = len(raw)
    k_values = list(range(2, min(26 if scope == "pooled" else 16, n_samples - 1)))
    lof_values = list(range(10, 26)) if scope == "pooled" else list(range(2, min(16, n_samples - 1)))
    if_seeds = list(range(seed, seed + settings.n_isolation_forest_seeds))
    tables = []

    def add(table: pd.DataFrame, detector: str, representation: str, parameter_name: str, counts: bool | Callable) -> None:
        """Label one detector's long table; ``counts`` says which settings enter the stability rule."""
        table["detector"], table["representation"], table["parameter_name"], table["scope"] = detector, representation, parameter_name, scope
        table["counts_for_stability"] = table["parameter_value"].map(counts) if callable(counts) else counts
        tables.append(table)

    for representation, (scores, eigenvalues) in pcs.items():
        # k-nearest-neighbour mean distance, plain and eigenvalue-whitened ("soft")
        weightings = {"isotropic": scores}
        if eigenvalues.min() >= 0.01 * eigenvalues.max():
            weightings["soft"] = scores.divide(np.sqrt(eigenvalues), axis=1)
        for name, frame in weightings.items():
            observed = detectors.knn_mean_distance_sweep(frame, k_values=k_values)
            observed.columns = k_values
            add(_two_stage(frame, observed, lambda rows, frame=frame: calibration.calibrate_knn_sweep(
                frame, k_values=k_values, calibration_values=rows, n_replicates=n, random_state=seed)),
                f"knn_{name}", representation, "k", True)
        # local outlier factor (supporting evidence only within regions: n = 20 is small)
        observed = detectors.local_outlier_factor_sweep(scores, k_values=lof_values)
        observed.columns = lof_values
        add(_two_stage(scores, observed, lambda rows, scores=scores: calibration.calibrate_lof_sweep(
            scores, k_values=lof_values, calibration_values=rows, n_replicates=n, random_state=seed)),
            "lof" if scope == "pooled" else "lof_supporting", representation, "k", scope == "pooled")
        # isolation forest averaged over seeds
        observed = detectors.isolation_forest_seed_scores(scores, seeds=if_seeds, n_estimators=settings.n_isolation_forest_trees).mean(axis=1).to_frame(len(if_seeds))
        def if_calibrate(rows, scores=scores):
            """Isolation-forest calibration in the (limits, nulls) shape that _two_stage expects."""
            limit, null, rate = calibration.calibrate_isolation_forest(
                scores, seeds=if_seeds, calibration_values=rows, n_estimators=settings.n_isolation_forest_trees,
                n_replicates=max(5, n // 4), random_state=seed)
            return pd.DataFrame({len(if_seeds): {**limit.to_dict(), "false_positive_rate": rate}}).T, {len(if_seeds): null}
        add(_two_stage(scores, observed, if_calibrate), "isolation_forest", representation, "seeds", True)
        # kernel-PCA reconstruction error over a grid of RBF widths
        q = scores.shape[1]
        scorers = {g: (lambda frame, g=g: detectors.kernel_pca_reconstruction_distance(frame, parameter=g, n_components=q)) for g in GAMMAS}
        observed = pd.DataFrame({g: s(scores) for g, s in scorers.items()})
        add(_two_stage(scores, observed, _generic_calibrator(scores, scorers, max(5, n // 4), seed)),
            "kernel_pca", representation, "gamma", True)

    # Ledoit-Wolf shrunk Mahalanobis distance on the full feature space.
    auto = detectors.ledoit_wolf_auto_intensity(raw)
    shrinkages = [*SHRINKAGES, auto]
    observed = detectors.ledoit_wolf_mahalanobis_sweep(raw, shrinkages=shrinkages)
    observed.columns = shrinkages
    add(_two_stage(raw, observed, lambda rows: calibration.calibrate_ledoit_wolf_sweep(
        raw, shrinkages=shrinkages, calibration_values=rows, n_replicates=max(5, n // 4), random_state=seed)),
        "mahalanobis", "raw", "shrinkage", lambda p: p != str(auto))
    return tables


def run_ensemble(settings: Settings, omics: str = "lipidomics") -> pd.DataFrame:
    """Multi-detector, calibrated outlier ensemble; returns the per-sample verdict."""
    plotting.apply_plot_style()
    out = settings.out(omics, "02_outliers", "ensemble")
    prepared = prepare(settings, omics)
    meta = prepared.metadata
    scopes = {"pooled": meta["Sample"].tolist()} | {r.lower(): region_samples(meta, r) for r in config.REGIONS}

    long_tables, contrast = [], []
    for scope, samples in scopes.items():
        matrix = for_multivariate(prepared.values.loc[:, samples])
        model, scores, _ = fit_pca(matrix, n_components=max(config.ENSEMBLE_PCA_COMPONENTS))
        pcs = {f"pc{q}": (scores.iloc[:, :q], model.explained_variance_[:q]) for q in config.ENSEMBLE_PCA_COMPONENTS}
        # Distances concentrate in high dimension: compare the raw space with PC spaces.
        contrast.append({"scope": scope, "representation": "raw", "relative_contrast": detectors.distance_contrast(matrix.T, k=2)})
        contrast += [{"scope": scope, "representation": r, "relative_contrast": detectors.distance_contrast(s, k=2)} for r, (s, _) in pcs.items()]
        long_tables += _detector_settings(scope, matrix.T, pcs, settings)

    scores = pd.concat(long_tables, ignore_index=True)
    stable = stability_fraction(scores.loc[scores["counts_for_stability"]]).reset_index()
    screen = pd.read_csv(settings.results_dir / omics / "02_outliers" / "outlier_screen.csv").set_index("Sample")
    verdicts = []
    for scope, samples in scopes.items():
        rules = {}
        for stage in ("contaminated", "trimmed"):
            subset = stable.loc[stable["scope"].eq(scope) & stable["calibration_stage"].eq(stage)]
            rules[stage] = apply_stability_rule(subset, minimum_fraction=config.ENSEMBLE_STABILITY_FRACTION).reindex(samples).fillna(0)
        verdict = rules["trimmed"].join(rules["contaminated"].add_suffix("_stage1"))
        verdict["candidate_outlier"] = verdict["candidate_outlier"].astype(bool)
        verdict["calibration_sensitive"] = verdict["candidate_outlier"] != verdict["candidate_outlier_stage1"].astype(bool)
        verdict["robust_candidate"] = verdict["candidate_outlier"] & ~verdict["calibration_sensitive"]
        # Descriptive: mean null percentile over one representative setting per detector.
        stage2 = scores.loc[scores["scope"].eq(scope) & scores["calibration_stage"].eq("trimmed") & scores["representation"].isin(["pc3", "raw"])]
        verdict["mean_null_percentile"] = stage2.groupby(["Sample", "detector"])["null_percentile"].median().groupby("Sample").mean()
        verdict["hotelling_t2_95_flag"] = screen["hotelling_t2_95_flag"].reindex(samples).to_numpy() if scope != "pooled" else np.nan
        verdict["scope"] = scope
        verdicts.append(verdict.rename_axis("Sample").reset_index())
    verdict = pd.concat(verdicts, ignore_index=True).merge(meta[["Sample", "Sample number"]], on="Sample")

    write_csv(scores, out / "detector_scores_long.csv")
    write_csv(stable, out / "stability_fractions.csv")
    write_csv(verdict, out / "ensemble_verdict.csv")
    write_csv(pd.DataFrame(contrast), out / "distance_concentration.csv")
    limits = scores.drop_duplicates(["scope", "detector", "representation", "parameter_value", "calibration_stage"])
    write_csv(limits.drop(columns=["Sample", "score", "null_percentile", "flag_95", "flag_99"]), out / "null_limits.csv")
    _ensemble_figures(scores, stable, meta, out / "figures")
    return verdict


def _ensemble_figures(scores: pd.DataFrame, stable: pd.DataFrame, meta: pd.DataFrame, out) -> None:
    """Stability heatmaps, parameter sweeps and the false-positive-rate check."""
    labels = meta.set_index("Sample")["Sample number"]
    # 1. stability heatmap: share of settings flagging each sample, per detector (trimmed stage)
    for scope, block in stable.loc[stable["calibration_stage"].eq("trimmed")].groupby("scope"):
        matrix = block.pivot(index="detector", columns="Sample", values="stability_fraction")
        matrix = matrix.loc[:, sorted(matrix.columns, key=lambda s: labels[s])]
        fig, ax = plt.subplots(figsize=(max(6, 0.22 * matrix.shape[1]), 3))
        image = ax.imshow(matrix, cmap="Reds", vmin=0, vmax=1, aspect="auto")
        ax.set(xticks=range(matrix.shape[1]), yticks=range(matrix.shape[0]), yticklabels=matrix.index,
               title=f"{scope}: share of settings flagging each sample")
        ax.set_xticklabels([labels[s] for s in matrix.columns], fontsize=6, rotation=90)
        fig.colorbar(image, ax=ax, label="stability")
        plotting.save_figure(fig, out / f"stability_{scope}")
    # 2. parameter sweeps: how many samples each setting flags
    sweep = scores.loc[scores["calibration_stage"].eq("trimmed")].groupby(
        ["scope", "detector", "representation", "parameter_value"])["flag_95"].mean().reset_index()
    for scope, block in sweep.groupby("scope"):
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, ((detector, rep), rows) in enumerate(block.groupby(["detector", "representation"])):
            ax.plot(range(len(rows)), rows["flag_95"], marker="o", markersize=3,
                    color=plotting.COLORBLIND_COLORS[i % 8], label=f"{detector} ({rep})")
        ax.axhline(0.05, color="#444444", linestyle="--", linewidth=0.8)
        ax.set(xlabel="parameter setting (sweep order)", ylabel="share of samples flagged", title=f"{scope}: detector sweeps")
        ax.legend(fontsize=6, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left")
        plotting.save_figure(fig, out / f"sweeps_{scope}")
    # 3. calibration check: achieved false-positive rate on fresh null data (target 5 %)
    rates = scores.drop_duplicates(["scope", "detector", "representation", "parameter_value", "calibration_stage"])
    rates = rates.dropna(subset=["false_positive_rate"])
    if len(rates):
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, (detector, rows) in enumerate(rates.groupby("detector")):
            ax.scatter(np.full(len(rows), i) + np.random.default_rng(0).uniform(-0.2, 0.2, len(rows)),
                       rows["false_positive_rate"], s=8, color=plotting.COLORBLIND_COLORS[i % 8])
        ax.axhline(0.05, color="#444444", linestyle="--")
        ax.set_xticks(range(rates["detector"].nunique()), sorted(rates["detector"].unique()), rotation=30, ha="right")
        ax.set(ylabel="false-positive rate on null data", title="Calibration check (target 5 %)")
        plotting.save_figure(fig, out / "calibration_check")


# ---------------------------------------------------------------- cross-assay concordance
def run_concordance(settings: Settings) -> pd.DataFrame:
    """Do lipidomics and metabolomics flag the same samples? (writes to results/cross_omics)."""
    out = settings.out("cross_omics", "outlier_concordance")
    lipid = pd.read_csv(settings.results_dir / "lipidomics" / "02_outliers" / "outlier_screen.csv")
    metabolite = pd.read_csv(settings.results_dir / "metabolomics" / "02_outliers" / "outlier_screen.csv")
    table = pd.concat([qc.outlier_flag_concordance(lipid, metabolite, s) for s in ("hotelling_t2", "normalized_dmodx")],
                      ignore_index=True)
    write_csv(table, out / "outlier_flag_concordance.csv")
    return table

