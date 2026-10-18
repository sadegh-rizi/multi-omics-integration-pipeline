"""Stage 8 - multi-omics integration (lipids + metabolites + proteins, same 60 samples).

Brain region is the strong, known signal (positive control) and MIA is the
characterised null (negative control), so the question for every method is:
does it find region, and does it correctly find *nothing* for MIA?

* :func:`build_blocks`            aligned sample x feature blocks + region-marker sanity check
* :func:`run_mofa`                MOFA+ on all 60 samples (seed-stable factors, shuffled-data control)
* :func:`run_mofa_within_region`  multi-group MOFA+ with region means removed: what do mice share within a region?
* :func:`run_benchmark`           region recovery by cross-validation; can one omics predict another within region?
* :func:`run_cross_omics_screen`  lipid summaries vs metabolites/proteins: correlation modules, shuffled-pairing null
* :func:`run_enrichment`          GO/Reactome over-representation of the proteins behind factors and modules
* :func:`run_covariate_map`       which covariates (sex, cage, date, mass, oligodendrocyte markers, MIA) track each axis
* :func:`run_inference_checks`    simulated null scores: why tests must be done at the litter level

Output: ``results/integration/``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from .. import config, plotting
from ..config import Settings
from ..integration import enrichment
from ..integration.common import spearman_matrix
from ..integration.inference import mouse_grouped_folds, null_calibration, treatment_score_test
from ..integration.mofa import fit_mofa, seed_stability, top_features
from ..lipids import build_lipid_summary_features
from ..preprocessing import complete_features
from ..stats.common import bh_adjust, r_squared
from .common import prepare, write_csv, write_json

COVARIATES = (config.REGION_COL, config.SEX_COL, config.GROUP_COL, config.CAGE_COL, config.COHORT_COL, "Tissue mass (mg)")


# ---------------------------------------------------------------- inputs
def build_blocks(settings: Settings, population: str = config.PRIMARY_POPULATION, *, write: bool = True
                 ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, pd.DataFrame]]:
    """Samples x features blocks for the three omics (complete features only, no new imputation)."""
    prepared = {view: prepare(settings, view, population) for view in config.VIEWS}
    metadata = prepared["lipidomics"].metadata
    samples = metadata["Sample"].tolist()
    blocks = {view: complete_features(p.values.loc[:, samples]).T for view, p in prepared.items()}
    features = {view: p.features for view, p in prepared.items()}
    if write:
        out = settings.out("integration", "00_inputs")
        for view, block in blocks.items():      # layout also read by the optional R scripts in r/
            write_csv(block.rename_axis("Sample").reset_index(), out / "blocks" / f"{view}.csv")
        write_csv(metadata, out / "sample_metadata.csv")
        write_csv(_marker_checks(blocks["proteomics"], features["proteomics"], metadata), out / "proteomics_marker_checks.csv")
        write_json({"population": population, "n_samples": len(samples),
                    "n_features": {v: b.shape[1] for v, b in blocks.items()},
                    "missing_values": "complete features only; proteomics delivered already imputed"}, out / "manifest.json")
    return blocks, metadata, features


def _marker_checks(proteins: pd.DataFrame, annotation: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Known region markers must be higher in their region (catches swapped sample labels)."""
    rows = []
    meta = metadata.set_index("Sample")
    for gene, region in config.REGION_MARKER_GENES:
        ids = annotation.index[annotation["Gene"].str.casefold().eq(gene.casefold())]
        if not len(ids):
            rows.append({"Gene": gene, "expected_region": region, "status": "not_detected"})
            continue
        values = proteins[ids[0]]
        differences = []
        for _, mouse in meta.groupby(config.MOUSE_COL):
            target = values[mouse.index[mouse[config.REGION_COL].eq(region)]]
            other = values[mouse.index[~mouse[config.REGION_COL].eq(region)]]
            differences.append(float(target.mean() - other.mean()))
        rows.append({"Gene": gene, "expected_region": region, "status": "detected",
                     "mean_target_minus_other_log2": np.mean(differences),
                     "direction_matches": bool(np.mean(differences) > 0)})
    return pd.DataFrame(rows)


def _associations(scores: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """R2 of every factor score with every design covariate."""
    rows = []
    for factor in scores:
        for covariate in COVARIATES:
            predictor = metadata[covariate]
            predictor = predictor.astype(str) if covariate == config.COHORT_COL else predictor
            rows.append({"Factor": factor, "Covariate": covariate, "r_squared": r_squared(scores[factor].to_numpy(), predictor)})
    return pd.DataFrame(rows)


def _treatment_tests(scores: pd.DataFrame, metadata: pd.DataFrame, settings: Settings, prefix: str) -> pd.DataFrame:
    """Litter-aware MIA test (Welch, mouse permutation, exact cage permutation) of every score column."""
    return pd.DataFrame([treatment_score_test(metadata, scores[c].reset_index(drop=True), score_name=f"{prefix}:{c}",
                                              n_mouse_permutations=settings.n_mouse_permutations) for c in scores])


# ---------------------------------------------------------------- MOFA+ on all samples
def run_mofa(settings: Settings) -> pd.DataFrame:
    """All-sample MOFA+: which factors are shared, stable, and what do they track?"""
    plotting.apply_plot_style()
    out = settings.out("integration", "01_mofa")
    blocks, metadata, _ = build_blocks(settings, write=False)
    # 1. Fit MOFA+ with several random seeds; keep the factors that recur in >= 80 % of them.
    fits = [fit_mofa(blocks, metadata, seed=settings.seed + s, n_factors=settings.mofa_factors,
                     iterations=settings.mofa_iterations) for s in range(settings.mofa_seeds)]
    stability, aligned = seed_stability(fits, by="scores")
    reference = fits[0]
    stable = [int(f.removeprefix("Factor")) - 1 for f in stability.loc[stability["stable"], "Factor"]]
    names = [f"Factor{k + 1}" for k in range(reference.factors.shape[1])]
    scores = pd.DataFrame(reference.factors, index=reference.samples, columns=names).loc[metadata["Sample"]]

    # 2. Variance explained per omics and factor; a factor "active" in >= 2 omics is shared.
    variance = pd.DataFrame(reference.variance[0], index=reference.views, columns=names)
    variance_long = variance.rename_axis("View").reset_index().melt(id_vars="View", var_name="Factor", value_name="variance_explained")
    variance_long["active_in_view"] = variance_long["variance_explained"].ge(0.02)
    variance_long["stable"] = variance_long["Factor"].isin(stability.loc[stability["stable"], "Factor"])
    write_csv(stability, out / "factor_stability.csv")
    write_csv(variance_long, out / "variance_explained.csv")
    write_csv(scores.rename_axis("Sample").reset_index(), out / "factor_scores.csv")
    # 3. What each factor tracks (R² with region, group, sex, cage, ...), its top features,
    #    and litter-aware tests of the stable factors against MIA.
    associations = _associations(scores, metadata.reset_index(drop=True).set_index(scores.index))
    write_csv(associations, out / "factor_associations.csv")
    median_weights = [np.nanmedian(np.stack([w[v] for w in aligned]), axis=0) for v in range(len(blocks))]
    write_csv(top_features(median_weights, blocks, stable, n=50), out / "top_features.csv")
    write_csv(_treatment_tests(scores[[names[k] for k in stable]], metadata, settings, "mofa"), out / "mia_tests.csv")

    # Negative control: shuffle samples independently in every block - shared factors should vanish.
    rng = np.random.default_rng(settings.seed)
    control = []
    for shuffle in range(settings.mofa_shuffles):
        shuffled = {v: b.iloc[rng.permutation(len(b))].set_axis(b.index) for v, b in blocks.items()}
        fit = fit_mofa(shuffled, metadata, seed=settings.seed + 100 + shuffle, n_factors=settings.mofa_factors,
                       iterations=settings.mofa_iterations)
        active = (fit.variance[0] >= 0.02).sum(axis=0)
        shared = fit.variance[0].sum(axis=0)[active > 1]
        control.append({"shuffle": shuffle, "n_shared_factors": int((active > 1).sum()),
                        "max_shared_variance": float(shared.max()) if len(shared) else 0.0})
    write_csv(pd.DataFrame(control), out / "shuffled_negative_control.csv")

    region_r2 = associations.loc[associations["Covariate"].eq(config.REGION_COL)].set_index("Factor")["r_squared"]
    shared = variance_long.loc[variance_long["active_in_view"]].groupby("Factor").size()
    write_json({"stable_factors": [names[k] for k in stable],
                "positive_control_region_factor": bool(any(region_r2.get(names[k], 0) >= 0.5 and shared.get(names[k], 0) >= 2 for k in stable)),
                "n_seeds": settings.mofa_seeds, "n_factors_requested": settings.mofa_factors}, out / "manifest.json")
    _mofa_figures(variance, scores, metadata, stable, out / "figures")
    return variance_long


def _mofa_figures(variance: pd.DataFrame, scores: pd.DataFrame, metadata: pd.DataFrame, stable: list[int], out) -> None:
    """Heatmap of % variance per omics and factor, and the Factor 1 vs Factor 2 score plot."""
    shown = variance.loc[:, variance.max(axis=0) >= 0.01]
    fig, ax = plt.subplots(figsize=(max(4, 0.7 * shown.shape[1] + 2), 2.8))
    image = ax.imshow(shown * 100, cmap="Blues", aspect="auto")
    ax.set_xticks(range(shown.shape[1]), shown.columns, rotation=45, ha="right")
    ax.set_yticks(range(shown.shape[0]), shown.index)
    for (i, j), value in np.ndenumerate(shown.to_numpy() * 100):
        ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="% variance explained")
    ax.set_title("MOFA+: variance explained per omics")
    plotting.save_figure(fig, out / "variance_explained")
    if len(scores.columns) >= 2:
        frame = scores.iloc[:, :2].set_axis(["PC 1", "PC 2"], axis=1)
        fig = plotting.scores_plot(frame, metadata, config.REGION_COL, title="MOFA+ Factor 1 vs Factor 2")
        fig.axes[0].set(xlabel="Factor 1", ylabel="Factor 2")
        plotting.save_figure(fig, out / "factor1_vs_factor2")


# ---------------------------------------------------------------- multi-group MOFA+ within region
def run_mofa_within_region(settings: Settings) -> pd.DataFrame:
    """Region means removed: shared variation between mice inside a region."""
    out = settings.out("integration", "02_mofa_within_region")
    blocks, metadata, _ = build_blocks(settings, write=False)
    fit_kwargs = dict(n_factors=settings.mofa_factors, iterations=settings.mofa_iterations, ard_factors=True)
    # Multi-group MOFA+: one group per region, each centred, so region differences are removed.
    # Sample order differs between groups, so factors are matched across seeds by their loadings.
    fits = [fit_mofa(blocks, metadata, seed=settings.seed + s, groups=config.REGIONS, **fit_kwargs)
            for s in range(settings.mofa_seeds)]
    stability, _ = seed_stability(fits, by="loadings", threshold=0.8)
    reference = fits[0]
    names = [f"Factor{k + 1}" for k in range(reference.factors.shape[1])]
    stable = [int(f.removeprefix("Factor")) - 1 for f in stability.loc[stability["stable"], "Factor"]]
    scores = pd.DataFrame(reference.factors, index=reference.samples, columns=names).loc[metadata["Sample"]]
    variance = pd.concat([pd.DataFrame(reference.variance[g], index=reference.views, columns=names)
                          .rename_axis("View").reset_index().melt(id_vars="View", var_name="Factor", value_name="variance_explained")
                          .assign(Region=region) for g, region in enumerate(config.REGIONS)], ignore_index=True)
    variance["stable"] = variance["Factor"].isin([names[k] for k in stable])
    write_csv(stability, out / "factor_stability.csv")
    write_csv(variance, out / "variance_explained_by_region.csv")
    write_csv(scores.rename_axis("Sample").reset_index(), out / "factor_scores.csv")
    write_csv(top_features(reference.weights, blocks, stable, fraction=0.05), out / "top_features_5pct.csv")
    write_csv(top_features(reference.weights, blocks, stable, fraction=0.10), out / "top_features_10pct.csv")
    write_csv(_associations(scores, metadata.set_index(scores.index)), out / "factor_associations.csv")
    if stable:
        write_csv(_treatment_tests(scores[[names[k] for k in stable]], metadata, settings, "mofa_within_region"), out / "mia_tests.csv")

    # Do factors found in one region reappear in another?  (single-region fits, loading correlation)
    from ..integration.mofa import match_factors, stacked_loadings
    region_fits = {}
    for region in config.REGIONS:
        mask = metadata[config.REGION_COL].eq(region).to_numpy()
        region_fits[region] = fit_mofa({v: b.loc[mask] for v, b in blocks.items()}, metadata.loc[mask].reset_index(drop=True),
                                       seed=settings.seed, groups=(region,), **fit_kwargs)
    replication = []
    for i, left in enumerate(config.REGIONS):
        for right in config.REGIONS[i + 1:]:
            _, strength = match_factors(stacked_loadings(region_fits[left].weights), stacked_loadings(region_fits[right].weights))
            replication.append({"pair": f"{left} vs {right}", "median_loading_correlation": float(np.nanmedian(strength)),
                                "n_matches_ge_0.5": int(np.sum(strength >= 0.5))})
    write_csv(pd.DataFrame(replication), out / "region_replication.csv")

    # Negative control: break the pairing between omics within each region.
    rng = np.random.default_rng(settings.seed)
    control = []
    for shuffle in range(settings.mofa_shuffles):
        shuffled = {v: b.copy() for v, b in blocks.items()}
        for region in config.REGIONS:
            positions = np.flatnonzero(metadata[config.REGION_COL].eq(region).to_numpy())
            for view in config.VIEWS[1:]:
                shuffled[view].iloc[positions] = shuffled[view].iloc[rng.permutation(positions)].to_numpy()
        fit = fit_mofa(shuffled, metadata, seed=settings.seed + 200 + shuffle, groups=config.REGIONS, **fit_kwargs)
        shared = (fit.variance >= 0.02).sum(axis=1).max(axis=0)
        control.append({"shuffle": shuffle, "n_factors": fit.factors.shape[1],
                        "n_factors_shared_by_2plus_views": int((shared >= 2).sum())})
    write_csv(pd.DataFrame(control), out / "pairing_shuffle_negative_control.csv")
    _within_region_figure(variance, out / "figures")
    return variance


def _within_region_figure(variance: pd.DataFrame, out) -> None:
    """Heatmap of % variance explained by the stable factors, per region and omics."""
    stable = variance.loc[variance["stable"]]
    if stable.empty:
        return
    table = stable.pivot_table(index=["Region", "View"], columns="Factor", values="variance_explained") * 100
    fig, ax = plt.subplots(figsize=(1.0 * table.shape[1] + 3, 0.35 * len(table) + 1.5))
    image = ax.imshow(table, cmap="Blues", aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(len(table)), [f"{r} - {v}" for r, v in table.index])
    for (i, j), value in np.ndenumerate(table.to_numpy()):
        ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="% variance")
    ax.set_title("Within-region MOFA+: stable factors")
    plotting.save_figure(fig, out / "variance_stable_factors")


# ---------------------------------------------------------------- benchmark
def _pca_fold(train: pd.DataFrame, test: pd.DataFrame, n: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Scale and fit PCA on the training samples only, then project the test samples."""
    centre, scale = train.mean(), train.std(ddof=1).replace(0, np.nan)
    keep = scale.notna()
    model = PCA(n_components=min(n, len(train) - 1, int(keep.sum())))
    return (model.fit_transform(((train - centre) / scale).loc[:, keep]),
            model.transform(((test - centre) / scale).loc[:, keep]))


def _block_pls(train: dict, test: dict, labels: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Equal-weight per-block PLS-DA scores (a Python stand-in for mixOmics block.spls)."""
    response = pd.get_dummies(labels, dtype=float).to_numpy()
    train_scores, test_scores = [], []
    for view in train:
        centre, scale = train[view].mean(), train[view].std(ddof=1).replace(0, np.nan)
        keep = scale.notna()
        model = PLSRegression(n_components=2, scale=False).fit(((train[view] - centre) / scale).loc[:, keep], response)
        a = model.transform(((train[view] - centre) / scale).loc[:, keep])
        b = model.transform(((test[view] - centre) / scale).loc[:, keep])
        train_scores.append((a - a.mean(0)) / a.std(0, ddof=1))
        test_scores.append((b - a.mean(0)) / a.std(0, ddof=1))
    return np.hstack(train_scores), np.hstack(test_scores)


def _nearest_centroid(train: np.ndarray, labels: pd.Series, test: np.ndarray, truth: pd.Series) -> float:
    """Accuracy of assigning each test sample to the nearest training-class centroid."""
    classes = list(pd.unique(labels))
    centroids = np.vstack([train[labels.to_numpy() == c].mean(0) for c in classes])
    predicted = [classes[int(np.argmin(((row - centroids) ** 2).sum(1)))] for row in test]
    return float(np.mean(np.asarray(predicted) == truth.to_numpy()))


def run_benchmark(settings: Settings) -> pd.DataFrame:
    """Region recovery (positive control) and within-region cross-omics predictability."""
    out = settings.out("integration", "03_benchmark")
    blocks, metadata, _ = build_blocks(settings, write=False)
    concatenated = pd.concat([b.add_prefix(f"{v}:") for v, b in blocks.items()], axis=1)
    # Positive control: can region be predicted for mice never seen in training?
    # Folds hold out whole mice (all three regions), so a mouse never helps predict itself.
    rows = []
    folds = mouse_grouped_folds(metadata, random_state=settings.seed)
    for method in ("concatenated_pca", "block_pls"):
        accuracy = []
        for train, test in folds:
            labels, truth = metadata[config.REGION_COL].iloc[train], metadata[config.REGION_COL].iloc[test]
            if method == "concatenated_pca":
                a, b = _pca_fold(concatenated.iloc[train], concatenated.iloc[test])
            else:
                a, b = _block_pls({v: x.iloc[train] for v, x in blocks.items()}, {v: x.iloc[test] for v, x in blocks.items()}, labels)
            accuracy.append(_nearest_centroid(a, labels, b, truth))
        rows.append({"method": method, "metric": "region accuracy (mouse-grouped CV)", "mean": np.mean(accuracy), "sd": np.std(accuracy, ddof=1)})
    recovery = pd.DataFrame(rows)
    write_csv(recovery, out / "region_recovery.csv")

    # Within region: can PCs of one omics predict PCs of another (leave one litter out)?
    rng = np.random.default_rng(settings.seed)
    predict = []
    for region in config.REGIONS:
        mask = metadata[config.REGION_COL].eq(region).to_numpy()
        cages = metadata.loc[mask, config.CAGE_COL].to_numpy()
        region_blocks = {v: b.loc[mask].reset_index(drop=True) for v, b in blocks.items()}
        cv = [(np.flatnonzero(cages != c), np.flatnonzero(cages == c)) for c in pd.unique(cages)]
        # Per omics and fold: PCA fitted on the training litters, test litter projected onto it.
        reps = {v: [_pca_fold(b.iloc[tr], b.iloc[te]) for tr, te in cv] for v, b in region_blocks.items()}
        for target in config.VIEWS:
            # Null: shuffle which mouse the target omics belongs to (one shuffle per replicate,
            # then the same folds), which breaks any real cross-omics link.
            null_reps = []
            for _ in range(max(3, settings.n_cross_omics_shuffles // 3)):
                shuffled = region_blocks[target].iloc[rng.permutation(mask.sum())].reset_index(drop=True)
                null_reps.append([_pca_fold(shuffled.iloc[tr], shuffled.iloc[te]) for tr, te in cv])
            for source in config.VIEWS:
                if source == target:
                    continue
                observed = _oof_r2(reps[source], reps[target])
                null = np.array([_oof_r2(reps[source], n) for n in null_reps])
                predict.append({"Region": region, "source": source, "target": target, "observed_oof_r2": observed,
                                "null_mean": null.mean(), "pairing_shuffle_p": (1 + np.sum(null >= observed)) / (1 + len(null))})
    write_csv(pd.DataFrame(predict), out / "within_region_predictability.csv")

    # Negative control: the first component of all omics together should not track MIA.
    pc1, _ = _pca_fold(concatenated, concatenated, n=2)
    scores = pd.DataFrame(pc1, columns=["Component1", "Component2"])
    write_csv(pd.DataFrame([treatment_score_test(metadata, scores["Component1"], score_name="concatenated PCA: Component1",
                                                 n_mouse_permutations=settings.n_mouse_permutations)]), out / "mia_negative_control.csv")
    return recovery


def _oof_r2(source: list, target: list) -> float:
    """Out-of-fold R² of predicting the target omics' PCs from the source omics' PCs (ridge regression)."""
    predicted, actual = [], []
    for (s_train, s_test), (t_train, t_test) in zip(source, target):
        predicted.append(Ridge(alpha=1.0).fit(s_train, t_train).predict(s_test))
        actual.append(t_test)
    predicted, actual = np.vstack(predicted), np.vstack(actual)
    total = np.square(actual - actual.mean(0)).sum()
    return float(1 - np.square(actual - predicted).sum() / total) if total else 0.0


# ---------------------------------------------------------------- cross-omics association screen
THRESHOLD = 0.6                  # |Spearman rho| counted as an association
MODULE_DISTANCE = 0.5            # correlation-distance cut for clustering association profiles


def _modules(correlation: np.ndarray, rows: list[str], columns: list[str]) -> list[dict]:
    """Bicluster the thresholded association matrix (rows = lipid summaries, columns = targets)."""
    active = np.abs(correlation) >= THRESHOLD
    r, c = np.flatnonzero(active.any(1)), np.flatnonzero(active.any(0))
    if len(r) < 2 or len(c) < 2:
        return []
    def cluster(profiles: np.ndarray) -> np.ndarray:
        """Average-linkage clusters of association profiles (correlation distance)."""
        distances = np.nan_to_num(pdist(profiles, metric="correlation"), nan=1.0)
        return fcluster(linkage(distances, method="average"), t=MODULE_DISTANCE, criterion="distance")
    # Cluster rows and columns separately; every row-cluster x column-cluster block with
    # at least two strong associations is a module.
    profiles = correlation[np.ix_(r, c)]
    row_labels, column_labels = cluster(profiles), cluster(profiles.T)
    modules = []
    for a in np.unique(row_labels):
        for b in np.unique(column_labels):
            rr, cc = r[row_labels == a], c[column_labels == b]
            block = active[np.ix_(rr, cc)]
            if block.sum() >= 2:
                modules.append({"module": len(modules) + 1, "lipid_features": ";".join(rows[i] for i in rr),
                                "target_features": ";".join(columns[i] for i in cc), "n_pairs": int(block.sum()),
                                "mean_abs_rho": float(np.abs(correlation[np.ix_(rr, cc)][block]).mean())})
    return modules


def _jaccard(a: str, b: str) -> float:
    """Overlap of two ';'-separated feature lists (|A and B| / |A or B|)."""
    a, b = set(a.split(";")), set(b.split(";"))
    return len(a & b) / len(a | b)


def run_cross_omics_screen(settings: Settings) -> pd.DataFrame:
    """Which lipid summaries co-vary with metabolites / proteins within a region, beyond chance?"""
    out = settings.out("integration", "04_cross_omics_screen")
    blocks, metadata, features = build_blocks(settings, write=False)
    annotation = features["lipidomics"].loc[blocks["lipidomics"].columns, ["Lipid Name", "Lipid Class"]]
    # Lipids are summarised first (class shares, chain-length bins, ...) to keep the number of pairs manageable.
    lipid_summaries, definitions = build_lipid_summary_features(blocks["lipidomics"], annotation)
    write_csv(definitions.reset_index(), out / "lipid_summary_definitions.csv")
    rng = np.random.default_rng(settings.seed)
    counts, modules = [], []
    for view in ("metabolomics", "proteomics"):
        for region in config.REGIONS:
            mask = metadata[config.REGION_COL].eq(region).to_numpy()
            left = lipid_summaries.loc[mask].reset_index(drop=True)
            right = blocks[view].loc[mask].reset_index(drop=True)
            # How many lipid-summary x target pairs have |rho| >= 0.6, compared with data whose
            # sample pairing between the two omics has been shuffled?
            correlation = spearman_matrix(left, right)
            observed = int(np.sum(np.abs(correlation) >= THRESHOLD))
            null = np.array([np.sum(np.abs(spearman_matrix(left, right.iloc[rng.permutation(len(right))].reset_index(drop=True))) >= THRESHOLD)
                             for _ in range(settings.n_cross_omics_shuffles)])
            counts.append({"Region": region, "target_view": view, "observed_pairs": observed, "null_mean": null.mean(),
                           "null_z": (observed - null.mean()) / null.std(ddof=1) if null.std(ddof=1) else np.nan,
                           "pairing_shuffle_p": (1 + np.sum(null >= observed)) / (1 + len(null))})
            # Modules, and how often each reappears when whole litters are resampled.
            found = _modules(correlation, list(left.columns), list(right.columns))
            cages = metadata.loc[mask, config.CAGE_COL].to_numpy()
            for module in found:
                hits = 0
                for _ in range(settings.n_cross_omics_bootstraps):     # cage bootstrap
                    chosen = np.concatenate([np.flatnonzero(cages == c) for c in rng.choice(pd.unique(cages), len(pd.unique(cages)))])
                    again = _modules(spearman_matrix(left.iloc[chosen], right.iloc[chosen]), list(left.columns), list(right.columns))
                    hits += any(min(_jaccard(module["lipid_features"], m["lipid_features"]),
                                    _jaccard(module["target_features"], m["target_features"])) >= 0.7 for m in again)
                modules.append({"Region": region, "target_view": view, **module,
                                "bootstrap_stability": hits / settings.n_cross_omics_bootstraps})
    modules = pd.DataFrame(modules)
    # A candidate finding is stable under the litter bootstrap and seen in at least two regions.
    if len(modules):
        modules["stable"] = modules["bootstrap_stability"].ge(0.7)
        modules["n_regions_matched"] = [
            1 + sum(1 for _, o in modules.loc[modules["target_view"].eq(m["target_view"]) & modules["Region"].ne(m["Region"])].iterrows()
                    if _jaccard(m["lipid_features"], o["lipid_features"]) >= 0.5 and _jaccard(m["target_features"], o["target_features"]) >= 0.5)
            for _, m in modules.iterrows()]
        modules["finding_candidate"] = modules["stable"] & modules["n_regions_matched"].ge(2)
    write_csv(pd.DataFrame(counts), out / "association_counts_vs_null.csv")
    write_csv(modules, out / "modules.csv")
    write_csv(lipid_summaries.rename_axis("Sample").reset_index(), out / "lipid_summary_values.csv")
    return pd.DataFrame(counts)


# ---------------------------------------------------------------- enrichment
def run_enrichment(settings: Settings) -> pd.DataFrame:
    """Hypergeometric GO / Reactome over-representation for the proteins behind factors and modules."""
    out = settings.out("integration", "05_enrichment")
    blocks, _, features = build_blocks(settings, write=False)
    annotation = features["proteomics"].loc[blocks["proteomics"].columns]
    # Background = all proteins in the integration block (not the whole genome).
    background = set(annotation.index)
    gene_of = {p: str(g).casefold() for p, g in annotation["Gene"].items() if pd.notna(g)}
    # Units to test: proteins behind each within-region MOFA+ factor and each cross-omics module.
    units = []
    for fraction in ("5pct", "10pct"):
        path = settings.results_dir / "integration" / "02_mofa_within_region" / f"top_features_{fraction}.csv"
        if path.exists() and path.stat().st_size > 30:
            top = pd.read_csv(path)
            for factor, rows in top.loc[top["View"].eq("proteomics")].groupby("Factor"):
                units.append((f"within-region {factor} (top {fraction})", set(rows["Feature"].astype(str))))
    modules_path = settings.results_dir / "integration" / "04_cross_omics_screen" / "modules.csv"
    if modules_path.exists() and modules_path.stat().st_size > 1:
        modules = pd.read_csv(modules_path)
        for _, m in modules.loc[modules["target_view"].eq("proteomics")].iterrows():
            units.append((f"{m['Region']} module {m['module']}", set(str(m["target_features"]).split(";"))))
    # Hypergeometric over-representation of every gene set (GMT files) in every unit; BH within unit.
    rows, references = [], []
    for path in sorted((settings.data_dir / config.REFERENCE_DIR).glob("*.gmt*")):
        collection = "Reactome" if "reactome" in path.name.casefold() else "GO"
        gene_sets = enrichment.parse_gmt(path)
        references.append({"collection": collection, "file": path.name, "version": enrichment.infer_gmt_version(path),
                           "n_gene_sets": len(gene_sets)})
        for term, members in gene_sets.items():
            members = {m.casefold() for m in members}
            term_proteins = {p for p in background if p.casefold() in members or gene_of.get(p) in members}
            if not term_proteins:
                continue
            for unit, selected in units:
                result = enrichment.hypergeometric_enrichment(background, selected & background, term_proteins)
                rows.append({"unit": unit, "collection": collection, "term": term, "n_term": result.gene_set_size,
                             "n_selected": result.selected_size, "n_overlap": result.overlap_size, "p_value": result.raw_p_value,
                             "overlap_genes": ";".join(sorted(annotation.loc[list(selected & term_proteins), "Gene"].astype(str)))})
    table = pd.DataFrame(rows)
    if len(table):
        table["q_value"] = table.groupby("unit")["p_value"].transform(bh_adjust)
        table = table.sort_values(["unit", "p_value"])
    write_csv(table, out / "protein_enrichment.csv")
    write_csv(pd.DataFrame(references), out / "reference_files.csv")
    return table


# ---------------------------------------------------------------- covariate map
def run_covariate_map(settings: Settings) -> pd.DataFrame:
    """R2 of each integration axis with sex, cage, date, mass, sample order, oligodendrocyte markers and MIA."""
    out = settings.out("integration", "06_covariate_map")
    blocks, metadata, features = build_blocks(settings, write=False)
    axes = pd.read_csv(settings.results_dir / "integration" / "02_mofa_within_region" / "factor_scores.csv").set_index("Sample")
    stable = pd.read_csv(settings.results_dir / "integration" / "02_mofa_within_region" / "factor_stability.csv")
    axes = axes[stable.loc[stable["stable"], "Factor"]].loc[metadata["Sample"]]
    genes = features["proteomics"].loc[blocks["proteomics"].columns, "Gene"]
    markers = genes.index[genes.isin(config.OLIGODENDROCYTE_GENES)]
    regions = metadata[config.REGION_COL].to_numpy()
    oligo = blocks["proteomics"][markers].groupby(regions).transform(lambda x: (x - x.mean()) / x.std(ddof=1)).mean(axis=1)
    covariates = metadata.set_index("Sample").assign(oligodendrocyte_score=oligo.to_numpy(),
                                                     date_numeric=pd.to_datetime(metadata[config.DATE_COL]).map(pd.Timestamp.toordinal).to_numpy())
    rows = []
    for axis in axes:
        for name, column in ((config.SEX_COL, config.SEX_COL), (config.CAGE_COL, config.CAGE_COL), ("Date collected", "date_numeric"),
                             ("Tissue mass (mg)", "Tissue mass (mg)"), ("Sample number", "Sample number"),
                             ("oligodendrocyte_score", "oligodendrocyte_score"), ("MIA", config.GROUP_COL)):
            rows.append({"Axis": axis, "Covariate": name, "r_squared": r_squared(axes[axis].to_numpy(), covariates[column])})
    table = pd.DataFrame(rows)
    write_csv(table, out / "axis_covariate_r2.csv")
    if len(axes.columns):
        write_csv(_treatment_tests(axes.reset_index(drop=True), metadata, settings, "axis"), out / "axis_mia_tests.csv")
        pivot = table.pivot(index="Axis", columns="Covariate", values="r_squared")
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(pivot) + 1.5))
        image = ax.imshow(pivot, cmap="Purples", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=40, ha="right")
        ax.set_yticks(range(len(pivot)), pivot.index)
        fig.colorbar(image, ax=ax, label="R²")
        ax.set_title("What do the within-region axes track?")
        plotting.save_figure(fig, out / "figures" / "covariate_map")
    return table


# ---------------------------------------------------------------- inference calibration
def run_inference_checks(settings: Settings) -> pd.DataFrame:
    """Shuffle-unit simulation: mouse-level tests are anti-conservative when litters differ."""
    out = settings.out("integration", "09_inference_checks")
    _, metadata, _ = build_blocks(settings, write=False)
    region_meta = metadata.loc[metadata[config.REGION_COL].eq(config.REGIONS[0])].reset_index(drop=True)
    simulation = null_calibration(region_meta, n_datasets=max(20, settings.n_power_datasets),
                                  n_mouse_permutations=min(999, settings.n_mouse_permutations), random_state=settings.seed)
    write_csv(simulation, out / "null_calibration.csv")
    summary = simulation.groupby("litter_effect")[["mouse_permutation_p", "cage_permutation_p"]].agg(lambda p: float(np.mean(p < 0.05)))
    summary = summary.rename(columns=lambda c: c.removesuffix("_p") + "_false_positive_rate").reset_index()
    write_csv(summary, out / "false_positive_rates.csv")
    folds = mouse_grouped_folds(metadata, random_state=settings.seed)
    write_csv(pd.DataFrame([{"fold": i, "n_train": len(a), "n_test": len(b)} for i, (a, b) in enumerate(folds)]), out / "grouped_folds.csv")
    return summary
