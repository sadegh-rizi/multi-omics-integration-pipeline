"""Stage 1c - structure among features: feature-space PCA and co-abundance modules.

Within each region (complete features only):

* **feature-space PCA** - PCA with features as observations, with bootstrap and
  split-half checks of whether the loadings are stable enough to interpret;
* **co-abundance modules** - hierarchical clustering of the signed Spearman
  correlation between features, with bootstrap co-membership stability and a
  permuted-data null for how large modules get by chance.

The result in the study: structure is real (modules far above the null) but
not resolvable at n = 20 per region.  Output: ``results/<omics>/01_qc/feature_modules/``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config, plotting
from ..config import Settings
from ..multivariate.feature_modules import (
    assign_hierarchical_modules,
    bootstrap_feature_loadings,
    bootstrap_module_comembership,
    fit_feature_margin_svd,
    fit_sample_space_loadings,
    loading_annotation_composition,
    loading_stability_summary,
    module_annotation_composition,
    module_stability_summary,
    permuted_null_module_sizes,
    spearman_correlation,
    split_half_loading_stability,
)
from ..preprocessing import complete_features
from .common import feature_class, prepare, region_samples, write_csv

SPLIT_HALF_FLOOR = 0.50          # loadings interpretable only if split-half correlation >= 0.5
MODULE_STABILITY_FLOOR = 0.80    # module reportable if members co-cluster in >= 80 % of bootstraps
MODULE_MIN_SIZE = 3
MODULE_DISTANCE_CUT = 0.30


def run(settings: Settings, omics: str = "metabolomics") -> pd.DataFrame:
    """Feature PCA and module analysis per region; returns one summary row per region."""
    plotting.apply_plot_style()
    out = settings.out(omics, "01_qc", "feature_modules")
    prepared = prepare(settings, omics)
    classes = feature_class(prepared.features)
    summary = []
    for number, region in enumerate(config.REGIONS):
        seed = settings.seed + 10 * number
        complete = complete_features(prepared.values.loc[:, region_samples(prepared.metadata, region)])
        tag = region.lower()

        # 1. Feature-space PCA and how stable its loadings are (bootstrap + split half).
        margin_loadings, _, margin_scree = fit_feature_margin_svd(complete)
        write_csv(margin_scree.assign(Region=region), out / f"feature_margin_scree_{tag}.csv")
        loadings, scree = fit_sample_space_loadings(complete, scale_features=True)
        write_csv(loadings.join(classes).rename_axis("Feature").reset_index(), out / f"feature_loadings_{tag}.csv")
        write_csv(loading_annotation_composition(loadings, classes.loc[loadings.index], annotation_name="Class"),
                  out / f"loading_class_composition_{tag}.csv")

        bootstrap = bootstrap_feature_loadings(complete, n_bootstraps=settings.n_feature_bootstraps, random_state=seed, n_components=2)
        stability = loading_stability_summary(bootstrap, complete.index)
        split = split_half_loading_stability(complete, random_state=seed + 1, n_components=2)
        split["interpretation_allowed"] = split["split_half_loading_correlation"].ge(SPLIT_HALF_FLOOR)
        write_csv(stability.assign(Region=region), out / f"loading_bootstrap_stability_{tag}.csv")
        write_csv(split.assign(Region=region), out / f"loading_split_half_{tag}.csv")

        # 2. Co-abundance modules, their bootstrap stability, and a permuted null for module size.
        correlation = spearman_correlation(complete)
        modules = assign_hierarchical_modules(correlation, distance_cut=MODULE_DISTANCE_CUT, signed=True)
        co_membership = bootstrap_module_comembership(complete, n_bootstraps=settings.n_module_bootstraps,
                                                      random_state=seed + 2, distance_cut=MODULE_DISTANCE_CUT, signed=True)
        module_summary = module_stability_summary(modules, co_membership, stability_floor=MODULE_STABILITY_FLOOR,
                                                  minimum_size=MODULE_MIN_SIZE)
        null_sizes = permuted_null_module_sizes(complete, n_permutations=settings.n_module_nulls, random_state=seed + 3,
                                                distance_cut=MODULE_DISTANCE_CUT, signed=True, minimum_size=MODULE_MIN_SIZE)
        write_csv(module_summary.assign(Region=region), out / f"module_stability_{tag}.csv")
        write_csv(module_annotation_composition(modules, classes.loc[modules.index], annotation_name="Class").assign(Region=region),
                  out / f"module_class_composition_{tag}.csv")
        write_csv(modules.rename("module").rename_axis("Feature").reset_index().assign(Region=region), out / f"module_assignments_{tag}.csv")
        write_csv(null_sizes.assign(Region=region), out / f"module_null_sizes_{tag}.csv")

        # 3. Is the largest real module bigger than the largest modules found in permuted data?
        largest_null = null_sizes.groupby("null_permutation")["n_features"].max()
        largest = int(modules.value_counts().max())
        summary.append({"Region": region, "n_complete_features": len(complete),
                        "feature_margin_pc1_variance": float(margin_scree["variance_explained"].iloc[0]),
                        "split_half_pc1": float(split["split_half_loading_correlation"].iloc[0]),
                        "n_modules": int(modules.nunique()), "n_reportable_modules": int(module_summary["reportable"].sum()),
                        "largest_module": largest, "largest_null_module_median": float(largest_null.median()),
                        "largest_module_z_vs_null": float((largest - largest_null.mean()) / largest_null.std(ddof=1))
                        if largest_null.std(ddof=1) > 0 else np.nan})
        _heatmap(correlation, modules, region, out / "figures" / f"correlation_heatmap_{tag}")
    summary = pd.DataFrame(summary)
    write_csv(summary, out / "summary.csv")
    return summary


def _heatmap(correlation: pd.DataFrame, modules: pd.Series, region: str, path) -> None:
    """Correlation heatmap with features ordered by module (at most 150 features shown)."""
    order = modules.sort_values().index
    if len(order) > 150:                   # keep the figure readable: largest modules first
        big = modules.value_counts().index
        order = modules.loc[modules.isin(big[:10])].sort_values().index[:150]
    fig, ax = plt.subplots(figsize=(6, 5.2))
    image = ax.imshow(correlation.loc[order, order], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set(xticks=[], yticks=[], title=f"{region}: Spearman correlation, features ordered by module")
    fig.colorbar(image, ax=ax, label="rho")
    plotting.save_figure(fig, path)
