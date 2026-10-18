"""Interactive companion to the pipeline (used by ``notebooks/supplementary_figures.ipynb``).

Build a :class:`View` from your own choices - which samples to exclude,
prevalence and blank thresholds, normalisation, scaling, FDR level - and draw
any of the figures of the original thesis repository from it:

    from mia_multiomics import explore
    view = explore.build_view(settings, explore.Options(exclude=[10, 45]))
    explore.volcano(view, region="Cortex")
    explore.loadings_heatmap(view, scope="pooled")

Every figure function returns a Matplotlib figure and prints the settings it
was drawn with underneath.
"""

from ._common import save_figures
from .differential_figures import (
    class_effects,
    effect_forest,
    feature_profile,
    hit_counts,
    p_value_histograms,
    small_p_counts,
    top_hits,
    volcano,
    volcano_grid,
)
from .outlier_figures import (
    EnsembleResults,
    dmodx_plot,
    ensemble_corroboration,
    ensemble_percentiles,
    ensemble_results,
    ensemble_stability,
    influence_plot,
    outlier_concordance,
    outlier_table,
    suggest_exclusions,
    t2_plot,
)
from .pca_figures import (
    loadings_class_composition,
    loadings_cumulative_share,
    loadings_heatmap,
    loadings_lollipop,
    loadings_scatter,
    loadings_top_features,
    pca_hotelling,
    pca_scores,
    region_group_composite,
    scree,
    scree_attribution,
    scree_grouped,
)
from .qc_figures import (
    abundance_similarity,
    feature_funnel,
    missingness_by_region_and_group,
    missingness_heatmap,
    missingness_per_feature,
    missingness_per_sample,
    prevalence_curve,
    signal_to_blank,
    techmix_cv,
    techmix_drift,
)
from .structure_figures import (
    asca_overview,
    cage_panels,
    feature_correlation_heatmap,
    feature_loading_stability,
    feature_loadings_view,
    module_stability,
    myelin_lipid_vs_protein,
    pca_by_cage,
    plsda_scores,
    region_differences,
    vendor_heatmap,
    within_cage_distance,
)
from .view import SCOPES, Options, View, build_view, resolve_exclusions

__all__ = [
    "EnsembleResults",
    "Options",
    "SCOPES",
    "View",
    "abundance_similarity",
    "asca_overview",
    "build_view",
    "cage_panels",
    "class_effects",
    "dmodx_plot",
    "effect_forest",
    "ensemble_corroboration",
    "ensemble_percentiles",
    "ensemble_results",
    "ensemble_stability",
    "feature_correlation_heatmap",
    "feature_funnel",
    "feature_loading_stability",
    "feature_loadings_view",
    "feature_profile",
    "hit_counts",
    "influence_plot",
    "loadings_class_composition",
    "loadings_cumulative_share",
    "loadings_heatmap",
    "loadings_lollipop",
    "loadings_scatter",
    "loadings_top_features",
    "missingness_by_region_and_group",
    "missingness_heatmap",
    "missingness_per_feature",
    "missingness_per_sample",
    "module_stability",
    "myelin_lipid_vs_protein",
    "outlier_concordance",
    "outlier_table",
    "p_value_histograms",
    "pca_by_cage",
    "pca_hotelling",
    "pca_scores",
    "plsda_scores",
    "prevalence_curve",
    "region_differences",
    "region_group_composite",
    "resolve_exclusions",
    "save_figures",
    "scree",
    "scree_attribution",
    "scree_grouped",
    "signal_to_blank",
    "small_p_counts",
    "suggest_exclusions",
    "t2_plot",
    "techmix_cv",
    "techmix_drift",
    "top_hits",
    "vendor_heatmap",
    "volcano",
    "volcano_grid",
    "within_cage_distance",
]
