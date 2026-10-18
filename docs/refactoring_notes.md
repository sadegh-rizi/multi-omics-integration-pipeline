# Refactoring notes: from the thesis repository to this pipeline

This repository is a cleaned-up copy of the analysis code from the thesis
repository (`LPJ_lipidomics`). The scientific methods are the same. The
organisation is new, and the copy has no real data.

## What changed, and why

| Before | After | Why |
|---|---|---|
| ~60 runner scripts under `runners/`, each reading the workbook, preprocessing and writing its own tree | 11 stage modules under `stages/`, all built on one `inputs` + `preprocessing` layer | The same loading and filtering code had been copied into many runners, with small differences |
| Separate lipidomics (`preprocessing.py`) and metabolomics (`metabolomics_preprocessing.py`) chains | One `preprocessing.preprocess(data, samples, ...)` for both | The chains were identical except for the detection floor and per-mode normalisation, which are now parameters |
| Stages handed work on through CSV files (`09_region_persistence/.../region_input_matrix.csv` read by ASCA, the integration inputs, ...) | Every stage calls `prepare(settings, omics, population)`, which is fast and cached | Removes hidden ordering dependencies and stale-file bugs. Only the outlier-screen table and the MOFA outputs are handed on |
| Benjamini-Hochberg written 5 times, R² 4 times, Welch t and empirical p in several places | `stats/common.py` | One tested implementation of each |
| `src/*.py` compatibility shims for old flat import paths (40 files) | removed | Not needed in a fresh repository |
| Settings spread across `config.py` and module constants (permutation counts, seeds, thresholds) | `config.py`, with a `Settings` dataclass (`fast` / `full`) | The notebook needs a quick run; the thesis needs the full one |
| Real-data specifics hard-coded (`"MB2_088-"` sample IDs, `N_DETECTED_PROTEINS = 3821`, date `2024-06-11`, vendor-excluded sample 26, expected sample counts 55/58) | Derived from the data or set in `config.py` | Lets the same code run on the toy data and the real data |
| Defensive write-then-reread assertions after almost every CSV | removed; validation stays in the readers (`inputs.check_design`) | Fail loudly on bad input once, at the boundary |
| `outliers/plots.py` (690 lines of plot helpers) and many near-identical PCA plotting functions | `plotting.py` (shared plot types used by the stages) and `explore/` (every other figure, drawn on demand) | One plot type per figure kind; the pipeline writes only the figures its story needs |
| Each figure saved as a PDF plus two PNGs | One PNG | Smaller output tree |
| PyShiny explorer app and deployment bundle | left out | Not part of the analysis pipeline |

## Where each original script went

| Original runner / module | New location |
|---|---|
| `io/lipidomics.py`, `io/metabolomics.py`, `io/proteomics.py`, `io/study_design.py`, `paths.py` | `inputs.py` |
| `preprocessing.py`, `metabolomics_preprocessing.py` | `preprocessing.py` |
| `qc/variance.py`, `qc/missingness.py`, `qc/cross_modality.py` | `qc.py` |
| `qc/abundance_similarity.py`, `noisyr.py` | `noise_floor.py`, `noisyr.py` |
| `aggregation.py`, `lipid_features.py` | `lipids.py` |
| `annotation.py` | `lipid_databases.py` (optional, needs internet) |
| `multivariate/pca.py`, `loadings.py`, `pca_metadata.py` | `multivariate/pca.py` |
| `multivariate/features.py` | `multivariate/feature_modules.py` |
| `statistics/variance.py` | `stats/permanova.py` |
| `integration/common.py`, `runners/run_integration_tests.py` | `integration/common.py`, `integration/inference.py` |
| `run_mia_vs_control.py`, `run_qc_filters.py`, `run_qc_completion.py`, `run_missingness_report.py`, `run_metabolomics_missingness.py`, `run_metabolomics_pca.py`, `run_loading_composition.py` | `stages/qc.py` |
| `run_abundance_similarity.py`, `run_noisyr_crosscheck.py`, `run_noisyr_parameter_sensitivity.py` | `stages/noise_floor.py` |
| `run_metabolomics_feature_pca.py` | `stages/feature_modules.py` (now for both omics) |
| `run_outlier_screen.py`, `run_pca_ellipse.py`, `run_metabolomics_hotelling.py`, `run_metabolomics_outlier_screen.py` | `stages/outliers.py: run_screen, run_concordance` |
| `run_outlier_ensemble.py`, `run_knn_outlier_sweep.py`, `run_isolation_forest.py`, `run_ensemble_calibration.py`, `run_mahalanobis_calibration_fix.py`, `run_outlier_diagnostic_plots.py` | `stages/outliers.py: run_ensemble` |
| `run_pipeline_variant.py`, `run_full_60_differential.py`, `pipeline_stages.py` | `stages/differential.py` |
| `run_specification_curve.py`, `run_pipeline_rf.py`, `run_pqn_sensitivity.py`, `run_cage_aware_sensitivity.py` | `stages/sensitivity.py` |
| `run_region_persistence.py` | `stages/region_persistence.py` |
| `run_asca.py` | `stages/asca.py` |
| `run_effect_size_estimation.py` | `stages/effect_sizes.py` |
| `run_integration_inputs.py`, `run_integration_mofa.py`, `run_mofa_within_region.py`, `run_integration_benchmark.py`, `run_cross_omics_screen.py`, `run_integration_enrichment.py`, `run_covariate_map.py` | `stages/integration.py` (+ `integration/mofa.py`) |
| `run_vendor_fidelity.py`, `multivariate/vendor.py` | `stages/vendor_figures.py`, `multivariate/vendor.py` |
| `run_known_broken_fixes.py`, `check_known_broken_fixes.py` | dropped: one-off regression checks against the real result tree |
| `R/*.R`, `src/differential_moderated.R` | `r/` (optional) |

## Figures of the original repository

The pipeline stages write the figures that the walkthrough uses. Every other
figure type of the thesis repository is a function in `mia_multiomics.explore`,
drawn by `notebooks/supplementary_figures.ipynb` from user-chosen settings
(sample exclusions, thresholds, scaling) instead of from a fixed population.

| Original figure | `explore` function |
|---|---|
| `qc_injection_drift`, Tech-Mix CV histogram | `techmix_drift`, `techmix_cv` |
| `missingness_per_lipid_histogram`, `missingness_per_sample_histogram`, `missingness_by_region_and_group`, `missingness_heatmap(_by_region)` | `missingness_per_feature`, `missingness_per_sample`, `missingness_by_region_and_group`, `missingness_heatmap` |
| `abundance_similarity_<region>`, `abundance_similarity_summary` | `abundance_similarity` |
| `scree_<scope>`, `scree_grouped_<scope>`, `scree_attribution_<scope>` | `scree`, `scree_grouped`, `scree_attribution` |
| `scores_<scope>_<factor>` (incl. `tissue_mass_binned`), `scores_pooled_region_group_composite` | `pca_scores`, `region_group_composite` |
| `loadings_lollipop`, `loadings_scatter`, `loadings_heatmap`, `loadings_class_stacked`, `loadings_cumulative_share`, `loadings_feature_stacked(_top20)` | `loadings_lollipop`, `loadings_scatter`, `loadings_heatmap`, `loadings_class_composition`, `loadings_cumulative_share`, `loadings_top_features` |
| `t2_range_plot`, `dmodx_plot`, `influence_t2_vs_dmodx`, `hotelling_fulldata` | `t2_plot`, `dmodx_plot`, `influence_plot`, `pca_hotelling` |
| `concordance_hotelling_t2`, `concordance_normalized_dmodx` | `outlier_concordance` |
| ensemble `stability_heatmap_<scope>`, `corroboration_scatter_all_scopes`, `percentile_strip_all_scopes` | `ensemble_stability`, `ensemble_corroboration`, `ensemble_percentiles` |
| `volcano_<region>`, `p_value_histograms`, `small_p_value_counts` | `volcano`, `volcano_grid`, `p_value_histograms`, `small_p_counts` |
| `feature_correlation_heatmap`, `feature_view_loadings`, `feature_loading_stability`, `module_comembership_stability` | `feature_correlation_heatmap`, `feature_loadings_view`, `feature_loading_stability`, `module_stability` |
| cage `scores_<region>_cage`, `scores_<region>_treatment_sex_cage_full`, `within_cage_distance` | `pca_by_cage`, `cage_panels`, `within_cage_distance` |
| `asca_composite` (variance part) | `asca_overview` |
| vendor-matched `heatmap_<region>`, `plsda_<region>` | `vendor_heatmap`, `plsda_scores` (with the permutation null drawn) |
| `hippocampus_myelin_lipid_vs_protein` | `myelin_lipid_vs_protein` (any region) |
| new | `feature_funnel`, `prevalence_curve`, `signal_to_blank` (to choose thresholds), `feature_profile`, `effect_forest`, `class_effects`, `region_differences` |

Not ported: the noisyR window-sensitivity grid and noisyR-vs-manual plots
(they need R; `stages/noise_floor.py --with-r` runs the cross-check) and the
"manuscript"/"presentation" duplicates of each figure.

## Bugs found while refactoring

* The pooled out-of-fold R² in the integration benchmark broadcast a
  1-D prediction against a 2-D target when only one component was predicted.
  This had no effect at the 5 components actually used, but the new
  `stages/integration._oof_r2` stacks predictions explicitly, and a test covers
  the one-component case.
* The integration enrichment hard-coded the size of the protein background. The
  background is now the set of proteins in the integration block.
* The null of the within-region cross-omics predictability benchmark shuffled
  the target omics separately for the training and the test rows of each fold,
  so one sample could appear on both sides of a fold. It now shuffles once per
  null replicate and then applies the folds (`stages/integration.run_benchmark`).
* The ASCA+ summary figure labelled its score panels "effect + residual"; they
  show the effect scores only, and are now labelled that way.
* A few stale or duplicated pieces were removed: a second hand-written
  Benjamini-Hochberg in `stats/permanova.py` (now `bh_adjust`), a copy of the
  model formula in `stats/diagnostics.py`, a duplicated class-composition
  function in `multivariate/feature_modules.py`, two identical calibration
  sweeps in `outliers/calibration.py` (now one `_calibrate_sweep`), a
  hard-coded "four multi-pup cages" caveat, and an out-of-date note about
  unresolved litter identity in `stats/power.py`.

## Things deliberately kept as they were

The statistical cores were moved without changing their logic: the OLS
differential model, the permutation null and effective-test estimate, cage
statistics, region contrasts with split-half disattenuation, ASCA+ coding and
permutation schemes, the outlier detectors and their calibration, and the
feature-module statistics. Their original unit tests were ported. The library
tests that still apply pass unchanged apart from import paths.
