"""One module per analysis step; each ``run(settings, ...)`` writes tables and figures to results/.

qc, noise_floor, feature_modules   1   data quality and structure
outliers                           2   T² screen, calibrated ensemble, assay concordance
differential                       3   group x sex model per region and population
sensitivity                        4   specification curve, imputation, PQN, litter-aware models
region_persistence                 5   do regional profiles persist after MIA?
asca                               6   ASCA+ with permutation tests
effect_sizes                       7   effect sizes, lipid composites, protein sets
integration                        8   MOFA+, benchmark, cross-omics screen, enrichment
vendor_figures                     9   the facility's report figures + permutation tests
common                                 shared loading/preprocessing (``prepare``) and writers
"""
