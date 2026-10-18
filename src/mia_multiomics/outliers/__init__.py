"""Sample-outlier detection.

screen       classical PCA screen: Hotelling T², robust Mahalanobis, DModX
detectors    six detector families (kNN, LOF, isolation forest, kernel PCA, Ledoit-Wolf, ...)
calibration  limits from simulated outlier-free data, with false-positive checks
ensemble     stability across parameter settings and the corroboration rule
"""
