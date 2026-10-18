import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import LocalOutlierFactor

from mia_multiomics.outliers.calibration import (
    calibrate_ensemble,
    calibrate_isolation_forest,
    calibrate_knn_sweep,
    calibrate_ledoit_wolf_sweep,
    calibrate_lof_sweep,
    score_to_percentile,
)
from mia_multiomics.outliers.detectors import (
    isolation_forest_score,
    isolation_forest_seed_scores,
    knn_mean_distance,
    ledoit_wolf_mahalanobis_sweep,
    local_outlier_factor,
    local_outlier_factor_sweep,
)
from mia_multiomics.outliers.ensemble import apply_stability_rule, combine_percentiles, stability_fraction


def test_knn_recovers_planted_outlier_and_soft_guard_fires():
    rng = np.random.default_rng(31)
    values = pd.DataFrame(rng.normal(size=(20, 3)), index=[f"S{i}" for i in range(20)])
    values.loc["S19"] += 8
    assert knn_mean_distance(values, parameter=4).idxmax() == "S19"
    with pytest.raises(ValueError, match="1%"):
        knn_mean_distance(values, parameter=4, weighting="soft", eigenvalues=[1, 0.001, 0.0001])


def test_isolation_forest_seed_average_is_deterministic():
    values = pd.DataFrame(np.random.default_rng(4).normal(size=(15, 3)))
    first = isolation_forest_score(values, parameter=[1, 2, 3], n_estimators=20)
    second = isolation_forest_score(values, parameter=[1, 2, 3], n_estimators=20)
    pd.testing.assert_series_equal(first, second)
    seed_scores = isolation_forest_seed_scores(values, seeds=[1, 2, 3], n_estimators=20, n_jobs=1)
    pd.testing.assert_series_equal(first, seed_scores.mean(axis=1).rename("isolation_forest"))
    parallel = isolation_forest_score(values, parameter=[1, 2, 3], n_estimators=20, n_jobs=2)
    pd.testing.assert_series_equal(first, parallel)


def test_percentiles_are_monotone_and_stability_requires_corrobation():
    percentiles = score_to_percentile(pd.Series([1.0, 2.0, 3.0]), pd.Series(np.arange(100)))
    assert percentiles.is_monotonic_increasing
    scores = pd.DataFrame({
        "scope": ["pooled", "pooled", "pooled", "pooled"],
        "calibration_stage": ["trimmed", "trimmed", "trimmed", "trimmed"],
        "Sample": ["A", "A", "B", "B"],
        "detector": ["knn_isotropic", "knn_isotropic", "mahalanobis", "mahalanobis"],
        "flag_95": [True, True, True, True],
    })
    stable = stability_fraction(scores).reset_index()
    verdict = apply_stability_rule(stable)
    assert not bool(verdict.loc["A", "candidate_outlier"])
    combined = combine_percentiles(pd.DataFrame({"mahalanobis": [0.9], "knn": [0.8], "isolation_forest": [0.7], "lof": [0.6], "kernel_pca": [0.5]}))
    assert combined.loc[0, "trinity_mean"] == pytest.approx(0.8)
    assert combined.loc[0, "trinity_max"] == pytest.approx(0.9)


def test_stability_fraction_does_not_blend_scopes():
    scores = pd.DataFrame({
        "scope": ["pooled", "pooled", "cortex", "cortex"],
        "calibration_stage": ["trimmed", "trimmed", "trimmed", "trimmed"],
        "Sample": ["A", "A", "A", "A"],
        "detector": ["lof", "lof", "lof", "lof"],
        "flag_95": [True, False, True, True],
    })
    stability = stability_fraction(scores).reset_index()
    fractions = stability.set_index(["scope", "calibration_stage", "Sample", "detector"])["stability_fraction"]
    assert fractions["pooled", "trimmed", "A", "lof"] == pytest.approx(0.5)
    assert fractions["cortex", "trimmed", "A", "lof"] == pytest.approx(1.0)


def test_regional_supporting_lof_is_displayed_but_not_counted():
    stability = pd.DataFrame({
        "scope": ["cortex", "cortex", "cortex"],
        "calibration_stage": ["trimmed", "trimmed", "trimmed"],
        "Sample": ["A", "A", "A"],
        "detector": ["lof_supporting", "knn_isotropic", "mahalanobis"],
        "stability_fraction": [1.0, 1.0, 1.0],
    })
    verdict = apply_stability_rule(stability)
    assert verdict.loc["A", "stable_detector_count"] == 2
    assert verdict.loc["A", "stable_non_covariance_count"] == 1


def test_ensemble_calibration_returns_limits_and_held_out_rates():
    values = pd.DataFrame(np.random.default_rng(8).normal(size=(12, 3)))
    scorers = {
        "mahalanobis": lambda frame: frame.pow(2).sum(axis=1),
        "knn": lambda frame: frame.abs().sum(axis=1),
        "isolation_forest": lambda frame: frame.max(axis=1),
        "lof": lambda frame: frame.min(axis=1),
        "kernel_pca": lambda frame: frame.mean(axis=1),
    }
    limits, rates = calibrate_ensemble(values, scorers, n_replicates=20)
    assert set(limits.index) == {"trinity_mean", "trinity_max", "five_detector_mean"}
    assert rates["false_positive_rate"].between(0, 1).all()




def test_knn_sweep_calibration_reuses_simulations_for_each_k():
    values = pd.DataFrame(np.random.default_rng(12).normal(size=(12, 3)))
    limits, null_scores = calibrate_knn_sweep(values, k_values=[2, 4], n_replicates=30, batch_size=10)
    assert list(limits.index) == [2, 4]
    assert all(len(scores) == 360 for scores in null_scores.values())
    assert limits["false_positive_rate"].between(0, 1).all()


def test_isolation_forest_calibration_returns_null_limits_and_rate():
    values = pd.DataFrame(np.random.default_rng(13).normal(size=(8, 3)))
    limits, null_scores, false_positive_rate = calibrate_isolation_forest(
        values,
        seeds=[1, 2],
        n_estimators=2,
        n_replicates=2,
        n_jobs=1,
    )
    assert limits["limit_95"] <= limits["limit_99"]
    assert len(null_scores) == 16
    assert 0 <= false_positive_rate <= 1


def test_lof_sweep_matches_sklearn_for_each_k():
    values = pd.DataFrame(np.random.default_rng(14).normal(size=(12, 4)))
    sweep = local_outlier_factor_sweep(values, k_values=[2, 5])
    for k in [2, 5]:
        expected = -LocalOutlierFactor(n_neighbors=k).fit(values).negative_outlier_factor_
        np.testing.assert_allclose(sweep[str(k)], expected)
    pd.testing.assert_series_equal(local_outlier_factor(values, parameter=5), sweep["5"], check_names=False)


def test_ledoit_wolf_sweep_matches_shrinkage_pseudoinverse():
    values = pd.DataFrame(np.random.default_rng(15).normal(size=(10, 16)))
    matrix = values.to_numpy()
    centered = matrix - matrix.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)
    target = np.eye(matrix.shape[1]) * np.trace(covariance) / matrix.shape[1]
    expected = []
    for alpha in [0, 0.5, 1]:
        expected.append(np.einsum("ij,jk,ik->i", centered, np.linalg.pinv((1 - alpha) * covariance + alpha * target), centered))
    np.testing.assert_allclose(
        ledoit_wolf_mahalanobis_sweep(values, shrinkages=[0, 0.5, 1]),
        np.column_stack(expected),
        rtol=1e-10,
        atol=1e-10,
    )


def test_lof_and_ledoit_wolf_calibrations_return_all_parameters():
    values = pd.DataFrame(np.random.default_rng(16).normal(size=(10, 5)))
    lof_limits, lof_null = calibrate_lof_sweep(values, k_values=[2, 4], n_replicates=3, batch_size=2)
    lw_limits, lw_null = calibrate_ledoit_wolf_sweep(values, shrinkages=[0, 0.5, 1], n_replicates=3, batch_size=2)
    assert list(lof_limits.index) == [2, 4]
    assert list(lw_limits.index) == [0.0, 0.5, 1.0]
    assert all(len(scores) == 30 for scores in lof_null.values())
    assert all(len(scores) == 30 for scores in lw_null.values())


def test_kernel_pca_reconstruction_is_stable_for_regional_sample_size():
    from mia_multiomics.outliers.detectors import kernel_pca_reconstruction_distance

    values = pd.DataFrame(np.random.default_rng(17).normal(size=(20, 10)))
    scores = kernel_pca_reconstruction_distance(values, parameter=0.1, n_components=10)
    assert np.isfinite(scores).all()


def test_kernel_pca_handles_constant_kernel_limit():
    from mia_multiomics.outliers.detectors import kernel_pca_reconstruction_distance

    values = pd.DataFrame(np.random.default_rng(18).normal(scale=1e-12, size=(20, 3)))
    scores = kernel_pca_reconstruction_distance(values, parameter=0.01, n_components=3)
    assert np.isfinite(scores).all()
