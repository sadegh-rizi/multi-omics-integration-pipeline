"""Stability and ensemble rules for calibrated outlier detector results."""

from __future__ import annotations

import pandas as pd


def stability_fraction(scores: pd.DataFrame) -> pd.Series:
    """Fraction of settings flagged within each scope and calibration stage."""
    required = {"scope", "calibration_stage", "Sample", "detector", "flag_95"}
    if missing := required.difference(scores.columns):
        raise ValueError(f"Missing stability columns: {sorted(missing)}")
    return scores.groupby(["scope", "calibration_stage", "Sample", "detector"], observed=True)["flag_95"].mean().rename("stability_fraction")


def combine_percentiles(percentiles: pd.DataFrame) -> pd.DataFrame:
    """Return the predeclared percentile-scale ensemble combinations."""
    required = {"mahalanobis", "knn", "isolation_forest", "lof", "kernel_pca"}
    if missing := required.difference(percentiles.columns):
        raise ValueError(f"Missing ensemble columns: {sorted(missing)}")
    primary = percentiles.loc[:, ["mahalanobis", "knn", "isolation_forest"]]
    return pd.DataFrame({
        "trinity_mean": primary.mean(axis=1),
        "trinity_max": primary.max(axis=1),
        "five_detector_mean": percentiles.loc[:, sorted(required)].mean(axis=1),
    }, index=percentiles.index)


def apply_stability_rule(stability: pd.DataFrame, *, minimum_fraction: float = 0.60) -> pd.DataFrame:
    """Apply the predeclared parameter-stability and family-corroboration rule."""
    required = {"scope", "calibration_stage", "Sample", "detector", "stability_fraction"}
    if missing := required.difference(stability.columns):
        raise ValueError(f"Missing stability-rule columns: {sorted(missing)}")
    if stability["calibration_stage"].nunique(dropna=True) != 1:
        raise ValueError("Apply the stability rule to one calibration stage at a time.")
    qualifying = stability.assign(stable=stability["stability_fraction"].ge(minimum_fraction))
    pivot = qualifying.pivot(index="Sample", columns="detector", values="stable").fillna(False)
    counted = [name for name in pivot.columns if name != "lof_supporting"]
    non_covariance = [name for name in ("knn_isotropic", "lof", "isolation_forest", "kernel_pca") if name in pivot]
    result = pd.DataFrame(index=pivot.index)
    result["stable_detector_count"] = pivot.loc[:, counted].sum(axis=1).astype(int)
    result["stable_non_covariance_count"] = pivot.loc[:, non_covariance].sum(axis=1).astype(int)
    result["candidate_outlier"] = result["stable_detector_count"].ge(2) & result["stable_non_covariance_count"].ge(1)
    return result
