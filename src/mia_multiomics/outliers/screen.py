"""Multivariate sample-outlier diagnostics for PCA score spaces."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta, chi2
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA


def _numeric_scores(scores: pd.DataFrame) -> pd.DataFrame:
    numeric = scores.apply(pd.to_numeric, errors="coerce")
    if numeric.empty or numeric.isna().to_numpy().any():
        raise ValueError("Outlier diagnostics require complete PCA scores.")
    if len(numeric) <= numeric.shape[1]:
        raise ValueError("Outlier diagnostics require more samples than retained PCs.")
    return numeric


def hotelling_t2(scores: pd.DataFrame) -> pd.DataFrame:
    """Calculate Hotelling T-squared and 95%/99% limits in PCA score space."""
    numeric = _numeric_scores(scores)
    n_samples, n_components = numeric.shape
    centered = numeric.to_numpy(dtype=float) - numeric.to_numpy(dtype=float).mean(axis=0)
    covariance_inverse = np.linalg.pinv(np.cov(centered, rowvar=False))
    statistic = np.einsum("ij,jk,ik->i", centered, covariance_inverse, centered)
    factor = (n_samples - 1) ** 2 / n_samples
    return pd.DataFrame({
        "hotelling_t2": statistic,
        "hotelling_t2_95_limit": factor * beta.ppf(0.95, n_components / 2, (n_samples - n_components - 1) / 2),
        "hotelling_t2_99_limit": factor * beta.ppf(0.99, n_components / 2, (n_samples - n_components - 1) / 2),
    }, index=numeric.index)


def robust_mahalanobis(
    scores: pd.DataFrame,
    *,
    random_state: int = 20_260_721,
    support_fraction: float = 0.75,
) -> pd.DataFrame:
    """Calculate MCD-based squared robust Mahalanobis distances and limits."""
    numeric = _numeric_scores(scores)
    estimator = MinCovDet(
        random_state=random_state,
        support_fraction=support_fraction,
    ).fit(numeric.to_numpy(dtype=float))
    n_components = numeric.shape[1]
    return pd.DataFrame({
        "robust_mahalanobis": estimator.mahalanobis(numeric.to_numpy(dtype=float)),
        "robust_mahalanobis_95_limit": chi2.ppf(0.95, n_components),
        "robust_mahalanobis_99_limit": chi2.ppf(0.99, n_components),
    }, index=numeric.index)


def dmodx(values: pd.DataFrame, model: PCA) -> pd.Series:
    """Return each sample's residual distance to a fitted PCA model plane."""
    numeric = values.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().to_numpy().any():
        raise ValueError("DModX requires a complete feature-by-sample matrix.")
    reconstructed = model.inverse_transform(model.transform(numeric.T.to_numpy(dtype=float)))
    residuals = numeric.T.to_numpy(dtype=float) - reconstructed
    return pd.Series(np.sqrt(np.mean(np.square(residuals), axis=1)), index=numeric.columns, name="dmodx")


def sample_outlier_table(
    values: pd.DataFrame,
    model: PCA,
    scores: pd.DataFrame,
    *,
    robust_scores: pd.DataFrame | None = None,
    random_state: int = 20_260_721,
) -> pd.DataFrame:
    """Combine PCA diagnostics, sorted worst-first by robust distance.

    ``robust_scores`` permits robust distance to use fewer PCs than the
    descriptive/Hotelling model, which is necessary in small regional groups.
    """
    if robust_scores is None:
        robust_scores = scores
    table = hotelling_t2(scores).join(
        robust_mahalanobis(robust_scores, random_state=random_state)
    )
    table = table.join(dmodx(values, model))
    table["hotelling_t2_95_flag"] = table["hotelling_t2"].gt(table["hotelling_t2_95_limit"])
    table["hotelling_t2_99_flag"] = table["hotelling_t2"].gt(table["hotelling_t2_99_limit"])
    table["robust_mahalanobis_95_flag"] = table["robust_mahalanobis"].gt(table["robust_mahalanobis_95_limit"])
    table["robust_mahalanobis_99_flag"] = table["robust_mahalanobis"].gt(table["robust_mahalanobis_99_limit"])
    return table.sort_values("robust_mahalanobis", ascending=False, kind="stable")
