"""Small statistical helpers shared by several analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd


def bh_adjust(p_values: pd.Series, denominator: int | None = None) -> pd.Series:
    """Benjamini-Hochberg q-values.

    ``denominator`` is the size of the multiple-testing family.  It defaults
    to the number of non-missing p-values; pass the full panel size when
    features that could not be tested should still count in the family.
    Missing p-values stay missing.
    """
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    ordered = p_values.dropna().sort_values()
    if ordered.empty:
        return adjusted
    m = len(ordered) if denominator is None else int(denominator)
    if m < len(ordered):
        raise ValueError("The BH denominator cannot be smaller than the number of p-values.")
    ranks = np.arange(1, len(ordered) + 1)
    q = np.minimum.accumulate((ordered.to_numpy(dtype=float) * m / ranks)[::-1])[::-1]
    adjusted.loc[ordered.index] = np.clip(q, 0.0, 1.0)
    return adjusted


def r_squared(values: pd.Series | np.ndarray, predictor: pd.Series) -> float:
    """Share of variance in ``values`` explained by a categorical or numeric predictor."""
    y = np.asarray(values, dtype=float)
    if not np.isfinite(y).all() or np.var(y) == 0:
        return 0.0
    if predictor.dtype.kind in "biuf":
        x = np.column_stack([np.ones(len(y)), predictor.to_numpy(dtype=float)])
    else:
        x = pd.get_dummies(predictor.astype(str), drop_first=False, dtype=float).to_numpy()
    fitted = x @ np.linalg.lstsq(x, y, rcond=None)[0]
    total = np.sum((y - y.mean()) ** 2)
    return float(np.clip(1 - np.sum((y - fitted) ** 2) / total, 0.0, 1.0))



def empirical_p(observed: float, null: np.ndarray, *, two_sided: bool = True) -> float:
    """Permutation p-value (1 + exceedances) / (1 + draws); never exactly zero."""
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not len(null) or not np.isfinite(observed):
        return np.nan
    if two_sided:
        exceed = np.sum(np.abs(null) >= abs(observed))
    else:
        exceed = np.sum(null >= observed)
    return float((1 + exceed) / (1 + len(null)))


def exact_empirical_p(observed: float, null: np.ndarray) -> float:
    """p-value from an *exhaustive* set of permutations (the observed one included).

    Its floor is 1 / (number of permutations): with few cages this floor is
    the honest limit of what the design can show.
    """
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not len(null) or not np.isfinite(observed):
        return np.nan
    return float(max(1 / len(null), np.mean(np.abs(null) >= abs(observed))))


def safe_correlation(left: pd.Series, right: pd.Series, method: str = "pearson") -> float:
    """Correlation that returns NaN (instead of warning) for constant or tiny inputs."""
    valid = pd.concat([left, right], axis=1).dropna()
    if len(valid) < 3 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method=method))
