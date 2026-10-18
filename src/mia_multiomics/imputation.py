"""Censoring-aware, label-free imputation for regional lipidomics matrices.

The main analysis never imputes.  These imputers exist only for sensitivity
analyses (does the answer change if missing values are filled in?).

A missing value can mean two different things, and they need different fills:

* **left-censored** - the peak was zero or below the detection limit: the true
  value is *small*, so it gets half the feature minimum ("half-min");
* **missing at random (MAR)** - absent from the delivered file for no
  abundance-related reason: predicted from the other features by a random
  forest (``IterativeImputer``).

The imputers never see the group labels ("label-free"), so they cannot create
a group difference.  Each returns the completed values and a same-shaped table
saying which route filled every cell.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


def censoring_mask(intensity: pd.DataFrame, below_detection: pd.DataFrame) -> pd.DataFrame:
    """Label source-MAR, zero, and detection-limit cells before log transformation."""
    numeric = intensity.apply(pd.to_numeric, errors="coerce")
    if not intensity.index.equals(below_detection.index) or not intensity.columns.equals(below_detection.columns):
        raise ValueError("Intensity and detection-limit masks must align exactly.")
    return pd.DataFrame(
        np.select(
            [numeric.isna(), numeric.eq(0), below_detection.notna()],
            ["mar", "left_censored", "left_censored"],
            default="observed",
        ),
        index=intensity.index,
        columns=intensity.columns,
    )


def _validate(values: pd.DataFrame, mask: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Numeric values and a censoring mask of exactly the same shape and labels."""
    numeric = values.apply(pd.to_numeric, errors="coerce")
    if not numeric.index.equals(mask.index) or not numeric.columns.equals(mask.columns):
        raise ValueError("Values and censoring mask must align exactly.")
    return numeric, mask


def _half_min(values: pd.DataFrame, fill: pd.DataFrame) -> pd.DataFrame:
    """On log2 scale, half the feature minimum is one log2 unit below it."""
    minimum = values.min(axis=1, skipna=True)
    available = minimum.notna()
    fill_values = pd.DataFrame(
        np.repeat(minimum.subtract(1.0).to_numpy()[:, np.newaxis], values.shape[1], axis=1),
        index=values.index,
        columns=values.columns,
    )
    return values.mask(fill & available.to_numpy()[:, np.newaxis], fill_values)


def impute_none(values: pd.DataFrame, mask: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return values unchanged and record no imputed cells."""
    numeric, _ = _validate(values, mask)
    return numeric.copy(), pd.DataFrame("none", index=values.index, columns=values.columns)


def impute_mnar_halfmin(values: pd.DataFrame, mask: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill every missing value with feature half-min on the log2 scale."""
    numeric, labels = _validate(values, mask)
    missing = numeric.isna()
    routes = pd.DataFrame("none", index=values.index, columns=values.columns).mask(missing, "halfmin")
    return _half_min(numeric, missing), routes


def _rf_impute(values: pd.DataFrame, fill: pd.DataFrame, *, random_state: int) -> pd.DataFrame:
    """Fill the cells marked in ``fill`` with a random-forest IterativeImputer; other cells are kept.

    Samples are the rows the imputer learns from, so each feature is predicted
    from its 10 most correlated features.  Features with no value at all are left missing.
    """
    available = values.notna().any(axis=1)
    fill = fill & available.to_numpy()[:, np.newaxis]
    if not fill.to_numpy().any():
        return values.copy()
    # Regional matrices contain only 19 samples; a deliberately small forest
    # keeps this exploratory specification curve tractable without tuning to results.
    estimator = RandomForestRegressor(n_estimators=10, min_samples_leaf=2, random_state=random_state, n_jobs=1)
    imputer = IterativeImputer(
        estimator=estimator,
        random_state=random_state,
        max_iter=1,
        n_nearest_features=10,
        initial_strategy="median",
        skip_complete=True,
    )
    # The imputer works on samples x features; transpose in and out.
    with warnings.catch_warnings():
        # max_iter=1 is deliberate (one pass, small forest); silence the "not converged" notice.
        warnings.simplefilter("ignore", ConvergenceWarning)
        completed = imputer.fit_transform(values.loc[available].T)
    result = values.copy()
    result.loc[available] = pd.DataFrame(completed.T, index=values.index[available], columns=values.columns)
    return values.where(~fill, result)


def impute_rf_mar_only(values: pd.DataFrame, mask: pd.DataFrame, *, random_state: int = 20_260_807) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Half-min left-censored values and RF-impute only genuinely MAR cells."""
    numeric, labels = _validate(values, mask)
    left = labels.eq("left_censored") & numeric.isna()
    mar = labels.eq("mar") & numeric.isna()
    halfmin = _half_min(numeric, left)
    completed = _rf_impute(halfmin, mar, random_state=random_state)
    route = pd.DataFrame("none", index=values.index, columns=values.columns)
    route = route.mask(left, "halfmin").mask(mar, "rf_mar")
    return completed, route


def impute_rf_all(values: pd.DataFrame, mask: pd.DataFrame, *, random_state: int = 20_260_807) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Naive contrast: RF-impute every missing cell regardless of censoring."""
    numeric, _ = _validate(values, mask)
    missing = numeric.isna()
    completed = _rf_impute(numeric, missing, random_state=random_state)
    return completed, pd.DataFrame("none", index=values.index, columns=values.columns).mask(missing, "rf_all")
