"""Small helpers shared by the integration analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def spearman_matrix(left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray:
    """Feature-by-feature Spearman correlations between two samples x features tables."""
    def ranks(table: pd.DataFrame) -> np.ndarray:
        r = rankdata(table.to_numpy(dtype=float), axis=0)
        sd = r.std(axis=0, ddof=1)
        return np.divide(r - r.mean(axis=0), sd, out=np.zeros_like(r), where=sd > 0)
    return ranks(left).T @ ranks(right) / (len(left) - 1)
