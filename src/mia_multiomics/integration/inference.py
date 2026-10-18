"""Treatment tests for integration scores (factor scores, module scores, composites).

The design has ~20 mice in 6-8 litters, so the MIA label can only be shuffled
in a few ways if the litter (cage) is respected.  Every score is therefore
tested three ways:

* Welch t on mouse means (assumes mice are independent - optimistic);
* mouse-level permutation within sex (also ignores litters);
* exact whole-cage permutation (honest, but its smallest possible p-value is
  1 / number of cage assignments).

A score is called "treatment associated" only if the cage p sits at its floor,
the mouse p < 0.05 and the group R2 exceeds the cage R2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from .. import config
from ..stats.clustered import cage_group_assignments
from ..stats.common import exact_empirical_p, r_squared


def mouse_grouped_folds(metadata: pd.DataFrame, *, n_splits: int = 5, n_repeats: int = 10,
                        random_state: int = 20_261_011) -> list[tuple[np.ndarray, np.ndarray]]:
    """Repeated cross-validation folds that never split one mouse's samples."""
    mice = metadata[config.MOUSE_COL].drop_duplicates().to_numpy()
    rng = np.random.default_rng(random_state)
    folds = []
    for _ in range(n_repeats):
        for test_mice in np.array_split(rng.permutation(mice), n_splits):
            test = metadata[config.MOUSE_COL].isin(test_mice).to_numpy()
            folds.append((np.flatnonzero(~test), np.flatnonzero(test)))
    return folds


def _mouse_means(metadata: pd.DataFrame, scores: pd.Series, labels: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"Mouse": metadata[config.MOUSE_COL].to_numpy(), "Cage": metadata[config.CAGE_COL].to_numpy(),
                          "Score": np.asarray(scores, dtype=float), "Group": np.asarray(labels)})
    return frame.groupby("Mouse", sort=False).agg(Score=("Score", "mean"), Cage=("Cage", "first"),
                                                  Group=("Group", "first")).reset_index()


def _welch(mice: pd.DataFrame) -> tuple[float, float]:
    test = ttest_ind(mice.loc[mice["Group"].eq(config.MIA), "Score"],
                     mice.loc[mice["Group"].eq(config.CONTROL), "Score"], equal_var=False)
    return float(test.statistic), float(test.pvalue)


def _mouse_permutation_t(mice: pd.DataFrame, sexes: pd.Series, n: int, seed: int) -> np.ndarray:
    """Welch t for ``n`` sex-stratified shuffles of the MIA label across mice (vectorised)."""
    rng = np.random.default_rng(seed)
    labels = mice["Group"].to_numpy()
    values = mice["Score"].to_numpy(dtype=float)
    strata = [np.flatnonzero(sexes.to_numpy() == s) for s in pd.unique(sexes)]
    shuffled = np.empty((n, len(labels)), dtype=bool)
    for i in range(n):
        current = labels.copy()
        for index in strata:
            current[index] = rng.permutation(labels[index])
        shuffled[i] = current == config.MIA
    a, b = shuffled, ~shuffled
    na, nb = a.sum(1), b.sum(1)
    ma, mb = a @ values / na, b @ values / nb
    va = (a @ values ** 2 - na * ma ** 2) / (na - 1)
    vb = (b @ values ** 2 - nb * mb ** 2) / (nb - 1)
    se = np.sqrt(va / na + vb / nb)
    return np.divide(ma - mb, se, out=np.full(n, np.nan), where=se > 0)


def treatment_score_test(metadata: pd.DataFrame, scores: pd.Series, *, score_name: str,
                         n_mouse_permutations: int = 9_999, random_state: int = 20_261_013) -> dict:
    """Test one per-sample score for an MIA effect at mouse and cage level (see module docstring)."""
    metadata = metadata.reset_index(drop=True)
    mice = _mouse_means(metadata, scores, metadata[config.GROUP_COL])
    observed_t, welch_p = _welch(mice)
    cages = mice.groupby("Cage", sort=False).agg(Score=("Score", "mean"), Group=("Group", "first"))
    group_r2 = r_squared(mice["Score"], mice["Group"])
    cage_r2 = r_squared(cages["Score"], cages["Group"])
    cage_null = np.array([_welch(_mouse_means(metadata, scores, assignment))[0]
                          for assignment in cage_group_assignments(metadata)])
    cage_p = exact_empirical_p(observed_t, cage_null)
    sexes = metadata.drop_duplicates(config.MOUSE_COL).set_index(config.MOUSE_COL).loc[mice["Mouse"], config.SEX_COL]
    mouse_p = exact_empirical_p(observed_t, _mouse_permutation_t(mice, sexes, n_mouse_permutations, random_state))
    floor = 1 / len(cage_null)
    associated = bool(np.isclose(cage_p, floor) and mouse_p < 0.05 and group_r2 > cage_r2)
    return {"Score name": score_name, "observed_welch_t": observed_t, "mouse_welch_p": welch_p,
            "mouse_permutation_p": mouse_p, "cage_permutation_p": cage_p, "cage_p_floor": floor,
            "group_r_squared": group_r2, "cage_r_squared": cage_r2,
            "treatment_associated": associated, "n_cage_assignments": len(cage_null)}


def null_calibration(metadata: pd.DataFrame, *, n_datasets: int = 200, n_mouse_permutations: int = 999,
                     random_state: int = 20_261_014) -> pd.DataFrame:
    """Simulate pure-noise scores, with and without a shared litter offset, and record the p-values.

    With litter offsets the mouse-level test rejects far more than 5 % of the
    time; the cage-level test stays calibrated.  This is why tests are run at
    the litter level.
    """
    rng = np.random.default_rng(random_state)
    rows = []
    for litter_effect in (False, True):
        for dataset in range(n_datasets):
            offsets = {c: rng.normal() if litter_effect else 0.0 for c in metadata[config.CAGE_COL].unique()}
            scores = pd.Series([rng.normal() + offsets[c] for c in metadata[config.CAGE_COL]])
            result = treatment_score_test(metadata, scores, score_name="null", n_mouse_permutations=n_mouse_permutations,
                                          random_state=random_state + dataset)
            rows.append({"litter_effect": litter_effect, "dataset": dataset,
                         "mouse_permutation_p": result["mouse_permutation_p"],
                         "cage_permutation_p": result["cage_permutation_p"]})
    return pd.DataFrame(rows)
