import numpy as np
import pandas as pd

from mia_multiomics.multivariate.pca import fit_pca
from mia_multiomics.outliers.screen import sample_outlier_table


def test_outlier_screen_flags_planted_outlier():
    rng = np.random.default_rng(11)
    values = pd.DataFrame(
        rng.normal(size=(20, 15)),
        index=[f"Lipid{number}" for number in range(20)],
        columns=[f"S{number}" for number in range(15)],
    )
    values["S14"] += 12
    model, scores, _ = fit_pca(values, n_components=5)
    diagnostics = sample_outlier_table(values, model, scores)

    assert diagnostics.index[0] == "S14"
    assert diagnostics.loc["S14", "robust_mahalanobis_99_flag"]


def test_outlier_screen_can_use_fewer_pcs_for_robust_distance():
    rng = np.random.default_rng(15)
    values = pd.DataFrame(rng.normal(size=(20, 15)), columns=[f"S{number}" for number in range(15)])
    model, scores, _ = fit_pca(values, n_components=5)

    diagnostics = sample_outlier_table(values, model, scores, robust_scores=scores.iloc[:, :3])

    assert len(diagnostics) == 15
