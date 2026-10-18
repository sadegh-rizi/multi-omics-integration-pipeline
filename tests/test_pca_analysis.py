import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta

from mia_multiomics.multivariate.pca import (
    fit_pca,
    hotelling_ellipse,
    hotelling_t2_2d,
    loadings_table,
    scree_table,
)
from mia_multiomics.plotting import confidence_ellipse


def test_scree_values_sum_to_one_and_loadings_keep_annotations():
    values = pd.DataFrame(
        [[-1, 0, 1, 2], [2, 1, 0, -1], [0, 1, 0, 1]],
        index=["A", "B", "C"],
        columns=["S1", "S2", "S3", "S4"],
    )
    model, _, loadings = fit_pca(values)
    scree = scree_table(model)
    table = loadings_table(
        loadings,
        pd.DataFrame({"Lipid Name": ["a", "b", "c"], "Lipid Class": ["x", "y", "z"]}, index=values.index),
        n_features=2,
    )

    assert np.isclose(scree["variance_explained"].sum(), 1.0)
    assert set(table["Lipid Name"]) <= {"a", "b", "c"}


def test_confidence_ellipse_has_expected_normal_theory_radius():
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ellipse = confidence_ellipse(2 * np.cos(angles), np.sin(angles))
    radii = np.linalg.norm(ellipse - ellipse.mean(axis=0), axis=1)

    assert ellipse.shape == (100, 2)
    assert radii.max() > radii.min()


def test_hotelling_ellipse_matches_two_dimensional_beta_boundary_and_is_axis_aligned():
    scores = pd.DataFrame({"PC 1": np.arange(10, dtype=float), "PC 2": np.arange(10, dtype=float) ** 2})

    ellipse, limit = hotelling_ellipse(scores, alpha=0.95)

    assert limit == pytest.approx(9 ** 2 / 10 * beta.ppf(0.95, 1, 3.5))
    assert ellipse.shape == (100, 2)
    assert ellipse[:, 0].max() - scores["PC 1"].mean() == pytest.approx(scores["PC 1"].std(ddof=1) * np.sqrt(limit))
    assert ellipse[:, 1].mean() == pytest.approx(scores["PC 2"].mean())


def test_hotelling_t2_2d_has_approximately_five_percent_outside_for_isotropic_cloud():
    rng = np.random.default_rng(20260807)
    scores = pd.DataFrame(rng.normal(size=(10_000, 2)), columns=["PC 1", "PC 2"])
    _, limit = hotelling_ellipse(scores)

    outside_share = hotelling_t2_2d(scores).gt(limit).mean()

    assert outside_share == pytest.approx(0.05, abs=0.01)


@pytest.mark.parametrize("scores", [pd.DataFrame({"PC 1": [1, 2, 3]}), pd.DataFrame(np.ones((4, 3)))])
def test_hotelling_helpers_refuse_anything_but_two_score_columns(scores):
    with pytest.raises(ValueError, match="exactly two"):
        hotelling_ellipse(scores)
    with pytest.raises(ValueError, match="exactly two"):
        hotelling_t2_2d(scores)
