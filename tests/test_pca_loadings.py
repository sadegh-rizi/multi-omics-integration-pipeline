import numpy as np
import pandas as pd
import pytest

from mia_multiomics.multivariate.pca import (
    align_loading_signs,
    class_composition,
    cumulative_share_curve,
    loading_concentration,
    top_loadings,
)


def test_align_loading_signs_is_deterministic_and_orients_largest_loading_positive():
    loadings = pd.DataFrame({"PC 1": [-0.8, 0.2], "PC 2": [0.1, -0.9]}, index=["A", "B"])

    first = align_loading_signs(loadings)
    second = align_loading_signs(loadings)

    pd.testing.assert_frame_equal(first, second)
    assert first.loc["A", "PC 1"] > 0
    assert first.loc["B", "PC 2"] > 0


def test_loading_concentration_is_monotone_and_bounded():
    loadings = pd.DataFrame({"PC 1": [0.8, 0.4, 0.2]}, index=["A", "B", "C"])

    concentration = loading_concentration(loadings, (1, 2, 3))

    assert concentration["squared_loading_share"].is_monotonic_increasing
    assert np.all(concentration["squared_loading_share"].between(0, 1))
    assert concentration.iloc[-1]["squared_loading_share"] == 1.0


def test_top_loadings_ranks_by_absolute_value():
    loadings = pd.DataFrame({"PC 1": [-0.8, 0.7, 0.1]}, index=["A", "B", "C"])

    ranked = top_loadings(loadings, 2)

    assert ranked["Feature"].tolist() == ["A", "B"]


def test_class_composition_sums_to_one_and_cumulative_curve_is_bounded():
    loadings = pd.DataFrame({"PC 1": [0.8, 0.4, 0.2]}, index=["A", "B", "C"])
    composition = class_composition(loadings, pd.Series(["PC", "PE", "PC"], index=loadings.index))
    curve = cumulative_share_curve(loadings)

    assert composition["squared_loading_share"].sum() == pytest.approx(1.0)
    assert curve["cumulative_squared_loading_share"].is_monotonic_increasing
    assert curve["cumulative_squared_loading_share"].iloc[-1] == pytest.approx(1.0)
