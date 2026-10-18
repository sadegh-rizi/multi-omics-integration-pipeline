"""Scientific PQN checks, including the explicitly chosen log-median convention."""

import numpy as np
import pandas as pd
import pytest

from mia_multiomics.preprocessing import normalize, pqn_normalize


def test_pqn_recovers_dilution_with_twenty_percent_changed_features():
    profile = np.linspace(-5, 5, 101)
    dilution = np.array([-0.7, -0.3, 0, 0.3, 0.7])
    values = pd.DataFrame(profile[:, None] + dilution)
    values.iloc[:20, 4] += 1.0
    normalized = pqn_normalize(values, pd.Series("region", index=values.columns))
    np.testing.assert_allclose((values - normalized).median(), dilution, atol=0.02)
    # Odd-sized reference and quotient medians commute with log.
    raw = np.exp2(values)
    reference = raw.median(axis=1)
    reference /= reference.median()
    raw_pqn = raw.divide(raw.divide(reference, axis=0).median(axis=0), axis=1)
    np.testing.assert_allclose(normalized, np.log2(raw_pqn), atol=1e-10)


def test_even_medians_do_not_in_general_commute_with_log():
    values = pd.DataFrame({"a": [0., 4.], "b": [4., 0.]})
    normalized = pqn_normalize(values, pd.Series("region", index=values.columns))
    np.testing.assert_allclose(normalized, values - 2)
    raw = np.exp2(values)
    raw_normalized = np.log2(raw.divide(raw.median(), axis=1))
    assert not np.allclose(normalized, raw_normalized)


def test_per_region_reference_preserves_composition_without_pooled_dilution_bias():
    # Thirty of 101 features change before expressing the region profiles on
    # the declared common-median (relative) scale. That anchoring is essential:
    # no method here identifies absolute loading separately from composition.
    profile = np.linspace(-5, 5, 101)
    groups = pd.Series(np.repeat(["A", "B", "C"], 5))
    dilution = np.tile([-0.6, -0.2, 0, 0.2, 0.6], 3)
    values = pd.DataFrame(profile[:, None] + dilution)
    changed = profile.copy()
    changed[:30] += 6
    changed -= np.median(changed)
    values.iloc[:, 10:] = changed[:, None] + dilution[10:]
    normalized = pqn_normalize(values, groups)
    shifts = (values - normalized).median()
    np.testing.assert_allclose(shifts, dilution, atol=0.02)
    pooled = pqn_normalize(values, pd.Series("pooled", index=values.columns))
    bias = (values - pooled).median() - dilution
    assert abs(bias[groups.eq("C")].mean() - bias[groups.eq("A")].mean()) > 0.1


def test_pqn_preserves_missingness_and_requires_half_panel_overlap():
    values = pd.DataFrame({"a": [0., 1., 2., 3.], "b": [0., np.nan, np.nan, 3.]})
    groups = pd.Series("region", index=values.columns)
    before = values.copy()
    normalized = pqn_normalize(values, groups)
    pd.testing.assert_frame_equal(normalized.isna(), values.isna())
    pd.testing.assert_frame_equal(values, before)
    values.loc[0, "b"] = np.nan
    with pytest.raises(ValueError, match="half"):
        pqn_normalize(values, groups)


def test_pqn_by_mode_uses_separate_references_and_loading_shifts():
    values = pd.DataFrame({"a": [-1., 0., 1., 4., 5., 6.], "b": [1., 2., 3., 1., 2., 3.]})
    methods = pd.Series(["pos"] * 3 + ["neg"] * 3)
    normalized = normalize(values, "pqn", modes=methods, regions=pd.Series("region", index=values.columns))
    shifts = (values - normalized).groupby(methods).median()
    np.testing.assert_allclose(normalized, np.tile([-1., 0., 1.], 2)[:, None] * np.ones((1, 2)))
    np.testing.assert_allclose(shifts.loc["pos"], [0, 2])
    np.testing.assert_allclose(shifts.loc["neg"], [5, 2])
