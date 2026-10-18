import numpy as np
import pandas as pd

from mia_multiomics.multivariate.plsda import fit_plsda, permutation_test_plsda
from mia_multiomics.multivariate.vendor import (
    cluster_heatmap,
    select_top_n_by_nominal_p,
    vendor_column_order,
)


def test_select_top_n_by_nominal_p_filters_to_complete_features_and_ranks_unique_features():
    effects = pd.DataFrame({
        "Feature": [1, 2, 3, 1, 2, 3],
        "Region": ["Cortex", "Cortex", "Cortex", "Hippocampus", "Hippocampus", "Hippocampus"],
        "Effect": ["MIA vs Saline Control"] * 6,
        "p_value": [0.04, 0.01, 0.03, 0.02, 0.05, 0.001],
    })

    cortex = select_top_n_by_nominal_p(
        effects, region="Cortex", eligible_features=[1, 3], n_features=2
    )
    all_groups = select_top_n_by_nominal_p(
        effects, region=None, eligible_features=[1, 2, 3], n_features=3
    )

    assert cortex.tolist() == [3, 1]
    assert all_groups.tolist() == [3, 2, 1]


def test_cluster_heatmap_returns_stable_complete_ordering():
    values = pd.DataFrame(
        [[0.0, 0.1, 4.0, 4.1], [0.2, 0.0, 4.2, 4.0], [4.0, 4.1, 0.0, 0.1]],
        index=["A", "B", "C"],
        columns=["S1", "S2", "S3", "S4"],
    )

    first, first_rows, first_columns = cluster_heatmap(values)
    second, second_rows, second_columns = cluster_heatmap(values)

    pd.testing.assert_frame_equal(first, second)
    np.testing.assert_allclose(first_rows, second_rows)
    np.testing.assert_allclose(first_columns, second_columns)


def test_vendor_column_order_matches_group_sex_and_sample_number_sort():
    metadata = pd.DataFrame({
        "Sample": ["S4", "S1", "S3", "S2"],
        "Bio group": ["MIA", "Saline Control", "MIA", "Saline Control"],
        "Sex": ["Female", "Female", "Male", "Male"],
        "Sample number": [4, 1, 3, 2],
    })

    assert vendor_column_order(metadata).tolist() == ["S2", "S1", "S3", "S4"]


def _synthetic_plsda_values(group_difference: float) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(4)
    samples = [f"S{number}" for number in range(20)]
    labels = pd.Series(["Saline Control"] * 10 + ["MIA"] * 10, index=samples)
    sexes = pd.Series(["Male", "Female"] * 10, index=samples)
    values = pd.DataFrame(rng.normal(size=(12, 20)), index=[f"Lipid{number}" for number in range(12)], columns=samples)
    values.loc[values.index[:4], labels.eq("MIA")] += group_difference
    return values, labels, sexes


def test_plsda_permutation_is_more_significant_for_an_injected_group_difference():
    null_values, null_labels, sexes = _synthetic_plsda_values(group_difference=0.0)
    signal_values, signal_labels, _ = _synthetic_plsda_values(group_difference=4.0)

    null_result, _ = permutation_test_plsda(
        null_values, null_labels, sexes, n_permutations=199, random_state=3
    )
    signal_result, _ = permutation_test_plsda(
        signal_values, signal_labels, sexes, n_permutations=199, random_state=3
    )
    scores, variance_explained = fit_plsda(signal_values, signal_labels)

    assert signal_result["permutation_p_value"] < null_result["permutation_p_value"]
    assert signal_result["permutation_p_value"] <= 0.01
    assert scores.shape == (20, 2)
    assert np.all(np.diff(variance_explained) >= 0)
    assert np.all((variance_explained >= 0) & (variance_explained <= 1))
