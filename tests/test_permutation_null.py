import numpy as np
import pandas as pd

from mia_multiomics.stats.permutation import (
    _empirical_null_fdr,
    estimate_effective_tests,
    permute_group_labels,
    permute_labels_for_effect,
    run_permutation_null,
)


def _metadata() -> pd.DataFrame:
    rows = []
    for group in ("Saline Control", "MIA"):
        for sex in ("Male", "Female"):
            for replicate in range(3):
                rows.append({
                    "Sample": f"S_{group[:1]}_{sex[:1]}_{replicate}",
                    "Sample type": "Cortex",
                    "Bio group": group,
                    "Sex": sex,
                })
    return pd.DataFrame(rows)


def test_sample_permutation_preserves_group_counts_within_sex():
    metadata = _metadata()
    labels = permute_group_labels(metadata, np.random.default_rng(7))

    original = metadata.groupby(["Sex", "Bio group"]).size()
    permuted = pd.DataFrame({"Sex": metadata["Sex"], "Bio group": labels}).groupby(["Sex", "Bio group"]).size()
    pd.testing.assert_series_equal(original.sort_index(), permuted.sort_index())

    sex_permuted = permute_labels_for_effect(metadata.assign(BioGroup=metadata["Bio group"]), np.random.default_rng(7), "Female vs Male")
    original_by_group = metadata.groupby(["Bio group", "Sex"]).size()
    permuted_by_group = sex_permuted.groupby(["BioGroup", "Sex"]).size()
    pd.testing.assert_series_equal(
        original_by_group.sort_index().rename_axis(["BioGroup", "Sex"]),
        permuted_by_group.sort_index(),
    )


def test_permutation_null_returns_matched_empirical_p_values():
    metadata = _metadata()
    other = metadata.copy()
    other["Sample"] += "_other"
    other["Sample type"] = "Cerebellum"
    metadata = pd.concat([metadata, other], ignore_index=True)
    values = pd.DataFrame(
        {
            sample: [
                float(index),
                float(index % 4),
            ]
            for index, sample in enumerate(metadata["Sample"])
        },
        index=["lipid_1", "lipid_2"],
    )

    feature_statistics, null_summary, effective_tests = run_permutation_null(
        values,
        metadata,
        n_permutations=9,
        random_state=11,
    )

    assert set(feature_statistics["Effect"]) == {"MIA vs Saline Control", "Female vs Male", "MIA x Female interaction"}
    assert feature_statistics.columns[1] == "Region"
    assert set(feature_statistics["Region"]) == {"Cortex", "Cerebellum"}
    assert not feature_statistics.duplicated(["Feature", "Effect", "Region"]).any()
    assert feature_statistics["empirical_p_floor"].eq(1 / 10).all()
    assert feature_statistics["p_value_empirical"].between(0, 1).all()
    assert feature_statistics["p_value_empirical_fdr"].between(0, 1).all()
    assert feature_statistics["empirical_null_fdr"].between(0, 1).all()
    assert set(feature_statistics["fdr_denominator"]) == {len(values.index)}
    assert null_summary["Permutation"].min() == 1
    assert null_summary["Permutation"].max() == 9
    assert set(effective_tests["Effect"]) == set(feature_statistics["Effect"])


def test_empirical_null_fdr_uses_less_stringent_thresholds_and_ignores_empty_draws():
    observed = np.array([3.0, 2.0, 1.0])
    null = np.array([
        [3.0] * 8 + [np.nan] * 2,
        [0.0] * 10,
        [1.0] * 7 + [0.0] * 3,
    ])

    result = _empirical_null_fdr(observed, null)

    np.testing.assert_allclose(result, [0.5, 0.5, 1.7 / 3])


def test_effective_tests_uses_null_discovery_count_variance():
    observed = pd.DataFrame({"Feature": ["a", "b", "c"], "residual_df": [20, 20, 20]})
    null = np.array([
        [-3.0, -2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])

    result = estimate_effective_tests(observed, null)

    assert result["n_features_complete_null"] == 3
    assert result["effective_tests_status"] == "estimated from permutation discovery-count variance"
    assert 1 <= result["effective_number_of_tests"] <= 3
