import numpy as np
import pandas as pd
import pytest

from mia_multiomics.noise_floor import (
    abundance_similarity_curve,
    candidate_thresholds,
    compare_noise_criteria,
    pairwise_window_correlation,
)


def test_noisyr_reference_windows_cover_the_full_rank_range_without_gaps():
    values = pd.DataFrame(
        {"S1": np.arange(11, dtype=float), "S2": np.arange(11, dtype=float) * 2},
        index=[f"L{number}" for number in range(11)],
    )

    result = pairwise_window_correlation(values, window_size=4, step=3, method="pearson")

    for _, reference_rows in result.groupby("sample_i", sort=False):
        assert reference_rows["window_start"].tolist() == [0, 3, 6, 7]
        assert reference_rows["window_index"].tolist() == [3, 2, 1, 0]
        covered = set()
        for start, stop in reference_rows.loc[:, ["window_start", "window_stop"]].itertuples(index=False):
            covered.update(range(start, stop))
        assert covered == set(range(11))


def test_noisyr_window_excludes_a_comparator_missing_lipid_only_from_that_correlation():
    values = pd.DataFrame(
        {
            "S1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "S2": [1.1, 2.2, np.nan, 4.4, 5.5],
        },
        index=[f"L{number}" for number in range(5)],
    )

    result = pairwise_window_correlation(values, window_size=4, step=1, method="pearson")

    s1_rows = result.loc[result["sample_i"].eq("S1")]
    s2_rows = result.loc[result["sample_i"].eq("S2")]
    assert len(s1_rows) == 2
    assert s1_rows["n_lipids_in_window"].eq(4).all()
    assert s1_rows["n_lipids_in_correlation"].eq(3).all()
    assert len(s2_rows) == 1
    assert s2_rows.iloc[0]["n_lipids_in_window"] == 4
    assert s2_rows.iloc[0]["n_lipids_in_correlation"] == 4


def test_noisyr_uses_each_reference_samples_own_ranking():
    values = pd.DataFrame(
        {
            "S1": [100.0, 80.0, 3.0, 2.0],
            "S2": [1.0, 4.0, 50.0, 40.0],
        },
        index=["L1", "L2", "L3", "L4"],
    )

    result = pairwise_window_correlation(values, window_size=2, step=2, method="pearson")

    s1_highest = result.loc[
        result["sample_i"].eq("S1") & result["window_start"].eq(0)
    ].iloc[0]
    s2_highest = result.loc[
        result["sample_i"].eq("S2") & result["window_start"].eq(0)
    ].iloc[0]
    assert s1_highest["mean_log2_abundance"] == pytest.approx(90.0)
    assert s2_highest["mean_log2_abundance"] == pytest.approx(45.0)


def test_pairwise_window_rejects_a_window_larger_than_the_lipid_panel():
    values = pd.DataFrame({"S1": [1.0, 2.0], "S2": [3.0, 4.0]})

    with pytest.raises(ValueError, match="cannot exceed"):
        pairwise_window_correlation(values, window_size=3, step=1, method="pearson")


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_pairwise_window_correlation_matches_pandas_for_each_complete_pair(method: str):
    values = pd.DataFrame(
        {
            "S1": [1.0, 1.0, 2.0, 3.0, 4.0],
            "S2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "S3": [1.5, 2.0, 3.5, 4.0, 6.0],
        }
    )

    result = pairwise_window_correlation(values, window_size=5, step=1, method=method)

    for row in result.itertuples(index=False):
        expected = values.loc[:, [row.sample_i, row.sample_j]].corr(method=method).iloc[0, 1]
        assert row.correlation == pytest.approx(expected)


def test_curve_averages_comparators_per_reference_then_summarizes_references():
    values = pd.DataFrame(
        {
            "S1": [1.0, 2.0, 3.0, 4.0],
            "S2": [1.0, 2.0, 4.0, 8.0],
            "S3": [4.0, 3.0, 2.0, 1.0],
        }
    )
    pairwise = pairwise_window_correlation(values, window_size=4, step=1, method="pearson")

    result = abundance_similarity_curve(
        values,
        window_size=4,
        step=1,
        min_pairwise_correlations_per_point=6,
        min_comparators_per_reference=2,
    )
    dropped = abundance_similarity_curve(
        values,
        window_size=4,
        step=1,
        min_pairwise_correlations_per_point=7,
        min_comparators_per_reference=2,
    )

    reference_means = pairwise.groupby("sample_i", sort=False)["correlation"].mean()
    assert result.loc[0, "n_reference_samples"] == 3
    assert result.loc[0, "n_pairwise_correlations"] == 6
    assert result.loc[0, "median_pearson"] == pytest.approx(reference_means.median())
    assert result.loc[0, "median_log2_abundance"] == pytest.approx(
        pairwise.groupby("sample_i", sort=False)["mean_log2_abundance"].first().median()
    )
    assert dropped.empty


def test_curve_requires_two_valid_comparators_for_each_reference_window():
    values = pd.DataFrame(
        {
            "S1": [1.0, 2.0, 3.0, 4.0],
            "S2": [1.5, 2.5, 3.5, 4.5],
            "S3": [np.nan, np.nan, np.nan, np.nan],
        }
    )

    curve = abundance_similarity_curve(
        values,
        window_size=4,
        step=1,
        min_pairwise_correlations_per_point=1,
        min_comparators_per_reference=2,
    )

    assert curve.empty


def test_biological_and_techmix_inputs_return_the_same_curve_schema():
    values = pd.DataFrame(
        np.arange(32, dtype=float).reshape(8, 4) + 1,
        columns=["S1", "S2", "S3", "S4"],
    )
    techmix = values.rename(columns={column: f"TechMix_{number}" for number, column in enumerate(values.columns)})

    biological_curve = abundance_similarity_curve(
        values,
        window_size=4,
        step=2,
        min_pairwise_correlations_per_point=1,
    )
    techmix_curve = abundance_similarity_curve(
        techmix,
        window_size=4,
        step=2,
        min_pairwise_correlations_per_point=1,
    )

    assert list(biological_curve.columns) == list(techmix_curve.columns)
    assert not biological_curve.empty
    assert not techmix_curve.empty


def test_planted_noise_floor_is_recovered_near_the_known_abundance():
    generator = np.random.default_rng(20_260_813)
    n_features = 500
    baseline = np.repeat(
        np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.2, 8.2, 9.2, 10.2, 11.2]),
        50,
    )
    known_floor = 7.0
    values = np.empty((n_features, 20))
    low_abundance = baseline < known_floor
    values[low_abundance] = baseline[low_abundance, np.newaxis] + generator.normal(
        0, 0.5, size=(low_abundance.sum(), 20)
    )
    shared_signal = generator.normal(0, 0.75, size=(~low_abundance).sum())
    values[~low_abundance] = (
        baseline[~low_abundance, np.newaxis]
        + shared_signal[:, np.newaxis]
        + generator.normal(0, 0.1, size=((~low_abundance).sum(), 20))
    )
    curve = abundance_similarity_curve(
        pd.DataFrame(values, columns=[f"S{number}" for number in range(20)]),
        window_size=50,
        step=25,
        min_pairwise_correlations_per_point=10,
        min_comparators_per_reference=2,
    )
    thresholds = candidate_thresholds(curve, rules=("pearson_gt_0_50",))

    recovered = thresholds.iloc[0]
    assert recovered["status"] == "found"
    assert recovered["threshold_log2_abundance"] == pytest.approx(known_floor, abs=1.0)


def test_independent_null_columns_produce_an_approximately_zero_noisyr_curve():
    generator = np.random.default_rng(20_260_821)
    values = pd.DataFrame(
        generator.normal(size=(1_000, 12)),
        columns=[f"S{number}" for number in range(12)],
    )

    curve = abundance_similarity_curve(
        values,
        window_size=100,
        step=50,
        min_pairwise_correlations_per_point=10,
        min_comparators_per_reference=2,
    )

    assert not curve.empty
    assert curve["median_pearson"].abs().max() < 0.15
    assert curve["median_pearson"].mean() == pytest.approx(0.0, abs=0.04)


def test_noise_criteria_reports_all_contingency_cells_and_jaccards():
    index = pd.Index(["L1", "L2", "L3", "L4"])
    abundance = pd.Series([True, True, False, False], index=index)
    blank = pd.Series([True, False, True, False], index=index)
    cv = pd.Series([True, False, False, False], index=index)

    contingency, jaccard = compare_noise_criteria(abundance, blank, cv)

    assert len(contingency) == 8
    assert contingency["n_features"].sum() == 4
    abundance_blank = jaccard.query(
        "criterion_a == 'abundance_below_floor' and criterion_b == 'blank_dominated'"
    ).iloc[0]
    assert abundance_blank["intersection"] == 1
    assert abundance_blank["union"] == 3
    assert abundance_blank["jaccard"] == pytest.approx(1 / 3)


def test_candidate_thresholds_flag_nonpersistent_crossings_and_abundance_alignment():
    curve = pd.DataFrame({
        "window_index": [0, 1, 2, 3, 4],
        "median_log2_abundance": [10.0, 11.0, 12.0, 13.0, 14.0],
        "median_pearson": [0.30, -0.75, -0.50, 0.10, 0.20],
        "median_spearman": [0.30, -0.75, -0.50, 0.10, 0.20],
    })
    technical = pd.DataFrame({
        "window_index": [0, 1, 2, 3, 4],
        "median_log2_abundance": [9.0, 10.0, 12.0, 13.0, 14.0],
        "median_pearson": [0.80, -1.0, -0.50, 0.10, 0.20],
        "median_spearman": [0.80, -1.0, -0.50, 0.10, 0.20],
    })

    thresholds = candidate_thresholds(
        curve,
        technical_curve=technical,
        rules=("pearson_90pct_plateau", "bio_tech_gap_le_0_05"),
    ).set_index("threshold_rule")

    plateau = thresholds.loc["pearson_90pct_plateau"]
    assert plateau["threshold_window_index"] == 0
    assert not plateau["threshold_persists_through_high_abundance"]
    assert plateau["n_later_windows_failing_rule"] == 3

    convergence = thresholds.loc["bio_tech_gap_le_0_05"]
    assert convergence["comparison_alignment"] == "interpolated_technical_at_biological_abundance"
    assert convergence["threshold_log2_abundance"] == 11.0
    assert convergence["technical_log2_abundance_at_threshold"] == 11.0
    assert convergence["technical_median_pearson_at_threshold"] == pytest.approx(-0.75)
    assert convergence["n_abundance_aligned_curve_points"] == 5
    assert convergence["n_biological_points_outside_technical_abundance_range"] == 0


def test_candidate_thresholds_does_not_extrapolate_technical_correlation():
    curve = pd.DataFrame({
        "window_index": [0, 1],
        "median_log2_abundance": [10.0, 11.0],
        "median_pearson": [0.2, 0.3],
        "median_spearman": [0.2, 0.3],
    })
    technical = pd.DataFrame({
        "window_index": [0, 1],
        "median_log2_abundance": [1.0, 2.0],
        "median_pearson": [0.2, 0.3],
        "median_spearman": [0.2, 0.3],
    })

    threshold = candidate_thresholds(
        curve,
        technical_curve=technical,
        rules=("bio_tech_gap_le_0_05",),
    ).iloc[0]

    assert threshold["status"] == "no_shared_technical_abundance_range"
    assert threshold["n_abundance_aligned_curve_points"] == 0

