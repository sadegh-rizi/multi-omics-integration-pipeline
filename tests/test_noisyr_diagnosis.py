import numpy as np
import pandas as pd

from mia_multiomics.noise_floor import (
    abundance_similarity_curve,
    candidate_thresholds,
)
from mia_multiomics.noisyr import (
    panel_scaled_window_parameters,
    threshold_difference,
    threshold_median,
)


def test_pairwise_overlap_guard_drops_points_without_enough_observations():
    values = pd.DataFrame(
        np.tile(np.arange(1.0, 11.0)[:, np.newaxis], (1, 4)),
        columns=["S1", "S2", "S3", "S4"],
    )
    values.loc[0, "S2"] = np.nan

    guarded = abundance_similarity_curve(
        values,
        window_size=10,
        step=1,
        min_pairwise_correlations_per_point=3,
        min_comparators_per_reference=3,
        min_pairwise_observations=10,
    )
    supported = abundance_similarity_curve(
        values,
        window_size=10,
        step=1,
        min_pairwise_correlations_per_point=3,
        min_comparators_per_reference=3,
        min_pairwise_observations=9,
    )

    assert guarded.empty
    assert not supported.empty
    assert supported["n_pairwise_correlations"].ge(3).all()
    assert supported["median_lipids_in_correlation"].ge(9).all()


def test_sustained_crossing_ignores_a_leftmost_j_shaped_spike():
    curve = pd.DataFrame({
        "window_index": range(8),
        "median_log2_abundance": np.arange(10.0, 18.0),
        "median_pearson": [0.31, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
        "median_spearman": [0.31, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
    })

    thresholds = candidate_thresholds(
        curve,
        rules=("pearson_gt_0_25", "pearson_sustained_gt_0_25"),
        sustained_windows=3,
    ).set_index("threshold_rule")

    assert thresholds.loc["pearson_gt_0_25", "threshold_window_index"] == 0
    assert thresholds.loc["pearson_sustained_gt_0_25", "threshold_window_index"] == 3
    assert thresholds.loc["pearson_sustained_gt_0_25", "sustained_windows_required"] == 3


def test_failed_noisyr_status_cannot_supply_a_threshold_or_difference():
    failed_thresholds = pd.DataFrame({"threshold_log2_abundance": [0.0]})

    assert np.isnan(threshold_median(failed_thresholds, "no_finite_noisyr_similarity"))
    assert np.isnan(
        threshold_difference(
            0.0,
            10.0,
            status="no_finite_noisyr_similarity",
        )
    )
    assert threshold_median(
        pd.DataFrame({"threshold_log2_abundance": [10.0, 12.0]}),
        "completed",
    ) == 11.0


def test_panel_scaled_window_has_a_floor_and_records_realized_fraction():
    window, step = panel_scaled_window_parameters(987, 0.025)

    assert window == 25
    assert step == 5
    assert window / 987 == 25 / 987
