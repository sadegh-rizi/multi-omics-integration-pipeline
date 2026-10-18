import numpy as np
import pandas as pd
import pytest

from mia_multiomics import config
from mia_multiomics.inputs import OmicsData
from mia_multiomics.preprocessing import (
    autoscale,
    below_detection_reasons,
    blank_dominated,
    drop_zero_variance,
    exclude_nonpositive,
    filter_low_iqr,
    for_multivariate,
    log2_transform,
    median_normalize,
    missing_fraction,
    pareto_scale,
    preprocess,
)


def exclude_blank_dominated_features(intensity, blanks):
    return intensity.drop(index=blank_dominated(intensity, blanks))


def exclude_below_detection(intensity, blanks):
    return intensity.mask(below_detection_reasons(intensity, blanks).notna())


def exclude_low_prevalence_features(values, max_missing_fraction):
    return values.loc[missing_fraction(values).le(max_missing_fraction)]


@pytest.fixture
def raw_intensity() -> pd.DataFrame:
    """3 lipids x 4 samples with a zero, a negative, and a NaN planted
    deliberately, mirroring the real workbook's shape (lipids as rows,
    samples as columns)."""
    return pd.DataFrame(
        {
            "S1": [100.0, 200.0, 0.0],
            "S2": [110.0, 190.0, 5.0],
            "S3": [90.0, -10.0, 8.0],
            "S4": [105.0, 210.0, np.nan],
        },
        index=["LipidA", "LipidB", "LipidC"],
    )


@pytest.fixture
def normalized_with_flat_feature() -> pd.DataFrame:
    """Already normalized-scale values (as if post log2 + median-normalize)
    with one feature deliberately constant across samples. Note: a feature
    that is constant in *raw* units is generally NOT constant after
    median-normalization, since that step subtracts a different offset per
    sample -- so the zero-variance guard is tested here directly on
    normalized-scale data, isolated from the normalization step itself."""
    return pd.DataFrame(
        {
            "S1": [0.10, -0.30, 0.00],
            "S2": [0.20, 0.10, 0.00],
            "S3": [-0.15, 0.25, 0.00],
            "S4": [0.05, -0.05, 0.00],
        },
        index=["LipidVar1", "LipidVar2", "LipidFlat"],
    )


@pytest.fixture
def blank_intensity(raw_intensity: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        0.0,
        index=raw_intensity.index,
        columns=["Blank1", "Blank2", "Blank3"],
    )


def test_exclude_nonpositive_masks_bad_values(raw_intensity):
    result = exclude_nonpositive(raw_intensity)
    assert np.isnan(result.loc["LipidB", "S3"])  # negative -> NaN
    assert np.isnan(result.loc["LipidC", "S1"])  # zero -> NaN
    assert np.isnan(result.loc["LipidC", "S4"])  # NaN stays NaN
    assert result.loc["LipidA", "S1"] == 100.0  # untouched positive value


def test_exclude_blank_dominated_features_removes_only_features_with_blank_signal():
    intensity = pd.DataFrame({"S1": [50.0, 50.0, 50.0], "S2": [55.0, 55.0, 55.0]}, index=["dominated", "clean", "zero_blank"])
    blanks = pd.DataFrame({"B1": [20.0, 1.0, 0.0], "B2": [20.0, 1.0, 0.0]}, index=intensity.index)

    filtered = exclude_blank_dominated_features(intensity, blanks)

    assert filtered.index.tolist() == ["clean", "zero_blank"]


def test_exclude_below_detection_masks_blank_derived_and_absolute_floor_values():
    intensity = pd.DataFrame(
        {
            "S1": [50.0, 61.0, 0.5, 5.0],
            "S2": [70.0, 75.0, 2.0, 6.0],
        },
        index=["below_blank", "above_blank", "below_floor", "zero_blank_normal"],
    )
    blanks = pd.DataFrame(
        {
            "Blank1": [20.0, 20.0, 0.0, 0.0],
            "Blank2": [20.0, 20.0, 0.0, 0.0],
        },
        index=intensity.index,
    )

    reasons = below_detection_reasons(intensity, blanks)
    result = exclude_below_detection(intensity, blanks)

    assert reasons.loc["below_blank", "S1"] == "below_blank_limit"
    assert pd.isna(reasons.loc["below_blank", "S2"])
    assert reasons.loc["below_floor", "S1"] == "below_absolute_floor"
    assert pd.isna(reasons.loc["zero_blank_normal", "S1"])
    assert np.isnan(result.loc["below_blank", "S1"])
    assert np.isnan(result.loc["below_floor", "S1"])
    assert result.loc["above_blank", "S1"] == 61.0
    assert result.loc["zero_blank_normal", "S1"] == 5.0
    assert result.index.tolist() == intensity.index.tolist()
    assert result.columns.tolist() == intensity.columns.tolist()


def test_exclude_below_detection_preserves_existing_missing_values(raw_intensity, blank_intensity):
    result = exclude_below_detection(exclude_nonpositive(raw_intensity), blank_intensity)
    assert np.isnan(result.loc["LipidB", "S3"])
    assert np.isnan(result.loc["LipidC", "S1"])
    assert np.isnan(result.loc["LipidC", "S4"])


def test_exclude_low_prevalence_features_uses_strict_missing_fraction_threshold():
    values = pd.DataFrame(
        {"S1": [1.0, 1.0, np.nan], "S2": [np.nan, 1.0, np.nan], "S3": [1.0, 1.0, 1.0], "S4": [1.0, 1.0, np.nan], "S5": [1.0, np.nan, np.nan]},
        index=["at_threshold", "complete", "above_threshold"],
    )

    filtered = exclude_low_prevalence_features(values, max_missing_fraction=0.20)

    assert filtered.index.tolist() == ["at_threshold", "complete"]


def test_log2_transform_matches_manual_calculation(raw_intensity):
    positive = exclude_nonpositive(raw_intensity)
    result = log2_transform(positive)
    assert result.loc["LipidA", "S1"] == pytest.approx(np.log2(100.0))
    assert np.isnan(result.loc["LipidC", "S1"])


def test_log2_transform_rejects_nonpositive_input(raw_intensity):
    with pytest.raises(ValueError):
        log2_transform(raw_intensity)  # forgot exclude_nonpositive first


def test_median_normalize_centers_each_sample_median_at_zero(raw_intensity):
    normalized = median_normalize(log2_transform(exclude_nonpositive(raw_intensity)))
    for sample in normalized.columns:
        assert normalized[sample].median(skipna=True) == pytest.approx(0.0, abs=1e-9)


def test_drop_zero_variance_features_removes_flat_row(normalized_with_flat_feature):
    filtered = drop_zero_variance(normalized_with_flat_feature)
    assert "LipidFlat" not in filtered.index
    assert "LipidVar1" in filtered.index
    assert "LipidVar2" in filtered.index


def test_autoscale_gives_each_feature_zero_mean_unit_variance(normalized_with_flat_feature):
    scaled = autoscale(drop_zero_variance(normalized_with_flat_feature))
    for feature in scaled.index:
        row = scaled.loc[feature].dropna()
        assert row.mean() == pytest.approx(0.0, abs=1e-9)
        assert row.std(ddof=1) == pytest.approx(1.0, abs=1e-9)


def test_autoscale_rejects_zero_variance_feature(normalized_with_flat_feature):
    with pytest.raises(ValueError):
        autoscale(normalized_with_flat_feature)  # LipidFlat is still present


def test_filter_low_iqr_features_removes_bottom_fraction_deterministically():
    values = pd.DataFrame(
        {
            "S1": [1.0, 1.0, 1.0, 1.0],
            "S2": [1.0, 2.0, 3.0, 4.0],
            "S3": [1.0, 3.0, 5.0, 7.0],
            "S4": [1.0, 4.0, 7.0, 10.0],
        },
        index=["flat", "low", "high", "highest"],
    )

    filtered, table = filter_low_iqr(values, remove_fraction=0.25)

    assert filtered.index.tolist() == ["low", "high", "highest"]
    assert table.loc[table["removed"], "Feature"].tolist() == ["flat"]


def test_pareto_scale_mean_centers_and_uses_square_root_sd(normalized_with_flat_feature):
    scaled = pareto_scale(drop_zero_variance(normalized_with_flat_feature))
    for feature in scaled.index:
        source = normalized_with_flat_feature.loc[feature]
        expected_sd = np.sqrt(source.std(ddof=1))
        assert scaled.loc[feature].mean() == pytest.approx(0.0, abs=1e-9)
        assert scaled.loc[feature].std(ddof=1) == pytest.approx(np.sqrt(source.std(ddof=1)), abs=1e-9)
        assert scaled.loc[feature, "S1"] == pytest.approx((source["S1"] - source.mean()) / expected_sd)


def _omics(raw, blanks):
    metadata = pd.DataFrame({"Sample": raw.columns, config.REGION_COL: "Cortex"})
    features = pd.DataFrame({"Lipid Name": raw.index, "Lipid Class": "PC"}, index=raw.index)
    return OmicsData("lipidomics", raw, features, metadata, blanks=blanks, techmix=blanks)


def test_preprocess_matches_stepwise_calls_and_logs_every_removed_feature(raw_intensity, blank_intensity):
    result = preprocess(_omics(raw_intensity, blank_intensity), max_missing_fraction=0.25)
    # LipidB misses 1 of 4 values (25 %, kept at the threshold); LipidC misses 2 of 4 (removed)
    kept = ["LipidA", "LipidB"]
    stepwise = median_normalize(log2_transform(exclude_below_detection(exclude_nonpositive(raw_intensity.loc[kept]), blank_intensity.loc[kept])))
    pd.testing.assert_frame_equal(result.values, drop_zero_variance(stepwise))
    strict = preprocess(_omics(raw_intensity, blank_intensity), max_missing_fraction=0.20)
    assert set(strict.exclusions.loc[strict.exclusions["stage"].eq("prevalence"), "Feature"]) == {"LipidB", "LipidC"}
    # a single remaining feature is all zeros after median normalisation, so it is logged as zero-variance
    assert strict.exclusions.loc[strict.exclusions["stage"].eq("zero_variance"), "Feature"].tolist() == ["LipidA"]


def test_for_multivariate_scales_every_remaining_feature(raw_intensity, blank_intensity):
    result = for_multivariate(preprocess(_omics(raw_intensity, blank_intensity), max_missing_fraction=None).values)
    assert not result.empty and not result.isna().any().any()
    for feature in result.index:
        assert result.loc[feature].std(ddof=1) == pytest.approx(1.0, abs=1e-9)
