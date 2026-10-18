import pandas as pd

from mia_multiomics.stats.diagnostics import missingness_se_crosscheck, summarize_p_values


def test_summarize_p_values_reports_uniform_expectations():
    effects = pd.DataFrame({
        "Region": ["Cortex"] * 4,
        "Effect": ["MIA vs Saline Control"] * 4,
        "p_value": [0.001, 0.02, 0.7, 0.9],
    })

    result = summarize_p_values(effects, source="filtered")

    row = result.loc[result["p_threshold"].eq(0.05)].iloc[0]
    assert row["n_features"] == 4
    assert row["observed_count"] == 2
    assert row["expected_uniform_count"] == 0.2


def test_missingness_se_crosscheck_preserves_distinct_missingness_sources():
    feature_qc = pd.DataFrame({
        "Feature": ["a", "b", "c"],
        "Region": ["Cortex"] * 3,
        "n_missing_after_preprocessing": [0, 1, 2],
    })
    effects = pd.DataFrame({
        "Feature": ["a", "b", "c"],
        "Region": ["Cortex"] * 3,
        "Effect": ["MIA vs Saline Control"] * 3,
        "standard_error": [0.2, 0.3, 0.4],
    })
    detection = pd.DataFrame({
        "Feature": ["b", "c"],
        "Sample type": ["Cortex", "Cortex"],
    })

    result = missingness_se_crosscheck(feature_qc, effects, detection)

    assert set(result["metric"]) == {
        "standard_error_vs_n_missing_after_preprocessing_spearman_r",
        "standard_error_vs_n_detection_masked_spearman_r",
    }
    assert result.set_index("metric").loc[
        "standard_error_vs_n_missing_after_preprocessing_spearman_r", "n_features"
    ] == 3
    assert result.set_index("metric").loc[
        "standard_error_vs_n_detection_masked_spearman_r", "n_features"
    ] == 2
