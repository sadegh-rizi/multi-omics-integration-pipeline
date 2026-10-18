import pandas as pd
import pytest

from mia_multiomics.stats.power import detectable_effect_sizes, fdr_hit_table


def test_detectable_effect_sizes_reports_sample_and_illustrative_scales():
    effects = pd.DataFrame({
        "Feature": ["a", "b"],
        "Region": ["Cortex", "Cortex"],
        "Effect": ["MIA vs Saline Control"] * 2,
        "standard_error": [0.3, 0.4],
        "n_usable": [19, 19],
    })

    feature_mde, summary = detectable_effect_sizes(effects)

    assert feature_mde["mde_log2"].iloc[0] == pytest.approx(0.3 * (1.959963984540054 + 0.8416212335729143))
    assert set(summary["power_scale"]) == {"observed_sample_level", "illustrative_litter_n"}
    assert summary.loc[summary["power_scale"].eq("illustrative_litter_n"), "illustrative_litter_n"].iloc[0] == 8


def test_fdr_hit_table_ranks_and_limits_labels_per_region_effect():
    effects = pd.DataFrame({
        "Feature": ["a", "b", "c"],
        "Lipid Name": ["PC(16:0)", "PC(18:1)", "PC(20:4)"],
        "Region": ["Cortex"] * 3,
        "Effect": ["MIA vs Saline Control"] * 3,
        "p_value": [0.001, 0.002, 0.003],
        "p_value_fdr": [0.01, 0.02, 0.06],
    })

    hits, top = fdr_hit_table(effects, top_n=1)

    assert hits["Feature"].tolist() == ["a", "b"]
    assert top["Feature"].tolist() == ["a"]
    assert top["label"].tolist() == ["PC(16:0)"]
