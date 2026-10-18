import numpy as np
import pandas as pd

from mia_multiomics.lipids import (
    aggregate_class_profiles,
    annotate_lipid_structures,
    parse_lipid_structure,
    structural_index_profiles,
)


def test_parse_lipid_structure_handles_sum_and_chain_composition_names():
    parsed = parse_lipid_structure("PC(O-18:1_20:4)")
    assert parsed["chain_lengths"] == (18, 20)
    assert parsed["double_bonds"] == (1, 4)
    assert parsed["total_chain_length"] == 38
    assert parsed["total_double_bond_count"] == 5

    sum_composition = parse_lipid_structure("PC(38:4)")
    assert sum_composition["chain_lengths"] == (38,)
    assert sum_composition["double_bonds"] == (4,)


def test_class_profiles_use_unweighted_log_scale_summaries():
    values = pd.DataFrame(
        {"S1": [1.0, 3.0, 10.0], "S2": [2.0, np.nan, 20.0]},
        index=["a", "b", "c"],
    )
    annotations = pd.DataFrame(
        {
            "Lipid Name": ["PC(16:0)", "PC(18:1)", "TG(16:0_18:1_18:1)"],
            "Lipid Class": ["PC", "PC", "TG"],
        },
        index=values.index,
    )

    means, mean_annotations = aggregate_class_profiles(values, annotations, statistic="mean")
    medians, _ = aggregate_class_profiles(values, annotations, statistic="median")

    assert means.loc["class_mean::PC", "S1"] == 2.0
    assert means.loc["class_mean::PC", "S2"] == 2.0
    assert medians.loc["class_median::PC", "S1"] == 2.0
    assert mean_annotations.loc["class_mean::PC", "Aggregation"] == "log2_mean"


def test_structural_profiles_keep_missing_samples_as_nan():
    values = pd.DataFrame(
        {"S1": [1.0, 2.0], "S2": [np.nan, np.nan]},
        index=["sat", "pufa"],
    )
    annotations = pd.DataFrame(
        {
            "Lipid Name": ["PC(16:0_18:0)", "PC(18:1_18:2)"],
            "Lipid Class": ["PC", "PC"],
        },
        index=values.index,
    )

    profiles, profile_annotations, long_values = structural_index_profiles(values, annotations)

    assert profiles.loc["structural::PC::mean_double_bond_count", "S1"] == 1.5
    assert profiles.loc["structural::PC::pufa_fraction", "S1"] == 0.5
    assert np.isnan(profiles.loc["structural::PC::pufa_fraction", "S2"])
    assert profile_annotations.loc["structural::Overall::pufa_fraction", "Scope"] == "Overall"
    assert set(long_values["Sample"]) == {"S1"}
    pufa_long = long_values.loc[long_values["Metric"].eq("pufa_fraction")]
    assert set(pufa_long["n_determinate_pufa_features"]) == {2}
    assert set(pufa_long["n_ambiguous_pufa_features"]) == {0}


def test_structural_pufa_fraction_does_not_infer_sum_composition():
    values = pd.DataFrame({"S1": [1.0]}, index=["sum"])
    annotations = pd.DataFrame({"Lipid Name": ["PC(38:2)"], "Lipid Class": ["PC"]}, index=values.index)

    profiles, _, long_values = structural_index_profiles(values, annotations)
    parsed = annotate_lipid_structures(annotations)

    assert bool(parsed.loc["sum", "structural_ambiguous"])
    assert np.isnan(profiles.loc["structural::PC::pufa_fraction", "S1"])
    pufa_long = long_values.loc[long_values["Metric"].eq("pufa_fraction")]
    assert set(pufa_long["n_determinate_pufa_features"]) == {0}
    assert set(pufa_long["n_ambiguous_pufa_features"]) == {1}
