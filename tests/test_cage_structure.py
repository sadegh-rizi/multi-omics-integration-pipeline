import numpy as np
import pandas as pd

from mia_multiomics import config, inputs
from mia_multiomics.stats.clustered import (
    cage_group_assignments,
    cage_permutation_ceiling,
    collapse_to_cage_means,
    intraclass_correlation,
    sex_exchangeability_summary,
)
from mia_multiomics.stats.common import exact_empirical_p
from mia_multiomics.stats.permutation import run_permutation_null


def test_cage_join_forward_fills_all_60_samples_and_matches_mouse_prefixes(toy_settings):
    metadata = inputs.load_metadata(toy_settings)
    assert len(metadata) == 60
    assert metadata[config.MOUSE_COL].nunique() == 20
    assert metadata[[config.CAGE_COL, config.DATE_COL, config.COHORT_COL]].notna().all().all()
    prefixes = metadata[config.MOUSE_COL].astype(str).str.split(" - ", n=1).str[0]
    assert set(prefixes) == set(metadata[config.CAGE_COL])
    assert metadata.groupby(config.CAGE_COL)[config.GROUP_COL].nunique().eq(1).all()
    assert metadata.groupby(config.CAGE_COL)[config.DATE_COL].nunique().eq(1).all()


def test_cage_permutation_space_is_28_and_has_exact_floor(toy_settings):
    metadata = inputs.load_metadata(toy_settings)
    ceiling = cage_permutation_ceiling(metadata, fdr_denominator=950)
    assignments = cage_group_assignments(metadata)
    assert ceiling["n_distinct_assignments"] == 28          # 2 control vs 6 MIA litters
    assert ceiling["minimum_empirical_p_value"] == 1 / 28
    assert len(assignments) == 28
    assert all(metadata.loc[a.eq(config.MIA), config.CAGE_COL].nunique() == 6 for a in assignments)
    assert exact_empirical_p(10.0, np.arange(28, dtype=float)) >= 1 / 28
    assert ceiling["minimum_bh_q_value"] > 1


def test_design_reports_mixed_control_cages_for_both_sex_schemes(toy_settings):
    summary = sex_exchangeability_summary(inputs.load_metadata(toy_settings))
    assert summary["within_cage"] == "available"
    assert summary["cage_level"] == "not_identifiable_mixed_sex_cages"
    assert summary["n_mixed_sex_cages"] == 2


def test_cage_means_and_icc_use_only_multi_pup_cages():
    metadata = pd.DataFrame({
        "Sample": [f"S{index}" for index in range(12)],
        "Cage": np.repeat(["A", "B", "C", "D"], 3),
        "Bio group": ["Saline Control"] * 6 + ["MIA"] * 6,
        "Sex": ["Male"] * 12,
        "Date collected": pd.to_datetime(["2024-06-11"] * 12),
        "Collection cohort": pd.Categorical(["2024-06-11"] * 12, ordered=True),
    })
    values = pd.DataFrame(
        {
            sample: [float(cage_number * 2 + replicate * 0.1)]
            for cage_number, (sample, replicate) in enumerate(
                zip(metadata["Sample"], np.tile([0, 1, 2], 4))
            )
        },
        index=["lipid_1"],
    )

    cage_values, cage_metadata = collapse_to_cage_means(values, metadata)
    icc = intraclass_correlation(values, metadata, n_bootstraps=50, random_state=7)

    assert cage_values.shape == (1, 4)
    assert cage_metadata["n_samples"].tolist() == [3, 3, 3, 3]
    assert icc.loc[0, "n_multi_pup_cages"] == 4
    assert icc.loc[0, "icc"] > 0.5


def test_zero_cluster_effect_is_clipped_to_a_finite_nonnegative_design_value():
    metadata = pd.DataFrame({
        "Sample": [f"S{index}" for index in range(12)],
        "Cage": np.repeat(["A", "B", "C", "D"], 3),
        "Bio group": ["Saline Control"] * 6 + ["MIA"] * 6,
    })
    values = pd.DataFrame(
        np.tile([0.0, 1.0, 2.0], (1, 4)),
        index=["lipid_1"],
        columns=metadata["Sample"],
    )

    icc = intraclass_correlation(values, metadata, n_bootstraps=50, random_state=7)

    assert np.isfinite(icc.loc[0, "icc"])
    assert 0 <= icc.loc[0, "icc"] < 0.1


def test_cage_permutation_null_is_exhaustive_and_records_the_assignment_floor():
    cages = [f"C{number}" for number in range(8)]
    rows = []
    for cage_number, cage in enumerate(cages):
        group = "Saline Control" if cage_number < 2 else "MIA"
        for replicate in range(6):
            rows.append({
                "Sample": f"S{cage_number}_{replicate}",
                "Sample type": "Cortex",
                "Bio group": group,
                "Sex": "Male" if replicate < 3 else "Female",
                "Cage": cage,
            })
    metadata = pd.DataFrame(rows)
    values = pd.DataFrame(
        np.arange(len(metadata), dtype=float).reshape(1, -1),
        index=["lipid_1"],
        columns=metadata["Sample"],
    )

    features, summary, effective = run_permutation_null(
        values,
        metadata,
        min_per_cell=3,
        exchangeability_unit="cage",
        block_column="Cage",
        effects=("MIA vs Saline Control",),
    )

    assert set(features["n_distinct_assignments"]) == {28}
    assert features["empirical_p_floor"].eq(1 / 28).all()
    assert features["n_permutations"].eq(28).all()
    assert features["p_value_empirical"].ge(1 / 28).all()
    assert summary["Permutation"].max() == 28
    assert effective.loc[0, "n_distinct_assignments"] == 28
