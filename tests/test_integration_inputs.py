from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mia_multiomics import config
from mia_multiomics.inputs import check_design, read_perseus_proteomics, sample_id
from mia_multiomics.integration.inference import mouse_grouped_folds, treatment_score_test
from mia_multiomics.stages.integration import _marker_checks


def test_perseus_reader_skips_metadata_rows_and_aligns_sample_numbers(tmp_path: Path):
    source = tmp_path / "proteomics_Cerebellum.txt"
    source.write_text(
        "LAB_Cerebellum_MIA_05 MaxLFQ Unique Intensity\t"
        "LAB_Cerebellum_control_02 MaxLFQ Unique Intensity\tProtein ID\tGene\n"
        "#!{Type}\tE\tT\tT\n"
        "16.0\t17.0\tsp|P1|X\tGabra6\n"
        "18.0\t19.0\tsp|P2|Y\tCalb1\n",
        encoding="utf-8",
    )
    values, annotations = read_perseus_proteomics(source, region="Cerebellum")
    assert list(values.columns) == [sample_id(5), sample_id(2)]
    assert list(values.index) == ["sp|P1|X", "sp|P2|Y"]
    assert annotations.loc["sp|P1|X", "Gene"] == "Gabra6"
    assert values.loc["sp|P2|Y", sample_id(2)] == 19.0


def test_perseus_reader_rejects_a_file_from_another_region(tmp_path: Path):
    source = tmp_path / "proteomics_Cortex.txt"
    source.write_text("LAB_Cortex_MIA_05 MaxLFQ Unique Intensity\tProtein ID\tGene\n16.0\tP1\tA\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Cerebellum"):
        read_perseus_proteomics(source, region="Cerebellum")


def _metadata() -> pd.DataFrame:
    rows = []
    for mouse in range(20):
        for region in config.REGIONS:
            number = mouse * 3 + {"Cerebellum": 1, "Cortex": 2, "Hippocampus": 3}[region]
            rows.append({
                "Sample": sample_id(number), "Sample number": number, config.REGION_COL: region,
                config.MOUSE_COL: f"C{mouse // 2} - {mouse % 2 + 1}",
                config.GROUP_COL: config.MIA if mouse >= 10 else config.CONTROL,
                config.SEX_COL: "Female" if mouse % 2 else "Male",
                config.CAGE_COL: f"C{mouse // 2}", config.DATE_COL: "2024-06-11", "Tissue mass (mg)": 1.0,
            })
    return pd.DataFrame(rows)


def test_design_check_requires_three_regions_per_mouse():
    metadata = _metadata()
    check_design(metadata)
    with pytest.raises(ValueError, match="60 samples"):
        check_design(metadata.drop(index=0))


def test_design_check_rejects_a_cage_with_two_groups():
    metadata = _metadata()
    metadata.loc[metadata[config.CAGE_COL].eq("C0") & metadata[config.MOUSE_COL].str.endswith("2"), config.GROUP_COL] = config.MIA
    with pytest.raises(ValueError, match="one treatment group"):
        check_design(metadata)


def test_marker_checks_use_within_mouse_region_contrasts():
    metadata = _metadata()
    proteins = pd.DataFrame(0.0, index=metadata["Sample"], columns=["P1"])
    proteins.loc[metadata.loc[metadata[config.REGION_COL].eq("Cerebellum"), "Sample"], "P1"] = 2.0
    annotation = pd.DataFrame({"Gene": ["Gabra6"]}, index=["P1"])
    checks = _marker_checks(proteins, annotation, metadata).set_index("Gene")
    assert checks.loc["Gabra6", "direction_matches"]
    assert np.isclose(checks.loc["Gabra6", "mean_target_minus_other_log2"], 2.0)
    assert checks.loc["Satb2", "status"] == "not_detected"


def test_grouped_folds_preserve_mouse_boundaries():
    metadata = _metadata()
    folds = mouse_grouped_folds(metadata, n_splits=5, n_repeats=2, random_state=4)
    assert len(folds) == 10
    for train, test in folds:
        assert not set(metadata.iloc[train][config.MOUSE_COL]) & set(metadata.iloc[test][config.MOUSE_COL])


def test_treatment_test_uses_cage_assignments_and_mouse_means():
    metadata = _metadata()
    scores = pd.Series(np.where(metadata[config.GROUP_COL].eq(config.MIA), 1.0, 0.0) + np.linspace(-0.1, 0.1, len(metadata)))
    result = treatment_score_test(metadata, scores, score_name="synthetic", n_mouse_permutations=50)
    assert result["n_cage_assignments"] == 252          # C(10, 5) whole-cage assignments
    assert np.isclose(result["cage_p_floor"], 1 / 252)
    assert result["cage_permutation_p"] <= 2 / 252     # a perfect separation sits at the floor
    assert 0 < result["mouse_permutation_p"] <= 1
