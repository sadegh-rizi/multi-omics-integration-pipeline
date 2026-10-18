"""End-to-end smoke tests: the cheap stages run on the toy data and find the planted truth."""

import pandas as pd

from mia_multiomics import config
from mia_multiomics.stages import differential, effect_sizes, outliers, qc, region_persistence


def test_qc_stage_finds_region_as_the_dominant_axis(toy_settings):
    headline = qc.run(toy_settings, "proteomics")
    assert headline["pc1_region_r2"] > 0.8
    assert headline["max_group_r2_pc1_5"] < 0.5


def test_outlier_screen_flags_a_planted_outlier(toy_settings):
    screen = outliers.run_screen(toy_settings, "lipidomics")
    assert len(screen) == config.N_SAMPLES
    flagged = set(screen.loc[screen["hotelling_t2_95_flag"], "Sample number"])
    assert {10, 45} & flagged           # samples 10 and 45 were made outlying by the simulator


def test_differential_light_run_writes_effects_for_every_region(toy_settings):
    effects = differential.run(toy_settings, "metabolomics", "all60", light=True)
    assert set(effects["Region"]) == set(config.REGIONS)
    assert set(effects["Effect"]) == {"MIA vs Saline Control", "Female vs Male", "MIA x Female interaction"}
    assert effects["fdr_denominator"].nunique() == 1


def test_region_profiles_persist_in_the_toy_data(toy_settings):
    table = region_persistence.run(toy_settings.with_(n_split_halves=20, n_bootstraps=20), "metabolomics")
    assert (table["disattenuated_concordance"] > 0.7).all()


def test_mia_effect_sizes_are_small(toy_settings):
    tables = effect_sizes.run(toy_settings.with_(n_mouse_permutations=99))
    assert isinstance(tables["lipid_composites"], pd.DataFrame)
    assert tables["effects_ruled_out"]["median_abs_fold_change"].lt(1.25).all()
