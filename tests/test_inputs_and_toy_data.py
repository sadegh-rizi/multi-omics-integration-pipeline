"""The readers parse the toy files exactly like the real deliverables."""

import pandas as pd

from mia_multiomics import config, inputs
from mia_multiomics.preprocessing import preprocess


def test_lipidomics_reader_drops_embedded_header_rows(toy_settings):
    lipids = inputs.load_lipidomics(toy_settings)
    assert lipids.values.shape[1] == config.N_SAMPLES
    assert not lipids.features["Lipid Name"].isin(["Lipid Name", "Sample Number"]).any()
    assert set(lipids.features["Polarity"].str[-1]) == {"+", "-"}       # both acquisition blocks were read
    assert lipids.blanks.shape[1] == 6 and lipids.techmix.shape[1] == 6
    assert lipids.values.columns.tolist() == lipids.metadata["Sample"].tolist()


def test_all_three_omics_use_the_same_padded_sample_ids(toy_settings):
    ids = [inputs.load(toy_settings, o).values.columns.tolist() for o in config.VIEWS]
    assert ids[0] == ids[1] == ids[2]
    assert ids[0][0] == inputs.sample_id(1) == f"{config.SAMPLE_PREFIX}-01"


def test_metadata_joins_sample_sheet_and_cage_workbook(toy_settings):
    meta = inputs.load_metadata(toy_settings)
    assert meta.groupby(config.REGION_COL).size().eq(20).all()
    assert meta[config.GROUP_COL].value_counts().eq(30).all()
    assert meta[config.CAGE_COL].nunique() == 8
    assert meta.loc[meta["Sample number"].eq(1), config.REGION_COL].item() == "Cortex"


def test_toy_truth_region_strong_and_no_planted_mia_effect(toy_settings):
    prepared = preprocess(inputs.load_lipidomics(toy_settings))
    values, meta = prepared.values, prepared.metadata.set_index("Sample")
    region_means = values.T.groupby(meta[config.REGION_COL]).mean()
    group_means = values.T.groupby(meta[config.GROUP_COL]).mean()
    region_spread = region_means.std().median()
    group_spread = (group_means.loc[config.MIA] - group_means.loc[config.CONTROL]).abs().median()
    assert region_spread > 3 * group_spread


def test_population_samples_remove_flagged_samples_only(toy_settings, tmp_path):
    meta = inputs.load_metadata(toy_settings)
    flags = pd.DataFrame({"Sample": meta["Sample"], "hotelling_t2_95_flag": False, "hotelling_t2_99_flag": False})
    flags.loc[[0, 5], "hotelling_t2_95_flag"] = True
    path = toy_settings.results_dir / "lipidomics" / "02_outliers"
    path.mkdir(parents=True, exist_ok=True)
    flags.to_csv(path / "outlier_screen.csv", index=False)
    kept = inputs.population_samples(toy_settings, meta, "t2_95")
    assert len(kept) == 58 and meta["Sample"][0] not in kept
    # the other regions of the same mouse stay in
    mouse = meta.loc[0, config.MOUSE_COL]
    assert meta.loc[meta[config.MOUSE_COL].eq(mouse) & ~meta.index.isin([0]), "Sample"].isin(kept).all()
    assert len(inputs.population_samples(toy_settings, meta, "all60")) == 60
    (path / "outlier_screen.csv").unlink()
