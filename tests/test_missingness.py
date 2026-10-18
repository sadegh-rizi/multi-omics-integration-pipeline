import numpy as np
import pandas as pd

from mia_multiomics import config
from mia_multiomics.qc import (
    filtering_stage_missingness,
    missingness_by_class,
    missingness_by_feature,
    missingness_by_group,
    missingness_by_sample,
)


def _toy():
    intensity = pd.DataFrame([[1.0, 2.0, 3.0, 4.0], [np.nan, 2.0, 3.0, 4.0], [np.nan, 2.0, np.nan, np.nan],
                              [np.nan, np.nan, np.nan, np.nan]], index=["f0", "f1", "f2", "f3"], columns=["S1", "S2", "S3", "S4"])
    annotations = pd.DataFrame({"Lipid Class": ["C1", "C1", "C2", "C2"]}, index=intensity.index)
    metadata = pd.DataFrame({"Sample": ["S1", "S2", "S3", "S4"],
                             config.GROUP_COL: [config.CONTROL, config.CONTROL, config.MIA, config.MIA]})
    return intensity, annotations, metadata


def test_missingness_per_feature_sample_and_group():
    intensity, annotations, metadata = _toy()
    by_feature = missingness_by_feature(intensity, annotations).set_index("Feature")
    assert by_feature["n_missing"].to_dict() == {"f3": 4, "f2": 3, "f1": 1, "f0": 0}
    by_sample = missingness_by_sample(intensity, metadata).set_index("Sample")
    assert by_sample.loc["S1", "n_missing"] == 3 and by_sample.loc["S2", "n_missing"] == 1
    by_group = missingness_by_group(by_sample.reset_index(), config.GROUP_COL).set_index(config.GROUP_COL)
    assert by_group.loc[config.CONTROL, "mean_pct_missing"] == 50.0


def test_missingness_by_class_runs_a_fisher_test_per_class():
    intensity, annotations, metadata = _toy()
    table = missingness_by_class(intensity, annotations["Lipid Class"], metadata).set_index("Class")
    assert table.loc["C1", f"pct_missing_{config.CONTROL}"] == 25.0
    assert table.loc["C1", f"pct_missing_{config.MIA}"] == 0.0
    assert 0 <= table.loc["C2", "p_value"] <= 1


def test_stage_missingness_reports_group_difference():
    intensity, _, metadata = _toy()
    table = filtering_stage_missingness([("raw", intensity), ("complete", intensity.loc[["f0"]])], metadata)
    assert table.loc[1, "pct_missing"] == 0.0
    assert table.loc[0, "n_features"] == 4
    assert table.loc[0, "group_difference"] == abs(table.loc[0, f"pct_missing_{config.CONTROL}"] - table.loc[0, f"pct_missing_{config.MIA}"])
