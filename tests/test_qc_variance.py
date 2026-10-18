import numpy as np
import pandas as pd
import pytest

from mia_multiomics import config
from mia_multiomics.qc import qc_injection_cv, variance_ratio_table


def test_qc_injection_cv_normalizes_injection_scale_before_calculating_cv():
    techmix = pd.DataFrame({"QC1": [10.0], "QC2": [20.0], "QC3": [30.0], "QC4": [40.0]}, index=["stable_after_normalization"])
    result = qc_injection_cv(techmix, min_injections=4)
    assert result.loc["stable_after_normalization", "raw_cv_percent"] == pytest.approx(np.std([10, 20, 30, 40], ddof=1) / 25 * 100)
    assert result.loc["stable_after_normalization", "normalized_cv_percent"] == pytest.approx(0.0)


def test_qc_injection_cv_marks_features_with_too_few_injections_not_assessable():
    techmix = pd.DataFrame({"QC1": [100.0, 10.0], "QC2": [200.0, 20.0], "QC3": [300.0, np.nan], "QC4": [400.0, np.nan]},
                           index=["reference", "too_few"])
    result = qc_injection_cv(techmix, min_injections=4)
    assert result.loc["too_few", "cv_status"] == "not_assessable"
    assert np.isnan(result.loc["too_few", "normalized_cv_percent"])


def test_variance_ratio_is_region_specific_and_needs_enough_values():
    columns = [f"S{number}" for number in range(8)]
    metadata = pd.DataFrame({"Sample": columns, config.REGION_COL: ["Cortex"] * 4 + ["Hippocampus"] * 4})
    samples = pd.DataFrame([np.arange(8, dtype=float), [1.0, 2.0, 1.0, 2.0, np.nan, np.nan, np.nan, np.nan]],
                           index=["variable", "missing"], columns=columns)
    techmix = pd.DataFrame([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 1.0, 1.0, 1.0, np.nan, np.nan]],
                           index=["variable", "missing"], columns=[f"T{i}" for i in range(6)])
    table = variance_ratio_table(techmix, samples, metadata, min_injections=4, min_per_region=4)
    cortex = table.query("Feature == 'variable' and Region == 'Cortex'").iloc[0]
    assert cortex["n_biological_used"] == 4 and cortex["n_techmix_used"] == 6
    assert np.isclose(cortex["variance_ratio"], np.var(np.arange(6.0), ddof=1) / np.var([0.0, 1.0, 2.0, 3.0], ddof=1))
    assert bool(cortex["technical_dominated"])
    missing = table.query("Feature == 'missing' and Region == 'Hippocampus'").iloc[0]
    assert np.isnan(missing["biological_sd_log2"])
    assert pd.isna(missing["technical_dominated"])
