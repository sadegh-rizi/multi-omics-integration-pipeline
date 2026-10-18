import numpy as np
import pandas as pd

from mia_multiomics.imputation import impute_mnar_halfmin, impute_rf_all, impute_rf_mar_only


def _values_and_mask():
    values = pd.DataFrame({"S1": [5.0, 7.0, 2.0], "S2": [6.0, np.nan, 3.0], "S3": [7.0, 9.0, np.nan], "S4": [8.0, 10.0, 5.0]}, index=["A", "B", "C"])
    mask = pd.DataFrame("observed", index=values.index, columns=values.columns)
    mask.loc["B", "S2"] = "left_censored"
    mask.loc["C", "S3"] = "mar"
    return values, mask


def test_halfmin_and_rf_mar_only_keep_censored_values_below_observed_minimum():
    values, mask = _values_and_mask()
    for imputer in (impute_mnar_halfmin, impute_rf_mar_only):
        completed, _ = imputer(values, mask)
        assert completed.loc["B", "S2"] < values.loc["B"].min()


def test_rf_imputation_is_deterministic_and_routes_only_mar_cells_to_rf():
    values, mask = _values_and_mask()
    first, routes = impute_rf_mar_only(values, mask, random_state=4)
    second, _ = impute_rf_mar_only(values, mask, random_state=4)
    pd.testing.assert_frame_equal(first, second)
    assert routes.loc["B", "S2"] == "halfmin"
    assert routes.loc["C", "S3"] == "rf_mar"


def test_rf_all_marks_censored_cells_as_rf_imputed():
    values, mask = _values_and_mask()
    _, routes = impute_rf_all(values, mask, random_state=4)
    assert routes.loc["B", "S2"] == "rf_all"
