import numpy as np
import pandas as pd

from mia_multiomics.stats.common import bh_adjust as _bh_adjust
from mia_multiomics.stats.differential import fit_regionwise_differential


def _metadata() -> pd.DataFrame:
    rows = []
    for region in ("Cortex", "Hippocampus"):
        for group in ("Saline Control", "MIA"):
            for sex in ("Male", "Female"):
                for replicate in range(3):
                    rows.append({"Sample": f"{region[:1]}_{group[:1]}_{sex[:1]}_{replicate}", "Sample type": region, "Bio group": group, "Sex": sex, "Mouse number": f"{region}_{group}_{sex}_{replicate}"})
    return pd.DataFrame(rows)


def test_differential_returns_effects_and_missingness_qc():
    metadata = _metadata()
    values = pd.DataFrame([np.arange(len(metadata), dtype=float), np.ones(len(metadata))], index=["lipid_ok", "lipid_missing"], columns=metadata["Sample"])
    values.loc["lipid_missing", metadata.iloc[0]["Sample"]] = np.nan
    result = fit_regionwise_differential(values, metadata, min_per_cell=3)
    assert set(result.effects["Effect"]) == {"MIA vs Saline Control", "Female vs Male", "MIA x Female interaction"}
    assert set(result.effects.query("Region == 'Cortex'")["Feature"]) == {"lipid_ok"}
    assert "lipid_missing" in set(result.effects.query("Region == 'Hippocampus'")["Feature"])
    excluded = result.feature_qc.query("Feature == 'lipid_missing' and Region == 'Cortex'").iloc[0]
    assert excluded["n_missing_after_preprocessing"] == 1
    assert excluded["fit_status"] == "excluded"
    assert result.effects["p_value_fdr"].notna().all()


def test_bh_adjust_preserves_nan_p_values_without_poisoning_finite_values():
    p_values = pd.Series([0.001, 0.02, 0.5, np.nan], index=list("abcd"))

    adjusted = _bh_adjust(p_values, denominator=4)

    np.testing.assert_allclose(adjusted.loc[["a", "b", "c"]], [0.004, 0.04, 2 / 3])
    assert pd.isna(adjusted.loc["d"])


def test_pareto_effects_do_not_report_fold_changes():
    metadata = _metadata()
    values = pd.DataFrame(
        [np.arange(len(metadata), dtype=float)],
        index=["lipid_ok"],
        columns=metadata["Sample"],
    )

    result = fit_regionwise_differential(
        values,
        metadata,
        min_per_cell=3,
        effect_scale="pareto_scaled_log2",
    )

    assert set(result.effects["effect_scale"]) == {"pareto_scaled_log2"}
    assert result.effects["fold_change"].isna().all()
