import numpy as np
import pandas as pd

from mia_multiomics.stats.permanova import permanova_by_region


def test_permanova_returns_three_terms_and_missing_feature_diagnostics():
    metadata = pd.DataFrame({
        "Sample": [f"S{i}" for i in range(12)],
        "Sample type": ["Cortex"] * 12,
        "Bio group": ["Saline Control"] * 6 + ["MIA"] * 6,
        "Sex": ["Male", "Female"] * 6,
    })
    values = pd.DataFrame(np.arange(48, dtype=float).reshape(4, 12), index=["a", "b", "c", "missing"], columns=metadata["Sample"])
    values.loc["missing", "S0"] = np.nan
    results, diagnostics = permanova_by_region(values, metadata, n_permutations=19, random_state=1)
    assert results["Term"].tolist() == ["BioGroup", "Sex", "BioGroup:Sex"]
    assert results["p_value"].between(0, 1).all()
    assert diagnostics.loc[0, "n_features_excluded_missing"] == 1
