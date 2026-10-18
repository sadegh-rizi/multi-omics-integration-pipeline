import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from mia_multiomics.multivariate.asca import (
    EFFECT_ORDER,
    effect_svd,
    effect_variance_table,
    fit_asca,
    procrustes_align,
)
from mia_multiomics.stats.asca_permutation import (
    cage_group_permutations,
    control_cage_sex_permutations,
    effect_ssq_by_scaling,
    enumerate_cage_assignment_count,
    make_spike_profile,
    mouse_group_permutations,
    region_label_permutations,
    stratified_mouse_bootstrap_indices,
)


def _metadata(n_mice: int = 8) -> pd.DataFrame:
    rows = []
    for number in range(n_mice):
        group = "MIA" if number in {4, 5, 6, 7} else "Saline Control"
        sex = "Female" if number in {2, 3, 6, 7} else "Male"
        for region in ("Cerebellum", "Cortex", "Hippocampus"):
            rows.append({
                "Sample": f"S{number}_{region[:2]}",
                "Sample type": region,
                "Bio group": group,
                "Sex": sex,
                "Mouse number": f"M{number}",
                "Cage": f"C{number}",
            })
    return pd.DataFrame(rows)


def _permutation_metadata() -> pd.DataFrame:
    metadata = _metadata()
    metadata["Bio group"] = np.where(
        metadata["Mouse number"].isin({"M0", "M2"}), "MIA", "Saline Control"
    )
    return metadata


def _values(metadata: pd.DataFrame, n_features: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(20261008)
    matrix = rng.normal(size=(n_features, len(metadata)))
    return pd.DataFrame(matrix, index=[str(i) for i in range(n_features)], columns=metadata["Sample"])


def test_balanced_design_decomposes_centered_data_and_is_orthogonal():
    metadata = _metadata()
    fit = fit_asca(_values(metadata), metadata, scaling="unscaled")

    reconstructed = sum(fit.effects.values(), np.zeros_like(fit.centered_data)) + fit.residual
    np.testing.assert_allclose(reconstructed, fit.centered_data, atol=1e-10)
    for left_index, left in enumerate(EFFECT_ORDER):
        for right in EFFECT_ORDER[left_index + 1:]:
            assert abs(float(np.sum(fit.effects[left] * fit.effects[right]))) < 1e-10
    assert fit.design.shape[1] == fit.model_rank
    np.testing.assert_allclose(
        effect_variance_table(fit)["variance_share"].sum(), 1.0, atol=1e-10
    )


def test_unbalanced_fit_matches_statsmodels_fitted_values():
    metadata = _metadata().drop(index=[0, 7]).reset_index(drop=True)
    values = _values(_metadata())
    values = values.loc[:, metadata["Sample"]]
    fit = fit_asca(values, metadata, scaling="unscaled")
    long = metadata.copy()
    long = long.rename(columns={"Sample type": "Region", "Bio group": "Group", "Mouse number": "Mouse"})
    long["y"] = values.iloc[0].to_numpy()
    model = smf.ols(
        "y ~ C(Region, Sum) * C(Group, Sum) * C(Sex, Sum) + C(Mouse, Sum)",
        data=long,
    ).fit()
    ours_fitted = fit.prepared_data[:, 0] - fit.raw_residual[:, 0]
    expected_fitted = model.fittedvalues.to_numpy() - long["y"].mean()
    np.testing.assert_allclose(ours_fitted, expected_fitted, atol=1e-10)


def test_svd_and_procrustes_have_declared_component_count():
    metadata = _metadata()
    fit = fit_asca(_values(metadata), metadata)
    scores, loadings, singular_values = effect_svd(fit, "Region:Group")
    assert list(scores.columns) == ["Sample", "PC1", "PC2"]
    assert loadings.shape == (12, 2)
    assert len(singular_values) == 2
    aligned = procrustes_align(loadings.to_numpy(), loadings.to_numpy())
    np.testing.assert_allclose(aligned, loadings.to_numpy(), atol=1e-10)


def test_permutation_units_preserve_the_declared_structure():
    metadata = _permutation_metadata()
    assert enumerate_cage_assignment_count(metadata) == 28
    original_by_mouse = metadata.groupby("Mouse number", observed=True)["Sample type"].apply(set)
    for _, permuted in region_label_permutations(metadata, n_random=10, random_state=1):
        current = permuted.groupby("Mouse number", observed=True)["Sample type"].apply(set)
        assert current.to_dict() == original_by_mouse.to_dict()
    for _, permuted in mouse_group_permutations(metadata, n_random=10, random_state=2):
        counts = permuted.drop_duplicates("Mouse number").groupby("Sex")["Bio group"].value_counts()
        original = metadata.drop_duplicates("Mouse number").groupby("Sex")["Bio group"].value_counts()
        assert counts.to_dict() == original.to_dict()
    for _, permuted in control_cage_sex_permutations(metadata, n_random=10, random_state=3):
        mia_original = metadata.loc[metadata["Bio group"].eq("MIA"), "Sex"].to_numpy()
        mia_current = permuted.loc[permuted["Bio group"].eq("MIA"), "Sex"].to_numpy()
        assert np.array_equal(mia_original, mia_current)
    assert all(len(list(cage_group_permutations(metadata))) == 28 for _ in range(1))


def test_bootstrap_retains_complete_mouse_clusters_and_spike_profile():
    metadata = _metadata()
    indices = stratified_mouse_bootstrap_indices(metadata, random_state=4)
    boot = metadata.iloc[indices]
    assert len(boot) == len(metadata)
    assert all(len(rows) % 3 == 0 for _, rows in boot.groupby("Mouse number", observed=True))
    profile = make_spike_profile(metadata)
    assert profile[metadata["Sample type"].eq("Cortex")].mean() == 0.5
    assert profile[metadata["Sample type"].eq("Cerebellum")].mean() == -0.5


def test_multi_scaling_ssq_helper_uses_one_design_fit():
    metadata = _metadata()
    values = _values(metadata)
    fits = {
        "unscaled": fit_asca(values, metadata, scaling="unscaled"),
        "autoscaled": fit_asca(values, metadata, scaling="autoscaled"),
    }
    result = effect_ssq_by_scaling(
        {name: fit.prepared_data for name, fit in fits.items()},
        metadata,
        ("Region", "Region:Group"),
    )
    assert set(result) == set(fits)
    assert all(set(stats) == {"Region", "Region:Group"} for stats in result.values())
