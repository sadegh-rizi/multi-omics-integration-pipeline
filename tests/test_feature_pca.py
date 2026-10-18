import numpy as np
import pandas as pd

from mia_multiomics.multivariate.feature_modules import (
    assign_hierarchical_modules,
    bootstrap_feature_loadings,
    bootstrap_module_comembership,
    fit_feature_margin_svd,
    fit_sample_space_loadings,
    loading_annotation_composition,
    loading_stability_summary,
    module_annotation_composition,
    module_stability_summary,
    spearman_correlation,
)


def _values(array: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        array,
        index=[f"Metabolite_{number}" for number in range(array.shape[0])],
        columns=[f"Sample_{number}" for number in range(array.shape[1])],
    )


def test_sample_space_loadings_match_transposed_svd_feature_side_up_to_sign():
    generator = np.random.default_rng(20260908)
    values = _values(generator.normal(size=(7, 6)))
    loadings, scree = fit_sample_space_loadings(values, n_components=2, scale_features=False)
    centered = values.to_numpy() - values.to_numpy().mean(axis=1, keepdims=True)
    _, transposed_singular_values, transposed_right = np.linalg.svd(centered.T, full_matrices=False)
    transposed_feature_scores = transposed_right.T[:, :2]

    for component in range(2):
        assert abs(np.corrcoef(loadings.iloc[:, component], transposed_feature_scores[:, component])[0, 1]) > 0.999999
    np.testing.assert_allclose(scree["singular_value"], transposed_singular_values[:2])


def test_planted_module_has_high_bootstrap_comembership():
    generator = np.random.default_rng(11)
    latent = generator.normal(size=18)
    array = generator.normal(size=(12, 18))
    array[:4] = latent + generator.normal(scale=0.06, size=(4, 18))
    values = _values(array)

    correlation = spearman_correlation(values)
    modules = assign_hierarchical_modules(correlation, distance_cut=0.30, signed=True)
    assert modules.iloc[:4].nunique() == 1

    comembership = bootstrap_module_comembership(
        values,
        n_bootstraps=40,
        random_state=12,
        distance_cut=0.30,
        signed=True,
    )
    summary = module_stability_summary(
        modules,
        comembership,
        stability_floor=0.80,
        minimum_size=3,
    )
    planted_module = modules.iloc[0]
    planted_summary = summary.loc[summary["module"].eq(planted_module)].iloc[0]
    assert planted_summary["mean_co_membership"] > 0.90
    assert planted_summary["reportable"]


def test_independent_noise_does_not_produce_a_reportable_module():
    values = _values(np.random.default_rng(13).normal(size=(30, 18)))
    correlation = spearman_correlation(values)
    modules = assign_hierarchical_modules(correlation, distance_cut=0.30, signed=True)
    comembership = bootstrap_module_comembership(
        values,
        n_bootstraps=40,
        random_state=14,
        distance_cut=0.30,
        signed=True,
    )
    summary = module_stability_summary(
        modules,
        comembership,
        stability_floor=0.80,
        minimum_size=3,
    )
    assert not summary["reportable"].any()


def test_bootstrap_loadings_are_sign_aligned_before_summary():
    generator = np.random.default_rng(15)
    latent = generator.normal(size=16)
    array = generator.normal(size=(10, 16))
    array[:3] = latent + generator.normal(scale=0.1, size=(3, 16))
    values = _values(array)
    bootstraps = bootstrap_feature_loadings(
        values,
        n_bootstraps=30,
        random_state=16,
        n_components=2,
    )
    for resample in bootstraps:
        largest = np.argmax(np.abs(resample), axis=0)
        assert np.all(resample[largest, np.arange(2)] > 0)
    summary = loading_stability_summary(bootstraps, values.index)
    assert summary["sign_alignment"].eq("largest_absolute_loading_positive").all()
    assert summary["n_bootstraps"].eq(30).all()


def test_metabolite_annotation_composition_uses_the_supplied_annotation_name():
    values = _values(np.arange(24, dtype=float).reshape(4, 6) + 1)
    loadings, _, _ = fit_feature_margin_svd(values, n_components=2)
    annotations = pd.Series(["Amino acids", "Amino acids", "Lipids", "Lipids"], index=values.index)
    loading_summary = loading_annotation_composition(loadings, annotations, annotation_name="Pathway")
    modules = pd.Series(["module_1", "module_1", "module_2", "module_2"], index=values.index, name="module")
    module_summary = module_annotation_composition(modules, annotations, annotation_name="Pathway")

    assert "Pathway" in loading_summary.columns
    assert "Lipid Class" not in loading_summary.columns
    assert "Pathway" in module_summary.columns
    assert "Lipid Class" not in module_summary.columns
