"""The explore package: options are validated, exclusions and thresholds reach the data,
and every figure function draws on the toy data."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from mia_multiomics import config, explore  # noqa: E402
from mia_multiomics.preprocessing import preprocess  # noqa: E402
from mia_multiomics.stages.common import load  # noqa: E402


@pytest.fixture(scope="module")
def view(toy_settings):
    return explore.build_view(toy_settings, n_permutations=19, n_bootstraps=12)


def test_default_view_matches_the_pipeline_preprocessing(toy_settings, view):
    pipeline = preprocess(load(toy_settings, "lipidomics"))
    assert view.values.equals(pipeline.values)
    assert view.excluded == []


def test_exclusions_accept_numbers_and_ids_and_flag_whole_mice(toy_settings):
    meta = load(toy_settings, "lipidomics").metadata
    mouse = meta.loc[meta["Sample number"].eq(1), config.MOUSE_COL].iloc[0]
    numbers = meta.loc[meta[config.MOUSE_COL].eq(mouse), "Sample number"].tolist()
    ids = meta.loc[meta["Sample number"].isin(numbers), "Sample"].tolist()
    view = explore.build_view(toy_settings, exclude=[numbers[0], *ids[1:]])
    assert len(view.kept_samples) == config.N_SAMPLES - len(numbers)
    assert view.removed_whole_mice() == [mouse]
    assert "excluded" in view.tag()


def test_unknown_samples_and_bad_options_are_rejected(toy_settings):
    with pytest.raises(ValueError, match="Unknown samples"):
        explore.build_view(toy_settings, exclude=[999])
    with pytest.raises(ValueError):
        explore.Options(normalization="mean")
    with pytest.raises(ValueError):
        explore.Options(max_missing_fraction=1.5)


def test_thresholds_change_the_feature_panel(toy_settings):
    strict = explore.build_view(toy_settings, max_missing_fraction=0.0, blank_multiplier=10)
    loose = explore.build_view(toy_settings, max_missing_fraction=None, blank_multiplier=1)
    assert len(strict.values) < len(loose.values)
    assert strict.prepared.summary["blank_multiplier"].iloc[0] == 10


def test_outlier_suggestions_find_the_planted_outliers(toy_settings):
    flagged = set(explore.suggest_exclusions(toy_settings)["Sample number"])
    assert {10, 45} <= flagged


def test_sibling_views_share_the_exclusions(toy_settings):
    view = explore.build_view(toy_settings, exclude=[10])
    protein = view.sibling("proteomics")
    assert protein.excluded == view.excluded and protein.omics == "proteomics"


FIGURES = [
    ("feature_funnel", {}), ("prevalence_curve", {}), ("signal_to_blank", {}), ("techmix_cv", {}),
    ("missingness_per_feature", {}), ("missingness_heatmap", {"stage": "filtered"}), ("scree_grouped", {}),
    ("scree_attribution", {}), ("pca_scores", {"color_by": "Tissue mass (binned)"}), ("loadings_heatmap", {}),
    ("loadings_class_composition", {}), ("loadings_cumulative_share", {}), ("influence_plot", {}),
    ("outlier_concordance", {}), ("volcano", {"region": "cortex"}), ("p_value_histograms", {}), ("small_p_counts", {}),
    ("effect_forest", {}), ("class_effects", {}), ("feature_profile", {"feature": "Cer(d18:0_14:0)", "effect": "Female vs Male"}), ("feature_correlation_heatmap", {}), ("feature_loading_stability", {}),
    ("module_stability", {}), ("cage_panels", {}), ("within_cage_distance", {}), ("region_differences", {}),
    ("asca_overview", {}), ("vendor_heatmap", {}), ("plsda_scores", {}), ("myelin_lipid_vs_protein", {}),
]


@pytest.mark.parametrize("name,kwargs", FIGURES, ids=[name for name, _ in FIGURES])
def test_figure_functions_draw(view, name, kwargs):
    fig = getattr(explore, name)(view, **kwargs)
    assert isinstance(fig, plt.Figure) and fig.axes
    assert any(view.tag() in text.get_text() for text in fig.texts)
    plt.close(fig)


def test_blank_based_figures_explain_why_proteomics_is_not_possible(toy_settings):
    with pytest.raises(ValueError, match="blank"):
        explore.techmix_cv(explore.build_view(toy_settings, omics="proteomics"))
