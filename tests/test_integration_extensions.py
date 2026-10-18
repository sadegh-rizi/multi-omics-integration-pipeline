"""Focused tests for the integration helpers."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from mia_multiomics.integration.common import spearman_matrix
from mia_multiomics.integration.mofa import MofaFit, match_factors, seed_stability, top_features
from mia_multiomics.lipids import build_lipid_summary_features
from mia_multiomics.stages.integration import _modules, _oof_r2


def test_lipid_summary_features_are_complete_and_fixed_size():
    classes = [f"Class{number}" for number in range(25)]
    feature_names = [f"{classes[index % len(classes)]}({16 + index % 8}:0_{18 + index % 6}:1)" for index in range(70)]
    values = pd.DataFrame(np.arange(60 * len(feature_names), dtype=float).reshape(60, len(feature_names)) / 100,
                          columns=[str(index) for index in range(len(feature_names))])
    annotations = pd.DataFrame({"Lipid Name": feature_names, "Lipid Class": classes * 2 + classes[:20]}, index=values.columns)
    summaries, definitions = build_lipid_summary_features(values, annotations)
    assert 40 <= summaries.shape[1] <= 100
    assert summaries.index.equals(values.index)
    assert summaries.notna().all().all()
    assert definitions.index.is_unique


def test_spearman_matrix_matches_perfect_rank_relationship():
    left = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
    right = pd.DataFrame({"x": [2, 4, 6, 8], "y": [8, 6, 4, 2]})
    np.testing.assert_allclose(spearman_matrix(left, right), [[1, -1], [-1, 1]])


def test_factor_matching_finds_permuted_and_sign_flipped_factors():
    rng = np.random.default_rng(1)
    reference = rng.normal(size=(30, 3))
    candidate = np.column_stack([-reference[:, 2], reference[:, 0], reference[:, 1]])
    pairing, strength = match_factors(reference, candidate)
    assert pairing[:, 0].tolist() == [1, 2, 0]
    assert pairing[0, 1] == 1 and pairing[2, 1] == -1
    np.testing.assert_allclose(strength, 1.0)


def test_seed_stability_requires_recovery_above_the_threshold():
    rng = np.random.default_rng(2)
    scores = rng.normal(size=(20, 2))
    weights = [rng.normal(size=(15, 2))]
    samples = [f"s{i}" for i in range(20)]
    fits = [MofaFit(scores, weights, np.zeros((1, 1, 2)), samples, ["v"]) for _ in range(5)]
    noisy = [w + rng.normal(scale=[0.01, 5.0], size=w.shape) for w in weights]   # factor 2 not recovered
    fits[1] = MofaFit(scores, noisy, np.zeros((1, 1, 2)), samples, ["v"])
    fits[2] = MofaFit(scores, [w + rng.normal(scale=[0.01, 5.0], size=w.shape) for w in weights], np.zeros((1, 1, 2)), samples, ["v"])
    table, _ = seed_stability(fits, by="loadings", threshold=0.8)
    assert table.loc[0, "stable"]
    assert not table.loc[1, "stable"]


def test_top_features_ranks_by_absolute_weight():
    blocks = {"v": pd.DataFrame(np.zeros((2, 4)), columns=list("abcd"))}
    table = top_features([np.array([[0.1], [-0.9], [0.5], [0.0]])], blocks, [0], n=2)
    assert table["Feature"].tolist() == ["b", "c"]


def test_modules_group_features_with_shared_association_profiles():
    correlation = np.array([[0.9, 0.7, 0.8, 0.0], [0.95, 0.75, 0.85, 0.0], [0.0, 0.1, 0.0, 0.0]])
    modules = _modules(correlation, ["l1", "l2", "l3"], ["t1", "t2", "t3", "t4"])
    assert len(modules) == 1
    assert modules[0]["lipid_features"] == "l1;l2"
    assert modules[0]["target_features"] == "t1;t2;t3"


def test_predictability_uses_pooled_out_of_fold_r2():
    source = [(np.asarray([[0.0], [1.0], [2.0]]), np.asarray([[3.0]])), (np.asarray([[0.0], [1.0], [2.0]]), np.asarray([[4.0]]))]
    target = [(np.asarray([[0.0], [1.0], [2.0]]), np.asarray([[3.0]])), (np.asarray([[0.0], [1.0], [2.0]]), np.asarray([[4.0]]))]
    predictions = np.vstack([Ridge(alpha=1.0).fit(a, ta).predict(b).reshape(len(b), -1) for (a, b), (ta, _) in zip(source, target)])
    actual = np.concatenate([b for _, b in target])
    expected = 1 - np.square(actual - predictions).sum() / np.square(actual - actual.mean(0)).sum()
    assert np.isclose(_oof_r2(source, target), expected)
