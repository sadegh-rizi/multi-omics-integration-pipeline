import numpy as np
import pandas as pd
import pytest

import mia_multiomics.stats.region_contrast as region_contrast
from mia_multiomics.stats.region_contrast import (
    REGION_CONTRASTS,
    build_region_contrasts,
    disattenuated_concordance,
    estimate_split_half_reliability,
    interaction_tests,
)


def _metadata(mice: list[str], missing: tuple[str, str] | None = None) -> pd.DataFrame:
    rows = []
    for mouse in mice:
        group = "MIA" if mouse.startswith("M") else "Saline Control"
        for region in ("Cerebellum", "Cortex", "Hippocampus"):
            if missing == (mouse, region):
                continue
            rows.append({
                "Sample": f"{mouse}_{region[:2]}",
                "Mouse number": mouse,
                "Sample type": region,
                "Bio group": group,
                "Cage": f"Cage_{mouse[0]}",
            })
    return pd.DataFrame(rows)


def test_missing_region_mouse_is_removed_only_from_affected_contrasts():
    metadata = _metadata(["C1", "C2", "M1"], missing=("M1", "Cortex"))
    values = pd.DataFrame(
        np.arange(8 * len(metadata), dtype=float).reshape(8, len(metadata)),
        index=[f"feature_{number}" for number in range(8)],
        columns=metadata["Sample"],
    )

    contrasts = build_region_contrasts(values, metadata)

    assert set(contrasts) == {name for name, _, _ in REGION_CONTRASTS}
    for name, contrast in contrasts.items():
        expected = ["C1", "C2", "M1"] if name == "Hippocampus - Cerebellum" else ["C1", "C2"]
        assert list(contrast.values.columns) == expected
        assert contrast.n_mice == len(expected)
        assert contrast.dropped_mice == (() if len(expected) == 3 else ("M1",))
    metadata.loc[metadata["Sample"].eq("M1_Hi"), "Cage"] = "inconsistent"
    with pytest.raises(ValueError, match="inconsistent Cage"):
        build_region_contrasts(values, metadata)


def test_interaction_is_a_two_group_difference_not_a_difference_of_significance_flags():
    metadata = _metadata(["C1", "C2", "C3", "M1", "M2", "M3"])
    values = pd.DataFrame(
        {
            sample: [
                10.0 if sample.endswith("Co") else 8.0 if sample.endswith("Ce") else 6.0,
                1.0,
            ]
            for sample in metadata["Sample"]
        },
        index=["feature_1", "feature_2"],
    )
    contrasts = build_region_contrasts(values, metadata)
    result = interaction_tests(contrasts["Cortex - Cerebellum"])

    assert result.loc[result["Feature"].eq("feature_1"), "interaction_definition"].item().startswith("MIA minus")
    assert result["Feature"].nunique() == 2
    assert "q_value_raw" in result


def test_disattenuation_recovers_shared_profile_with_different_noise():
    rng = np.random.default_rng(20260919)
    n_features = 160
    feature_names = [f"feature_{number}" for number in range(n_features)]
    mice = [f"C{number}" for number in range(12)] + [f"M{number}" for number in range(12)]
    metadata = _metadata(mice)
    samples_by_mouse = metadata.set_index("Sample")["Mouse number"].to_dict()
    baseline = rng.normal(size=n_features)
    regional_profile = rng.normal(size=n_features)
    values = pd.DataFrame(index=feature_names)
    for sample, mouse in samples_by_mouse.items():
        region_offset = {"Ce": 0.0, "Co": 1.0, "Hi": 0.25}[sample.rsplit("_", 1)[1]]
        group_noise = 0.10 if mouse.startswith("C") else 0.70
        noise = rng.normal(scale=group_noise, size=n_features)
        values[sample] = baseline + region_offset * regional_profile + noise

    contrast = build_region_contrasts(values, metadata)["Cortex - Cerebellum"]
    reliability, draws = estimate_split_half_reliability(
        contrast,
        n_splits=250,
        random_state=20260919,
    )
    result = disattenuated_concordance(
        contrast,
        reliability,
        draws,
        n_bootstraps=250,
        random_state=20260920,
    )

    assert result.loc[0, "r_observed"] < 0.99
    assert result.loc[0, "disattenuated_concordance"] > 0.90
    assert result.loc[0, "n_valid_bootstraps"] > 0
    assert result.loc[0, "point_inside_interval"]
    assert result.loc[0, "bootstrap_q025"] <= 1 <= result.loc[0, "bootstrap_q975"]


def test_duplicate_mouse_copies_never_straddle_split_halves(monkeypatch):
    ids = np.array([0, 0, 1, 2, 3, 3, 4, 5])
    values = np.random.default_rng(4).normal(size=(20, len(ids)))
    values[0] = ids
    halves = []
    original = region_contrast._array_profile_mean

    def record_half(current):
        halves.append(set(current[0]))
        return original(current)

    monkeypatch.setattr(region_contrast, "_array_profile_mean", record_half)
    region_contrast._split_half_reliability(values, ids, 40, np.random.default_rng(7))
    for left, right in zip(halves[::2], halves[1::2]):
        assert left.isdisjoint(right)
        assert left | right == set(ids)


def test_shared_profile_bootstrap_coverage_over_100_seeds():
    # A Monte Carlo unit check; full project regressions use 1,000 x 100.
    covered = 0
    for seed in range(100):
        rng = np.random.default_rng(seed)
        mice = [f"C{i}" for i in range(10)] + [f"M{i}" for i in range(10)]
        profile = rng.normal(size=(80, 1))
        values = pd.DataFrame(profile + rng.normal(scale=1.5, size=(80, 20)), columns=mice)
        metadata = pd.DataFrame({"Sample": mice, "Bio group": ["Saline Control"] * 10 + ["MIA"] * 10})
        contrast = region_contrast.RegionContrast("synthetic", "A", "B", values, metadata)
        reliability, draws = estimate_split_half_reliability(contrast, n_splits=100, random_state=seed)
        result = disattenuated_concordance(contrast, reliability, draws,
                                          n_bootstraps=200, n_inner_splits=40, random_state=seed + 1000).iloc[0]
        covered += result["bootstrap_q025"] <= 1 <= result["bootstrap_q975"]
    assert covered >= 90, f"Shared-profile coverage: {covered}/100"
