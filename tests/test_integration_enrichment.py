from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mia_multiomics.integration.enrichment import hypergeometric_enrichment, infer_gmt_version, parse_gmt
from mia_multiomics.stats.common import bh_adjust


def test_parse_gmt_skips_comments_and_deduplicates_members(tmp_path: Path):
    path = tmp_path / "go_mouse_v2026.1.gmt"
    path.write_text(
        "# version: 2026.1\nterm-a\tdescription\tGene1\tGene1\tGene2\n",
        encoding="utf-8",
    )

    assert parse_gmt(path) == {"term-a": frozenset({"Gene1", "Gene2"})}


def test_hypergeometric_enrichment_matches_hand_calculation():
    result = hypergeometric_enrichment(
        population=range(10),
        selected={0, 1, 5},
        gene_set={0, 1, 2, 3},
    )

    assert result.population_size == 10
    assert result.selected_size == 3
    assert result.gene_set_size == 4
    assert result.overlap_size == 2
    assert np.isclose(result.raw_p_value, (36 + 4) / 120)


def test_benjamini_hochberg_preserves_input_order():
    assert np.allclose(
        bh_adjust(pd.Series([0.01, 0.04, 0.03, 1.0])),
        [0.04, 0.05333333333333334, 0.05333333333333334, 1.0],
    )


def test_benjamini_hochberg_counts_untested_features_in_the_family():
    q = bh_adjust(pd.Series([0.01, np.nan]), denominator=10)
    assert np.isclose(q.iloc[0], 0.1)
    assert np.isnan(q.iloc[1])
    with pytest.raises(ValueError):
        bh_adjust(pd.Series([0.01, 0.02, 0.03]), denominator=2)


def test_gmt_version_is_read_from_comments_or_filename(tmp_path: Path):
    path = tmp_path / "reactome_v97.gmt"
    path.write_text("term\tdesc\tA\n", encoding="utf-8")
    assert infer_gmt_version(path) == "97"

