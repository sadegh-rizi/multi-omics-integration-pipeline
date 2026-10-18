from pathlib import Path
from urllib.error import HTTPError

import pandas as pd
import pytest

import mia_multiomics.lipid_databases as lipid_annotation


def test_map_lipid_ids_marks_exact_match(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lipid_annotation, "query_lipid_maps", lambda name, cache_dir: {"Row1": {"lm_id": "LMGP1", "hmdb_id": "HMDB1", "kegg_id": "cpd:C00157", "chebi_id": "123", "pubchem_cid": "456", "core": "Glycerophospholipids", "main_class": "PC", "sub_class": "Diacyl-PC"}})
    monkeypatch.setattr(lipid_annotation, "_cached_query_date", lambda cache_dir, source, key: "2026-07-30")
    mapped = lipid_annotation.map_lipid_ids(pd.DataFrame({"Feature": [1], "Lipid Name": ["PC(16:0_18:1)"]}), tmp_path)
    assert mapped.loc[0, "mapping_status"] == "matched"
    assert mapped.loc[0, "hmdb_id"] == "HMDB1"
    assert mapped.loc[0, "lipidmaps_functional_category"] == "Glycerophospholipids; PC; Diacyl-PC"


def test_map_lipid_ids_keeps_unmapped_and_ambiguous_blank(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    responses = {"missing": {}, "ambiguous": {"Row1": {"lm_id": "LM1"}, "Row2": {"lm_id": "LM2"}}}
    monkeypatch.setattr(lipid_annotation, "query_lipid_maps", lambda name, cache_dir: responses[name])
    monkeypatch.setattr(lipid_annotation, "_cached_query_date", lambda cache_dir, source, key: "2026-07-30")
    mapped = lipid_annotation.map_lipid_ids(pd.DataFrame({"Feature": [1, 2], "Lipid Name": ["missing", "ambiguous"]}), tmp_path)
    assert mapped["mapping_status"].tolist() == ["unmapped", "ambiguous"]
    assert mapped["lm_id"].tolist() == ["", ""]


def test_query_lipid_maps_uses_chain_endpoint_and_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def request(url: str) -> dict[str, dict[str, str]]:
        calls.append(url)
        return {"Row1": {"lm_id": "LM1"}}

    monkeypatch.setattr(lipid_annotation, "_request", request)
    monkeypatch.setattr(lipid_annotation.time, "sleep", lambda seconds: None)
    assert lipid_annotation.query_lipid_maps("PC(16:0_18:1)", tmp_path)["Row1"]["lm_id"] == "LM1"
    lipid_annotation.query_lipid_maps("PC(16:0_18:1)", tmp_path)
    assert len(calls) == 1
    assert "/abbrev_chains/" in calls[0]


def test_attach_kegg_pathways_parses_ids_and_names(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def cached_request(cache_dir: Path, source: str, key: str, url: str) -> str:
        if key == "pathway_list":
            return "path:map00564\tGlycerophospholipid metabolism\n"
        return "cpd:C00157\tpath:map00564\n"

    monkeypatch.setattr(lipid_annotation, "_cached_request", cached_request)
    mapped = lipid_annotation.attach_kegg_pathways(pd.DataFrame({"kegg_id": ["cpd:C00157", ""], "kegg_pathway_id": ["", ""], "kegg_pathway_name": ["", ""]}), tmp_path)
    assert mapped.loc[0, "kegg_pathway_id"] == "map00564"
    assert mapped.loc[0, "kegg_pathway_name"] == "Glycerophospholipid metabolism"


def test_request_returns_empty_response_for_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(url: str, timeout: int):
        raise HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr(lipid_annotation, "urlopen", missing)
    assert lipid_annotation._request("https://example.test/missing") == {}
