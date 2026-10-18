"""Optional: map lipid shorthand names to LIPID MAPS identifiers and KEGG pathways (needs internet).

This script deliberately uses exact shorthand matching only. A response with multiple
structures is retained in the raw cache but reported as ambiguous rather than choosing
an isomer that the source workbook does not identify.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import date
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd

LIPID_MAPS_URL = "https://www.lipidmaps.org/rest/compound"
KEGG_URL = "https://rest.kegg.jp"
REQUEST_DELAY_SECONDS = 0.25
MAX_ATTEMPTS = 4

OUTPUT_COLUMNS = [
    "Feature",
    "Lipid Name",
    "lm_id",
    "hmdb_id",
    "kegg_id",
    "chebi_id",
    "pubchem_cid",
    "kegg_pathway_id",
    "kegg_pathway_name",
    "lipidmaps_functional_category",
    "mapping_status",
    "database_query_date",
]


def _cache_path(cache_dir: Path, source: str, key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return cache_dir / source / f"{digest}.json"


def _request(url: str) -> Any:
    """Request a JSON or text response without treating transient failures as no match."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            with urlopen(url, timeout=30) as response:
                body = response.read().decode("utf-8")
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return body
        except HTTPError as error:
            if error.code == 404:
                return {}
            if error.code < 500 or attempt == MAX_ATTEMPTS - 1:
                raise
        except URLError:
            if attempt == MAX_ATTEMPTS - 1:
                raise
        time.sleep(REQUEST_DELAY_SECONDS * (2**attempt))
    raise RuntimeError("Unreachable retry state.")


def _cached_request(cache_dir: Path, source: str, key: str, url: str) -> Any:
    path = _cache_path(cache_dir, source, key)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["response"]
    response = _request(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"query_key": key, "url": url, "queried_at": date.today().isoformat(), "response": response}, indent=2),
        encoding="utf-8",
    )
    time.sleep(REQUEST_DELAY_SECONDS)
    return response


def _cached_query_date(cache_dir: Path, source: str, key: str) -> str:
    """Return the original retrieval date rather than relabeling cached data as new."""
    payload = json.loads(_cache_path(cache_dir, source, key).read_text(encoding="utf-8"))
    return str(payload["queried_at"])


def query_lipid_maps(name: str, cache_dir: Path) -> dict[str, dict[str, Any]]:
    """Return exact LIPID MAPS abbreviation matches, cached by supplied shorthand."""
    input_item = "abbrev_chains" if "_" in name else "abbrev"
    url = f"{LIPID_MAPS_URL}/{input_item}/{quote(name, safe='')}/all"
    response = _cached_request(cache_dir, "lipidmaps", name, url)
    if not isinstance(response, dict):
        raise ValueError(f"Unexpected LIPID MAPS response for {name!r}.")
    return response


def _records(response: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in response.values() if isinstance(record, dict) and record.get("lm_id")]


def _functional_category(record: dict[str, Any]) -> str:
    return "; ".join(str(record[field]) for field in ("core", "main_class", "sub_class") if record.get(field))


def map_lipid_ids(annotations: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    """Map each feature using exact shorthand and preserve unresolved uncertainty."""
    required = {"Feature", "Lipid Name"}
    missing = required.difference(annotations.columns)
    if missing:
        raise KeyError(f"Annotations are missing columns: {sorted(missing)}")
    if annotations["Feature"].duplicated().any():
        raise ValueError("Feature must uniquely identify annotation rows.")

    rows: list[dict[str, object]] = []
    for annotation in annotations.loc[:, [column for column in annotations.columns if column in {"Feature", "Lipid Name", "Lipid Class"}]].to_dict("records"):
        name = annotation["Lipid Name"]
        query_date = ""
        if not isinstance(name, str) or not name.strip():
            records: list[dict[str, Any]] = []
        else:
            query_name = name.strip()
            records = _records(query_lipid_maps(query_name, cache_dir))
            query_date = _cached_query_date(cache_dir, "lipidmaps", query_name)
        row: dict[str, object] = {column: "" for column in OUTPUT_COLUMNS}
        row["Feature"] = annotation["Feature"]
        row["Lipid Name"] = name
        row["database_query_date"] = query_date
        if len(records) == 1:
            record = records[0]
            for field in ("lm_id", "hmdb_id", "kegg_id", "chebi_id", "pubchem_cid"):
                row[field] = record.get(field, "")
            row["lipidmaps_functional_category"] = _functional_category(record)
            row["mapping_status"] = "matched"
        elif records:
            row["mapping_status"] = "ambiguous"
        else:
            row["mapping_status"] = "unmapped"
        rows.append(row)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def _kegg_pathway_names(cache_dir: Path) -> dict[str, str]:
    response = _cached_request(cache_dir, "kegg", "pathway_list", f"{KEGG_URL}/list/pathway")
    if not isinstance(response, str):
        raise ValueError("Unexpected KEGG pathway-list response.")
    return {
        parts[0].replace("path:", ""): parts[1]
        for line in response.splitlines()
        if len(parts := line.split("\t", maxsplit=1)) == 2
    }


def _kegg_pathway_ids(kegg_id: str, cache_dir: Path) -> list[str]:
    response = _cached_request(cache_dir, "kegg", f"pathways_{kegg_id}", f"{KEGG_URL}/link/pathway/{quote(kegg_id, safe=':')}")
    if not isinstance(response, str):
        raise ValueError(f"Unexpected KEGG pathway response for {kegg_id!r}.")
    return [line.split("\t")[1].replace("path:", "") for line in response.splitlines() if "\t" in line]


def attach_kegg_pathways(mapped: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    """Add semicolon-joined KEGG pathway identifiers and names for resolved IDs."""
    if "kegg_id" not in mapped:
        raise KeyError("Mapped annotations are missing 'kegg_id'.")
    result = mapped.copy()
    pathway_names = _kegg_pathway_names(cache_dir)
    for index, kegg_id in result["kegg_id"].items():
        if not kegg_id:
            continue
        pathway_ids = _kegg_pathway_ids(str(kegg_id), cache_dir)
        result.loc[index, "kegg_pathway_id"] = "; ".join(pathway_ids)
        result.loc[index, "kegg_pathway_name"] = "; ".join(pathway_names.get(pathway_id, "") for pathway_id in pathway_ids)
    return result


def run(settings) -> pd.DataFrame:
    """Annotate every lipid of the report (needs internet; responses are cached in results/)."""
    from .stages.common import load
    out = settings.out("lipidomics", "08_database_annotation")
    features = load(settings, "lipidomics").features
    annotations = features[["Lipid Name", "Lipid Class"]].rename_axis("Feature").reset_index()
    mapped = attach_kegg_pathways(map_lipid_ids(annotations, out / "cache"), out / "cache")
    mapped.to_csv(out / "lipid_database_annotations.csv", index=False)
    return mapped
