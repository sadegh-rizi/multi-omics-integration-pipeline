"""Small local-GMT and over-representation helpers for integration analyses."""

from __future__ import annotations

import gzip
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from scipy.stats import hypergeom


@dataclass(frozen=True)
class HypergeometricResult:
    """Counts and the one-sided over-representation p-value."""

    population_size: int
    selected_size: int
    gene_set_size: int
    overlap_size: int
    raw_p_value: float


def _open_gmt(path: Path):
    if path.name.casefold().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8-sig")
    return path.open("r", encoding="utf-8-sig")


def parse_gmt(path: Path) -> dict[str, frozenset[str]]:
    """Parse a tab-delimited GMT file into named, deduplicated gene sets."""
    if not path.is_file():
        raise FileNotFoundError(f"GMT file does not exist: {path}")

    gene_sets: dict[str, frozenset[str]] = {}
    with _open_gmt(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split("\t")
            if len(fields) < 3:
                raise ValueError(
                    f"Invalid GMT row in {path} at line {line_number}: "
                    "expected a name, description, and at least one member."
                )
            name = fields[0].strip()
            if not name:
                raise ValueError(f"GMT row has an empty gene-set name: {path}:{line_number}")
            if name in gene_sets:
                raise ValueError(f"Duplicate gene-set name {name!r} in {path}:{line_number}")
            members = frozenset(member.strip() for member in fields[2:] if member.strip())
            if not members:
                raise ValueError(f"GMT gene set {name!r} has no members: {path}:{line_number}")
            gene_sets[name] = members
    if not gene_sets:
        raise ValueError(f"GMT file contains no gene sets: {path}")
    return gene_sets


_VERSION_PATTERNS = (
    re.compile(r"(?:version|release)[\s_=-]*([0-9][A-Za-z0-9._-]*)", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])v([0-9][A-Za-z0-9._-]*)", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])(20[0-9]{2}(?:[._-][0-9]+)?)"),
)


def infer_gmt_version(path: Path) -> str:
    """Read a version token from GMT comments or the filename when available."""
    comment_lines: list[str] = []
    with _open_gmt(path) as handle:
        for line in handle:
            if line.lstrip().startswith("#"):
                comment_lines.append(line.strip())
            if len(comment_lines) >= 20:
                break

    filename = path.name
    if filename.casefold().endswith(".gz"):
        filename = filename[:-3]
    if filename.casefold().endswith(".gmt"):
        filename = filename[:-4]
    for source in [*comment_lines, filename]:
        for pattern in _VERSION_PATTERNS:
            match = pattern.search(source)
            if match:
                return match.group(1)
    return "unspecified"


def hypergeometric_enrichment(
    population: Iterable[str],
    selected: Iterable[str],
    gene_set: Iterable[str],
) -> HypergeometricResult:
    """Calculate one-sided over-representation for identifiers in one universe."""
    population_set = set(population)
    selected_set = set(selected)
    gene_set_values = set(gene_set)
    if not population_set:
        raise ValueError("The enrichment population cannot be empty.")
    if not selected_set.issubset(population_set):
        raise ValueError("Selected enrichment identifiers must be in the population.")

    population_size = len(population_set)
    selected_size = len(selected_set)
    gene_set_size = len(population_set & gene_set_values)
    overlap_size = len(selected_set & gene_set_values)
    raw_p_value = float(
        hypergeom.sf(overlap_size - 1, population_size, gene_set_size, selected_size)
    )
    return HypergeometricResult(
        population_size=population_size,
        selected_size=selected_size,
        gene_set_size=gene_set_size,
        overlap_size=overlap_size,
        raw_p_value=raw_p_value,
    )
