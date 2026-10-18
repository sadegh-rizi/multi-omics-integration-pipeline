"""Helpers shared by the pipeline stages (loading + preprocessing in one call, small writers)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .. import config, inputs
from ..config import Settings
from ..preprocessing import Preprocessed, preprocess


@lru_cache(maxsize=8)
def _load_cached(settings: Settings, omics: str) -> inputs.OmicsData:
    return inputs.load(settings, omics)


def load(settings: Settings, omics: str) -> inputs.OmicsData:
    """Raw data of one omics (cached: the workbooks are read only once per session)."""
    return _load_cached(settings, omics)


def prepare(settings: Settings, omics: str, population: str = config.PRIMARY_POPULATION,
            **options) -> Preprocessed:
    """Load one omics and preprocess it for one analysis population.

    ``options`` are passed to :func:`mia_multiomics.preprocessing.preprocess`
    (``max_missing_fraction``, ``normalization``).
    """
    data = load(settings, omics)
    samples = inputs.population_samples(settings, data.metadata, population, omics="lipidomics")
    return preprocess(data, samples, **options)


def feature_class(features: pd.DataFrame) -> pd.Series:
    """The grouping column of a feature table: lipid class, metabolite pathway or nothing."""
    for column in ("Lipid Class", "Pathway"):
        if column in features:
            return features[column].astype(str).rename("Class")
    return pd.Series("protein", index=features.index, name="Class")


def feature_label(features: pd.DataFrame) -> pd.Series:
    """Human-readable feature name: lipid name, metabolite name or gene symbol."""
    for column in ("Lipid Name", "Metabolite", "Gene"):
        if column in features:
            return features[column].astype(str)
    return features.index.to_series().astype(str)


def region_samples(metadata: pd.DataFrame, region: str) -> list[str]:
    return metadata.loc[metadata[config.REGION_COL].eq(region), "Sample"].tolist()


def write_csv(table: pd.DataFrame, path: Path, **kwargs) -> Path:
    """Write a table, creating the folder first."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=kwargs.pop("index", False), **kwargs)
    return path


def write_json(data: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    return path
