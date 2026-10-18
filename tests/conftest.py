"""Shared test fixtures.  The package is importable without installation."""

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mia_multiomics.config import Settings  # noqa: E402
from mia_multiomics.toy_data import write_toy_dataset  # noqa: E402


@pytest.fixture(scope="session")
def toy_settings(tmp_path_factory) -> Settings:
    """A fresh toy dataset in a temporary project folder, with fast run settings."""
    root = tmp_path_factory.mktemp("toy_project")
    write_toy_dataset(root / "data")
    return Settings.fast(root)
