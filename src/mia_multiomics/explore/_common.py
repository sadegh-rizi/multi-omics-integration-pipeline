"""Small helpers shared by the explore figure modules."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .. import config, plotting
from .view import View

MIA_EFFECT = "MIA vs Saline Control"
EFFECTS = (MIA_EFFECT, "Female vs Male", "MIA x Female interaction")


def finish(fig: plt.Figure, view: View | None, note: str | None = None) -> plt.Figure:
    """Print the data description (and an optional note) under a figure, then return it."""
    lines = [view.tag()] if view is not None else []
    if note:
        lines.append(note)
    if lines:
        fig.text(0.01, -0.01, "\n".join(lines), fontsize=7, color="#555555", ha="left", va="top")
    return fig


def require_blanks(view: View, what: str) -> None:
    if view.raw.blanks is None:
        raise ValueError(f"{what} needs blank / Tech-Mix injections; {view.omics} has none (use lipidomics or metabolomics).")


def region_list(region: str | None) -> list[str]:
    return list(config.REGIONS) if region is None else [region]


def augmented_metadata(view: View) -> pd.DataFrame:
    """Metadata plus derived columns that are handy for colouring plots."""
    meta = view.metadata.copy()
    meta["Date collected"] = pd.to_datetime(meta["Date collected"]).dt.strftime("%Y-%m-%d")
    mass = pd.to_numeric(meta["Tissue mass (mg)"], errors="coerce")
    meta["Tissue mass (binned)"] = pd.qcut(mass, 3, labels=["low", "middle", "high"]).astype(str)
    meta["Group x sex"] = meta[config.GROUP_COL].astype(str) + " / " + meta[config.SEX_COL].astype(str)
    return meta


def class_colors(classes: pd.Series, n_largest: int = 7) -> tuple[pd.Series, dict]:
    """Keep the n largest classes, lump the rest into 'other', and give each a colour."""
    counts = classes.value_counts()
    keep = counts.index[:n_largest]
    lumped = classes.where(classes.isin(keep), "other")
    names = [*keep, "other"] if (~classes.isin(keep)).any() else list(keep)
    colors = {name: plotting.COLORBLIND_COLORS[i % len(plotting.COLORBLIND_COLORS)] for i, name in enumerate(names)}
    if "other" in colors:
        colors["other"] = "#BBBBBB"
    return lumped, colors


def save_figures(figures: dict[str, plt.Figure], folder: Path) -> list[Path]:
    """Write a dict of named figures as PNG files into ``folder``."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    return [plotting.save_figure(fig, folder / name) for name, fig in figures.items()]
