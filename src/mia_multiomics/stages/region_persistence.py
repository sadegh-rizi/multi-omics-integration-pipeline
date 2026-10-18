"""Stage 5 - do regional differences persist after MIA?

Treatment is weak and confounded with litter, but region is strong and every
mouse contributes all three regions, so each mouse is its own control.  For
each pair of regions (e.g. cortex - cerebellum) we compute the per-mouse
difference profile and ask whether the MIA profile has the same shape as the
control profile:

* **concordance** - correlation of the control and MIA mean difference profiles;
* **disattenuated concordance** - that correlation divided by the ceiling set
  by each group's split-half reliability (1 = profile fully preserved), with
  mouse-bootstrap intervals;
* **per-feature interaction** - Welch test of control vs MIA differences
  (secondary, descriptive), FDR over all features x contrasts;
* cage ICCs of the difference profiles.

Output: ``results/<omics>/05_region_persistence/<population>/``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config, plotting
from ..config import Settings
from ..stats.clustered import intraclass_correlation
from ..stats.region_contrast import (
    REGION_CONTRASTS,
    apply_interaction_fdr,
    build_region_contrasts,
    concordance_summary,
    disattenuated_concordance,
    estimate_split_half_reliability,
    interaction_tests,
    one_sample_region_tests,
)
from .common import prepare, write_csv, write_json


def run(settings: Settings, omics: str = "lipidomics", population: str = config.PRIMARY_POPULATION) -> pd.DataFrame:
    """Region-persistence analysis for one omics and population; returns the disattenuation table."""
    out = settings.out(omics, "05_region_persistence", population)
    prepared = prepare(settings, omics, population)
    values, meta = prepared.values, prepared.metadata
    contrasts = build_region_contrasts(values, meta)

    within, interactions, iccs, concordance, reliability, disattenuation = [], [], [], [], [], []
    for number, (name, _, _) in enumerate(REGION_CONTRASTS):
        contrast = contrasts[name]
        seed = settings.seed + number
        within.append(one_sample_region_tests(contrast))
        icc = intraclass_correlation(contrast.values, contrast.metadata[["Sample", config.CAGE_COL, config.GROUP_COL]],
                                     n_bootstraps=settings.n_bootstraps, random_state=seed)
        iccs.append(icc.assign(Contrast=name))
        interactions.append(interaction_tests(contrast, icc))
        concordance.append(concordance_summary(contrast))
        summary, draws = estimate_split_half_reliability(contrast, n_splits=settings.n_split_halves, random_state=seed)
        reliability.append(summary)
        disattenuation.append(disattenuated_concordance(contrast, summary, draws, n_bootstraps=settings.n_bootstraps,
                                                        random_state=seed))

    interactions = apply_interaction_fdr(pd.concat(interactions, ignore_index=True), n_features=len(values),
                                         n_contrasts=len(REGION_CONTRASTS))
    disattenuation = pd.concat(disattenuation, ignore_index=True).assign(population=population, omics=omics)
    write_csv(pd.concat(within, ignore_index=True), out / "within_group_region_tests.csv")
    write_csv(interactions, out / "region_by_group_interactions.csv")
    write_csv(pd.concat(iccs, ignore_index=True), out / "cage_icc.csv")
    write_csv(pd.concat(concordance, ignore_index=True), out / "concordance.csv")
    write_csv(pd.concat(reliability, ignore_index=True), out / "split_half_reliability.csv")
    write_csv(disattenuation, out / "disattenuated_concordance.csv")
    write_json({"omics": omics, "population": population, "n_samples": values.shape[1], "n_features": len(values),
                "contrasts": [c[0] for c in REGION_CONTRASTS], "n_split_halves": settings.n_split_halves,
                "n_bootstraps": settings.n_bootstraps,
                "paired_mice": {name: len(c.metadata) for name, c in contrasts.items()}}, out / "manifest.json")
    return disattenuation


def plot_summary(settings: Settings, omics: str = "lipidomics") -> None:
    """Observed vs noise-corrected concordance per contrast and population (1 = preserved)."""
    plotting.apply_plot_style()
    tables = [pd.read_csv(path) for path in sorted((settings.results_dir / omics / "05_region_persistence").glob("*/disattenuated_concordance.csv"))]
    if not tables:
        return
    table = pd.concat(tables, ignore_index=True)
    write_csv(table, settings.results_dir / omics / "05_region_persistence" / "disattenuated_concordance_all.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    populations = list(dict.fromkeys(table["population"]))
    contrasts = list(dict.fromkeys(table["Contrast"]))
    width = 0.8 / len(populations)
    for i, population in enumerate(populations):
        rows = table.loc[table["population"].eq(population)].set_index("Contrast").loc[contrasts]
        x = np.arange(len(contrasts)) + (i - (len(populations) - 1) / 2) * width
        ax.scatter(x, rows["r_observed"], marker="o", facecolor="white", edgecolor=plotting.COLORBLIND_COLORS[i], zorder=3)
        ax.errorbar(x, rows["disattenuated_concordance"],
                    yerr=[rows["disattenuated_concordance"] - rows["bootstrap_q025"], rows["bootstrap_q975"] - rows["disattenuated_concordance"]],
                    fmt="o", color=plotting.COLORBLIND_COLORS[i], label=population, capsize=3)
    ax.axhline(1, color="#444444", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(len(contrasts)), contrasts)
    ax.set(ylabel="control vs MIA profile correlation", title=f"{omics}: regional profiles persist (open = observed, filled = corrected)")
    ax.legend(title="population", fontsize=8)
    plotting.save_figure(fig, settings.results_dir / omics / "05_region_persistence" / "figures" / "disattenuated_concordance")
