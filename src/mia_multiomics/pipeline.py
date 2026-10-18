"""Run the whole analysis, start to finish, in the right order.

    python -m mia_multiomics.pipeline            # full-size run (slow)
    python -m mia_multiomics.pipeline --fast     # small resampling counts (minutes)
    python -m mia_multiomics.pipeline --fast --only qc outliers

Each step reads the raw files (cached) and writes its tables and figures
under ``results/``; the only hand-over between steps is the outlier-screen
table (it defines the t2_95 / t2_99 populations) and the integration outputs
that later integration steps summarise.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

from . import config
from .config import Settings
from .stages import (
    asca,
    differential,
    effect_sizes,
    feature_modules,
    integration,
    noise_floor,
    outliers,
    qc,
    region_persistence,
    sensitivity,
    vendor_figures,
)

OMICS_WITH_BLANKS = ("lipidomics", "metabolomics")


def _for_each(function: Callable, omics: tuple[str, ...] = OMICS_WITH_BLANKS, **kwargs) -> Callable[[Settings], None]:
    """Turn ``function(settings, omics)`` into a step that runs it for every omics in ``omics``."""
    def run(settings: Settings) -> None:
        for name in omics:
            function(settings, name, **kwargs)
    return run


def _differential_all(settings: Settings) -> None:
    """The differential model for every omics and population, then the population comparison."""
    for omics in OMICS_WITH_BLANKS:
        for population in config.POPULATIONS:
            differential.run(settings, omics, population)
        differential.compare_populations(settings, omics)


def _persistence_all(settings: Settings) -> None:
    """Region persistence for every omics and population, then the summary figure."""
    for omics in OMICS_WITH_BLANKS:
        for population in config.POPULATIONS:
            region_persistence.run(settings, omics, population)
        region_persistence.plot_summary(settings, omics)


def _asca_all(settings: Settings) -> None:
    """ASCA+ for every omics and population."""
    for omics in OMICS_WITH_BLANKS:
        for population in config.POPULATIONS:
            asca.run(settings, omics, population)


def _sensitivity_all(settings: Settings) -> None:
    """All sensitivity analyses; the slow pooled mixed model only in full-size runs."""
    sensitivity.specification_curve(settings, "lipidomics")
    sensitivity.rf_variants(settings, "lipidomics")
    sensitivity.normalization_comparison(settings, "lipidomics")
    sensitivity.normalization_comparison(settings, "metabolomics")
    sensitivity.cage_structure(settings, "lipidomics")
    sensitivity.cage_structure(settings, "metabolomics")
    if not settings.quick:
        sensitivity.pooled_mixed_model(settings, "lipidomics")


# (name, what it does, function) in run order
STEPS: list[tuple[str, str, Callable[[Settings], object]]] = [
    ("qc", "preprocessing log, pooled QC, missingness, descriptive PCA", _for_each(qc.run, config.VIEWS)),
    ("noise_floor", "abundance-similarity noise floor (lipidomics)", lambda s: noise_floor.run(s, "lipidomics")),
    ("feature_modules", "feature-space PCA and co-abundance modules", _for_each(feature_modules.run)),
    ("outliers", "regional PCA outlier screen (defines t2_95 / t2_99)", _for_each(outliers.run_screen)),
    ("outlier_ensemble", "calibrated multi-detector outlier ensemble", lambda s: outliers.run_ensemble(s, "lipidomics")),
    ("outlier_concordance", "do the two assays flag the same samples?", outliers.run_concordance),
    ("differential", "group x sex model per region, all three populations", _differential_all),
    ("sensitivity", "specification curve, RF/scaling grid, PQN, litter-aware models", _sensitivity_all),
    ("region_persistence", "do regional profiles persist after MIA?", _persistence_all),
    ("asca", "ASCA+ multivariate ANOVA with permutation tests", _asca_all),
    ("effect_sizes", "effect sizes, lipid composites, protein sets", effect_sizes.run),
    ("integration_inputs", "aligned blocks for the three omics", lambda s: integration.build_blocks(s)),
    ("mofa", "MOFA+ on all 60 samples", integration.run_mofa),
    ("mofa_within_region", "multi-group MOFA+ (region means removed)", integration.run_mofa_within_region),
    ("benchmark", "region recovery and cross-omics predictability", integration.run_benchmark),
    ("cross_omics_screen", "lipid summaries vs metabolites / proteins", integration.run_cross_omics_screen),
    ("enrichment", "GO / Reactome enrichment of factor and module proteins", integration.run_enrichment),
    ("covariate_map", "what each integration axis tracks", integration.run_covariate_map),
    ("inference_checks", "why tests must be done at the litter level", integration.run_inference_checks),
    ("vendor_figures", "facility-style heatmaps and PLS-DA, with permutation tests", vendor_figures.run),
]


def run_all(settings: Settings, only: list[str] | None = None, verbose: bool = True) -> dict[str, float]:
    """Run every step (or only the named ones) and return the run time of each, in seconds."""
    timings = {}
    for name, description, step in STEPS:
        if only and name not in only:
            continue
        start = time.time()
        if verbose:
            print(f"[{name}] {description} ...", flush=True)
        step(settings)
        timings[name] = round(time.time() - start, 1)
        if verbose:
            print(f"[{name}] done in {timings[name]} s", flush=True)
    return timings


def main() -> None:
    """Command-line entry point: ``python -m mia_multiomics.pipeline [--fast] [--with-r] [--only ...]``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--fast", action="store_true", help="small resampling counts (demonstration run)")
    parser.add_argument("--with-r", action="store_true", help="also run the optional R steps (needs R)")
    parser.add_argument("--only", nargs="*", help=f"run only these steps: {[s[0] for s in STEPS]}")
    args = parser.parse_args()
    settings = (Settings.fast if args.fast else Settings.full)(args.project_root).with_(use_r=args.with_r)
    run_all(settings, args.only)


if __name__ == "__main__":
    main()
