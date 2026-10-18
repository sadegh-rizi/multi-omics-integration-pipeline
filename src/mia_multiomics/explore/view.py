"""One analysis-ready view of one omics layer, built from *your* settings.

The pipeline fixes its choices (all 60 samples or a T2 population, a 20 %
prevalence filter, 3 x blank, median normalisation).  For exploration you may
want to change them, so :func:`build_view` takes every choice as an option:

    view = build_view(settings, Options(exclude=[10, 45], max_missing_fraction=0.3))
    view.values          # normalised log2 abundances, features x kept samples
    view.pca("cortex")   # PCA of one region with the chosen scaling (cached)
    view.differential()  # group x sex model per region (cached)

Every figure function in :mod:`mia_multiomics.explore` takes a view, so all
figures in a notebook run come from the same data and the same choices.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, fields, replace

import numpy as np
import pandas as pd

from .. import config
from ..config import Settings
from ..inputs import OmicsData
from ..multivariate.pca import align_loading_signs, fit_pca
from ..preprocessing import Preprocessed, complete_features, drop_zero_variance, preprocess, scale
from ..stages.common import feature_class, feature_label, load
from ..stats.differential import fit_regionwise_differential

SCOPES = ("pooled", *config.REGIONS)


@dataclass(frozen=True)
class Options:
    """Every choice that shapes the data behind the figures.

    The defaults reproduce the pipeline's primary analysis (all 60 samples,
    current filtering).  ``exclude`` takes sample numbers (``[10, 45]``) or
    sample IDs (``["TOY_001-10"]``).
    """

    omics: str = "lipidomics"
    exclude: tuple = ()
    max_missing_fraction: float | None = config.MAX_MISSING_FRACTION
    blank_multiplier: float = config.BLANK_MULTIPLIER
    detection_floor: float | None = None
    normalization: str = "median"
    scaling: str = "autoscaled"
    min_per_cell: int = config.MIN_PER_CELL
    fdr_alpha: float = config.FDR_ALPHA
    outlier_pcs: int = config.OUTLIER_SCREEN_PCS
    top_n: int = 20
    n_permutations: int = 199
    n_bootstraps: int = 100
    seed: int = 2026

    def __post_init__(self) -> None:
        """Store ``exclude`` as a tuple (hashable) and reject impossible settings early."""
        object.__setattr__(self, "exclude", tuple(self.exclude or ()))
        if self.omics not in config.VIEWS:
            raise ValueError(f"omics must be one of {config.VIEWS}.")
        if self.normalization not in {"median", "pqn"}:
            raise ValueError("normalization must be 'median' or 'pqn'.")
        if self.scaling not in {"autoscaled", "pareto", "unscaled"}:
            raise ValueError("scaling must be 'autoscaled', 'pareto' or 'unscaled'.")
        if self.max_missing_fraction is not None and not 0 <= self.max_missing_fraction <= 1:
            raise ValueError("max_missing_fraction must be between 0 and 1 (or None to switch the filter off).")
        if self.blank_multiplier <= 0:
            raise ValueError("blank_multiplier must be positive.")
        if not 0 < self.fdr_alpha < 1:
            raise ValueError("fdr_alpha must be between 0 and 1.")


@dataclass
class View:
    """Preprocessed data of one omics plus everything needed to plot it."""

    settings: Settings
    options: Options
    raw: OmicsData
    prepared: Preprocessed
    excluded: list[str]
    _cache: dict = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------- data
    @property
    def omics(self) -> str:
        return self.options.omics

    @property
    def values(self) -> pd.DataFrame:
        return self.prepared.values

    @property
    def metadata(self) -> pd.DataFrame:
        return self.prepared.metadata

    @property
    def features(self) -> pd.DataFrame:
        return self.prepared.features

    @property
    def classes(self) -> pd.Series:
        """Lipid class / metabolite pathway of every feature in the raw panel."""
        return feature_class(self.raw.features)

    @property
    def labels(self) -> pd.Series:
        """Readable name of every feature in the raw panel."""
        return feature_label(self.raw.features)

    @property
    def kept_samples(self) -> list[str]:
        return self.metadata["Sample"].tolist()

    def samples(self, scope: str) -> list[str]:
        """Kept samples of ``'pooled'`` (all regions) or one region (any capitalisation)."""
        scope = resolve_scope(scope)
        if scope == "pooled":
            return self.kept_samples
        return self.metadata.loc[self.metadata[config.REGION_COL].eq(scope), "Sample"].tolist()

    def matrix(self, scope: str = "pooled") -> pd.DataFrame:
        """Complete features of one scope, scaled as chosen (the input of PCA-type methods)."""
        values = complete_features(self.values.loc[:, self.samples(scope)])
        return scale(drop_zero_variance(values), self.options.scaling)

    def pca(self, scope: str = "pooled"):
        """(model, scores, sign-aligned loadings) of one scope; computed once per view."""
        key = ("pca", resolve_scope(scope))
        if key not in self._cache:
            model, scores, loadings = fit_pca(self.matrix(scope))
            self._cache[key] = (model, scores, align_loading_signs(loadings))
        return self._cache[key]

    def differential(self) -> pd.DataFrame:
        """Group x sex OLS per feature and region (FDR over the whole panel); computed once."""
        if "differential" not in self._cache:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                effects = fit_regionwise_differential(
                    self.values, self.metadata, annotations=None, min_per_cell=self.options.min_per_cell,
                    fdr_denominator="all_features").effects
            if len(effects):
                effects["Label"] = effects["Feature"].map(self.labels)
                effects["Class"] = effects["Feature"].map(self.classes)
            self._cache["differential"] = effects
        return self._cache["differential"]

    def sibling(self, omics: str) -> View:
        """The same settings (and exclusions) applied to another omics layer."""
        if omics == self.omics:
            return self
        key = ("sibling", omics)
        if key not in self._cache:
            self._cache[key] = build_view(self.settings, replace(self.options, omics=omics, detection_floor=None))
        return self._cache[key]

    # ---------------------------------------------------------------- description
    def tag(self) -> str:
        """One line describing the data behind a figure (printed under every figure)."""
        o = self.options
        parts = [o.omics, f"{len(self.kept_samples)} samples"]
        if self.excluded:
            numbers = self.raw.metadata.set_index("Sample").loc[self.excluded, "Sample number"]
            parts.append("excluded " + ", ".join(str(n) for n in sorted(numbers)))
        if self.raw.blanks is not None:
            prevalence = "off" if o.max_missing_fraction is None else f"<= {o.max_missing_fraction:.0%} missing"
            parts += [f"prevalence {prevalence}", f"blank x{o.blank_multiplier:g}", o.normalization]
        parts += [f"{len(self.values)} features", o.scaling]
        return " | ".join(parts)

    def summary(self) -> pd.DataFrame:
        """Feature and sample counts after each preprocessing step."""
        table = self.prepared.summary.T.rename(columns={0: "value"})
        table.loc["excluded_samples"] = ", ".join(map(str, self.excluded_numbers())) or "none"
        return table

    def excluded_numbers(self) -> list[int]:
        meta = self.raw.metadata.set_index("Sample")
        return sorted(int(n) for n in meta.loc[self.excluded, "Sample number"])

    def removed_whole_mice(self) -> list[str]:
        """Mice whose every regional sample was excluded."""
        meta = self.raw.metadata
        excluded = meta["Sample"].isin(self.excluded)
        per_mouse = excluded.groupby(meta[config.MOUSE_COL]).all()
        return per_mouse.index[per_mouse].tolist()

    def cell_counts(self) -> pd.DataFrame:
        """Samples per region x group x sex after exclusion (the model needs >= min_per_cell)."""
        return (self.metadata.groupby([config.REGION_COL, config.GROUP_COL, config.SEX_COL], observed=True)
                .size().unstack([config.GROUP_COL, config.SEX_COL]).fillna(0).astype(int))


def resolve_scope(scope: str) -> str:
    """'pooled', 'cortex', 'Cortex' ... -> canonical scope name."""
    lookup = {s.lower(): s for s in SCOPES}
    try:
        return lookup[str(scope).lower()]
    except KeyError:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}.") from None


def resolve_exclusions(metadata: pd.DataFrame, exclude) -> list[str]:
    """Turn sample numbers and/or sample IDs into sample IDs (unknown entries are an error)."""
    by_number = dict(zip(metadata["Sample number"].astype(int), metadata["Sample"]))
    ids = set(metadata["Sample"])
    resolved, unknown = [], []
    for entry in exclude:
        if isinstance(entry, (int, np.integer)) or (isinstance(entry, str) and entry.strip().isdigit()):
            number = int(entry)
            (resolved.append(by_number[number]) if number in by_number else unknown.append(entry))
        elif entry in ids:
            resolved.append(entry)
        else:
            unknown.append(entry)
    if unknown:
        raise ValueError(f"Unknown samples in exclude: {unknown}. Use sample numbers 1-{len(metadata)} or sample IDs.")
    return list(dict.fromkeys(resolved))


def build_view(settings: Settings, options: Options | None = None, **changes) -> View:
    """Load one omics and preprocess it with the given options.

    Keyword ``changes`` override single options:
    ``build_view(settings, omics="metabolomics", exclude=[10])``.
    """
    options = replace(options or Options(), **changes) if changes else (options or Options())
    raw = load(settings, options.omics)
    excluded = resolve_exclusions(raw.metadata, options.exclude)
    samples = [s for s in raw.metadata["Sample"] if s not in excluded]
    if len(samples) < 6:
        raise ValueError("Fewer than 6 samples would remain; exclude fewer samples.")
    prepared = preprocess(raw, samples, max_missing_fraction=options.max_missing_fraction,
                          normalization=options.normalization, blank_multiplier=options.blank_multiplier,
                          absolute_floor=options.detection_floor)
    return View(settings=settings, options=options, raw=raw, prepared=prepared, excluded=excluded)


def option_names() -> list[str]:
    return [f.name for f in fields(Options)]
