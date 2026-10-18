"""From raw peak areas to normalised log2 abundances.

Input convention everywhere: features are rows, samples are columns.

The chain applied by :func:`preprocess` (same for lipidomics and metabolomics):

1. **prevalence**  - drop features missing in > 20 % of all 60 raw samples
   (computed before any sample is removed, so every population sees the same panel);
2. **blank filter** - drop features whose median sample signal is < 3 x mean blank;
3. **detection limit** - set single values below max(3 x mean blank, floor) to missing;
4. **sample selection** - keep the samples of the requested population;
5. **log2** and **normalisation** - subtract each sample's median (or log-space PQN),
   per acquisition mode for metabolomics.

Nothing is imputed: missing values stay missing (see :mod:`.imputation` for
the optional sensitivity imputers).  Every removed feature is recorded with
its reason in ``Preprocessed.exclusions``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .inputs import OmicsData


# ---------------------------------------------------------------- value-level filters
def exclude_nonpositive(intensity: pd.DataFrame) -> pd.DataFrame:
    """Mask zero, negative and non-numeric intensities as missing (never as zero)."""
    numeric = intensity.apply(pd.to_numeric, errors="coerce")
    return numeric.where(numeric > 0)


def _check_blanks(intensity: pd.DataFrame, blanks: pd.DataFrame) -> pd.DataFrame:
    """Blanks must match the samples' features, exist, and be non-negative; missing blank values count as 0."""
    if not intensity.index.equals(blanks.index):
        raise ValueError("Sample and blank tables must have the same features in the same order.")
    if blanks.shape[1] == 0:
        raise ValueError("At least one blank injection is required.")
    numeric = blanks.apply(pd.to_numeric, errors="coerce")
    if numeric.lt(0).to_numpy().any():
        raise ValueError("Blank intensities must be non-negative.")
    return numeric.fillna(0)


def blank_dominated(intensity: pd.DataFrame, blanks: pd.DataFrame,
                    multiplier: float = config.BLANK_MULTIPLIER) -> pd.Index:
    """Features whose median positive sample signal is below ``multiplier`` x mean blank."""
    mean_blank = _check_blanks(intensity, blanks).mean(axis=1)
    median = exclude_nonpositive(intensity).median(axis=1)
    return intensity.index[mean_blank.gt(0) & median.lt(multiplier * mean_blank)]


def below_detection_reasons(intensity: pd.DataFrame, blanks: pd.DataFrame, *,
                            multiplier: float = config.BLANK_MULTIPLIER,
                            absolute_floor: float = config.LIPID_ABSOLUTE_FLOOR) -> pd.DataFrame:
    """Label positive values below the per-feature detection limit.

    The limit is ``max(multiplier x mean blank, absolute_floor)``; the label
    says which of the two set it ("below_blank_limit" / "below_absolute_floor").
    """
    if multiplier <= 0 or absolute_floor <= 0:
        raise ValueError("multiplier and absolute_floor must be positive.")
    blank_limit = multiplier * _check_blanks(intensity, blanks).mean(axis=1)
    blank_sets_limit = blank_limit.gt(absolute_floor).to_numpy()[:, None]
    numeric = intensity.apply(pd.to_numeric, errors="coerce")
    positive = numeric.gt(0)
    below_blank = positive & numeric.lt(blank_limit, axis="index") & blank_sets_limit
    below_floor = positive & numeric.lt(absolute_floor) & ~blank_sets_limit
    reasons = pd.DataFrame(pd.NA, index=intensity.index, columns=intensity.columns, dtype="string")
    return reasons.mask(below_blank, "below_blank_limit").mask(below_floor, "below_absolute_floor")


def missing_fraction(intensity: pd.DataFrame) -> pd.Series:
    """Share of samples in which each feature is missing or non-positive."""
    return exclude_nonpositive(intensity).isna().mean(axis=1)


# ---------------------------------------------------------------- transformation
def log2_transform(positive: pd.DataFrame) -> pd.DataFrame:
    """log2 of a table that contains only positive values or NaN."""
    if positive.le(0).to_numpy().any():
        raise ValueError("log2_transform received non-positive values; call exclude_nonpositive first.")
    return np.log2(positive)


def median_normalize(log_values: pd.DataFrame) -> pd.DataFrame:
    """Subtract each sample's median log2 intensity (corrects loading differences)."""
    return log_values.subtract(log_values.median(axis=0), axis=1)


def pqn_normalize(log_values: pd.DataFrame, regions: pd.Series) -> pd.DataFrame:
    """Probabilistic quotient normalisation in log space, one reference per region.

    The reference is the region's median profile (both treatment groups
    pooled), centred to median zero; each sample is shifted by its median
    log-ratio to that reference.  ``regions`` is indexed by sample.
    """
    regions = regions.reindex(log_values.columns)
    if regions.isna().any():
        raise ValueError("PQN needs a region label for every sample.")
    normalized = log_values.copy()
    for _, samples in regions.groupby(regions, sort=False).groups.items():
        current = log_values.loc[:, samples]
        reference = current.median(axis=1)
        reference -= reference.median()
        quotients = current.subtract(reference, axis=0)
        if quotients.notna().sum(axis=0).lt(len(log_values) / 2).any():
            raise ValueError("PQN needs at least half of the features observed in every sample.")
        normalized.loc[:, samples] = current.subtract(quotients.median(axis=0), axis=1)
    return normalized


def normalize(log_values: pd.DataFrame, method: str, *, modes: pd.Series | None = None,
              regions: pd.Series | None = None) -> pd.DataFrame:
    """Median or PQN normalisation, done separately per acquisition mode if ``modes`` is given."""
    if method not in {"median", "pqn"}:
        raise ValueError("method must be 'median' or 'pqn'.")
    groups = modes.reindex(log_values.index) if modes is not None else pd.Series("all", index=log_values.index)
    out = log_values.copy()
    for _, features in groups.groupby(groups, sort=True).groups.items():
        block = log_values.loc[features]
        out.loc[features] = median_normalize(block) if method == "median" else pqn_normalize(block, regions)
    return out


def drop_zero_variance(values: pd.DataFrame) -> pd.DataFrame:
    """Drop features with zero or undefined variance (required before scaling)."""
    sd = values.std(axis=1, ddof=1)
    return values.loc[sd.gt(0) & sd.notna()]


def autoscale(values: pd.DataFrame) -> pd.DataFrame:
    """Centre each feature and divide by its SD (unit variance; for PCA/PERMANOVA)."""
    sd = values.std(axis=1, ddof=1)
    if sd.le(0).any() or sd.isna().any():
        raise ValueError("autoscale received zero-variance features; call drop_zero_variance first.")
    return values.subtract(values.mean(axis=1), axis=0).divide(sd, axis=0)


def pareto_scale(values: pd.DataFrame) -> pd.DataFrame:
    """Centre each feature and divide by the square root of its SD."""
    sd = values.std(axis=1, ddof=1)
    if sd.le(0).any() or sd.isna().any():
        raise ValueError("pareto_scale received zero-variance features; call drop_zero_variance first.")
    return values.subtract(values.mean(axis=1), axis=0).divide(np.sqrt(sd), axis=0)


def scale(values: pd.DataFrame, method: str) -> pd.DataFrame:
    """'autoscaled', 'pareto' or 'unscaled' (centred only)."""
    if method == "autoscaled":
        return autoscale(values)
    if method == "pareto":
        return pareto_scale(values)
    if method == "unscaled":
        return values.subtract(values.mean(axis=1), axis=0)
    raise ValueError(f"Unknown scaling {method!r}.")


def filter_low_iqr(values: pd.DataFrame, remove_fraction: float = config.IQR_REMOVE_FRACTION
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove the lowest-IQR features (label-free variance filter; sensitivity analysis only)."""
    if not 0 <= remove_fraction < 1:
        raise ValueError("remove_fraction must be in [0, 1).")
    iqr = values.quantile(0.75, axis=1) - values.quantile(0.25, axis=1)
    ranking = pd.DataFrame({"Feature": values.index, "iqr": iqr.to_numpy()})
    ranking = ranking.assign(_key=ranking["Feature"].astype(str)).sort_values(
        ["iqr", "_key"], na_position="first", kind="mergesort").drop(columns="_key").reset_index(drop=True)
    ranking["iqr_rank"] = np.arange(1, len(ranking) + 1)
    ranking["removed"] = ranking["iqr_rank"].le(int(np.ceil(len(ranking) * remove_fraction)))
    return values.drop(index=ranking.loc[ranking["removed"], "Feature"]), ranking


# ---------------------------------------------------------------- the whole chain
@dataclass
class Preprocessed:
    """Output of :func:`preprocess`."""

    values: pd.DataFrame        # normalised log2, features x kept samples (NaN = missing)
    metadata: pd.DataFrame      # metadata of the kept samples
    features: pd.DataFrame      # annotation of the kept features
    exclusions: pd.DataFrame    # one row per removed feature with stage and reason
    detection: pd.DataFrame     # long table of values masked below the detection limit
    summary: pd.DataFrame       # one-row feature counts per stage


def preprocess(
    data: OmicsData,
    samples: list[str] | None = None,
    *,
    max_missing_fraction: float | None = config.MAX_MISSING_FRACTION,
    normalization: str = "median",
    blank_multiplier: float = config.BLANK_MULTIPLIER,
    absolute_floor: float | None = None,
) -> Preprocessed:
    """Apply the full preprocessing chain to one omics layer (see module docstring).

    ``samples`` selects the population (default: all 60).  ``blank_multiplier``
    is used by both the blank filter and the detection limit;
    ``absolute_floor`` overrides the detection floor (default: 1.0 for lipids,
    the smallest positive value for metabolomics).  Proteomics is delivered
    already normalised and imputed, so only sample selection is applied to it.
    """
    samples = list(data.metadata["Sample"]) if samples is None else list(samples)
    metadata = data.metadata.set_index("Sample").loc[samples].reset_index()
    if data.blanks is None:                       # proteomics: nothing to filter
        values = data.values.loc[:, samples]
        empty = pd.DataFrame(columns=["Feature", "stage", "reason"])
        summary = pd.DataFrame([{"omics": data.name, "n_features_input": len(values),
                                 "n_features_output": len(values), "n_samples": len(samples)}])
        return Preprocessed(values, metadata, data.features, empty, empty.copy(), summary)

    raw = data.values.apply(pd.to_numeric, errors="coerce")
    blanks = data.blanks
    exclusions: list[pd.DataFrame] = []

    def record(features: pd.Index, stage: str, reason: str) -> None:
        exclusions.append(pd.DataFrame({"Feature": features, "stage": stage, "reason": reason}))

    # 1. prevalence on all 60 raw samples
    if max_missing_fraction is not None:
        low = raw.index[missing_fraction(raw).gt(max_missing_fraction)]
        record(low, "prevalence", f"missing in > {max_missing_fraction:.0%} of all samples")
        raw, blanks = raw.drop(index=low), blanks.drop(index=low)
    n_after_prevalence = len(raw)

    # 2. blank filter
    dominated = blank_dominated(raw, blanks, multiplier=blank_multiplier)
    record(dominated, "blank_filter", f"median signal < {blank_multiplier:g} x mean blank")
    raw, blanks = raw.drop(index=dominated), blanks.drop(index=dominated)

    # 3. value-level detection limit (metabolomics floor = its smallest positive value)
    if absolute_floor is not None:
        floor = float(absolute_floor)
    else:
        floor = config.LIPID_ABSOLUTE_FLOOR if data.name == "lipidomics" else float(raw[raw > 0].min().min())
    reasons = below_detection_reasons(raw, blanks, multiplier=blank_multiplier, absolute_floor=floor)
    detectable = exclude_nonpositive(raw).mask(reasons.notna())
    empty = detectable.index[detectable.isna().all(axis=1)]
    record(empty, "detection_limit", "no value above the detection limit")
    detectable = detectable.drop(index=empty)

    # 4-5. population, log2, normalisation
    logged = log2_transform(detectable.loc[:, samples])
    modes = data.features["Method"] if "Method" in data.features else None
    regions = metadata.set_index("Sample")[config.REGION_COL]
    values = normalize(logged, normalization, modes=modes, regions=regions)
    variable = drop_zero_variance(values)
    record(values.index.difference(variable.index, sort=False), "zero_variance", "no variation across samples")

    detection = (reasons.loc[:, samples].rename_axis(index="Feature", columns="Sample").stack()
                 .dropna().rename("reason").reset_index())
    exclusion_table = pd.concat(exclusions, ignore_index=True)
    summary = pd.DataFrame([{
        "omics": data.name,
        "n_samples": len(samples),
        "n_features_input": len(data.values),
        "n_after_prevalence": n_after_prevalence,
        "n_blank_dominated": len(dominated),
        "n_no_detectable_value": len(empty),
        "n_features_output": len(variable),
        "n_values_below_detection": len(detection),
        "detection_floor": floor,
        "blank_multiplier": blank_multiplier,
        "normalization": normalization,
    }])
    return Preprocessed(variable, metadata, data.features.loc[variable.index], exclusion_table, detection, summary)


def complete_features(values: pd.DataFrame) -> pd.DataFrame:
    """Keep only features observed in every sample (needed by PCA, ASCA, MOFA inputs)."""
    return values.dropna(axis=0, how="any")


def for_multivariate(values: pd.DataFrame) -> pd.DataFrame:
    """Complete features, autoscaled - the standard input of PCA and the outlier screen."""
    return autoscale(drop_zero_variance(complete_features(values)))
