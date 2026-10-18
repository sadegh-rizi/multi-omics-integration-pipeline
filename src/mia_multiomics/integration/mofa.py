"""MOFA+ helpers: fit a model, align factors across random seeds, summarise.

MOFA+ approximates every omics block (samples x features) by the same small
set of sample scores (factors, Z) times block-specific weights (W).  A factor
that explains variance in several blocks is a pattern the omics share.

MOFA+ solutions depend on the random seed, so a factor is only trusted if it
is recovered (matched by score or loading correlation) in most seeds.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .. import config


@dataclass
class MofaFit:
    factors: np.ndarray            # samples x factors (samples in `samples` order)
    weights: list[np.ndarray]      # one features x factors array per view
    variance: np.ndarray           # groups x views x factors (share of variance explained)
    samples: list[str]
    views: list[str]


def fit_mofa(blocks: dict[str, pd.DataFrame], metadata: pd.DataFrame, *, seed: int, n_factors: int,
             iterations: int, groups: tuple[str, ...] | None = None, ard_factors: bool = False) -> MofaFit:
    """Fit MOFA+ to ``blocks`` (view -> samples x features, same sample order as ``metadata``).

    ``groups`` splits the samples by region (multi-group MOFA+, region means
    removed by ``center_groups``); ``None`` fits all samples as one group.
    """
    from mofapy2.run.entry_point import entry_point

    views = list(blocks)
    if groups is None:
        group_index = [np.arange(len(metadata))]
        group_names = ["all"]
    else:
        group_index = [np.flatnonzero(metadata[config.REGION_COL].eq(g).to_numpy()) for g in groups]
        group_names = list(groups)
    sample_names = [[str(s) for s in metadata["Sample"].iloc[index]] for index in group_index]

    with contextlib.redirect_stdout(io.StringIO()):      # mofapy2 prints a lot even with quiet=True
        return _fit(entry_point, blocks, views, group_index, group_names, sample_names, seed, n_factors, iterations, ard_factors)


def _fit(entry_point, blocks, views, group_index, group_names, sample_names, seed, n_factors, iterations, ard_factors) -> MofaFit:
    model = entry_point()
    model.set_data_matrix(
        [[blocks[v].iloc[index].to_numpy(dtype=float) for index in group_index] for v in views],
        views_names=views, groups_names=group_names, samples_names=sample_names,
        features_names=[[f"{v}::{f}" for f in blocks[v].columns.astype(str)] for v in views])
    model.set_data_options(scale_views=True, center_groups=True)
    model.set_model_options(factors=n_factors, ard_factors=ard_factors)
    model.set_train_options(iter=iterations, startELBO=1, freqELBO=1, convergence_mode="fast", dropR2=0.02,
                            seed=seed, quiet=True, verbose=False)
    model.build()
    model.run()
    expectations = model.model.getExpectations()
    variance = np.asarray(model.model.calculate_variance_explained(), dtype=float)
    return MofaFit(factors=np.asarray(expectations["Z"]["E"], dtype=float),
                   weights=[np.asarray(w["E"], dtype=float) for w in expectations["W"]],
                   variance=variance, samples=[s for group in sample_names for s in group], views=views)


def match_factors(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pair candidate columns with reference columns by |correlation| (Hungarian matching).

    Returns, for every reference column, the matched candidate column (-1 if
    none), the sign, and the absolute correlation.
    """
    correlation = np.corrcoef(reference.T, candidate.T)[:reference.shape[1], reference.shape[1]:]
    correlation = np.nan_to_num(correlation)
    rows, columns = linear_sum_assignment(-np.abs(correlation))
    matched = np.full(reference.shape[1], -1)
    strength = np.full(reference.shape[1], np.nan)
    signs = np.ones(reference.shape[1])
    for r, c in zip(rows, columns):
        matched[r], strength[r] = c, abs(correlation[r, c])
        signs[r] = 1.0 if correlation[r, c] >= 0 else -1.0
    return np.column_stack([matched, signs]), strength


def stacked_loadings(weights: list[np.ndarray]) -> np.ndarray:
    """All views' weights stacked into one features x factors matrix."""
    return np.vstack([np.nan_to_num(w) for w in weights])


def seed_stability(fits: list[MofaFit], *, by: str = "scores", threshold: float = 0.0,
                   min_fraction: float = 0.8) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Match every fit's factors to the first fit; a factor is stable if found in >= 80 % of seeds.

    ``by='scores'`` matches sample scores (all-sample model), ``by='loadings'``
    matches feature weights (multi-group model, whose sample order may differ).
    Matches weaker than ``threshold`` do not count.  Also returns each fit's
    weights re-ordered and sign-flipped to the reference factors.
    """
    reference = fits[0]
    ref = reference.factors if by == "scores" else stacked_loadings(reference.weights)
    n = ref.shape[1]
    strengths, aligned_weights = [np.ones(n)], [reference.weights]
    for fit in fits[1:]:
        if by == "scores":
            order = [fit.samples.index(s) for s in reference.samples]
            candidate = fit.factors[order]
        else:
            candidate = stacked_loadings(fit.weights)
        pairing, strength = match_factors(ref, candidate)
        strength = np.where(strength >= threshold, strength, np.nan)
        strengths.append(strength)
        aligned = []
        for w in fit.weights:
            out = np.full((w.shape[0], n), np.nan)
            for r, (c, sign) in enumerate(pairing):
                if c >= 0 and c < w.shape[1]:
                    out[:, r] = sign * w[:, int(c)]
            aligned.append(out)
        aligned_weights.append(aligned)
    matrix = np.vstack(strengths)
    recovered = np.isfinite(matrix).sum(axis=0)
    table = pd.DataFrame({
        "Factor": [f"Factor{k + 1}" for k in range(n)],
        "n_recovered": recovered,
        "n_seeds": len(fits),
        "median_abs_correlation": np.nanmedian(matrix, axis=0),
        "stable": recovered >= np.ceil(min_fraction * len(fits)),
    })
    return table, aligned_weights


def top_features(weights: list[np.ndarray], blocks: dict[str, pd.DataFrame], factors: list[int], *,
                 n: int | None = None, fraction: float | None = None) -> pd.DataFrame:
    """Largest |weight| features per view and factor (fixed number or top fraction)."""
    rows = []
    for view, weight in zip(blocks, weights):
        for k in factors:
            count = n if n is not None else max(1, int(np.ceil(weight.shape[0] * fraction)))
            column = weight[:, k]
            order = np.argsort(-np.abs(np.nan_to_num(column)))[:count]
            for rank, position in enumerate(order, start=1):
                rows.append({"View": view, "Factor": f"Factor{k + 1}", "Feature": blocks[view].columns[position],
                             "Weight": float(column[position]), "rank": rank})
    return pd.DataFrame(rows, columns=["View", "Factor", "Feature", "Weight", "rank"])
