"""Sequential PERMANOVA per region: is there *any* multivariate group or sex difference?

PERMANOVA partitions the Euclidean distances between samples (here on
autoscaled features) like an ANOVA: how much of the total spread is explained
by group, then sex, then their interaction (sequential, "type I" sums of
squares).  P-values come from Freedman-Lane permutations, which permute the
residuals of the reduced model so that earlier terms are held fixed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy.spatial.distance import pdist, squareform

from .common import bh_adjust


def _projection(design: np.ndarray) -> np.ndarray:
    """Hat matrix H = X (X'X)^+ X' of a design matrix."""
    return design @ np.linalg.pinv(design.T @ design) @ design.T


def _gower_matrix(values: np.ndarray) -> np.ndarray:
    """Gower-centred matrix G = -1/2 C D^2 C (samples x samples); trace(G) = total sum of squares."""
    distances = squareform(pdist(values, metric="euclidean"))
    n_samples = len(values)
    centering = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
    return -0.5 * centering @ (distances ** 2) @ centering


def _sequential_statistics(gower: np.ndarray, designs: list[np.ndarray]) -> list[dict[str, float]]:
    """Sum of squares, R² and pseudo-F of each term added in turn (``designs`` are nested)."""
    total_ss = float(np.trace(gower))
    fitted_ss = 0.0
    rows: list[dict[str, float]] = []
    for design in designs:
        projection = _projection(design)
        current_ss = float(np.trace(projection @ gower))
        term_ss = current_ss - fitted_ss
        fitted_ss = current_ss
        rows.append({"sum_of_squares": max(term_ss, 0.0), "r_squared": max(term_ss, 0.0) / total_ss})
    residual_ss = max(total_ss - fitted_ss, 0.0)
    full_rank = np.linalg.matrix_rank(designs[-1])
    residual_df = gower.shape[0] - full_rank
    for index, row in enumerate(rows):
        previous_rank = 1 if index == 0 else np.linalg.matrix_rank(designs[index - 1])
        term_df = np.linalg.matrix_rank(designs[index]) - previous_rank
        row["df"] = term_df
        row["residual_df"] = residual_df
        row["pseudo_f"] = (row["sum_of_squares"] / term_df) / (residual_ss / residual_df)
    return rows


def permanova_by_region(
    values: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    sample_column: str = "Sample",
    region_column: str = "Sample type",
    group_column: str = "Bio group",
    sex_column: str = "Sex",
    n_permutations: int = 9_999,
    random_state: int = 20_260_721,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run sequential per-region PERMANOVA with Freedman-Lane permutations.

    ``values`` is an auto-scaled feature-by-sample matrix. Features with any
    missing value within a region are excluded from that region's distance
    matrix; the returned diagnostics record their count.
    """
    required = {sample_column, region_column, group_column, sex_column}
    if missing := required.difference(metadata):
        raise KeyError(f"Metadata is missing columns: {sorted(missing)}")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive.")
    metadata = metadata.set_index(sample_column, drop=False).loc[values.columns].copy().rename(
        columns={region_column: "Region", group_column: "BioGroup", sex_column: "Sex"}
    )
    rng = np.random.default_rng(random_state)
    result_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    terms = ("BioGroup", "Sex", "BioGroup:Sex")
    for region, region_metadata in metadata.groupby("Region", sort=False, observed=True):
        samples = region_metadata[sample_column].tolist()
        complete = values.loc[:, samples].dropna(axis=0, how="any")
        diagnostics.append({"Region": region, "n_samples": len(samples), "n_features_input": len(values), "n_features_used": len(complete), "n_features_excluded_missing": len(values) - len(complete)})
        if complete.shape[0] < 2:
            raise ValueError(f"{region}: fewer than two complete features remain for PERMANOVA.")
        # Three nested designs: group; group + sex; group + sex + interaction.
        design_frame = region_metadata.loc[:, ["BioGroup", "Sex"]].copy()
        group_formula = "1 + C(BioGroup, Treatment(reference='Saline Control'))"
        group_sex_formula = group_formula + " + C(Sex, Treatment(reference='Male'))"
        full_formula = group_sex_formula + " + C(BioGroup, Treatment(reference='Saline Control')):C(Sex, Treatment(reference='Male'))"
        designs = [
            dmatrix(formula, design_frame, return_type="dataframe").to_numpy()
            for formula in (group_formula, group_sex_formula, full_formula)
        ]
        gower = _gower_matrix(complete.T.to_numpy(dtype=float))
        observed = _sequential_statistics(gower, designs)
        for term_index, term in enumerate(terms):
            # Freedman-Lane: split G into the part fitted by the earlier terms and the
            # residual part, permute only the residual part, and recompute the term's F.
            reduced = np.ones((len(samples), 1)) if term_index == 0 else designs[term_index - 1]
            reduced_projection = _projection(reduced)
            residualizer = np.eye(len(samples)) - reduced_projection
            residual = residualizer @ gower @ residualizer
            fitted = gower - residual
            permuted_f: list[float] = []
            for _ in range(n_permutations):
                permutation = rng.permutation(len(samples))
                permuted = fitted + residual[np.ix_(permutation, permutation)]
                permuted_f.append(_sequential_statistics(permuted, designs)[term_index]["pseudo_f"])
            result_rows.append({
                "Region": region,
                "Term": term,
                **observed[term_index],
                "p_value": (1 + sum(value >= observed[term_index]["pseudo_f"] for value in permuted_f)) / (n_permutations + 1),
                "n_permutations": n_permutations,
                "permutation_method": "Freedman-Lane",
            })
    results = pd.DataFrame(result_rows)
    results["p_value_fdr"] = bh_adjust(results["p_value"])     # BH over all regions x terms
    return results, pd.DataFrame(diagnostics)
