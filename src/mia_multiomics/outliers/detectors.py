"""Outlier scores for sample-by-variable matrices.

All scores increase with unusualness.  The proximity and partition detectors
are applied to PCA score matrices by the ensemble driver; Ledoit-Wolf retains
the complete feature space because shrinkage makes that covariance invertible.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import IsolationForest


def _numeric(values: pd.DataFrame) -> np.ndarray:
    numeric = values.apply(pd.to_numeric, errors="coerce")
    if numeric.empty or numeric.isna().to_numpy().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Outlier detectors require a non-empty, finite sample-by-variable matrix.")
    return numeric.to_numpy(dtype=float)


def distance_contrast(values: pd.DataFrame, *, k: int) -> float:
    """Return the relative spread of each sample's mean k-neighbour distance."""
    matrix = _numeric(values)
    distances = np.sqrt(((matrix[:, None, :] - matrix[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    nearest = np.partition(distances, k - 1, axis=1)[:, :k].mean(axis=1)
    return float((nearest.max() - nearest.min()) / nearest.min())


def knn_mean_distance(
    values: pd.DataFrame,
    *,
    parameter: int,
    weighting: Literal["isotropic", "soft"] = "isotropic",
    eigenvalues: Sequence[float] | None = None,
) -> pd.Series:
    """Mean Euclidean distance to the ``parameter`` nearest other samples."""
    matrix = _numeric(values)
    if not 1 <= parameter < len(matrix):
        raise ValueError("k must be between 1 and the number of samples minus one.")
    if weighting == "soft":
        if eigenvalues is None or len(eigenvalues) != matrix.shape[1]:
            raise ValueError("Soft k-NN requires one PCA eigenvalue per score column.")
        scales = np.asarray(eigenvalues, dtype=float)
        if np.any(~np.isfinite(scales)) or np.any(scales <= 0) or scales.min() < 0.01 * scales.max():
            raise ValueError("Soft k-NN requires retained eigenvalues at least 1% of the largest.")
        matrix = matrix / np.sqrt(scales)
    elif weighting != "isotropic":
        raise ValueError("weighting must be 'isotropic' or 'soft'.")
    distances = np.sqrt(((matrix[:, None, :] - matrix[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    return pd.Series(np.partition(distances, parameter - 1, axis=1)[:, :parameter].mean(axis=1), index=values.index)


def knn_mean_distance_sweep(values: pd.DataFrame, *, k_values: Sequence[int]) -> pd.DataFrame:
    """Calculate all requested k-NN mean distances from one distance sort."""
    matrix = _numeric(values)
    k_values = tuple(sorted(set(k_values)))
    if not k_values or min(k_values) < 1 or max(k_values) >= len(matrix):
        raise ValueError("k must be between 1 and the number of samples minus one.")
    distances = _pairwise_distances(matrix)
    ordered = np.sort(distances, axis=1)
    cumulative = np.cumsum(ordered, axis=1)
    return pd.DataFrame(
        {str(k): cumulative[:, k - 1] / k for k in k_values},
        index=values.index,
    )


def local_outlier_factor(values: pd.DataFrame, *, parameter: int) -> pd.Series:
    """LOF score, with larger values denoting more anomalous samples."""
    return local_outlier_factor_sweep(values, k_values=[parameter]).iloc[:, 0]


def local_outlier_factor_sweep(
    values: pd.DataFrame,
    *,
    k_values: Sequence[int],
) -> pd.DataFrame:
    """Calculate all requested in-sample LOF scores from one distance sort."""
    matrix = _numeric(values)
    k_values = tuple(sorted(set(k_values)))
    if not k_values or min(k_values) < 1 or max(k_values) >= len(matrix):
        raise ValueError("LOF neighbours must be between 1 and the number of samples minus one.")
    distances = _pairwise_distances(matrix)
    sorted_indices = np.argsort(distances, axis=1)
    sorted_distances = np.take_along_axis(distances, sorted_indices, axis=1)
    scores = {}
    for k in k_values:
        neighbors = sorted_indices[:, :k]
        neighbor_distances = sorted_distances[:, :k]
        kth_neighbor_distance = sorted_distances[:, k - 1]
        reachability = np.maximum(neighbor_distances, kth_neighbor_distance[neighbors])
        lrd = 1.0 / (reachability.mean(axis=1) + 1e-10)
        scores[k] = np.mean(lrd[neighbors] / lrd[:, np.newaxis], axis=1)
    return pd.DataFrame(scores, index=values.index).rename(columns=str)


def local_outlier_factor_sweep_array(
    simulations: np.ndarray,
    *,
    k_values: Sequence[int],
) -> np.ndarray:
    """Calculate all LOF k values for a batch of simulated matrices."""
    if simulations.ndim != 3 or not np.isfinite(simulations).all():
        raise ValueError("LOF simulations must be a finite 3-D array.")
    k_values = tuple(sorted(set(k_values)))
    n_samples = simulations.shape[1]
    if not k_values or min(k_values) < 1 or max(k_values) >= n_samples:
        raise ValueError("LOF neighbours must be between 1 and n_samples minus one.")
    distances = np.sqrt(((simulations[:, :, None, :] - simulations[:, None, :, :]) ** 2).sum(axis=3))
    diagonal = np.arange(n_samples)
    distances[:, diagonal, diagonal] = np.inf
    sorted_indices = np.argsort(distances, axis=2)
    sorted_distances = np.take_along_axis(distances, sorted_indices, axis=2)
    result = np.empty((len(simulations), n_samples, len(k_values)), dtype=float)
    for column, k in enumerate(k_values):
        neighbors = sorted_indices[:, :, :k]
        neighbor_distances = sorted_distances[:, :, :k]
        kth_neighbor_distance = sorted_distances[:, :, k - 1]
        neighbor_kth_distances = np.take_along_axis(
            kth_neighbor_distance[:, :, np.newaxis], neighbors, axis=1
        )
        reachability = np.maximum(neighbor_distances, neighbor_kth_distances)
        lrd = 1.0 / (reachability.mean(axis=2) + 1e-10)
        neighbor_lrd = np.take_along_axis(lrd[:, :, np.newaxis], neighbors, axis=1)
        result[:, :, column] = (neighbor_lrd / lrd[:, :, np.newaxis]).mean(axis=2)
    return result


def _pairwise_distances(matrix: np.ndarray) -> np.ndarray:
    """Return a finite Euclidean distance matrix with zero diagonal."""
    distances = np.sqrt(((matrix[:, None, :] - matrix[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    return distances


def isolation_forest_score(
    values: pd.DataFrame,
    *,
    parameter: Sequence[int],
    n_estimators: int = 500,
    n_jobs: int = -1,
) -> pd.Series:
    """Average Isolation Forest anomaly score across the supplied seed list."""
    _numeric(values)  # validates the input (numeric, complete)
    seeds = tuple(parameter)
    if not seeds:
        raise ValueError("Isolation Forest requires at least one random seed.")
    seed_scores = isolation_forest_seed_scores(
        values,
        seeds=seeds,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
    )
    return seed_scores.mean(axis=1).rename("isolation_forest")


def isolation_forest_seed_scores(
    values: pd.DataFrame,
    *,
    seeds: Sequence[int],
    n_estimators: int = 500,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Return one anomaly-score column per Isolation Forest seed."""
    matrix = _numeric(values)
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("Isolation Forest requires at least one random seed.")
    scores = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_isolation_forest_single_score)(matrix, seed, n_estimators)
        for seed in seeds
    )
    return pd.DataFrame(np.column_stack(scores), index=values.index, columns=[str(seed) for seed in seeds])


def _isolation_forest_single_score(matrix: np.ndarray, seed: int, n_estimators: int) -> np.ndarray:
    """Fit one serial forest; independent seeds are parallelized by the caller."""
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination="auto",
        random_state=seed,
        n_jobs=1,
    ).fit(matrix)
    return -model.score_samples(matrix)


def isolation_forest_null_scores(
    simulations: np.ndarray,
    *,
    seeds: Sequence[int],
    n_estimators: int = 500,
    n_jobs: int = -1,
) -> np.ndarray:
    """Score many simulated matrices while parallelizing across seed workers.

    ``simulations`` has shape ``(replicate, sample, variable)``.  Each worker
    keeps one seed's 500-tree forest configuration and loops over replicates,
    avoiding process creation and data serialization for every single matrix.
    """
    if simulations.ndim != 3 or not np.isfinite(simulations).all():
        raise ValueError("Isolation Forest simulations must be a finite 3-D array.")
    if not seeds:
        raise ValueError("Isolation Forest requires at least one random seed.")
    seed_scores = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        max_nbytes="1M",
        mmap_mode="r",
    )(
        delayed(_isolation_forest_scores_for_seed)(simulations, seed, n_estimators)
        for seed in seeds
    )
    return np.mean(np.stack(seed_scores, axis=0), axis=0)


def _isolation_forest_scores_for_seed(
    simulations: np.ndarray,
    seed: int,
    n_estimators: int,
) -> np.ndarray:
    """Return one seed's scores for every simulated matrix."""
    return np.stack([
        _isolation_forest_single_score(matrix, seed, n_estimators)
        for matrix in simulations
    ])




def ledoit_wolf_mahalanobis_sweep(
    values: pd.DataFrame,
    *,
    shrinkages: Sequence[float],
) -> pd.DataFrame:
    """Calculate shrinkage distances from one eigendecomposition of X'X."""
    matrix = _numeric(values)
    shrinkages = tuple(float(alpha) for alpha in shrinkages)
    if not shrinkages or min(shrinkages) < 0 or max(shrinkages) > 1:
        raise ValueError("Ledoit-Wolf shrinkage intensities must be in [0, 1].")
    return pd.DataFrame(
        _ledoit_wolf_scores_array(matrix[np.newaxis, :, :], shrinkages)[0],
        index=values.index,
        columns=[str(alpha) for alpha in shrinkages],
    )


def ledoit_wolf_mahalanobis_sweep_array(
    simulations: np.ndarray,
    *,
    shrinkages: Sequence[float],
) -> np.ndarray:
    """Calculate shrinkage distances for a batch without p-by-p inverses."""
    if simulations.ndim != 3 or not np.isfinite(simulations).all():
        raise ValueError("Ledoit-Wolf simulations must be a finite 3-D array.")
    shrinkages = tuple(float(alpha) for alpha in shrinkages)
    if not shrinkages or min(shrinkages) < 0 or max(shrinkages) > 1:
        raise ValueError("Ledoit-Wolf shrinkage intensities must be in [0, 1].")
    return _ledoit_wolf_scores_array(simulations, shrinkages)


def _ledoit_wolf_scores_array(simulations: np.ndarray, shrinkages: Sequence[float]) -> np.ndarray:
    """Score simulated observations from the sample-space covariance spectrum."""
    n_samples, n_features = simulations.shape[1:]
    centered = simulations - simulations.mean(axis=1, keepdims=True)
    gram = np.einsum("bnp,bmp->bnm", centered, centered) / (n_samples - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    positive = eigenvalues > (np.finfo(float).eps * np.maximum(eigenvalues[:, -1:], 1.0))
    clipped = np.maximum(eigenvalues, 0)
    projections = eigenvectors * np.sqrt((n_samples - 1) * clipped)[:, np.newaxis, :]
    projected_squares = projections**2
    residual = np.maximum(
        np.sum(centered**2, axis=2) - projected_squares.sum(axis=2),
        0,
    )
    trace_target = eigenvalues.clip(min=0).sum(axis=1) / n_features
    scores = np.empty((len(simulations), n_samples, len(shrinkages)), dtype=float)
    for column, alpha in enumerate(shrinkages):
        if alpha == 0:
            denominator = np.where(positive, eigenvalues, np.inf)
            scores[:, :, column] = np.sum(projected_squares / denominator[:, np.newaxis, :], axis=2)
        else:
            denominator = (1 - alpha) * eigenvalues + alpha * trace_target[:, np.newaxis]
            scores[:, :, column] = np.sum(
                projected_squares / denominator[:, np.newaxis, :], axis=2
            )
            if np.any(trace_target <= 0):
                raise ValueError("Ledoit-Wolf target variance must be positive.")
            scores[:, :, column] += residual / (alpha * trace_target[:, np.newaxis])
    return scores


def ledoit_wolf_auto_intensity(values: pd.DataFrame) -> float:
    """Return the data-selected Ledoit-Wolf shrinkage intensity for reporting."""
    return float(LedoitWolf().fit(_numeric(values)).shrinkage_)


def kernel_pca_reconstruction_distance(
    values: pd.DataFrame,
    *,
    parameter: float,
    n_components: int,
) -> pd.Series:
    """RBF kernel-PCA reconstruction RMSE per sample."""
    matrix = _numeric(values)
    if parameter <= 0:
        raise ValueError("Kernel-PCA gamma must be positive.")
    if not 1 <= n_components < len(matrix):
        raise ValueError("Kernel-PCA components must be between 1 and n_samples - 1.")
    model = KernelPCA(
        n_components=n_components,
        kernel="rbf",
        gamma=parameter,
        fit_inverse_transform=True,
        # The kernel matrix is at most 60 x 60; dense eigendecomposition is
        # stable here, whereas ARPACK can fail near the regional rank.
        eigen_solver="dense",
        random_state=20_260_826,
    )
    try:
        reconstructed = model.inverse_transform(model.fit_transform(matrix))
    except ValueError as error:
        if "zero-size array" not in str(error):
            raise
        # At very small gamma, floating-point centering can collapse the RBF
        # kernel to a constant. Its fitted manifold is then the input mean.
        reconstructed = np.broadcast_to(matrix.mean(axis=0), matrix.shape)
    return pd.Series(np.sqrt(np.mean((matrix - reconstructed) ** 2, axis=1)), index=values.index)
