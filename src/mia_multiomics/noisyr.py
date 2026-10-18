"""Small adapter for the external noisyR cross-check script."""

from __future__ import annotations

import shutil
import subprocess
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

NOISYR_WINDOW_MIN_FEATURES = 25   # smallest window, so tiny panels do not give unstable windows
NOISYR_STEP_FRACTION = 0.20       # windows overlap: step = 20 % of the window


def panel_scaled_window_parameters(
    n_features: int,
    fraction: float,
    *,
    minimum_features: int = NOISYR_WINDOW_MIN_FEATURES,
    step_fraction: float = NOISYR_STEP_FRACTION,
) -> tuple[int, int]:
    """Return a panel-scaled window and its overlapping step size."""
    if not isinstance(n_features, int) or n_features <= 0:
        raise ValueError("n_features must be a positive integer.")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")
    if not isinstance(minimum_features, int) or minimum_features <= 0:
        raise ValueError("minimum_features must be a positive integer.")
    if not 0 < step_fraction <= 1:
        raise ValueError("step_fraction must be in (0, 1].")

    window = max(ceil(n_features * fraction), minimum_features)
    if window > n_features:
        raise ValueError("The scaled window cannot exceed the feature panel.")
    step = max(1, round(window * step_fraction))
    return window, step


def threshold_median(thresholds: pd.DataFrame | None, status: str) -> float:
    """Return a noisyR threshold only for a completed, finite run."""
    if status != "completed" or thresholds is None or "threshold_log2_abundance" not in thresholds:
        return float("nan")
    values = pd.to_numeric(thresholds["threshold_log2_abundance"], errors="coerce")
    finite = values[np.isfinite(values)]
    return float(finite.median()) if not finite.empty else float("nan")


def threshold_difference(
    noisy_threshold: float,
    manual_threshold: float,
    *,
    status: str,
) -> float:
    """Compare thresholds only when the noisyR run completed successfully."""
    if status != "completed" or not np.isfinite(noisy_threshold) or not np.isfinite(manual_threshold):
        return float("nan")
    return noisy_threshold - manual_threshold


def run_noisyr(
    values: pd.DataFrame,
    work_dir: Path,
    label: str,
    project_root: Path,
    *,
    window: int | None = None,
    step: int | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
    """Run noisyR and return its curve, thresholds, and a readable status."""
    executable = shutil.which("Rscript")
    if executable is None:
        raise RuntimeError("Rscript is required for the noisyR cross-check.")
    if (window is None) != (step is None):
        raise ValueError("window and step must be supplied together.")

    tag = label if window is None else f"{label}_window{window}_step{step}"
    input_path = work_dir / f"noisyr_input_{tag}.csv"
    curve_path = work_dir / f"noisyr_curve_{tag}.csv"
    threshold_path = work_dir / f"noisyr_thresholds_{tag}.csv"
    values.to_csv(input_path, index_label="Feature")
    command = [
        executable,
        str(project_root / "r" / "noisyr_crosscheck.R"),
        str(input_path),
        str(curve_path),
        str(threshold_path),
    ]
    if window is not None and step is not None:
        command.extend((str(window), str(step)))
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        lines = completed.stdout.strip().splitlines()
        return None, None, next(
            (
                line.strip()
                for line in lines
                if "too many NAs" in line or "must be a finite" in line
            ),
            lines[-1] if lines else "noisyr_failed_without_output",
        )

    curve = pd.read_csv(curve_path)
    thresholds = pd.read_csv(threshold_path)
    if not np.isfinite(curve["median_pearson"]).any():
        thresholds["threshold_log2_abundance"] = np.nan
        return curve, thresholds, "no_finite_noisyr_similarity"
    if not np.isfinite(pd.to_numeric(thresholds["threshold_log2_abundance"], errors="coerce")).any():
        thresholds["threshold_log2_abundance"] = np.nan
        return curve, thresholds, "no_finite_noisyr_threshold"
    return curve, thresholds, "completed"
