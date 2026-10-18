"""Exchangeability-respecting permutation and bootstrap helpers for ASCA."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

from ..multivariate.asca import build_effect_coded_design
from .clustered import cage_group_assignments

GROUP_COLUMN = "Bio group"
CONTROL_GROUP = "Saline Control"
MIA_GROUP = "MIA"


def _check_random_count(n_random: int) -> None:
    if n_random < 0:
        raise ValueError("n_random must be non-negative.")


def region_label_permutations(
    metadata: pd.DataFrame,
    *,
    n_random: int,
    random_state: int,
) -> Iterator[tuple[int, pd.DataFrame]]:
    """Yield the observed Region labels and random within-mouse shuffles."""
    _check_random_count(n_random)
    if "Mouse number" not in metadata or "Sample type" not in metadata:
        raise KeyError("Region permutations require Mouse number and Sample type.")
    rng = np.random.default_rng(random_state)
    observed = metadata.reset_index(drop=True).copy()
    yield 0, observed
    for permutation_id in range(1, n_random + 1):
        current = observed.copy()
        for _, positions in current.groupby("Mouse number", sort=False, observed=True).groups.items():
            positions = np.asarray(list(positions), dtype=int)
            if len(positions) < 2:
                raise ValueError("Every mouse must have at least two regional observations.")
            current.loc[positions, "Sample type"] = rng.permutation(
                observed.loc[positions, "Sample type"].to_numpy()
            )
        yield permutation_id, current


def cage_group_permutations(metadata: pd.DataFrame) -> Iterator[tuple[int, pd.DataFrame]]:
    """Yield every whole-cage Group assignment, including the observed labels."""
    observed = metadata.reset_index(drop=True).copy()
    assignments = cage_group_assignments(observed)
    for permutation_id, labels in enumerate(assignments):
        current = observed.copy()
        current[GROUP_COLUMN] = labels.to_numpy()
        yield permutation_id, current


def mouse_group_permutations(
    metadata: pd.DataFrame,
    *,
    n_random: int,
    random_state: int,
) -> Iterator[tuple[int, pd.DataFrame]]:
    """Yield observed and Group shuffles across mice within Sex."""
    _check_random_count(n_random)
    required = {"Mouse number", "Sex", GROUP_COLUMN}
    missing = required.difference(metadata.columns)
    if missing:
        raise KeyError(f"Mouse Group permutations require columns: {sorted(missing)}")
    observed = metadata.reset_index(drop=True).copy()
    rng = np.random.default_rng(random_state)
    mice = observed[["Mouse number", "Sex", GROUP_COLUMN]].drop_duplicates("Mouse number")
    if mice["Mouse number"].duplicated().any():
        raise ValueError("Each mouse must have one Sex and Group label.")
    yield 0, observed
    for permutation_id in range(1, n_random + 1):
        labels: dict[object, str] = {}
        for sex, cell in mice.groupby("Sex", sort=False, observed=True):
            cell_mice = cell["Mouse number"].to_numpy()
            n_mia = int(cell[GROUP_COLUMN].eq(MIA_GROUP).sum())
            selected = set(rng.choice(cell_mice, size=n_mia, replace=False).tolist())
            labels.update({mouse: (MIA_GROUP if mouse in selected else CONTROL_GROUP) for mouse in cell_mice})
        current = observed.copy()
        current[GROUP_COLUMN] = current["Mouse number"].map(labels)
        yield permutation_id, current


def control_cage_sex_permutations(
    metadata: pd.DataFrame,
    *,
    n_random: int,
    random_state: int,
) -> Iterator[tuple[int, pd.DataFrame]]:
    """Shuffle Sex among mice within control cages and preserve MIA cages."""
    _check_random_count(n_random)
    required = {"Mouse number", "Sex", "Cage", GROUP_COLUMN}
    missing = required.difference(metadata.columns)
    if missing:
        raise KeyError(f"Sex permutations require columns: {sorted(missing)}")
    observed = metadata.reset_index(drop=True).copy()
    rng = np.random.default_rng(random_state)
    yield 0, observed
    controls = observed.loc[observed[GROUP_COLUMN].eq(CONTROL_GROUP)]
    cages = controls["Cage"].drop_duplicates().tolist()
    for permutation_id in range(1, n_random + 1):
        labels: dict[object, str] = {}
        for cage in cages:
            cage_rows = controls.loc[controls["Cage"].eq(cage)]
            mouse_sex = cage_rows[["Mouse number", "Sex"]].drop_duplicates("Mouse number")
            if mouse_sex["Mouse number"].duplicated().any():
                raise ValueError("Each mouse must have one Sex label within a control cage.")
            shuffled = rng.permutation(mouse_sex["Sex"].to_numpy())
            labels.update(dict(zip(mouse_sex["Mouse number"], shuffled)))
        current = observed.copy()
        current["Sex"] = current["Mouse number"].map(labels).fillna(current["Sex"])
        yield permutation_id, current


def stratified_mouse_bootstrap_indices(
    metadata: pd.DataFrame,
    *,
    random_state: int,
) -> np.ndarray:
    """Sample mice with replacement within Group x Sex, retaining all rows."""
    required = {"Mouse number", GROUP_COLUMN, "Sex"}
    missing = required.difference(metadata.columns)
    if missing:
        raise KeyError(f"Mouse bootstrap requires columns: {sorted(missing)}")
    frame = metadata.reset_index(drop=True)
    mouse_info = frame[["Mouse number", GROUP_COLUMN, "Sex"]].drop_duplicates("Mouse number")
    if mouse_info["Mouse number"].duplicated().any():
        raise ValueError("Each mouse must have one Group and Sex label.")
    rng = np.random.default_rng(random_state)
    sampled_mice: list[object] = []
    for _, cell in mouse_info.groupby([GROUP_COLUMN, "Sex"], sort=True, observed=True):
        current = cell["Mouse number"].tolist()
        sampled_mice.extend(rng.choice(current, size=len(current), replace=True).tolist())
    rows: list[int] = []
    for mouse in sampled_mice:
        rows.extend(frame.index[frame["Mouse number"].eq(mouse)].tolist())
    return np.asarray(rows, dtype=int)


def effect_ssq_by_scaling(
    prepared_by_scaling: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    effects: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    """Calculate term SSQs for several scaling matrices using one design fit."""
    # Some cage relabelings leave one Group x Sex cell with a single mouse.
    # The fixed nested term then aliases that cell effect; use the deterministic
    # Moore-Penrose fit for the permutation statistic rather than dropping the
    # valid assignment from the exact exchangeability set.
    design, term_columns = build_effect_coded_design(metadata, allow_rank_deficient=True)
    pseudo_inverse = np.linalg.pinv(design)
    result = {scaling: {} for scaling in prepared_by_scaling}
    for scaling, prepared in prepared_by_scaling.items():
        data = np.asarray(prepared, dtype=float)
        if data.shape[0] != design.shape[0] or not np.isfinite(data).all():
            raise ValueError("Prepared permutation data is not aligned with metadata.")
        coefficients = pseudo_inverse @ data
        for effect in effects:
            columns = term_columns[effect]
            contribution = design[:, columns] @ coefficients[list(columns)]
            result[scaling][effect] = float(np.square(contribution).sum())
    return result


def enumerate_cage_assignment_count(metadata: pd.DataFrame) -> int:
    """Return the number of exact whole-cage assignments."""
    return sum(1 for _ in cage_group_permutations(metadata))


def make_spike_profile(metadata: pd.DataFrame, *, region_a: str = "Cortex", region_b: str = "Cerebellum") -> np.ndarray:
    """Return a unit contrast profile with a region difference of one."""
    result = np.zeros(len(metadata), dtype=float)
    result[metadata["Sample type"].eq(region_a).to_numpy()] = 0.5
    result[metadata["Sample type"].eq(region_b).to_numpy()] = -0.5
    return result
