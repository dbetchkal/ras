"""Classical percentile references and observed-spectrum representatives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geometry import normalize_rows
from .states import closest_state


def closest_row_euclidean(
    A: pd.DataFrame | np.ndarray,
    target: np.ndarray | pd.Series,
    chunk_size: int = 200_000,
) -> int:
    """Index of the row with minimum Euclidean distance, using chunked arrays."""
    target_array = np.asarray(target, dtype=np.float32)
    target_norm2 = float(np.dot(target_array, target_array))
    best_idx, best_d2 = -1, np.inf

    n = len(A)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        if isinstance(A, pd.DataFrame):
            block = A.iloc[start:stop].to_numpy(dtype=np.float32, copy=False)
        else:
            block = np.asarray(A[start:stop], dtype=np.float32)
        d2 = np.sum(block * block, axis=1) - 2.0 * (block @ target_array) + target_norm2
        local = int(np.argmin(d2))
        if d2[local] < best_d2:
            best_d2 = float(d2[local])
            best_idx = start + local
    if best_idx < 0:
        raise ValueError("Cannot search an empty spectrum matrix.")
    return best_idx


@dataclass(frozen=True, slots=True)
class ReferenceStates:
    """L90/L50/L10 reference spectra mapped through observed rows to states."""

    quantile_spectra: pd.DataFrame
    nearest_row_positions: dict[str, int]
    state_indices: dict[str, int]
    state_observation_positions: dict[str, np.ndarray]
    medoid_spectrum: pd.Series
    ecc_spectrum: pd.Series
    medoid_observation_positions: np.ndarray
    ecc_observation_positions: np.ndarray


def build_reference_states(
    S: pd.DataFrame,
    S_band: pd.DataFrame,
    M: np.ndarray,
    X_N: np.ndarray,
    assignment: np.ndarray,
    medoid_idx: int,
    ecc_idx: int,
) -> ReferenceStates:
    """Reproduce the notebook's percentile -> nearest frame -> state map."""
    quantiles = pd.DataFrame(
        {
            "L90": S_band.quantile(0.10, axis=0),
            "L50": S_band.quantile(0.50, axis=0),
            "L10": S_band.quantile(0.90, axis=0),
        }
    ).T

    nearest: dict[str, int] = {}
    state: dict[str, int] = {}
    state_positions: dict[str, np.ndarray] = {}

    for label in ("L90", "L50", "L10"):
        row_pos = closest_row_euclidean(S_band, quantiles.loc[label])
        nearest[label] = row_pos
        state[label] = closest_state(M[row_pos], X_N)
        state_positions[label] = np.flatnonzero(assignment == state[label])

    medoid_positions = np.flatnonzero(assignment == medoid_idx)
    far_positions = np.flatnonzero(assignment == ecc_idx)
    medoid_observations = S.iloc[medoid_positions]
    far_observations = S.iloc[far_positions]

    medoid_spectrum = (
        medoid_observations.median(axis=0)
        if len(medoid_observations)
        else pd.Series(np.nan, index=S.columns)
    )
    far_spectrum = (
        far_observations.median(axis=0)
        if len(far_observations)
        else pd.Series(np.nan, index=S.columns)
    )

    return ReferenceStates(
        quantile_spectra=quantiles,
        nearest_row_positions=nearest,
        state_indices=state,
        state_observation_positions=state_positions,
        medoid_spectrum=medoid_spectrum,
        ecc_spectrum=far_spectrum,
        medoid_observation_positions=medoid_positions,
        ecc_observation_positions=far_positions,
    )


def state_percentile_spectra(
    S_band: pd.DataFrame,
    assignment: np.ndarray,
    n_states: int,
    percentile: float = 50.0,
) -> np.ndarray:
    """Percentile original spectrum for each state, NaN for empty cells."""
    assignment = np.asarray(assignment)
    out = np.full((n_states, S_band.shape[1]), np.nan, dtype=np.float64)
    values = S_band.to_numpy(dtype=np.float64)
    for k in range(n_states):
        mask = assignment == k
        if mask.any():
            out[k] = np.nanpercentile(values[mask], percentile, axis=0)
    return out


def diam_family_percentile_spectra(
    S_band: pd.DataFrame,
    diam_family_by_observation: np.ndarray,
    n_families: int,
    percentile: float = 50.0,
) -> np.ndarray:
    return state_percentile_spectra(
        S_band, diam_family_by_observation, n_families, percentile
    )


def furthest_observed_spectrum(
    q: np.ndarray,
    M_df: pd.DataFrame,
    S: pd.DataFrame,
) -> pd.Series:
    """Return original S row whose M direction is furthest from q."""
    idx = int(np.argmin(M_df.to_numpy() @ np.asarray(q)))
    return S.iloc[idx]
