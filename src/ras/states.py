"""State assignment, occupancy-weighted medoid, and state lookups."""

from __future__ import annotations

import numpy as np


def assign_states(
    M: np.ndarray,
    X_N: np.ndarray,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """Assign each unit spectral state to its nearest spherical state."""
    M = np.asarray(M)
    X = np.asarray(X_N, dtype=np.float32)
    if M.ndim != 2 or X.ndim != 2 or M.shape[1] != X.shape[1]:
        raise ValueError("M and X_N must be 2-D with matching feature dimensions.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    assignment = np.empty(len(M), dtype=np.int32)
    for start in range(0, len(M), chunk_size):
        stop = min(start + chunk_size, len(M))
        block = np.asarray(M[start:stop], dtype=np.float32)
        similarity = block @ X.T
        assignment[start:stop] = np.argmax(similarity, axis=1)
    return assignment


def closest_state(q: np.ndarray, X: np.ndarray) -> int:
    X = np.asarray(X, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    q = q / max(np.linalg.norm(q), 1e-12)
    return int(np.argmax(X @ q))


def furthest_state(
    q: np.ndarray,
    X: np.ndarray,
    mask: np.ndarray | None = None,
) -> int:
    X = np.asarray(X, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    q = q / max(np.linalg.norm(q), 1e-12)
    cosine = X @ q
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(X),) or not np.any(mask):
            raise ValueError("mask must select at least one state.")
        candidate_idx = np.flatnonzero(mask)
        return int(candidate_idx[np.argmin(cosine[candidate_idx])])
    return int(np.argmin(cosine))


def spherical_medoid(
    X: np.ndarray,
    w: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Source-exact occupancy-weighted spherical medoid over states."""
    X = np.asarray(X, dtype=np.float64)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    weights = np.asarray(w, dtype=np.float64).copy()
    if X.ndim != 2 or weights.shape != (len(X),):
        raise ValueError("w must have one value per state.")
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Medoid weights must be finite, nonnegative, and nonzero.")
    weights /= weights.sum()
    D = np.arccos(np.clip(X @ X.T, -1.0, 1.0))
    objective = D @ weights
    idx = int(np.argmin(objective))
    return X[idx].copy(), idx


def state_indices(assignment: np.ndarray, state: int) -> np.ndarray:
    return np.flatnonzero(np.asarray(assignment) == int(state))
