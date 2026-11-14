"""Legacy row-entropy functional and quantile-stratified sampling."""

from __future__ import annotations

import numpy as np
import pandas as pd


def entropy_per_row(M: np.ndarray) -> np.ndarray:
    """Source-exact positive-component entropy-like functional.

    Nonpositive components contribute zero. This is not necessarily Shannon
    entropy of a probability simplex because M may be signed.
    """
    values = np.asarray(M, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_values = np.where(values > 0, np.log(values), 0.0)
    return -np.sum(values * log_values, axis=1)


def stratified_entropy_positions(
    M: np.ndarray,
    N: int = 2000,
    n_bins: int = 100,
    random_state: int | None = 42,
) -> np.ndarray:
    """Return selected integer row positions using the source sampler logic."""
    values = np.asarray(M, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("M must be a two-dimensional matrix.")
    if N <= 0:
        return np.empty(0, dtype=int)
    if N >= len(values):
        return np.arange(len(values), dtype=int)
    if n_bins < 1:
        raise ValueError("n_bins must be positive.")

    H = entropy_per_row(values)
    q = min(n_bins, len(H))
    labels = pd.qcut(H, q=q, labels=False, duplicates="drop")
    labels = np.asarray(labels, dtype=float)

    rng = np.random.default_rng(random_state)
    per_bin = max(1, N // n_bins)
    picks: list[int] = []

    for b in np.unique(labels[np.isfinite(labels)]).astype(int):
        positions = np.flatnonzero(labels == b)
        n_take = min(per_bin, len(positions))
        if n_take:
            picks.extend(
                rng.choice(positions, size=n_take, replace=False).tolist()
            )

    selected = np.asarray(sorted(picks), dtype=int)
    return selected


def stratified_entropy_sample(
    M_df: pd.DataFrame,
    N: int = 2000,
    n_bins: int = 100,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Return entropy-stratified observations in original temporal order."""
    positions = stratified_entropy_positions(
        M_df.to_numpy(), N=N, n_bins=n_bins, random_state=random_state
    )
    return M_df.iloc[positions]
