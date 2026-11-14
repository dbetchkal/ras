"""State run statistics and hypersphere-aware temporal velocity."""

from __future__ import annotations

import numpy as np
import pandas as pd


def state_runs(q: np.ndarray):
    """Convert a non-empty state sequence into contiguous runs."""
    q = np.asarray(q)
    if q.ndim != 1 or len(q) == 0:
        raise ValueError("q must be a non-empty one-dimensional array.")
    change = np.r_[True, q[1:] != q[:-1]]
    starts = np.flatnonzero(change)
    states = q[starts]
    ends = np.r_[starts[1:], len(q)]
    durations = ends - starts
    return states, starts, durations


def state_temporal_dynamics(
    q: np.ndarray,
    n_states: int,
    *,
    time_step_seconds: float = 1.0,
) -> pd.DataFrame:
    """Persistence/return statistics; retain sample counts and add seconds."""
    q = np.asarray(q)
    states, starts, durations = state_runs(q)
    records = []
    for state in range(n_states):
        idx = np.flatnonzero(states == state)
        if len(idx) == 0:
            record = {
                "state": state,
                "occupancy": 0.0,
                "n_visits": 0,
                "mean_dwell": np.nan,
                "median_dwell": np.nan,
                "mean_return": np.nan,
                "median_return": np.nan,
                "cv_return": np.nan,
            }
        else:
            dwell = durations[idx]
            if len(idx) > 1:
                returns = np.diff(starts[idx])
                mean_return = returns.mean()
                median_return = np.median(returns)
                cv_return = (
                    returns.std(ddof=1) / mean_return
                    if len(returns) > 1 and mean_return > 0
                    else np.nan
                )
            else:
                mean_return = median_return = cv_return = np.nan
            record = {
                "state": state,
                "occupancy": dwell.sum() / len(q),
                "n_visits": len(idx),
                "mean_dwell": dwell.mean(),
                "median_dwell": np.median(dwell),
                "mean_return": mean_return,
                "median_return": median_return,
                "cv_return": cv_return,
            }
        record["mean_dwell_seconds"] = record["mean_dwell"] * time_step_seconds
        record["median_dwell_seconds"] = record["median_dwell"] * time_step_seconds
        record["mean_return_seconds"] = record["mean_return"] * time_step_seconds
        record["median_return_seconds"] = record["median_return"] * time_step_seconds
        records.append(record)
    return pd.DataFrame.from_records(records)


def hypersphere_velocity_weighted(M_df: pd.DataFrame):
    """Notebook v_H = -(d-2) log(sin(theta)) and adjacent angular separation."""
    M = M_df.to_numpy(dtype=float)
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    M = M / norms
    dots = np.sum(M[1:] * M[:-1], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    theta = np.arccos(dots)
    dimension = M.shape[1]
    log_p = (dimension - 2) * np.log(np.sin(theta) + 1e-12)
    velocity = np.insert(-log_p, 0, np.nan)
    theta = np.insert(theta, 0, np.nan)
    return (
        pd.Series(velocity, index=M_df.index, name="v_H"),
        pd.Series(theta, index=M_df.index, name="theta"),
    )
