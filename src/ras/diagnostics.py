"""Temporal/exposure occupancy and effective-number diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def effective_number(mu: np.ndarray) -> float:
    p = np.asarray(mu, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return float("nan")
    p = p / p.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def renyi_effective_number(p_i: np.ndarray, q: float) -> float:
    p = np.asarray(p_i, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if p.size == 0:
        return float("nan")
    p = p / p.sum()
    if q == 0:
        return float(len(p))
    if q == 1:
        return float(np.exp(-np.sum(p * np.log(p))))
    power_sum = np.sum(p ** q)
    return float(power_sum ** (1.0 / (1.0 - q)))


def occupancy_diagnostics(
    S_band: pd.DataFrame,
    assignment: np.ndarray,
    n_prototypes: int,
) -> dict:
    """Compute notebook mu_T and mu_E from frame counts and linear band energy."""
    assignment = np.asarray(assignment, dtype=int)
    if len(assignment) != len(S_band):
        raise ValueError("assignment must contain one prototype per S row.")
    counts = np.bincount(assignment, minlength=n_prototypes).astype(np.float64)
    mu_T = counts / counts.sum()

    levels = S_band.to_numpy(dtype=np.float64)
    row_energy = np.power(10.0, levels / 10.0).sum(axis=1)
    exposure = np.bincount(
        assignment, weights=row_energy, minlength=n_prototypes
    ).astype(np.float64)
    mu_E = exposure / exposure.sum()

    N_eff_T = effective_number(mu_T)
    N_eff_E = effective_number(mu_E)
    order = np.argsort(mu_E)[::-1]
    cumulative = np.cumsum(mu_E[order])
    k = max(1, int(np.floor(N_eff_E))) if np.isfinite(N_eff_E) else 0
    exposure_at_Neff = (
        float(cumulative[min(k, len(cumulative)) - 1])
        if len(cumulative) and k > 0 else np.nan
    )

    return {
        "mean_activity": counts,
        "mu_T": mu_T,
        "mu_E": mu_E,
        "N_eff_T": N_eff_T,
        "N_eff_E": N_eff_E,
        "N_eff_T_fraction": N_eff_T / n_prototypes,
        "N_eff_E_fraction": N_eff_E / n_prototypes,
        "exposure_at_Neff": exposure_at_Neff,
        "N_q_T": {q: renyi_effective_number(mu_T, q) for q in (0, 1, 2)},
        "N_q_E": {q: renyi_effective_number(mu_E, q) for q in (0, 1, 2)},
    }
