"""Results and labeled accessors for a completed RAS analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import RASConfig
from .data import Spectrogram
from .preprocessing import ExposureResult
from .references import ReferenceStates


@dataclass(frozen=True, slots=True)
class RASResult:
    """All canonical RAS intermediates and derived morphology."""

    S: Spectrogram
    exposure: ExposureResult
    X_N: np.ndarray
    loss_history: np.ndarray
    state_assignment: np.ndarray
    mean_activity: np.ndarray
    weights: np.ndarray
    x_star: np.ndarray
    medoid_idx: int
    ecc_idx: int
    references: ReferenceStates
    G_MST: Any
    D_geo: np.ndarray
    diam: Any
    diam_distance: dict
    diam_attachment: dict
    diam_family_by_state: dict
    families: dict
    diam_family_by_observation: np.ndarray
    state_dynamics: pd.DataFrame
    diam_nodes: np.ndarray
    diam_occupancy: np.ndarray
    diam_diameter_order: np.ndarray
    diam_family_percentile_spectra: np.ndarray
    state_median_spectra: np.ndarray
    state_first_percentile_spectra: np.ndarray
    v_H: pd.Series
    theta_temporal: pd.Series
    tangent: dict[str, np.ndarray]
    diagnostics: dict
    config: RASConfig
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def S_O(self) -> pd.DataFrame:
        return self.exposure.S_O.to_frame()

    @property
    def S_E(self) -> pd.DataFrame:
        return self.exposure.S_E.to_frame()

    @property
    def M(self) -> pd.DataFrame:
        return self.exposure.M.to_frame()

    @property
    def M_values(self) -> np.ndarray:
        return self.exposure.M.values

    @property
    def X_N_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.X_N,
            columns=self.exposure.M.frequency.labels,
            index=pd.Index(np.arange(len(self.X_N)), name="state"),
        )

    @property
    def S_Pi(self) -> pd.DataFrame:
        """Original measured spectra with state and diameter-family labels."""
        labels = pd.DataFrame(
            {
                "state": self.state_assignment,
                "diam_family": self.diam_family_by_observation,
            },
            index=self.S.time,
        )
        return self.S.to_frame().join(labels)

    @property
    def S_family(self) -> pd.DataFrame:
        """Notebook-compatible alias for S_Pi."""
        return self.S_Pi

    @property
    def G(self):
        return self.G_MST

    @property
    def dynamics(self) -> pd.DataFrame:
        return self.state_dynamics

    @property
    def mu_T(self) -> np.ndarray:
        return self.diagnostics["mu_T"]

    @property
    def mu_E(self) -> np.ndarray:
        return self.diagnostics["mu_E"]

    @property
    def N_eff_T(self) -> float:
        return self.diagnostics["N_eff_T"]

    @property
    def N_eff_E(self) -> float:
        return self.diagnostics["N_eff_E"]

    @property
    def L90(self) -> pd.Series:
        return self.references.quantile_spectra.loc["L90"]

    @property
    def L50(self) -> pd.Series:
        return self.references.quantile_spectra.loc["L50"]

    @property
    def L10(self) -> pd.Series:
        return self.references.quantile_spectra.loc["L10"]

    @property
    def medoid_spectrum(self) -> pd.Series:
        return self.references.medoid_spectrum

    @property
    def ecc_spectrum(self) -> pd.Series:
        return self.references.ecc_spectrum

    @property
    def medoid_observations(self) -> pd.DataFrame:
        return self.S.to_frame(copy=False).iloc[
            self.references.medoid_observation_positions
        ]

    @property
    def ecc_observations(self) -> pd.DataFrame:
        return self.S.to_frame(copy=False).iloc[
            self.references.ecc_observation_positions
        ]
