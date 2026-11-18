"""Scale-selective acoustic exposure: S -> S_O -> S_E -> M."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, grey_opening

from .config import PreprocessingConfig
from .data import Spectrogram
from .exceptions import InputContractError


@dataclass(frozen=True, slots=True)
class ExposureResult:
    """Numerical intermediates and frequency-selected hyperspherical states."""

    S_O: Spectrogram
    S_E: Spectrogram
    M: Spectrogram
    metadata: dict


def estimate_exposure(
    S: Spectrogram,
    config: PreprocessingConfig,
) -> ExposureResult:
    """Port the supplied `compute_exposure` sequence using index-derived dt.

    Source order is preserved: opening subtraction yields S_O; temporal
    Gaussian subtraction yields S_E; frequency selection and row-wise L2
    normalization yield M.
    """
    dt_seconds = S.sampling_interval_seconds()
    epsilon_samples = max(1, int(round(config.epsilon_t_seconds / dt_seconds)))
    footprint = np.ones(
        (epsilon_samples, config.frequency_footprint_bands),
        dtype=bool,
    )

    opened = grey_opening(S.values, footprint=footprint)
    S_O_values = S.values - opened
    del opened

    sigma_samples = (
        config.epsilon_t_seconds / dt_seconds / config.gaussian_fwhm_divisor
    )
    smooth = gaussian_filter1d(S_O_values, sigma=sigma_samples, axis=0)
    S_E_values = S_O_values - smooth
    del smooth

    S_O = Spectrogram(
        S_O_values, S.time, S.frequency,
        {**S.attrs, "ras_stage": "morphological_opening_subtraction"},
    )
    S_E = Spectrogram(
        S_E_values, S.time, S.frequency,
        {
            **S.attrs,
            "ras_stage": "temporal_gaussian_highpass",
            "epsilon_t_seconds": config.epsilon_t_seconds,
            "sampling_interval_seconds": dt_seconds,
            "epsilon_samples": epsilon_samples,
            "frequency_footprint_bands": config.frequency_footprint_bands,
            "gaussian_sigma_samples": sigma_samples,
        },
    )

    positions = S.frequency.select_positions(
        config.f_lower_hz, config.f_upper_hz
    )
    selected = S_E_values[:, positions]
    norms = np.linalg.norm(selected.astype(np.float64), axis=1, keepdims=True)
    M_values = selected.astype(np.float64) / np.maximum(
        norms, config.normalization_epsilon
    )
    M = Spectrogram(
        M_values,
        S.time,
        S.frequency.subset(positions),
        {
            **S_E.attrs,
            "ras_stage": "unit_l2_spectral_directions",
            "zero_norm_rows": int(np.count_nonzero(norms[:, 0] <= config.normalization_epsilon)),
        },
    )
    return ExposureResult(S_O=S_O, S_E=S_E, M=M, metadata=dict(M.attrs))


def compute_exposure(
    S: pd.DataFrame,
    f_lower: float | None = 20,
    f_upper: float | None = 1250,
    epsilon_t: float = 12 * 60,
    *,
    frequency_centers_hz: np.ndarray | pd.Series | None = None,
    frequency_footprint_bands: int = 33,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """DataFrame compatibility wrapper with the supplied function's return order.

    Returns `(S_E, S_O, M_df)`. Note that the notebook's actual call unpacks
    `S_E, S_O, M_df`, even though the first two intermediates are computed in
    the opposite order.
    """
    spec = Spectrogram.from_frame(
        S, frequency_centers_hz=frequency_centers_hz
    )
    cfg = PreprocessingConfig(
        epsilon_t_seconds=float(epsilon_t),
        frequency_footprint_bands=frequency_footprint_bands,
        f_lower_hz=f_lower,
        f_upper_hz=f_upper,
    )
    result = estimate_exposure(spec, cfg)
    return result.S_E.to_frame(), result.S_O.to_frame(), result.M.to_frame()
