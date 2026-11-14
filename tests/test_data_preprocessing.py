import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter1d, grey_opening

from ras.config import PreprocessingConfig
from ras.data import Spectrogram, select_frequency_band
from ras.exceptions import InputContractError
from ras.preprocessing import compute_exposure, estimate_exposure


def test_input_preserves_datetime_and_frequency_labels(spectrogram):
    spec = Spectrogram.from_frame(spectrogram)
    assert spec.time.equals(spectrogram.index)
    assert spec.frequency.labels.equals(spectrogram.columns)
    assert spec.to_frame().equals(spectrogram)


def test_custom_frequency_labels_with_explicit_centers():
    index = pd.date_range("2025-01-01", periods=4, freq="s")
    frame = pd.DataFrame(
        np.arange(12).reshape(4, 3),
        index=index,
        columns=["low", "mid", "high"],
    )
    spec = Spectrogram.from_frame(
        frame, frequency_centers_hz=np.array([12.5, 31.5, 100.0])
    )
    assert spec.frequency.centers_hz.tolist() == [12.5, 31.5, 100.0]
    assert spec.frequency.labels.tolist() == ["low", "mid", "high"]


def test_frequency_band_selection_inclusive_and_order_preserving():
    index = pd.date_range("2025-01-01", periods=4, freq="s")
    frame = pd.DataFrame(
        np.arange(16).reshape(4, 4),
        index=index,
        columns=["100", "12.5", "50", "25"],
    )
    selected = select_frequency_band(frame, 25, 100)
    assert selected.columns.tolist() == ["100", "50", "25"]


def test_nonnumeric_frequency_labels_require_centers():
    index = pd.date_range("2025-01-01", periods=4, freq="s")
    frame = pd.DataFrame(np.ones((4, 2)), index=index, columns=["low", "high"])
    with pytest.raises(InputContractError, match="frequency_centers_hz"):
        Spectrogram.from_frame(frame)


def test_irregular_datetime_index_rejected(spectrogram):
    frame = spectrogram.copy()
    idx = list(frame.index)
    idx[10] += pd.Timedelta(milliseconds=200)
    frame.index = pd.DatetimeIndex(idx)
    with pytest.raises(InputContractError, match="regularly sampled"):
        Spectrogram.from_frame(frame).sampling_interval_seconds()


def test_compute_exposure_matches_direct_source_formula(spectrogram):
    spec = Spectrogram.from_frame(spectrogram)
    cfg = PreprocessingConfig(
        epsilon_t_seconds=5,
        frequency_footprint_bands=3,
        f_lower_hz=12.5,
        f_upper_hz=200.0,
    )
    result = estimate_exposure(spec, cfg)

    O = grey_opening(spec.values, footprint=np.ones((5, 3), dtype=bool))
    S_O = spec.values - O
    G = gaussian_filter1d(S_O, sigma=5 / 2.355, axis=0)
    S_E = S_O - G
    selected = S_E
    norms = np.linalg.norm(selected, axis=1, keepdims=True)
    M = selected / np.maximum(norms, 1e-12)

    np.testing.assert_allclose(result.S_O.values, S_O)
    np.testing.assert_allclose(result.S_E.values, S_E)
    np.testing.assert_allclose(result.M.values, M)
    assert result.S_O.time.equals(spec.time)
    assert result.S_E.frequency.labels.equals(spec.frequency.labels)


def test_compute_exposure_dataframe_return_order(spectrogram):
    S_E, S_O, M_df = compute_exposure(
        spectrogram,
        f_lower=25,
        f_upper=100,
        epsilon_t=5,
        frequency_footprint_bands=3,
    )
    assert S_E.shape == spectrogram.shape
    assert S_O.shape == spectrogram.shape
    assert list(M_df.columns) == ["25", "50", "100"]



def test_one_sided_frequency_bounds_are_supported():
    index = pd.date_range("2025-01-01", periods=4, freq="s")
    frame = pd.DataFrame(
        np.ones((4, 4)),
        index=index,
        columns=["12.5", "25", "50", "100"],
    )
    assert select_frequency_band(frame, f_lower=50).columns.tolist() == ["50", "100"]
    assert select_frequency_band(frame, f_upper=25).columns.tolist() == ["12.5", "25"]
