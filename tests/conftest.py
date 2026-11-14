import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def spectrogram():
    rng = np.random.default_rng(491)
    index = pd.date_range("2025-06-01", periods=96, freq="s", name="time")
    centers = np.array([12.5, 25.0, 50.0, 100.0, 200.0])
    # Distinct low-amplitude spectral variation around plausible dB levels.
    values = 45 + rng.normal(0, 3, size=(len(index), len(centers)))
    frame = pd.DataFrame(values, index=index, columns=[f"{x:g}" for x in centers])
    return frame
