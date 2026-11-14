import numpy as np
import pandas as pd

from ras import RAS, RASConfig
from ras.config import (
    AssignmentConfig, GraphConfig, PreprocessingConfig,
    SamplingConfig, STSWConfig, TangentConfig,
)


def test_end_to_end_pipeline_from_dataframe(spectrogram):
    def fake_stsw(samples, config, random_state):
        # Deterministic unit representatives selected from real M observations.
        rows = np.linspace(0, len(samples)-1, config.n_components, dtype=int)
        X = samples[rows].astype(float)
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        return X, [1.0, 0.5, 0.25]

    cfg = RASConfig(
        preprocessing=PreprocessingConfig(
            epsilon_t_seconds=5,
            frequency_footprint_bands=3,
            f_lower_hz=12.5,
            f_upper_hz=200.0,
        ),
        sampling=SamplingConfig(n_samples=80, n_bins=8, random_state=679),
        stsw=STSWConfig(
            n_components=5, n_steps=3, batch_size=16, ntrees=4, nlines=2
        ),
        assignment=AssignmentConfig(chunk_size=13),
        graph=GraphConfig(),
        tangent=TangentConfig(n_components=2),
        random_state=679,
    )
    model = RAS(cfg, stsw_backend=fake_stsw)
    result = model.analyze(spectrogram)

    assert result.S_O.shape == spectrogram.shape
    assert result.S_E.shape == spectrogram.shape
    assert result.M.shape == (len(spectrogram), 5)
    assert result.X_N.shape == (5, 5)
    assert result.prototype_assignment.shape == (len(spectrogram),)
    assert result.S_Pi.index.equals(spectrogram.index)
    assert {"prototype", "pith_family"}.issubset(result.S_Pi.columns)
    assert nx_is_tree(result.G_MST)
    assert set(result.pith_family_by_observation).issubset(
        set(result.pith_family_by_prototype.values())
    )
    assert result.N_eff_T > 0 and result.N_eff_E > 0
    assert result.tangent["scores_pole"].shape[0] == 5
    assert result.v_H.index.equals(spectrogram.index)
    assert result.references.quantile_spectra.index.tolist() == ["L90","L50","L10"]
    assert model.X_N_.shape == (5,5)


def nx_is_tree(graph):
    import networkx as nx
    return nx.is_tree(graph)
