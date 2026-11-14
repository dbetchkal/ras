import importlib.util
import numpy as np
import pytest

from ras.config import STSWConfig
from ras.stsw import run_stsw


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch optional")
def test_stsw_respects_n_components_not_batch_size():
    import torch

    samples = np.eye(4, dtype=np.float32)
    config = STSWConfig(
        n_components=3, n_steps=2, batch_size=7, lr=1e-2,
        ntrees=2, nlines=1, device="cpu",
    )

    def fake_objective(batch, prototypes, **kwargs):
        return ((batch.mean(dim=0) - prototypes.mean(dim=0)) ** 2).sum()

    X, losses = run_stsw(
        samples, config, random_state=17, stswd_function=fake_objective
    )
    assert X.shape == (3, 4)
    assert len(losses) == 2
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-6)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch optional")
def test_stsw_seed_reproducibility_with_deterministic_objective():
    import torch

    samples = np.eye(3, dtype=np.float32)
    config = STSWConfig(
        n_components=2, n_steps=2, batch_size=4, lr=1e-2,
        ntrees=2, nlines=1, device="cpu",
    )

    def fake_objective(batch, prototypes, **kwargs):
        return ((batch.mean(dim=0) - prototypes.mean(dim=0)) ** 2).sum()

    a, la = run_stsw(samples, config, random_state=91, stswd_function=fake_objective)
    b, lb = run_stsw(samples, config, random_state=91, stswd_function=fake_objective)
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(la, lb)
