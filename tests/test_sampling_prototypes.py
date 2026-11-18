import numpy as np
import pandas as pd

from ras.sampling import entropy_per_row, stratified_entropy_positions, stratified_entropy_sample
from ras.prototypes import assign_prototypes, spherical_medoid, closest_prototype, furthest_prototype


def test_entropy_is_source_exact_for_signed_vectors():
    M = np.array([[0.5, 0.5], [1.0, -0.2], [-0.2, -0.8]])
    expected = np.array([
        -0.5*np.log(0.5) - 0.5*np.log(0.5),
        -1.0*np.log(1.0),
        0.0,
    ])
    np.testing.assert_allclose(entropy_per_row(M), expected)


def test_stratified_sampler_reproducible_and_temporally_ordered():
    rng = np.random.default_rng(3)
    M = rng.normal(size=(200, 8))
    a = stratified_entropy_positions(M, N=100, n_bins=10, random_state=679)
    b = stratified_entropy_positions(M, N=100, n_bins=10, random_state=679)
    np.testing.assert_array_equal(a, b)
    assert np.all(np.diff(a) >= 0)
    df = pd.DataFrame(M, index=pd.date_range("2025", periods=200, freq="s"))
    sample = stratified_entropy_sample(df, N=100, n_bins=10, random_state=679)
    assert sample.index.is_monotonic_increasing


def test_assign_prototypes_chunked():
    X = np.eye(3, dtype=float)
    M = np.array([[1, 0, 0], [0, 1, 0], [0.8, 0.2, 0]], dtype=float)
    assigned = assign_prototypes(M, X, chunk_size=2)
    assert assigned.tolist() == [0, 1, 0]


def test_spherical_medoid_and_extremal_prototype():
    X = np.eye(3, dtype=float)
    medoid, idx = spherical_medoid(X, np.array([0.8, 0.1, 0.1]))
    assert idx == 0
    np.testing.assert_allclose(medoid, X[0])
    assert closest_prototype(X[0], X) == 0
    assert furthest_prototype(X[0], X) in (1, 2)
