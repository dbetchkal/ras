"""Basic geometry on the unit hypersphere and spectral contrast."""

from __future__ import annotations

import numpy as np

from .exceptions import GeometryError


def normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be two-dimensional.")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def normalize_vector(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x)
    if norm <= eps:
        raise GeometryError("Cannot normalize a zero-length vector.")
    return x / norm


def spherical_distance(x: np.ndarray, y: np.ndarray) -> float:
    x = normalize_vector(x)
    y = normalize_vector(y)
    return float(np.arccos(np.clip(np.dot(x, y), -1.0, 1.0)))


def angular_separation_from_medoid(X: np.ndarray, m_star: np.ndarray) -> np.ndarray:
    """Vectorized angular distance helper implied by the notebook's later cell."""
    X = normalize_rows(X)
    m = normalize_vector(m_star)
    return np.arccos(np.clip(X @ m, -1.0, 1.0))


def contrast_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Wan et al. spectral contrast angle in degrees, as defined in the notebook."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator <= 0:
        raise GeometryError("Contrast angle is undefined for zero vectors.")
    cosine = np.clip(np.dot(a, b) / denominator, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))
