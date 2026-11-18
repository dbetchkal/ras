"""Medoid-centered spherical log map, tangent basis, and weighted PCA."""

from __future__ import annotations

import numpy as np
from scipy.linalg import null_space

from .geometry import normalize_rows, normalize_vector


def sphere_log_map(
    X: np.ndarray,
    m_star: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Riemannian log map from unit-sphere points to T_m S^(D-1)."""
    X = normalize_rows(X)
    m = normalize_vector(m_star)
    cos_theta = np.clip(X @ m, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    V = X - cos_theta[:, None] * m[None, :]
    sin_theta = np.sin(theta)
    scale = np.ones_like(theta)
    mask = sin_theta > eps
    scale[mask] = theta[mask] / sin_theta[mask]
    V *= scale[:, None]
    V[theta < eps] = 0.0
    return V, theta


def tangent_basis(m_star: np.ndarray) -> np.ndarray:
    """Orthonormal basis for the tangent space at m_star."""
    m = normalize_vector(m_star)
    return null_space(m[None, :])


def weighted_pca(
    Z: np.ndarray,
    w: np.ndarray,
    n_components: int = 2,
):
    """Port the notebook's occupancy-weighted covariance eigendecomposition."""
    Z = np.asarray(Z, dtype=float)
    weights = np.asarray(w, dtype=float).copy()
    if Z.ndim != 2 or weights.shape != (len(Z),):
        raise ValueError("w must have one weight per row of Z.")
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("PCA weights must be finite, nonnegative, and nonzero.")
    if n_components < 1 or n_components > Z.shape[1]:
        raise ValueError("n_components must be between 1 and tangent dimension.")
    weights /= weights.sum()

    mu = np.sum(Z * weights[:, None], axis=0)
    Zc = Z - mu
    covariance = (Zc * weights[:, None]).T @ Zc
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    components = eigvecs[:, :n_components]
    scores = Zc @ components
    return scores, components, eigvals, mu


def analyze_tangent_space(
    X_N: np.ndarray,
    m_star: np.ndarray,
    weights: np.ndarray,
    n_components: int = 2,
) -> dict[str, np.ndarray]:
    """Compute both weighted-centered and medoid-anchored PCA coordinates."""
    X = normalize_rows(X_N)
    V, theta = sphere_log_map(X, m_star)
    basis = tangent_basis(m_star)
    Z = V @ basis
    scores_centered, components, eigvals, tangent_mean = weighted_pca(
        Z, weights, n_components=n_components
    )
    scores_pole = Z @ components
    direction = np.zeros_like(V)
    norms = np.linalg.norm(V, axis=1)
    mask = norms > 1e-12
    direction[mask] = V[mask] / norms[mask, None]
    return {
        "V": V,
        "theta": theta,
        "tangent_basis": basis,
        "Z": Z,
        "scores_centered": scores_centered,
        "scores_pole": scores_pole,
        "components": components,
        "eigenvalues": eigvals,
        "tangent_mean": tangent_mean,
        "direction": direction,
    }
