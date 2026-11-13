import cmocean
from collections import deque
import glob
from itertools import cycle
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.linalg import null_space
from scipy.ndimage import gaussian_filter1d, grey_opening
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from scipy.special import softmax
from sklearn.decomposition import PCA
import sys
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import umap

# ----------------------------------------------------------------------
repo_dir = "/home/waxy/Documents/SCRIPTS/GITHUB"
# ----------------------------------------------------------------------
sys.path.append(os.path.join(repo_dir, "iyore"))
sys.path.append(os.path.join(repo_dir, "NPS-ActiveSpace"))
sys.path.append(os.path.join(repo_dir, "STSW"))
import iyore
from nps_active_space.utils.models import Nvspl, Srcid
from methods import stswd, wd

__all__ = [
    "aggregate_pith_occupancy",
    "assign_prototypes",
    "closest_prototype",
    "closest_row_euclidean",
    "compute_exposure",
    "entropy_per_row",
    "event_plotter",
    "family_nodes",
    "furthest_observed_spectrum",
    "furthest_prototype",
    "normalize_rows",
    "normalize_vector",
    "pith_geometry",
    "pith_layers",
    "prototype_indices",
    "prototype_runs",
    "prototype_temporal_dynamics",
    "run_stsw",
    "select_frequency_band",
    "sphere_log_map",
    "spherical_medoid",
    "stratified_entropy_sample",
    "tangent_basis",
    "tree_diameter_graph",
    "tree_diameter_path",
    "weighted_pca",
]

def aggregate_pith_occupancy(G, pith, occupancy, pith_attachment=None):
    """
    Aggregate prototype occupancy onto pith nodes.

    Each prototype contributes its occupancy to the pith node
    defining its morphological family.
    """
    occupancy = np.asarray(occupancy, dtype=float)

    if pith_attachment is None:
        _, pith_attachment, _ = pith_geometry(
            G,
            pith,
            weight="geo_distance"
        )

    pith_nodes = list(pith.nodes)

    result = {
        node: 0.0
        for node in pith_nodes
    }

    for node in G.nodes:
        attachment = pith_attachment[node]
        result[attachment] += occupancy[node]

    return (
        np.asarray(pith_nodes, dtype=int),
        np.asarray(
            [result[node] for node in pith_nodes],
            dtype=float
        )
    )


def assign_prototypes(M, X_N, chunk_size=100_000):
    """
    Assign each spectral state to its nearest spherical prototype.

    Parameters
    ----------
    M : array-like, shape (T, D)
        Unit-normalized spectral-state coordinates.

    X_N : array-like, shape (K, D)
        Unit-normalized prototypes.

    Returns
    -------
    assignment : ndarray, shape (T,)
        Prototype index assigned to each observation.
    """
    M = np.asarray(M, dtype=np.float32)
    X_N = normalize_rows(
        np.asarray(X_N, dtype=np.float32)
    )

    assignment = np.empty(
        len(M),
        dtype=np.int16
    )

    for start in range(0, len(M), chunk_size):

        stop = min(
            start + chunk_size,
            len(M)
        )

        similarity = M[start:stop] @ X_N.T

        assignment[start:stop] = np.argmax(
            similarity,
            axis=1
        )

    return assignment


def closest_prototype(q, X):
    """
    Return the prototype minimizing spherical/geodesic angle
    to unit vector q.
    """
    cos_sim = X @ q
    return np.argmax(cos_sim)


def closest_row_euclidean(A, target, chunk_size=200_000):
    target = np.asarray(target, dtype=np.float32)
    target_norm2 = np.dot(target, target)

    best_idx = -1
    best_d2 = np.inf

    for start in range(0, len(A), chunk_size):
        stop = min(start + chunk_size, len(A))

        X = np.asarray(
            A.iloc[start:stop].values,
            dtype=np.float32
        )

        d2 = (
            np.sum(X * X, axis=1)
            - 2.0 * (X @ target)
            + target_norm2
        )

        local = np.argmin(d2)

        if d2[local] < best_d2:
            best_d2 = d2[local]
            best_idx = start + local

    return best_idx


def compute_exposure(S, f_lower=20, f_upper=1250, epsilon_t=12*60):
    """
    Compute scale-selective acoustic exposure and project it onto a
    hypersphere using an optional frequency sub-band.

    Parameters
    ----------
    S : pandas.DataFrame
        Spectrogram. Columns represent frequency bands in Hz.

    f_lower : float or None
        Lower frequency bound in Hz, inclusive. Use None for no lower bound.

    f_upper : float or None
        Upper frequency bound in Hz, inclusive. Use None for no upper bound.

    epsilon_t : int
        Temporal scale of the grayscale opening, in seconds.

    Returns
    -------
    S_E : pandas.DataFrame
        Scale-selective exposure over the full frequency range.

    S_O : pandas.DataFrame
        Scale-selective subtraction of morphological opening over the full frequency range.

    M_df : pandas.DataFrame
        Unit-normalized spectral states restricted to the requested
        frequency sub-band.
    """

    dt_ = 1
    sigma = epsilon_t / dt_ / 2.355
    footprint = np.ones((epsilon_t, 33), dtype=bool)

    print("Apply greyscale opening...")

    O = pd.DataFrame(
        grey_opening(S, footprint=footprint),
        index=S.index,
        columns=S.columns
    )

    S_O = S - O

    print("Apply Gaussian high-pass filter...")

    G = pd.DataFrame(
        gaussian_filter1d(
            S_O.values,
            sigma=sigma,
            axis=0
        ),
        index=S.index,
        columns=S.columns
    )

    S_E = S_O - G

    print("Select frequency band...")

    S_band = select_frequency_band(S_E,
        f_lower=f_lower,
        f_upper=f_upper)

    print(
        f"Using {S_band.shape[1]} frequency bands "
        f"from {float(pd.to_numeric(S_band.columns).min()):g} "
        f"to {float(pd.to_numeric(S_band.columns).max()):g} Hz."
    )

    print("Normalize to hypersphere...")

    A = S_band.values.astype(np.float64)

    norms = np.linalg.norm(A, axis=1, keepdims=True)

    M = A / np.maximum(norms, 1e-12)

    M_df = pd.DataFrame(
        M,
        index=S_band.index,
        columns=S_band.columns
    )

    return S_E, S_O, M_df


def entropy_per_row(M):

    M = np.asarray(M, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        logM = np.where(M > 0, np.log(M), 0)

    return -np.sum(M * logM, axis=1)


def event_plotter(event, ax, vmin=-10, vmax=90, xtickscale="diel", cmap="plasma"):

    x_lims = mdates.date2num(event.index)

    im = ax.imshow(
        event.T,
        extent=[x_lims[0], x_lims[-1], 0, event.shape[1]],
        cmap=cmap,
        origin="lower",
        interpolation="none",
        aspect="auto",
        vmin=vmin,
        vmax=vmax
    )

    ax.set_yticks(np.arange(event.shape[1])[::8])
    ax.set_yticklabels(event.columns.astype(float)[::8], fontsize=8)
    ax.xaxis_date()

    if xtickscale == "diel":
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d\n%H:%M"))

    return im


def family_nodes(G, pith_family, attachment):
    """
    Return prototype indices grouped by pith attachment.
    """
    families = {}

    for node in G.nodes:
        p = attachment[node]
        families.setdefault(p, []).append(node)

    return {
        p: np.asarray(nodes, dtype=int)
        for p, nodes in families.items()
    }


def furthest_observed_spectrum(q, M_df, S, X_N):
    """
    Find the observed spectrum whose normalized exposure state
    is closest in spherical angle to prototype q.
    """
    cos_sim = M_df.values @ q
    idx = np.argmin(cos_sim)
    return S.iloc[idx, :]


def furthest_prototype(q, X, mask=None):
    """
    Return the prototype maximizing spherical/geodesic distance
    from unit vector q.

    Parameters
    ----------
    q : ndarray
        Reference unit vector.
    X : ndarray
        Unit-normalized prototype matrix, shape (N, B).
    mask : ndarray of bool, optional
        Restricts the search to eligible prototypes.

    Returns
    -------
    int
        Index of the furthest eligible prototype.
    """
    cos_sim = X @ q

    if mask is not None:
        if not np.any(mask):
            raise ValueError("No prototypes satisfy the supplied mask.")

        candidate_idx = np.flatnonzero(mask)
        return candidate_idx[np.argmin(cos_sim[candidate_idx])]

    return np.argmin(cos_sim)

    
def normalize_rows(X, eps=1e-12):
    """Normalize each row of X to unit Euclidean length."""
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def normalize_vector(x, eps=1e-12):
    """Normalize a single vector."""
    x = np.asarray(x, dtype=float)
    return x / max(np.linalg.norm(x), eps)


def pith_geometry(G, pith, weight="geo_distance"):
    """
    Decompose an MST relative to its diameter (pith).

    Returns
    -------
    pith_distance : dict
        Weighted geodesic distance from each node to the pith.

    pith_attachment : dict
        Pith node to which each node is attached.

    pith_family : dict
        Morphological family indexed by its pith attachment node.
    """
    if not nx.is_tree(G):
        raise ValueError("G must be a connected tree.")

    pith_nodes = set(pith.nodes)

    distance, paths = nx.multi_source_dijkstra(
        G,
        sources=pith_nodes,
        weight=weight
    )

    attachment = {
        node: path[0]
        for node, path in paths.items()
    }

    family = attachment.copy()

    return distance, attachment, family


def pith_layers(G, pith):
    """
    Assign every prototype to its nearest pith node
    in topological distance.
    """
    pith_nodes = set(pith.nodes)

    layer = {}
    attachment = {}

    queue = deque()

    for node in pith_nodes:
        layer[node] = 0
        attachment[node] = node
        queue.append(node)

    while queue:
        node = queue.popleft()

        for neighbor in G.neighbors(node):

            if neighbor not in layer:
                layer[neighbor] = layer[node] + 1
                attachment[neighbor] = attachment[node]
                queue.append(neighbor)

    return layer, attachment
 

def prototype_indices(assignment, prototype):
    return np.flatnonzero(assignment == prototype)  


def prototype_runs(q):
    """
    Convert a prototype sequence into contiguous runs.

    Parameters
    ----------
    q : array-like, shape (T,)
        Prototype index at each 1-second time step.

    Returns
    -------
    states : ndarray
        Prototype index for each run.
    starts : ndarray
        Start time of each run, in samples/seconds.
    durations : ndarray
        Duration of each run, in samples/seconds.
    """
    q = np.asarray(q)

    if q.ndim != 1 or len(q) == 0:
        raise ValueError("q must be a non-empty 1-D array.")

    change = np.r_[True, q[1:] != q[:-1]]

    starts = np.flatnonzero(change)
    states = q[starts]

    ends = np.r_[starts[1:], len(q)]

    durations = ends - starts

    return states, starts, durations

    
def prototype_temporal_dynamics(q, n_prototypes):
    """
    Calculate persistence and return-time statistics for each prototype.

    A "visit" is one contiguous run of a prototype.

    Returns
    -------
    stats : pandas.DataFrame
        One row per prototype.
    """
    import pandas as pd

    q = np.asarray(q)

    states, starts, durations = prototype_runs(q)

    records = []

    for n in range(n_prototypes):

        idx = np.flatnonzero(states == n)

        if len(idx) == 0:
            records.append({
                "prototype": n,
                "occupancy": 0.0,
                "n_visits": 0,
                "mean_dwell": np.nan,
                "median_dwell": np.nan,
                "mean_return": np.nan,
                "median_return": np.nan,
                "cv_return": np.nan,
            })
            continue

        d = durations[idx]

        # Occupancy: fraction of all seconds assigned to this prototype
        occupancy = d.sum() / len(q)

        # Return interval:
        # time from the START of one visit to the START of the next visit.
        if len(idx) > 1:
            return_times = np.diff(starts[idx])

            mean_return = return_times.mean()
            median_return = np.median(return_times)

            if mean_return > 0:
                cv_return = return_times.std(ddof=1) / mean_return
            else:
                cv_return = np.nan

        else:
            mean_return = np.nan
            median_return = np.nan
            cv_return = np.nan

        records.append({
            "prototype": n,
            "occupancy": occupancy,
            "n_visits": len(idx),
            "mean_dwell": d.mean(),
            "median_dwell": np.median(d),
            "mean_return": mean_return,
            "median_return": median_return,
            "cv_return": cv_return,
        })

    return pd.DataFrame(records)


def run_stsw(Xt, n_components=256, n_steps=300, batch_size=512, lr=1e-2, ntrees=16, nlines=3, device="cpu"):

    n, d = Xt.shape

    Xt = Xt.to(device)

    X0 = torch.randn(
        (batch_size, d),
        device=device
    )

    X0 = X0 / torch.norm(
        X0,
        dim=1,
        keepdim=True
    )

    X0.requires_grad_(True)

    optimizer = torch.optim.Adam(
        [X0],
        lr=lr
    )

    loss_history = []

    for step in range(n_steps):

        optimizer.zero_grad()

        # Random minibatch, with replacement
        idx = torch.randint(
            0,
            n,
            (batch_size,),
            device=device
        )

        batch = Xt[idx]

        sw = stswd.stswd(
            batch,
            X0,
            ntrees=ntrees,
            nlines=nlines,
            device=device
        )

        sw.backward()

        optimizer.step()

        with torch.no_grad():
            X0 /= torch.norm(
                X0,
                dim=1,
                keepdim=True
            )

        loss_history.append(sw.item())

    return X0.detach().cpu().numpy(), loss_history


def select_frequency_band(S, f_lower=None, f_upper=None):
    """
    Select a numerical frequency sub-band from a spectrogram.

    Parameters
    ----------
    S : pandas.DataFrame
        Spectrogram with frequency bands in columns. Column labels may be
        strings, integers, or floats representing frequency in Hz.

    f_lower : float or None
        Lower frequency bound in Hz, inclusive.

    f_upper : float or None
        Upper frequency bound in Hz, inclusive.

    Returns
    -------
    S_band : pandas.DataFrame
        Spectrogram restricted to the requested frequency band.

    Raises
    ------
    ValueError
        If no frequency bands fall within the requested interval.
    """

    freq = pd.to_numeric(
        pd.Index(S.columns),
        errors="coerce"
    )

    if np.any(np.isnan(freq)):
        bad = pd.Index(S.columns)[np.isnan(freq)]
        raise ValueError(
            f"Non-numeric frequency column labels encountered: {list(bad)}"
        )

    mask = np.ones(len(freq), dtype=bool)

    if f_lower is not None:
        mask &= freq >= float(f_lower)

    if f_upper is not None:
        mask &= freq <= float(f_upper)

    if not np.any(mask):
        raise ValueError(
            f"No frequency bands found between "
            f"{f_lower} and {f_upper} Hz."
        )

    return S.iloc[:, mask]


def sphere_log_map(X, m_star, eps=1e-12):
    """
    Riemannian logarithm map on S^(D-1).

    Maps each X[i] to a tangent vector at m_star:

        log_m*(X[i])

    Parameters
    ----------
    X : (N, D)
        Unit vectors on the sphere.
    m_star : (D,)
        Unit reference point / data-defined pole.

    Returns
    -------
    V : (N, D)
        Tangent vectors at m_star.
    theta : (N,)
        Geodesic distances from m_star.
    """

    X = normalize_rows(X)
    m_star = normalize_vector(m_star)

    # cos(theta) = m* dot n
    cos_theta = np.clip(X @ m_star, -1.0, 1.0)

    theta = np.arccos(cos_theta)

    # Projection of X onto tangent space at m*
    V = X - cos_theta[:, None] * m_star[None, :]

    sin_theta = np.sin(theta)

    # theta / sin(theta)
    scale = np.ones_like(theta)

    mask = sin_theta > eps
    scale[mask] = theta[mask] / sin_theta[mask]

    V *= scale[:, None]

    # At m* itself, log_m*(m*) = 0
    near_zero = theta < eps
    V[near_zero] = 0.0

    return V, theta

   
def spherical_medoid(X, w):

    """
    Occupancy-weighted spherical medoid.

    Finds the observed state X[i] minimizing the
    occupancy-weighted geodesic distance to all states.

    Parameters
    ----------
    X : (N, D)
        Unit vectors on S^(D-1).
    w : (N,)
        Nonnegative occupancy weights.

    Returns
    -------
    m_star : (D,)
        Occupancy-weighted spherical medoid.
    idx : int
        Index of the medoid in X.
    """
    X = normalize_rows(X)

    w = np.asarray(w, dtype=np.float64)
    w /= w.sum()

    D = np.arccos(np.clip(X @ X.T, -1.0, 1.0)) # pairwise geodesic distances

    objective = D @ w # expected geodesic distance from each observed state
    idx = np.argmin(objective)

    m_star = X[idx]

    return m_star, idx


def stratified_entropy_sample(M_df, N=2000, n_bins=100, random_state=42):

    rng = np.random.default_rng(random_state)

    H = entropy_per_row(M_df.values)

    labels = pd.qcut(H, q=n_bins, labels=False, duplicates="drop")
    labels = pd.Series(labels, index=M_df.index)

    picks = []
    per_bin = max(1, N // n_bins)
    
    for b in np.unique(labels):
        idx = labels[labels == b].index
        n_take = min(per_bin, len(idx))
        sel = rng.choice(idx, size=n_take, replace=False)
        picks.extend(sel)

    picks = pd.Index(picks)

    return M_df.loc[picks].sort_index()


def tangent_basis(m_star):
    """
    Orthonormal basis for T_m S^(D-1).
    """
    m_star = normalize_vector(m_star)

    # Columns span the orthogonal complement of m_star
    E = null_space(m_star[None, :])

    return E


def tree_diameter_graph(G):
    path = tree_diameter_path(G)

    H = G.subgraph(path).copy()

    return H


def tree_diameter_path(G):
    start = next(iter(G.nodes))

    lengths = nx.single_source_dijkstra_path_length(G,
                                                    start,
                                                    weight="geo_distance")
    u = max(lengths, key=lengths.get)

    lengths, paths = nx.single_source_dijkstra(G,
                                               u,
                                               weight="geo_distance")
    v = max(lengths, key=lengths.get)

    return paths[v]


def weighted_pca(Z, w, n_components=2):
    """
    Occupancy-weighted PCA of tangent-space coordinates.

    Parameters
    ----------
    Z : (N, K)
        Tangent-space coordinates.
    w : (N,)
        Occupancy weights.

    Returns
    -------
    scores : (N, n_components)
        PCA coordinates.
    components : (K, n_components)
        PCA directions.
    eigvals : (K,)
        Eigenvalues of weighted covariance matrix.
    mu : (K,)
        Weighted tangent-space mean.
    """

    Z = np.asarray(Z, dtype=float)

    w = np.asarray(w, dtype=float)
    w = w / w.sum()

    # Weighted mean
    mu = np.sum(Z * w[:, None], axis=0)

    # Center
    Zc = Z - mu

    # Weighted covariance
    C = (Zc * w[:, None]).T @ Zc

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(C)

    # Largest eigenvalues first
    order = np.argsort(eigvals)[::-1]

    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    components = eigvecs[:, :n_components]

    scores = Zc @ components

    return scores, components, eigvals, mu





   





