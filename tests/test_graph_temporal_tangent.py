import networkx as nx
import numpy as np
import pandas as pd
import pytest

from ras.graph import (
    aggregate_pith_occupancy, build_geodesic_mst, family_nodes,
    ordered_path_nodes, pith_geometry, pith_layers, tree_diameter_path,
)
from ras.tangent import sphere_log_map, tangent_basis, weighted_pca
from ras.temporal import prototype_runs, prototype_temporal_dynamics, hypersphere_velocity_weighted


def test_mst_pith_and_attachment_geometry():
    # Four nonduplicate points on S^2.
    X = np.array([
        [1, 0, 0],
        [0.7, 0.7, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=float)
    G, D = build_geodesic_mst(X)
    assert nx.is_tree(G)
    path = tree_diameter_path(G)
    pith = G.subgraph(path).copy()
    distance, attachment, family = pith_geometry(G, pith)
    assert set(attachment) == set(G.nodes)
    assert set(family) == set(G.nodes)
    assert set(ordered_path_nodes(pith)) == set(pith.nodes)
    fam_nodes = family_nodes(G, family, attachment)
    assert sum(len(x) for x in fam_nodes.values()) == len(G)
    nodes, occ = aggregate_pith_occupancy(G, pith, np.ones(4), attachment)
    assert len(nodes) == len(occ)
    layer, attach_topo = pith_layers(G, pith)
    assert set(layer) == set(G.nodes)


def test_zero_distance_duplicate_prototypes_report_mst_issue():
    X = np.array([[1,0,0], [1,0,0], [1,0,0]], dtype=float)
    from ras.exceptions import GraphConstructionError
    with pytest.raises(GraphConstructionError):
        build_geodesic_mst(X)


def test_prototype_runs_and_dynamics():
    q = np.array([0,0,1,1,1,0,2,2])
    states, starts, durations = prototype_runs(q)
    assert states.tolist() == [0,1,0,2]
    assert starts.tolist() == [0,2,5,6]
    assert durations.tolist() == [2,3,1,2]
    stats = prototype_temporal_dynamics(q, 3, time_step_seconds=2.0)
    assert stats.loc[0, "occupancy"] == 3/8
    assert stats.loc[1, "n_visits"] == 1
    assert stats.loc[0, "mean_dwell_seconds"] == 3.0


def test_sphere_log_map_and_tangent_basis():
    X = np.eye(3)
    m = np.array([1.0,0,0])
    V, theta = sphere_log_map(X, m)
    assert theta[0] == pytest.approx(0)
    assert np.linalg.norm(V[0]) == pytest.approx(0)
    E = tangent_basis(m)
    assert E.shape == (3,2)
    np.testing.assert_allclose(m @ E, np.zeros(2), atol=1e-12)


def test_weighted_pca():
    Z = np.array([[0.,0.], [1.,0.], [0.,1.]])
    score, components, eigvals, mu = weighted_pca(Z, np.array([0.5,0.25,0.25]), 2)
    assert score.shape == (3,2)
    assert components.shape == (2,2)
    assert np.all(np.diff(eigvals) <= 0)


def test_hypersphere_velocity_preserves_index():
    idx = pd.date_range("2025-01-01", periods=3, freq="s")
    M = pd.DataFrame([[1,0], [0.9,0.1], [0.8,0.2]], index=idx, columns=["a","b"])
    v, theta = hypersphere_velocity_weighted(M)
    assert v.index.equals(idx) and theta.index.equals(idx)
    assert np.isnan(v.iloc[0])
