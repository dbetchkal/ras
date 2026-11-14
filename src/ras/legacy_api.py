"""Compatibility façade exposing the supplied sound_geometry public API."""

from __future__ import annotations

from .audibility import (
    audibility_sir,
    binary_audibility_time_series,
    join_src_to_nvspl,
)
from .diagnostics import effective_number, renyi_effective_number
from .geometry import (
    angular_separation_from_medoid,
    contrast_1d,
    normalize_rows,
    normalize_vector,
)
from .graph import (
    aggregate_pith_occupancy,
    family_nodes,
    ordered_path_edges,
    ordered_path_nodes,
    pith_geometry,
    pith_layers,
    tree_diameter_graph,
    tree_diameter_path,
)
from .plotting import event_plotter
from .data import select_frequency_band
from .preprocessing import compute_exposure
from .prototypes import (
    assign_prototypes,
    closest_prototype,
    furthest_prototype,
    prototype_indices,
    spherical_medoid,
)
from .references import closest_row_euclidean, furthest_observed_spectrum
from .sampling import entropy_per_row, stratified_entropy_sample
from .stsw import run_stsw as _run_stsw
from .config import STSWConfig
from .tangent import sphere_log_map, tangent_basis, weighted_pca
from .temporal import (
    hypersphere_velocity_weighted,
    prototype_runs,
    prototype_temporal_dynamics,
)


def run_stsw(
    Xt,
    n_components=256,
    n_steps=300,
    batch_size=512,
    lr=1e-2,
    ntrees=16,
    nlines=3,
    device="cpu",
    random_state=679,
):
    """Compatibility signature matching the supplied sound_geometry function."""
    config = STSWConfig(
        n_components=n_components,
        n_steps=n_steps,
        batch_size=batch_size,
        lr=lr,
        ntrees=ntrees,
        nlines=nlines,
        device=device,
    )
    return _run_stsw(Xt, config, random_state=random_state)


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
    # notebook-local helpers
    "ordered_path_edges",
    "ordered_path_nodes",
    "contrast_1d",
    "binary_audibility_time_series",
    "join_src_to_nvspl",
    "audibility_sir",
    "effective_number",
    "renyi_effective_number",
    "hypersphere_velocity_weighted",
    "angular_separation_from_medoid",
]
