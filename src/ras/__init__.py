"""Representative Acoustic States package."""

from .config import (
    AssignmentConfig,
    GraphConfig,
    PreprocessingConfig,
    RASConfig,
    SamplingConfig,
    STSWConfig,
    TangentConfig,
)
from .data import FrequencyAxis, Spectrogram, select_frequency_band
from .exceptions import (
    GeometryError,
    GraphConstructionError,
    InputContractError,
    RASDependencyError,
    RASError,
)
from .model import RAS
from .graph import build_geodesic_mst, diam_geometry, tree_diameter_path
from .states import assign_states, spherical_medoid
from .tangent import sphere_log_map, tangent_basis, weighted_pca
from .temporal import state_runs, state_temporal_dynamics, hypersphere_velocity_weighted
from .result import RASResult
from .sampling import entropy_per_row, stratified_entropy_sample
from .preprocessing import compute_exposure, estimate_exposure

__all__ = [
    "RAS",
    "RASConfig",
    "RASResult",
    "Spectrogram",
    "FrequencyAxis",
    "PreprocessingConfig",
    "SamplingConfig",
    "STSWConfig",
    "AssignmentConfig",
    "GraphConfig",
    "TangentConfig",
    "select_frequency_band",
    "compute_exposure",
    "estimate_exposure",
    "entropy_per_row",
    "stratified_entropy_sample",
    "assign_states",
    "spherical_medoid",
    "build_geodesic_mst",
    "diam_geometry",
    "tree_diameter_path",
    "sphere_log_map",
    "tangent_basis",
    "weighted_pca",
    "state_runs",
    "state_temporal_dynamics",
    "hypersphere_velocity_weighted",
    "RASError",
    "InputContractError",
    "GeometryError",
    "GraphConstructionError",
    "RASDependencyError",
]
