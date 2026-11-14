"""Explicit, immutable configuration for the notebook-derived RAS workflow."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Configuration for S -> S_O -> S_E -> M.

    The input notebook works with 1-second NVSPL rows. The implementation
    converts `epsilon_t_seconds` to samples from the input DatetimeIndex.
    `frequency_footprint_bands` retains the source's 33-band opening width.
    """

    epsilon_t_seconds: float = 30.0 * 60.0
    frequency_footprint_bands: int = 33
    f_lower_hz: float | None = 12.5
    f_upper_hz: float | None = 20_000.0
    gaussian_fwhm_divisor: float = 2.355
    normalization_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.epsilon_t_seconds <= 0:
            raise ValueError("epsilon_t_seconds must be positive.")
        if self.frequency_footprint_bands < 1:
            raise ValueError("frequency_footprint_bands must be positive.")
        if self.gaussian_fwhm_divisor <= 0 or self.normalization_epsilon <= 0:
            raise ValueError("Gaussian divisor and normalization epsilon must be positive.")
        if self.f_lower_hz is not None and self.f_lower_hz <= 0:
            raise ValueError("f_lower_hz must be positive.")
        if self.f_upper_hz is not None and self.f_upper_hz <= 0:
            raise ValueError("f_upper_hz must be positive.")
        if (
            self.f_lower_hz is not None
            and self.f_upper_hz is not None
            and self.f_upper_hz < self.f_lower_hz
        ):
            raise ValueError("f_upper_hz must be greater than or equal to f_lower_hz.")


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Notebook entropy sampler parameters."""

    n_samples: int = 604_800
    n_bins: int = 800
    random_state: int = 679

    def __post_init__(self) -> None:
        if self.n_samples < 1 or self.n_bins < 1:
            raise ValueError("n_samples and n_bins must be positive.")


@dataclass(frozen=True, slots=True)
class STSWConfig:
    """Parameters used in the notebook's STSW call."""

    n_components: int = 128
    n_steps: int = 500
    batch_size: int = 1024
    lr: float = 5e-3
    ntrees: int = 400
    nlines: int = 30
    device: str = "auto"
    dtype: Literal["float32"] = "float32"

    def __post_init__(self) -> None:
        for name in ("n_components", "n_steps", "batch_size", "ntrees", "nlines"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.lr <= 0:
            raise ValueError("lr must be positive.")


@dataclass(frozen=True, slots=True)
class AssignmentConfig:
    """Chunk size and output index dtype for temporal prototype assignment."""

    chunk_size: int = 100_000


@dataclass(frozen=True, slots=True)
class GraphConfig:
    """Prototype MST / pith configuration."""

    edge_weight: str = "geo_distance"


@dataclass(frozen=True, slots=True)
class TangentConfig:
    """Occupancy-weighted tangent PCA settings."""

    n_components: int = 2


@dataclass(frozen=True, slots=True)
class RASConfig:
    """Top-level settings; defaults mirror the supplied notebook."""

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    stsw: STSWConfig = field(default_factory=STSWConfig)
    assignment: AssignmentConfig = field(default_factory=AssignmentConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    tangent: TangentConfig = field(default_factory=TangentConfig)
    random_state: int = 679
