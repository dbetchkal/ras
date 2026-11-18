"""Domain-specific exceptions."""


class RASError(Exception):
    """Base class for RAS errors."""


class InputContractError(RASError, ValueError):
    """Input spectrogram or coordinate contract failure."""


class RASDependencyError(RASError, ImportError):
    """Optional external dependency required for a selected analysis stage."""


class GeometryError(RASError, ValueError):
    """Invalid or degenerate geometric input."""


class GraphConstructionError(RASError, ValueError):
    """The prototype distance graph cannot be represented as the requested tree."""
