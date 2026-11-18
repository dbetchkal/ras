"""Input validation for the notebook-derived pipeline."""

from .data import Spectrogram


def validate_spectrogram(spec: Spectrogram) -> float:
    """Validate core input and return its regular sampling interval in seconds."""
    return spec.sampling_interval_seconds()
