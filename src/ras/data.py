"""Coordinate-aware data structures and frequency selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .exceptions import InputContractError


def _numeric_frequency_label(label: Any) -> float:
    if isinstance(label, pd.Interval):
        value = (float(label.left) + float(label.right)) / 2.0
    elif isinstance(label, str):
        value = float(label.strip().lower().removesuffix("hz").strip())
    else:
        value = float(label)
    if not np.isfinite(value) or value <= 0:
        raise InputContractError(f"Invalid frequency center: {label!r}")
    return value


@dataclass(frozen=True, slots=True)
class FrequencyAxis:
    """Frequency centers and original DataFrame column labels.

    Column order is preserved. Custom labels require explicitly supplied
    `centers_hz` in the same order as the original DataFrame columns.
    """

    labels: pd.Index
    centers_hz: np.ndarray

    def __post_init__(self) -> None:
        centers = np.asarray(self.centers_hz, dtype=np.float64)
        if centers.shape != (len(self.labels),):
            raise InputContractError("centers_hz must match the number of columns.")
        if not np.isfinite(centers).all() or np.any(centers <= 0):
            raise InputContractError("Frequency centers must be finite and positive.")
        if len(np.unique(centers)) != len(centers):
            raise InputContractError("Frequency centers must be unique.")
        object.__setattr__(self, "centers_hz", centers)
        object.__setattr__(self, "labels", pd.Index(self.labels))

    @classmethod
    def from_columns(
        cls,
        columns: pd.Index,
        centers_hz: np.ndarray | pd.Series | None = None,
    ) -> "FrequencyAxis":
        labels = pd.Index(columns)
        if centers_hz is None:
            try:
                centers = np.asarray(
                    [_numeric_frequency_label(x) for x in labels],
                    dtype=np.float64,
                )
            except (TypeError, ValueError) as exc:
                raise InputContractError(
                    "Non-numeric frequency labels require frequency_centers_hz."
                ) from exc
        else:
            if isinstance(centers_hz, pd.Series):
                if centers_hz.index.equals(labels):
                    centers = centers_hz.to_numpy(dtype=np.float64)
                else:
                    centers = centers_hz.reindex(labels).to_numpy(dtype=np.float64)
            else:
                centers = np.asarray(centers_hz, dtype=np.float64)
        return cls(labels=labels, centers_hz=centers)

    def select_positions(
        self,
        lower_hz: float | None = None,
        upper_hz: float | None = None,
    ) -> np.ndarray:
        """Return selected positions with inclusive bounds, preserving column order."""
        if lower_hz is None and upper_hz is None:
            return np.arange(len(self.centers_hz), dtype=int)
        mask = np.ones(len(self.centers_hz), dtype=bool)
        if lower_hz is not None:
            mask &= self.centers_hz >= float(lower_hz)
        if upper_hz is not None:
            mask &= self.centers_hz <= float(upper_hz)
        positions = np.flatnonzero(mask)
        if not len(positions):
            raise InputContractError(
                f"No bands found between {lower_hz} and {upper_hz} Hz."
            )
        return positions

    def subset(self, positions: np.ndarray) -> "FrequencyAxis":
        positions = np.asarray(positions, dtype=int)
        return FrequencyAxis(
            labels=self.labels.take(positions),
            centers_hz=self.centers_hz[positions],
        )

    def __len__(self) -> int:
        return len(self.labels)


@dataclass(frozen=True, slots=True)
class Spectrogram:
    """A (time, frequency) spectrogram with explicit coordinates."""

    values: np.ndarray
    time: pd.DatetimeIndex
    frequency: FrequencyAxis
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.time, pd.DatetimeIndex):
            raise InputContractError("Spectrogram time coordinate must be DatetimeIndex.")
        if len(self.time) < 2:
            raise InputContractError("At least two timestamps are required.")
        if self.time.has_duplicates:
            raise InputContractError("DatetimeIndex contains duplicate timestamps.")
        if not self.time.is_monotonic_increasing:
            raise InputContractError("DatetimeIndex must be monotonically increasing.")
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 2 or values.shape != (len(self.time), len(self.frequency)):
            raise InputContractError("values must have shape (len(time), len(frequency)).")
        if not np.isfinite(values).all():
            raise InputContractError("Spectrogram contains NaN or infinite measurements.")
        object.__setattr__(self, "values", np.ascontiguousarray(values))
        object.__setattr__(self, "attrs", dict(self.attrs))

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        frequency_centers_hz: np.ndarray | pd.Series | None = None,
    ) -> "Spectrogram":
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("S must be a pandas.DataFrame.")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise InputContractError("S.index must be a pandas.DatetimeIndex.")
        if frame.shape[1] == 0:
            raise InputContractError("S must have at least one frequency column.")
        frequency = FrequencyAxis.from_columns(frame.columns, frequency_centers_hz)
        return cls(
            values=frame.to_numpy(dtype=np.float64, copy=False),
            time=frame.index.copy(),
            frequency=frequency,
            attrs=dict(frame.attrs),
        )

    def to_frame(self, *, copy: bool = True) -> pd.DataFrame:
        frame = pd.DataFrame(
            self.values.copy() if copy else self.values,
            index=self.time,
            columns=self.frequency.labels,
            copy=False,
        )
        frame.attrs.update(self.attrs)
        return frame

    def sampling_interval_seconds(self, *, atol_seconds: float = 1e-6) -> float:
        delta = np.diff(self.time.asi8).astype(np.float64) / 1e9
        median = float(np.median(delta))
        # if median <= 0 or not np.allclose(delta, median, rtol=0, atol=atol_seconds):
        #     raise InputContractError(
        #         "Preprocessing requires a regularly sampled DatetimeIndex. "
        #         "Resample explicitly before analysis."
        #     )
        return median

    def select_frequency(
        self,
        lower_hz: float | None,
        upper_hz: float | None,
    ) -> "Spectrogram":
        pos = self.frequency.select_positions(lower_hz, upper_hz)
        return Spectrogram(
            self.values[:, pos],
            self.time,
            self.frequency.subset(pos),
            self.attrs,
        )


def select_frequency_band(
    S: pd.DataFrame,
    f_lower: float | None = None,
    f_upper: float | None = None,
    *,
    frequency_centers_hz: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """Select inclusive frequency-center bounds, preserving input columns."""
    spec = Spectrogram.from_frame(S, frequency_centers_hz=frequency_centers_hz)
    return spec.select_frequency(f_lower, f_upper).to_frame()
