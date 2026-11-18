"""Pure SRCID interval/audibility helpers used by the notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd


def binary_audibility_time_series(src, resolution: str = "1s") -> pd.Series:
    """Expand SRCID-like start-index + duration objects to a binary time series."""
    if not hasattr(src, "index") or not hasattr(src, "len"):
        raise TypeError("src must expose an index and len duration attribute.")
    interval_ends = src.index + src.len
    start_time = interval_ends.index.min()
    end_time = interval_ends.max()
    time_index = pd.date_range(start=start_time, end=end_time, freq=resolution)
    binary = pd.Series(0, index=time_index, name="Audible", dtype=np.int8)
    for start, end in zip(interval_ends.index, interval_ends.values):
        binary.loc[start:end] = 1
    return binary


def join_src_to_nvspl(nv: pd.DataFrame, src) -> pd.DataFrame:
    """Outer-join the SRCID audibility intervals to an NVSPL-like DataFrame."""
    audible = binary_audibility_time_series(src, resolution="1s")
    joined = nv.join(audible, how="outer")
    joined["Audible"] = joined["Audible"].fillna(0).astype(np.int8)
    return joined


def audibility_sir(family: pd.DataFrame, overall_ta: float) -> float:
    """Observed/expected audibility ratio for one state or pith family."""
    audible = family["Audible"].dropna()
    n = len(audible)
    observed = float(audible.sum())
    expected = n * float(overall_ta)
    return observed / expected if expected > 0 else float("nan")
