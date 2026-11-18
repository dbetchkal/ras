"""Lazy Matplotlib visualizations for spectrograms and prototype trees."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Spectrogram


def event_plotter(
    event: pd.DataFrame | Spectrogram,
    ax,
    vmin: float = -10,
    vmax: float = 90,
    xtickscale: str = "diel",
    cmap: str = "plasma",
):
    """Port the attached source event_plotter; returns the image artist."""
    import matplotlib.dates as mdates

    frame = event.to_frame(copy=False) if isinstance(event, Spectrogram) else event
    x_lims = mdates.date2num(frame.index)
    image = ax.imshow(
        frame.to_numpy().T,
        extent=[x_lims[0], x_lims[-1], 0, frame.shape[1]],
        cmap=cmap,
        origin="lower",
        interpolation="none",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    centers = (
        event.frequency.centers_hz
        if isinstance(event, Spectrogram)
        else pd.to_numeric(pd.Index(frame.columns)).to_numpy()
    )
    ax.set_yticks(np.arange(frame.shape[1])[::8])
    ax.set_yticklabels(centers[::8], fontsize=8)
    ax.xaxis_date()
    if xtickscale == "diel":
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d\n%H:%M"))
    elif xtickscale == "annual":
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d\n%Y"))
    return image


def plot_mst(G, pos=None, *, ax=None, node_color="k", edge_color="silver"):
    """Draw the MST without requiring a particular layout policy."""
    import matplotlib.pyplot as plt
    import networkx as nx

    if ax is None:
        _, ax = plt.subplots()
    if pos is None:
        pos = nx.kamada_kawai_layout(G, dim=2, weight="geo_distance")
    nx.draw(
        G, pos, node_size=8, node_color=node_color,
        edge_color=edge_color, width=0.4, ax=ax,
    )
    return ax, pos
