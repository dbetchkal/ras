"""Public orchestrator for the notebook-derived RAS analysis."""

from __future__ import annotations

from typing import Callable

import logging
import numpy as np
import pandas as pd

from .config import RASConfig
from .data import Spectrogram
from .diagnostics import occupancy_diagnostics
from .graph import (
    aggregate_diam_occupancy,
    build_geodesic_mst,
    family_nodes,
    ordered_path_nodes,
    diam_geometry,
    tree_diameter_graph,
)
from .preprocessing import estimate_exposure
from .states import (
    assign_states,
    furthest_state,
    spherical_medoid,
)
from .references import (
    build_reference_states,
    diam_family_percentile_spectra,
    state_percentile_spectra,
)
from .result import RASResult
from .stsw import run_stsw
from .tangent import analyze_tangent_space
from .temporal import (
    hypersphere_velocity_weighted,
    state_temporal_dynamics,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ras.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class RAS:
    """Analyze a DataFrame spectrogram using the supplied notebook workflow.

    `stsw_backend` can replace the optional external STSW optimizer during
    tests or controlled experiments. It receives `(sample_values, config,
    random_state)` and must return `(X_N, loss_history)`.
    """

    def __init__(
        self,
        config: RASConfig | None = None,
        *,
        stsw_backend: Callable | None = None,
    ) -> None:
        self.config = config or RASConfig()
        self.stsw_backend = stsw_backend
        self.result_: RASResult | None = None

    def analyze(
        self,
        S: pd.DataFrame,
        *,
        frequency_centers_hz: np.ndarray | pd.Series | None = None,
    ) -> RASResult:
        spec = Spectrogram.from_frame(
            S, frequency_centers_hz=frequency_centers_hz
        )
        dt_seconds = spec.sampling_interval_seconds()

        exposure = estimate_exposure(spec, self.config.preprocessing)
        M_df = exposure.M.to_frame(copy=False)
        logger.info("Hypersphere M has been estimated.")

        from .sampling import stratified_entropy_sample
        sampled = stratified_entropy_sample(
            M_df,
            N=self.config.sampling.n_samples,
            n_bins=self.config.sampling.n_bins,
            random_state=self.config.sampling.random_state,
        )
        if sampled.empty:
            raise ValueError("Entropy-stratified sampling returned no observations.")
        
        logger.info("Sampled hyperspherical points.")

        if self.stsw_backend is None:
            X_N, loss_history = run_stsw(
                sampled.to_numpy(dtype=np.float32),
                self.config.stsw,
                random_state=self.config.random_state,
            )
        else:
            X_N, loss_history = self.stsw_backend(
                sampled.to_numpy(dtype=np.float32),
                self.config.stsw,
                self.config.random_state,
            )

        logger.info("Representative acoustic states X_N have been learned.")

        X_N = np.asarray(X_N, dtype=np.float64)
        if X_N.ndim != 2 or X_N.shape[1] != M_df.shape[1]:
            raise ValueError("STSW backend must return X_N with shape (K, selected_B).")
        norms = np.linalg.norm(X_N, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise ValueError("STSW backend returned a zero state.")
        X_N = X_N / norms

        assignment = assign_states(
            M_df.to_numpy(),
            X_N,
            chunk_size=self.config.assignment.chunk_size,
        )
        n_states = len(X_N)
        mean_activity = np.bincount(assignment, 
                                    minlength=n_states).astype(np.float64)
        if mean_activity.sum() == 0:
            raise ValueError("No observations were assigned to states.")
        weights = mean_activity / mean_activity.sum()

        logger.info("A representative state has been assigned to each frame in M")

        x_star, medoid_idx = spherical_medoid(X_N, weights)
        ecc_idx = furthest_state(X_N[medoid_idx], X_N)

        logger.info("The medoid and it's eccentricity have been calculated.")

        S_band = spec.to_frame(copy=False).iloc[
            :, spec.frequency.select_positions(
                self.config.preprocessing.f_lower_hz,
                self.config.preprocessing.f_upper_hz,
            )
        ]
        references = build_reference_states(
            spec.to_frame(copy=False),
            S_band,
            M_df.to_numpy(),
            X_N,
            assignment,
            medoid_idx,
            ecc_idx,
        )

        tangent = analyze_tangent_space(
            X_N,
            x_star,
            weights,
            n_components=min(
                self.config.tangent.n_components,
                max(1, X_N.shape[1] - 1),
            ),
        )

        G_MST, D_geo = build_geodesic_mst(X_N)
        diam = tree_diameter_graph(
            G_MST, weight=self.config.graph.edge_weight
        )
        diam_distance, attachment, family = diam_geometry(
            G_MST,
            diam,
            weight=self.config.graph.edge_weight,
        )
        families = family_nodes(G_MST, family, attachment)

        diam_family_by_observation = np.asarray(
            [attachment[int(state)] for state in assignment],
            dtype=np.int32,
        )
        state_dynamics = state_temporal_dynamics(
            assignment,
            n_states,
            time_step_seconds=dt_seconds,
        )
        diam_nodes, diam_occupancy = aggregate_diam_occupancy(
            G_MST,
            diam,
            state_dynamics["occupancy"].to_numpy(),
            diam_attachment=attachment,
            weight=self.config.graph.edge_weight,
        )

        logger.info("The Minimum Spanning Tree Graph has been created and relevant structure analyzed.")

        diagnostics = occupancy_diagnostics(
            S_band,
            assignment,
            n_states,
        )
        diagnostics["spectral_bias"] = np.sum(
            X_N - X_N[medoid_idx], axis=1
        )
        diagnostics["spectral_bias_order"] = np.argsort(
            diagnostics["spectral_bias"]
        )
        diagnostics["n_states"] = n_states
        diagnostics["n_diam_nodes"] = len(diam)
        diagnostics["n_families"] = len(families)
        diagnostics["n_unassigned_states"] = len(
            set(G_MST.nodes) - set(attachment)
        )
        diagnostics["diam_compression_fraction"] = (
            1.0 - len(diam) / len(spec.time)
        )
        diagnostics["family_sizes"] = {
            int(node): int(len(nodes)) for node, nodes in families.items()
        }
        diam_diameter_order = np.asarray(ordered_path_nodes(diam), dtype=int)
        diam_family_spectra = diam_family_percentile_spectra(
            S_band,
            diam_family_by_observation,
            n_families=n_states,
            percentile=50.0,
        )
        state_median_spectra = state_percentile_spectra(
            S_band, assignment, n_states, percentile=50.0
        )
        state_first_percentile_spectra = state_percentile_spectra(
            S_band, assignment, n_states, percentile=1.0
        )
        v_H, theta_temporal = hypersphere_velocity_weighted(M_df)

        result = RASResult(
            S=spec,
            exposure=exposure,
            X_N=X_N,
            loss_history=np.asarray(loss_history, dtype=np.float64),
            state_assignment=assignment,
            mean_activity=mean_activity,
            weights=weights,
            x_star=x_star,
            medoid_idx=medoid_idx,
            ecc_idx=ecc_idx,
            references=references,
            G_MST=G_MST,
            D_geo=D_geo,
            diam=diam,
            diam_distance=diam_distance,
            diam_attachment=attachment,
            diam_family_by_state=family,
            families=families,
            diam_family_by_observation=diam_family_by_observation,
            state_dynamics=state_dynamics,
            diam_nodes=diam_nodes,
            diam_occupancy=diam_occupancy,
            diam_diameter_order=diam_diameter_order,
            diam_family_percentile_spectra=diam_family_spectra,
            state_median_spectra=state_median_spectra,
            state_first_percentile_spectra=state_first_percentile_spectra,
            v_H=v_H,
            theta_temporal=theta_temporal,
            tangent=tangent,
            diagnostics=diagnostics,
            config=self.config,
            metadata={
                "time_step_seconds": dt_seconds,
                "sampling_target": self.config.sampling.n_samples,
                "sampling_realized": len(sampled),
                "sampling_bins": self.config.sampling.n_bins,
                "state_count": n_states,
                "source_pipeline": "S -> S_O -> S_E -> M -> X_N -> q(t) -> G_MST -> Diameter",
            },
        )
        self.result_ = result
        return result

    def fit_transform(self, S, *, frequency_centers_hz=None) -> RASResult:
        return self.analyze(S, frequency_centers_hz=frequency_centers_hz)

    @property
    def S_O_(self):
        return self._require_result().S_O

    @property
    def S_E_(self):
        return self._require_result().S_E

    @property
    def M_(self):
        return self._require_result().M

    @property
    def X_N_(self):
        return self._require_result().X_N

    @property
    def S_Pi_(self):
        return self._require_result().S_Pi

    def _require_result(self) -> RASResult:
        if self.result_ is None:
            raise AttributeError("Call analyze(S) before accessing fitted attributes.")
        return self.result_
