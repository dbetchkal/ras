"""Lazy adapter to the external STSW implementation."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .config import STSWConfig
from .exceptions import RASDependencyError


def run_stsw(
    Xt,
    config: STSWConfig,
    *,
    random_state: int = 679,
    stswd_function: Callable | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Learn unit-sphere prototypes with the supplied STSW objective.

    The source's optimization is retained: random unit prototypes, random
    minibatches drawn with replacement, Adam, one `stswd(..., ntrees, nlines)`
    objective per step, and unit renormalization after every update.

    Source correction: the supplied file allocates X0 with `batch_size` rows
    despite taking `n_components`. This implementation uses n_components for
    the vocabulary size and batch_size only for minibatch sampling.
    """
    try:
        import torch
    except ImportError as exc:
        raise RASDependencyError(
            "STSW training requires PyTorch. Install ras-acoustics[stsw]."
        ) from exc

    if stswd_function is None:
        try:
            from methods import stswd
        except ImportError as exc:
            raise RASDependencyError(
                "STSW training requires the external `methods.stswd` module "
                "from the STSW repository."
            ) from exc
        stswd_function = stswd.stswd

    if not torch.is_tensor(Xt):
        Xt = torch.as_tensor(np.array(Xt, dtype=np.float32, copy=True))
    else:
        Xt = Xt.detach().to(dtype=torch.float32)

    if Xt.ndim != 2 or Xt.shape[0] == 0:
        raise ValueError("Xt must have nonempty shape (n_samples, n_features).")

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if config.device == "auto"
        else config.device
    )
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    Xt = Xt.to(device)
    n_samples, n_features = Xt.shape

    generator = torch.Generator(device=device)
    generator.manual_seed(int(random_state))

    X0 = torch.randn(
        (config.batch_size, n_features),
        device=device,
        dtype=Xt.dtype,
        generator=generator,
    )
    X0 = X0 / torch.clamp(
        torch.linalg.vector_norm(X0, dim=1, keepdim=True), min=1e-12
    )
    X0.requires_grad_(True)

    optimizer = torch.optim.Adam([X0], lr=config.lr)
    loss_history: list[float] = []

    for _ in range(config.n_steps):
        idx = torch.randint(
            0, n_samples, (config.batch_size,),
            device=device, generator=generator,
        )
        batch = Xt[idx]

        optimizer.zero_grad(set_to_none=True)
        loss = stswd_function(
            batch,
            X0,
            ntrees=config.ntrees,
            nlines=config.nlines,
            device=device,
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            X0 /= torch.clamp(
                torch.linalg.vector_norm(X0, dim=1, keepdim=True), min=1e-12
            )
        loss_history.append(float(loss.detach().cpu().item()))

    return X0.detach().cpu().numpy(), loss_history
