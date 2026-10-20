"""BOSS UCB construction and continuous candidate optimization."""

import math
from typing import Tuple

import numpy as np
import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples

from boss.config import BOConfig


DTYPE = torch.float64


def beta_schedule(t: int, delta: float) -> float:
    if t < 1:
        raise ValueError("t must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    return 2.0 * math.log((t * t * math.pi * math.pi) / (6.0 * delta))


def _is_duplicate(candidate: np.ndarray, observed: np.ndarray, tolerance: float) -> bool:
    return bool(np.any(np.linalg.norm(observed - candidate[None, :], axis=1) <= tolerance))


def select_candidate(
    model: object,
    observed_normalized: np.ndarray,
    t: int,
    config: BOConfig,
) -> Tuple[np.ndarray, float, float, str]:
    """Maximize time-varying UCB and return a unique normalized candidate."""
    beta = beta_schedule(t, config.delta)
    acquisition = UpperConfidenceBound(model=model, beta=beta, maximize=True)
    bounds = torch.stack(
        [
            torch.zeros(config.dimension, dtype=DTYPE),
            torch.ones(config.dimension, dtype=DTYPE),
        ]
    )

    for retry in range(2):
        torch.manual_seed(config.seed + t + retry)
        options = {"maxiter": config.acquisition_maxiter, "batch_limit": 5}
        if config.sampling_profile == "journal":
            options["method"] = "L-BFGS-B"
        candidate, value = optimize_acqf(
            acq_function=acquisition,
            bounds=bounds,
            q=1,
            num_restarts=config.num_restarts,
            raw_samples=config.raw_samples,
            options=options,
        )
        candidate_np = candidate.detach().cpu().numpy().reshape(-1)
        if not _is_duplicate(candidate_np, observed_normalized, config.duplicate_tolerance):
            source = "optimize_acqf" if retry == 0 else "optimize_acqf_retry"
            return candidate_np, float(value.detach().cpu().reshape(())), beta, source

    samples = draw_sobol_samples(
        bounds=bounds,
        n=config.fallback_samples,
        q=1,
        seed=config.seed + 10000 + t,
    )
    sample_np = samples[:, 0, :].cpu().numpy()
    keep = np.array(
        [
            not _is_duplicate(row, observed_normalized, config.duplicate_tolerance)
            for row in sample_np
        ]
    )
    if not np.any(keep):
        raise RuntimeError("unable to find a nonduplicate acquisition candidate")
    kept_samples = samples[torch.as_tensor(keep), :, :]
    with torch.no_grad():
        values = acquisition(kept_samples)
    best = int(torch.argmax(values).item())
    return (
        kept_samples[best, 0, :].cpu().numpy(),
        float(values[best].cpu()),
        beta,
        "sobol_fallback",
    )
