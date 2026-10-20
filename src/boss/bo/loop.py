from typing import List


import logging


import numpy as np


from botorch.utils.sampling import draw_sobol_samples


import torch


from boss.config import BOConfig


from boss.evaluator import Evaluator


from boss.types import BOResult, TraceRecord


from .acquisition import select_candidate


from .surrogate import (
    build_surrogate,
    model_hyperparameters,
    normalized_to_physical,
    physical_to_normalized,
)


LOGGER = logging.getLogger(__name__)


def _initial_design(config: BOConfig, bounds: np.ndarray) -> np.ndarray:
    if config.dimension == 1:
        return np.linspace(bounds[0, 0], bounds[1, 0], config.num_initial)[:, None]
    unit_bounds = torch.stack(
        [torch.zeros(config.dimension), torch.ones(config.dimension)]
    )
    normalized = draw_sobol_samples(
        unit_bounds, n=config.num_initial, q=1, seed=config.seed
    )[:, 0, :].numpy()
    return normalized_to_physical(normalized, bounds)


def _run_journal(evaluator: Evaluator, config: BOConfig) -> BOResult:
    """Revised journal sampling rules, with BoTorch continuous optimization."""
    bounds = np.asarray(config.bounds, dtype=float)
    evaluator.initialize()
    x_physical = _initial_design(config, bounds)
    evaluations = [evaluator.evaluate(row) for row in x_physical]
    y_absolute = np.array([item.objective for item in evaluations])
    offset = float(y_absolute.mean())
    centered = y_absolute-offset
    normalized = physical_to_normalized(x_physical, bounds)
    trace = [TraceRecord(iteration=i, evaluation=item,
                         centered_objective=float(item.objective-offset), center_offset=offset)
             for i, item in enumerate(evaluations, 1)]
    outputscale = max(float(np.var(y_absolute, ddof=1)), 1e-12)
    model = build_surrogate(normalized, centered, config, fit_hyperparameters=True,
                            journal_outputscale=outputscale, fit_observation_count=len(evaluations))
    while len(evaluations) < config.budget:
        observed = len(evaluations)
        initial_fit = observed == config.num_initial
        fitted = initial_fit or config.should_fit_hyperparameters(observed)
        if not initial_fit:
            if fitted:
                outputscale = max(float(np.var(y_absolute, ddof=1)), 1e-12)
            model = build_surrogate(normalized, centered, config, previous_model=model,
                                    fit_hyperparameters=fitted, journal_outputscale=outputscale,
                                    fit_observation_count=observed)
        lengthscale, _ = model_hyperparameters(model)
        # Journal UCB uses i + num_initial, i.e. the next evaluation number.
        candidate, acquisition, beta, source = select_candidate(model, normalized, observed+1, config)
        physical = normalized_to_physical(candidate, bounds)
        evaluation = evaluator.evaluate(physical)
        evaluations.append(evaluation)
        x_physical = np.vstack([x_physical, physical])
        normalized = np.vstack([normalized, candidate])
        y_absolute = np.append(y_absolute, evaluation.objective)
        offset = float(y_absolute.mean())
        centered = y_absolute-offset
        trace.append(TraceRecord(
            iteration=observed+1, evaluation=evaluation,
            centered_objective=float(evaluation.objective-offset), center_offset=offset,
            beta=beta, acquisition_value=acquisition, lengthscale=lengthscale.tolist(),
            outputscale=outputscale, hyperparameters_fitted=fitted,
            fit_observation_count=observed if fitted else None,
            fit_success=True if fitted else None, candidate_source=source))
        LOGGER.info("completed journal evaluation %d/%d at alpha=%s", observed+1, config.budget, physical.tolist())
    model = build_surrogate(normalized, centered, config, previous_model=model,
                            journal_outputscale=outputscale, fit_hyperparameters=False)
    return BOResult(model=model, train_x_physical=x_physical, train_x_normalized=normalized,
                    train_y_absolute=y_absolute, train_y_centered=centered,
                    center_offset=offset, trace=trace, bounds=bounds)


run_boss = _run_journal
