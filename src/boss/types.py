from dataclasses import dataclass, field


from typing import Any, Dict, List, Mapping, Optional


import numpy as np


import pandas as pd


@dataclass(frozen=True)
class Evaluation:
    alpha: np.ndarray
    log_marginal: float
    log_prior: float
    objective: float
    elapsed_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        alpha = np.asarray(self.alpha, dtype=float)
        object.__setattr__(self, "alpha", alpha)
        if alpha.ndim != 1 or not np.all(np.isfinite(alpha)):
            raise ValueError("alpha must be a finite one-dimensional vector")
        values = (self.log_marginal, self.log_prior, self.objective)
        if not np.all(np.isfinite(values)):
            raise ValueError("evaluation values must be finite")
        if not np.isclose(self.log_marginal + self.log_prior, self.objective):
            raise ValueError("objective must equal log_marginal + log_prior")


@dataclass
class TraceRecord:
    iteration: int
    evaluation: Evaluation
    centered_objective: float
    beta: Optional[float] = None
    acquisition_value: Optional[float] = None
    lengthscale: Optional[List[float]] = None
    outputscale: Optional[float] = None
    hyperparameters_fitted: bool = False
    candidate_source: str = "initial"
    fit_observation_count: Optional[int] = None
    candidate_grid_index: Optional[int] = None
    lengthscale_grid_likelihoods: Optional[List[float]] = None
    center_offset: Optional[float] = None
    fit_success: Optional[bool] = None


@dataclass
class BOResult:
    model: Any
    train_x_physical: np.ndarray
    train_x_normalized: np.ndarray
    train_y_absolute: np.ndarray
    train_y_centered: np.ndarray
    center_offset: float
    trace: List[TraceRecord]
    bounds: np.ndarray

    def trace_frame(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for record in self.trace:
            row: Dict[str, Any] = {
                "iteration": record.iteration,
                "log_marginal": record.evaluation.log_marginal,
                "log_prior": record.evaluation.log_prior,
                "objective": record.evaluation.objective,
                "centered_objective": record.centered_objective,
                "beta": record.beta,
                "acquisition_value": record.acquisition_value,
                "outputscale": record.outputscale,
                "hyperparameters_fitted": record.hyperparameters_fitted,
                "candidate_source": record.candidate_source,
                "fit_observation_count": record.fit_observation_count,
                "candidate_grid_index": record.candidate_grid_index,
                "lengthscale_grid_likelihoods": record.lengthscale_grid_likelihoods,
                "elapsed_seconds": record.evaluation.elapsed_seconds,
                "center_offset": record.center_offset,
                "fit_success": record.fit_success,
            }
            for j, value in enumerate(record.evaluation.alpha, start=1):
                row[f"alpha_{j}"] = value
            if record.lengthscale is not None:
                for j, value in enumerate(record.lengthscale, start=1):
                    row[f"lengthscale_{j}"] = value
            rows.append(row)
        return pd.DataFrame(rows)





