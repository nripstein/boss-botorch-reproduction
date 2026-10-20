from typing import Optional, Tuple


import warnings


import numpy as np


import torch


from botorch.fit import fit_gpytorch_mll


from botorch.optim.fit import fit_gpytorch_mll_scipy


from botorch.exceptions.errors import ModelFittingError


from botorch.models import SingleTaskGP


from botorch.models.utils.assorted import InputDataWarning


from gpytorch.kernels import RBFKernel, ScaleKernel


from gpytorch.means import ZeroMean


from gpytorch.mlls import ExactMarginalLogLikelihood


from boss.config import BOConfig


DTYPE = torch.float64


def physical_to_normalized(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    return (x - bounds[0]) / (bounds[1] - bounds[0])


def normalized_to_physical(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    return bounds[0] + x * (bounds[1] - bounds[0])


def _hyperparameters(model: SingleTaskGP) -> Tuple[torch.Tensor, torch.Tensor]:
    lengthscale = model.covar_module.base_kernel.lengthscale.detach().clone()
    outputscale = model.covar_module.outputscale.detach().clone()
    return lengthscale, outputscale


def build_surrogate(
    train_x_normalized: np.ndarray,
    train_y_centered: np.ndarray,
    config: BOConfig,
    previous_model: Optional[SingleTaskGP] = None,
    fit_hyperparameters: bool = False,
    journal_outputscale: Optional[float] = None,
    fit_observation_count: Optional[int] = None,
) -> SingleTaskGP:
    """Build an exact GP, optionally fitting or carrying forward hyperparameters."""
    train_x = torch.as_tensor(train_x_normalized, dtype=DTYPE)
    train_y = torch.as_tensor(train_y_centered, dtype=DTYPE).reshape(-1, 1)
    train_yvar = torch.full_like(train_y, config.noise_variance)

    covar_module = ScaleKernel(RBFKernel(ard_num_dims=train_x.shape[-1]))
    # Centering is explicit in the sampling loop; avoid an additional outcome
    # transform and suppress BoTorch's generic standardization advice.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InputDataWarning)
        model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y,
            train_Yvar=train_yvar,
            mean_module=ZeroMean(),
            covar_module=covar_module,
            outcome_transform=None,
        )

    if previous_model is None:
        model.covar_module.base_kernel.lengthscale = config.initial_lengthscale
        model.covar_module.outputscale = config.initial_outputscale
    else:
        lengthscale, outputscale = _hyperparameters(previous_model)
        model.covar_module.base_kernel.lengthscale = lengthscale
        model.covar_module.outputscale = outputscale

    if config.sampling_profile == "journal":
        if journal_outputscale is None:
            raise ValueError("journal GP requires the carried sample variance")
        model.covar_module.outputscale = torch.tensor(journal_outputscale, dtype=DTYPE)
        model.covar_module.raw_outputscale.requires_grad_(False)
        if fit_hyperparameters:
            _fit_journal_lengthscale(model, config, fit_observation_count or len(train_x))
        model.eval()
        model.likelihood.eval()
        return model

    raise ValueError('Only journal sampling is supported in this submission')

def _fit_journal_lengthscale(model: SingleTaskGP, config: BOConfig, observed: int) -> None:
    """Fit only lengthscale, using bounded multistart BoTorch/SciPy fits."""
    model.train()
    model.likelihood.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    kernel = model.covar_module.base_kernel
    constraint = kernel.raw_lengthscale_constraint
    raw_bounds = tuple(float(constraint.inverse_transform(torch.tensor(v, dtype=DTYPE)))
                       for v in (.01, .99))
    parameter_name = "model.covar_module.base_kernel.raw_lengthscale"
    rng = np.random.default_rng(config.seed + observed)
    starts = np.r_[float(kernel.lengthscale.detach().reshape(())),
                   rng.uniform(.01, .99, config.lengthscale_fit_restarts - 1)]
    best_value, best_lengthscale = -np.inf, None
    for start in starts:
        kernel.lengthscale = torch.tensor(float(start), dtype=DTYPE)
        try:
            outcome = fit_gpytorch_mll_scipy(
                mll, parameters={parameter_name: kernel.raw_lengthscale},
                bounds={parameter_name: raw_bounds}, method="L-BFGS-B",
                options={"maxiter": 1000})
            if outcome.status.name != "SUCCESS":
                continue
            with torch.no_grad():
                value = float(mll(model(model.train_inputs[0]), model.train_targets))
            lengthscale = float(kernel.lengthscale.detach().reshape(()))
            if np.isfinite(value) and .01-1e-12 <= lengthscale <= .99+1e-12 and value > best_value:
                best_value, best_lengthscale = value, lengthscale
        except (RuntimeError, ValueError):
            continue
    if best_lengthscale is None:
        raise ModelFittingError(f"All {len(starts)} journal lengthscale fits failed at {observed} observations")
    kernel.lengthscale = torch.tensor(best_lengthscale, dtype=DTYPE)


def model_hyperparameters(model: SingleTaskGP) -> Tuple[np.ndarray, float]:
    lengthscale, outputscale = _hyperparameters(model)
    return (
        lengthscale.reshape(-1).cpu().numpy(),
        float(outputscale.reshape(()).cpu()),
    )


