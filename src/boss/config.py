from dataclasses import dataclass


from typing import Literal, Tuple


@dataclass(frozen=True)
class BOConfig:
    bounds: Tuple[Tuple[float, ...], Tuple[float, ...]]
    budget: int = 30
    num_initial: int = 5
    delta: float = 0.01
    noise_variance: float = 1e-6
    initial_lengthscale: float = 0.1
    initial_outputscale: float = 1000.0
    hyperparameter_update_every: int = 10
    hyperparameter_update_schedule: Literal["total_observations", "bo_iterations"] = "total_observations"
    raw_samples: int = 512
    num_restarts: int = 20
    fallback_samples: int = 4096
    duplicate_tolerance: float = 1e-6
    seed: int = 123
    # ``released_r_parity`` is retained as an input-compatible alias.
    mode: Literal["general_botorch", "released_r_live", "released_artifact_replay", "released_r_parity"] = "general_botorch"
    grid_points: int = 1000
    sampling_profile: Literal["existing", "journal"] = "existing"
    lengthscale_fit_restarts: int = 50
    acquisition_maxiter: int = 300

    def __post_init__(self) -> None:
        lower, upper = self.bounds
        if not lower or len(lower) != len(upper):
            raise ValueError("bounds must contain equally sized lower and upper vectors")
        if any(lo >= hi for lo, hi in zip(lower, upper)):
            raise ValueError("each lower bound must be strictly less than its upper bound")
        if self.num_initial < 1 or self.budget < self.num_initial:
            raise ValueError("budget must be at least num_initial >= 1")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must lie in (0, 1)")
        if self.noise_variance <= 0:
            raise ValueError("noise_variance must be positive")
        if self.acquisition_maxiter < 1:
            raise ValueError("acquisition_maxiter must be positive")
        if self.mode not in {"general_botorch", "released_r_live", "released_artifact_replay", "released_r_parity"}:
            raise ValueError("unknown BO mode")
        if self.mode in {"released_r_live", "released_artifact_replay", "released_r_parity"} and self.noise_variance != 1e-8:
            raise ValueError("released-R modes require noise_variance=1e-8")
        if self.grid_points < self.num_initial:
            raise ValueError("grid_points must be at least num_initial")
        if self.hyperparameter_update_schedule not in {"total_observations", "bo_iterations"}:
            raise ValueError("unknown hyperparameter update schedule")
        if self.sampling_profile not in {"existing", "journal"}:
            raise ValueError("unknown sampling profile")
        if self.sampling_profile == "journal":
            if self.mode != "general_botorch" or self.dimension != 1:
                raise ValueError("journal sampling requires one-dimensional general_botorch mode")
            if self.num_initial < 2 or self.lengthscale_fit_restarts < 1:
                raise ValueError("journal sampling requires at least two initials and one fitting start")
            if not .01 <= self.initial_lengthscale <= .99:
                raise ValueError("journal initial lengthscale must be in [0.01, 0.99]")

    @classmethod
    def journal(cls, bounds: Tuple[Tuple[float, ...], Tuple[float, ...]], **kwargs) -> "BOConfig":
        """Journal sampling rules with BoTorch optimization; not literal R parity."""
        settings = dict(sampling_profile="journal", noise_variance=1e-6,
                        hyperparameter_update_every=5,
                        hyperparameter_update_schedule="bo_iterations", num_restarts=50)
        settings.update(kwargs)
        return cls(bounds=bounds, **settings)

    @property
    def dimension(self) -> int:
        return len(self.bounds[0])

    @property
    def canonical_mode(self) -> str:
        return "released_r_live" if self.mode == "released_r_parity" else self.mode

    def should_fit_hyperparameters(self, observed: int) -> bool:
        """Whether to fit before selecting the next candidate.

        ``bo_iterations`` matches released examples whose counter starts after
        their initial design; ``total_observations`` preserves the framework's
        established behaviour.
        """
        if self.hyperparameter_update_schedule == "total_observations":
            return observed % self.hyperparameter_update_every == 0
        bo_iteration = observed - self.num_initial + 1
        return bo_iteration > 0 and bo_iteration % self.hyperparameter_update_every == 0


