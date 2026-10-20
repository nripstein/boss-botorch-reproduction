"""Persistent rpy2 session for the change-point evaluator and inference stage."""

from pathlib import Path
import os
import json
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from boss.types import Evaluation


class RBridgeError(RuntimeError):
    """An R evaluator or conversion failure with boundary context."""

class PersistentRSession:
    """Small reusable rpy2 boundary for experiment-owned R contracts.

    Concrete adapters own their initialization arguments and R function names;
    this class centralizes script sourcing and pandas conversion only.
    """

    def __init__(self, scripts: Sequence[Path]) -> None:
        self.scripts = tuple((Path(path).resolve() for path in scripts))
        self._ro: Optional[Any] = None

    def _source_scripts(self) -> Any:
        for path in self.scripts:
            if not path.exists():
                raise RBridgeError(f'required R bridge path does not exist: {path}')
        try:
            import rpy2.robjects as ro
            self._ro = ro
            for path in self.scripts:
                ro.r['source'](str(path))
            return ro
        except Exception as exc:
            raise RBridgeError(f'failed to source R bridge scripts: {exc}') from exc

class RChangePointEvaluator(PersistentRSession):

    def __init__(self, data_path: Path, evaluator_script: Path, bounds: Sequence[float]=(0.1, 9.9), prior_bounds: Sequence[float]=(0.0, 10.0), basis_k: int=30, inner_aghq_k: int=3, seed: int=123, profile=None) -> None:
        super().__init__((evaluator_script,))
        self.data_path = Path(data_path).resolve()
        self.evaluator_script = Path(evaluator_script).resolve()
        
        self.bounds = np.asarray(bounds, dtype=float)
        self.prior_bounds = np.asarray(prior_bounds, dtype=float)
        self.basis_k = int(basis_k)
        self.inner_aghq_k = int(inner_aghq_k)
        self.seed = int(seed)
        self.profile = profile
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        if sys.platform == 'darwin':
            try:
                r_home = subprocess.run(['R', 'RHOME'], check=True, capture_output=True, text=True).stdout.strip()
            except Exception as exc:
                raise RBridgeError(f'could not determine R_HOME: {exc}') from exc
            r_library = str(Path(r_home) / 'lib')
            dyld_paths = os.environ.get('DYLD_LIBRARY_PATH', '').split(':')
            if r_library not in dyld_paths:
                raise RBridgeError('macOS must start Python with R_HOME/lib first in DYLD_LIBRARY_PATH so PyTorch and R/data.table share one OpenMP runtime; use scripts/boss-python')
        try:
            ro = self._source_scripts()
            initialize = ro.globalenv['boss_initialize_change_point']
            initialize(str(self.data_path), ro.FloatVector(self.bounds), ro.FloatVector(self.prior_bounds), self.basis_k, self.inner_aghq_k, self.seed, False)
        except Exception as exc:
            raise RBridgeError(f'failed to initialize persistent R session: {exc}') from exc
        self._initialized = True

    @property
    def ro(self) -> Any:
        if not self._initialized or self._ro is None:
            raise RBridgeError('initialize() must be called before accessing R')
        return self._ro

    def evaluate(self, alpha: np.ndarray) -> Evaluation:
        self.initialize()
        alpha = np.asarray(alpha, dtype=float)
        if alpha.shape != (1,) or not np.all(np.isfinite(alpha)):
            raise ValueError('change-point alpha must have shape (1,) and be finite')
        started = time.perf_counter()
        try:
            raw = self.ro.globalenv['boss_evaluate_change_point'](self.ro.FloatVector(alpha))
            values = dict(zip(list(raw.names), np.asarray(raw, dtype=float)))
        except Exception as exc:
            raise RBridgeError(f'R evaluation failed at alpha={alpha.tolist()}: {exc}') from exc
        return Evaluation(alpha=alpha, log_marginal=float(values['log_marginal']), log_prior=float(values['log_prior']), objective=float(values['objective']), elapsed_seconds=time.perf_counter() - started)

class RSeasonalityEvaluator(PersistentRSession):
    """Persistent evaluator for the paper's one-dimensional seasonal period model."""

    def __init__(self, data_path: Optional[Path], evaluator_script: Path, bounds: Sequence[float]=(0.5, 4.5), seed: int=123) -> None:
        super().__init__((evaluator_script,))
        self.data_path = Path(data_path).resolve() if data_path is not None else None
        self.evaluator_script = Path(evaluator_script).resolve()
        self.bounds = np.asarray(bounds, dtype=float)
        self.seed = int(seed)
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        if sys.platform == 'darwin':
            try:
                r_home = subprocess.run(['R', 'RHOME'], check=True, capture_output=True, text=True).stdout.strip()
            except Exception as exc:
                raise RBridgeError(f'could not determine R_HOME: {exc}') from exc
            r_library = str(Path(r_home) / 'lib')
            if r_library not in os.environ.get('DYLD_LIBRARY_PATH', '').split(':'):
                raise RBridgeError('macOS must start Python with R_HOME/lib first in DYLD_LIBRARY_PATH; use scripts/boss-python')
        try:
            ro = self._source_scripts()
            ro.globalenv['boss_initialize_seasonality'](str(self.data_path) if self.data_path is not None else ro.NULL, ro.FloatVector(self.bounds), self.seed)
        except Exception as exc:
            raise RBridgeError(f'failed to initialize seasonality R session: {exc}') from exc
        self._initialized = True

    @property
    def ro(self) -> Any:
        if not self._initialized or self._ro is None:
            raise RBridgeError('initialize() must be called before accessing R')
        return self._ro

    def evaluate(self, alpha: np.ndarray) -> Evaluation:
        self.initialize()
        alpha = np.asarray(alpha, dtype=float)
        if alpha.shape != (1,) or not np.all(np.isfinite(alpha)):
            raise ValueError('seasonality alpha must have shape (1,) and be finite')
        started = time.perf_counter()
        try:
            raw = self.ro.globalenv['boss_evaluate_seasonality'](self.ro.FloatVector(alpha))
            values = dict(zip(list(raw.names), np.asarray(raw, dtype=float)))
        except Exception as exc:
            raise RBridgeError(f'R seasonality evaluation failed at alpha={alpha.tolist()}: {exc}') from exc
        return Evaluation(alpha=alpha, log_marginal=float(values['log_marginal']), log_prior=float(values['log_prior']), objective=float(values['objective']), elapsed_seconds=time.perf_counter() - started)

