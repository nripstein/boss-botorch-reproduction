"""Evaluator abstraction used by the generic ask-tell loop."""

from typing import Protocol

import numpy as np

from .types import Evaluation


class Evaluator(Protocol):
    def initialize(self) -> None:
        """Initialize any persistent external state exactly once."""

    def evaluate(self, alpha: np.ndarray) -> Evaluation:
        """Return the absolute log objective components at physical ``alpha``."""

