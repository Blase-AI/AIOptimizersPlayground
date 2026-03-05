"""Gradient descent optimizer."""
import numpy as np
from typing import List, Optional, Callable
from .base import BaseOptimizer
from .dtime import timed


class GradientDescent(BaseOptimizer):
    """Batch gradient descent with regularization, history, and hooks."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        track_history: bool = False,
        track_interval: int = 1,
        on_step: Optional[Callable[[List[np.ndarray], List[np.ndarray], List[np.ndarray]], None]] = None,
        reg_type: str = 'none',
        weight_decay: float = 0.0,
        l1_ratio: float = 0.5,
        decay_rate: float = 1.0,
        verbose: bool = False
    ):
        """Initialize gradient descent.

        Args:
            learning_rate: Step size.
            track_history: Whether to store parameter history.
            track_interval: Store history every N steps.
            on_step: Callback (params, grads, updated) after each step.
            reg_type: 'none' | 'l1' | 'l2' | 'enet'.
            weight_decay: Regularization strength.
            l1_ratio: L1 fraction for elastic net (0-1).
            decay_rate: LR decay per step.
            verbose: Log per-step info.
        """
        super().__init__(
            learning_rate=learning_rate,
            track_history=track_history,
            track_interval=track_interval,
            on_step=on_step,
            reg_type=reg_type,
            weight_decay=weight_decay,
            l1_ratio=l1_ratio,
            verbose=verbose,
            decay_rate=decay_rate,
        )

    @timed
    def step(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        """Update parameters: theta = theta - lr * grad."""
        lr = self._effective_lr()
        updated_params = [p - lr * g for p, g in zip(params, grads)]
        return updated_params
