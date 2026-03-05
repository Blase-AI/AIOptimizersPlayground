"""SGD optimizer with optional momentum."""
import numpy as np
from typing import List, Optional, Callable
import logging
from .base import BaseOptimizer
from .dtime import timed


class StochasticGradientDescent(BaseOptimizer):
    """SGD with optional momentum, regularization, gradient clipping, history."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        reg_type: str = 'none',
        weight_decay: float = 0.0,
        l1_ratio: float = 0.5,
        clip_norm: Optional[float] = None,
        decay_rate: float = 1.0,
        track_history: bool = False,
        track_interval: int = 1,
        on_step: Optional[Callable[[List[np.ndarray], List[np.ndarray], List[np.ndarray]], None]] = None,
        verbose: bool = False
    ):
        """Initialize SGD.

        Args:
            learning_rate: Step size.
            momentum: Momentum coefficient (0 = plain SGD).
            reg_type: 'none' | 'l1' | 'l2' | 'enet'.
            weight_decay: Regularization strength.
            l1_ratio: L1 fraction for elastic net (0-1).
            clip_norm: Max gradient norm (None = no clip).
            decay_rate: LR decay per step.
            track_history: Store parameter history.
            track_interval: Store every N steps.
            on_step: Callback after each step.
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
            clip_norm=clip_norm,
            decay_rate=decay_rate,
        )
        assert 0 <= momentum <= 1, "momentum must be in [0, 1]"
        self.momentum = momentum
        self.velocities: Optional[List[np.ndarray]] = None

    def reset(self):
        """Reset iteration, history, and velocities."""
        super().reset()
        self.velocities = None

    @timed
    def step(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        """SGD step with momentum: v = mu*v + lr*g, theta = theta - v."""
        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p in params]

        lr = self._effective_lr()
        updated_params: List[np.ndarray] = []
        for i, (p, g) in enumerate(zip(params, grads)):
            g = self._clip_gradient(g)
            v_prev = self.velocities[i]
            v_new = self.momentum * v_prev + lr * g
            self.velocities[i] = v_new

            new_param = p - v_new
            updated_params.append(new_param)

            if self.verbose:
                logging.info(
                    f"[SGD] Iter {self.iteration+1} | Param {i} | reg={self.reg_type} | "
                    f"||grad||={np.linalg.norm(g):.4f} | ||update||={np.linalg.norm(v_new):.4f}"
                )

        return updated_params

    def get_config(self) -> dict:
        """Return config dict including momentum."""
        cfg = super().get_config()
        cfg.update({"momentum": self.momentum})
        return cfg
