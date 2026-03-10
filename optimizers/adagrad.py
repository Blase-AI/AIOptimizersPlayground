import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Callable
import logging
from .base import BaseOptimizer
from .dtime import timed

logger = logging.getLogger(__name__)


class Adagrad(BaseOptimizer):
    """Adagrad: per-parameter adaptive learning rate. Regularization, clipping, LR decay, history."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        eps: float = 1e-8,
        clip_norm: Optional[float] = None,
        decay_rate: float = 1.0,
        track_history: bool = False,
        track_interval: int = 1,
        on_step: Optional[Callable[[List[np.ndarray], List[np.ndarray], List[np.ndarray]], None]] = None,
        reg_type: str = 'none',
        weight_decay: float = 0.0,
        l1_ratio: float = 0.5,
        verbose: bool = False
    ):
        """Initialize Adagrad. eps for numerical stability; clip_norm optional."""
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
        assert eps > 0, "eps must be positive"
        self.eps = eps
        self.sum_g2: Optional[List[NDArray[np.float64]]] = None

    def reset(self):
        """Reset iteration, history, and sum of squared gradients."""
        super().reset()
        self.sum_g2 = None

    @timed
    def step(
        self,
        params: List[NDArray[np.float64]],
        grads: List[NDArray[np.float64]]
    ) -> List[NDArray[np.float64]]:
        """Adagrad step: scale by cumulative sum of squared gradients."""
        if self.sum_g2 is None:
            self.sum_g2 = [np.zeros_like(p) for p in params]

        lr = self._effective_lr()
        t = self.iteration + 1
        updated_params: List[NDArray[np.float64]] = []

        for i, (p, g) in enumerate(zip(params, grads)):
            g = self._clip_gradient(g)
            sum_g2_prev = self.sum_g2[i]
            sum_g2_new = sum_g2_prev + (g * g)
            self.sum_g2[i] = sum_g2_new

            update = lr * g / (np.sqrt(sum_g2_new) + self.eps)
            new_param = p - update
            updated_params.append(new_param)

            if self.verbose:
                logger.debug(
                    "[Adagrad] iter %d param %d reg=%s ||grad||=%.4f ||update||=%.4f",
                    t, i, self.reg_type,
                    float(np.linalg.norm(g)), float(np.linalg.norm(update)),
                )

        return updated_params

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({"eps": self.eps})
        return cfg

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(')')
        return f"{base}, eps={self.eps})"
