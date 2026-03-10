import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Callable
import logging
from .base import BaseOptimizer
from .dtime import timed

logger = logging.getLogger(__name__)


class AdamW(BaseOptimizer):
    """Adam with decoupled weight decay (Loshchilov & Hutter). Gradient clipping, LR decay, history."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        clip_norm: Optional[float] = None,
        decay_rate: float = 1.0,
        track_history: bool = False,
        track_interval: int = 1,
        on_step: Optional[Callable[[List[np.ndarray], List[np.ndarray], List[np.ndarray]], None]] = None,
        verbose: bool = False
    ):
        """Initialize AdamW. Uses decoupled weight decay; reg_type in base is set to 'none'."""
        super().__init__(
            learning_rate=learning_rate,
            track_history=track_history,
            track_interval=track_interval,
            on_step=on_step,
            reg_type='none',
            weight_decay=0.0,
            l1_ratio=0.5,
            verbose=verbose,
            clip_norm=clip_norm,
            decay_rate=decay_rate,
        )
        assert 0 <= beta1 < 1, "beta1 must be in [0, 1)"
        assert 0 <= beta2 < 1, "beta2 must be in [0, 1)"
        assert eps > 0, "eps must be positive"
        assert weight_decay >= 0, "weight_decay must be non-negative"
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m: Optional[List[NDArray[np.float64]]] = None
        self.v: Optional[List[NDArray[np.float64]]] = None

    def reset(self):
        """Reset iteration, history, and moment estimates."""
        super().reset()
        self.m = None
        self.v = None

    @timed
    def step(
        self,
        params: List[NDArray[np.float64]],
        grads: List[NDArray[np.float64]]
    ) -> List[NDArray[np.float64]]:
        """AdamW step: Adam update plus decoupled weight decay."""
        if self.m is None or self.v is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        lr = self._effective_lr()
        t = self.iteration + 1
        updated_params: List[NDArray[np.float64]] = []

        for i, (p, g) in enumerate(zip(params, grads)):
            g = self._clip_gradient(g)
            m_prev, v_prev = self.m[i], self.v[i]
            m_new = self.beta1 * m_prev + (1 - self.beta1) * g
            v_new = self.beta2 * v_prev + (1 - self.beta2) * (g * g)
            self.m[i], self.v[i] = m_new, v_new

            m_hat = m_new / (1 - self.beta1 ** t)
            v_hat = v_new / (1 - self.beta2 ** t)

            update = lr * m_hat / (np.sqrt(v_hat) + self.eps)

            decayed_param = p * (1 - lr * self.weight_decay)
            new_param = decayed_param - update
            updated_params.append(new_param)

            if self.verbose:
                logger.debug(
                    "[AdamW] iter %d param %d ||grad||=%.4f ||update||=%.4f wd=%.6f",
                    t, i, float(np.linalg.norm(g)), float(np.linalg.norm(update)),
                    self.weight_decay,
                )

        return updated_params

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
        })
        return cfg

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(')')
        return (
            f"{base}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, "
            f"weight_decay={self.weight_decay})"
        )
