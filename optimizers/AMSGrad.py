import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Callable
import logging
from .base import BaseOptimizer
from .dtime import timed

logger = logging.getLogger(__name__)


class AMSGrad(BaseOptimizer):
    """Adam variant keeping max of second moment. Decoupled weight decay, gradient clipping, LR decay."""

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
        """Initialize AMSGrad. Uses max of second moment for denominator."""
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
        self.v_hat: Optional[List[NDArray[np.float64]]] = None

    def reset(self):
        """Reset iteration, history, and moment estimates."""
        super().reset()
        self.m = None
        self.v = None
        self.v_hat = None

    @timed
    def step(
        self,
        params: List[NDArray[np.float64]],
        grads: List[NDArray[np.float64]]
    ) -> List[NDArray[np.float64]]:
        """AMSGrad step: Adam-like update with max of second moment."""
        if self.m is None or self.v is None or self.v_hat is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
            self.v_hat = [np.zeros_like(p) for p in params]

        lr = self._effective_lr()
        t = self.iteration + 1
        updated_params: List[NDArray[np.float64]] = []

        for i, (p, g) in enumerate(zip(params, grads)):
            g = self._clip_gradient(g)
            m_prev = self.m[i]
            v_prev = self.v[i]
            v_hat_prev = self.v_hat[i]
            m_new = self.beta1 * m_prev + (1 - self.beta1) * g
            v_new = self.beta2 * v_prev + (1 - self.beta2) * (g * g)
            v_hat_new = np.maximum(v_hat_prev, v_new)
            self.m[i] = m_new
            self.v[i] = v_new
            self.v_hat[i] = v_hat_new

            update = lr * m_new / (np.sqrt(v_hat_new) + self.eps)
            decayed = p * (1 - lr * self.weight_decay)
            new_param = decayed - update
            updated_params.append(new_param)

            if self.verbose:
                logger.debug(
                    "[AMSGrad] iter %d param %d ||grad||=%.4f ||update||=%.4f",
                    t, i, float(np.linalg.norm(g)), float(np.linalg.norm(update)),
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
            f"wd={self.weight_decay})"
        )
