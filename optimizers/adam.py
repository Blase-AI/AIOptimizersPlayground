import logging
from typing import List, Optional, Callable

import numpy as np
from numpy.typing import NDArray

from .base import BaseOptimizer
from .dtime import timed

logger = logging.getLogger(__name__)


class Adam(BaseOptimizer):
    """Adam with optional AMSGrad, regularization, gradient clipping, history."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        amsgrad: bool = False,
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
        """Initialize Adam.

        Args:
            learning_rate: Step size.
            beta1: First moment decay (0 <= beta1 < 1).
            beta2: Second moment decay (0 <= beta2 < 1).
            eps: Numerical stability constant.
            amsgrad: Use max of second moment (AMSGrad).
            clip_norm: Max gradient norm (None = no clip).
            decay_rate: LR decay per step.
            track_history: Store parameter history.
            track_interval: Store every N steps.
            on_step: Callback after each step.
            reg_type: 'none' | 'l1' | 'l2' | 'enet'.
            weight_decay: Regularization strength.
            l1_ratio: L1 fraction for elastic net (0-1).
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
        assert 0 <= beta1 < 1, "beta1 must be in [0, 1)"
        assert 0 <= beta2 < 1, "beta2 must be in [0, 1)"
        assert eps > 0, "eps must be positive"
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.amsgrad = amsgrad
        self.m: Optional[List[NDArray[np.float64]]] = None
        self.v: Optional[List[NDArray[np.float64]]] = None
        self.v_hat_max: Optional[List[NDArray[np.float64]]] = None

    def reset(self):
        """Reset iteration, history, and moment estimates."""
        super().reset()
        self.m = None
        self.v = None
        self.v_hat_max = None

    @timed
    def step(
        self,
        params: List[NDArray[np.float64]],
        grads: List[NDArray[np.float64]]
    ) -> List[NDArray[np.float64]]:
        """Adam step: update params using biased-corrected first and second moments."""
        if self.m is None or self.v is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
            if self.amsgrad:
                self.v_hat_max = [np.zeros_like(p) for p in params]

        lr = self._effective_lr()
        t = self.iteration + 1
        updated_params: List[NDArray[np.float64]] = []

        for i, (p, g) in enumerate(zip(params, grads)):
            g = self._clip_gradient(g)
            m_prev = self.m[i]
            v_prev = self.v[i]
            m_new = self.beta1 * m_prev + (1 - self.beta1) * g
            v_new = self.beta2 * v_prev + (1 - self.beta2) * (g * g)
            self.m[i] = m_new
            self.v[i] = v_new

            if self.amsgrad:
                v_hat_max_prev = self.v_hat_max[i]
                v_hat_max_new = np.maximum(v_hat_max_prev, v_new)
                self.v_hat_max[i] = v_hat_max_new
                denom = v_hat_max_new
            else:
                denom = v_new

            m_hat = m_new / (1 - self.beta1 ** t)
            v_hat = denom / (1 - self.beta2 ** t)

            update = lr * m_hat / (np.sqrt(v_hat) + self.eps)
            new_param = p - update
            updated_params.append(new_param)

            if self.verbose:
                logger.debug(
                    "[Adam] iter %d param %d reg=%s ||grad||=%.4f ||update||=%.4f",
                    t, i, self.reg_type,
                    float(np.linalg.norm(g)), float(np.linalg.norm(update)),
                )

        return updated_params

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "amsgrad": self.amsgrad,
        })
        return cfg

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(')')
        return (
            f"{base}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, "
            f"amsgrad={self.amsgrad})"
        )
