import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Callable
import logging
from .base import BaseOptimizer
from .dtime import timed

logger = logging.getLogger(__name__)


class RMSProp(BaseOptimizer):
    """RMSProp with optional momentum, regularization, gradient clipping, LR decay, history."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        rho: float = 0.9,
        eps: float = 1e-8,
        momentum: float = 0.0,
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
        """Initialize RMSProp. rho smooths E[g^2]; momentum is optional."""
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
        assert 0 <= rho < 1, "rho must be in [0, 1)"
        assert eps > 0, "eps must be positive"
        assert 0 <= momentum <= 1, "momentum must be in [0, 1]"
        self.rho = rho
        self.eps = eps
        self.momentum = momentum
        self.eg2: Optional[List[NDArray[np.float64]]] = None
        self.velocity: Optional[List[NDArray[np.float64]]] = None

    def reset(self):
        """Reset iteration, history, E[g^2], and velocity."""
        super().reset()
        self.eg2 = None
        self.velocity = None

    @timed
    def step(
        self,
        params: List[NDArray[np.float64]],
        grads: List[NDArray[np.float64]]
    ) -> List[NDArray[np.float64]]:
        """RMSProp step: adaptive scaling by running average of squared gradients."""
        if self.eg2 is None:
            self.eg2 = [np.zeros_like(p) for p in params]
            if self.momentum > 0.0:
                self.velocity = [np.zeros_like(p) for p in params]

        lr = self._effective_lr()
        t = self.iteration + 1
        updated_params: List[NDArray[np.float64]] = []

        for i, (p, g) in enumerate(zip(params, grads)):
            g = self._clip_gradient(g)
            eg2_prev = self.eg2[i]
            eg2_new = self.rho * eg2_prev + (1 - self.rho) * (g * g)
            self.eg2[i] = eg2_new

            step_update = lr * g / (np.sqrt(eg2_new) + self.eps)

            if self.momentum > 0.0:
                v_prev = self.velocity[i]
                v_new = self.momentum * v_prev + step_update
                self.velocity[i] = v_new
                update = v_new
            else:
                update = step_update

            new_param = p - update
            updated_params.append(new_param)

            if self.verbose:
                logger.debug(
                    "[RMSProp] iter %d param %d reg=%s ||grad||=%.4f ||update||=%.4f",
                    t, i, self.reg_type,
                    float(np.linalg.norm(g)), float(np.linalg.norm(update)),
                )

        return updated_params

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "rho": self.rho,
            "eps": self.eps,
            "momentum": self.momentum,
        })
        return cfg

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(')')
        return f"{base}, rho={self.rho}, eps={self.eps}, momentum={self.momentum})"
