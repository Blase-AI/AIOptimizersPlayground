import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Callable
import logging
from .BOptimizer import BaseOptimizer
from .dtime import timed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Lion(BaseOptimizer):
    """
    Lion: Efficient sign-momentum optimizer (EvoLved Sign Momentum) with optional bias correction,
    learning rate scheduling, decoupled weight decay, and gradient clipping.
    """

    def __init__(
        self,
        learning_rate: float = 1e-4,
        beta: float = 0.9,
        weight_decay: float = 0.01,
        clip_norm: Optional[float] = None,
        bias_correction: bool = False,
        lr_scheduler: Optional[Callable[[int], float]] = None,
        track_history: bool = False,
        on_step: Optional[Callable[[List[np.ndarray], List[np.ndarray], List[np.ndarray]], None]] = None
    ):
        """
        :param learning_rate: step size α
        :param beta: momentum coefficient β
        :param weight_decay: decoupled weight decay coefficient
        :param clip_norm: threshold for gradient clipping (by norm)
        :param bias_correction: apply bias correction to momentum
        :param lr_scheduler: function(epoch) -> new learning rate
        :param track_history: save history of parameters
        :param on_step: hook called after each update (params, grads, updated)
        """
        super().__init__(
            learning_rate=learning_rate,
            track_history=track_history,
            on_step=on_step,
            reg_type='none',
            weight_decay=0.0,
            l1_ratio=0.5
        )
        self.beta = beta
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        self.bias_correction = bias_correction
        self.lr_scheduler = lr_scheduler
        self.v: Optional[List[NDArray[np.float64]]] = None

    def reset(self):
        """Reset optimizer state."""
        super().reset()
        self.v = None

    @timed
    def step(
        self,
        params: List[NDArray[np.float64]],
        grads: List[NDArray[np.float64]]
    ) -> List[NDArray[np.float64]]:
        t = self.iteration + 1

        if self.lr_scheduler:
            self.learning_rate = self.lr_scheduler(t)

        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]

        updated_params: List[NDArray[np.float64]] = []
        for i, (p, g) in enumerate(zip(params, grads)):
            if self.clip_norm is not None:
                norm = np.linalg.norm(g)
                if norm > self.clip_norm:
                    g = g * (self.clip_norm / (norm + 1e-6))

            v_prev = self.v[i]
            v_new = self.beta * v_prev + (1 - self.beta) * g
            self.v[i] = v_new

            v_hat = v_new / (1 - self.beta**t) if self.bias_correction else v_new

            decayed = p * (1 - self.learning_rate * self.weight_decay)

            update = self.learning_rate * np.sign(v_hat)
            new_param = decayed - update
            updated_params.append(new_param)

            logger.info(
                f"[Lion] Iter {t} | Param {i} | ||grad||={np.linalg.norm(g):.4f} | "
                f"sign-step sum={np.sum(np.sign(v_hat)):.4f}"
            )

        return updated_params

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            'beta': self.beta,
            'weight_decay': self.weight_decay,
            'clip_norm': self.clip_norm,
            'bias_correction': self.bias_correction,
            'lr_scheduler': self.lr_scheduler.__name__ if self.lr_scheduler else None
        })
        return cfg

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(')')
        return (
            f"{base}, beta={self.beta}, wd={self.weight_decay}, "
            f"clip_norm={self.clip_norm}, bias_correction={self.bias_correction})"
        )
