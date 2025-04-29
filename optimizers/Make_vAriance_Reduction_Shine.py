import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Callable
import logging
from .BOptimizer import BaseOptimizer
from .dtime import timed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MARS(BaseOptimizer):
    """
    MARS (Momentum-Averaging Regularization Strategy) optimizer.
    Combines momentum, gradient averaging, configurable regularization,
    gradient clipping, bias correction, and learning rate scheduling.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        avg_beta: float = 0.1,
        reg_type: str = 'none',  
        weight_decay: float = 0.0,
        l1_ratio: float = 0.5,
        clip_norm: Optional[float] = None,
        bias_correction: bool = False,
        lr_scheduler: Optional[Callable[[int], float]] = None,
        track_history: bool = False,
        on_step: Optional[Callable[[List[np.ndarray], List[np.ndarray], List[np.ndarray]], None]] = None,
    ):
        """
        :param learning_rate: initial step size α
        :param momentum: momentum coefficient β for parameter updates
        :param avg_beta: coefficient for exponential averaging of raw gradients
        :param reg_type: type of regularization: 'none' | 'l1' | 'l2' | 'enet'
        :param weight_decay: coefficient for regularization strength
        :param l1_ratio: ratio of L1 in ElasticNet
        :param clip_norm: maximum norm for gradient clipping
        :param bias_correction: whether to apply bias correction to momentum
        :param lr_scheduler: optional function(epoch) → new learning rate
        :param track_history: whether to track parameter history
        :param on_step: hook called after each update (params, grads, updated)
        """
        super().__init__(
            learning_rate=learning_rate,
            track_history=track_history,
            on_step=on_step,
            reg_type=reg_type,
            weight_decay=weight_decay,
            l1_ratio=l1_ratio,
        )
        self.momentum = momentum
        self.avg_beta = avg_beta
        self.clip_norm = clip_norm
        self.bias_correction = bias_correction
        self.lr_scheduler = lr_scheduler
        self.velocities: Optional[List[NDArray[np.float64]]] = None
        self.avg_grads: Optional[List[NDArray[np.float64]]] = None

    def reset(self):
        """Reset optimizer state."""
        super().reset()
        self.velocities = None
        self.avg_grads = None

    @timed
    def step(
        self,
        params: List[NDArray[np.float64]],
        grads: List[NDArray[np.float64]]
    ) -> List[NDArray[np.float64]]:
        step_idx = self.iteration + 1
        if self.lr_scheduler:
            self.learning_rate = self.lr_scheduler(step_idx)

        if self.velocities is None or self.avg_grads is None:
            self.velocities = [np.zeros_like(p) for p in params]
            self.avg_grads = [np.zeros_like(g) for g in grads]

        updated_params: List[NDArray[np.float64]] = []

        for i, (p, g_orig) in enumerate(zip(params, grads)):
            g = g_orig
            if self.clip_norm is not None:
                norm = np.linalg.norm(g)
                if norm > self.clip_norm:
                    g = g * (self.clip_norm / (norm + 1e-6))

            avg_prev = self.avg_grads[i]
            avg_new = (1 - self.avg_beta) * avg_prev + self.avg_beta * g
            self.avg_grads[i] = avg_new

            v_prev = self.velocities[i]
            v_new = self.momentum * v_prev + (1 - self.momentum) * avg_new
            self.velocities[i] = v_new

            v_hat = v_new / (1 - self.momentum**step_idx) if self.bias_correction else v_new

            new_param = p - self.learning_rate * v_hat
            updated_params.append(new_param)

            logger.info(
                f"[MARS] Iter {step_idx} | Param {i} | reg={self.reg_type} | "
                f"||avg_grad||={np.linalg.norm(avg_new):.4f} | "
                f"||vel||={np.linalg.norm(v_hat):.4f}"
            )

        return updated_params

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "momentum": self.momentum,
            "avg_beta": self.avg_beta,
            "clip_norm": self.clip_norm,
            "bias_correction": self.bias_correction,
            "lr_scheduler": self.lr_scheduler.__name__ if self.lr_scheduler else None
        })
        return cfg

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(')')
        return (
            f"{base}, momentum={self.momentum}, avg_beta={self.avg_beta}, "
            f"bias_correction={self.bias_correction}, clip_norm={self.clip_norm})"
        )
