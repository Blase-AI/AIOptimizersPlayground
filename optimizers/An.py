import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Callable
import logging
from .BOptimizer import BaseOptimizer
from .dtime import timed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Adan(BaseOptimizer):
    """
    Adan: Adaptive Nesterov Momentum optimizer (An All-Round Universal Optimizer).
    Combines adaptive moments with Nesterov momentum on gradient differences.
    """

    def __init__(
        self,
        learning_rate: float = 0.002,
        beta1: float = 0.9,
        beta2: float = 0.999,
        beta3: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        clip_norm: Optional[float] = None,
        track_history: bool = False,
        on_step: Optional[Callable[[List[np.ndarray], List[np.ndarray], List[np.ndarray]], None]] = None
    ):
        """
        :param learning_rate: step size α
        :param beta1: smoothing coefficient for first moment
        :param beta2: smoothing coefficient for second moment
        :param beta3: smoothing coefficient for gradient difference
        :param eps: numerical stability term
        :param weight_decay: decoupled weight decay coefficient
        :param clip_norm: threshold for gradient clipping by norm
        :param track_history: whether to track parameter history
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
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.eps = eps
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        self.m: Optional[List[NDArray[np.float64]]] = None
        self.v: Optional[List[NDArray[np.float64]]] = None
        self.g_prev: Optional[List[NDArray[np.float64]]] = None

    def reset(self):
        super().reset()
        self.m = None
        self.v = None
        self.g_prev = None

    @timed
    def step(
        self,
        params: List[NDArray[np.float64]],
        grads: List[NDArray[np.float64]]
    ) -> List[NDArray[np.float64]]:
        t = self.iteration + 1

        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
            self.g_prev = [np.zeros_like(p) for p in params]

        updated_params: List[NDArray[np.float64]] = []

        for i, (p, g) in enumerate(zip(params, grads)):
            if self.clip_norm is not None:
                norm = np.linalg.norm(g)
                if norm > self.clip_norm:
                    g = g * (self.clip_norm / (norm + 1e-6))

            u = g + self.beta3 * (g - self.g_prev[i])

            m_prev = self.m[i]; v_prev = self.v[i]
            m_new = self.beta1 * m_prev + (1 - self.beta1) * g
            v_new = self.beta2 * v_prev + (1 - self.beta2) * (u * u)
            self.m[i] = m_new; self.v[i] = v_new

            m_hat = m_new / (1 - self.beta1 ** t)
            v_hat = v_new / (1 - self.beta2 ** t)

            update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

            decayed = p * (1 - self.learning_rate * self.weight_decay)
            new_param = decayed - update
            updated_params.append(new_param)

            logger.info(
                f"[Adan] Iter {t} | Param {i} | ||grad||={np.linalg.norm(g):.4f} | "
                f"||update||={np.linalg.norm(update):.4f}"
            )

            # save prev gradient
            self.g_prev[i] = g

        return updated_params

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            'beta1': self.beta1,
            'beta2': self.beta2,
            'beta3': self.beta3,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'clip_norm': self.clip_norm
        })
        return cfg

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(')')
        return (
            f"{base}, beta1={self.beta1}, beta2={self.beta2}, beta3={self.beta3}, "
            f"eps={self.eps}, wd={self.weight_decay}, clip_norm={self.clip_norm})"
        )