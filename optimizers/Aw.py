import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Callable
import logging
from .BOptimizer import BaseOptimizer
from .dtime import timed

logging.basicConfig(level=logging.INFO)


class AdamW(BaseOptimizer):
    """
    AdamW: Adam с decoupled weight decay (Loshchilov & Hutter).
    Поддержка gradient clipping и истории.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        clip_norm: Optional[float] = None,
        track_history: bool = False,
        on_step: Optional[Callable[[List[np.ndarray], List[np.ndarray], List[np.ndarray]], None]] = None
    ):
        """
        :param learning_rate: шаг обучения
        :param beta1: сглаживающий коэффициент для первого момента
        :param beta2: сглаживающий коэффициент для второго момента
        :param eps: малое число для численной стабильности
        :param weight_decay: коэффициент decoupled weight decay
        :param clip_norm: порог для gradient clipping по норме
        :param track_history: сохранять историю параметров
        :param on_step: hook после шага (params, grads, updated)
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
        self.eps = eps
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        self.m: Optional[List[NDArray[np.float64]]] = None
        self.v: Optional[List[NDArray[np.float64]]] = None

    def reset(self):
        """
        Сброс состояния оптимизатора.
        """
        super().reset()
        self.m = None
        self.v = None

    @timed
    def step(
        self,
        params: List[NDArray[np.float64]],
        grads: List[NDArray[np.float64]]
    ) -> List[NDArray[np.float64]]:
        if self.m is None or self.v is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        t = self.iteration + 1
        updated_params: List[NDArray[np.float64]] = []

        for i, (p, g) in enumerate(zip(params, grads)):
            if self.clip_norm is not None:
                norm = np.linalg.norm(g)
                if norm > self.clip_norm:
                    g = g * (self.clip_norm / (norm + 1e-6))

            m_prev, v_prev = self.m[i], self.v[i]
            m_new = self.beta1 * m_prev + (1 - self.beta1) * g
            v_new = self.beta2 * v_prev + (1 - self.beta2) * (g * g)
            self.m[i], self.v[i] = m_new, v_new

            m_hat = m_new / (1 - self.beta1 ** t)
            v_hat = v_new / (1 - self.beta2 ** t)

            update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

            decayed_param = p * (1 - self.learning_rate * self.weight_decay)
            new_param = decayed_param - update
            updated_params.append(new_param)

            logging.info(
                f"[AdamW] Iter {t} | Param {i} | ||grad||={np.linalg.norm(g):.4f} | "
                f"||update||={np.linalg.norm(update):.4f} | wd={self.weight_decay}"
            )

        return updated_params

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "clip_norm": self.clip_norm
        })
        return cfg

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(')')
        return (
            f"{base}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, "
            f"weight_decay={self.weight_decay}, clip_norm={self.clip_norm})"
        )
