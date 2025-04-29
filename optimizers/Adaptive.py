import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Callable
import logging
from .BOptimizer import BaseOptimizer
from .dtime import timed

logging.basicConfig(level=logging.INFO)


class Adam(BaseOptimizer):
    """
    Адаптивный оптимизатор Adam с поддержкой AMSGrad, регуляризацией, gradient clipping и историей.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        amsgrad: bool = False,
        clip_norm: Optional[float] = None,
        track_history: bool = False,
        on_step: Optional[Callable[[List[np.ndarray], List[np.ndarray], List[np.ndarray]], None]] = None,
        reg_type: str = 'none',
        weight_decay: float = 0.0,
        l1_ratio: float = 0.5,
    ):
        """
        :param learning_rate: шаг обучения
        :param beta1: сглаживающий коэффициент для первого момента
        :param beta2: сглаживающий коэффициент для второго момента
        :param eps: шум для численной стабильности
        :param amsgrad: использовать ли AMSGrad (max-вариант второго момента)
        :param clip_norm: порог для gradient clipping по норме
        :param track_history: сохранять историю параметров
        :param on_step: hook после шага (params, grads, updated)
        :param reg_type: тип регуляризации: 'none' | 'l1' | 'l2' | 'enet'
        :param weight_decay: сила регуляризации λ
        :param l1_ratio: для Enet — соотношение L1/L2 (0 ≤ l1_ratio ≤ 1)
        """
        super().__init__(
            learning_rate=learning_rate,
            track_history=track_history,
            on_step=on_step,
            reg_type=reg_type,
            weight_decay=weight_decay,
            l1_ratio=l1_ratio,
        )
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.amsgrad = amsgrad
        self.clip_norm = clip_norm
        self.m: Optional[List[NDArray[np.float64]]] = None
        self.v: Optional[List[NDArray[np.float64]]] = None
        self.v_hat_max: Optional[List[NDArray[np.float64]]] = None

    def reset(self):
        """
        Сброс состояния оптимизатора.
        """
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

        if self.m is None or self.v is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
            if self.amsgrad:
                self.v_hat_max = [np.zeros_like(p) for p in params]

        t = self.iteration + 1
        updated_params: List[NDArray[np.float64]] = []

        for i, (p, g) in enumerate(zip(params, grads)):
            if self.clip_norm is not None:
                norm = np.linalg.norm(g)
                if norm > self.clip_norm:
                    g = g * (self.clip_norm / (norm + 1e-6))

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

            update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
            new_param = p - update
            updated_params.append(new_param)

            logging.info(
                f"[Adam] Iter {t} | Param {i} | ||grad||={np.linalg.norm(g):.4f} | "
                f"||update||={np.linalg.norm(update):.4f}"
            )

        return updated_params

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "amsgrad": self.amsgrad,
            "clip_norm": self.clip_norm
        })
        return cfg

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(')')
        return (
            f"{base}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, "
            f"amsgrad={self.amsgrad})"
        )
