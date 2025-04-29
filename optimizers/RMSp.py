import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Callable
import logging
from .BOptimizer import BaseOptimizer
from .dtime import timed

logging.basicConfig(level=logging.INFO)


class RMSProp(BaseOptimizer):
    """
    Оптимизатор RMSProp с поддержкой momentum, регуляризации, gradient clipping и историей.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        rho: float = 0.9,
        eps: float = 1e-8,
        momentum: float = 0.0,
        clip_norm: Optional[float] = None,
        track_history: bool = False,
        on_step: Optional[Callable[[List[np.ndarray], List[np.ndarray], List[np.ndarray]], None]] = None,
        reg_type: str = 'none',
        weight_decay: float = 0.0,
        l1_ratio: float = 0.5,
    ):
        """
        :param learning_rate: шаг обучения
        :param rho: коэффициент экспонента для сглаживания E[g^2] (обычно ~0.9)
        :param eps: малое число для стабилизации вычислений
        :param momentum: коэффициент momentum (0.0 = без)
        :param clip_norm: порог для gradient clipping
        :param track_history: сохранять историю параметров
        :param on_step: hook после шага (params, grads, updated)
        :param reg_type: тип регуляризации: 'none' | 'l1' | 'l2' | 'enet'
        :param weight_decay: сила регуляризации λ
        :param l1_ratio: для ElasticNet — соотношение L1/L2
        """
        super().__init__(
            learning_rate=learning_rate,
            track_history=track_history,
            on_step=on_step,
            reg_type=reg_type,
            weight_decay=weight_decay,
            l1_ratio=l1_ratio
        )
        self.rho = rho
        self.eps = eps
        self.momentum = momentum
        self.clip_norm = clip_norm
        self.eg2: Optional[List[NDArray[np.float64]]] = None
        self.velocity: Optional[List[NDArray[np.float64]]] = None

    def reset(self):
        """
        Сброс состояния RMSProp (Eg2, velocity).
        """
        super().reset()
        self.eg2 = None
        self.velocity = None

    @timed
    def step(
        self,
        params: List[NDArray[np.float64]],
        grads: List[NDArray[np.float64]]
    ) -> List[NDArray[np.float64]]:
        if self.eg2 is None:
            self.eg2 = [np.zeros_like(p) for p in params]
            if self.momentum > 0.0:
                self.velocity = [np.zeros_like(p) for p in params]

        t = self.iteration + 1
        updated_params: List[NDArray[np.float64]] = []

        for i, (p, g) in enumerate(zip(params, grads)):
            if self.clip_norm is not None:
                norm = np.linalg.norm(g)
                if norm > self.clip_norm:
                    g = g * (self.clip_norm / (norm + 1e-6))

            eg2_prev = self.eg2[i]
            eg2_new = self.rho * eg2_prev + (1 - self.rho) * (g * g)
            self.eg2[i] = eg2_new

            step_update = self.learning_rate * g / (np.sqrt(eg2_new) + self.eps)

            if self.momentum > 0.0:
                v_prev = self.velocity[i]  
                v_new = self.momentum * v_prev + step_update
                self.velocity[i] = v_new  
                update = v_new
            else:
                update = step_update

            new_param = p - update
            updated_params.append(new_param)

            logging.info(
                f"[RMSProp] Iter {t} | Param {i} | ||grad||={np.linalg.norm(g):.4f} | "
                f"||update||={np.linalg.norm(update):.4f}"
            )

        return updated_params

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "rho": self.rho,
            "eps": self.eps,
            "momentum": self.momentum,
            "clip_norm": self.clip_norm
        })
        return cfg

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(')')
        return (
            f"{base}, rho={self.rho}, eps={self.eps}, momentum={self.momentum}, "
            f"clip_norm={self.clip_norm})"
        )
