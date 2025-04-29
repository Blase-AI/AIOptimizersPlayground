import numpy as np
from typing import List, Optional, Callable
import logging
from .BOptimizer import BaseOptimizer
from .dtime import timed

logging.basicConfig(level=logging.INFO)


class StochasticGradientDescent(BaseOptimizer):
    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        reg_type: str = 'none',
        weight_decay: float = 0.0,
        l1_ratio: float = 0.5,
        clip_norm: Optional[float] = None,
        track_history: bool = False,
        on_step: Optional[Callable[[List[np.ndarray], List[np.ndarray], List[np.ndarray]], None]] = None,
    ):
        """
        SGD с опциональным momentum, выбором регуляризации (none, l1, l2, enet), gradient clipping и историей.

        :param learning_rate: шаг обучения
        :param momentum: коэффициент импульса (0.0 = чистый SGD)
        :param reg_type: тип регуляризации: 'none' | 'l1' | 'l2' | 'enet'
        :param weight_decay: сила регуляризации λ (L1/L2/Enet)
        :param l1_ratio: для Enet — доля L1 в elastic net (0 ≤ l1_ratio ≤ 1)
        :param clip_norm: если не None, обрезаем градиент по норме
        :param track_history: сохранять ли историю параметров
        :param on_step: hook после каждого шага (params, grads, updated)
        """
        super().__init__(
            learning_rate=learning_rate,
            track_history=track_history,
            on_step=on_step,
            reg_type=reg_type,
            weight_decay=weight_decay,
            l1_ratio=l1_ratio
        )
        self.momentum = momentum
        self.clip_norm = clip_norm
        self.velocities: Optional[List[np.ndarray]] = None

    @timed
    def step(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:

        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p in params]

        updated_params: List[np.ndarray] = []
        for i, (p, g) in enumerate(zip(params, grads)):
            if self.clip_norm is not None:
                norm = np.linalg.norm(g)
                if norm > self.clip_norm:
                    g = g * (self.clip_norm / (norm + 1e-6))

            v_prev = self.velocities[i]
            v_new = self.momentum * v_prev + self.learning_rate * g
            self.velocities[i] = v_new

            new_param = p - v_new
            updated_params.append(new_param)

            logging.info(
                f"[SGD] Iter {self.iteration+1} | Param {i} | reg={self.reg_type} | "
                f"||grad||={np.linalg.norm(g):.4f} | ||update||={np.linalg.norm(v_new):.4f}"
            )

        return updated_params

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "momentum": self.momentum,
            "clip_norm": self.clip_norm
        })
        return cfg
