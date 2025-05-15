import numpy as np
from typing import List, Optional, Callable
import logging
from .BOptimizer import BaseOptimizer
from .dtime import timed


class StochasticGradientDescent(BaseOptimizer):
    """
    Стохастический градиентный спуск (SGD) с опциональным импульсом, регуляризацией,
    обрезкой градиентов, таймерами и историей.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        reg_type: str = 'none',
        weight_decay: float = 0.0,
        l1_ratio: float = 0.5,
        clip_norm: Optional[float] = None,
        decay_rate: float = 1.0,
        track_history: bool = False,
        track_interval: int = 1,
        on_step: Optional[Callable[[List[np.ndarray], List[np.ndarray], List[np.ndarray]], None]] = None,
        verbose: bool = False
    ):
        """
        :param learning_rate: начальный шаг обучения
        :param momentum: коэффициент импульса (0.0 = чистый SGD)
        :param reg_type: тип регуляризации: 'none' | 'l1' | 'l2' | 'enet'
        :param weight_decay: сила регуляризации λ (L1/L2/Enet)
        :param l1_ratio: для Enet — доля L1 в elastic net (0 ≤ l1_ratio ≤ 1)
        :param clip_norm: максимальная норма градиента (если не None)
        :param decay_rate: коэффициент экспоненциального затухания learning rate (1.0 = нет затухания)
        :param track_history: сохранять ли историю параметров
        :param track_interval: интервал сохранения истории (каждую N-ю итерацию)
        :param on_step: hook после каждого шага (params, grads, updated)
        :param verbose: выводить ли информацию об итерациях
        :example:
            params = [np.array([1.0]), np.array([2.0])]
            grads = [np.array([0.5]), np.array([1.0])]
            optimizer = StochasticGradientDescent(learning_rate=0.1, momentum=0.9)
            updated = optimizer.step(params, grads)  # Обновленные параметры с учетом импульса
        """
        super().__init__(
            learning_rate=learning_rate,
            track_history=track_history,
            track_interval=track_interval,
            on_step=on_step,
            reg_type=reg_type,
            weight_decay=weight_decay,
            l1_ratio=l1_ratio,
            verbose=verbose
        )
        assert 0 <= momentum <= 1, "momentum must be in [0, 1]"
        assert clip_norm is None or clip_norm > 0, "clip_norm must be positive or None"
        assert 0 < decay_rate <= 1, "decay_rate must be in (0, 1]"
        self.momentum = momentum
        self.clip_norm = clip_norm
        self.decay_rate = decay_rate
        self.velocities: Optional[List[np.ndarray]] = None

    def reset(self):
        """
        Сброс состояния оптимизатора (итерация, история, скорости).
        """
        super().reset()
        self.velocities = None

    @timed
    def step(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        """
        Шаг стохастического градиентного спуска с импульсом:
            v = μ * v + α * ∇J(θ), θ = θ - v
        :param params: список параметров (np.ndarray), например, веса и смещения модели
        :param grads: список градиентов (np.ndarray), уже включающих регуляризацию
        :return: список обновлённых параметров
        """
        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p in params]

        lr = self.learning_rate * self.decay_rate ** self.iteration
        updated_params: List[np.ndarray] = []
        for i, (p, g) in enumerate(zip(params, grads)):
            if self.clip_norm is not None:
                norm = np.linalg.norm(g)
                if norm > self.clip_norm:
                    g = g * (self.clip_norm / (norm + 1e-6))

            v_prev = self.velocities[i]
            v_new = self.momentum * v_prev + lr * g
            self.velocities[i] = v_new

            new_param = p - v_new
            updated_params.append(new_param)

            if self.verbose:
                logging.info(
                    f"[SGD] Iter {self.iteration+1} | Param {i} | reg={self.reg_type} | "
                    f"||grad||={np.linalg.norm(g):.4f} | ||update||={np.linalg.norm(v_new):.4f}"
                )

        return updated_params

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "momentum": self.momentum,
            "clip_norm": self.clip_norm,
            "decay_rate": self.decay_rate
        })
        return cfg
