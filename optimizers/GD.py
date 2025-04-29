import numpy as np
from typing import List, Optional, Callable
from .BOptimizer import BaseOptimizer
from .dtime import timed


class GradientDescent(BaseOptimizer):
    """
    Классический пакетный градиентный спуск (Batch GD) с поддержкой регуляризации,
    таймерами и историей.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        track_history: bool = False,
        on_step: Optional[Callable[[List[np.ndarray], List[np.ndarray], List[np.ndarray]], None]] = None,
        reg_type: str = 'none',
        weight_decay: float = 0.0,
        l1_ratio: float = 0.5,
    ):
        """
        :param learning_rate: шаг обучения
        :param track_history: сохранять ли историю обновлений параметров
        :param on_step: hook после каждого шага (params, grads, updated)
        :param reg_type: тип регуляризации: 'none' | 'l1' | 'l2' | 'enet'
        :param weight_decay: сила регуляризации λ (L2/L1/Enet)
        :param l1_ratio: для Enet — соотношение L1/L2 (0 ≤ l1_ratio ≤ 1)
        """
        super().__init__(
            learning_rate=learning_rate,
            track_history=track_history,
            on_step=on_step,
            reg_type=reg_type,
            weight_decay=weight_decay,
            l1_ratio=l1_ratio
        )

    @timed
    def step(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        """
        Шаг пакетного градиентного спуска:
            θ = θ - α * ∇J(θ)
        :param params: список параметров (np.ndarray)
        :param grads: список градиентов (np.ndarray) уже с учётом регуляризации
        :return: список обновлённых параметров
        """
        updated_params: List[np.ndarray] = [
            p - self.learning_rate * g for p, g in zip(params, grads)
        ]
        return updated_params
