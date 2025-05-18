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
        track_interval: int = 1,
        on_step: Optional[Callable[[List[np.ndarray], List[np.ndarray], List[np.ndarray]], None]] = None,
        reg_type: str = 'none',
        weight_decay: float = 0.0,
        l1_ratio: float = 0.5,
        decay_rate: float = 1.0,
        verbose: bool = False
    ):
        """
        :param learning_rate: начальный шаг обучения
        :param track_history: сохранять ли историю обновлений параметров
        :param track_interval: интервал сохранения истории (каждую N-ю итерацию)
        :param on_step: hook после каждого шага (params, grads, updated)
        :param reg_type: тип регуляризации: 'none' | 'l1' | 'l2' | 'enet'
        :param weight_decay: сила регуляризации λ (L2/L1/Enet)
        :param l1_ratio: для Enet — соотношение L1/L2 (0 ≤ l1_ratio ≤ 1)
        :param decay_rate: коэффициент экспоненциального затухания learning rate (1.0 = нет затухания)
        :param verbose: выводить ли информацию об итерациях
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
        assert 0 < decay_rate <= 1, "decay_rate must be in (0, 1]"
        self.decay_rate = decay_rate

    @timed
    def step(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        """
        Шаг пакетного градиентного спуска: θ = θ - α * ∇J(θ).
        :param params: список параметров (np.ndarray), например, веса и смещения модели
        :param grads: список градиентов (np.ndarray), уже включающих регуляризацию
        :return: список обновлённых параметров
        :example:
            params = [np.array([1.0]), np.array([2.0])]
            grads = [np.array([0.5]), np.array([1.0])]
            optimizer = GradientDescent(learning_rate=0.1)
            updated = optimizer.step(params, grads) 
        """
        lr = self.learning_rate * self.decay_rate ** self.iteration
        updated_params = [p - lr * g for p, g in zip(params, grads)]

        if self.on_step:
            self.on_step(params, grads, updated_params)

        return updated_params
