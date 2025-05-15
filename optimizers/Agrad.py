import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Callable
import logging
from .BOptimizer import BaseOptimizer
from .dtime import timed

logger = logging.getLogger(__name__)


class Adagrad(BaseOptimizer):
    """
    Adagrad: Adaptive Gradient Algorithm с поддержкой регуляризации, gradient clipping,
    затухания learning rate и историей.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        eps: float = 1e-8,
        clip_norm: Optional[float] = None,
        decay_rate: float = 1.0,
        track_history: bool = False,
        track_interval: int = 1,
        on_step: Optional[Callable[[List[np.ndarray], List[np.ndarray], List[np.ndarray]], None]] = None,
        reg_type: str = 'none',
        weight_decay: float = 0.0,
        l1_ratio: float = 0.5,
        verbose: bool = False
    ):
        """
        :param learning_rate: начальный шаг обучения
        :param eps: малое число для численной стабильности
        :param clip_norm: максимальная норма градиента (если не None)
        :param decay_rate: коэффициент экспоненциального затухания learning rate (1.0 = нет затухания)
        :param track_history: сохранять ли историю параметров
        :param track_interval: интервал сохранения истории (каждую N-ю итерацию)
        :param on_step: hook после каждого шага (params, grads, updated)
        :param reg_type: тип регуляризации: 'none' | 'l1' | 'l2' | 'enet'
        :param weight_decay: сила регуляризации λ (L1/L2/Enet)
        :param l1_ratio: для ElasticNet — соотношение L1/L2 (0 ≤ l1_ratio ≤ 1)
        :param verbose: выводить ли информацию об итерациях
        :example:
            params = [np.array([1.0]), np.array([2.0])]
            grads = [np.array([0.5]), np.array([1.0])]
            optimizer = Adagrad(learning_rate=0.01, eps=1e-8)
            updated = optimizer.step(params, grads)  # Обновленные параметры с адаптивным масштабированием
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
        assert eps > 0, "eps must be positive"
        assert clip_norm is None or clip_norm > 0, "clip_norm must be positive or None"
        assert 0 < decay_rate <= 1, "decay_rate must be in (0, 1]"
        self.eps = eps
        self.clip_norm = clip_norm
        self.decay_rate = decay_rate
        self.sum_g2: Optional[List[NDArray[np.float64]]] = None

    def reset(self):
        """
        Сброс состояния оптимизатора (итерация, история, сумма квадратов градиентов).
        """
        super().reset()
        self.sum_g2 = None

    @timed
    def step(
        self,
        params: List[NDArray[np.float64]],
        grads: List[NDArray[np.float64]]
    ) -> List[NDArray[np.float64]]:
        """
        Шаг Adagrad: обновление параметров с адаптивным масштабированием градиентов.
        sum_g^2_t = sum_g^2_{t-1} + g_t^2
        θ = θ - α * g_t / (√sum_g^2_t + ε)
        :param params: список параметров (np.ndarray), например, веса и смещения модели
        :param grads: список градиентов (np.ndarray), уже включающих регуляризацию
        :return: список обновлённых параметров
        """
        if self.sum_g2 is None:
            self.sum_g2 = [np.zeros_like(p) for p in params]

        lr = self.learning_rate * self.decay_rate ** self.iteration
        t = self.iteration + 1
        updated_params: List[NDArray[np.float64]] = []

        for i, (p, g) in enumerate(zip(params, grads)):
            if self.clip_norm is not None:
                norm = np.linalg.norm(g)
                if norm > self.clip_norm:
                    g = g * (self.clip_norm / (norm + 1e-6))

            sum_g2_prev = self.sum_g2[i]
            sum_g2_new = sum_g2_prev + (g * g)
            self.sum_g2[i] = sum_g2_new

            update = lr * g / (np.sqrt(sum_g2_new) + self.eps)
            new_param = p - update
            updated_params.append(new_param)

            if self.verbose:
                logger.info(
                    f"[Adagrad] Iter {t} | Param {i} | reg={self.reg_type} | "
                    f"||grad||={np.linalg.norm(g):.4f} | ||update||={np.linalg.norm(update):.4f}"
                )

        return updated_params

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "eps": self.eps,
            "clip_norm": self.clip_norm,
            "decay_rate": self.decay_rate
        })
        return cfg

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(')')
        return (
            f"{base}, eps={self.eps}, clip_norm={self.clip_norm})"
        )