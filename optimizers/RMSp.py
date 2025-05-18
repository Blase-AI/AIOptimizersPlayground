import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Callable
import logging
from .BOptimizer import BaseOptimizer
from .dtime import timed

logger = logging.getLogger(__name__)


class RMSProp(BaseOptimizer):
    """
    Оптимизатор RMSProp с поддержкой momentum, регуляризации, gradient clipping, затухания learning rate и историей.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        rho: float = 0.9,
        eps: float = 1e-8,
        momentum: float = 0.0,
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
        :param rho: коэффициент экспоненциального сглаживания для E[g^2] (0 ≤ rho < 1)
        :param eps: малое число для численной стабильности
        :param momentum: коэффициент импульса (0 ≤ momentum ≤ 1, 0.0 = без импульса)
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
            optimizer = RMSProp(learning_rate=0.001, rho=0.9, momentum=0.9)
            updated = optimizer.step(params, grads) 
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
        assert 0 <= rho < 1, "rho must be in [0, 1)"
        assert eps > 0, "eps must be positive"
        assert 0 <= momentum <= 1, "momentum must be in [0, 1]"
        assert clip_norm is None or clip_norm > 0, "clip_norm must be positive or None"
        assert 0 < decay_rate <= 1, "decay_rate must be in (0, 1]"
        self.rho = rho
        self.eps = eps
        self.momentum = momentum
        self.clip_norm = clip_norm
        self.decay_rate = decay_rate
        self.eg2: Optional[List[NDArray[np.float64]]] = None
        self.velocity: Optional[List[NDArray[np.float64]]] = None

    def reset(self):
        """
        Сброс состояния оптимизатора (итерация, история, E[g^2], velocity).
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
        """
        Шаг RMSProp: обновление параметров с использованием адаптивного масштабирования градиентов.
        E[g^2]_t = ρ * E[g^2]_{t-1} + (1 - ρ) * g_t^2
        step = α * g_t / (√E[g^2]_t + ε)
        v_t = μ * v_{t-1} + step (если momentum > 0)
        θ = θ - v_t (или step, если momentum = 0)
        :param params: список параметров (np.ndarray), например, веса и смещения модели
        :param grads: список градиентов (np.ndarray), уже включающих регуляризацию
        :return: список обновлённых параметров
        """
        if self.eg2 is None:
            self.eg2 = [np.zeros_like(p) for p in params]
            if self.momentum > 0.0:
                self.velocity = [np.zeros_like(p) for p in params]

        lr = self.learning_rate * self.decay_rate ** self.iteration
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

            step_update = lr * g / (np.sqrt(eg2_new) + self.eps)

            if self.momentum > 0.0:
                v_prev = self.velocity[i]
                v_new = self.momentum * v_prev + step_update
                self.velocity[i] = v_new
                update = v_new
            else:
                update = step_update

            new_param = p - update
            updated_params.append(new_param)

            if self.verbose:
                logger.info(
                    f"[RMSProp] Iter {t} | Param {i} | reg={self.reg_type} | "
                    f"||grad||={np.linalg.norm(g):.4f} | ||update||={np.linalg.norm(update):.4f}"
                )

        return updated_params

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "rho": self.rho,
            "eps": self.eps,
            "momentum": self.momentum,
            "clip_norm": self.clip_norm,
            "decay_rate": self.decay_rate
        })
        return cfg

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(')')
        return (
            f"{base}, rho={self.rho}, eps={self.eps}, momentum={self.momentum}, "
            f"clip_norm={self.clip_norm})"
        )
