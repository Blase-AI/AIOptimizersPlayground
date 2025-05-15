import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Callable
import logging
from .BOptimizer import BaseOptimizer
from .dtime import timed

logger = logging.getLogger(__name__)


class AdamW(BaseOptimizer):
    """
    AdamW: Adam с декуплированной регуляризацией весов (Loshchilov & Hutter).
    Поддержка gradient clipping, затухания learning rate и истории.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        clip_norm: Optional[float] = None,
        decay_rate: float = 1.0,
        track_history: bool = False,
        track_interval: int = 1,
        on_step: Optional[Callable[[List[np.ndarray], List[np.ndarray], List[np.ndarray]], None]] = None,
        verbose: bool = False
    ):
        """
        :param learning_rate: начальный шаг обучения
        :param beta1: сглаживающий коэффициент для первого момента (0 ≤ beta1 < 1)
        :param beta2: сглаживающий коэффициент для второго момента (0 ≤ beta2 < 1)
        :param eps: малое число для численной стабильности
        :param weight_decay: коэффициент декуплированной регуляризации весов
        :param clip_norm: максимальная норма градиента (если не None)
        :param decay_rate: коэффициент экспоненциального затухания learning rate (1.0 = нет затухания)
        :param track_history: сохранять ли историю параметров
        :param track_interval: интервал сохранения истории (каждую N-ю итерацию)
        :param on_step: hook после каждого шага (params, grads, updated)
        :param verbose: выводить ли информацию об итерациях
        :example:
            params = [np.array([1.0]), np.array([2.0])]
            grads = [np.array([0.5]), np.array([1.0])]
            optimizer = AdamW(learning_rate=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01)
            updated = optimizer.step(params, grads)  # Обновленные параметры с учетом моментов и регуляризации
        """
        super().__init__(
            learning_rate=learning_rate,
            track_history=track_history,
            track_interval=track_interval,
            on_step=on_step,
            reg_type='none',
            weight_decay=0.0,
            l1_ratio=0.5,
            verbose=verbose
        )
        assert 0 <= beta1 < 1, "beta1 must be in [0, 1)"
        assert 0 <= beta2 < 1, "beta2 must be in [0, 1)"
        assert eps > 0, "eps must be positive"
        assert weight_decay >= 0, "weight_decay must be non-negative"
        assert clip_norm is None or clip_norm > 0, "clip_norm must be positive or None"
        assert 0 < decay_rate <= 1, "decay_rate must be in (0, 1]"
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        self.decay_rate = decay_rate
        self.m: Optional[List[NDArray[np.float64]]] = None
        self.v: Optional[List[NDArray[np.float64]]] = None

    def reset(self):
        """
        Сброс состояния оптимизатора (итерация, история, моменты).
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
        """
        Шаг AdamW: обновление параметров с использованием адаптивных моментов и декуплированной регуляризации.
        m_t = β1 * m_{t-1} + (1 - β1) * g_t
        v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
        m_hat = m_t / (1 - β1^t)
        v_hat = v_t / (1 - β2^t)
        θ = θ * (1 - α * λ) - α * m_hat / (√v_hat + ε)
        :param params: список параметров (np.ndarray), например, веса и смещения модели
        :param grads: список градиентов (np.ndarray), уже включающих регуляризацию
        :return: список обновлённых параметров
        """
        if self.m is None or self.v is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        lr = self.learning_rate * self.decay_rate ** self.iteration
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

            update = lr * m_hat / (np.sqrt(v_hat) + self.eps)

            decayed_param = p * (1 - lr * self.weight_decay)
            new_param = decayed_param - update
            updated_params.append(new_param)

            if self.verbose:
                logger.info(
                    f"[AdamW] Iter {t} | Param {i} | reg=none | "
                    f"||grad||={np.linalg.norm(g):.4f} | ||update||={np.linalg.norm(update):.4f} | wd={self.weight_decay}"
                )

        return updated_params

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "clip_norm": self.clip_norm,
            "decay_rate": self.decay_rate
        })
        return cfg

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(')')
        return (
            f"{base}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, "
            f"weight_decay={self.weight_decay}, clip_norm={self.clip_norm})"
        )
