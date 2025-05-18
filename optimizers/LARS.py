import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Callable
import logging
from .BOptimizer import BaseOptimizer
from .dtime import timed

logger = logging.getLogger(__name__)

class LARS(BaseOptimizer):
    """
    LARS: Layer-wise Adaptive Rate Scaling optimizer.
    Adapts learning rate per layer based on the ratio of parameter and gradient norms,
    with decoupled weight decay and gradient clipping.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        trust_coeff: float = 0.001,
        clip_norm: Optional[float] = None,
        decay_rate: float = 1.0,
        track_history: bool = False,
        track_interval: int = 1,
        on_step: Optional[Callable[[List[np.ndarray], List[np.ndarray], List[np.ndarray]], None]] = None,
        verbose: bool = False
    ):
        """
        :param learning_rate: начальный шаг обучения
        :param momentum: коэффициент импульса (0 ≤ momentum < 1)
        :param weight_decay: коэффициент декуплированной регуляризации весов
        :param trust_coeff: коэффициент доверия для масштабирования learning rate
        :param clip_norm: максимальная норма градиента (если не None)
        :param decay_rate: коэффициент экспоненциального затухания learning rate (1.0 = нет затухания)
        :param track_history: сохранять ли историю параметров
        :param track_interval: интервал сохранения истории (каждую N-ю итерацию)
        :param on_step: hook после каждого шага (params, grads, updated)
        :param verbose: выводить ли информацию об итерациях
        :example:
            params = [np.array([1.0]), np.array([2.0])]
            grads = [np.array([0.5]), np.array([1.0])]
            optimizer = LARS(learning_rate=0.01, momentum=0.9, trust_coeff=0.001)
            updated = optimizer.step(params, grads)  
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
        assert 0 <= momentum < 1, "momentum must be in [0, 1)"
        assert weight_decay >= 0, "weight_decay must be non-negative"
        assert trust_coeff > 0, "trust_coeff must be positive"
        assert clip_norm is None or clip_norm > 0, "clip_norm must be positive or None"
        assert 0 < decay_rate <= 1, "decay_rate must be in (0, 1]"
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.trust_coeff = trust_coeff
        self.clip_norm = clip_norm
        self.decay_rate = decay_rate
        self.velocities: Optional[List[NDArray[np.float64]]] = None

    def reset(self):
        """
        Сброс состояния оптимизатора (итерация, история, скорости).
        """
        super().reset()
        self.velocities = None

    @timed
    def step(
        self,
        params: List[NDArray[np.float64]],
        grads: List[NDArray[np.float64]]
    ) -> List[NDArray[np.float64]]:
        """
        Шаг LARS: обновление параметров с адаптивным масштабированием learning rate для каждого слоя.
        η_i = trust_coeff * ||p_i|| / (||g_i|| + weight_decay * ||p_i||)
        v_t = momentum * v_{t-1} + (η_i * g_t + weight_decay * p_t)
        θ = θ - α * v_t
        :param params: список параметров (np.ndarray), например, веса и смещения модели
        :param grads: список градиентов (np.ndarray), уже включающих регуляризацию
        :return: список обновлённых параметров
        """
        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p in params]

        lr = self.learning_rate * self.decay_rate ** self.iteration
        t = self.iteration + 1
        updated_params: List[NDArray[np.float64]] = []

        for i, (p, g) in enumerate(zip(params, grads)):
            if self.clip_norm is not None:
                norm = np.linalg.norm(g)
                if norm > self.clip_norm:
                    g = g * (self.clip_norm / (norm + 1e-6))

            param_norm = np.linalg.norm(p)
            grad_norm = np.linalg.norm(g)
            if param_norm > 0 and grad_norm > 0:
                local_lr = self.trust_coeff * param_norm / (grad_norm + self.weight_decay * param_norm + 1e-6)
            else:
                local_lr = 1.0

            v_prev = self.velocities[i]
            v_new = self.momentum * v_prev + (local_lr * g + self.weight_decay * p)
            self.velocities[i] = v_new

            update = lr * v_new
            new_param = p - update
            updated_params.append(new_param)

            if self.verbose:
                logger.info(
                    f"[LARS] Iter {t} | Param {i} | reg=none | "
                    f"||grad||={grad_norm:.4f} | local_lr={local_lr:.4f} | "
                    f"||update||={np.linalg.norm(update):.4f}"
                )

        return updated_params

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "trust_coeff": self.trust_coeff,
            "clip_norm": self.clip_norm,
            "decay_rate": self.decay_rate
        })
        return cfg

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(')')
        return (
            f"{base}, momentum={self.momentum}, wd={self.weight_decay}, "
            f"trust_coeff={self.trust_coeff}, clip_norm={self.clip_norm})"
        )
