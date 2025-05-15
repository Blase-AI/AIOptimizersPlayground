import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Callable
import logging
from .BOptimizer import BaseOptimizer
from .dtime import timed

logger = logging.getLogger(__name__)


class MARS(BaseOptimizer):
    """
    MARS (Momentum-Averaging Regularization Strategy) optimizer.
    Combines momentum, gradient averaging, configurable regularization,
    gradient clipping, bias correction, and learning rate scheduling.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        avg_beta: float = 0.1,
        reg_type: str = 'none',
        weight_decay: float = 0.0,
        l1_ratio: float = 0.5,
        clip_norm: Optional[float] = None,
        bias_correction: bool = False,
        lr_scheduler: Optional[Callable[[int], float]] = None,
        decay_rate: float = 1.0,
        track_history: bool = False,
        track_interval: int = 1,
        on_step: Optional[Callable[[List[np.ndarray], List[np.ndarray], List[np.ndarray]], None]] = None,
        verbose: bool = False
    ):
        """
        :param learning_rate: начальный шаг обучения
        :param momentum: коэффициент импульса для обновления параметров (0 ≤ momentum < 1)
        :param avg_beta: коэффициент экспоненциального усреднения градиентов (0 ≤ avg_beta < 1)
        :param reg_type: тип регуляризации: 'none' | 'l1' | 'l2' | 'enet'
        :param weight_decay: коэффициент силы регуляризации (L1/L2/Enet)
        :param l1_ratio: доля L1 в ElasticNet (0 ≤ l1_ratio ≤ 1)
        :param clip_norm: максимальная норма градиента (если не None)
        :param bias_correction: применять ли коррекцию смещения для импульса
        :param lr_scheduler: функция (epoch) → новый learning rate (имеет приоритет над decay_rate)
        :param decay_rate: коэффициент экспоненциального затухания learning rate (1.0 = нет затухания)
        :param track_history: сохранять ли историю параметров
        :param track_interval: интервал сохранения истории (каждую N-ю итерацию)
        :param on_step: hook после каждого шага (params, grads, updated)
        :param verbose: выводить ли информацию об итерациях
        :example:
            params = [np.array([1.0]), np.array([2.0])]
            grads = [np.array([0.5]), np.array([1.0])]
            optimizer = MARS(learning_rate=0.01, momentum=0.9, avg_beta=0.1)
            updated = optimizer.step(params, grads)  # Обновленные параметры с учетом усреднения и импульса
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
        assert 0 <= momentum < 1, "momentum must be in [0, 1)"
        assert 0 <= avg_beta < 1, "avg_beta must be in [0, 1)"
        assert clip_norm is None or clip_norm > 0, "clip_norm must be positive or None"
        assert 0 < decay_rate <= 1, "decay_rate must be in (0, 1]"
        self.momentum = momentum
        self.avg_beta = avg_beta
        self.clip_norm = clip_norm
        self.bias_correction = bias_correction
        self.lr_scheduler = lr_scheduler
        self.decay_rate = decay_rate
        self.velocities: Optional[List[NDArray[np.float64]]] = None
        self.avg_grads: Optional[List[NDArray[np.float64]]] = None

    def reset(self):
        """
        Сброс состояния оптимизатора (итерация, история, скорости, усредненные градиенты).
        """
        super().reset()
        self.velocities = None
        self.avg_grads = None

    @timed
    def step(
        self,
        params: List[NDArray[np.float64]],
        grads: List[NDArray[np.float64]]
    ) -> List[NDArray[np.float64]]:
        """
        Шаг MARS: обновление параметров с использованием усреднения градиентов и импульса.
        avg_grad_t = (1 - avg_beta) * avg_grad_{t-1} + avg_beta * g_t
        v_t = momentum * v_{t-1} + (1 - momentum) * avg_grad_t
        v_hat = v_t / (1 - momentum^t) если bias_correction, иначе v_t
        θ = θ - α * v_hat
        :param params: список параметров (np.ndarray), например, веса и смещения модели
        :param grads: список градиентов (np.ndarray), уже включающих регуляризацию
        :return: список обновлённых параметров
        """
        step_idx = self.iteration + 1

        if self.lr_scheduler:
            lr = self.lr_scheduler(step_idx)
        else:
            lr = self.learning_rate * self.decay_rate ** self.iteration

        if self.velocities is None or self.avg_grads is None:
            self.velocities = [np.zeros_like(p) for p in params]
            self.avg_grads = [np.zeros_like(g) for g in grads]

        updated_params: List[NDArray[np.float64]] = []

        for i, (p, g_orig) in enumerate(zip(params, grads)):
            g = g_orig
            if self.clip_norm is not None:
                norm = np.linalg.norm(g)
                if norm > self.clip_norm:
                    g = g * (self.clip_norm / (norm + 1e-6))

            avg_prev = self.avg_grads[i]
            avg_new = (1 - self.avg_beta) * avg_prev + self.avg_beta * g
            self.avg_grads[i] = avg_new

            v_prev = self.velocities[i]
            v_new = self.momentum * v_prev + (1 - self.momentum) * avg_new
            self.velocities[i] = v_new

            v_hat = v_new / (1 - self.momentum ** step_idx) if self.bias_correction else v_new

            new_param = p - lr * v_hat
            updated_params.append(new_param)

            if self.verbose:
                logger.info(
                    f"[MARS] Iter {step_idx} | Param {i} | reg={self.reg_type} | "
                    f"||avg_grad||={np.linalg.norm(avg_new):.4f} | ||vel||={np.linalg.norm(v_hat):.4f}"
                )

        return updated_params

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "momentum": self.momentum,
            "avg_beta": self.avg_beta,
            "clip_norm": self.clip_norm,
            "bias_correction": self.bias_correction,
            "lr_scheduler": self.lr_scheduler.__name__ if self.lr_scheduler else None,
            "decay_rate": self.decay_rate
        })
        return cfg

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(')')
        return (
            f"{base}, momentum={self.momentum}, avg_beta={self.avg_beta}, "
            f"bias_correction={self.bias_correction}, clip_norm={self.clip_norm})"
        )
