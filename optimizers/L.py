import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Callable
import logging
from .BOptimizer import BaseOptimizer
from .dtime import timed

logger = logging.getLogger(__name__)


class Lion(BaseOptimizer):
    """
    Lion: Efficient sign-momentum optimizer (EvoLved Sign Momentum) with optional bias correction,
    learning rate scheduling, decoupled weight decay, and gradient clipping.
    """

    def __init__(
        self,
        learning_rate: float = 1e-4,
        beta: float = 0.9,
        weight_decay: float = 0.01,
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
        :param beta: коэффициент импульса (0 ≤ beta < 1)
        :param weight_decay: коэффициент декуплированной регуляризации весов
        :param clip_norm: максимальная норма градиента (если не None)
        :param bias_correction: применять ли коррекцию смещения для импульса
        :param lr_scheduler: функция (epoch) -> новый learning rate (имеет приоритет над decay_rate)
        :param decay_rate: коэффициент экспоненциального затухания learning rate (1.0 = нет затухания)
        :param track_history: сохранять ли историю параметров
        :param track_interval: интервал сохранения истории (каждую N-ю итерацию)
        :param on_step: hook после каждого шага (params, grads, updated)
        :param verbose: выводить ли информацию об итерациях
        :example:
            params = [np.array([1.0]), np.array([2.0])]
            grads = [np.array([0.5]), np.array([1.0])]
            optimizer = Lion(learning_rate=1e-4, beta=0.9, weight_decay=0.01)
            updated = optimizer.step(params, grads)  # Обновленные параметры с учетом знакового импульса
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
        assert 0 <= beta < 1, "beta must be in [0, 1)"
        assert weight_decay >= 0, "weight_decay must be non-negative"
        assert clip_norm is None or clip_norm > 0, "clip_norm must be positive or None"
        assert 0 < decay_rate <= 1, "decay_rate must be in (0, 1]"
        self.beta = beta
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        self.bias_correction = bias_correction
        self.lr_scheduler = lr_scheduler
        self.decay_rate = decay_rate
        self.v: Optional[List[NDArray[np.float64]]] = None

    def reset(self):
        """
        Сброс состояния оптимизатора (итерация, история, импульс).
        """
        super().reset()
        self.v = None

    @timed
    def step(
        self,
        params: List[NDArray[np.float64]],
        grads: List[NDArray[np.float64]]
    ) -> List[NDArray[np.float64]]:
        """
        Шаг Lion: обновление параметров с использованием знакового импульса.
        v_t = β * v_{t-1} + (1 - β) * g_t
        v_hat = v_t / (1 - β^t) если bias_correction, иначе v_t
        θ = θ * (1 - α * λ) - α * sign(v_hat)
        :param params: список параметров (np.ndarray), например, веса и смещения модели
        :param grads: список градиентов (np.ndarray), уже включающих регуляризацию
        :return: список обновлённых параметров
        """
        t = self.iteration + 1

        # Применяем lr_scheduler или экспоненциальное затухание
        if self.lr_scheduler:
            lr = self.lr_scheduler(t)
        else:
            lr = self.learning_rate * self.decay_rate ** self.iteration

        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]

        updated_params: List[NDArray[np.float64]] = []
        for i, (p, g) in enumerate(zip(params, grads)):
            if self.clip_norm is not None:
                norm = np.linalg.norm(g)
                if norm > self.clip_norm:
                    g = g * (self.clip_norm / (norm + 1e-6))

            v_prev = self.v[i]
            v_new = self.beta * v_prev + (1 - self.beta) * g
            self.v[i] = v_new

            v_hat = v_new / (1 - self.beta ** t) if self.bias_correction else v_new

            decayed = p * (1 - lr * self.weight_decay)
            update = lr * np.sign(v_hat)
            new_param = decayed - update
            updated_params.append(new_param)

            if self.verbose:
                logger.info(
                    f"[Lion] Iter {t} | Param {i} | reg=none | "
                    f"||grad||={np.linalg.norm(g):.4f} | sign-step sum={np.sum(np.sign(v_hat)):.4f}"
                )

        return updated_params

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            'beta': self.beta,
            'weight_decay': self.weight_decay,
            'clip_norm': self.clip_norm,
            'bias_correction': self.bias_correction,
            'lr_scheduler': self.lr_scheduler.__name__ if self.lr_scheduler else None,
            'decay_rate': self.decay_rate
        })
        return cfg

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(')')
        return (
            f"{base}, beta={self.beta}, wd={self.weight_decay}, "
            f"clip_norm={self.clip_norm}, bias_correction={self.bias_correction})"
        )
