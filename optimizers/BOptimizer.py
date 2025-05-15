from abc import ABC, abstractmethod
from typing import List, Callable, Optional
import numpy as np
import copy


class BaseOptimizer(ABC):
    def __init__(
        self,
        learning_rate: float = 0.01,
        track_history: bool = False,
        track_interval: int = 1,
        on_step: Optional[Callable[[List[np.ndarray], List[np.ndarray], List[np.ndarray]], None]] = None,
        reg_type: str = 'none',
        weight_decay: float = 0.0,
        l1_ratio: float = 0.5,
        verbose: bool = False
    ):
        """
        Базовый класс для оптимизаторов.

        :param learning_rate: начальный шаг обучения
        :param track_history: сохранять ли историю обновлений параметров
        :param track_interval: интервал сохранения истории (каждую N-ю итерацию)
        :param on_step: функция-хук, вызываемая после каждого шага
        :param reg_type: тип регуляризации: 'none' | 'l1' | 'l2' | 'enet'
        :param weight_decay: сила регуляризации λ (L1/L2/ENet)
        :param l1_ratio: для ENet — доля L1 в elastic net (0 ≤ l1_ratio ≤ 1)
        :param verbose: выводить ли информацию об итерациях
        """
        assert reg_type in {'none', 'l1', 'l2', 'enet'}, \
            "reg_type must be one of 'none','l1','l2','enet'"
        assert learning_rate > 0, "learning_rate must be positive"
        assert weight_decay >= 0, "weight_decay must be non-negative"
        assert 0 <= l1_ratio <= 1, "l1_ratio must be in [0, 1]"
        assert track_interval >= 1, "track_interval must be at least 1"

        self.learning_rate = learning_rate
        self.iteration = 0
        self.track_history = track_history
        self.track_interval = track_interval
        self.history: List[List[np.ndarray]] = []
        self.on_step = on_step
        self.reg_type = reg_type
        self.weight_decay = weight_decay
        self.l1_ratio = l1_ratio
        self.verbose = verbose

    @abstractmethod
    def step(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        """
        Обновляет параметры модели.
        :param params: список параметров (np.ndarray)
        :param grads: список градиентов (np.ndarray), уже включающих регуляризацию
        :return: новый список параметров
        """
        pass

    def apply_regularization(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        """
        Применяет выбранную регуляризацию к градиентам.
        """
        reg_grads: List[np.ndarray] = []
        for p, g in zip(params, grads):
            if self.reg_type == 'l2':
                reg_grads.append(g + self.weight_decay * p)
            elif self.reg_type == 'l1':
                reg_grads.append(g + self.weight_decay * np.sign(p))
            elif self.reg_type == 'enet':
                reg_grads.append(
                    g + self.weight_decay * (
                        self.l1_ratio * np.sign(p) + (1.0 - self.l1_ratio) * p
                    )
                )
            else:
                reg_grads.append(g)
        return reg_grads

    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        """
        Хелпер: применяет регуляризацию, вызывает step(), инкремент итерации, сохраняет историю и вызывает hook.
        """
        assert len(params) == len(grads), "params and grads must have the same length"

        regs = self.apply_regularization(params, grads)
        updated = self.step(params, regs)
        self.iteration += 1

        if self.track_history and self.iteration % self.track_interval == 0:
            self.history.append(copy.deepcopy(updated))

        if self.verbose:
            param_norm = sum(np.linalg.norm(p) for p in updated)
            print(f"Iteration {self.iteration}, param_norm: {param_norm:.4f}")

        if self.on_step:
            self.on_step(params, regs, updated)

        return updated

    def reset(self):
        """
        Сброс состояния оптимизатора (итерация, история).
        """
        self.iteration = 0
        self.history.clear()

    def set_learning_rate(self, lr: float):
        assert lr > 0, "learning_rate must be positive"
        self.learning_rate = lr

    def get_learning_rate(self) -> float:
        return self.learning_rate

    def get_config(self) -> dict:
        cfg = {
            "optimizer": self.__class__.__name__,
            "learning_rate": self.learning_rate,
            "iteration": self.iteration,
            "track_history": self.track_history,
            "track_interval": self.track_interval,
            "reg_type": self.reg_type,
            "weight_decay": self.weight_decay,
            "l1_ratio": self.l1_ratio,
            "verbose": self.verbose
        }
        return cfg

    def get_history(self) -> List[List[np.ndarray]]:
        return self.history

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(lr={self.learning_rate}, "
            f"iter={self.iteration}, reg={self.reg_type})"
        )
