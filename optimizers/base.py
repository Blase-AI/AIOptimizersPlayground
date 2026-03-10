"""Base optimizer and gradient clipping utility."""
from abc import ABC, abstractmethod
import copy
import logging
from typing import List, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


def clip_gradient(g: np.ndarray, clip_norm: Optional[float], eps: float = 1e-10) -> np.ndarray:
    """Clip gradient by L2 norm. Returns new array if clipped; in-place safe.

    Args:
        g: Gradient array.
        clip_norm: Max norm; None disables clipping.
        eps: Small value for numerical stability.

    Returns:
        Clipped gradient (same as g if norm <= clip_norm).
    """
    if clip_norm is None:
        return g
    norm = np.linalg.norm(g)
    if norm <= clip_norm:
        return g
    return g * (clip_norm / (norm + eps))


class BaseOptimizer(ABC):
    """Base class for all optimizers with regularization, history, and hooks."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        track_history: bool = False,
        track_interval: int = 1,
        on_step: Optional[Callable[[List[np.ndarray], List[np.ndarray], List[np.ndarray]], None]] = None,
        reg_type: str = 'none',
        weight_decay: float = 0.0,
        l1_ratio: float = 0.5,
        verbose: bool = False,
        clip_norm: Optional[float] = None,
        decay_rate: float = 1.0,
    ):
        """Initialize base optimizer.

        Args:
            learning_rate: Initial step size.
            track_history: Whether to store parameter history.
            track_interval: Store history every N steps.
            on_step: Callback (params, grads, updated) after each step.
            reg_type: 'none' | 'l1' | 'l2' | 'enet'.
            weight_decay: Regularization strength.
            l1_ratio: L1 fraction for elastic net (0-1).
            verbose: Log per-step info.
            clip_norm: Max gradient norm (None = no clipping).
            decay_rate: LR decay per step: lr * decay_rate^iteration.
        """
        assert reg_type in {'none', 'l1', 'l2', 'enet'}, \
            "reg_type must be one of 'none','l1','l2','enet'"
        assert learning_rate > 0, "learning_rate must be positive"
        assert weight_decay >= 0, "weight_decay must be non-negative"
        assert 0 <= l1_ratio <= 1, "l1_ratio must be in [0, 1]"
        assert track_interval >= 1, "track_interval must be at least 1"
        assert clip_norm is None or clip_norm > 0, "clip_norm must be positive or None"
        assert 0 < decay_rate <= 1, "decay_rate must be in (0, 1]"

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
        self.clip_norm = clip_norm
        self.decay_rate = decay_rate

    def get_local_lr(self, params: List[np.ndarray], grads: List[np.ndarray]) -> Optional[float]:
        """Return per-step local learning rate if the optimizer tracks it (e.g. LARS). Otherwise None."""
        return None

    @abstractmethod
    def step(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        """Update parameters given gradients.

        Args:
            params: List of parameter arrays.
            grads: List of gradient arrays (already regularized if applicable).

        Returns:
            New list of parameter arrays.
        """
        pass

    def _clip_gradient(self, g: np.ndarray) -> np.ndarray:
        """Return gradient, optionally clipped by self.clip_norm."""
        return clip_gradient(g, self.clip_norm)

    def _effective_lr(self) -> float:
        """Effective learning rate at current iteration: lr * decay_rate^iteration."""
        return self.learning_rate * (self.decay_rate ** self.iteration)

    def apply_regularization(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        """Apply L1/L2/elastic net to gradients and return modified list."""
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
        """Apply regularization, step(), increment iteration, optional history and hook."""
        assert len(params) == len(grads), "params and grads must have the same length"

        regs = self.apply_regularization(params, grads)
        updated = self.step(params, regs)
        self.iteration += 1

        if self.track_history and self.iteration % self.track_interval == 0:
            self.history.append(copy.deepcopy(updated))

        if self.verbose:
            param_norm = sum(np.linalg.norm(p) for p in updated)
            logger.debug("Iteration %d, param_norm: %.4f", self.iteration, param_norm)

        if self.on_step:
            self.on_step(params, regs, updated)

        return updated

    def reset(self):
        """Reset iteration count and history."""
        self.iteration = 0
        self.history.clear()

    def set_learning_rate(self, lr: float):
        """Set learning rate (must be positive)."""
        assert lr > 0, "learning_rate must be positive"
        self.learning_rate = lr

    def get_learning_rate(self) -> float:
        """Return current learning rate."""
        return self.learning_rate

    def get_config(self) -> dict:
        """Return dict of optimizer config for serialization."""
        cfg = {
            "optimizer": self.__class__.__name__,
            "learning_rate": self.learning_rate,
            "iteration": self.iteration,
            "track_history": self.track_history,
            "track_interval": self.track_interval,
            "reg_type": self.reg_type,
            "weight_decay": self.weight_decay,
            "l1_ratio": self.l1_ratio,
            "verbose": self.verbose,
            "clip_norm": self.clip_norm,
            "decay_rate": self.decay_rate,
        }
        return cfg

    def get_history(self) -> List[List[np.ndarray]]:
        """Return stored parameter history."""
        return self.history

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(lr={self.learning_rate}, "
            f"iter={self.iteration}, reg={self.reg_type})"
        )
