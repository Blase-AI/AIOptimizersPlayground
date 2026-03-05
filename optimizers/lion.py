import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Callable
import logging
from .base import BaseOptimizer
from .dtime import timed

logger = logging.getLogger(__name__)


class Lion(BaseOptimizer):
    """Lion: sign-momentum optimizer. Optional bias correction, LR scheduler, decoupled weight decay, clipping."""

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
        """Initialize Lion. lr_scheduler(epoch) overrides decay_rate if set."""
        super().__init__(
            learning_rate=learning_rate,
            track_history=track_history,
            track_interval=track_interval,
            on_step=on_step,
            reg_type='none',
            weight_decay=0.0,
            l1_ratio=0.5,
            verbose=verbose,
            clip_norm=clip_norm,
            decay_rate=decay_rate,
        )
        assert 0 <= beta < 1, "beta must be in [0, 1)"
        assert weight_decay >= 0, "weight_decay must be non-negative"
        self.beta = beta
        self.weight_decay = weight_decay
        self.bias_correction = bias_correction
        self.lr_scheduler = lr_scheduler
        self.v: Optional[List[NDArray[np.float64]]] = None

    def reset(self):
        """Reset iteration, history, and momentum."""
        super().reset()
        self.v = None

    @timed
    def step(
        self,
        params: List[NDArray[np.float64]],
        grads: List[NDArray[np.float64]]
    ) -> List[NDArray[np.float64]]:
        """Lion step: sign-momentum update with optional weight decay."""
        t = self.iteration + 1

        if self.lr_scheduler:
            lr = self.lr_scheduler(t)
        else:
            lr = self._effective_lr()

        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]

        updated_params: List[NDArray[np.float64]] = []
        for i, (p, g) in enumerate(zip(params, grads)):
            g = self._clip_gradient(g)
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
            'bias_correction': self.bias_correction,
            'lr_scheduler': getattr(self.lr_scheduler, '__name__', repr(self.lr_scheduler)) if self.lr_scheduler else None,
        })
        return cfg

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(')')
        return f"{base}, beta={self.beta}, wd={self.weight_decay}, bias_correction={self.bias_correction})"
