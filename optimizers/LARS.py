import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Callable
import logging
from .base import BaseOptimizer
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
        """Initialize LARS. trust_coeff scales per-layer LR by param_norm / (grad_norm + wd*param_norm)."""
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
        assert 0 <= momentum < 1, "momentum must be in [0, 1)"
        assert weight_decay >= 0, "weight_decay must be non-negative"
        assert trust_coeff > 0, "trust_coeff must be positive"
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.trust_coeff = trust_coeff
        self.velocities: Optional[List[NDArray[np.float64]]] = None

    def reset(self):
        """Reset iteration, history, and velocities."""
        super().reset()
        self.velocities = None

    @timed
    def step(
        self,
        params: List[NDArray[np.float64]],
        grads: List[NDArray[np.float64]]
    ) -> List[NDArray[np.float64]]:
        """LARS step: layer-wise adaptive LR and momentum with weight decay."""
        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p in params]

        lr = self._effective_lr()
        t = self.iteration + 1
        updated_params: List[NDArray[np.float64]] = []

        for i, (p, g) in enumerate(zip(params, grads)):
            g = self._clip_gradient(g)
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
        })
        return cfg

    def __repr__(self) -> str:
        base = super().__repr__().rstrip(')')
        return (
            f"{base}, momentum={self.momentum}, wd={self.weight_decay}, "
            f"trust_coeff={self.trust_coeff})"
        )
