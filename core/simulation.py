"""Optimization loop: run optimizer steps with optional gradient noise. No Streamlit."""
import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


async def run_optimization(
    opt,
    start_params: np.ndarray,
    test_functions: Dict[str, Dict[str, Any]],
    test_func: str,
    iterations: int,
    noise_level: float,
    bounds: float,
    add_noise: bool,
) -> Tuple[List[np.ndarray], List[float], List[float], Optional[List[float]]]:
    """Run optimizer for a given number of steps.

    Uses opt.update(params, grads) as single API. Applies optional gradient noise.
    Clips params to [-bounds, bounds] after each step.

    Args:
        opt: Optimizer instance with update() method.
        start_params: Initial parameter vector.
        test_functions: Dict of name -> {func, grad}.
        test_func: Name of test function to use.
        iterations: Number of steps.
        noise_level: Std of Gaussian noise added to gradients if add_noise.
        bounds: Clip params to [-bounds, bounds].
        add_noise: Whether to add noise to gradients.

    Returns:
        Tuple of (trajectory, losses, grad_norms, local_lrs or None for non-LARS).

    Raises:
        ValueError: If gradients contain NaN or Inf.
    """
    params = [start_params.copy()]
    trajectory = [start_params.copy()]
    func_and_grad = test_functions[test_func]
    losses = [func_and_grad["func"](start_params)]
    grad_norms: List[float] = []
    local_lrs: Optional[List[float]] = []

    for i in range(iterations):
        grad = func_and_grad["grad"](params[0]).copy()
        if add_noise:
            grad += np.random.normal(0, noise_level, size=grad.shape)

        if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
            raise ValueError(f"Invalid gradients for {opt.__class__.__name__}")

        local_lr = opt.get_local_lr(params, [grad])
        if local_lr is not None:
            local_lrs.append(local_lr)
            logger.debug("%s | local_lr: %.6f", opt.__class__.__name__, local_lr)

        updated = opt.update(params, [grad])
        updated[0] = np.clip(updated[0], -bounds, bounds)
        trajectory.append(updated[0].copy())
        loss = func_and_grad["func"](updated[0])
        losses.append(loss)
        grad_norms.append(float(np.linalg.norm(grad)))
        params = updated

        logger.debug(
            "%s | Iter %d | Loss: %.4f | Grad norm: %.4f",
            opt.__class__.__name__, i + 1, loss, grad_norms[-1],
        )

        await asyncio.sleep(0)

    return trajectory, losses, grad_norms, local_lrs if local_lrs else None
