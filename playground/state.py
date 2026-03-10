"""Session state defaults and cached helpers for mesh and optimizer creation."""
import streamlit as st

from core import generate_test_data as core_generate_test_data
from optimizers.registry import create_optimizer

SESSION_DEFAULTS = [
    ("trajectories", {}),
    ("run_simulation", False),
    ("learning_rate", 0.001),
    ("momentum", 0.9),
    ("weight_decay", 0.01),
    ("iterations", 100),
    ("resolution", 100),
    ("bounds", 5.0),
    ("noise_level", 0.1),
    ("color_scheme", "Viridis"),
    ("random_start", True),
    ("start_x0", 2.0),
    ("start_y0", -2.0),
    ("random_seed", None),
]


def init_session_state():
    """Initialize st.session_state with default keys if missing."""
    for key, default in SESSION_DEFAULTS:
        if key not in st.session_state:
            st.session_state[key] = default


@st.cache_data
def generate_test_data(name: str, res: int, bnd: float):
    """Return (X, Y, Z) mesh for the given test function. Cached.

    Args:
        name: Test function name (e.g. from TEST_FUNCTION_NAMES).
        res: Grid resolution.
        bnd: Axis bound (-bnd, bnd).

    Returns:
        Tuple of 2D arrays (X, Y, Z) for surface/contour plots.
    """
    return core_generate_test_data(name, res, bnd)


def _params_from_tuple(params_tuple):
    """Convert list of (k, v) to dict; return {} if empty or None."""
    return dict(params_tuple) if params_tuple else {}


@st.cache_resource
def init_optimizer(opt_name, lr, momentum, wd, params_tuple):
    """Create and cache an optimizer instance from the registry.

    Args:
        opt_name: Registered optimizer name.
        lr: Learning rate.
        momentum: Momentum (used if optimizer supports it).
        wd: Weight decay.
        params_tuple: Sorted iterable of (key, value) for extra params.

    Returns:
        Optimizer instance (cached per arguments).
    """
    params = _params_from_tuple(params_tuple)
    return create_optimizer(opt_name, lr, momentum, wd, params)
