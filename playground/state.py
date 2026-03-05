"""Session state initialization and cached helpers (mesh, optimizer)."""
import streamlit as st

from optimizers.registry import create_optimizer
from core import generate_test_data as core_generate_test_data

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
]


def init_session_state():
    """Set default keys in st.session_state if not present."""
    for key, default in SESSION_DEFAULTS:
        if key not in st.session_state:
            st.session_state[key] = default


@st.cache_data
def generate_test_data(name: str, res: int, bnd: float):
    """Cached wrapper for core.generate_test_data (mesh X, Y, Z)."""
    return core_generate_test_data(name, res, bnd)


def _params_from_tuple(params_tuple):
    """Convert list of (k,v) to dict; return {} if empty or None."""
    return dict(params_tuple) if params_tuple else {}


@st.cache_resource
def init_optimizer(opt_name, lr, momentum, wd, params_tuple):
    """Create and cache optimizer instance from registry."""
    params = _params_from_tuple(params_tuple)
    return create_optimizer(opt_name, lr, momentum, wd, params)
