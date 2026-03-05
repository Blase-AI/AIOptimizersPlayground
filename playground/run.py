"""Simulation loop: run optimizers and fill trajectories in session_state."""
import asyncio
import streamlit as st
import numpy as np

from core import get_test_functions, run_optimization as core_run_optimization
from .state import init_optimizer


def run_simulation_loop(optimizers, params, test_func, add_noise):
    """Run simulation for selected optimizers and update st.session_state.trajectories."""
    if not optimizers:
        return
    test_functions = get_test_functions()
    progress_bar = st.progress(0)
    start = np.random.uniform(
        -st.session_state.bounds, st.session_state.bounds, size=(2,)
    )
    for idx, opt_name in enumerate(optimizers):
        if opt_name not in st.session_state.trajectories:
            opt = init_optimizer(
                opt_name,
                st.session_state.learning_rate,
                st.session_state.momentum,
                st.session_state.weight_decay,
                tuple(sorted(params.items())),
            )
            try:
                trajectory, losses, grad_norms, local_lrs = asyncio.run(
                    core_run_optimization(
                        opt,
                        start,
                        test_functions,
                        test_func,
                        st.session_state.iterations,
                        st.session_state.noise_level,
                        st.session_state.bounds,
                        add_noise,
                    )
                )
            except ValueError as e:
                st.error(str(e))
                trajectory, losses, grad_norms, local_lrs = [], [], [], None
            st.session_state.trajectories[opt_name] = {
                "traj": trajectory,
                "loss": losses,
                "grad_norms": grad_norms,
                "local_lrs": local_lrs,
            }
        progress_bar.progress((idx + 1) / len(optimizers))
