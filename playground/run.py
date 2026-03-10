"""Run optimizers and write trajectories (and optional no-reg comparison) to session_state."""
import asyncio

import numpy as np
import streamlit as st

from core import get_test_functions, run_optimization as core_run_optimization
from .state import init_optimizer


def run_simulation_loop(optimizers, params, test_func, add_noise, compare_with_without_reg=False):
    """Run optimization for each selected optimizer; store trajectories in session_state.

    If compare_with_without_reg is True and weight_decay > 0, runs each optimizer
    again with weight_decay=0 and stores results in session_state["trajectories_no_reg"].

    Args:
        optimizers: List of optimizer names.
        params: Dict of extra optimizer params (e.g. from sidebar).
        test_func: Test function name.
        add_noise: Whether to add gradient noise.
        compare_with_without_reg: If True and weight_decay > 0, also run without regularization.
    """
    if not optimizers:
        return
    st.session_state["trajectories_no_reg"] = {}
    test_functions = get_test_functions()
    progress_bar = st.progress(0)
    if st.session_state.get("random_start", True):
        seed = st.session_state.get("random_seed")
        if seed is not None:
            np.random.seed(int(seed))
        start = np.random.uniform(
            -st.session_state.bounds, st.session_state.bounds, size=(2,)
        )
        st.session_state["_last_start"] = start.tolist()
        st.session_state["_last_seed"] = seed
        st.session_state["_last_random_start"] = True
    else:
        start = np.array(
            [
                float(st.session_state.get("start_x0", 2.0)),
                float(st.session_state.get("start_y0", -2.0)),
            ],
            dtype=np.float64,
        )
        st.session_state["_last_start"] = start.tolist()
        st.session_state["_last_seed"] = None
        st.session_state["_last_random_start"] = False
    st.session_state["_last_run_test_func"] = test_func
    st.session_state["_last_run_iterations"] = st.session_state.iterations
    st.session_state["_last_run_optimizers"] = list(optimizers)
    st.session_state["_last_run_random_start"] = st.session_state.get("random_start", True)
    total = len(optimizers) + (len(optimizers) if (compare_with_without_reg and st.session_state.get("weight_decay", 0) > 0) else 0)
    done = 0
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
        done += 1
        progress_bar.progress(done / total)
    wd = st.session_state.get("weight_decay", 0)
    if compare_with_without_reg and wd > 0:
        for idx, opt_name in enumerate(optimizers):
            opt_no_reg = init_optimizer(
                opt_name,
                st.session_state.learning_rate,
                st.session_state.momentum,
                0.0,
                tuple(sorted(params.items())),
            )
            try:
                trajectory, losses, grad_norms, local_lrs = asyncio.run(
                    core_run_optimization(
                        opt_no_reg,
                        start,
                        test_functions,
                        test_func,
                        st.session_state.iterations,
                        st.session_state.noise_level,
                        st.session_state.bounds,
                        add_noise,
                    )
                )
            except ValueError:
                trajectory, losses, grad_norms, local_lrs = [], [], [], None
            st.session_state["trajectories_no_reg"][opt_name] = {
                "traj": trajectory,
                "loss": losses,
                "grad_norms": grad_norms,
                "local_lrs": local_lrs,
            }
            done += 1
            progress_bar.progress(done / total)
