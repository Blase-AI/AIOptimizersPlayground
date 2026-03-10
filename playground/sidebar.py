"""Sidebar: global params, optimizer selection, simulation and visualization options, buttons."""
from typing import Optional

import streamlit as st

from core.i18n import render_language_switcher, t
from optimizers.registry import get_optimizer_names, get_param_spec
from core import TEST_FUNCTION_NAMES, get_preset_names, get_preset_by_name, get_param_help


def _rerun_safe() -> None:
    """Trigger Streamlit rerun; on exception show translated error in the UI."""
    try:
        st.rerun()
    except Exception as e:
        st.error(f"{t('error.rerun')}: {e}")


def _render_number_with_slider(
    state_key: str,
    label: str,
    min_val: float,
    max_val: float,
    step: float,
    fmt: str,
    key_prefix: str,
    default: Optional[float] = None,
    help_text: Optional[str] = None,
    short_label: Optional[str] = None,
):
    """Render two columns: number_input and slider, both bound to st.session_state[state_key].

    Args:
        state_key: Key in st.session_state to read/write the value.
        label: Label for the number input.
        min_val: Minimum value.
        max_val: Maximum value.
        step: Step size.
        fmt: Format string for display (e.g. "%.4f", "%d").
        key_prefix: Prefix for Streamlit widget keys (key_prefix_input, key_prefix_slider).
        default: Optional initial value if state_key is missing.
        help_text: Optional tooltip for the number input.
        short_label: Optional short label for the slider (collapsed visibility).
    """
    if default is not None and state_key not in st.session_state:
        st.session_state[state_key] = default
    value = st.session_state.get(state_key, default if default is not None else min_val)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.session_state[state_key] = st.number_input(
            label,
            min_value=min_val,
            max_value=max_val,
            value=value,
            step=step,
            format=fmt,
            key=f"{key_prefix}_input",
            help=help_text,
        )
    with col2:
        st.session_state[state_key] = st.slider(
            short_label or " ",
            min_val,
            max_val,
            st.session_state[state_key],
            step=step,
            format=fmt,
            label_visibility="collapsed",
            key=f"{key_prefix}_slider",
        )


def render_sidebar():
    """Render sidebar, write to session_state. Return dict with optimizers, params, test_func, UI flags."""
    with st.sidebar:
        render_language_switcher()
        _render_presets()
        st.markdown("### " + t("sidebar.settings"))
        st.caption(t("sidebar.settings_caption"))
        st.markdown("---")
        _render_global_params()
        st.markdown("---")
        params = _render_optimizer_params()
        st.markdown("---")
        test_func = _render_simulation_settings()
        st.markdown("---")
        add_noise, show_surface, show_3d, show_colorbar, realtime_update = _render_visualization_options()
        st.markdown("---")
        save_results = _render_buttons()
    optimizers = st.session_state.get("_pg_optimizers", ["AdamW"])
    compare_with_without_reg = st.session_state.get("compare_with_without_reg", False)
    return {
        "optimizers": optimizers,
        "params": params,
        "test_func": test_func,
        "add_noise": add_noise,
        "show_surface": show_surface,
        "show_3d": show_3d,
        "show_colorbar": show_colorbar,
        "realtime_update": realtime_update,
        "save_results": save_results,
        "compare_with_without_reg": compare_with_without_reg,
    }


def _render_presets():
    """Render preset selector and Apply button; apply preset to session state on click."""
    preset_names = get_preset_names()
    if "pg_preset_choice" not in st.session_state:
        st.session_state.pg_preset_choice = preset_names[0]
    idx = preset_names.index(st.session_state.get("pg_preset_choice", preset_names[0]))
    if idx < 0:
        idx = 0
    selected = st.selectbox(
        t("sidebar.preset"),
        preset_names,
        index=idx,
        key="pg_preset_select",
        help=t("sidebar.preset_help"),
    )
    st.session_state.pg_preset_choice = selected
    if st.button(t("sidebar.apply_preset"), key="pg_apply_preset"):
        preset = get_preset_by_name(selected)
        if preset:
            st.session_state.learning_rate = preset.get("learning_rate", st.session_state.learning_rate)
            st.session_state.momentum = preset.get("momentum", st.session_state.momentum)
            st.session_state.weight_decay = preset.get("weight_decay", st.session_state.weight_decay)
            st.session_state.iterations = preset.get("iterations", st.session_state.iterations)
            opts = preset.get("optimizers", ["AdamW"])
            st.session_state["_pg_optimizers"] = opts
            st.session_state["pg_optimizers"] = opts
            if preset.get("test_func") and preset["test_func"] in TEST_FUNCTION_NAMES:
                st.session_state["pg_test_func"] = preset["test_func"]
            st.session_state.trajectories = {}
            st.session_state.run_simulation = False
            _rerun_safe()
    st.markdown("---")


def _render_global_params():
    """Render learning rate, momentum, weight decay inputs in an expander."""
    with st.expander(t("sidebar.global_params"), expanded=True):
        _render_number_with_slider(
            "learning_rate",
            t("sidebar.learning_rate"),
            0.0001, 0.1, 0.0001, "%.4f", "pg_lr",
            help_text=t("sidebar.lr_help"), short_label="LR",
        )
        _render_number_with_slider(
            "momentum",
            t("sidebar.momentum"),
            0.0, 1.0, 0.01, "%.2f", "pg_mom",
            help_text=t("sidebar.momentum_help"), short_label="Mom",
        )
        _render_number_with_slider(
            "weight_decay",
            "Weight Decay",
            0.0, 0.1, 0.01, "%.2f", "pg_wd",
            help_text=t("sidebar.weight_decay_help"), short_label="WD",
        )


def _render_optimizer_params():
    """Render multiselect for optimizers and per-optimizer param inputs. Return params dict."""
    with st.expander(t("sidebar.optimizer_params")):
        optimizers = st.multiselect(
            t("sidebar.select_optimizers"),
            get_optimizer_names(),
            default=["AdamW"],
            key="pg_optimizers",
        )
        st.session_state["_pg_optimizers"] = optimizers
        params = {}
        for opt in optimizers:
            param_spec = get_param_spec(opt)
            if param_spec:
                st.markdown(f"**{opt}**")
                for param, (min_val, max_val, default, step, fmt) in param_spec.items():
                    param_key = f"pg_{opt}_{param}"
                    _render_number_with_slider(
                        param_key,
                        param,
                        min_val, max_val, step, fmt,
                        param_key,
                        default=default,
                        help_text=get_param_help(param) or None,
                    )
                    params[f"{opt}_{param}"] = st.session_state[param_key]
    return params


def _render_simulation_settings():
    """Render iterations, resolution, bounds, test function. Return selected test_func."""
    with st.expander(t("sidebar.simulation")):
        _render_number_with_slider(
            "iterations",
            t("sidebar.iterations"),
            10, 5000, 1, "%d", "pg_iter",
            short_label="Iter",
        )
        _render_number_with_slider(
            "resolution",
            t("sidebar.resolution"),
            50, 300, 1, "%d", "pg_res",
            short_label="Res",
        )
        _render_number_with_slider(
            "bounds",
            t("sidebar.bounds"),
            1.0, 500.0, 0.5, "%.1f", "pg_bnd",
            short_label="Bounds",
        )
        random_start = st.checkbox(
            t("sidebar.random_start"),
            value=st.session_state.get("random_start", True),
            key="pg_random_start",
            help=t("sidebar.random_start_help"),
        )
        st.session_state.random_start = random_start
        if random_start:
            seed_val = st.session_state.get("random_seed")
            seed_input = st.number_input(
                t("sidebar.seed_label"),
                min_value=0,
                value=int(seed_val) if seed_val is not None else 0,
                step=1,
                format="%d",
                key="pg_seed_input",
                help=t("sidebar.seed_help"),
            )
            st.session_state.random_seed = seed_input if seed_input else None
        else:
            col_x, col_y = st.columns(2)
            with col_x:
                st.session_state.start_x0 = st.number_input(
                    "x0",
                    value=float(st.session_state.get("start_x0", 2.0)),
                    format="%.2f",
                    key="pg_start_x0",
                )
            with col_y:
                st.session_state.start_y0 = st.number_input(
                    "y0",
                    value=float(st.session_state.get("start_y0", -2.0)),
                    format="%.2f",
                    key="pg_start_y0",
                )
        test_func = st.selectbox(t("sidebar.test_func"), TEST_FUNCTION_NAMES, key="pg_test_func")
        compare_with_without_reg = st.checkbox(
            t("sidebar.compare_reg"),
            value=st.session_state.get("compare_with_without_reg", False),
            key="pg_compare_reg",
            help=t("sidebar.compare_reg_help"),
        )
        st.session_state.compare_with_without_reg = compare_with_without_reg
    return test_func


def _render_visualization_options():
    """Render surface, 3D, colorbar, color scheme, noise, realtime. Return (add_noise, show_surface, show_3d, show_colorbar, realtime_update)."""
    with st.expander(t("sidebar.visualization")):
        show_surface = st.checkbox(t("sidebar.show_surface"), value=True, key="pg_show_surface")
        show_3d = st.checkbox(t("sidebar.show_3d"), value=True, key="pg_show_3d")
        show_colorbar = st.checkbox(t("sidebar.show_colorbar"), value=False, key="pg_show_colorbar")
        st.session_state.color_scheme = st.selectbox(
            t("sidebar.color_scheme"),
            ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Plotly"],
            index=["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Plotly"].index(
                st.session_state.color_scheme
            ),
            key="pg_color_scheme",
        )
        add_noise = st.checkbox(t("sidebar.add_noise"), value=False, key="pg_add_noise")
        if add_noise:
            _render_number_with_slider(
                "noise_level",
                t("sidebar.noise_level"),
                0.0, 1.0, 0.01, "%.2f", "pg_noise",
                default=0.0, short_label="Noise",
            )
        else:
            st.session_state.noise_level = 0.0
        realtime_update = st.checkbox(t("sidebar.realtime"), value=True, key="pg_realtime")
    return add_noise, show_surface, show_3d, show_colorbar, realtime_update


def _render_buttons():
    """Render Run simulation, Reset, Save results buttons. Return True if Save was clicked."""
    if st.button(t("sidebar.run"), key="pg_run_btn"):
        st.session_state.run_simulation = True
        st.session_state.trajectories = {}
        _rerun_safe()
    if st.button(t("sidebar.reset"), key="pg_reset_btn"):
        st.session_state.trajectories = {}
        st.session_state["trajectories_no_reg"] = {}
        st.session_state.run_simulation = False
        st.cache_data.clear()
        _rerun_safe()
    return st.button(t("sidebar.save_results"), key="pg_save_btn")
