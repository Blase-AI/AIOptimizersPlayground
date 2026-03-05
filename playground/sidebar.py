"""Sidebar: global params, optimizer selection, simulation and visualization options, buttons."""
import streamlit as st

from optimizers.registry import get_optimizer_names, get_param_spec
from core import TEST_FUNCTION_NAMES


def render_sidebar():
    """Render sidebar, write to session_state. Return dict with optimizers, params, test_func, UI flags."""
    with st.sidebar:
        st.markdown("### Настройки")
        st.caption("Параметры и запуск симуляции")
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
    }


def _render_global_params():
    """Render learning rate, momentum, weight decay inputs in an expander."""
    with st.expander("Глобальные параметры", expanded=True):
        lr_col1, lr_col2 = st.columns([1, 1])
        with lr_col1:
            st.session_state.learning_rate = st.number_input(
                "Скорость обучения",
                min_value=0.0001,
                max_value=0.1,
                value=st.session_state.learning_rate,
                step=0.0001,
                format="%.4f",
                key="pg_lr_input",
            )
        with lr_col2:
            st.session_state.learning_rate = st.slider(
                "LR", 0.0001, 0.1, st.session_state.learning_rate,
                step=0.0001, format="%.4f", label_visibility="collapsed", key="pg_lr_slider"
            )
        mom_col1, mom_col2 = st.columns([1, 1])
        with mom_col1:
            st.session_state.momentum = st.number_input(
                "Моментум (β)",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.momentum,
                step=0.01,
                format="%.2f",
                key="pg_mom_input",
            )
        with mom_col2:
            st.session_state.momentum = st.slider(
                "Mom", 0.0, 1.0, st.session_state.momentum,
                step=0.01, format="%.2f", label_visibility="collapsed", key="pg_mom_slider"
            )
        wd_col1, wd_col2 = st.columns([1, 1])
        with wd_col1:
            st.session_state.weight_decay = st.number_input(
                "Weight Decay",
                min_value=0.0,
                max_value=0.1,
                value=st.session_state.weight_decay,
                step=0.01,
                format="%.2f",
                key="pg_wd_input",
            )
        with wd_col2:
            st.session_state.weight_decay = st.slider(
                "WD", 0.0, 0.1, st.session_state.weight_decay,
                step=0.01, format="%.2f", label_visibility="collapsed", key="pg_wd_slider"
            )


def _render_optimizer_params():
    """Render multiselect for optimizers and per-optimizer param inputs. Return params dict."""
    with st.expander("Параметры оптимизаторов"):
        optimizers = st.multiselect(
            "Выберите оптимизаторы",
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
                    if param_key not in st.session_state:
                        st.session_state[param_key] = default
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.session_state[param_key] = st.number_input(
                            param,
                            min_value=min_val,
                            max_value=max_val,
                            value=st.session_state[param_key],
                            step=step,
                            format=fmt,
                            key=f"{param_key}_input",
                        )
                    with col2:
                        st.session_state[param_key] = st.slider(
                            param, min_val, max_val, st.session_state[param_key],
                            step=step, format=fmt, label_visibility="collapsed", key=f"{param_key}_slider"
                        )
                    params[f"{opt}_{param}"] = st.session_state[param_key]
    return params


def _render_simulation_settings():
    """Render iterations, resolution, bounds, test function. Return selected test_func."""
    with st.expander("Настройки симуляции"):
        iter_col1, iter_col2 = st.columns([1, 1])
        with iter_col1:
            st.session_state.iterations = st.number_input(
                "Количество итераций",
                min_value=10,
                max_value=5000,
                value=st.session_state.iterations,
                step=1,
                format="%d",
                key="pg_iter_input",
            )
        with iter_col2:
            st.session_state.iterations = st.slider(
                "Iter", 10, 5000, st.session_state.iterations,
                step=1, format="%d", label_visibility="collapsed", key="pg_iter_slider"
            )
        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.session_state.resolution = st.number_input(
                "Разрешение сетки",
                min_value=50,
                max_value=300,
                value=st.session_state.resolution,
                step=1,
                format="%d",
                key="pg_res_input",
            )
        with res_col2:
            st.session_state.resolution = st.slider(
                "Res", 50, 300, st.session_state.resolution,
                step=1, format="%d", label_visibility="collapsed", key="pg_res_slider"
            )
        bnd_col1, bnd_col2 = st.columns([1, 1])
        with bnd_col1:
            st.session_state.bounds = st.number_input(
                "Диапазон осей (X, Y)",
                min_value=1.0,
                max_value=500.0,
                value=st.session_state.bounds,
                step=0.5,
                format="%.1f",
                key="pg_bnd_input",
            )
        with bnd_col2:
            st.session_state.bounds = st.slider(
                "Bounds", 1.0, 500.0, st.session_state.bounds,
                step=0.5, format="%.1f", label_visibility="collapsed", key="pg_bnd_slider"
            )
        test_func = st.selectbox("Тестовая функция", TEST_FUNCTION_NAMES, key="pg_test_func")
    return test_func


def _render_visualization_options():
    """Render surface, 3D, colorbar, color scheme, noise, realtime. Return (add_noise, show_surface, show_3d, show_colorbar, realtime_update)."""
    with st.expander("Визуализация"):
        show_surface = st.checkbox("Показать поверхность", value=True, key="pg_show_surface")
        show_3d = st.checkbox("3D визуализация", value=True, key="pg_show_3d")
        show_colorbar = st.checkbox("Показать шкалу цветов", value=False, key="pg_show_colorbar")
        st.session_state.color_scheme = st.selectbox(
            "Цветовая палитра",
            ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Plotly"],
            index=["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Plotly"].index(
                st.session_state.color_scheme
            ),
            key="pg_color_scheme",
        )
        add_noise = st.checkbox("Добавить шум в градиенты", value=False, key="pg_add_noise")
        if add_noise:
            noise_col1, noise_col2 = st.columns([1, 1])
            with noise_col1:
                st.session_state.noise_level = st.number_input(
                    "Уровень шума",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.noise_level,
                    step=0.01,
                    format="%.2f",
                    key="pg_noise_input",
                )
            with noise_col2:
                st.session_state.noise_level = st.slider(
                    "Noise", 0.0, 1.0, st.session_state.noise_level,
                    step=0.01, format="%.2f", label_visibility="collapsed", key="pg_noise_slider"
                )
        else:
            st.session_state.noise_level = 0.0
        realtime_update = st.checkbox("Режим реального времени", value=True, key="pg_realtime")
    return add_noise, show_surface, show_3d, show_colorbar, realtime_update


def _render_buttons():
    """Render Run simulation, Reset, Save results buttons. Return True if Save was clicked."""
    if st.button("Запустить симуляцию", key="pg_run_btn"):
        st.session_state.run_simulation = True
        st.session_state.trajectories = {}
        try:
            st.rerun()
        except Exception as e:
            st.error(f"Ошибка при перезапуске: {e}")
    if st.button("Сбросить", key="pg_reset_btn"):
        st.session_state.trajectories = {}
        st.session_state.run_simulation = False
        st.cache_data.clear()
        try:
            st.rerun()
        except Exception as e:
            st.error(f"Ошибка при перезапуске: {e}")
    return st.button("Сохранить результаты", key="pg_save_btn")
