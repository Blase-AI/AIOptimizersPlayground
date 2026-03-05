"""Playground page: optimizer comparison (thin entry point, logic in playground/)."""
import logging

import streamlit as st

from playground import (
    handle_export,
    init_session_state,
    render_description_tab,
    render_guide_tab,
    render_metrics_tab,
    render_sidebar,
    render_visualization_tab,
    run_simulation_loop,
)
from playground.styles import PLAYGROUND_CSS

logging.basicConfig(level=logging.INFO, filename="optimizer_debug.log")

st.markdown(PLAYGROUND_CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="pg-header">'
    '<h1>Сравнение оптимизаторов</h1>'
    '<p>Запустите симуляцию и сравните траектории на 2D/3D ландшафте тестовой функции.</p>'
    '</div>',
    unsafe_allow_html=True,
)

init_session_state()
sidebar_result = render_sidebar()
optimizers = sidebar_result["optimizers"]
params = sidebar_result["params"]
test_func = sidebar_result["test_func"]
add_noise = sidebar_result["add_noise"]
show_surface = sidebar_result["show_surface"]
show_3d = sidebar_result["show_3d"]
show_colorbar = sidebar_result["show_colorbar"]
realtime_update = sidebar_result["realtime_update"]
save_results = sidebar_result["save_results"]

tab1, tab2, tab3, tab4 = st.tabs(["Визуализация", "Метрики", "Описание", "Руководство"])

if st.session_state.run_simulation and optimizers:
    run_simulation_loop(optimizers, params, test_func, add_noise)

render_visualization_tab(
    tab1, optimizers, test_func, show_surface, show_3d, show_colorbar, realtime_update
)
render_metrics_tab(tab2, optimizers)
render_description_tab(tab3, test_func, optimizers)
render_guide_tab(tab4)
handle_export(save_results, test_func, optimizers, tab2)
