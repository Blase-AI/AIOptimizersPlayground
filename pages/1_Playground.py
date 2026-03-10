"""Playground page: optimizer comparison (thin entry point, logic in playground/)."""
import streamlit as st

from core.i18n import t
from playground import (
    handle_export,
    init_session_state,
    render_description_tab,
    render_metrics_tab,
    render_sidebar,
    render_visualization_tab,
    run_simulation_loop,
)
from playground.styles import PLAYGROUND_CSS

st.markdown(PLAYGROUND_CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="pg-header">'
    f'<h1>{t("pg.title")}</h1>'
    f'<p>{t("pg.caption")}</p>'
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
compare_with_without_reg = sidebar_result.get("compare_with_without_reg", False)

tab1, tab2, tab3 = st.tabs(
    [t("pg.tab_viz"), t("pg.tab_metrics"), t("pg.tab_desc")]
)

if st.session_state.run_simulation and optimizers:
    run_simulation_loop(optimizers, params, test_func, add_noise, compare_with_without_reg)

render_visualization_tab(
    tab1, optimizers, test_func, show_surface, show_3d, show_colorbar, realtime_update
)
render_metrics_tab(tab2, optimizers)
render_description_tab(tab3, test_func, optimizers)
handle_export(save_results, test_func, optimizers, tab2)
