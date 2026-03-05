"""Description tab: text for test function and selected optimizers."""
import streamlit as st

from core.descriptions import FUNCTION_DESCRIPTIONS, OPTIMIZER_DESCRIPTIONS


def render_description_tab(tab, test_func, optimizers):
    """Render tab with test function and optimizer descriptions."""
    with tab:
        st.markdown(f"#### Тестовая функция: **{test_func}**")
        st.markdown(FUNCTION_DESCRIPTIONS.get(test_func, ""))
        st.markdown("---")
        st.markdown("#### Оптимизаторы")
        for opt_name in optimizers:
            with st.expander(opt_name):
                st.markdown(OPTIMIZER_DESCRIPTIONS.get(opt_name, "Нет описания."))
