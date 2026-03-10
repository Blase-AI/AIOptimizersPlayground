"""Description tab: text for test function and selected optimizers, with formulas."""
import streamlit as st

from core.descriptions import get_function_description, get_optimizer_description
from core.formulas import get_function_formula, get_optimizer_formula
from core.code_snippets import get_code_snippet
from core.i18n import t


def render_description_tab(tab, test_func, optimizers):
    """Render tab with test function and optimizer descriptions and formulas."""
    with tab:
        st.markdown(f"#### {t('desc.test_func')}: **{test_func}**")
        func_info = get_function_formula(test_func)
        if func_info.get("formula"):
            st.markdown("$" + func_info["formula"] + "$")
            if func_info.get("minimum"):
                st.caption(f"{t('desc.global_min')}: {func_info['minimum']}")
        st.markdown(get_function_description(test_func))
        st.markdown("---")
        st.markdown("#### " + t("desc.optimizers"))
        for opt_name in optimizers:
            with st.expander(opt_name):
                st.markdown(get_optimizer_description(opt_name) or t("desc.no_description"))
                formula = get_optimizer_formula(opt_name)
                if formula:
                    st.markdown("**" + t("desc.formulas") + "**")
                    st.markdown(formula)
                snippet = get_code_snippet(opt_name)
                if snippet:
                    st.markdown("**" + t("desc.code_step") + "**")
                    st.code(snippet, language="python")
