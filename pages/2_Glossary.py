"""Glossary page: formulas for optimizers and test functions, regularization viz, term definitions."""
import streamlit as st

from core import (
    TEST_FUNCTION_NAMES,
    get_optimizer_formula,
    get_function_formula,
    get_glossary_entries,
)
from core.i18n import render_language_switcher, t
from core.regularization_viz import build_l2_figure, build_l1_figure, build_elastic_figure
from optimizers.registry import get_optimizer_names


st.set_page_config(page_title="Glossary | AI Optimizers Playground", layout="wide")
if "lang" not in st.session_state:
    st.session_state["lang"] = "ru"
render_language_switcher()
st.markdown("# " + t("glossary.title"))
st.caption(t("glossary.caption"))

section_labels = [t("glossary.terms"), t("glossary.optimizers"), t("glossary.test_functions"), t("glossary.regularization")]
section = st.radio(
    t("glossary.section_label"),
    section_labels,
    horizontal=True,
)

if section == section_labels[0]:
    st.markdown("## " + t("glossary.terms"))
    for term, definition in get_glossary_entries():
        with st.expander(term, expanded=False):
            st.markdown(definition)
elif section == section_labels[1]:
    st.markdown("## " + t("glossary.optimizers"))
    for opt_name in get_optimizer_names():
        with st.expander(opt_name, expanded=False):
            formula = get_optimizer_formula(opt_name)
            if formula:
                st.markdown(formula)
            else:
                st.caption(t("glossary.no_formula"))
elif section == section_labels[2]:
    st.markdown("## " + t("glossary.test_functions"))
    for func_name in TEST_FUNCTION_NAMES:
        info = get_function_formula(func_name)
        with st.expander(func_name, expanded=False):
            if info.get("formula"):
                st.markdown("$" + info["formula"] + "$")
            if info.get("minimum"):
                st.caption(f"{t('desc.global_min')}: {info['minimum']}")
else:
    st.markdown("## " + t("glossary.regularization"))
    st.markdown(t("glossary.reg_intro"))
    st.markdown("### " + t("glossary.math_title"))
    with st.expander(t("glossary.math_expand"), expanded=True):
        st.markdown(r"""
        **Исходная задача:** минимизировать $L(\theta)$ (например, квадратичные потери $L = \frac{1}{2}\|\theta - c\|^2$).

        - **L2 (Ridge):** $J = L(\theta) + \frac{\lambda}{2}\|\theta\|_2^2$. Градиент штрафа: $\lambda\theta$.  
          Решение в явном виде (для квадратичного $L$): $\theta^* = c/(1+\lambda)$ — сдвиг к нулю по всем координатам.

        - **L1 (Lasso):** $J = L(\theta) + \lambda\|\theta\|_1$. Субградиент штрафа: $\lambda\,\mathrm{sign}(\theta)$.  
          Решение: **soft-thresholding** по каждой координате: $\theta^*_i = \mathrm{sign}(c_i)\max(|c_i|-\lambda, 0)$.  
          При достаточно большом $\lambda$ координаты обнуляются → **разреженность (sparsity)**.

        - **Elastic Net:** $J = L(\theta) + \lambda\bigl(\alpha\|\theta\|_1 + (1-\alpha)\frac{1}{2}\|\theta\|_2^2\bigr)$.  
          Комбинация L1 и L2: и сжатие к нулю, и групповой эффект.
        """)
    st.markdown("### " + t("glossary.viz_heading"))
    st.caption(t("glossary.viz_caption"))
    lam = st.slider(t("glossary.slider_lam"), 0.0, 3.0, 0.5, 0.1, key="gl_reg_lambda")
    tab_l2, tab_l1, tab_el = st.tabs([t("glossary.tab_l2"), t("glossary.tab_l1"), "Elastic Net"])
    with tab_l2:
        st.plotly_chart(build_l2_figure(lam), use_container_width=True)
    with tab_l1:
        st.plotly_chart(build_l1_figure(lam), use_container_width=True)
    with tab_el:
        l1_ratio = st.slider(t("glossary.slider_l1_ratio"), 0.0, 1.0, 0.5, 0.1, key="gl_elastic_ratio")
        st.plotly_chart(build_elastic_figure(lam, l1_ratio), use_container_width=True)
