"""AI Optimizers Playground — entry point.

Usage:
    streamlit run app.py

Multipage app: Home (this file), Playground (pages/1_Playground.py), Glossary (pages/2_Glossary.py).
"""
import logging

import streamlit as st

from core.i18n import render_language_switcher, t

logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("optimizers").setLevel(logging.WARNING)
logging.getLogger("core").setLevel(logging.WARNING)

st.set_page_config(
    page_title="AI Optimizers Playground",
    page_icon="🧪",
    layout="wide"
)

if "lang" not in st.session_state:
    st.session_state["lang"] = "ru"
render_language_switcher()

st.markdown("""
<style>
    .main { background-color: #f9fafb; color: #333; }
    .stButton>button {
        background-color: #0288d1; color: white; border: none; border-radius: 6px;
        padding: 8px 16px; transition: background-color 0.2s;
    }
    .stButton>button:hover { background-color: #0277bd; }
    h1, h2, h3 { color: #333; font-family: 'Helvetica Neue', sans-serif; }
</style>
""", unsafe_allow_html=True)

st.title(t("app.title"))
st.markdown(t("app.subtitle"))

st.info(t("app.info_playground"))

st.markdown("---")
st.markdown("### " + t("app.guide_title"))
col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.markdown("**" + t("app.how_to_use") + "**")
        st.markdown(t("app.how_to_use_text"))
with col2:
    with st.container(border=True):
        st.markdown("**" + t("app.params_tips") + "**")
        st.markdown(t("app.params_tips_text"))

st.markdown("---")
st.markdown("### " + t("app.about_title"))
st.markdown(t("app.about_text"))
