"""AI Optimizers Playground application entry point.

Run: streamlit run app.py

Pages (Streamlit multipage):
    Home (this file)
    Playground: optimizer comparison (pages/1_Playground.py)
"""
import streamlit as st

st.set_page_config(
    page_title="AI Optimizers Playground",
    page_icon=None,
    layout="wide",
)

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

st.title("AI Optimizers Playground")
st.markdown("Интерактивная площадка для сравнения алгоритмов оптимизации.")

st.info(
    "Выберите **Playground** в боковой панели, чтобы запустить сравнение оптимизаторов "
    "на тестовых функциях (Rastrigin, Rosenbrock, Ackley и др.)."
)

st.markdown("---")
st.markdown("""
### О проекте
- **SGD, GD, RMSProp, Adagrad, Adam, AdamW, AMSGrad, Sophia, Lion, Adan, MARS, LARS** — все оптимизаторы в одном месте.
- Визуализация траекторий в 2D/3D, метрики и экспорт результатов.
- Единый API: везде используется `optimizer.update(params, grads)` для шага с учётом итерации и регуляризации.
""")
