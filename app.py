import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from optimizers import StochasticGradientDescent, GradientDescent, RMSProp, AdamW, Lion, Adan, MARS

st.set_page_config(
    page_title="AI Optimizers Playground",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stButton>button {border-radius: 8px; padding: 8px 16px;}
    .stSelectbox, .stSlider {margin-bottom: 20px;}
    .metric-card {border-radius: 10px; padding: 15px; background: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .plot-container {margin-top: 30px;}
</style>
""", unsafe_allow_html=True)

st.title("🧠 AI Optimizers Playground")
st.markdown("Интерактивная площадка для сравнения алгоритмов оптимизации в машинном обучении")

with st.sidebar:
    st.header("⚙️ Настройки оптимизации и тестовой функции")

    optimizer = st.selectbox(
        "Выберите оптимизатор",
        ("SGD", "GD", "RMSProp", "AdamW", "Lion", "Adan", "MARS"),
        index=2
    )

    learning_rate = st.slider("Скорость обучения", 0.0001, 0.1, 0.001, format="%.4f")
    momentum = st.slider("Моментум (β)", 0.0, 1.0, 0.9)
    weight_decay = st.slider("Weight Decay", 0.0, 0.1, 0.01)
    iterations = st.slider("Количество итераций", 10, 500, 100)

    st.markdown("---")
    test_func = st.selectbox(
        "Выберите тестовую функцию",
        ("Quadratic", "Rastrigin", "Rosenbrock", "Himmelblau")
    )
    resolution = st.slider("Разрешение сетки", 50, 300, 100)
    bounds = st.slider("Диапазон осей для X и Y", 1.0, 10.0, 5.0, step=0.5)

    st.markdown("---")
    show_surface = st.checkbox("Показать поверхность", value=True)
    show_3d = st.checkbox("3D визуализация", value=True)
    realtime_update = st.checkbox("Режим реального времени", value=True)

    if st.button("🔄 Сбросить и начать заново"):
        st.experimental_rerun()

if 'history' not in st.session_state:
    st.session_state.history = {'loss': [], 'grad_norm': [], 'param_norm': [], 'iteration': []}

@st.cache_data
def generate_test_data(name, res, bnd):
    x = np.linspace(-bnd, bnd, res)
    y = np.linspace(-bnd, bnd, res)
    X, Y = np.meshgrid(x, y)
    if name == "Quadratic":
        Z = X**2 + Y**2
    elif name == "Rastrigin":
        Z = 10 * 2 + (X**2 - 10*np.cos(2*np.pi*X)) + (Y**2 - 10*np.cos(2*np.pi*Y))
    elif name == "Rosenbrock":
        Z = (1 - X)**2 + 100 * (Y - X**2)**2
    elif name == "Himmelblau":
        Z = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2
    return X, Y, Z

X, Y, Z = generate_test_data(test_func, resolution, bounds)

col1, col2 = st.columns([1, 1])

with col1:
    st.header("📈 Траектория оптимизации")
    if optimizer == "SGD":
        opt = StochasticGradientDescent(learning_rate=learning_rate, momentum=momentum)
    elif optimizer == "GD":
        opt = GradientDescent(learning_rate=learning_rate)
    elif optimizer == "RMSProp":
        opt = RMSProp(learning_rate=learning_rate, momentum=momentum)
    elif optimizer == "AdamW":
        opt = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    elif optimizer == "Lion":
        opt = Lion(learning_rate=learning_rate, weight_decay=weight_decay)
    elif optimizer == "Adan":
        opt = Adan(learning_rate=learning_rate, weight_decay=weight_decay)
    elif optimizer == "MARS":
        opt = MARS(learning_rate=learning_rate, momentum=momentum)

    start = np.random.uniform(-bounds, bounds, size=(2,))
    params = [start.copy()]
    trajectory = [start.copy()]
    losses = []
    plot_placeholder = st.empty()

    for i in range(iterations):
        grad = None
        if test_func == "Quadratic":
            grad = 2 * params[0]
        elif test_func == "Rastrigin":
            x0, y0 = params[0]
            grad = np.array([2*x0 + 20*np.pi*np.sin(2*np.pi*x0), 2*y0 + 20*np.pi*np.sin(2*np.pi*y0)])
        elif test_func == "Rosenbrock":
            x0, y0 = params[0]
            grad = np.array([-2*(1-x0) - 400*x0*(y0-x0**2), 200*(y0-x0**2)])
        elif test_func == "Himmelblau":
            x0, y0 = params[0]
            grad = np.array([4*x0*(x0**2 + y0 - 11) + 2*(x0 + y0**2 - 7), 2*(x0**2 + y0 - 11) + 4*y0*(x0 + y0**2 - 7)])
        grads = [grad]

        updated = opt.update(params, grads)
        trajectory.append(updated[0].copy())
        loss = opt._loss(updated[0], test_func) if hasattr(opt, '_loss') else np.sum(updated[0]**2)
        losses.append(loss)
        params = updated

        if realtime_update or i == iterations-1:
            df = pd.DataFrame({
                'x': [p[0] for p in trajectory],
                'y': [p[1] for p in trajectory],
                'loss': losses if len(losses)==len(trajectory) else [None]+losses,
                'iter': list(range(len(trajectory)))
            })
            if show_3d:
                fig = go.Figure(data=[
                    go.Surface(x=X, y=Y, z=Z, opacity=0.7, showscale=False) if show_surface else None,
                    go.Scatter3d(x=df['x'], y=df['y'], z=df['loss'], mode='markers+lines', marker=dict(size=4), line=dict(width=2))
                ])
                fig.update_layout(title=f"{optimizer} на {test_func} (3D)")
            else:
                fig = px.contour(X, Y, Z, labels={'x':'X','y':'Y','z':'Z'}, title=f"{optimizer} на {test_func} (2D)")
                fig.add_scatter(x=df['x'], y=df['y'], mode='markers+lines')
            plot_placeholder.plotly_chart(fig, use_container_width=True)
            if realtime_update:
                time.sleep(0.05)

with col2:
    st.header("📊 Метрики оптимизации")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"**Финальный Loss:** {losses[-1]:.4f}")
    with m2:
        st.markdown(f"**Итерации:** {iterations}")
    with m3:
        st.markdown(f"**LR:** {learning_rate:.4f}")

    st.subheader("История Loss")
    df_loss = pd.DataFrame({'Loss': losses, 'Iteration': range(1, len(losses)+1)})
    fig_loss = px.line(df_loss, x='Iteration', y='Loss', title='Loss на лог. шкале', log_y=True)
    st.plotly_chart(fig_loss, use_container_width=True)

    st.subheader("Норма градиентов")
    grad_norms = [np.linalg.norm(opt._last_grad) if hasattr(opt, '_last_grad') else None for _ in losses]
    df_grad = pd.DataFrame({'GradNorm': grad_norms, 'Iteration': range(1, len(grad_norms)+1)})
    fig_grad = px.line(df_grad, x='Iteration', y='GradNorm', title='Норма градиентов')
    st.plotly_chart(fig_grad, use_container_width=True)

st.markdown("---")
st.header(f"ℹ️ О {test_func} и {optimizer}")
if test_func == "Quadratic":
    st.markdown("**Quadratic** — простая параболическая функция с единственным глобальным минимумом в (0,0).")
elif test_func == "Rastrigin":
    st.markdown("**Rastrigin** — мульти-модальная функция с глобальным минимумом в (0,0) и множеством локальных минимумов.")
elif test_func == "Rosenbrock":
    st.markdown("**Rosenbrock** — нелинейная функция-«долина», сложная для оптимизации из-за узкой изогнутой долины.")
elif test_func == "Himmelblau":
    st.markdown("**Himmelblau** — мульти-модальная функция с четырьмя идентичными локальными минимумами.")

if optimizer == "SGD":
    st.markdown("**SGD** — стохастический градиентный спуск с моментумом.")
elif optimizer == "AdamW":
    st.markdown("**AdamW** — Adam с decoupled weight decay.")
