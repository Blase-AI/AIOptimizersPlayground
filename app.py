import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import asyncio
from optimizers import StochasticGradientDescent, GradientDescent, RMSProp, AdamW, Lion, Adan, MARS

st.set_page_config(
    page_title="AI Optimizers Playground",
    page_icon="📊",
    layout="wide"
)

def set_theme(theme):
    if theme == "Темная":
        return """
        <style>
            .main {background-color: #1e1e1e; color: white;}
            .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px; padding: 8px 16px;}
            .metric-card {background: #2e2e2e; color: white; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
            .stSelectbox, .stSlider {margin-bottom: 20px;}
            .plot-container {margin-top: 30px;}
        </style>
        """
    return """
    <style>
        .main {background-color: #f8f9fa;}
        .stButton>button {border-radius: 8px; padding: 8px 16px;}
        .metric-card {border-radius: 10px; padding: 15px; background: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
        .stSelectbox, .stSlider {margin-bottom: 20px;}
        .plot-container {margin-top: 30px;}
    </style>
    """

st.title("🧠 AI Optimizers Playground")
st.markdown("Интерактивная площадка для сравнения алгоритмов оптимизации в машинном обучении")

with st.sidebar:
    st.header("⚙️ Настройки")
    theme = st.selectbox("Тема", ["Светлая", "Темная"], index=0)
    st.markdown(set_theme(theme), unsafe_allow_html=True)
    
    optimizers = st.multiselect(
        "Выберите оптимизаторы",
        ["SGD", "GD", "RMSProp", "AdamW", "Lion", "Adan", "MARS"],
        default=["AdamW"]
    )
    learning_rate = st.slider("Скорость обучения", 0.0001, 0.1, 0.001, format="%.4f")
    momentum = st.slider("Моментум (β)", 0.0, 1.0, 0.9)
    weight_decay = st.slider("Weight Decay", 0.0, 0.1, 0.01)
    iterations = st.slider("Количество итераций", 10, 500, 100)
    
    st.markdown("---")
    test_func = st.selectbox(
        "Выберите тестовую функцию",
        ["Quadratic", "Rastrigin", "Rosenbrock", "Himmelblau"]
    )
    resolution = st.slider("Разрешение сетки", 50, 300, 100)
    bounds = st.slider("Диапазон осей для X и Y", 1.0, 10.0, 5.0, step=0.5)
    
    st.markdown("---")
    show_surface = st.checkbox("Показать поверхность", value=True)
    show_3d = st.checkbox("3D визуализация", value=True)
    add_noise = st.checkbox("Добавить шум в градиенты", value=False)
    noise_level = st.slider("Уровень шума", 0.0, 1.0, 0.1) if add_noise else 0.0
    realtime_update = st.checkbox("Режим реального времени", value=True)
    
    if st.button("🔄 Сбросить и начать заново"):
        st.cache_data.clear()
        st.experimental_rerun()
    
    save_results = st.button("💾 Сохранить результаты")

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

test_functions = {
    "Quadratic": {
        "func": lambda p: np.sum(p**2),
        "grad": lambda p: 2 * p
    },
    "Rastrigin": {
        "func": lambda p: 10 * 2 + (p[0]**2 - 10*np.cos(2*np.pi*p[0])) + (p[1]**2 - 10*np.cos(2*np.pi*p[1])),
        "grad": lambda p: np.array([2*p[0] + 20*np.pi*np.sin(2*np.pi*p[0]), 2*p[1] + 20*np.pi*np.sin(2*np.pi*p[1])])
    },
    "Rosenbrock": {
        "func": lambda p: (1 - p[0])**2 + 100 * (p[1] - p[0]**2)**2,
        "grad": lambda p: np.array([-2*(1-p[0]) - 400*p[0]*(p[1]-p[0]**2), 200*(p[1]-p[0]**2)])
    },
    "Himmelblau": {
        "func": lambda p: (p[0]**2 + p[1] - 11)**2 + (p[0] + p[1]**2 - 7)**2,
        "grad": lambda p: np.array([4*p[0]*(p[0]**2 + p[1] - 11) + 2*(p[0] + p[1]**2 - 7), 
                                  2*(p[0]**2 + p[1] - 11) + 4*p[1]*(p[0] + p[1]**2 - 7)])
    }
}

@st.cache_resource
def init_optimizer(opt_name, lr, momentum, wd):
    if opt_name == "SGD":
        return StochasticGradientDescent(learning_rate=lr, momentum=momentum)
    elif opt_name == "GD":
        return GradientDescent(learning_rate=lr)
    elif opt_name == "RMSProp":
        return RMSProp(learning_rate=lr, momentum=momentum)
    elif opt_name == "AdamW":
        return AdamW(learning_rate=lr, weight_decay=wd)
    elif opt_name == "Lion":
        return Lion(learning_rate=lr, weight_decay=wd)
    elif opt_name == "Adan":
        return Adan(learning_rate=lr, weight_decay=wd)
    elif opt_name == "MARS":
        return MARS(learning_rate=lr, momentum=momentum)
    return None

async def run_optimization(opt, start_params, test_func, iterations, noise_level, bounds):
    params = [start_params.copy()]
    trajectory = [start_params.copy()]
    losses = [test_functions[test_func]["func"](start_params)]  # Начальное значение потерь
    grad_norms = []
    
    try:
        for i in range(iterations):
            grad = test_functions[test_func]["grad"](params[0])
            if add_noise:
                grad += np.random.normal(0, noise_level, size=grad.shape)
                
            if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
                st.error(f"Ошибка: Некорректные градиенты для {opt.__class__.__name__}")
                break
                
            updated = opt.update(params, [grad])
            # Ограничение параметров в пределах bounds
            updated[0] = np.clip(updated[0], -bounds, bounds)
            trajectory.append(updated[0].copy())
            loss = test_functions[test_func]["func"](updated[0])
            losses.append(loss)
            grad_norms.append(np.linalg.norm(grad) if grad is not None else 0)
            params = updated
            
            await asyncio.sleep(0)  # Для отзывчивости UI
        return trajectory, losses, grad_norms
    except Exception as e:
        st.error(f"Ошибка в оптимизации {opt.__class__.__name__}: {e}")
        return trajectory, losses, grad_norms

X, Y, Z = generate_test_data(test_func, resolution, bounds)
progress_bar = st.progress(0)
trajectories = {}
plot_placeholder = st.empty()

start = np.random.uniform(-bounds, bounds, size=(2,))
for idx, opt_name in enumerate(optimizers):
    opt = init_optimizer(opt_name, learning_rate, momentum, weight_decay)
    trajectory, losses, grad_norms = asyncio.run(
        run_optimization(opt, start, test_func, iterations, noise_level, bounds)
    )
    trajectories[opt_name] = {
        'traj': trajectory,
        'loss': losses,
        'grad_norms': grad_norms
    }
    progress_bar.progress((idx + 1) / len(optimizers))

fig = go.Figure()
if show_3d:
    if show_surface:
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, opacity=0.7, showscale=False, showlegend=False))
    for opt_name, data in trajectories.items():
        min_len = min(len(data['traj']), len(data['loss']))
        if min_len == 0:
            st.warning(f"Нет данных для {opt_name}")
            continue
        df = pd.DataFrame({
            'x': [p[0] for p in data['traj'][:min_len]],
            'y': [p[1] for p in data['traj'][:min_len]],
            'loss': data['loss'][:min_len]
        })
        fig.add_trace(go.Scatter3d(
            x=df['x'], y=df['y'], z=df['loss'],
            mode='markers+lines', name=opt_name,
            marker=dict(size=4), line=dict(width=2),
            showlegend=True
        ))
    fig.update_layout(title=f"Сравнение оптимизаторов на {test_func} (3D)")
else:
    fig.add_trace(go.Contour(
        x=X[0, :], y=Y[:, 0], z=Z,
        colorscale='Viridis',
        showscale=True,
        showlegend=False  # Отключаем легенду для контура
    ))
    for opt_name, data in trajectories.items():
        min_len = min(len(data['traj']), len(data['loss']))
        if min_len == 0:
            st.warning(f"Нет данных для {opt_name}")
            continue
        df = pd.DataFrame({
            'x': [p[0] for p in data['traj'][:min_len]],
            'y': [p[1] for p in data['traj'][:min_len]]
        })
        fig.add_trace(go.Scatter(
            x=df['x'], y=df['y'],
            mode='markers+lines', name=opt_name,
            marker=dict(size=8), line=dict(width=2),
            showlegend=True
        ))
    fig.update_layout(title=f"Сравнение оптимизаторов на {test_func} (2D)")

frames = []
max_traj_len = max(len(data['traj']) for data in trajectories.values()) if trajectories else 1
for i in range(1, max_traj_len):
    frame_data = []
    if show_3d and show_surface:
        frame_data.append(go.Surface(x=X, y=Y, z=Z, opacity=0.7, showlegend=False))
    for opt_name, data in trajectories.items():
        if i <= len(data['traj']):
            min_len = min(len(data['traj'][:i]), len(data['loss']))
            if min_len == 0:
                continue
            df = pd.DataFrame({
                'x': [p[0] for p in data['traj'][:min_len]],
                'y': [p[1] for p in data['traj'][:min_len]],
                'loss': data['loss'][:min_len]
            })
            if show_3d:
                frame_data.append(go.Scatter3d(
                    x=df['x'], y=df['y'], z=df['loss'],
                    mode='markers+lines', name=opt_name,
                    marker=dict(size=4), line=dict(width=2),
                    showlegend=True
                ))
            else:
                frame_data.append(go.Scatter(
                    x=df['x'], y=df['y'],
                    mode='markers+lines', name=opt_name,
                    marker=dict(size=8), line=dict(width=2),
                    showlegend=True
                ))
    frames.append(go.Frame(data=frame_data, name=str(i)))

fig.update(frames=frames)
fig.update_layout(
    updatemenus=[dict(
        type="buttons",
        buttons=[dict(label="Play", method="animate", args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}])]
    )],
    showlegend=True
)


if realtime_update or not trajectories:
    plot_placeholder.plotly_chart(fig, use_container_width=True)


st.header("📊 Метрики оптимизации")
cols = st.columns(len(optimizers))
for idx, opt_name in enumerate(optimizers):
    with cols[idx]:
        st.markdown(f"**{opt_name}**")
        if trajectories[opt_name]['loss']:
            st.markdown(f"Финальный Loss: {trajectories[opt_name]['loss'][-1]:.4f}")
        else:
            st.markdown("Финальный Loss: N/A")
        st.markdown(f"Итерации: {len(trajectories[opt_name]['loss']) - 1}")  # -1, т.к. начальное значение
        st.markdown(f"LR: {learning_rate:.4f}")

st.subheader("История Loss")
loss_fig = go.Figure()
for opt_name, data in trajectories.items():
    if data['loss']:
        df_loss = pd.DataFrame({'Loss': data['loss'], 'Iteration': range(len(data['loss']))})
        loss_fig.add_trace(go.Scatter(x=df_loss['Iteration'], y=df_loss['Loss'], mode='lines', name=opt_name))
loss_fig.update_layout(title='Loss на лог. шкале', yaxis_type="log")
st.plotly_chart(loss_fig, use_container_width=True)

st.subheader("Норма градиентов")
grad_fig = go.Figure()
for opt_name, data in trajectories.items():
    if data['grad_norms']:
        df_grad = pd.DataFrame({'GradNorm': data['grad_norms'], 'Iteration': range(1, len(data['grad_norms']) + 1)})
        grad_fig.add_trace(go.Scatter(x=df_grad['Iteration'], y=df_grad['GradNorm'], mode='lines', name=opt_name))
grad_fig.update_layout(title='Норма градиентов')
st.plotly_chart(grad_fig, use_container_width=True)

if save_results:
    for opt_name, data in trajectories.items():
        min_len = min(len(data['traj']), len(data['loss']))
        if min_len == 0:
            st.warning(f"Нет данных для сохранения для {opt_name}")
            continue
        df = pd.DataFrame({
            'x': [p[0] for p in data['traj'][:min_len]],
            'y': [p[1] for p in data['traj'][:min_len]],
            'loss': data['loss'][:min_len],
            'grad_norm': data['grad_norms'][:min_len] if data['grad_norms'] else [0] * min_len
        })
        df.to_csv(f"{opt_name}_{test_func}_results.csv", index=False)
    st.success("Результаты сохранены!")

st.markdown("---")
st.header(f"ℹ️ О {test_func} и оптимизаторах")

# Описание тестовой функции
st.subheader("Тестовая функция")
if test_func == "Quadratic":
    st.markdown("""
    **Quadratic** — простая параболическая функция f(x, y) = x² + y² с единственным глобальным минимумом в (0,0).
    """)
elif test_func == "Rastrigin":
    st.markdown("""
    **Rastrigin** — мульти-модальная функция f(x, y) = 20 + x² + y² - 10(cos(2πx) + cos(2πy)) с глобальным минимумом в (0,0) и множеством локальных минимумов.
    """)
elif test_func == "Rosenbrock":
    st.markdown("""
    **Rosenbrock** — нелинейная функция f(x, y) = (1 - x)² + 100(y - x²)², сложная из-за узкой изогнутой долины.
    """)
elif test_func == "Himmelblau":
    st.markdown("""
    **Himmelblau** — мульти-модальная функция f(x, y) = (x² + y - 11)² + (x + y² - 7)² с четырьмя локальными минимумами.
    """)

st.subheader("Оптимизаторы")
for opt_name in optimizers:
    if opt_name == "SGD":
        with st.expander("SGD — Стохастический градиентный спуск с моментумом"):
            st.markdown("""
            **SGD** обновляет параметры модели, используя градиенты на небольших подвыборках данных. Моментум сглаживает изменения, ускоряя обучение.
            
            **Особенности**:
            - Прост и эффективен для больших наборов данных.
            - Требует тщательной настройки скорости обучения.
            - Подходит для задач компьютерного зрения и обработки текстов.
            
            """)
    elif opt_name == "GD":
        with st.expander("GD — Классический градиентный спуск"):
            st.markdown("""
            **GD** использует градиенты, вычисленные на всём наборе данных, для обновления параметров модели.
            
            **Особенности**:
            - Надёжен для простых функций с одним минимумом.
            - Медленный на больших данных из-за полного вычисления градиентов.
            - Используется в небольших задачах или для теоретического анализа.
            
            """)
    elif opt_name == "RMSProp":
        with st.expander("RMSProp — Адаптивный метод с экспоненциальным сглаживанием"):
            st.markdown("""
            **RMSProp** автоматически подстраивает скорость обучения для каждого параметра, используя среднее квадратов градиентов.
            
            **Особенности**:
            - Хорошо работает с неравномерными градиентами.
            - Эффективен для рекуррентных нейронных сетей.
            - Требует настройки дополнительных параметров.
            
            """)
    elif opt_name == "AdamW":
        with st.expander("AdamW — Adam с регуляризацией весов"):
            st.markdown("""
            **AdamW** сочетает адаптивное обучение с регуляризацией, улучшая способность модели обобщать данные.
            
            **Особенности**:
            - Быстро сходится и устойчив к шуму.
            - Идеален для глубоких нейронных сетей, таких как CNN и Transformer.
            - Использует больше памяти, чем SGD.
            
            """)
    elif opt_name == "Lion":
        with st.expander("Lion — Современный оптимизатор с высокой эффективностью"):
            st.markdown("""
            **Lion** обновляет параметры, используя только направление градиента, что снижает вычислительные затраты.
            
            **Особенности**:
            - Прост в реализации и эффективен.
            - Подходит для больших моделей, таких как Transformer.
            - Менее изучен, чем Adam.
            
            """)
    elif opt_name == "Adan":
        with st.expander("Adan — Адаптивный метод для ускоренного обучения"):
            st.markdown("""
            **Adan** ускоряет обучение, комбинируя адаптивные обновления с предсказанием изменений градиентов.
            
            **Особенности**:
            - Быстрее Adam на больших моделях.
            - Устойчив к шумным данным.
            - Требует настройки нескольких параметров.
            
            """)
    elif opt_name == "MARS":
        with st.expander("MARS — кастомный оптимизатор"):
            st.markdown("""
            **MARS** — уникальный оптимизатор, созданный для специфических задач машинного обучения.
            
            **Особенности**:
            - Настраивается под ваши нужды.
            - Может быть лучше стандартных методов в определённых случаях.
            - Требует тестирования для проверки эффективности.

            """)
