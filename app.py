import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import asyncio
import json
import logging
from optimizers import StochasticGradientDescent, GradientDescent, RMSProp, AdamW, Lion, Adan, MARS, AMSGrad, Adagrad, Adam, Sophia, LARS

logging.basicConfig(level=logging.INFO, filename="optimizer_debug.log")
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="🧪AI Optimizers Playground",
    page_icon="🧪",
    layout="wide"
)

st.markdown("""
<style>
    .main {background-color: #f9fafb; color: #333;}
    .stButton>button {
        background-color: #0288d1; color: white; border: none; border-radius: 6px; 
        padding: 8px 16px; transition: background-color 0.2s;
    }
    .stButton>button:hover {background-color: #0277bd;}
    .metric-card {
        background: white; color: #333; border-radius: 8px; 
        padding: 15px; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stSelectbox, .stSlider, .stNumberInput {margin-bottom: 15px;}
    .st-expander {background: #fff; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.1);}
    .plot-container {margin-top: 20px;}
    h1, h2, h3 {color: #333; font-family: 'Helvetica Neue', sans-serif;}
</style>
""", unsafe_allow_html=True)

st.title("AI Optimizers Playground")
st.markdown("Интерактивная площадка для сравнения алгоритмов оптимизации", unsafe_allow_html=True)

if 'trajectories' not in st.session_state:
    st.session_state.trajectories = {}
if 'run_simulation' not in st.session_state:
    st.session_state.run_simulation = False
if 'learning_rate' not in st.session_state:
    st.session_state.learning_rate = 0.001
if 'momentum' not in st.session_state:
    st.session_state.momentum = 0.9
if 'weight_decay' not in st.session_state:
    st.session_state.weight_decay = 0.01
if 'iterations' not in st.session_state:
    st.session_state.iterations = 100
if 'resolution' not in st.session_state:
    st.session_state.resolution = 100
if 'bounds' not in st.session_state:
    st.session_state.bounds = 5.0
if 'noise_level' not in st.session_state:
    st.session_state.noise_level = 0.1
if 'color_scheme' not in st.session_state:
    st.session_state.color_scheme = 'Viridis'

with st.sidebar:
    with st.expander("Глобальные параметры", expanded=True):
        lr_col1, lr_col2 = st.columns([1, 1])
        with lr_col1:
            st.session_state.learning_rate = st.number_input(
                "Скорость обучения", min_value=0.0001, max_value=0.1, value=st.session_state.learning_rate, 
                step=0.0001, format="%.4f", key="learning_rate_input"
            )
        with lr_col2:
            st.session_state.learning_rate = st.slider(
                "", 0.0001, 0.1, st.session_state.learning_rate, step=0.0001, format="%.4f", 
                label_visibility="collapsed", key="learning_rate_slider"
            )
        mom_col1, mom_col2 = st.columns([1, 1])
        with mom_col1:
            st.session_state.momentum = st.number_input(
                "Моментум (β)", min_value=0.0, max_value=1.0, value=st.session_state.momentum, 
                step=0.01, format="%.2f", key="momentum_input"
            )
        with mom_col2:
            st.session_state.momentum = st.slider(
                "", 0.0, 1.0, st.session_state.momentum, step=0.01, format="%.2f", 
                label_visibility="collapsed", key="momentum_slider"
            )
        wd_col1, wd_col2 = st.columns([1, 1])
        with wd_col1:
            st.session_state.weight_decay = st.number_input(
                "Weight Decay", min_value=0.0, max_value=0.1, value=st.session_state.weight_decay, 
                step=0.01, format="%.2f", key="weight_decay_input"
            )
        with wd_col2:
            st.session_state.weight_decay = st.slider(
                "", 0.0, 0.1, st.session_state.weight_decay, step=0.01, format="%.2f", 
                label_visibility="collapsed", key="weight_decay_slider"
            )

    st.markdown("---")
    with st.expander("Параметры оптимизаторов"):
        optimizers = st.multiselect(
            "Выберите оптимизаторы",
            ["SGD", "GD", "RMSProp", "AMSGrad", "Adagrad", "Adam", "AdamW", "Sophia", "Lion", "Adan", "MARS", "LARS"],
            default=["AdamW"],
            key="optimizers_select"
        )
        optimizer_params = {
            "AdamW": {"beta1": (0.0, 1.0, 0.9, 0.01, "%.2f"), "beta2": (0.0, 1.0, 0.999, 0.001, "%.3f")},
            "AMSGrad": {"beta1": (0.0, 1.0, 0.9, 0.01, "%.2f"), "beta2": (0.0, 1.0, 0.999, 0.001, "%.3f")},
            "Adam": {"beta1": (0.0, 1.0, 0.9, 0.01, "%.2f"), "beta2": (0.0, 1.0, 0.999, 0.001, "%.3f")},
            "Sophia": {"beta1": (0.0, 1.0, 0.9, 0.01, "%.2f"), "beta2": (0.0, 1.0, 0.999, 0.001, "%.3f")},
            "Lion": {"beta": (0.0, 1.0, 0.9, 0.01, "%.2f")},
            "LARS": {"trust_coeff": (0.0001, 0.01, 0.001, 0.0001, "%.4f")}
        }
        params = {}
        for opt in optimizers:
            if opt in optimizer_params:
                st.markdown(f"**{opt}**")
                for param, (min_val, max_val, default, step, fmt) in optimizer_params[opt].items():
                    param_key = f"{opt}_{param}"
                    if param_key not in st.session_state:
                        st.session_state[param_key] = default
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.session_state[param_key] = st.number_input(
                            param, min_value=min_val, max_value=max_val, value=st.session_state[param_key], 
                            step=step, format=fmt, key=f"{param_key}_input"
                        )
                    with col2:
                        st.session_state[param_key] = st.slider(
                            "", min_val, max_val, st.session_state[param_key], step=step, format=fmt, 
                            label_visibility="collapsed", key=f"{param_key}_slider"
                        )
                    params[param_key] = st.session_state[param_key]

    st.markdown("---")
    with st.expander("Настройки симуляции"):
        iter_col1, iter_col2 = st.columns([1, 1])
        with iter_col1:
            st.session_state.iterations = st.number_input(
                "Количество итераций", min_value=10, max_value=500, value=st.session_state.iterations, 
                step=1, format="%d", key="iterations_input"
            )
        with iter_col2:
            st.session_state.iterations = st.slider(
                "", 10, 500, st.session_state.iterations, step=1, format="%d", 
                label_visibility="collapsed", key="iterations_slider"
            )
        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.session_state.resolution = st.number_input(
                "Разрешение сетки", min_value=50, max_value=300, value=st.session_state.resolution, 
                step=1, format="%d", key="resolution_input"
            )
        with res_col2:
            st.session_state.resolution = st.slider(
                "", 50, 300, st.session_state.resolution, step=1, format="%d", 
                label_visibility="collapsed", key="resolution_slider"
            )
        bnd_col1, bnd_col2 = st.columns([1, 1])
        with bnd_col1:
            st.session_state.bounds = st.number_input(
                "Диапазон осей (X, Y)", min_value=1.0, max_value=500.0, value=st.session_state.bounds, 
                step=0.5, format="%.1f", key="bounds_input"
            )
        with bnd_col2:
            st.session_state.bounds = st.slider(
                "", 1.0, 500.0, st.session_state.bounds, step=0.5, format="%.1f", 
                label_visibility="collapsed", key="bounds_slider"
            )
        test_func = st.selectbox(
            "Тестовая функция", [
                "Quadratic", "Rastrigin", "Rosenbrock", "Himmelblau", 
                "Ackley", "Griewank", "Schwefel", "Levy", "Beale"
            ], 
            key="test_func_select"
        )

    st.markdown("---")
    with st.expander("Визуализация"):
        show_surface = st.checkbox("Показать поверхность", value=True, key="show_surface_cb")
        show_3d = st.checkbox("3D визуализация", value=True, key="show_3d_cb")
        show_colorbar = st.checkbox("Показать шкалу цветов", value=True, key="show_colorbar_cb")
        st.session_state.color_scheme = st.selectbox(
            "Цветовая палитра", ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Plotly"],
            index=["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Plotly"].index(st.session_state.color_scheme),
            key="color_scheme_select"
        )
        add_noise = st.checkbox("Добавить шум в градиенты", value=False, key="add_noise_cb")
        if add_noise:
            noise_col1, noise_col2 = st.columns([1, 1])
            with noise_col1:
                st.session_state.noise_level = st.number_input(
                    "Уровень шума", min_value=0.0, max_value=1.0, value=st.session_state.noise_level, 
                    step=0.01, format="%.2f", key="noise_level_input"
                )
            with noise_col2:
                st.session_state.noise_level = st.slider(
                    "", 0.0, 1.0, st.session_state.noise_level, step=0.01, format="%.2f", 
                    label_visibility="collapsed", key="noise_level_slider"
                )
        else:
            st.session_state.noise_level = 0.0
        realtime_update = st.checkbox("Режим реального времени", value=True, key="realtime_update_cb")

    st.markdown("---")
    if st.button("🚀 Запустить симуляцию", key="run_simulation_btn"):
        st.session_state.run_simulation = True
        st.session_state.trajectories = {}
        try:
            st.rerun()
        except Exception as e:
            st.error(f"Ошибка при перезапуске: {e}")
    
    if st.button("🔄 Сбросить", key="reset_btn"):
        st.session_state.trajectories = {}
        st.session_state.run_simulation = False
        st.cache_data.clear()
        try:
            st.rerun()
        except Exception as e:
            st.error(f"Ошибка при перезапуске: {e}")
    
    #save_results = st.button("💾 Сохранить результаты", key="save_results_btn")

tab1, tab2, tab3, tab4 = st.tabs(["Визуализация", "Метрики", "Описание", "Руководство"])

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
    elif name == "Ackley":
        Z = -20 * np.exp(-0.2 * np.sqrt(0.5 * (X**2 + Y**2))) - \
            np.exp(0.5 * (np.cos(2*np.pi*X) + np.cos(2*np.pi*Y))) + np.e + 20
    elif name == "Griewank":
        Z = (X**2 + Y**2) / 4000 - np.cos(X) * np.cos(Y / np.sqrt(2)) + 1
    elif name == "Schwefel":
        Z = 418.9829 * 2 - X * np.sin(np.sqrt(np.abs(X))) - Y * np.sin(np.sqrt(np.abs(Y)))
    elif name == "Levy":
        w_x = 1 + (X - 1) / 4
        w_y = 1 + (Y - 1) / 4
        Z = np.sin(np.pi * w_x)**2 + \
            (w_x - 1)**2 * (1 + 10 * np.sin(np.pi * w_x + 1)**2) + \
            (w_y - 1)**2 * (1 + np.sin(2 * np.pi * w_y)**2)
    elif name == "Beale":
        Z = (1.5 - X + X*Y)**2 + (2.25 - X + X*Y**2)**2 + (2.625 - X + X*Y**3)**2
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
    },
    "Ackley": {
        "func": lambda p: -20 * np.exp(-0.2 * np.sqrt(0.5 * (p[0]**2 + p[1]**2))) - \
                         np.exp(0.5 * (np.cos(2*np.pi*p[0]) + np.cos(2*np.pi*p[1]))) + np.e + 20,
        "grad": lambda p: np.array([
            2 * np.exp(-0.2 * np.sqrt(0.5 * (p[0]**2 + p[1]**2))) * p[0] / np.sqrt(0.5 * (p[0]**2 + p[1]**2)) + \
            np.pi * np.sin(2*np.pi*p[0]) * np.exp(0.5 * (np.cos(2*np.pi*p[0]) + np.cos(2*np.pi*p[1]))),
            2 * np.exp(-0.2 * np.sqrt(0.5 * (p[0]**2 + p[1]**2))) * p[1] / np.sqrt(0.5 * (p[0]**2 + p[1]**2)) + \
            np.pi * np.sin(2*np.pi*p[1]) * np.exp(0.5 * (np.cos(2*np.pi*p[0]) + np.cos(2*np.pi*p[1])))
        ])
    },
    "Griewank": {
        "func": lambda p: (p[0]**2 + p[1]**2) / 4000 - np.cos(p[0]) * np.cos(p[1] / np.sqrt(2)) + 1,
        "grad": lambda p: np.array([
            p[0] / 2000 + np.sin(p[0]) * np.cos(p[1] / np.sqrt(2)),
            p[1] / 2000 + np.cos(p[0]) * np.sin(p[1] / np.sqrt(2)) / np.sqrt(2)
        ])
    },
    "Schwefel": {
        "func": lambda p: 418.9829 * 2 - p[0] * np.sin(np.sqrt(np.abs(p[0]))) - p[1] * np.sin(np.sqrt(np.abs(p[1]))),
        "grad": lambda p: np.array([
            -np.sin(np.sqrt(np.abs(p[0]))) - (p[0] * np.cos(np.sqrt(np.abs(p[0])))) / (2 * np.sqrt(np.abs(p[0]))) if p[0] != 0 else 0,
            -np.sin(np.sqrt(np.abs(p[1]))) - (p[1] * np.cos(np.sqrt(np.abs(p[1])))) / (2 * np.sqrt(np.abs(p[1]))) if p[1] != 0 else 0
        ])
    },
    "Levy": {
        "func": lambda p: np.sin(np.pi * (1 + (p[0] - 1) / 4))**2 + \
                         ((1 + (p[0] - 1) / 4) - 1)**2 * (1 + 10 * np.sin(np.pi * (1 + (p[0] - 1) / 4) + 1)**2) + \
                         ((1 + (p[1] - 1) / 4) - 1)**2 * (1 + np.sin(2 * np.pi * (1 + (p[1] - 1) / 4))**2),
        "grad": lambda p: np.array([
            np.pi * np.cos(np.pi * (1 + (p[0] - 1) / 4)) * np.sin(np.pi * (1 + (p[0] - 1) / 4)) / 2 + \
            (p[0] - 5) / 8 * (1 + 10 * np.sin(np.pi * (1 + (p[0] - 1) / 4) + 1)**2) + \
            5 * np.pi * (p[0] - 5)**2 / 16 * np.cos(np.pi * (1 + (p[0] - 1) / 4) + 1) * \
            np.sin(np.pi * (1 + (p[0] - 1) / 4) + 1),
            (p[1] - 5) / 8 * (1 + np.sin(2 * np.pi * (1 + (p[1] - 1) / 4))**2) + \
            np.pi * (p[1] - 5)**2 / 8 * np.cos(2 * np.pi * (1 + (p[1] - 1) / 4)) * \
            np.sin(2 * np.pi * (1 + (p[1] - 1) / 4))
        ])
    },
    "Beale": {
        "func": lambda p: (1.5 - p[0] + p[0]*p[1])**2 + (2.25 - p[0] + p[0]*p[1]**2)**2 + (2.625 - p[0] + p[0]*p[1]**3)**2,
        "grad": lambda p: np.array([
            2 * (1.5 - p[0] + p[0]*p[1]) * (-1 + p[1]) + \
            2 * (2.25 - p[0] + p[0]*p[1]**2) * (-1 + p[1]**2) + \
            2 * (2.625 - p[0] + p[0]*p[1]**3) * (-1 + p[1]**3),
            2 * (1.5 - p[0] + p[0]*p[1]) * p[0] + \
            2 * (2.25 - p[0] + p[0]*p[1]**2) * (2*p[0]*p[1]) + \
            2 * (2.625 - p[0] + p[0]*p[1]**3) * (3*p[0]*p[1]**2)
        ])
    }
}

@st.cache_resource
def init_optimizer(opt_name, lr, momentum, wd, params):
    if opt_name == "SGD":
        return StochasticGradientDescent(learning_rate=lr, momentum=momentum)
    elif opt_name == "GD":
        return GradientDescent(learning_rate=lr)
    elif opt_name == "RMSProp":
        return RMSProp(learning_rate=lr, momentum=momentum)
    elif opt_name == "AMSGrad":
        return AMSGrad(learning_rate=lr, beta1=params.get(f"{opt_name}_beta1", 0.9), beta2=params.get(f"{opt_name}_beta2", 0.999))
    elif opt_name == "Adagrad":
        return Adagrad(learning_rate=lr)
    elif opt_name == "Adam":
        return Adam(learning_rate=lr, beta1=params.get(f"{opt_name}_beta1", 0.9), beta2=params.get(f"{opt_name}_beta2", 0.999))
    elif opt_name == "AdamW":
        return AdamW(learning_rate=lr, beta1=params.get(f"{opt_name}_beta1", 0.9), beta2=params.get(f"{opt_name}_beta2", 0.999), weight_decay=wd)
    elif opt_name == "Sophia":
        return Sophia(learning_rate=lr, beta1=params.get(f"{opt_name}_beta1", 0.9), beta2=params.get(f"{opt_name}_beta2", 0.999))
    elif opt_name == "Lion":
        return Lion(learning_rate=lr, beta=params.get(f"{opt_name}_beta", 0.9), weight_decay=wd)
    elif opt_name == "Adan":
        return Adan(learning_rate=lr, weight_decay=wd)
    elif opt_name == "MARS":
        return MARS(learning_rate=lr, momentum=momentum)
    elif opt_name == "LARS":
        return LARS(learning_rate=lr, momentum=momentum, trust_coeff=params.get(f"{opt_name}_trust_coeff", 0.001), weight_decay=wd)
    return None

async def run_optimization(opt, start_params, test_func, iterations, noise_level, bounds):
    params = [start_params.copy()]
    trajectory = [start_params.copy()]
    losses = [test_functions[test_func]["func"](start_params)]
    grad_norms = []
    local_lrs = [] if opt.__class__.__name__ == "LARS" else None
    
    try:
        for i in range(iterations):
            grad = test_functions[test_func]["grad"](params[0])
            if add_noise:
                grad += np.random.normal(0, noise_level, size=grad.shape)
                
            if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
                st.error(f"Ошибка: Некорректные градиенты для {opt.__class__.__name__}")
                break
                
            updated = opt.step(params, [grad])
            updated[0] = np.clip(updated[0], -bounds, bounds)
            trajectory.append(updated[0].copy())
            loss = test_functions[test_func]["func"](updated[0])
            losses.append(loss)
            grad_norms.append(np.linalg.norm(grad) if grad is not None else 0)
            params = updated
            
            logger.info(f"{opt.__class__.__name__} | Iter {i+1} | Loss: {loss:.4f} | Grad Norm: {grad_norms[-1]:.4f}")
            if opt.__class__.__name__ == "LARS":
                param_norm = np.linalg.norm(params[0])
                grad_norm = np.linalg.norm(grad)
                local_lr = opt.trust_coeff * param_norm / (grad_norm + opt.weight_decay * param_norm + 1e-6) if param_norm > 0 and grad_norm > 0 else 1.0
                local_lrs.append(local_lr)
                logger.info(f"LARS | local_lr: {local_lr:.6f}")
            
            await asyncio.sleep(0)
        return trajectory, losses, grad_norms, local_lrs
    except Exception as e:
        st.error(f"Ошибка в оптимизации {opt.__class__.__name__}: {e}")
        return trajectory, losses, grad_norms, local_lrs

X, Y, Z = generate_test_data(test_func, st.session_state.resolution, st.session_state.bounds)

progress_bar = st.progress(0)
if st.session_state.run_simulation and optimizers:
    start = np.random.uniform(-st.session_state.bounds, st.session_state.bounds, size=(2,))
    for idx, opt_name in enumerate(optimizers):
        if opt_name not in st.session_state.trajectories:
            opt = init_optimizer(opt_name, st.session_state.learning_rate, st.session_state.momentum, st.session_state.weight_decay, params)
            trajectory, losses, grad_norms, local_lrs = asyncio.run(
                run_optimization(opt, start, test_func, st.session_state.iterations, st.session_state.noise_level, st.session_state.bounds)
            )
            st.session_state.trajectories[opt_name] = {
                'traj': trajectory,
                'loss': losses,
                'grad_norms': grad_norms,
                'local_lrs': local_lrs
            }
        progress_bar.progress((idx + 1) / len(optimizers))

with tab1:
    if not st.session_state.run_simulation and not st.session_state.trajectories:
        st.info("Настройте параметры и нажмите 'Запустить симуляцию'.")
    elif st.session_state.run_simulation and not st.session_state.trajectories:
        st.info("Симуляция выполняется...")
    elif st.session_state.trajectories:
        st.success("Симуляция завершена.")

    if st.session_state.trajectories:
        fig = go.Figure()
        opt_colors = px.colors.qualitative.Plotly
        if len(optimizers) > len(opt_colors):
            opt_colors = opt_colors * (len(optimizers) // len(opt_colors) + 1)

        if show_3d:
            if show_surface:
                fig.add_trace(go.Surface(
                    x=X, y=Y, z=Z, opacity=0.7, 
                    colorscale=st.session_state.color_scheme, 
                    showscale=show_colorbar,
                    showlegend=False
                ))
            for idx, (opt_name, data) in enumerate(st.session_state.trajectories.items()):
                min_len = min(len(data['traj']), len(data['loss']))
                if min_len == 0:
                    st.warning(f"Нет данных для {opt_name}")
                    continue
                df = pd.DataFrame({
                    'x': [p[0] for p in data['traj'][:min_len]],
                    'y': [p[1] for p in data['traj'][:min_len]],
                    'loss': data['loss'][:min_len],
                    'iteration': list(range(min_len))
                })
                fig.add_trace(go.Scatter3d(
                    x=df['x'], y=df['y'], z=df['loss'],
                    mode='markers+lines', name=opt_name,
                    marker=dict(size=4, color=opt_colors[idx]), 
                    line=dict(width=2, color=opt_colors[idx]),
                    showlegend=True,
                    customdata=df[['iteration', 'loss']],
                    hovertemplate="Итерация: %{customdata[0]}<br>Loss: %{customdata[1]:.4f}<br>x: %{x:.2f}<br>y: %{y:.2f}"
                ))
            fig.update_layout(title=f"Сравнение оптимизаторов на {test_func} (3D)")
        else:
            fig.add_trace(go.Contour(
                x=X[0, :], y=Y[:, 0], z=Z,
                colorscale=st.session_state.color_scheme,
                showscale=show_colorbar,
                showlegend=False,
                colorbar=dict(
                    x=1.02,  
                    len=0.8,  
                    thickness=15
                )
            ))
            for idx, (opt_name, data) in enumerate(st.session_state.trajectories.items()):
                min_len = min(len(data['traj']), len(data['loss']))
                if min_len == 0:
                    st.warning(f"Нет данных для {opt_name}")
                    continue
                df = pd.DataFrame({
                    'x': [p[0] for p in data['traj'][:min_len]],
                    'y': [p[1] for p in data['traj'][:min_len]],
                    'iteration': list(range(min_len))
                })
                fig.add_trace(go.Scatter(
                    x=df['x'], y=df['y'],
                    mode='markers+lines', name=opt_name,
                    marker=dict(size=8, color=opt_colors[idx]),
                    line=dict(width=2, color=opt_colors[idx]),
                    showlegend=True,
                    customdata=df[['iteration']],
                    hovertemplate="Итерация: %{customdata[0]}<br>x: %{x:.2f}<br>y: %{y:.2f}"
                ))
            fig.update_layout(
                title=f"Сравнение оптимизаторов на {test_func} (2D)",
                margin=dict(r=100)  
            )

        frames = []
        max_traj_len = max(len(data['traj']) for data in st.session_state.trajectories.values()) if st.session_state.trajectories else 1
        frame_step = max(1, max_traj_len // 50)
        for i in range(1, max_traj_len, frame_step):
            frame_data = []
            if show_3d and show_surface:
                frame_data.append(go.Surface(
                    x=X, y=Y, z=Z, opacity=0.7, 
                    colorscale=st.session_state.color_scheme, 
                    showlegend=False
                ))
            for idx, (opt_name, data) in enumerate(st.session_state.trajectories.items()):
                if i <= len(data['traj']):
                    min_len = min(len(data['traj'][:i]), len(data['loss']))
                    if min_len == 0:
                        continue
                    df = pd.DataFrame({
                        'x': [p[0] for p in data['traj'][:min_len]],
                        'y': [p[1] for p in data['traj'][:min_len]],
                        'loss': data['loss'][:min_len],
                        'iteration': list(range(min_len))
                    })
                    if show_3d:
                        frame_data.append(go.Scatter3d(
                            x=df['x'], y=df['y'], z=df['loss'],
                            mode='markers+lines', name=opt_name,
                            marker=dict(size=4, color=opt_colors[idx]),
                            line=dict(width=2, color=opt_colors[idx]),
                            showlegend=True
                        ))
                    else:
                        frame_data.append(go.Scatter(
                            x=df['x'], y=df['y'],
                            mode='markers+lines', name=opt_name,
                            marker=dict(size=8, color=opt_colors[idx]),
                            line=dict(width=2, color=opt_colors[idx]),
                            showlegend=True
                        ))
            frames.append(go.Frame(data=frame_data, name=str(i)))

        fig.update(frames=frames)
        fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play", method="animate", args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}])]
            )],
            showlegend=True,
            margin=dict(l=20, r=100, t=50, b=20)
        )

        if realtime_update or not st.session_state.trajectories:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Запустите симуляцию, чтобы увидеть визуализацию.")

with tab2:
    if st.session_state.trajectories:
        st.markdown("### Метрики оптимизации")
        metrics_data = []
        for opt_name, data in st.session_state.trajectories.items():
            if data['loss']:
                final_loss = data['loss'][-1]
                iterations_done = len(data['loss']) - 1
                avg_grad_norm = np.mean(data['grad_norms']) if data['grad_norms'] else 0
                avg_local_lr = np.mean(data['local_lrs']) if data['local_lrs'] else None
                metric = {
                    "Оптимизатор": opt_name,
                    "Финальный Loss": f"{final_loss:.4f}",
                    "Итерации": iterations_done,
                    "Средняя норма градиентов": f"{avg_grad_norm:.4f}"
                }
                if avg_local_lr is not None:
                    metric["Средний local_lr (LARS)"] = f"{avg_local_lr:.6f}"
                metrics_data.append(metric)
        if metrics_data:
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
        else:
            st.warning("Нет данных для отображения метрик")

        st.markdown("### История Loss")
        loss_fig = go.Figure()
        for idx, (opt_name, data) in enumerate(st.session_state.trajectories.items()):
            if data['loss']:
                df_loss = pd.DataFrame({'Loss': data['loss'], 'Iteration': range(len(data['loss']))})
                loss_fig.add_trace(go.Scatter(
                    x=df_loss['Iteration'], y=df_loss['Loss'], 
                    mode='lines', name=opt_name,
                    line=dict(color=opt_colors[idx])
                ))
        loss_fig.update_layout(title='Loss (лог. шкала)', yaxis_type="log", margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(loss_fig, use_container_width=True)

        st.markdown("### Норма градиентов")
        grad_fig = go.Figure()
        for idx, (opt_name, data) in enumerate(st.session_state.trajectories.items()):
            if data['grad_norms']:
                df_grad = pd.DataFrame({'GradNorm': data['grad_norms'], 'Iteration': range(1, len(data['grad_norms']) + 1)})
                grad_fig.add_trace(go.Scatter(
                    x=df_grad['Iteration'], y=df_grad['GradNorm'], 
                    mode='lines', name=opt_name,
                    line=dict(color=opt_colors[idx])
                ))
        grad_fig.update_layout(title='Норма градиентов', margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(grad_fig, use_container_width=True)
    else:
        st.info("Запустите симуляцию, чтобы увидеть метрики.")

with tab3:
    st.markdown(f"### О {test_func}")
    if test_func == "Quadratic":
        st.markdown("Простая параболическая функция f(x, y) = x² + y² с глобальным минимумом в (0,0). Идеальна для проверки базовой сходимости оптимизаторов.")
    elif test_func == "Rastrigin":
        st.markdown("Мультимодальная функция f(x, y) = 20 + x² + y² - 10(cos(2πx) + cos(2πy)) с глобальным минимумом в (0,0). Множество локальных минимумов усложняет оптимизацию.")
    elif test_func == "Rosenbrock":
        st.markdown("Нелинейная функция f(x, y) = (1 - x)² + 100(y - x²)² с узкой долиной и глобальным минимумом в (1,1). Тестирует способность следовать сложным траекториям.")
    elif test_func == "Himmelblau":
        st.markdown("Мультимодальная функция f(x, y) = (x² + y - 11)² + (x + y² - 7)² с четырьмя минимумами. Проверяет устойчивость к неоднозначным ландшафтам.")
    elif test_func == "Ackley":
        st.markdown("Мультимодальная функция f(x, y) = -20 exp(-0.2 √(0.5(x² + y²))) - exp(0.5(cos(2πx) + cos(2πy))) + e + 20 с глобальным минимумом в (0,0). Множество локальных минимумов затрудняет сходимость.")
    elif test_func == "Griewank":
        st.markdown("Мультимодальная функция f(x, y) = (x² + y²)/4000 - cos(x) cos(y/√2) + 1 с глобальным минимумом в (0,0). Широкая структура и локальные минимумы тестируют глобальный и локальный поиск.")
    elif test_func == "Schwefel":
        st.markdown("Мультимодальная функция f(x, y) = 418.9829·2 - x sin(√|x|) - y sin(√|y|) с глобальным минимумом в (420.9687, 420.9687). Глубокие локальные минимумы усложняют глобальный поиск.")
    elif test_func == "Levy":
        st.markdown("Нелинейная функция с глобальным минимумом в (1,1). Плоские участки и локальные минимумы делают её сложной для оптимизации, особенно на плато.")
    elif test_func == "Beale":
        st.markdown("Нелинейная функция f(x, y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)² с глобальным минимумом в (3, 0.5). Узкие долины тестируют точность оптимизаторов.")

    st.markdown("### Оптимизаторы")
    for opt_name in optimizers:
        with st.expander(opt_name):
            if opt_name == "SGD":
                st.markdown("""
                **Стохастический градиентный спуск (SGD)** обновляет параметры на основе градиентов, вычисленных на подвыборках данных. Моментум сглаживает изменения, ускоряя сходимость.
                
                - **Плюсы**: Простота, масштабируемость на больших датасетах, устойчивость с правильной настройкой.
                - **Минусы**: Чувствителен к выбору скорости обучения, может застревать в локальных минимумах.
                - **Применение**: Широко используется в глубоком обучении, особенно для больших моделей (CNN, RNN).
                - **Рекомендации**: Используйте `learning_rate=0.01–0.1`, `momentum=0.9`. Настройте `lr` с помощью шедулера для сложных функций.
                """)
            elif opt_name == "GD":
                st.markdown("""
                **Градиентный спуск (GD)** обновляет параметры, используя градиенты всего датасета. Это классический метод оптимизации.
                
                - **Плюсы**: Надежен на выпуклых функциях, гарантирует сходимость с малым `lr`.
                - **Минусы**: Медленный на больших данных, требует много вычислений.
                - **Применение**: Подходит для небольших задач или теоретического анализа.
                - **Рекомендации**: Используйте малый `learning_rate=0.001–0.01`. Избегайте на мультимодальных функциях.
                """)
            elif opt_name == "RMSProp":
                st.markdown("""
                **RMSProp** адаптирует скорость обучения, нормализуя градиенты по экспоненциально затухающей средней их квадратов. Это улучшает сходимость на нестационарных данных.
                
                - **Плюсы**: Устойчив к изменяющимся градиентам, хорош для глубоких сетей.
                - **Минусы**: Может быть чувствителен к настройке гиперпараметров.
                - **Применение**: Эффективен для рекуррентных нейронных сетей (RNN) и задач с неравномерными градиентами.
                - **Рекомендации**: Установите `learning_rate=0.001`, `momentum=0.9`. Тестируйте на функциях типа `Rosenbrock`.
                """)
            elif opt_name == "AMSGrad":
                st.markdown("""
                **AMSGrad** — модификация Adam, которая устраняет проблему нестабильности сходимости за счет хранения максимума экспоненциальной средней квадратов градиентов.
                
                - **Плюсы**: Более устойчив, чем Adam, на сложных ландшафтах.
                - **Минусы**: Может быть медленнее Adam на простых задачах.
                - **Применение**: Подходит для глубокого обучения, особенно если Adam нестабилен.
                - **Рекомендации**: Используйте `learning_rate=0.001`, `beta1=0.9`, `beta2=0.999`. Проверяйте на `Rastrigin` или `Ackley`.
                """)
            elif opt_name == "Adagrad":
                st.markdown("""
                **Adagrad** адаптирует скорость обучения для каждого параметра, деля градиент на сумму квадратов прошлых градиентов. Это делает его устойчивым для разреженных данных.
                
                - **Плюсы**: Хорошо работает на разреженных данных, не требует ручной настройки `lr`.
                - **Минусы**: Скорость обучения может слишком быстро уменьшаться, замедляя сходимость.
                - **Применение**: Используется в задачах обработки текста и логистической регрессии.
                - **Рекомендации**: Начните с `learning_rate=0.01`. Подходит для `Quadratic`, но избегайте на `Himmelblau`.
                """)
            elif opt_name == "Adam":
                st.markdown("""
                **Adam** сочетает моментум и адаптивное обучение, используя экспоненциальные средние градиентов и их квадратов. Это один из самых популярных оптимизаторов.
                
                - **Плюсы**: Быстрая сходимость, устойчивость на большинстве задач.
                - **Минусы**: Может быть нестабилен на мультимодальных функциях, требует настройки.
                - **Применение**: Стандарт для сверточных нейронных сетей (CNN) и общего глубокого обучения.
                - **Рекомендации**: Используйте `learning_rate=0.001`, `beta1=0.9`, `beta2=0.999`. Тестируйте на `Rosenbrock` или `Griewank`.
                """)
            elif opt_name == "AdamW":
                st.markdown("""
                **AdamW** — улучшенная версия Adam с декуплированной регуляризацией весов (weight decay). Это улучшает обобщающую способность моделей.
                
                - **Плюсы**: Лучше обобщает, чем Adam, устойчив на сложных задачах.
                - **Минусы**: Требует настройки `weight_decay`.
                - **Применение**: Широко используется в современных CNN и Transformer.
                - **Рекомендации**: Установите `learning_rate=0.001`, `beta1=0.9`, `beta2=0.999`, `weight_decay=0.01`. Проверяйте на `Levy` или `Beale`.
                """)
            elif opt_name == "Sophia":
                st.markdown("""
                **Sophia** — экспериментальный оптимизатор, использующий адаптивные обновления на основе гессиана. Подходит для больших моделей.
                
                - **Плюсы**: Эффективен для больших языковых моделей, устойчив к шуму.
                - **Минусы**: Высокая вычислительная сложность, требует тонкой настройки.
                - **Применение**: Используется в исследованиях больших моделей (LLM).
                - **Рекомендации**: Попробуйте `learning_rate=0.0001`, `beta1=0.9`, `beta2=0.999`. Тестируйте на `Ackley` или `Schwefel`.
                """)
            elif opt_name == "Lion":
                st.markdown("""
                **Lion** использует направление градиента (sign) вместо его величины, что упрощает обновления. Разработан для больших моделей.
                
                - **Плюсы**: Простота, меньшая потребность в памяти, эффективность на Transformer.
                - **Минусы**: Может быть менее точным на малых данных.
                - **Применение**: Подходит для Transformer и больших моделей компьютерного зрения.
                - **Рекомендации**: Используйте `learning_rate=0.0001`, `beta=0.9–0.95`, `weight_decay=0.01`. Проверяйте на `Griewank` или `Rastrigin`.
                """)
            elif opt_name == "Adan":
                st.markdown("""
                **Adan** комбинирует адаптивные обновления с предсказанием градиентов, ускоряя сходимость. Это экспериментальный метод для глубокого обучения.
                
                - **Плюсы**: Быстрая сходимость, устойчивость на больших моделях.
                - **Минусы**: Сложность настройки, высокая вычислительная нагрузка.
                - **Применение**: Подходит для Transformer и больших CNN.
                - **Рекомендации**: Начните с `learning_rate=0.001`, `weight_decay=0.01`. Тестируйте на `Himmelblau` или `Levy`.
                """)
            elif opt_name == "MARS":
                st.markdown("""
                **MARS** — кастомный оптимизатор с адаптивной регуляризацией, разработанный для специфичных задач. Использует моментум для сглаживания.
                
                - **Плюсы**: Гибкость, хорошая сходимость на кастомных задачах.
                - **Минусы**: Требует настройки под конкретную задачу.
                - **Применение**: Экспериментальные задачи, где стандартные оптимизаторы неэффективны.
                - **Рекомендации**: Попробуйте `learning_rate=0.001`, `momentum=0.9`. Проверяйте на `Quadratic` или `Beale`.
                """)
            elif opt_name == "LARS":
                st.markdown("""
                **LARS (Layer-wise Adaptive Rate Scaling)** адаптирует скорость обучения для каждого слоя, основываясь на нормах параметров и градиентов. Эффективен для больших моделей.
                
                - **Плюсы**: Устойчивость на больших батчах, хорошая масштабируемость.
                - **Минусы**: Требует настройки `trust_coeff`, может быть медленным на малых данных.
                - **Применение**: Используется в обучении больших CNN и Transformer, особенно с большими батчами.
                - **Рекомендации**: Установите `learning_rate=0.0001–0.01`, `trust_coeff=0.0005–0.002`, `momentum=0.9`. Тестируйте на `Schwefel` или `Rosenbrock`.
                """)

with tab4:
    with st.expander("Как использовать", expanded=True):
        st.markdown("""
        1. В боковой панели выберите оптимизаторы и настройте параметры.
        2. Выберите тестовую функцию для анализа ландшафта.
        3. Настройте цветовую палитру и параметры визуализации.
        4. Нажмите **"🚀 Запустить симуляцию"** для запуска.
        5. Переключайтесь между вкладками для анализа результатов.
        6. Сохраните результаты с помощью **"💾 Сохранить результаты"**.
        7. Нажмите **"🔄 Сбросить"** для новой симуляции.
        """)
    with st.expander("Параметры"):
        st.markdown("""
        - **Скорость обучения**: Размер шага обновления (0.0001–0.1).
        - **Моментум (β)**: Сглаживание градиентов (0.0–1.0).
        - **Weight Decay**: Регуляризация весов (0.0–0.1).
        - **Итерации**: Число шагов (10–500).
        - **Тестовая функция**: Ландшафт оптимизации (Quadratic, Rastrigin, Ackley и др.).
        - **Разрешение сетки**: Детализация поверхности (50–300).
        - **Диапазон осей**: Ограничение поиска (1.0–500.0, увеличено для Schwefel).
        - **Шум**: Случайные помехи в градиентах (0.0–1.0).
        - **Цветовая палитра**: Стиль графиков (Viridis, Plasma и др.).
        - **trust_coeff (LARS)**: Коэффициент доверия (0.0001–0.01).
        - **beta1, beta2 (Adam, AdamW, AMSGrad, Sophia)**: Сглаживание градиентов.
        - **beta (Lion)**: Сглаживание направления.
        """)
    with st.expander("Советы"):
        st.markdown("""
        - Для `LARS` используйте `trust_coeff=0.0005–0.002` и `learning_rate=0.0001–0.01`. Проверяйте на `Schwefel`.
        - Для `Lion` попробуйте `learning_rate=0.0001`, `beta=0.9–0.95`. Хорошо работает на `Griewank`.
        - Начните с `Quadratic` для проверки сходимости, затем переходите к `Ackley` или `Levy`.
        - Для мультимодальных функций (`Rastrigin`, `Ackley`) увеличивайте `iterations` до 200–500.
        - Используйте палитру `Plotly` для четкого различения оптимизаторов в 2D-режиме.
        - Отключите шкалу цветов в 2D, если она мешает аннотациям.
        - Отключите шум для стабильных результатов, особенно на `Beale` или `Rosenbrock`.
        """)


if save_results and st.session_state.trajectories:
    results = {}
    for opt_name, data in st.session_state.trajectories.items():
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
        results[opt_name] = df.to_dict()
    with open(f"{test_func}_results.json", "w") as f:
        json.dump(results, f)
    st.success("Результаты сохранены в CSV и JSON!")
    with tab2:
        st.markdown("### Сохраненные данные")
        for opt_name, data in results.items():
            st.markdown(f"**{opt_name}**")
            st.dataframe(pd.DataFrame(data))
