"""Visualization tab: 2D/3D trajectory and surface plots."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from .state import generate_test_data
from .styles import OPTIMIZER_COLORS, PLOTLY_LAYOUT_DEFAULTS


def get_opt_colors(optimizers):
    """Return a consistent color list for optimizers (contrast palette)."""
    n = len(optimizers)
    if n <= len(OPTIMIZER_COLORS):
        return OPTIMIZER_COLORS[:n]
    repeat = (n // len(OPTIMIZER_COLORS)) + 1
    return (OPTIMIZER_COLORS * repeat)[:n]


def render_visualization_tab(tab, optimizers, test_func, show_surface, show_3d, show_colorbar, realtime_update):
    """Render tab with 2D/3D trajectory visualization and optional surface/contour."""
    with tab:
        trajectories = st.session_state.get("trajectories", {})
        if not st.session_state.get("run_simulation") and not trajectories:
            st.markdown(
                '<div class="pg-empty-card">'
                '<div class="pg-empty-icon"></div>'
                '<div class="pg-empty-title">Нет данных для визуализации</div>'
                '<div class="pg-empty-text">Выберите оптимизаторы в боковой панели и нажмите '
                '<strong>«Запустить симуляцию»</strong>, чтобы построить траектории на ландшафте функции.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        elif st.session_state.get("run_simulation") and not trajectories:
            st.info("Симуляция выполняется...")
        elif trajectories:
            st.success("Симуляция завершена.")

        if not trajectories:
            return

        resolution = st.session_state.get("resolution", 100)
        bounds = st.session_state.get("bounds", 5.0)
        X, Y, Z = generate_test_data(test_func, resolution, bounds)
        opt_colors = get_opt_colors(optimizers)
        color_scheme = st.session_state.get("color_scheme", "Viridis")
        colorscale_plot = "plotly3" if color_scheme == "Plotly" else color_scheme

        fig = go.Figure()
        if show_3d:
            if show_surface:
                fig.add_trace(
                    go.Surface(
                        x=X, y=Y, z=Z,
                        opacity=0.7,
                        colorscale=colorscale_plot,
                        showscale=show_colorbar,
                        showlegend=False,
                    )
                )
            for idx, (opt_name, data) in enumerate(trajectories.items()):
                min_len = min(len(data["traj"]), len(data["loss"]))
                if min_len == 0:
                    continue
                df = pd.DataFrame({
                    "x": [p[0] for p in data["traj"][:min_len]],
                    "y": [p[1] for p in data["traj"][:min_len]],
                    "loss": data["loss"][:min_len],
                    "iteration": list(range(min_len)),
                })
                fig.add_trace(
                    go.Scatter3d(
                        x=df["x"], y=df["y"], z=df["loss"],
                        mode="markers+lines",
                        name=opt_name,
                        marker=dict(size=4, color=opt_colors[idx]),
                        line=dict(width=2, color=opt_colors[idx]),
                        showlegend=True,
                        customdata=df[["iteration", "loss"]],
                        hovertemplate="Итерация: %{customdata[0]}<br>Loss: %{customdata[1]:.4f}<br>x: %{x:.2f}<br>y: %{y:.2f}",
                    )
                )
            _no_title = {k: v for k, v in PLOTLY_LAYOUT_DEFAULTS.items() if k not in ("xaxis", "yaxis", "title")}
            fig.update_layout(
                title=dict(text=f"Сравнение оптимизаторов · {test_func} (3D)", font=dict(size=18)),
                scene=dict(
                    xaxis=dict(backgroundcolor="rgba(248,250,252,0.9)", gridcolor="#e2e8f0"),
                    yaxis=dict(backgroundcolor="rgba(248,250,252,0.9)", gridcolor="#e2e8f0"),
                    zaxis=dict(backgroundcolor="rgba(248,250,252,0.9)", gridcolor="#e2e8f0"),
                ),
                **_no_title,
            )
        else:
            fig.add_trace(
                go.Contour(
                    x=X[0, :], y=Y[:, 0], z=Z,
                    colorscale=colorscale_plot,
                    showscale=show_colorbar,
                    showlegend=False,
                    colorbar=dict(x=1.02, len=0.8, thickness=15),
                )
            )
            for idx, (opt_name, data) in enumerate(trajectories.items()):
                min_len = min(len(data["traj"]), len(data["loss"]))
                if min_len == 0:
                    continue
                df = pd.DataFrame({
                    "x": [p[0] for p in data["traj"][:min_len]],
                    "y": [p[1] for p in data["traj"][:min_len]],
                    "iteration": list(range(min_len)),
                })
                fig.add_trace(
                    go.Scatter(
                        x=df["x"], y=df["y"],
                        mode="markers+lines",
                        name=opt_name,
                        marker=dict(size=8, color=opt_colors[idx]),
                        line=dict(width=2, color=opt_colors[idx]),
                        showlegend=True,
                        customdata=df[["iteration"]],
                        hovertemplate="Итерация: %{customdata[0]}<br>x: %{x:.2f}<br>y: %{y:.2f}",
                    )
                )
            _layout_2d = {k: v for k, v in PLOTLY_LAYOUT_DEFAULTS.items() if k not in ("title", "margin")}
            fig.update_layout(
                **_layout_2d,
                title=dict(text=f"Сравнение оптимизаторов · {test_func} (2D)", font=dict(size=18)),
                margin=dict(l=56, r=120, t=56, b=48),
            )

        _chart_defaults = {k: v for k, v in PLOTLY_LAYOUT_DEFAULTS.items() if k not in ("title", "margin")}
        if show_3d:
            max_traj_len = max(len(d["traj"]) for d in trajectories.values()) if trajectories else 1
            frame_step = max(1, max_traj_len // 50)
            frames = []
            for i in range(1, max_traj_len, frame_step):
                frame_data = []
                if show_surface:
                    frame_data.append(
                        go.Surface(
                            x=X, y=Y, z=Z,
                            opacity=0.7,
                            colorscale=colorscale_plot,
                            showlegend=False,
                        )
                    )
                for idx, (opt_name, data) in enumerate(trajectories.items()):
                    if i <= len(data["traj"]):
                        min_len = min(len(data["traj"][:i]), len(data["loss"]))
                        if min_len == 0:
                            continue
                        df = pd.DataFrame({
                            "x": [p[0] for p in data["traj"][:min_len]],
                            "y": [p[1] for p in data["traj"][:min_len]],
                            "loss": data["loss"][:min_len],
                            "iteration": list(range(min_len)),
                        })
                        frame_data.append(
                            go.Scatter3d(
                                x=df["x"], y=df["y"], z=df["loss"],
                                mode="markers+lines",
                                name=opt_name,
                                marker=dict(size=4, color=opt_colors[idx]),
                                line=dict(width=2, color=opt_colors[idx]),
                                showlegend=True,
                            )
                        )
                frames.append(go.Frame(data=frame_data, name=str(i)))
            fig.update(frames=frames)
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=False,
                        buttons=[
                            dict(label="Play", method="animate", args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]),
                            dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
                        ],
                        x=0.02, xanchor="left", y=0, yanchor="top", pad=dict(t=0, r=10),
                        bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2e8f0", borderwidth=1,
                    )
                ],
                showlegend=True,
                margin=dict(l=56, r=120, t=56, b=48),
                **_chart_defaults,
            )
        else:
            fig.update_layout(showlegend=True, margin=dict(l=56, r=120, t=56, b=48), **_chart_defaults)
        if realtime_update or not trajectories:
            st.plotly_chart(fig, use_container_width=True)
