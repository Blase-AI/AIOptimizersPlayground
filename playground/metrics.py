"""Metrics tab: results table, Loss and gradient norm charts."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .styles import PLOTLY_LAYOUT_DEFAULTS
from .visualization import get_opt_colors


def render_metrics_tab(tab, optimizers):
    """Render tab with metrics table and loss/grad-norm plots."""
    with tab:
        trajectories = st.session_state.get("trajectories", {})
        if not trajectories:
            st.markdown(
                '<div class="pg-empty-card">'
                '<div class="pg-empty-icon"></div>'
                '<div class="pg-empty-title">Метрики появятся после симуляции</div>'
                '<div class="pg-empty-text">Запустите сравнение оптимизаторов, чтобы увидеть финальный loss, '
                'историю сходимости и нормы градиентов.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            return

        opt_colors = get_opt_colors(optimizers)
        metrics_data = []
        for opt_name, data in trajectories.items():
            if data["loss"]:
                final_loss = data["loss"][-1]
                iterations_done = len(data["loss"]) - 1
                avg_grad_norm = np.mean(data["grad_norms"]) if data["grad_norms"] else 0
                avg_local_lr = np.mean(data["local_lrs"]) if data["local_lrs"] else None
                metric = {
                    "Оптимизатор": opt_name,
                    "Финальный Loss": f"{final_loss:.4f}",
                    "Итерации": iterations_done,
                    "Средняя норма градиентов": f"{avg_grad_norm:.4f}",
                }
                if avg_local_lr is not None:
                    metric["Средний local_lr (LARS)"] = f"{avg_local_lr:.6f}"
                metrics_data.append(metric)
        if metrics_data:
            best_idx = min(range(len(metrics_data)), key=lambda i: float(metrics_data[i]["Финальный Loss"]))
            best_name = metrics_data[best_idx]["Оптимизатор"]
            best_loss = metrics_data[best_idx]["Финальный Loss"]
            st.markdown(
                f'<div class="pg-winner-badge">'
                f'<div class="pg-winner-label">Лучший результат</div>'
                f'<div class="pg-winner-name">{best_name}</div>'
                f'<div style="font-size:0.9rem; opacity:0.95;">Финальный Loss: {best_loss}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

        st.markdown("### История Loss")
        loss_fig = go.Figure()
        for idx, (opt_name, data) in enumerate(trajectories.items()):
            if data["loss"]:
                df_loss = pd.DataFrame({
                    "Loss": data["loss"],
                    "Iteration": range(len(data["loss"])),
                })
                loss_fig.add_trace(
                    go.Scatter(
                        x=df_loss["Iteration"],
                        y=df_loss["Loss"],
                        mode="lines",
                        name=opt_name,
                        line=dict(color=opt_colors[idx % len(opt_colors)]),
                    )
                )
        _layout = {k: v for k, v in PLOTLY_LAYOUT_DEFAULTS.items() if k not in ("title", "margin")}
        loss_fig.update_layout(
            **_layout,
            title=dict(text="Loss (лог. шкала)", font=dict(size=16)),
            yaxis_type="log",
            margin=dict(l=56, r=24, t=56, b=48),
        )
        st.plotly_chart(loss_fig, use_container_width=True)

        st.markdown("### Норма градиентов")
        grad_fig = go.Figure()
        for idx, (opt_name, data) in enumerate(trajectories.items()):
            if data["grad_norms"]:
                df_grad = pd.DataFrame({
                    "GradNorm": data["grad_norms"],
                    "Iteration": range(1, len(data["grad_norms"]) + 1),
                })
                grad_fig.add_trace(
                    go.Scatter(
                        x=df_grad["Iteration"],
                        y=df_grad["GradNorm"],
                        mode="lines",
                        name=opt_name,
                        line=dict(color=opt_colors[idx % len(opt_colors)], width=2),
                    )
                )
        grad_fig.update_layout(
            **_layout,
            title=dict(text="Норма градиентов", font=dict(size=16)),
            margin=dict(l=56, r=24, t=56, b=48),
        )
        st.plotly_chart(grad_fig, use_container_width=True)
