"""Metrics tab: results table, Loss and gradient norm charts."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.i18n import t
from .styles import PLOTLY_LAYOUT_DEFAULTS
from .visualization import get_opt_colors


def render_metrics_tab(tab, optimizers):
    """Render tab with metrics table and loss/grad-norm plots; optional baseline comparison."""
    with tab:
        trajectories = st.session_state.get("trajectories", {})
        baseline = st.session_state.get("_baseline_trajectories")
        baseline_meta = st.session_state.get("_baseline_meta", {})

        if trajectories and st.button(t("metrics.save_baseline"), key="pg_save_baseline"):
            import copy
            st.session_state["_baseline_trajectories"] = copy.deepcopy(trajectories)
            st.session_state["_baseline_meta"] = {
                "test_func": st.session_state.get("_last_run_test_func"),
                "optimizers": list(st.session_state.get("_last_run_optimizers", [])),
                "iterations": st.session_state.get("_last_run_iterations"),
                "learning_rate": st.session_state.get("learning_rate"),
                "start": st.session_state.get("_last_start"),
            }
            st.rerun()

        if baseline:
            st.caption(t("metrics.baseline_caption") + ": " + (baseline_meta.get("test_func") or "—") + " | " + ", ".join(baseline_meta.get("optimizers") or []))

        if not trajectories:
            st.markdown(
                '<div class="pg-empty-card">'
                '<div class="pg-empty-icon"></div>'
                f'<div class="pg-empty-title">{t("metrics.no_data_title")}</div>'
                f'<div class="pg-empty-text">{t("metrics.no_data_text")}</div>'
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
                    t("metrics.optimizer"): opt_name,
                    t("metrics.final_loss"): f"{final_loss:.4f}",
                    t("metrics.iterations"): iterations_done,
                    t("metrics.avg_grad_norm"): f"{avg_grad_norm:.4f}",
                }
                if avg_local_lr is not None:
                    metric[t("metrics.lars_lr")] = f"{avg_local_lr:.6f}"
                metrics_data.append(metric)
        if metrics_data:
            final_loss_key = t("metrics.final_loss")
            opt_key = t("metrics.optimizer")
            best_idx = min(range(len(metrics_data)), key=lambda i: float(metrics_data[i][final_loss_key]))
            best_name = metrics_data[best_idx][opt_key]
            best_loss = metrics_data[best_idx][final_loss_key]
            st.markdown(
                f'<div class="pg-winner-badge">'
                f'<div class="pg-winner-label">{t("metrics.best_result")}</div>'
                f'<div class="pg-winner-name">{best_name}</div>'
                f'<div style="font-size:0.9rem; opacity:0.95;">{final_loss_key}: {best_loss}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

        if baseline:
            st.markdown("#### " + t("metrics.baseline_section"))
            base_rows = []
            for opt_name, data in baseline.items():
                if data.get("loss"):
                    final = data["loss"][-1]
                    avg_gn = np.mean(data["grad_norms"]) if data.get("grad_norms") else 0
                    base_rows.append({
                        t("metrics.optimizer"): opt_name,
                        t("metrics.final_loss"): f"{final:.4f}",
                        t("metrics.avg_grad_norm"): f"{avg_gn:.4f}",
                    })
            if base_rows:
                st.dataframe(pd.DataFrame(base_rows), use_container_width=True)
            st.markdown("---")

        trajectories_no_reg = st.session_state.get("trajectories_no_reg", {})

        st.markdown("### " + t("metrics.history_loss"))
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
        for idx, opt_name in enumerate(optimizers):
            data_no = trajectories_no_reg.get(opt_name, {})
            if data_no.get("loss"):
                loss_fig.add_trace(
                    go.Scatter(
                        x=list(range(len(data_no["loss"]))),
                        y=data_no["loss"],
                        mode="lines",
                        name=f"{opt_name}{t('viz.no_reg')}",
                        line=dict(color=opt_colors[idx % len(opt_colors)], dash="dash", width=1.5),
                    )
                )
        if baseline:
            base_suffix = t("metrics.baseline_opt")
            for idx, (opt_name, data) in enumerate(baseline.items()):
                if data.get("loss"):
                    loss_fig.add_trace(
                        go.Scatter(
                            x=list(range(len(data["loss"]))),
                            y=data["loss"],
                            mode="lines",
                            name=f"{opt_name}{base_suffix}",
                            line=dict(color=opt_colors[idx % len(opt_colors)], dash="dot"),
                        )
                    )
        _layout = {k: v for k, v in PLOTLY_LAYOUT_DEFAULTS.items() if k not in ("title", "margin")}
        loss_fig.update_layout(
            **_layout,
            title=dict(text=t("metrics.loss_log"), font=dict(size=16)),
            yaxis_type="log",
            margin=dict(l=56, r=24, t=56, b=48),
        )
        st.plotly_chart(loss_fig, use_container_width=True)

        st.markdown("### " + t("metrics.grad_norm"))
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
        for idx, opt_name in enumerate(optimizers):
            data_no = trajectories_no_reg.get(opt_name, {})
            if data_no.get("grad_norms"):
                grad_fig.add_trace(
                    go.Scatter(
                        x=list(range(1, len(data_no["grad_norms"]) + 1)),
                        y=data_no["grad_norms"],
                        mode="lines",
                        name=f"{opt_name}{t('viz.no_reg')}",
                        line=dict(color=opt_colors[idx % len(opt_colors)], dash="dash", width=1.5),
                    )
                )
        grad_fig.update_layout(
            **_layout,
            title=dict(text=t("metrics.grad_norm"), font=dict(size=16)),
            margin=dict(l=56, r=24, t=56, b=48),
        )
        st.plotly_chart(grad_fig, use_container_width=True)
