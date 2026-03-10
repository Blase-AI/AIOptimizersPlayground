"""Export results: ZIP with CSV and JSON, Markdown report download. Metrics tab."""
import io
import json
import zipfile
import streamlit as st
import pandas as pd
import numpy as np

from core.i18n import t


def _build_report_md(trajectories, test_func, optimizers):
    """Build Markdown report text: function, params, start, metrics, best optimizer."""
    lr = st.session_state.get("learning_rate", "")
    it = st.session_state.get("_last_run_iterations", "")
    start = st.session_state.get("_last_start")
    seed = st.session_state.get("_last_seed")
    random_start = st.session_state.get("_last_run_random_start", True)
    lines = [
        "# " + t("export.report_title"),
        "",
        "## " + t("export.params_section"),
        f"- **{t('export.test_func')}:** {test_func}",
        f"- **{t('export.optimizers')}:** {', '.join(optimizers)}",
        f"- **{t('export.learning_rate')}:** {lr}",
        f"- **{t('export.iterations')}:** {it}",
        f"- **{t('export.start_random')}**" if random_start else f"- **{t('export.start_fixed')}** {start}",
    ]
    if seed is not None:
        lines.append(f"- **Seed:** {seed}")
    elif start and not random_start:
        lines.append(f"- **(x0, y0):** {start}")
    lines.append("")
    lines.append("## " + t("export.metrics_section"))
    rows = []
    for opt_name, data in trajectories.items():
        if data.get("loss"):
            final = data["loss"][-1]
            avg_gn = np.mean(data["grad_norms"]) if data.get("grad_norms") else 0
            rows.append((opt_name, f"{final:.4f}", f"{avg_gn:.4f}"))
    if rows:
        opt_h = t("metrics.optimizer")
        loss_h = t("metrics.final_loss")
        grad_h = t("metrics.avg_grad_norm")
        lines.append(f"| {opt_h} | {loss_h} | {grad_h} |")
        lines.append("|-------------|----------------|--------------------------|")
        for r in rows:
            lines.append(f"| {r[0]} | {r[1]} | {r[2]} |")
        best_name = min(rows, key=lambda x: float(x[1]))[0]
        best_loss = min(r[1] for r in rows)
        lines.append("")
        lines.append("## " + t("export.conclusion"))
        lines.append(f"{t('export.best_loss')}: **{best_name}** (loss = {best_loss}).")
    return "\n".join(lines)


def handle_export(save_results, test_func, optimizers, metrics_tab):
    """If save_results and trajectories exist, build ZIP, report MD, and show download in Metrics tab."""
    trajectories = st.session_state.get("trajectories", {})
    if not save_results or not trajectories:
        return
    results = {}
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for opt_name, data in trajectories.items():
            traj_len = len(data["traj"]) if data["traj"] else 0
            loss_len = len(data["loss"]) if data["loss"] else 0
            min_len = min(traj_len, loss_len)
            if min_len == 0:
                continue
            x = [p[0] for p in data["traj"][:min_len]]
            y = [p[1] for p in data["traj"][:min_len]]
            loss = data["loss"][:min_len]
            grad_norms = data["grad_norms"][:min_len] if data["grad_norms"] else [0] * min_len
            min_len = min(len(x), len(y), len(loss), len(grad_norms))
            df = pd.DataFrame({
                "x": x[:min_len],
                "y": y[:min_len],
                "loss": loss[:min_len],
                "grad_norm": grad_norms[:min_len],
            })
            csv_buffer = df.to_csv(index=False).encode("utf-8")
            zip_file.writestr(f"{opt_name}_{test_func}_results.csv", csv_buffer)
            results[opt_name] = df.to_dict()
        zip_file.writestr(f"{test_func}_results.json", json.dumps(results).encode("utf-8"))
        report_md = _build_report_md(trajectories, test_func, optimizers)
        zip_file.writestr(f"{test_func}_report.md", report_md.encode("utf-8"))
    zip_buffer.seek(0)
    report_md = _build_report_md(trajectories, test_func, optimizers)
    with metrics_tab:
        st.markdown("### " + t("export.saved_data"))
        st.download_button(
            label=t("export.download_zip"),
            data=zip_buffer,
            file_name=f"{test_func}_results.zip",
            mime="application/zip",
            key="pg_download_zip",
        )
        st.download_button(
            label=t("export.download_md"),
            data=report_md.encode("utf-8"),
            file_name=f"{test_func}_report.md",
            mime="text/markdown",
            key="pg_download_report",
        )
        for opt_name, data in results.items():
            st.markdown(f"**{opt_name}**")
            st.dataframe(pd.DataFrame(data))
    st.success(t("export.ready"))
