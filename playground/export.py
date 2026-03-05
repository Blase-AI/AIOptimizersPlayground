"""Export results: ZIP with CSV and JSON, download button in Metrics tab."""
import io
import json
import zipfile
import streamlit as st
import pandas as pd


def handle_export(save_results, test_func, optimizers, metrics_tab):
    """If save_results and trajectories exist, build ZIP and show download in Metrics tab."""
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
    zip_buffer.seek(0)
    with metrics_tab:
        st.markdown("### Сохраненные данные")
        st.download_button(
            label="Скачать результаты (ZIP)",
            data=zip_buffer,
            file_name=f"{test_func}_results.zip",
            mime="application/zip",
            key="pg_download_zip",
        )
        for opt_name, data in results.items():
            st.markdown(f"**{opt_name}**")
            st.dataframe(pd.DataFrame(data))
    st.success("Файлы готовы для скачивания! Нажмите на кнопку во вкладке 'Метрики'.")
