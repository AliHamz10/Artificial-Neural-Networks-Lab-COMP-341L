from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from inference import build_model_specs, load_model, preprocess_image_rgb, predict_topk


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent


st.set_page_config(
    page_title="Assignment 04 — Efficient CNNs (CIFAR-10)",
    layout="wide",
)


@st.cache_resource
def _cached_load_model(model_path: str):
    return load_model(Path(model_path))


@st.cache_data
def _load_comparison_table() -> pd.DataFrame | None:
    csv_path = PROJECT_ROOT / "results" / "tables" / "comparison_table.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def main():
    specs = build_model_specs()

    st.title("Assignment 04 — Efficient CNN Architectures (CIFAR-10)")
    st.caption(
        "MobileNetV2 vs DenseNet121 vs EfficientNetB0 (transfer learning). "
        "Upload an image and run inference using the exported `.keras` models."
    )

    with st.sidebar:
        st.header("Model")
        model_key = st.radio(
            "Select a model",
            options=list(specs.keys()),
            format_func=lambda k: specs[k].display_name,
        )
        top_k = st.slider("Top‑K", min_value=1, max_value=10, value=5, step=1)
        show_artifacts = st.checkbox("Show comparison artifacts", value=True)

    spec = specs[model_key]

    if not spec.keras_path.exists():
        st.error(
            f"Missing model file: `{spec.keras_path}`. "
            "Copy the exported `.keras` file(s) into the `models/` folder, then restart the app."
        )
        st.stop()

    model = _cached_load_model(str(spec.keras_path))

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input")
        up = st.file_uploader("Upload an RGB image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if up is None:
            st.info("Upload an image to run prediction.")
            st.stop()

        img = Image.open(up).convert("RGB")
        st.image(img, caption="Uploaded image", use_container_width=True)

        img_np = np.array(img, dtype=np.uint8)
        x = preprocess_image_rgb(img_np, spec.input_size, spec.preprocess)

    with col2:
        st.subheader("Prediction")
        t0 = time.perf_counter()
        topk = predict_topk(model, x, k=top_k)
        t1 = time.perf_counter()

        st.write(f"Inference time: `{(t1 - t0) * 1000:.2f} ms` (single image)")

        df = pd.DataFrame(topk, columns=["Class", "Probability"])
        st.dataframe(df, hide_index=True, use_container_width=True)

        st.bar_chart(df.set_index("Class")["Probability"], use_container_width=True)

    if show_artifacts:
        st.divider()
        st.subheader("Assignment artifacts (from training run)")

        table = _load_comparison_table()
        if table is not None:
            st.markdown("**Comparison table**")
            st.dataframe(table, hide_index=True, use_container_width=True)
        else:
            st.warning("`results/tables/comparison_table.csv` not found.")

        plots_dir = PROJECT_ROOT / "results" / "plots"
        if plots_dir.exists():
            plot_files = [
                "bar_accuracy.png",
                "bar_params.png",
                "bar_gpu_speed.png",
                "bar_model_size.png",
                "mobilenetv2_curves.png",
                "densenet121_curves.png",
                "efficientnetb0_curves.png",
            ]
            for fname in plot_files:
                p = plots_dir / fname
                if p.exists():
                    st.image(str(p), caption=fname, use_container_width=True)
        else:
            st.warning("`results/plots/` folder not found.")


if __name__ == "__main__":
    main()

