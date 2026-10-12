from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import streamlit as st
import torch
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src.model import CompactCNN
from src.preprocess import preprocess_sketch

ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "artifacts" / "model.pt"

st.set_page_config(page_title="Draw to Search", page_icon="✏️", layout="wide")
st.markdown(
    """
    <style>
      .block-container {max-width: 1120px; padding-top: 2rem;}
      .hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#38265c,#7258a6); color: white; margin-bottom: 1rem;}
      .hero h1 {color: white; margin: 0 0 .4rem 0; font-size: 2.2rem;}
      .hero p {color: #f2edff; margin: 0; font-size: 1.05rem;}
      .small-muted {color: #777; font-size: .9rem;}
    </style>
    <div class="hero"><h1>Draw it. Find it.</h1><p>A visual search prototype: sketch an object and let a compact CNN suggest what to search for.</p></div>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model(checkpoint_path: str):
    path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    classes = checkpoint["classes"]
    model = CompactCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes, checkpoint.get("metrics", {})

left, right = st.columns([1.15, 0.85], gap="large")
with left:
    st.subheader("1. Sketch your idea")
    st.caption("Use the mouse or touchpad. Aim for a simple outline, similar to a QuickDraw doodle.")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=10,
        stroke_color="#111111",
        background_color="#FFFFFF",
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="drawing_canvas",
        display_toolbar=True,
        update_streamlit=True,
    )
    predict_clicked = st.button("Find what I drew", type="primary", use_container_width=True)

with right:
    st.subheader("2. Search suggestions")
    if not CHECKPOINT_PATH.exists():
        st.info(
            "The model has not been trained yet. From the project folder, run `python scripts/download_data.py` and then `python train.py --samples-per-class 4000 --epochs 12`. Restart this app after training."
        )
        st.markdown("**What the model will return**")
        st.write("A ranked list of likely doodle labels with confidence scores, plus a search link for the most likely label.")
    else:
        try:
            model, classes, train_metrics = load_model(str(CHECKPOINT_PATH))
            if train_metrics.get("test_accuracy") is not None:
                st.metric("Held-out test accuracy", f"{100 * float(train_metrics['test_accuracy']):.1f}%")
                st.caption("This value comes from the saved local training run.")
        except Exception as exc:
            model = None
            classes = []
            st.error(f"Could not load the trained checkpoint: {exc}")

        if predict_clicked:
            if canvas_result.image_data is None:
                st.warning("Draw something on the canvas first.")
            elif model is None:
                st.error("The saved model could not be loaded. Re-run training and check artifacts/model.pt.")
            else:
                rgba = Image.fromarray(np.asarray(canvas_result.image_data, dtype=np.uint8), mode="RGBA")
                raw_sketch = preprocess_sketch(rgba)
                # QuickDraw dataset bitmaps use black background (0.0) with white strokes (1.0).
                # Invert the canvas sketch (light background, dark strokes) to align with training data.
                aligned_sketch = 1.0 - raw_sketch
                if aligned_sketch.max() < 0.1:
                    st.warning("The canvas looks blank. Draw something first!")
                else:
                    tensor = torch.from_numpy(aligned_sketch).unsqueeze(0)
                    with torch.no_grad():
                        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
                    top_indices = np.argsort(probabilities)[::-1][: min(3, len(classes))]
                    best_label = classes[int(top_indices[0])]
                    st.success(f"Best match: **{best_label}**")
                    st.progress(float(probabilities[int(top_indices[0])]), text=f"Confidence: {probabilities[int(top_indices[0])] * 100:.1f}%")
                    st.markdown("**Other possible matches**")
                    for idx in top_indices[1:]:
                        st.write(f"{classes[int(idx)]} - {probabilities[int(idx)] * 100:.1f}%")
                    query_url = "https://www.google.com/search?q=" + quote_plus(best_label)
                    st.markdown(f"[Search the web for **{best_label}** ↗]({query_url})")
                    st.caption("Confidence is the model's relative score, not a guarantee that the label is correct.")

st.divider()
st.markdown(
    "<p class='small-muted'>Model: compact CNN · Input: 28 × 28 grayscale bitmap · Dataset: Google Quick, Draw! · Research basis: “Drawing: A New Way To Search”</p>",
    unsafe_allow_html=True,
)
