
from __future__ import annotations

import csv
import io
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st
from PIL import Image
from src.config.config import Config

_cfg = Config()
API_URL = "http://127.0.0.1:8000"
TEST_DIR = Path("test_data")
IMG_SIZE = _cfg.img_size
PATCH_SIZE = _cfg.patch_size 

st.set_page_config(page_title="Patch Localization Demo", layout="wide")
st.title("Patch Localization Demo")

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", value=API_URL)


def _call_predict(api_url: str, source_bytes: bytes, patch_bytes: bytes):
    """Send source + patch to the API and return (y, x) or stop on error."""
    try:
        resp = requests.post(
            f"{api_url.rstrip('/')}/predict",
            files={
                "source": ("source.png", source_bytes, "image/png"),
                "patch": ("patch.png", patch_bytes, "image/png"),
            },
            timeout=30,
        )
        resp.raise_for_status()
    except requests.ConnectionError:
        st.error(f"Cannot connect to the API at **{api_url}**. Is it running?")
        st.stop()
    except requests.HTTPError:
        st.error(f"API error {resp.status_code}: {resp.text}")
        st.stop()
    data = resp.json()
    return data["y"], data["x"]


def _image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@st.cache_data
def load_test_coords() -> dict[int, tuple[int, int]]:
    coords: dict[int, tuple[int, int]] = {}
    with open(TEST_DIR / "coords.csv", newline="") as f:
        for row in csv.DictReader(f):
            idx = int(row["index"])
            coords[idx] = (int(row["y_start"]), int(row["x_start"]))
    return coords


@st.cache_data
def list_test_indices() -> list[int]:
    return sorted(
        int(p.stem) for p in (TEST_DIR / "source").iterdir() if p.suffix == ".png"
    )


# 
tab_test, tab_custom = st.tabs(["Test Samples", "Custom Image"])

with tab_test:
    st.markdown("Browse **test samples** and compare the model prediction against the ground-truth patch location.")

    coords_map = load_test_coords()
    test_indices = list_test_indices()

    with st.sidebar:
        st.markdown(f"**Test samples available:** {len(test_indices)}")

    sample_idx = st.selectbox("Select test sample index", test_indices)

    filename = f"{sample_idx:05d}.png"
    source_path = TEST_DIR / "source" / filename
    patch_path = TEST_DIR / "patch" / filename

    source_img = Image.open(source_path)
    patch_img = Image.open(patch_path)
    gt_y, gt_x = coords_map[sample_idx]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Source image")
        st.image(source_img, use_container_width=True)
    with col2:
        st.subheader("Patch")
        st.image(patch_img, use_container_width=True)

    st.info(f"Ground truth top-left corner: **(y={gt_y}, x={gt_x})**")

    if st.button("Predict", key="predict_test"):
        source_bytes = source_path.read_bytes()
        patch_bytes = patch_path.read_bytes()
        y_pred, x_pred = _call_predict(api_url, source_bytes, patch_bytes)

        ed = np.sqrt((y_pred - gt_y) ** 2 + (x_pred - gt_x) ** 2)
        st.success(
            f"Predicted: **(y={y_pred:.1f}, x={x_pred:.1f})** — "
            f"Euclidean distance to GT: **{ed:.2f} px**"
        )

        source_np = np.array(source_img.convert("L"))
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, title, (y, x), color in [
            (axes[0], "Ground Truth", (gt_y, gt_x), "lime"),
            (axes[1], "Prediction", (y_pred, x_pred), "red"),
        ]:
            ax.imshow(source_np, cmap="gray")
            rect = mpatches.Rectangle(
                (x, y), PATCH_SIZE, PATCH_SIZE,
                linewidth=2, edgecolor=color, facecolor="none",
            )
            ax.add_patch(rect)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"Sample #{sample_idx}  |  ED = {ed:.2f} px", fontsize=13)
        fig.tight_layout()
        st.pyplot(fig)

with tab_custom:
    st.markdown("Upload your own image and **select a patch** to locate.")

    uploaded = st.file_uploader("Upload a source image", type=["png", "jpg", "jpeg", "bmp", "tiff"])

    if uploaded is not None:
        raw_img = Image.open(uploaded).convert("RGB")
        # Resize to model working resolution so coordinates match predictions
        custom_img = raw_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        w, h = custom_img.size

        st.subheader("Select patch region")
        st.markdown(
            f"Image resized to **{w} x {h}** px (model resolution). "
            f"Use the sliders to pick the top-left corner of a {PATCH_SIZE}x{PATCH_SIZE} patch."
        )

        col_sl1, col_sl2 = st.columns(2)
        with col_sl1:
            patch_y = st.slider("Y (top-left row)", 0, max(h - PATCH_SIZE, 0), value=0, key="cy")
        with col_sl2:
            patch_x = st.slider("X (top-left col)", 0, max(w - PATCH_SIZE, 0), value=0, key="cx")

        custom_patch = custom_img.crop((patch_x, patch_y, patch_x + PATCH_SIZE, patch_y + PATCH_SIZE))

        col_src, col_preview = st.columns(2)
        with col_src:
            st.subheader("Source with selection")
            source_np = np.array(custom_img)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(source_np)
            rect = mpatches.Rectangle(
                (patch_x, patch_y), PATCH_SIZE, PATCH_SIZE,
                linewidth=2, edgecolor="cyan", facecolor="none",
            )
            ax.add_patch(rect)
            ax.axis("off")
            fig.tight_layout()
            st.pyplot(fig)
        with col_preview:
            st.subheader("Selected patch")
            st.image(custom_patch, use_container_width=True)

        st.info(f"Selected patch top-left: **(y={patch_y}, x={patch_x})**")

        if st.button("Predict", key="predict_custom"):
            source_bytes = _image_to_png_bytes(custom_img)
            patch_bytes = _image_to_png_bytes(custom_patch)
            y_pred, x_pred = _call_predict(api_url, source_bytes, patch_bytes)

            ed = np.sqrt((y_pred - patch_y) ** 2 + (x_pred - patch_x) ** 2)
            st.success(
                f"Predicted: **(y={y_pred:.1f}, x={x_pred:.1f})** — "
                f"Euclidean distance to selection: **{ed:.2f} px**"
            )

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            for ax, title, (y, x), color in [
                (axes[0], "Selected (GT)", (patch_y, patch_x), "lime"),
                (axes[1], "Prediction", (y_pred, x_pred), "red"),
            ]:
                ax.imshow(source_np)
                rect = mpatches.Rectangle(
                    (x, y), PATCH_SIZE, PATCH_SIZE,
                    linewidth=2, edgecolor=color, facecolor="none",
                )
                ax.add_patch(rect)
                ax.set_title(title)
                ax.axis("off")
            fig.suptitle(f"Custom image  |  ED = {ed:.2f} px", fontsize=13)
            fig.tight_layout()
            st.pyplot(fig)
