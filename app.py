"""
app.py
------
Streamlit web interface for the CLIP Image Search engine.

Self-bootstrapping: if no index is found on startup, the app automatically
downloads STL-10 (10 images/class) and builds the FAISS index. This means
the app works out-of-the-box on Streamlit Community Cloud with no manual setup.

Run locally:
    streamlit run app.py
"""

import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from search import ImageSearchEngine

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="CLIP Image Search",
    page_icon="🔍",
    layout="wide",
)

INDEX_DIR = "./index"
IMAGE_DIR = "./images"
PER_CLASS  = 10   # images per class to download on first run


# ------------------------------------------------------------------
# Auto-bootstrap: download dataset + build index if missing
# ------------------------------------------------------------------

def bootstrap():
    """
    Called once on first launch when no index exists.
    Downloads STL-10 subset and builds the FAISS index.
    Shows progress in the Streamlit UI so the user knows what's happening.
    """
    st.info("First launch — setting up the search engine. This takes 2–4 minutes.")

    # Step 1 — download and prepare images
    with st.spinner(f"Downloading STL-10 dataset ({PER_CLASS} images/class) ..."):
        result = subprocess.run(
            [sys.executable, "prepare_stl10.py", "--per_class", str(PER_CLASS)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            st.error("Failed to download dataset.")
            st.code(result.stderr)
            st.stop()

    st.success(f"Dataset ready — {PER_CLASS * 10} images saved.")

    # Step 2 — build FAISS index
    with st.spinner("Building FAISS index (encoding images with CLIP) ..."):
        result = subprocess.run(
            [sys.executable, "main.py", "--image_dir", IMAGE_DIR,
             "--index_dir", INDEX_DIR, "--mode", "build"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            st.error("Failed to build index.")
            st.code(result.stderr)
            st.stop()

    st.success("Index built! Reloading app ...")
    st.rerun()


# ------------------------------------------------------------------
# Load engine (cached so it only loads once per session)
# ------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading CLIP model and index ...")
def load_engine() -> ImageSearchEngine:
    if not Path(INDEX_DIR).exists():
        return None
    return ImageSearchEngine.load(INDEX_DIR)


# Run bootstrap if index doesn't exist yet
if not Path(INDEX_DIR).exists():
    bootstrap()

engine = load_engine()

if engine is None:
    st.error("Index not found. Refresh the page to trigger setup.")
    st.stop()


# ------------------------------------------------------------------
# UI — Sidebar / left column
# ------------------------------------------------------------------

st.title("🔍 CLIP Image Search")
st.markdown("Search your image collection with **natural language** or an **example image**.")

col_left, col_right = st.columns([1, 3])

with col_left:
    st.subheader("Query")
    search_mode = st.radio("Search mode", ["Text", "Image"], horizontal=True)
    k = st.slider("Results (top-k)", min_value=1, max_value=20, value=6)

    query_text  = ""
    query_image = None

    if search_mode == "Text":
        query_text = st.text_input(
            "Enter a description",
            placeholder="e.g. a red truck, a bird in flight"
        )
        run = st.button("Search", type="primary", disabled=not query_text)
    else:
        uploaded = st.file_uploader("Upload query image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded:
            query_image = Image.open(uploaded).convert("RGB")
            st.image(query_image, caption="Query image", use_column_width=True)
        run = st.button("Search", type="primary", disabled=query_image is None)

    st.markdown("---")
    st.caption(f"Index size: **{engine.index_size()} images**")

    # Show class breakdown
    images_root = Path(IMAGE_DIR)
    if images_root.exists():
        class_dirs = sorted([d for d in images_root.iterdir() if d.is_dir()])
        if class_dirs:
            st.markdown("**Classes in index:**")
            for d in class_dirs:
                count = len([
                    f for f in d.iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                ])
                st.caption(f"• {d.name}: {count} images")


# ------------------------------------------------------------------
# Results — right column
# ------------------------------------------------------------------

with col_right:
    st.subheader("Results")

    if run:
        with st.spinner("Searching ..."):
            if search_mode == "Text":
                results = engine.text_search(query_text, k=k)
                label   = query_text
            else:
                results = engine.image_pil_search(query_image, k=k)
                label   = "uploaded image"

        if not results:
            st.warning("No results found.")
        else:
            st.success(f"Top {len(results)} results for: *{label}*")
            cols = st.columns(3)
            for i, result in enumerate(results):
                with cols[i % 3]:
                    try:
                        img = Image.open(result.image_path)
                        st.image(img, use_column_width=True)
                    except Exception:
                        st.error("Could not load image")
                    p = Path(result.image_path)
                    class_label = (
                        p.parent.name
                        if p.parent.name != "images"
                        else p.stem.rsplit("_", 1)[0]
                    )
                    st.caption(
                        f"**#{result.rank}** · score `{result.score:.4f}`\n\n"
                        f"Class: **{class_label}** · `{p.name}`"
                    )
    else:
        st.info("Enter a query on the left and click **Search**.")
