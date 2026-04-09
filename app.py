"""
app.py
------
Streamlit web interface for the image search engine.

Run with:
    streamlit run app.py

Features:
  • Text-to-image search
  • Image-to-image search (upload a query image)
  • Adjustable top-k slider
  • Live result grid with similarity scores
"""

import io
import os
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

# ------------------------------------------------------------------
# Load engine (cached so it only loads once per session)
# ------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading CLIP model and index...")
def load_engine() -> ImageSearchEngine:
    if not Path(INDEX_DIR).exists():
        st.error(f"No index found at '{INDEX_DIR}'. Run `python main.py --mode build` first.")
        st.stop()
    return ImageSearchEngine.load(INDEX_DIR)


engine = load_engine()

# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------

st.title("🔍 CLIP Image Search")
st.markdown("Search your image collection with **natural language** or an **example image**.")

col_left, col_right = st.columns([1, 3])

with col_left:
    st.subheader("Query")
    search_mode = st.radio("Search mode", ["Text", "Image"], horizontal=True)
    k = st.slider("Results (top-k)", min_value=1, max_value=20, value=6)

    query_text = ""
    query_image = None

    if search_mode == "Text":
        query_text = st.text_input("Enter a description", placeholder="e.g. a red sports car")
        run = st.button("Search", type="primary", disabled=not query_text)
    else:
        uploaded = st.file_uploader("Upload query image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded:
            query_image = Image.open(uploaded).convert("RGB")
            st.image(query_image, caption="Query image", use_column_width=True)
        run = st.button("Search", type="primary", disabled=query_image is None)

    st.markdown("---")
    st.caption(f"Index size: **{engine.index_size()} images**")

# ------------------------------------------------------------------
# Results
# ------------------------------------------------------------------

with col_right:
    st.subheader("Results")

    if run:
        with st.spinner("Searching..."):
            if search_mode == "Text":
                results = engine.text_search(query_text, k=k)
                label = query_text
            else:
                results = engine.image_pil_search(query_image, k=k)
                label = "uploaded image"

        if not results:
            st.warning("No results found.")
        else:
            st.success(f"Top {len(results)} results for: *{label}*")
            # Display in a responsive grid (3 columns)
            cols = st.columns(3)
            for i, result in enumerate(results):
                with cols[i % 3]:
                    try:
                        img = Image.open(result.image_path)
                        st.image(img, use_column_width=True)
                    except Exception:
                        st.error("Could not load image")
                    st.caption(
                        f"**#{result.rank}** · score `{result.score:.4f}`\n\n"
                        f"`{Path(result.image_path).name}`"
                    )
    else:
        st.info("Enter a query on the left and click **Search**.")
   
   