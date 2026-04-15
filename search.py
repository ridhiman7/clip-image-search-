"""
search.py
---------
High-level search engine combining the embedder and FAISS index.

Provides two query modes:
  1. text_search(query_str, k)  → Text → embedding → top-k images
  2. image_search(image_path, k) → Image → embedding → top-k images
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from embedding import CLIPEmbedder
from index import FaissIndex


@dataclass
class SearchResult:
    """A single search result with its file path and similarity score."""
    rank: int
    image_path: str
    score: float       # Cosine similarity in [-1, 1]; higher = more similar

    def __repr__(self):
        return f"[{self.rank}] {self.image_path}  (score={self.score:.4f})"


class ImageSearchEngine:
    """
    Unified search engine for text-to-image and image-to-image retrieval.

    Usage
    -----
    # Build from scratch
    engine = ImageSearchEngine.build_from_folder("./images", index_dir="./index")

    # Or load a pre-built index
    engine = ImageSearchEngine.load("./index")

    # Query
    results = engine.text_search("a red sports car", k=5)
    results = engine.image_search("query.jpg", k=5)
    """

    def __init__(self, embedder: CLIPEmbedder, index: FaissIndex):
        self.embedder = embedder
        self.index = index

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def build_from_folder(
        cls,
        image_dir: str,
        index_dir: str = "./index",
        index_type: str = "flat",
        batch_size: int = 64,
        model_name: str = "openai/clip-vit-base-patch32",
    ) -> "ImageSearchEngine":
        """
        End-to-end pipeline: load images → embed → build index → save.

        Embeddings are cached alongside the index so subsequent rebuilds
        skip CLIP inference entirely if the image set hasn't changed.

        Args:
            image_dir:  Folder containing images to index.
            index_dir:  Where to persist the FAISS index.
            index_type: "flat" | "ivf" | "ivfpq"
            batch_size: Images per GPU/CPU forward pass.
            model_name: HuggingFace CLIP model id.
        """
        embedder = CLIPEmbedder(model_name=model_name)

        # Cache path lives inside index_dir so it's saved/loaded together
        cache_path = str(Path(index_dir) / "embeddings_cache")

        # Generate (or load cached) embeddings for every image in the folder
        embeddings, image_paths = embedder.encode_image_folder(
            image_dir, batch_size=batch_size, cache_path=cache_path
        )

        # Build and persist FAISS index
        idx = FaissIndex(embedding_dim=embedder.embedding_dim, index_type=index_type)
        idx.build(embeddings, image_paths)
        idx.save(index_dir)

        return cls(embedder, idx)

    @classmethod
    def load(
        cls,
        index_dir: str,
        model_name: str = "openai/clip-vit-base-patch32",
    ) -> "ImageSearchEngine":
        """Load a pre-built index and initialise the embedder."""
        embedder = CLIPEmbedder(model_name=model_name)
        idx = FaissIndex.load(index_dir)
        return cls(embedder, idx)

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def text_search(self, query: str, k: int = 10) -> List[SearchResult]:
        """
        Text → embedding → top-k image search.

        Pipeline:
          1. Tokenise `query` with CLIP text encoder
          2. Project to 512-d embedding space
          3. L2-normalise
          4. Inner-product search in FAISS
          5. Return ranked results

        Args:
            query: Natural-language description (e.g. "sunset over the ocean").
            k:     Number of results to return.

        Returns:
            List of SearchResult ordered by descending similarity.
        """
        query_emb = self.embedder.encode_text(query)   # (1, dim)
        paths, scores = self.index.search(query_emb, k=k)
        return self._to_results(paths, scores)

    def image_search(self, image_path: str, k: int = 10) -> List[SearchResult]:
        """
        Image → embedding → top-k similar image search.

        The query image is encoded through CLIP's vision encoder, so the
        results are semantically similar (same object/scene type), not just
        visually similar (texture/color).

        Args:
            image_path: Path to the query image file.
            k:          Number of results to return.
        """
        query_emb = self.embedder.encode_image_file(image_path)  # (1, dim)
        paths, scores = self.index.search(query_emb, k=k)
        return self._to_results(paths, scores)

    def image_pil_search(self, image: Image.Image, k: int = 10) -> List[SearchResult]:
        """Search using an in-memory PIL Image (useful for web uploads)."""
        query_emb = self.embedder.encode_images([image])
        paths, scores = self.index.search(query_emb, k=k)
        return self._to_results(paths, scores)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _to_results(paths: List[str], scores: np.ndarray) -> List[SearchResult]:
        return [
            SearchResult(rank=i + 1, image_path=p, score=float(s))
            for i, (p, s) in enumerate(zip(paths, scores))
        ]

    def index_size(self) -> int:
        return len(self.index)
