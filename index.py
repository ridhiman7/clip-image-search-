"""
index.py
--------
FAISS-based vector index for fast approximate nearest-neighbour (ANN) search.

Index types used
----------------
• IndexFlatIP  – exact inner-product search (cosine similarity after L2-norm).
                 Best for < 100k images; perfectly accurate.
• IndexIVFFlat – inverted-file index; partitions space into `nlist` Voronoi
                 cells so only a fraction are searched.  Trades a small
                 accuracy drop for large speed gains at 100k+ images.
• IndexIVFPQ   – adds product quantisation on top of IVF to compress vectors
                 and reduce RAM usage; good for millions of images.

For most use-cases (tens of thousands of images) IndexFlatIP is the
right choice – no training needed, exact results, still very fast.
"""

import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np


class FaissIndex:
    """
    Wrapper around a FAISS index that also stores the list of image paths
    so search results can be mapped back to file names.
    """

    def __init__(self, embedding_dim: int, index_type: str = "flat"):
        """
        Args:
            embedding_dim: Dimensionality of the embedding vectors (e.g. 512).
            index_type:    One of "flat" | "ivf" | "ivfpq".
                           "flat"  → IndexFlatIP  (exact, recommended default)
                           "ivf"   → IndexIVFFlat (fast, slight accuracy drop)
                           "ivfpq" → IndexIVFPQ   (fast + compressed)
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = self._build_empty_index(index_type, embedding_dim)
        self.image_paths: List[str] = []

    # ------------------------------------------------------------------
    # Building / adding vectors
    # ------------------------------------------------------------------

    def _build_empty_index(self, index_type: str, dim: int) -> faiss.Index:
        if index_type == "flat":
            # Exact cosine similarity (vectors must be L2-normalised)
            return faiss.IndexFlatIP(dim)

        elif index_type == "ivf":
            # IVF: train with data before adding vectors
            nlist = 100  # Number of Voronoi cells; tune for dataset size
            quantiser = faiss.IndexFlatIP(dim)
            return faiss.IndexIVFFlat(quantiser, dim, nlist, faiss.METRIC_INNER_PRODUCT)

        elif index_type == "ivfpq":
            # IVF + Product Quantisation: compressed, very scalable
            nlist = 100
            m = 8          # Number of sub-quantisers (must divide dim)
            bits = 8       # Bits per sub-quantiser
            quantiser = faiss.IndexFlatIP(dim)
            return faiss.IndexIVFPQ(quantiser, dim, nlist, m, bits)

        else:
            raise ValueError(f"Unknown index_type '{index_type}'. Choose flat | ivf | ivfpq.")

    def build(self, embeddings: np.ndarray, image_paths: List[str]) -> None:
        """
        Train (if needed) and add all embeddings to the index.

        Args:
            embeddings:   Float32 array of shape (N, embedding_dim).
            image_paths:  Corresponding file paths; len must equal N.
        """
        assert embeddings.shape[0] == len(image_paths), "Embedding/path count mismatch"
        assert embeddings.dtype == np.float32, "FAISS requires float32"

        self.image_paths = list(image_paths)

        if self.index_type in ("ivf", "ivfpq"):
            # IVF indexes need to see training data before accepting vectors
            print(f"[FaissIndex] Training {self.index_type} index ...")
            self.index.train(embeddings)

        print(f"[FaissIndex] Adding {len(embeddings)} vectors ...")
        self.index.add(embeddings)
        print(f"[FaissIndex] Index size: {self.index.ntotal} vectors")

    def add(self, embeddings: np.ndarray, image_paths: List[str]) -> None:
        """Incrementally add new vectors (useful for growing datasets)."""
        assert embeddings.dtype == np.float32
        self.image_paths.extend(image_paths)
        self.index.add(embeddings)

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        nprobe: int = 10,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Return the top-k most similar images for a query embedding.

        Args:
            query_embedding: Float32 array of shape (1, embedding_dim) or (embedding_dim,).
            k:               Number of results to return.
            nprobe:          IVF-only: how many Voronoi cells to inspect
                             (higher → more accurate, slower).

        Returns:
            (paths, scores): lists of image file paths and similarity scores.
        """
        query = query_embedding.astype("float32")
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # nprobe only applies to IVF indexes
        if self.index_type in ("ivf", "ivfpq"):
            self.index.nprobe = nprobe

        scores, indices = self.index.search(query, k)
        scores = scores[0]   # Flatten batch dimension
        indices = indices[0]

        # Filter out invalid indices (-1 returned when k > index size)
        valid = [(self.image_paths[i], float(s)) for i, s in zip(indices, scores) if i != -1]
        paths = [v[0] for v in valid]
        sims = np.array([v[1] for v in valid])
        return paths, sims

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """
        Save the FAISS index and image path list to disk.

        Creates:
            <directory>/faiss.index   – binary FAISS index
            <directory>/metadata.pkl  – image paths + config
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / "faiss.index"))

        metadata = {
            "image_paths": self.image_paths,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
        }
        with open(directory / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        print(f"[FaissIndex] Saved to '{directory}' ({self.index.ntotal} vectors)")

    @classmethod
    def load(cls, directory: str) -> "FaissIndex":
        """
        Load a previously saved index from disk.

        Returns a fully initialised FaissIndex ready for search.
        """
        directory = Path(directory)

        with open(directory / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        obj = cls.__new__(cls)
        obj.embedding_dim = metadata["embedding_dim"]
        obj.index_type = metadata["index_type"]
        obj.image_paths = metadata["image_paths"]
        obj.index = faiss.read_index(str(directory / "faiss.index"))

        print(f"[FaissIndex] Loaded from '{directory}' ({obj.index.ntotal} vectors)")
        return obj

    def __len__(self) -> int:
        return self.index.ntotal
