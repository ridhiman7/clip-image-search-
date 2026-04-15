"""
embedding.py
------------
CLIP-based embedding generation for images and text.

How CLIP embeddings work
------------------------
CLIP (Contrastive Language-Image Pretraining) is trained on 400M image-text
pairs so that matching pairs are pulled close together in a shared embedding
space.  This means:
  • encode_text("a dog")  →  a vector near encode_image(<dog photo>)
  • encode_image(img_A)   →  a vector near encode_image(similar images)

We use HuggingFace's `transformers` library to load the pretrained model,
so no custom CLIP repo is required.
"""

import os
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


DEFAULT_MODEL = "openai/clip-vit-base-patch32"


class CLIPEmbedder:
    """
    Wraps a HuggingFace CLIP model to produce L2-normalised embeddings
    for both images and text.

    Normalisation ensures cosine similarity = dot product, which is what
    FAISS IndexFlatIP (inner product) measures efficiently.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "auto"):
        """
        Args:
            model_name: HuggingFace model id (or local path).
            device:     "auto" picks CUDA if available, else CPU.
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[CLIPEmbedder] Loading '{model_name}' on {self.device} ...")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Disable dropout for deterministic outputs
        self.embedding_dim = self.model.config.projection_dim  # 512 for ViT-B/32
        print(f"[CLIPEmbedder] Ready. Embedding dim = {self.embedding_dim}")

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image], batch_size: int = 64) -> np.ndarray:
        """
        Encode a list of PIL images into L2-normalised embeddings.

        Args:
            images:     List of PIL.Image objects.
            batch_size: Number of images processed per forward pass.

        Returns:
            np.ndarray of shape (N, embedding_dim), float32.
        """
        all_embeddings = []
        for i in tqdm(range(0, len(images), batch_size), desc="Encoding images"):
            batch = images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            embeddings = self.model.get_image_features(**inputs)
            # Newer transformers versions may return a dataclass instead of a tensor
            if not isinstance(embeddings, torch.Tensor):
                embeddings = self.model.visual_projection(embeddings.pooler_output)
            embeddings = self._normalize(embeddings)
            all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings).astype("float32")

    @torch.no_grad()
    def encode_image_file(self, image_path: str) -> np.ndarray:
        """Convenience wrapper: load one image file and return its embedding."""
        image = Image.open(image_path).convert("RGB")
        return self.encode_images([image])  # Shape (1, dim)

    @torch.no_grad()
    def encode_image_folder(
        self, folder: str, batch_size: int = 64, extensions=None,
        cache_path: str = None,
    ) -> tuple:
        """
        Encode all images in a folder.

        If `cache_path` is provided and a valid cache exists for the exact
        same set of images, embeddings are loaded from disk instead of
        re-running CLIP — making index rebuilds near-instant.

        Returns:
            (embeddings np.ndarray, image_paths list[str])
        """
        import pickle

        if extensions is None:
            extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}

        folder = Path(folder)
        image_paths = sorted(
            p for p in folder.rglob("*") if p.suffix.lower() in extensions
        )
        if not image_paths:
            raise FileNotFoundError(f"No images found in '{folder}' (searched recursively)")

        str_paths = [str(p) for p in image_paths]

        # --- Try loading from cache ---
        if cache_path:
            cache_emb = Path(cache_path + ".npy")
            cache_meta = Path(cache_path + ".pkl")
            if cache_emb.exists() and cache_meta.exists():
                with open(cache_meta, "rb") as f:
                    cached_paths = pickle.load(f)
                if cached_paths == str_paths:
                    print(f"[CLIPEmbedder] Cache hit — loading embeddings from '{cache_emb}'")
                    return np.load(str(cache_emb)), str_paths
                else:
                    print("[CLIPEmbedder] Cache mismatch (images changed) — re-encoding ...")

        # --- Encode ---
        images = [Image.open(p).convert("RGB") for p in tqdm(image_paths, desc="Loading images")]
        embeddings = self.encode_images(images, batch_size=batch_size)

        # --- Save cache ---
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path + ".npy", embeddings)
            with open(cache_path + ".pkl", "wb") as f:
                pickle.dump(str_paths, f)
            print(f"[CLIPEmbedder] Embeddings cached to '{cache_path}'")

        return embeddings, str_paths

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        """
        Encode a list of text strings into L2-normalised embeddings.

        The same shared embedding space as images, so text and image
        vectors are directly comparable with cosine/dot-product similarity.

        Args:
            texts:      List of query strings.
            batch_size: Texts processed per forward pass.

        Returns:
            np.ndarray of shape (N, embedding_dim), float32.
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            embeddings = self.model.get_text_features(**inputs)
            # Newer transformers versions may return a dataclass instead of a tensor
            if not isinstance(embeddings, torch.Tensor):
                embeddings = self.model.text_projection(embeddings.pooler_output)
            embeddings = self._normalize(embeddings)
            all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings).astype("float32")

    def encode_text(self, text: str) -> np.ndarray:
        """Convenience wrapper for a single query string."""
        return self.encode_texts([text])  # Shape (1, dim)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(tensor: torch.Tensor) -> torch.Tensor:
        """L2-normalise so dot product == cosine similarity."""
        return tensor / tensor.norm(dim=-1, keepdim=True)
