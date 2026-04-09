"""
data_loader.py
--------------
Handles image loading, preprocessing, and batching.
Supports loading images from a folder (with optional captions).
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

# Standard CLIP preprocessing: resize to 224x224, normalize with ImageNet stats
CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP-specific mean
        std=[0.26862954, 0.26130258, 0.27577711],   # CLIP-specific std
    ),
])


class ImageDataset(Dataset):
    """
    PyTorch Dataset that loads images from a folder.
    Optionally pairs each image with a text caption for evaluation.
    """

    def __init__(
        self,
        image_dir: str,
        caption_file: Optional[str] = None,
        transform=None,
    ):
        """
        Args:
            image_dir:     Path to folder containing images.
            caption_file:  Optional path to a text file with one caption per line
                           (lines match image sort order).
            transform:     Torchvision transform applied to each image.
        """
        self.image_dir = Path(image_dir)
        self.transform = transform or CLIP_TRANSFORM
        self.image_paths = self._collect_images()
        self.captions = self._load_captions(caption_file)

    def _collect_images(self) -> List[Path]:
        """Return sorted list of image paths in the directory."""
        paths = [
            p for p in sorted(self.image_dir.iterdir())
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not paths:
            raise FileNotFoundError(f"No images found in '{self.image_dir}'")
        return paths

    def _load_captions(self, caption_file: Optional[str]) -> Optional[List[str]]:
        """Load captions from a plain-text file (one caption per line)."""
        if caption_file is None:
            return None
        with open(caption_file, "r", encoding="utf-8") as f:
            captions = [line.strip() for line in f.readlines()]
        assert len(captions) == len(self.image_paths), (
            f"Caption count ({len(captions)}) != image count ({len(self.image_paths)})"
        )
        return captions

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """Return (tensor, path_str) or (tensor, caption, path_str)."""
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        if self.captions is not None:
            return tensor, self.captions[idx], str(path)
        return tensor, str(path)


def get_dataloader(
    image_dir: str,
    batch_size: int = 64,
    num_workers: int = 0,
    caption_file: Optional[str] = None,
) -> Tuple[DataLoader, List[str]]:
    """
    Build a DataLoader for batch embedding generation.

    Returns:
        (dataloader, image_paths_list)
    """
    dataset = ImageDataset(image_dir, caption_file=caption_file)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,          # Keep order so indices map to paths
        num_workers=num_workers,
        pin_memory=True,
    )
    image_paths = [str(p) for p in dataset.image_paths]
    return loader, image_paths


def load_single_image(image_path: str) -> "torch.Tensor":
    """Load and preprocess a single image for query use."""
    image = Image.open(image_path).convert("RGB")
    return CLIP_TRANSFORM(image).unsqueeze(0)  # Add batch dimension
