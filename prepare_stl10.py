"""
prepare_stl10.py
----------------
Downloads STL-10 and saves a subset as PNG files into ./images/.

STL-10 images are 96×96 (vs CIFAR-10's 32×32), so they display sharply
without any upscaling tricks. Each image is optionally upscaled to 224×224
using a clean LANCZOS + unsharp-mask pipeline for best UI quality.

Each image is saved as:
    images/<class_name>/<class_name>_<index>.png

Usage
-----
# Default: 10 images per class (100 total)
python prepare_stl10.py

# Custom: 25 images per class
python prepare_stl10.py --per_class 25

# Flat layout (all in one folder, no subfolders):
python prepare_stl10.py --flat

# Keep native 96×96 resolution (no upscale):
python prepare_stl10.py --size 96
"""

import argparse
from pathlib import Path

from PIL import Image, ImageFilter, ImageEnhance
from torchvision.datasets import STL10


STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck"
]

# STL-10 images are 96×96. Upscale to 224×224 for sharper UI display.
DISPLAY_SIZE = (224, 224)


def upscale_clean(img: Image.Image, target: tuple) -> Image.Image:
    """
    Upscale a 96×96 STL-10 image to `target` size cleanly.

    STL-10 images are already clean (ImageNet-derived), so only a light
    unsharp mask is needed after upscaling — no heavy denoising required.
    """
    # High-quality upscale
    img = img.resize(target, Image.LANCZOS)

    # Unsharp mask to recover edges softened by resampling
    img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=100, threshold=3))

    return img


def save_subset(per_class: int, output_dir: str, flat: bool) -> None:
    """
    Download STL-10 (train split) and save `per_class` images per class
    as PNG files, upscaled to DISPLAY_SIZE.

    Args:
        per_class:   How many images to save per class.
        output_dir:  Root output folder.
        flat:        If True, save all images flat in output_dir.
                     If False, save in output_dir/<class_name>/.
    """
    root = Path(output_dir)

    print("[stl10] Downloading STL-10 (train split) ...")
    # STL-10 train split has 500 images per class (5,000 total)
    dataset = STL10(root="./stl10_raw", split="train", download=True)

    class_counts = {i: 0 for i in range(10)}
    saved_total = 0
    target_per_class = per_class

    print(f"[stl10] Saving {per_class} images per class → '{output_dir}' "
          f"(resized to {DISPLAY_SIZE[0]}×{DISPLAY_SIZE[1]}) ...")

    for img_pil, label in dataset:
        if class_counts[label] >= target_per_class:
            continue

        class_name = STL10_CLASSES[label]
        idx = class_counts[label]

        if flat:
            save_dir = root
        else:
            save_dir = root / class_name

        save_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{class_name}_{idx:04d}.png"

        if DISPLAY_SIZE != (96, 96):
            img_out = upscale_clean(img_pil, DISPLAY_SIZE)
        else:
            img_out = img_pil  # keep native resolution

        img_out.save(save_dir / filename)

        class_counts[label] += 1
        saved_total += 1

        if all(c >= target_per_class for c in class_counts.values()):
            break

    print(f"[stl10] Done! Saved {saved_total} images total "
          f"(each {DISPLAY_SIZE[0]}×{DISPLAY_SIZE[1]}).")
    print(f"[stl10] Class breakdown:")
    for i, name in enumerate(STL10_CLASSES):
        print(f"          {name:12s}: {class_counts[i]} images")

    if not flat:
        print(f"\n[stl10] Folder structure: {output_dir}/<class_name>/<image>.png")

    print(f"\nNext steps:")
    print(f"  python main.py --image_dir {output_dir} --mode build")
    print(f"  streamlit run app.py")


def main():
    parser = argparse.ArgumentParser(description="Prepare an STL-10 image subset")
    parser.add_argument(
        "--per_class", type=int, default=10,
        help="Number of images to save per class (default: 10, max: 500)"
    )
    parser.add_argument(
        "--output_dir", default="./images",
        help="Output folder for images (default: ./images)"
    )
    parser.add_argument(
        "--flat", action="store_true",
        help="Save all images flat in output_dir instead of per-class subfolders"
    )
    parser.add_argument(
        "--size", type=int, default=224,
        help="Pixel size to save images at (default: 224). "
             "STL-10 native size is 96 — use 96 to skip upscaling."
    )
    args = parser.parse_args()

    global DISPLAY_SIZE
    DISPLAY_SIZE = (args.size, args.size)

    save_subset(
        per_class=args.per_class,
        output_dir=args.output_dir,
        flat=args.flat,
    )


if __name__ == "__main__":
    main()
