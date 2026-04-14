"""
prepare_cifar10.py
------------------
Downloads CIFAR-10 and saves a subset as PNG files into ./images/.

Each image is saved as:
    images/<class_name>/<class_name>_<index>.png

This lets the search engine (and evaluator) know the ground-truth label
from the filename/folder name.

Usage
-----
# Default: 10 images per class (100 total)
python prepare_cifar10.py

# Custom: 25 images per class
python prepare_cifar10.py --per_class 25

# Flat layout (all images in one folder, no subfolders):
python prepare_cifar10.py --flat
"""

import argparse
from pathlib import Path

from PIL import Image, ImageFilter, ImageEnhance
from torchvision.datasets import CIFAR10


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# CIFAR-10 images are 32×32. Upscale to this size so they look sharp in the UI.
DISPLAY_SIZE = (224, 224)


def upscale_clean(img: Image.Image, target: tuple[int, int]) -> Image.Image:
    """
    Upscale a tiny CIFAR-10 image (32×32) to `target` size cleanly.

    Pipeline:
      1. Slight Gaussian blur on the raw 32×32 image — smooths harsh pixel
         block edges before they get magnified.
      2. LANCZOS upscale — best-quality resampling filter.
      3. Unsharp mask — sharpens real edges without amplifying noise.
      4. Mild contrast/colour boost — counters the slight wash-out from step 1.
    """
    # Step 1 – tame pixel-block noise at source resolution
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))

    # Step 2 – high-quality upscale
    img = img.resize(target, Image.LANCZOS)

    # Step 3 – unsharp mask: radius=2, strength=120%, edge threshold=3
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))

    # Step 4 – slight contrast/saturation recovery
    img = ImageEnhance.Contrast(img).enhance(1.1)
    img = ImageEnhance.Color(img).enhance(1.1)

    return img


def save_subset(per_class: int, output_dir: str, flat: bool) -> None:
    """
    Download CIFAR-10 (train split) and save `per_class` images per class
    as PNG files, upscaled to 224×224 using high-quality LANCZOS resampling.

    Args:
        per_class:   How many images to save per class.
        output_dir:  Root output folder.
        flat:        If True, save all images flat in output_dir.
                     If False, save in output_dir/<class_name>/.
    """
    root = Path(output_dir)

    # Download CIFAR-10 training set (50 000 images, 5 000/class)
    print("[cifar10] Downloading CIFAR-10 (train split) ...")
    dataset = CIFAR10(root="./cifar10_raw", train=True, download=True)

    # Count how many we've saved per class
    class_counts = {i: 0 for i in range(10)}
    saved_total = 0
    target_per_class = per_class

    print(f"[cifar10] Saving {per_class} images per class → '{output_dir}' (resized to {DISPLAY_SIZE[0]}×{DISPLAY_SIZE[1]}) ...")

    for img_pil, label in dataset:
        if class_counts[label] >= target_per_class:
            continue

        class_name = CIFAR10_CLASSES[label]
        idx = class_counts[label]

        if flat:
            save_dir = root
        else:
            save_dir = root / class_name

        save_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{class_name}_{idx:04d}.png"
        img_resized = upscale_clean(img_pil, DISPLAY_SIZE)
        img_resized.save(save_dir / filename)

        class_counts[label] += 1
        saved_total += 1

        # Stop early if all classes are done
        if all(c >= target_per_class for c in class_counts.values()):
            break

    print(f"[cifar10] Done! Saved {saved_total} images total (each resized to {DISPLAY_SIZE[0]}×{DISPLAY_SIZE[1]}).")
    print(f"[cifar10] Class breakdown:")
    for i, name in enumerate(CIFAR10_CLASSES):
        print(f"          {name:12s}: {class_counts[i]} images")

    if not flat:
        print(f"\n[cifar10] Folder structure: {output_dir}/<class_name>/<image>.png")
        print(f"[cifar10] NOTE: for the search engine, point --image_dir to the root")
        print(f"          or run with --flat to put all images in one folder.\n")

    print(f"\nNext step:")
    print(f"  python main.py --image_dir {output_dir} --mode build")
    print(f"  streamlit run app.py")


def main():
    parser = argparse.ArgumentParser(description="Prepare a CIFAR-10 image subset")
    parser.add_argument(
        "--per_class", type=int, default=10,
        help="Number of images to save per class (default: 10)"
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
        help="Pixel size to upscale images to (default: 224). "
             "CIFAR-10 images are 32×32; larger values look sharper in the UI."
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
