"""
main.py
-------
Demo script: build an index and run example searches.

Usage
-----
# Build a new index (first time):
python main.py --image_dir ./images --mode build

# Search with a text query:
python main.py --mode text --query "a sunset over the ocean"

# Search with an image:
python main.py --mode image --query ./images/example.jpg

# Build + immediately run a demo:
python main.py --image_dir ./images --mode demo
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless backend (no display required)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

from search import ImageSearchEngine

INDEX_DIR = "./index"


# ------------------------------------------------------------------
# Display helpers
# ------------------------------------------------------------------
 
def display_results(results, query_label: str, output_path: str = "results.png") -> None:
    """Save a grid of result images to a PNG file."""
    n = len(results)
    if n == 0:
        print("No results to display.")
        return

    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows + 1))
    fig.suptitle(f'Query: "{query_label}"', fontsize=14, fontweight="bold")

    # Flatten axes for uniform indexing
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, (ax, result) in enumerate(zip(axes, results)):
        try:
            img = mpimg.imread(result.image_path)
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "Error loading", ha="center", va="center")
        ax.set_title(
            f"#{result.rank}  score={result.score:.3f}\n{Path(result.image_path).name}",
            fontsize=8,
        )
        ax.axis("off")

    # Hide unused subplots
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[main] Results saved to '{output_path}'")


def print_results(results, query_label: str) -> None:
    """Print results as a ranked list."""
    print(f'\nTop-{len(results)} results for: "{query_label}"')
    print("-" * 60)
    for r in results:
        print(r)
    print()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CLIP Image Search Demo")
    parser.add_argument("--image_dir", default="./images", help="Folder of images to index")
    parser.add_argument("--index_dir", default=INDEX_DIR, help="Where to store/load the index")
    parser.add_argument(
        "--mode",
        choices=["build", "text", "image", "demo"],
        default="demo",
        help="build=index images; text=text query; image=image query; demo=all",
    )
    parser.add_argument("--query", default="", help="Text string or image path for search")
    parser.add_argument("--k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--index_type", default="flat", choices=["flat", "ivf", "ivfpq"])
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # ---- BUILD -------------------------------------------------------
    if args.mode in ("build", "demo"):
        if not Path(args.image_dir).exists():
            # Create sample images for demo purposes if folder is missing
            print(f"[main] '{args.image_dir}' not found – creating synthetic demo images ...")
            _create_demo_images(args.image_dir)

        print(f"\n[main] Building index from '{args.image_dir}' ...")
        engine = ImageSearchEngine.build_from_folder(
            image_dir=args.image_dir,
            index_dir=args.index_dir,
            index_type=args.index_type,
            batch_size=args.batch_size,
        )
        print(f"[main] Index built: {engine.index_size()} images")

        if args.mode == "build":
            return  # Done

        # Demo: run a few sample queries
        print("\n[main] Running demo queries ...")
        sample_queries = ["a red object", "something blue", "a circle shape"]
        for q in sample_queries:
            results = engine.text_search(q, k=args.k)
            print_results(results, q)
            display_results(results, q, output_path=f"results_demo_{q[:10].replace(' ','_')}.png")
        return

    # ---- LOAD existing index -----------------------------------------
    if not Path(args.index_dir).exists():
        print(f"[main] No index at '{args.index_dir}'. Run with --mode build first.")
        sys.exit(1)

    print(f"\n[main] Loading index from '{args.index_dir}' ...")
    engine = ImageSearchEngine.load(args.index_dir)
    print(f"[main] Loaded {engine.index_size()} images.")

    # ---- TEXT SEARCH -------------------------------------------------
    if args.mode == "text":
        if not args.query:
            print("Please provide --query 'your text here'")
            sys.exit(1)
        results = engine.text_search(args.query, k=args.k)
        print_results(results, args.query)
        display_results(results, args.query)

    # ---- IMAGE SEARCH ------------------------------------------------
    elif args.mode == "image":
        if not args.query or not Path(args.query).exists():
            print("Please provide --query path/to/image.jpg")
            sys.exit(1)
        results = engine.image_search(args.query, k=args.k)
        print_results(results, Path(args.query).name)
        display_results(results, Path(args.query).name)


# ------------------------------------------------------------------
# Demo image generator (no real images needed to try the system)
# ------------------------------------------------------------------

def _create_demo_images(folder: str, n: int = 30) -> None:
    """Create coloured synthetic images for a quick smoke-test."""
    import random
    import numpy as np

    Path(folder).mkdir(parents=True, exist_ok=True)
    colors = {
        "red":    (220, 50,  50),
        "blue":   (50,  100, 220),
        "green":  (50,  180, 80),
        "yellow": (240, 210, 50),
        "purple": (150, 50,  200),
    }
    shapes = ["circle", "square", "triangle"]

    for i in range(n):
        color_name = random.choice(list(colors.keys()))
        shape_name = random.choice(shapes)
        rgb = colors[color_name]

        # Plain coloured background
        canvas = np.full((224, 224, 3), rgb, dtype=np.uint8)
        img = Image.fromarray(canvas)
        fname = f"{color_name}_{shape_name}_{i:03d}.jpg"
        img.save(Path(folder) / fname)

    print(f"[demo] Created {n} synthetic images in '{folder}'")


if __name__ == "__main__":
    main()
