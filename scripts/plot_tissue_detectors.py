"""
Generate a tissue detector comparison figure from a random WSI.

Usage:
    python scripts/plot_tissue_detectors.py \
        --slide-dir /path/to/slides \
        --backend openslide \
        --output docs/assets/tissue_detectors.png

Re-run to get a different random slide each time.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts._helpers import get_backend
from wsistream.types import WSI_EXTENSIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slide-dir", required=True, help="Directory containing WSI files")
    parser.add_argument("--backend", default="openslide", choices=["openslide", "tiffslide"])
    parser.add_argument("--thumbnail-size", type=int, default=1024)
    parser.add_argument("--output", default="docs/assets/tissue_detectors.png")
    args = parser.parse_args()

    from wsistream.slide import SlideHandle
    from wsistream.tissue import (
        CLAMTissueDetector,
        CombinedTissueDetector,
        HSVTissueDetector,
        OtsuTissueDetector,
    )

    # Collect WSI files and pick a random one
    slide_dir = Path(args.slide_dir)
    slides = [p for p in slide_dir.iterdir() if p.suffix.lower() in WSI_EXTENSIONS]
    if not slides:
        raise FileNotFoundError(f"No WSI files found in {slide_dir}")

    chosen = random.choice(slides)
    print(f"Selected: {chosen.name}")

    # Open slide, get thumbnail
    backend = get_backend(args.backend)
    slide = SlideHandle(str(chosen), backend=backend)
    size = (args.thumbnail_size, args.thumbnail_size)
    thumbnail = slide.get_thumbnail(size)

    # Compute downsample as (scale_x, scale_y) tuple for CLAM
    th, tw = thumbnail.shape[:2]
    sw, sh = slide.properties.dimensions
    downsample = (sw / tw, sh / th)

    slide.close()

    # Run all detectors
    detectors = {
        "CLAM": CLAMTissueDetector(),
        "Otsu": OtsuTissueDetector(),
        "HSV": HSVTissueDetector(),
        "CLAM + HSV": CombinedTissueDetector(detectors=[CLAMTissueDetector(), HSVTissueDetector()]),
    }

    masks = {}
    for name, det in detectors.items():
        masks[name] = det.detect(thumbnail, downsample=downsample)

    # Plot
    n_cols = 1 + len(detectors)  # thumbnail + each mask
    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 6.4))

    # Row 0: thumbnail + binary masks
    axes[0, 0].imshow(thumbnail)
    axes[0, 0].set_title("Input thumbnail", fontsize=10, fontweight="bold")
    axes[0, 0].set_ylabel("Binary mask", fontsize=9)

    for i, (name, mask) in enumerate(masks.items(), start=1):
        frac = mask.sum() / mask.size
        axes[0, i].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"{name}\n({frac:.1%} tissue)", fontsize=10, fontweight="bold")

    # Row 1: thumbnail + overlay (mask in green on top of thumbnail)
    axes[1, 0].imshow(thumbnail)
    axes[1, 0].set_ylabel("Overlay", fontsize=9)

    for i, (name, mask) in enumerate(masks.items(), start=1):
        axes[1, i].imshow(thumbnail)
        # Green overlay where tissue is detected
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask, 1] = 1.0  # green channel
        overlay[mask, 3] = 0.35  # alpha
        axes[1, i].imshow(overlay)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Tissue detection comparison", fontsize=12, fontweight="bold")
    slide_name = chosen.stem
    fig.text(0.5, 0.01, slide_name, ha="center", fontsize=7, color="gray")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    svg_path = output_path.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output_path} and {svg_path}")


if __name__ == "__main__":
    main()
