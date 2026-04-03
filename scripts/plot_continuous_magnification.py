"""
Visualise continuous vs discrete magnification sampling on the same tissue region.

Shows the same tissue location at multiple mpp values: continuous sampling fills
the full spectrum while discrete sampling leaves gaps at intermediate magnifications.

Usage:
    python scripts/plot_continuous_magnification.py \
        --slide-dir /path/to/slides \
        --backend openslide \
        --output docs/assets/continuous_magnification.png
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from scripts._helpers import get_backend
from wsistream.sampling.continuous_magnification import _best_level_for_downsample
from wsistream.types import WSI_EXTENSIONS

OUTPUT_SIZE = 224

# Paper's 7 evaluation magnifications
CONTINUOUS_MPPS = [0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0]

# Standard 4 discrete magnifications
DISCRETE_MPPS = {0.25, 0.5, 1.0, 2.0}


def _find_tissue_center(slide, thumbnail_size=2048):
    """Find a tissue-rich center point in level-0 coordinates."""
    from wsistream.tissue import OtsuTissueDetector

    thumbnail = slide.get_thumbnail((thumbnail_size, thumbnail_size))
    th, tw = thumbnail.shape[:2]
    sw, sh = slide.properties.dimensions
    ds_x, ds_y = sw / tw, sh / th

    mask = OtsuTissueDetector().detect(thumbnail, downsample=(ds_x, ds_y))

    # Erode mask to avoid edges (we need room for large crops)
    margin = int(max(th, tw) * 0.15)
    kernel = np.ones((margin, margin), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel)

    if eroded.sum() == 0:
        eroded = mask.astype(np.uint8)

    # Pick centroid of eroded tissue
    ys, xs = np.where(eroded > 0)
    cy_thumb, cx_thumb = int(np.median(ys)), int(np.median(xs))

    # Convert to level-0 coordinates
    cx_l0 = int(cx_thumb * ds_x)
    cy_l0 = int(cy_thumb * ds_y)
    return cx_l0, cy_l0


def _extract_at_mpp(slide, center_x, center_y, target_mpp):
    """Crop-and-resize a patch centred at (center_x, center_y)."""
    props = slide.properties
    level = _best_level_for_downsample(props, target_mpp)
    level_mpp = props.mpp * props.level_downsamples[level]
    crop_size = max(1, round(OUTPUT_SIZE * target_mpp / level_mpp))

    ds = props.level_downsamples[level]
    crop_l0 = round(crop_size * ds)

    # Centre the crop in level-0 coordinates
    x0 = max(0, center_x - crop_l0 // 2)
    y0 = max(0, center_y - crop_l0 // 2)

    # Clamp to slide bounds
    x0 = min(x0, props.width - crop_l0)
    y0 = min(y0, props.height - crop_l0)

    patch = slide.read_region(x0, y0, crop_size, crop_size, level=level)

    if patch.shape[0] != OUTPUT_SIZE or patch.shape[1] != OUTPUT_SIZE:
        patch = cv2.resize(patch, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR)

    return patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slide-dir", required=True, help="Directory containing WSI files")
    parser.add_argument("--backend", default="openslide", choices=["openslide", "tiffslide"])
    parser.add_argument("--output", default="docs/assets/continuous_magnification.png")
    args = parser.parse_args()

    from wsistream.slide import SlideHandle

    # Pick a random slide
    slide_dir = Path(args.slide_dir)
    slides = sorted(p for p in slide_dir.rglob("*") if p.suffix.lower() in WSI_EXTENSIONS)
    if not slides:
        raise FileNotFoundError(f"No WSI files found in {slide_dir}")

    chosen = random.choice(slides)
    print(f"Selected: {chosen.name}")

    backend = get_backend(args.backend)
    slide = SlideHandle(str(chosen), backend=backend)
    print(f"  Dimensions: {slide.properties.dimensions}, MPP: {slide.properties.mpp}")

    # Find tissue centre
    cx, cy = _find_tissue_center(slide)
    print(f"  Tissue centre: ({cx}, {cy})")

    # Extract patches at each mpp
    patches = {}
    for mpp in CONTINUOUS_MPPS:
        patches[mpp] = _extract_at_mpp(slide, cx, cy, mpp)
        print(f"  {mpp:.3f} mpp: extracted")

    slide.close()

    # --- Plot ---
    n = len(CONTINUOUS_MPPS)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 5.2))

    for col, mpp in enumerate(CONTINUOUS_MPPS):
        # Top row: continuous (all magnifications)
        ax = axes[0, col]
        ax.imshow(patches[mpp])
        ax.set_title(f"{mpp} µm/px", fontsize=8, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        # Bottom row: discrete (only standard magnifications, gaps shown as gray)
        ax = axes[1, col]
        if mpp in DISCRETE_MPPS:
            ax.imshow(patches[mpp])
        else:
            ax.imshow(np.full((OUTPUT_SIZE, OUTPUT_SIZE, 3), 220, dtype=np.uint8))
            ax.text(
                0.5,
                0.5,
                "?",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=28,
                color="#aaa",
                fontweight="bold",
            )
        ax.set_xticks([])
        ax.set_yticks([])

    # Row labels
    axes[0, 0].set_ylabel("Continuous", fontsize=10, fontweight="bold", labelpad=10)
    axes[1, 0].set_ylabel("Discrete", fontsize=10, fontweight="bold", labelpad=10)

    slide_name = chosen.stem.split(".")[0]
    fig.suptitle(
        "Continuous vs Discrete Magnification Sampling\n"
        "(same tissue location, crop-and-resize to 224×224)",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(0.5, 0.01, slide_name, ha="center", fontsize=6, color="gray")
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    svg_path = output_path.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved to {output_path} and {svg_path}")


if __name__ == "__main__":
    main()
