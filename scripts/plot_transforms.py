"""
Generate a transform comparison figure showing each transform applied to patches.

Usage:
    python scripts/plot_transforms.py \
        --slide-dir /path/to/slides \
        --backend openslide \
        --output docs/assets/transforms.png

Re-run to get a different random slide each time.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from scripts._helpers import get_backend
from wsistream.types import WSI_EXTENSIONS

N_PATCHES = 6  # columns: one original + transforms applied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slide-dir", required=True, help="Directory containing WSI files")
    parser.add_argument("--backend", default="openslide", choices=["openslide", "tiffslide"])
    parser.add_argument("--thumbnail-size", type=int, default=1024)
    parser.add_argument("--output", default="docs/assets/transforms.png")
    args = parser.parse_args()

    from wsistream.filters import HSVPatchFilter
    from wsistream.sampling import RandomSampler
    from wsistream.slide import SlideHandle
    from wsistream.tissue import OtsuTissueDetector
    from wsistream.transforms import (
        HEDColorAugmentation,
        RandomFlipRotate,
        ResizeTransform,
    )
    from wsistream.types import TissueMask

    # Pick a random slide
    slide_dir = Path(args.slide_dir)
    slides = [p for p in slide_dir.iterdir() if p.suffix.lower() in WSI_EXTENSIONS]
    if not slides:
        raise FileNotFoundError(f"No WSI files found in {slide_dir}")

    chosen = random.choice(slides)
    print(f"Selected: {chosen.name}")

    # Open slide, detect tissue
    backend = get_backend(args.backend)
    slide = SlideHandle(str(chosen), backend=backend)
    size = (args.thumbnail_size, args.thumbnail_size)
    thumbnail = slide.get_thumbnail(size)

    th, tw = thumbnail.shape[:2]
    sw, sh = slide.properties.dimensions
    downsample_xy = (sw / tw, sh / th)
    downsample_scalar = max(downsample_xy)

    mask_arr = OtsuTissueDetector().detect(thumbnail, downsample=downsample_xy)
    tissue_mask = TissueMask(
        mask=mask_arr,
        downsample=downsample_scalar,
        slide_dimensions=slide.properties.dimensions,
    )

    # Sample patches (only accepted ones)
    sampler = RandomSampler(patch_size=256, num_patches=100, seed=None)
    coords = list(sampler.sample(slide, tissue_mask))
    patch_filter = HSVPatchFilter()

    source_patches = []
    for c in coords:
        try:
            patch = slide.read_region(
                x=c.x,
                y=c.y,
                width=c.patch_size,
                height=c.patch_size,
                level=c.level,
            )
        except Exception:
            continue
        if patch_filter.accept(patch):
            source_patches.append(patch)
        if len(source_patches) >= N_PATCHES:
            break

    slide.close()

    if not source_patches:
        raise RuntimeError("No accepted patches found")

    print(f"  Collected {len(source_patches)} source patches")

    # Define transforms to visualize
    # Each entry: (row label, transform instance or None for original)
    transforms = [
        ("Original", None),
        ("HEDColorAugmentation\n(sigma=0.05)", HEDColorAugmentation(sigma=0.05)),
        ("RandomFlipRotate", RandomFlipRotate()),
        ("ResizeTransform\n(256 -> 224)", ResizeTransform(target_size=224)),
    ]

    # Try to add an albumentations example
    try:
        import albumentations as A
        from wsistream.transforms import AlbumentationsWrapper

        albu_transform = AlbumentationsWrapper(
            A.Compose(
                [
                    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=1.0),
                ]
            )
        )
        transforms.append(("AlbumentationsWrapper\n(ColorJitter)", albu_transform))
    except ImportError:
        print("  albumentations not installed, skipping AlbumentationsWrapper")

    n_rows = len(transforms)
    n_cols = len(source_patches)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.2 * n_rows))

    # Ensure axes is 2D even with single row/col
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row_idx, (label, transform) in enumerate(transforms):
        for col_idx, src_patch in enumerate(source_patches):
            ax = axes[row_idx, col_idx]

            if transform is None:
                display = src_patch
            else:
                display = transform(src_patch.copy())

            # Handle float32 output from normalization
            if display.dtype != np.uint8:
                display = np.clip(
                    display * 255 if display.max() <= 1.0 else display, 0, 255
                ).astype(np.uint8)

            # Resize back to consistent display size if ResizeTransform changed it
            if display.shape[0] != 256 or display.shape[1] != 256:
                display = cv2.resize(display, (256, 256), interpolation=cv2.INTER_NEAREST)

            ax.imshow(display)
            ax.set_xticks([])
            ax.set_yticks([])

            # Row label on first column
            if col_idx == 0:
                ax.set_ylabel(
                    label,
                    fontsize=9,
                    fontweight="bold",
                    rotation=0,
                    labelpad=100,
                    ha="right",
                    va="center",
                )

    fig.suptitle("Transform comparison", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.005, chosen.stem, ha="center", fontsize=7, color="gray")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    svg_path = output_path.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved to {output_path} and {svg_path}")


if __name__ == "__main__":
    main()
