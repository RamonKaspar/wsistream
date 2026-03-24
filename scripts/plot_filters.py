"""
Generate a filter visualization showing accepted/rejected patches.

Usage:
    python scripts/plot_filters.py \
        --slide-dir /path/to/slides \
        --backend openslide \
        --output docs/assets/filters.png

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slide-dir", required=True, help="Directory containing WSI files")
    parser.add_argument("--backend", default="openslide", choices=["openslide", "tiffslide"])
    parser.add_argument("--thumbnail-size", type=int, default=1024)
    parser.add_argument("--output", default="docs/assets/filters.png")
    args = parser.parse_args()

    from wsistream.filters import HSVPatchFilter
    from wsistream.sampling import RandomSampler
    from wsistream.slide import SlideHandle
    from wsistream.tissue import OtsuTissueDetector
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

    # Sample many patches to get a good mix of accepted/rejected
    sampler = RandomSampler(patch_size=256, num_patches=200, seed=None)
    coords = list(sampler.sample(slide, tissue_mask))

    # Apply HSV filter
    patch_filter = HSVPatchFilter()
    accepted = []
    rejected = []

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
            accepted.append(patch)
        else:
            rejected.append(patch)

        # Collect enough of each
        if len(accepted) >= 16 and len(rejected) >= 16:
            break

    slide.close()

    n_accepted = min(16, len(accepted))
    n_rejected = min(16, len(rejected))
    print(f"  Accepted: {len(accepted)}, Rejected: {len(rejected)}")
    print(f"  Showing: {n_accepted} accepted, {n_rejected} rejected")

    # Build figure: two grids side by side
    grid_cols = 4
    grid_rows = max(
        (n_accepted + grid_cols - 1) // grid_cols,
        (n_rejected + grid_cols - 1) // grid_cols,
    )
    grid_rows = max(grid_rows, 1)

    fig, (ax_acc, ax_rej) = plt.subplots(1, 2, figsize=(11, 1.5 + grid_rows * 1.4))

    cell_sz = 80
    pad = 4
    border = 3
    cell = cell_sz + pad

    def _build_grid(patches, n_show, color):
        rows = (n_show + grid_cols - 1) // grid_cols
        rows = max(rows, 1)
        gw = grid_cols * cell + pad
        gh = rows * cell + pad
        grid_img = np.ones((gh, gw, 3), dtype=np.uint8) * 255

        for idx in range(n_show):
            row, col = divmod(idx, grid_cols)
            p = cv2.resize(patches[idx], (cell_sz, cell_sz))
            # Draw border
            p[:border, :] = color
            p[-border:, :] = color
            p[:, :border] = color
            p[:, -border:] = color
            y0, x0 = row * cell + pad, col * cell + pad
            grid_img[y0 : y0 + cell_sz, x0 : x0 + cell_sz] = p

        return grid_img

    green = (50, 180, 50)
    red = (220, 60, 60)

    grid_acc = _build_grid(accepted, n_accepted, green)
    grid_rej = _build_grid(rejected, n_rejected, red)

    ax_acc.imshow(grid_acc)
    ax_acc.set_title(f"Accepted ({n_accepted})", fontsize=11, fontweight="bold", color="#2c8c2c")
    ax_acc.axis("off")

    ax_rej.imshow(grid_rej)
    ax_rej.set_title(f"Rejected ({n_rejected})", fontsize=11, fontweight="bold", color="#c03030")
    ax_rej.axis("off")

    fig.suptitle(
        "HSVPatchFilter",
        fontsize=13,
        fontweight="bold",
    )

    # Subtitle with filter params
    fig.text(
        0.5,
        0.92,
        f"hue=[{patch_filter.hue_range[0]}, {patch_filter.hue_range[1]}]  "
        f"sat=[{patch_filter.sat_range[0]}, {patch_filter.sat_range[1]}]  "
        f"val=[{patch_filter.val_range[0]}, {patch_filter.val_range[1]}]  "
        f"min_pixel_fraction={patch_filter.min_pixel_fraction}",
        ha="center",
        fontsize=8,
        color="gray",
    )

    fig.text(0.5, 0.005, chosen.stem, ha="center", fontsize=7, color="gray")
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    svg_path = output_path.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved to {output_path} and {svg_path}")


if __name__ == "__main__":
    main()
