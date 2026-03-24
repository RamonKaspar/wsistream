"""
Generate a recipe comparison figure from a random WSI.

Shows tissue masks and sampled patch locations for each paper recipe
(Midnight, UNI, Virchow, GPFM, Prov-GigaPath) side by side.

Usage:
    python scripts/plot_recipe_comparison.py \
        --slide-dir /path/to/slides \
        --backend openslide \
        --output docs/assets/recipe_comparison.png

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

NUM_PATCHES = 64

COLORS_BY_LEVEL = {0: "#e74c3c", 1: "#3498db", 2: "#2ecc71", 3: "#f39c12"}
DEFAULT_COLOR = "#e74c3c"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slide-dir", required=True, help="Directory containing WSI files")
    parser.add_argument("--backend", default="openslide", choices=["openslide", "tiffslide"])
    parser.add_argument("--thumbnail-size", type=int, default=1024)
    parser.add_argument("--num-patches", type=int, default=NUM_PATCHES)
    parser.add_argument("--output", default="docs/assets/recipe_comparison.png")
    args = parser.parse_args()

    from wsistream.sampling import MultiMagnificationSampler, RandomSampler
    from wsistream.slide import SlideHandle
    from wsistream.tissue import (
        CLAMTissueDetector,
        HSVTissueDetector,
        OtsuTissueDetector,
    )
    from wsistream.types import TissueMask

    # Pick a random slide
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

    th, tw = thumbnail.shape[:2]
    sw, sh = slide.properties.dimensions
    downsample_tuple = (sw / tw, sh / th)
    downsample_scalar = max(sw / tw, sh / th)

    # Recipes matching the documentation exactly
    recipes = {
        "Midnight": {
            "detector": HSVTissueDetector(
                hue_range=(90, 180),
                sat_range=(8, 255),
                val_range=(103, 255),
            ),
            "sampler": MultiMagnificationSampler(
                target_mpps=[0.25, 0.5, 1.0, 2.0],  # ~40x, ~20x, ~10x, ~5x
                patch_size=256,
                num_patches=args.num_patches,
                tissue_threshold=0.4,
            ),
        },
        "UNI": {
            "detector": CLAMTissueDetector(),
            "sampler": RandomSampler(
                patch_size=256,
                num_patches=args.num_patches,
                target_mpp=0.5,  # 20x
                tissue_threshold=0.4,
            ),
        },
        "Virchow": {
            "detector": HSVTissueDetector(
                hue_range=(90, 180),
                sat_range=(8, 255),
                val_range=(103, 255),
            ),
            "sampler": RandomSampler(
                patch_size=224,
                num_patches=args.num_patches,
                target_mpp=0.5,  # 20x
                tissue_threshold=0.25,
            ),
        },
        "GPFM": {
            "detector": CLAMTissueDetector(),
            "sampler": RandomSampler(
                patch_size=512,
                num_patches=args.num_patches,
                level=0,  # native resolution
                tissue_threshold=0.4,
            ),
        },
        "Prov-GigaPath": {
            "detector": OtsuTissueDetector(),
            "sampler": RandomSampler(
                patch_size=256,
                num_patches=args.num_patches,
                target_mpp=0.5,  # 20x
                tissue_threshold=0.1,
            ),
        },
    }

    # Compute tissue masks
    masks = {}
    for name, cfg in recipes.items():
        masks[name] = cfg["detector"].detect(thumbnail, downsample=downsample_tuple)

    # Sample patch coordinates
    sampler_coords: dict[str, list] = {}
    for name, cfg in recipes.items():
        mask_arr = masks[name]
        tissue_mask = TissueMask(
            mask=mask_arr,
            downsample=downsample_scalar,
            slide_dimensions=slide.properties.dimensions,
        )
        coords = []
        for coord in cfg["sampler"].sample(slide, tissue_mask):
            coords.append(coord)
            if len(coords) >= args.num_patches:
                break
        sampler_coords[name] = coords
        frac = mask_arr.sum() / mask_arr.size
        print(f"  {name}: {len(coords)} patches, tissue={frac:.1%}")

    # Keep reference before closing
    level_downsamples = slide.properties.level_downsamples
    slide.close()

    # Scale factors for mapping level-0 coords to thumbnail
    sx, sy = tw / sw, th / sh

    # Plot: 3 rows x N cols
    # Row 0: tissue masks (binary)
    # Row 1: tissue overlay on thumbnail
    # Row 2: sampled patch locations on thumbnail
    n_cols = len(recipes)
    fig, axes = plt.subplots(3, n_cols, figsize=(3.5 * n_cols, 10.5))

    for col, (name, cfg) in enumerate(recipes.items()):
        mask = masks[name]
        frac = mask.sum() / mask.size

        # Row 0: binary mask
        axes[0, col].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"{name}\n({frac:.1%} tissue)", fontsize=9, fontweight="bold")

        # Row 1: overlay
        axes[1, col].imshow(thumbnail)
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask, 1] = 1.0
        overlay[mask, 3] = 0.35
        axes[1, col].imshow(overlay)

        # Row 2: patch locations, colored by pyramid level
        axes[2, col].imshow(thumbnail)
        coords = sampler_coords[name]
        for c in coords:
            color = COLORS_BY_LEVEL.get(c.level, DEFAULT_COLOR)
            cx = int(c.x * sx)
            cy = int(c.y * sy)
            ds = level_downsamples[c.level]
            ps = int(c.patch_size * ds * sx)
            ps = max(ps, 2)
            rect = plt.Rectangle(
                (cx, cy),
                ps,
                ps,
                linewidth=0.5,
                edgecolor=color,
                facecolor=color,
                alpha=0.4,
            )
            axes[2, col].add_patch(rect)

    # Row labels
    axes[0, 0].set_ylabel("Tissue mask", fontsize=10)
    axes[1, 0].set_ylabel("Overlay", fontsize=10)
    axes[2, 0].set_ylabel(f"Patches (n={args.num_patches})", fontsize=10)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # Level legend
    legend_labels = [f"Level {k}" for k in sorted(COLORS_BY_LEVEL.keys())]
    legend_handles = [
        plt.Line2D(
            [0], [0], marker="s", color="w", markerfacecolor=COLORS_BY_LEVEL[k], markersize=8
        )
        for k in sorted(COLORS_BY_LEVEL.keys())
    ]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower right",
        fontsize=9,
        title="Pyramid level",
        title_fontsize=9,
    )

    fig.suptitle("Recipe comparison", fontsize=14, fontweight="bold")
    slide_name = chosen.stem
    fig.text(0.5, 0.005, slide_name, ha="center", fontsize=7, color="gray")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    svg_path = output_path.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output_path} and {svg_path}")


if __name__ == "__main__":
    main()
