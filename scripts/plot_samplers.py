"""
Generate a sampler comparison figure from a random WSI.

Usage:
    python scripts/plot_samplers.py \
        --slide-dir /path/to/slides \
        --backend openslide \
        --output docs/assets/samplers.png

Re-run to get a different random slide each time.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt

from scripts._helpers import get_backend
from wsistream.types import WSI_EXTENSIONS

NUM_PATCHES = 64


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slide-dir", required=True, help="Directory containing WSI files")
    parser.add_argument("--backend", default="openslide", choices=["openslide", "tiffslide"])
    parser.add_argument("--thumbnail-size", type=int, default=1024)
    parser.add_argument("--num-patches", type=int, default=NUM_PATCHES)
    parser.add_argument("--output", default="docs/assets/samplers.png")
    args = parser.parse_args()

    from wsistream.sampling import GridSampler, MultiMagnificationSampler, RandomSampler
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
    downsample = max(downsample_xy)

    mask_arr = OtsuTissueDetector().detect(thumbnail, downsample=downsample_xy)
    tissue_mask = TissueMask(
        mask=mask_arr,
        downsample=downsample,
        slide_dimensions=slide.properties.dimensions,
    )

    tissue_frac = tissue_mask.tissue_fraction
    print(f"Tissue fraction: {tissue_frac:.1%}")

    # Define samplers
    samplers = {
        "Random": RandomSampler(
            patch_size=256,
            num_patches=args.num_patches,
            seed=None,
        ),
        "Grid": GridSampler(
            patch_size=256,
        ),
        "MultiMagnification": MultiMagnificationSampler(
            patch_size=256,
            num_patches=args.num_patches,
            seed=None,
        ),
    }

    # Collect coordinates from each sampler
    sampler_coords: dict[str, list] = {}
    for name, sampler in samplers.items():
        coords = []
        for coord in sampler.sample(slide, tissue_mask):
            coords.append(coord)
            # Grid can produce thousands — cap it for visualization
            if name != "Grid" and len(coords) >= args.num_patches:
                break
            if name == "Grid" and len(coords) >= 2000:
                break
        sampler_coords[name] = coords
        print(f"  {name}: {len(coords)} patches")

    slide.close()

    # Scale factors for mapping level-0 coords to thumbnail
    sx, sy = tw / sw, th / sh

    # Plot: 1 row x N cols — sampling locations on thumbnail
    n_cols = len(samplers)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    colors_by_level = {0: "#e74c3c", 1: "#3498db", 2: "#2ecc71", 3: "#f39c12"}
    default_color = "#e74c3c"

    for col, (name, coords) in enumerate(sampler_coords.items()):
        axes[col].imshow(thumbnail)
        for c in coords:
            color = colors_by_level.get(c.level, default_color)
            cx = int(c.x * sx)
            cy = int(c.y * sy)
            ps = int(c.patch_size * slide.properties.level_downsamples[c.level] * sx)
            ps = max(ps, 2)
            rect = plt.Rectangle(
                (cx, cy), ps, ps, linewidth=0.5, edgecolor=color, facecolor=color, alpha=0.4
            )
            axes[col].add_patch(rect)
        axes[col].set_title(f"{name}\n({len(coords)} patches)", fontsize=11, fontweight="bold")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # Add level legend for MultiMagnification
    if slide.properties.mpp is not None:
        legend_labels = [f"Level {k}" for k in sorted(colors_by_level.keys())]
        legend_handles = [
            plt.Line2D(
                [0], [0], marker="s", color="w", markerfacecolor=colors_by_level[k], markersize=8
            )
            for k in sorted(colors_by_level.keys())
        ]
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower right",
            fontsize=9,
            title="Pyramid level",
            title_fontsize=9,
        )

    fig.suptitle("Sampler comparison", fontsize=13, fontweight="bold")
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
