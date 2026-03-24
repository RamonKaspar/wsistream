"""
Generate a pipeline overview figure showing all stages on a real WSI.

Usage:
    python scripts/plot_pipeline_overview.py \
        --slide-dir /path/to/slides \
        --backend openslide \
        --output docs/assets/pipeline_overview.png

Re-run to get a different random slide each time.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from scripts._helpers import get_backend
from wsistream.types import WSI_EXTENSIONS


def _add_arrow(fig, ax_from, ax_to) -> None:
    """Draw a horizontal arrow between two axes."""
    bbox_from = ax_from.get_position()
    bbox_to = ax_to.get_position()
    x0 = bbox_from.x1
    x1 = bbox_to.x0
    y = (bbox_from.y0 + bbox_from.y1) / 2
    arrow = mpatches.FancyArrowPatch(
        (x0, y),
        (x1, y),
        transform=fig.transFigure,
        arrowstyle="->",
        mutation_scale=15,
        color="#555555",
        lw=1.5,
    )
    fig.add_artist(arrow)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slide-dir", required=True)
    parser.add_argument("--backend", default="openslide", choices=["openslide", "tiffslide"])
    parser.add_argument("--thumbnail-size", type=int, default=1024)
    parser.add_argument("--output", default="docs/assets/pipeline_overview.png")
    args = parser.parse_args()

    from wsistream.filters import HSVPatchFilter
    from wsistream.sampling import RandomSampler
    from wsistream.slide import SlideHandle
    from wsistream.tissue import OtsuTissueDetector
    from wsistream.transforms import (
        ComposeTransforms,
        HEDColorAugmentation,
        RandomFlipRotate,
    )
    from wsistream.types import TissueMask

    # ── Pick a random slide ──
    slide_dir = Path(args.slide_dir)
    slides = [p for p in slide_dir.iterdir() if p.suffix.lower() in WSI_EXTENSIONS]
    if not slides:
        raise FileNotFoundError(f"No WSI files found in {slide_dir}")

    chosen = random.choice(slides)
    print(f"Selected: {chosen.name}")

    # ── Stage 1: Open slide, get thumbnail ──
    backend = get_backend(args.backend)
    slide = SlideHandle(str(chosen), backend=backend)
    size = (args.thumbnail_size, args.thumbnail_size)
    thumbnail = slide.get_thumbnail(size)
    th, tw = thumbnail.shape[:2]
    sw, sh = slide.properties.dimensions
    downsample_xy = (sw / tw, sh / th)
    downsample_scalar = max(downsample_xy)
    sx, sy = tw / sw, th / sh

    print(f"  Dimensions: {sw} x {sh}, MPP: {slide.properties.mpp}")

    # ── Stage 2: Tissue detection ──
    detector = OtsuTissueDetector()
    mask_arr = detector.detect(thumbnail, downsample=downsample_xy)
    tissue_mask = TissueMask(
        mask=mask_arr,
        downsample=downsample_scalar,
        slide_dimensions=slide.properties.dimensions,
    )
    tissue_frac = tissue_mask.tissue_fraction
    print(f"  Tissue fraction: {tissue_frac:.1%}")

    # Build overlay image (green mask on thumbnail)
    overlay = thumbnail.copy()
    green = np.zeros_like(overlay)
    green[mask_arr] = (0, 200, 0)
    overlay = cv2.addWeighted(overlay, 0.65, green, 0.35, 0)

    # ── Stage 3: Sampling ──
    sampler = RandomSampler(patch_size=256, num_patches=100, seed=42)
    all_coords = list(sampler.sample(slide, tissue_mask))
    print(f"  Sampled {len(all_coords)} candidate coordinates")

    # Build sampling location image (cyan rectangles — visible on pink tissue)
    loc_img = thumbnail.copy()
    for c in all_coords:
        cx, cy = int(c.x * sx), int(c.y * sy)
        ps = max(int(256 * sx), 2)
        cv2.rectangle(loc_img, (cx, cy), (cx + ps, cy + ps), (0, 200, 255), 1)

    # ── Stage 4: Read patches + Stage 5: Filter ──
    patch_filter = HSVPatchFilter()
    accepted_patches = []
    accepted_coords = []
    rejected_patches = []
    for c in all_coords:
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
            accepted_patches.append(patch)
            accepted_coords.append(c)
        else:
            rejected_patches.append(patch)

        if len(accepted_patches) >= 12 and len(rejected_patches) >= 4:
            break

    print(f"  Accepted: {len(accepted_patches)}, Rejected: {len(rejected_patches)}")

    # ── Stage 6: Transform ──
    transform = ComposeTransforms(
        transforms=[
            HEDColorAugmentation(sigma=0.05, seed=42),
            RandomFlipRotate(seed=42),
        ]
    )
    transformed_patches = [transform(p.copy()) for p in accepted_patches[:8]]

    slide.close()

    # ── Build the figure ──
    # Layout: 2 rows x 3 cols
    #   Row 0: Thumbnail | Tissue Mask | Sampling Locations
    #   Row 1: Extract & Filter | Transform | Yield PatchResult
    # Adapt figure height to slide aspect ratio
    slide_aspect = sh / sw  # height / width
    row0_h = 4.5 * slide_aspect  # scale top row by slide shape
    row0_h = max(row0_h, 3.0)  # minimum readable height
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(20, row0_h + 5),
        gridspec_kw={"height_ratios": [row0_h, 4.5]},
    )

    stage_color = "#2c3e50"

    # ── Row 0: Slide-level stages ──
    ax_thumb = axes[0, 0]
    for ax, img, title, xlabel in [
        (ax_thumb, thumbnail, "1. Open Slide", f"{sw:,} x {sh:,} px"),
        (axes[0, 1], overlay, "2. Tissue Detection", f"Otsu | {tissue_frac:.0%} tissue"),
        (
            axes[0, 2],
            loc_img,
            "3. Sample Coordinates",
            f"RandomSampler | {len(all_coords)} patches",
        ),
    ]:
        ax.imshow(img, aspect="equal")
        ax.set_title(title, fontsize=12, fontweight="bold", color=stage_color)
        ax.set_xlabel(xlabel, fontsize=8, color="gray")
        ax.set_xticks([])
        ax.set_yticks([])

    ax_mask = axes[0, 1]
    ax_loc = axes[0, 2]

    # ── Row 1: Patch-level stages ──

    # Panel 4/5: Extracted patches (accepted=green border, rejected=red border)
    ax_extract = axes[1, 0]
    ax_extract.axis("off")
    ax_extract.set_title(
        "4. Extract & 5. Filter", fontsize=12, fontweight="bold", color=stage_color
    )

    n_show_accept = min(8, len(accepted_patches))
    n_show_reject = min(4, len(rejected_patches))
    grid_cols = 4
    grid_rows_a = (n_show_accept + grid_cols - 1) // grid_cols
    grid_rows_r = (n_show_reject + grid_cols - 1) // grid_cols
    total_rows = grid_rows_a + grid_rows_r
    cell_sz = 64
    pad = 3
    border = 2
    cell = cell_sz + pad
    gw = grid_cols * cell + pad
    gh = total_rows * cell + pad
    grid_img = np.ones((gh, gw, 3), dtype=np.uint8) * 255

    for idx in range(n_show_accept):
        row, col = divmod(idx, grid_cols)
        p = cv2.resize(accepted_patches[idx], (cell_sz, cell_sz))
        p[:border, :] = (50, 180, 50)
        p[-border:, :] = (50, 180, 50)
        p[:, :border] = (50, 180, 50)
        p[:, -border:] = (50, 180, 50)
        y0, x0 = row * cell + pad, col * cell + pad
        grid_img[y0 : y0 + cell_sz, x0 : x0 + cell_sz] = p

    for idx in range(n_show_reject):
        row, col = divmod(idx, grid_cols)
        row += grid_rows_a
        p = cv2.resize(rejected_patches[idx], (cell_sz, cell_sz))
        p[:border, :] = (220, 60, 60)
        p[-border:, :] = (220, 60, 60)
        p[:, :border] = (220, 60, 60)
        p[:, -border:] = (220, 60, 60)
        y0, x0 = row * cell + pad, col * cell + pad
        grid_img[y0 : y0 + cell_sz, x0 : x0 + cell_sz] = p

    ax_extract.imshow(grid_img)

    # Panel 6: Transformed patches
    ax_aug = axes[1, 1]
    ax_aug.axis("off")
    ax_aug.set_title("6. Transform", fontsize=12, fontweight="bold", color=stage_color)

    n_show_aug = min(8, len(transformed_patches))
    aug_rows = (n_show_aug + grid_cols - 1) // grid_cols
    aug_h = aug_rows * cell + pad
    aug_img = np.ones((aug_h, gw, 3), dtype=np.uint8) * 255

    for idx in range(n_show_aug):
        row, col = divmod(idx, grid_cols)
        p = transformed_patches[idx]
        if p.dtype != np.uint8:
            p = np.clip(p * 255 if p.max() <= 1.0 else p, 0, 255).astype(np.uint8)
        p = cv2.resize(p, (cell_sz, cell_sz))
        y0, x0 = row * cell + pad, col * cell + pad
        aug_img[y0 : y0 + cell_sz, x0 : x0 + cell_sz] = p

    ax_aug.imshow(aug_img)
    ax_aug.set_xlabel("HED + RandomFlipRotate", fontsize=8, color="gray")

    # Panel 7: Yield PatchResult (card deck)
    ax_yield = axes[1, 2]
    ax_yield.axis("off")
    ax_yield.set_title("7. Yield PatchResult", fontsize=12, fontweight="bold", color=stage_color)

    batch_patches = accepted_patches[:4] if len(accepted_patches) >= 4 else accepted_patches
    n_batch = len(batch_patches)
    batch_sz = 130
    offset = 14
    margin = 20
    canvas_h = batch_sz + offset * (n_batch - 1) + 2 * margin
    canvas_w = batch_sz + offset * (n_batch - 1) + 2 * margin
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    for i, p in enumerate(reversed(batch_patches)):
        p_resized = cv2.resize(p, (batch_sz, batch_sz))
        y0 = margin + i * offset
        x0 = margin + i * offset
        canvas[y0 : y0 + batch_sz, x0 : x0 + batch_sz] = p_resized
        cv2.rectangle(canvas, (x0, y0), (x0 + batch_sz - 1, y0 + batch_sz - 1), (180, 180, 180), 1)

    ax_yield.imshow(canvas)
    ax_yield.set_xlabel("image + coordinate + tissue_frac + metadata", fontsize=8, color="gray")

    # ── Layout and arrows ──
    fig.suptitle("wsistream Pipeline Overview", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.canvas.draw()  # finalize positions before reading bbox

    _add_arrow(fig, ax_thumb, ax_mask)
    _add_arrow(fig, ax_mask, ax_loc)
    _add_arrow(fig, ax_extract, ax_aug)
    _add_arrow(fig, ax_aug, ax_yield)

    # ── Legend for filter panel (below bottom row) ──
    bbox_ext = ax_extract.get_position()
    green_patch = mpatches.Patch(facecolor="#32b432", edgecolor="none", label="Accepted (HSV)")
    red_patch = mpatches.Patch(facecolor="#dc3c3c", edgecolor="none", label="Rejected")
    fig.legend(
        handles=[green_patch, red_patch],
        ncol=2,
        frameon=False,
        fontsize=9,
        loc="lower left",
        bbox_to_anchor=(bbox_ext.x0 + 0.02, 0.01),
    )

    fig.text(0.5, 0.005, chosen.stem, ha="center", fontsize=7, color="gray")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    svg_path = output_path.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved to {output_path} and {svg_path}")


if __name__ == "__main__":
    main()
