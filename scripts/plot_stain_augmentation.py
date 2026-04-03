"""
Compare stain augmentation methods on real tissue patches.

Shows the native HEDColorAugmentation alongside albumentations' HEStain
(Macenko, Vahadane, random preset) so users can see the visual differences.

Usage:
    python -m scripts.plot_stain_augmentation \
        --slide-dir /path/to/slides \
        --backend openslide \
        --output docs/assets/stain_augmentation.png
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np

from scripts._helpers import get_backend
from wsistream.types import WSI_EXTENSIONS

N_SAMPLES = 3  # augmented versions per method per patch
N_PATCHES = 3  # source patches (columns = 1 original + N_SAMPLES augmented)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slide-dir", required=True, help="Directory containing WSI files")
    parser.add_argument("--backend", default="openslide", choices=["openslide", "tiffslide"])
    parser.add_argument("--output", default="docs/assets/stain_augmentation.png")
    args = parser.parse_args()

    from wsistream.filters import HSVPatchFilter
    from wsistream.sampling import RandomSampler
    from wsistream.slide import SlideHandle
    from wsistream.tissue import OtsuTissueDetector
    from wsistream.transforms import AlbumentationsWrapper, HEDColorAugmentation
    from wsistream.types import TissueMask

    # Pick a random slide
    slide_dir = Path(args.slide_dir)
    slides = sorted(p for p in slide_dir.rglob("*") if p.suffix.lower() in WSI_EXTENSIONS)
    if not slides:
        raise FileNotFoundError(f"No WSI files found in {slide_dir}")

    chosen = random.choice(slides)
    print(f"Selected: {chosen.name}")

    # Open slide, detect tissue, sample patches
    backend = get_backend(args.backend)
    slide = SlideHandle(str(chosen), backend=backend)
    thumbnail = slide.get_thumbnail((1024, 1024))
    th, tw = thumbnail.shape[:2]
    sw, sh = slide.properties.dimensions
    ds_xy = (sw / tw, sh / th)

    mask_arr = OtsuTissueDetector().detect(thumbnail, downsample=ds_xy)
    tissue_mask = TissueMask(
        mask=mask_arr,
        downsample=max(ds_xy),
        slide_dimensions=(sw, sh),
    )

    sampler = RandomSampler(patch_size=256, num_patches=100, seed=None)
    patch_filter = HSVPatchFilter()

    source_patches = []
    for c in sampler.sample(slide, tissue_mask):
        try:
            patch = slide.read_region(c.x, c.y, c.patch_size, c.patch_size, level=c.level)
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

    # Define stain augmentation methods
    methods = [
        ("Original", None),
        ("HED\n(wsistream native)", HEDColorAugmentation(sigma=0.05)),
        (
            "HEStain\n(Macenko)",
            AlbumentationsWrapper(A.Compose([A.HEStain(method="macenko", p=1.0)])),
        ),
        (
            "HEStain\n(Vahadane)",
            AlbumentationsWrapper(A.Compose([A.HEStain(method="vahadane", p=1.0)])),
        ),
        (
            "HEStain\n(random preset)",
            AlbumentationsWrapper(A.Compose([A.HEStain(method="random_preset", p=1.0)])),
        ),
    ]

    # Build figure: rows = methods, columns = original + N_SAMPLES augmented per patch
    n_rows = len(methods)
    n_cols = N_PATCHES * (1 + N_SAMPLES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.4 * n_cols, 1.8 * n_rows))

    for row_idx, (label, transform) in enumerate(methods):
        col = 0
        for patch_idx, src in enumerate(source_patches):
            if transform is None:
                # Original row: show each patch once, leave augmented slots empty
                axes[row_idx, col].imshow(src)
                axes[row_idx, col].set_xticks([])
                axes[row_idx, col].set_yticks([])
                for s in range(N_SAMPLES):
                    axes[row_idx, col + 1 + s].axis("off")
            else:
                # Show original then N_SAMPLES augmented versions
                axes[row_idx, col].imshow(src)
                axes[row_idx, col].set_xticks([])
                axes[row_idx, col].set_yticks([])
                for s in range(N_SAMPLES):
                    aug = transform(src.copy())
                    if aug.dtype != np.uint8:
                        aug = np.clip(aug * 255, 0, 255).astype(np.uint8)
                    axes[row_idx, col + 1 + s].imshow(aug)
                    axes[row_idx, col + 1 + s].set_xticks([])
                    axes[row_idx, col + 1 + s].set_yticks([])

            col += 1 + N_SAMPLES

        # Row label
        axes[row_idx, 0].set_ylabel(
            label,
            fontsize=8,
            fontweight="bold",
            rotation=0,
            labelpad=70,
            ha="right",
            va="center",
        )

    # Column group headers
    for patch_idx in range(N_PATCHES):
        col_start = patch_idx * (1 + N_SAMPLES)
        center_col = col_start + N_SAMPLES / 2
        fig.text(
            (center_col + 0.5) / n_cols,
            0.98,
            f"Patch {patch_idx + 1}",
            ha="center",
            fontsize=8,
            fontweight="bold",
            transform=fig.transFigure,
        )

    fig.suptitle(
        "Stain Augmentation Comparison",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    slide_name = chosen.stem.split(".")[0]
    fig.text(0.5, -0.01, slide_name, ha="center", fontsize=6, color="gray")
    fig.tight_layout(rect=[0, 0.0, 1, 0.97])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    svg_path = output_path.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved to {output_path} and {svg_path}")


if __name__ == "__main__":
    main()
