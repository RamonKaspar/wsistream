"""
Generate two documentation figures illustrating multi-view patch generation.

Figure 1 (views_multicrop): DINO-style multi-crop — shared-transformed patch with
    colored crop boxes, with each output thumbnail framed in the matching color.

Figure 2 (views_twoview): Two independently augmented views of the same extracted
    patch, shown side by side.

Usage:
    python scripts/plot_views.py \
        --slide-dir /path/to/slides \
        --backend openslide \
        --output docs/assets/views.svg

Re-run to get a different random slide each time.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

try:
    from scripts._helpers import get_backend
except ModuleNotFoundError:
    from _helpers import get_backend
from wsistream.types import WSI_EXTENSIONS
from wsistream.views import CropParams, RandomResizedCrop

# Colorblind-friendly Wong palette
GLOBAL_COLORS = ["#0072B2", "#009E73"]  # blue, green
LOCAL_COLORS = ["#D55E00", "#E69F00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]

_N_GLOBAL = 2
_N_LOCAL = 6


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    img = image.astype(np.float32)
    if img.max() <= 1.0 + 1e-6:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def _hide_ticks(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_crop_box(ax, params: CropParams, color: str, linewidth: float = 2.5) -> None:
    rect = Rectangle(
        (params.x, params.y),
        params.width,
        params.height,
        linewidth=linewidth,
        edgecolor=color,
        facecolor="none",
        zorder=3,
    )
    ax.add_patch(rect)


def _frame_ax(ax, color: str, linewidth: float = 3.5) -> None:
    """Color the spines of an axes to match its source crop box."""
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(linewidth)


class _RecordingRandomResizedCrop(RandomResizedCrop):
    """RandomResizedCrop that records sampled parameters for plotting."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.params: list[CropParams] = []

    def __call__(self, image: np.ndarray) -> np.ndarray:
        params = self.sample_params(image)
        self.params.append(params)
        return self.apply_params(image, params)


def _run_dino_pipeline(slide_path: str, backend_name: str):
    from wsistream.pipeline import PatchPipeline
    from wsistream.sampling import RandomSampler
    from wsistream.tissue import OtsuTissueDetector
    from wsistream.transforms import ComposeTransforms, HEDColorAugmentation, RandomFlipRotate
    from wsistream.views import ViewConfig

    # 256x256 source tile matches Midnight (Karasikov et al., 2025) and UNI (Chen et al., 2024),
    # both of which extract 256x256 at 0.5 MPP and feed them to DINOv2 multi-crop.
    # DINO v1 scale ranges are used here (0.4-1.0 global, 0.05-0.4 local) for visual clarity;
    # DINOv2 tightens the global range to (0.32-1.0) and the local range to (0.05-0.32).
    global_crops = [
        _RecordingRandomResizedCrop(size=224, scale=(0.4, 1.0)) for _ in range(_N_GLOBAL)
    ]
    local_crop = _RecordingRandomResizedCrop(size=96, scale=(0.05, 0.4))

    pipeline = PatchPipeline(
        slide_paths=[slide_path],
        backend=get_backend(backend_name),
        tissue_detector=OtsuTissueDetector(),
        sampler=RandomSampler(patch_size=256, num_patches=1),
        views=[
            ViewConfig(name="source"),
            *[
                ViewConfig(
                    name=f"global{i + 1}",
                    crop=global_crops[i],
                    transforms=ComposeTransforms([HEDColorAugmentation(sigma=0.05)]),
                )
                for i in range(_N_GLOBAL)
            ],
            ViewConfig(
                name="local",
                crop=local_crop,
                count=_N_LOCAL,
                transforms=ComposeTransforms([HEDColorAugmentation(sigma=0.10)]),
            ),
        ],
        shared_transforms=RandomFlipRotate(),
        pool_size=1,
        patches_per_slide=1,
        seed=42,
    )
    result = next(iter(pipeline))
    global_params = [c.params[0] for c in global_crops]
    local_params = local_crop.params
    global_size = global_crops[0].size
    local_size = local_crop.size
    return result, global_params, local_params, global_size, local_size


def _run_twoview_pipeline(slide_path: str, backend_name: str):
    from wsistream.pipeline import PatchPipeline
    from wsistream.sampling import RandomSampler
    from wsistream.tissue import OtsuTissueDetector
    from wsistream.transforms import HEDColorAugmentation, RandomFlipRotate
    from wsistream.views import ViewConfig

    pipeline = PatchPipeline(
        slide_paths=[slide_path],
        backend=get_backend(backend_name),
        tissue_detector=OtsuTissueDetector(),
        sampler=RandomSampler(patch_size=256, num_patches=1),
        views=[
            ViewConfig(name="source"),
            ViewConfig(name="view1", transforms=HEDColorAugmentation(sigma=0.18)),
            ViewConfig(name="view2", transforms=RandomFlipRotate()),
        ],
        pool_size=1,
        patches_per_slide=1,
        seed=7,
    )
    return next(iter(pipeline))


def _save_multicrop_figure(
    source_img: np.ndarray,
    global_imgs: list[np.ndarray],
    local_imgs: list[np.ndarray],
    global_params: list[CropParams],
    local_params: list[CropParams],
    global_size: int,
    local_size: int,
    output_path: Path,
    slide_name: str = "",
) -> None:
    """Single-figure: source patch + crop boxes + color-framed output thumbnails."""
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    n_global = len(global_imgs)
    n_local = len(local_imgs)
    h_src, w_src = source_img.shape[:2]

    fig = plt.figure(figsize=(14, 7), facecolor="white")
    fig.suptitle(
        "DINO-style Multi-crop View Generation",
        fontsize=13,
        fontweight="bold",
        y=0.97,
    )

    # Main layout: source patch (left) | crop outputs (right).
    # width_ratios chosen so the source column is approximately square at this figsize.
    gs = GridSpec(
        1,
        2,
        figure=fig,
        left=0.02,
        right=0.98,
        top=0.90,
        bottom=0.10,
        width_ratios=[n_local - 1, n_local],
        wspace=0.05,
    )

    # --- Source patch ---
    ax_src = fig.add_subplot(gs[0, 0])
    ax_src.imshow(source_img)
    ax_src.set_title(f"Shared-transformed patch  ({w_src}\u00d7{h_src})", fontsize=10, pad=6)
    _hide_ticks(ax_src)
    for params, color in zip(global_params, GLOBAL_COLORS):
        _draw_crop_box(ax_src, params, color, linewidth=3.0)
    for params, color in zip(local_params, LOCAL_COLORS[:n_local]):
        _draw_crop_box(ax_src, params, color, linewidth=2.0)

    # --- Right panel: global crops (top) + local crops (bottom) ---
    # height_ratios reflect actual output pixel sizes.
    gs_right = GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=gs[0, 1],
        height_ratios=[global_size, local_size],
        hspace=0.55,
    )

    gs_global = GridSpecFromSubplotSpec(1, n_global, subplot_spec=gs_right[0], wspace=0.08)
    global_axes = []
    for i, (img, color) in enumerate(zip(global_imgs, GLOBAL_COLORS)):
        ax = fig.add_subplot(gs_global[0, i])
        ax.imshow(img)
        ax.set_title(f"global{i + 1}", fontsize=9, color=color, fontweight="bold", pad=4)
        _hide_ticks(ax)
        _frame_ax(ax, color, linewidth=3.5)
        global_axes.append(ax)

    gs_local = GridSpecFromSubplotSpec(1, n_local, subplot_spec=gs_right[1], wspace=0.06)
    local_axes = []
    for i, (img, color) in enumerate(zip(local_imgs, LOCAL_COLORS[:n_local])):
        ax = fig.add_subplot(gs_local[0, i])
        ax.imshow(img)
        ax.set_title(f"local_{i}", fontsize=7.5, color=color, pad=3)
        _hide_ticks(ax)
        _frame_ax(ax, color, linewidth=3.0)
        local_axes.append(ax)

    # Section labels: read actual GridSpec bounding boxes so labels land
    # precisely above each row of thumbnails, regardless of height_ratios.
    fig.canvas.draw()
    for axes_list, label in [
        (global_axes, f"Global crops  [{global_size}\u00d7{global_size}]"),
        (local_axes, f"Local crops  [{local_size}\u00d7{local_size}]"),
    ]:
        p0 = axes_list[0].get_position()
        p1 = axes_list[-1].get_position()
        x_ctr = (p0.x0 + p1.x1) / 2
        y_top = max(ax.get_position().y1 for ax in axes_list)
        fig.text(
            x_ctr,
            y_top + 0.018,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#444444",
        )

    # Caption below source explaining the non-square crop boxes.
    src_pos = ax_src.get_position()
    fig.text(
        (src_pos.x0 + src_pos.x1) / 2,
        src_pos.y0 - 0.012,
        "Crop boxes show source windows before resizing. "
        "Aspect-ratio jitter (ratio \u2208 [3/4,\u2009 4/3]) makes source windows non-square; "
        "the resize step produces square outputs.",
        ha="center",
        va="top",
        fontsize=7,
        color="#666666",
        style="italic",
    )

    if slide_name:
        fig.text(0.5, 0.02, slide_name, ha="center", fontsize=7, color="gray")

    for ext in (".svg", ".png"):
        p = output_path.with_suffix(ext)
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved \u2192 {p}")
    plt.close(fig)


def _save_twoview_figure(
    source_img: np.ndarray,
    view1: np.ndarray,
    view2: np.ndarray,
    output_path: Path,
    slide_name: str = "",
) -> None:
    """Three-panel figure: source + two independently augmented views."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 4), facecolor="white")
    fig.suptitle("Two-view Augmentation from One Patch", fontsize=11, fontweight="bold")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.05, wspace=0.08)

    data = [
        (source_img, "Source patch", "#888888", False),
        (view1, "View 1 — HED color jitter", "#0072B2", True),
        (view2, "View 2 — random flip / rotate", "#009E73", True),
    ]
    for ax, (img, title, color, framed) in zip(axes, data):
        ax.imshow(img)
        ax.set_title(title, fontsize=9, fontweight="bold" if framed else "normal", pad=5)
        _hide_ticks(ax)
        if framed:
            _frame_ax(ax, color, linewidth=3.5)
        else:
            for spine in ax.spines.values():
                spine.set_edgecolor("#cccccc")
                spine.set_linewidth(1.0)

    if slide_name:
        fig.text(0.5, 0.01, slide_name, ha="center", fontsize=7, color="gray")

    for ext in (".svg", ".png"):
        p = output_path.with_suffix(ext)
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved → {p}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slide-dir", required=True, help="Directory containing WSI files")
    parser.add_argument("--backend", default="openslide", choices=["openslide", "tiffslide"])
    parser.add_argument(
        "--output",
        default="docs/assets/views.svg",
        help=(
            "Base output path (e.g. docs/assets/views.svg). "
            "Two figures are saved: <stem>_multicrop.{svg,png} and <stem>_twoview.{svg,png}."
        ),
    )
    args = parser.parse_args()

    slide_dir = Path(args.slide_dir)
    slides = [p for p in slide_dir.rglob("*") if p.suffix.lower() in WSI_EXTENSIONS]
    if not slides:
        raise FileNotFoundError(f"No WSI files found in {slide_dir}")

    chosen = random.choice(slides)
    print(f"Selected: {chosen.name}")

    base = Path(args.output)
    out_dir = base.parent
    stem = base.stem  # e.g. "views"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: DINO-style multi-crop
    dino_result, global_params, local_params, global_size, local_size = _run_dino_pipeline(
        str(chosen), args.backend
    )
    source_img = _to_uint8(dino_result.views["source"])
    global_imgs = [_to_uint8(dino_result.views[f"global{i + 1}"]) for i in range(_N_GLOBAL)]
    local_imgs = [_to_uint8(dino_result.views[f"local_{i}"]) for i in range(_N_LOCAL)]
    _save_multicrop_figure(
        source_img,
        global_imgs,
        local_imgs,
        global_params,
        local_params,
        global_size,
        local_size,
        out_dir / f"{stem}_multicrop.svg",
        slide_name=chosen.stem,
    )

    # Figure 2: Two independent augmentations
    two_result = _run_twoview_pipeline(str(chosen), args.backend)
    _save_twoview_figure(
        _to_uint8(two_result.views["source"]),
        _to_uint8(two_result.views["view1"]),
        _to_uint8(two_result.views["view2"]),
        out_dir / f"{stem}_twoview.svg",
        slide_name=chosen.stem,
    )


if __name__ == "__main__":
    main()
