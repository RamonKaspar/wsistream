"""Visualization utilities for inspecting pipeline outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def plot_tissue_mask(
    thumbnail: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    color: tuple[int, int, int] = (0, 255, 0),
    save_path: str | Path | None = None,
) -> np.ndarray:
    """Overlay a tissue mask on a thumbnail. Returns the overlay image."""
    overlay = thumbnail.copy()
    mask_rgb = np.zeros_like(overlay)
    mask_rgb[mask] = color
    overlay = cv2.addWeighted(overlay, 1 - alpha, mask_rgb, alpha, 0)

    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)

    if save_path:
        _save_image(overlay, save_path)
    return overlay


def plot_patch_grid(
    patches: Sequence[np.ndarray],
    ncols: int = 8,
    patch_display_size: int = 128,
    titles: Sequence[str] | None = None,
    save_path: str | Path | None = None,
) -> np.ndarray:
    """Arrange patches in a grid. Returns the grid image."""
    n = len(patches)
    if n == 0:
        return np.zeros((patch_display_size, patch_display_size, 3), dtype=np.uint8)

    nrows = (n + ncols - 1) // ncols
    sz, pad = patch_display_size, 4
    cell = sz + pad
    grid = np.ones((nrows * cell + pad, ncols * cell + pad, 3), dtype=np.uint8) * 240

    for idx, patch in enumerate(patches):
        row, col = divmod(idx, ncols)
        p = patch
        if p.dtype != np.uint8:
            p = np.clip(p * 255 if p.max() <= 1.0 else p, 0, 255).astype(np.uint8)
        resized = cv2.resize(p, (sz, sz))
        y0, x0 = row * cell + pad, col * cell + pad
        grid[y0 : y0 + sz, x0 : x0 + sz] = resized
        if titles and idx < len(titles):
            cv2.putText(
                grid, titles[idx], (x0 + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1
            )

    if save_path:
        _save_image(grid, save_path)
    return grid


def compare_transforms(
    original: np.ndarray,
    transforms_dict: dict[str, object],
    n_samples: int = 5,
    save_path: str | Path | None = None,
) -> np.ndarray:
    """Compare augmentation strategies side-by-side. Each row = one transform."""
    sz, pad = 128, 4
    cell = sz + pad
    n_transforms = len(transforms_dict)
    ncols = n_samples + 1
    label_w = 160
    grid = (
        np.ones((n_transforms * cell + pad, label_w + ncols * cell + pad, 3), dtype=np.uint8) * 255
    )

    for row, (name, transform) in enumerate(transforms_dict.items()):
        y0 = row * cell + pad
        cv2.putText(grid, name, (8, y0 + sz // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        resized = cv2.resize(original, (sz, sz))
        grid[y0 : y0 + sz, label_w + pad : label_w + pad + sz] = resized

        for col in range(n_samples):
            aug = transform(original.copy())
            if aug.dtype != np.uint8:
                aug = np.clip(aug * 255 if aug.max() <= 1.0 else aug, 0, 255).astype(np.uint8)
            resized = cv2.resize(aug, (sz, sz))
            x0 = label_w + (col + 1) * cell + pad
            grid[y0 : y0 + sz, x0 : x0 + sz] = resized

    if save_path:
        _save_image(grid, save_path)
    return grid


def plot_sampling_locations(
    thumbnail: np.ndarray,
    coordinates: Sequence,
    slide_dimensions: tuple[int, int],
    color: tuple[int, int, int] = (255, 0, 0),
    marker_size: int = 3,
    save_path: str | Path | None = None,
) -> np.ndarray:
    """Plot sampled patch locations on a thumbnail."""
    vis = thumbnail.copy()
    th, tw = vis.shape[:2]
    sw, sh = slide_dimensions
    sx, sy = tw / sw, th / sh
    for coord in coordinates:
        cv2.circle(vis, (int(coord.x * sx), int(coord.y * sy)), marker_size, color, -1)
    if save_path:
        _save_image(vis, save_path)
    return vis


def _save_image(image: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
