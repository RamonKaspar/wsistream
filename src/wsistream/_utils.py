"""Internal helpers shared across wsistream modules."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def infer_batch_size(batch: Any) -> int:
    """Infer batch size from the first tensor-like field."""
    if isinstance(batch, dict):
        if "image" in batch:
            try:
                return int(batch["image"].shape[0])
            except (AttributeError, IndexError):
                return 1
        for value in batch.values():
            try:
                return int(value.shape[0])
            except (AttributeError, IndexError):
                continue
        for value in batch.values():
            if isinstance(value, (list, tuple)):
                return len(value)
    try:
        return int(batch.shape[0])
    except (AttributeError, IndexError):
        return 1


def resize_patch(patch: np.ndarray, size: int) -> np.ndarray:
    """Resize a square patch to ``size x size`` without aliasing when shrinking.

    Bilinear interpolation reads only the 2x2 source pixels nearest to each
    output pixel, so shrinking by more than 2x skips source pixels and aliases
    fine structure. Area interpolation averages every source pixel covered by
    an output pixel. Enlarging keeps bilinear interpolation.
    """
    interpolation = cv2.INTER_AREA if patch.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(patch, (size, size), interpolation=interpolation)
