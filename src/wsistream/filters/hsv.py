"""
HSV pixel filter (Midnight-style per-tile rejection).

Accepts a patch only if a sufficient fraction of its pixels fall within
the specified HSV ranges. This matches how Midnight uses the HSV filter:
applied to the extracted tile, not to the thumbnail.

References
----------
Karasikov et al., "Training state-of-the-art pathology foundation models
with orders of magnitude less data", 2025. arXiv:2504.05186
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from wsistream.filters.base import PatchFilter


@dataclass
class HSVPatchFilter(PatchFilter):
    """
    Per-tile HSV pixel filter matching Midnight's tile acceptance criterion.

    A tile is accepted if at least ``min_pixel_fraction`` of its pixels
    have HSV values within the specified ranges. Midnight uses 60%.

    Parameters
    ----------
    hue_range : tuple[int, int]
        Hue range (0-180 in OpenCV). Midnight default: (90, 180).
    sat_range : tuple[int, int]
        Saturation range (0-255). Midnight default: (8, 255).
    val_range : tuple[int, int]
        Value range (0-255). Midnight default: (103, 255).
    min_pixel_fraction : float
        Minimum fraction of pixels that must pass. Midnight default: 0.6.
    """

    hue_range: tuple[int, int] = (90, 180)
    sat_range: tuple[int, int] = (8, 255)
    val_range: tuple[int, int] = (103, 255)
    min_pixel_fraction: float = 0.6

    def accept(self, patch: np.ndarray) -> bool:
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        lower = np.array([self.hue_range[0], self.sat_range[0], self.val_range[0]])
        upper = np.array([self.hue_range[1], self.sat_range[1], self.val_range[1]])
        in_range = cv2.inRange(hsv, lower, upper)
        fraction = float(in_range.astype(bool).sum()) / in_range.size
        return fraction >= self.min_pixel_fraction