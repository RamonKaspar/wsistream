"""
HSV-space pixel filtering for tissue detection.

Per-pixel HSV range filtering applied to the thumbnail.

Defaults match the Midnight paper (Karasikov et al., 2025): hue in
[90, 180], saturation in [8, 255], value in [103, 255].

Note on how Midnight actually uses this: in the paper, this filter
is applied per-tile AFTER extraction (accept a tile if >=60% of its
pixels pass). Here we apply it to the thumbnail to produce a tissue
mask, which is a spatial approximation of the same filter. The 60%
threshold is handled downstream by the sampler's ``tissue_threshold``
parameter via ``TissueMask.contains_tissue()``.

For exact per-tile behavior, use ``HSVPatchFilter`` from
``wsistream.filters``.

References
----------
Karasikov et al., "Training state-of-the-art pathology foundation
models with orders of magnitude less data", 2025. arXiv:2504.05186
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from wsistream.tissue.base import TissueDetector


@dataclass
class HSVTissueDetector(TissueDetector):
    """
    Per-pixel HSV range filtering applied to the thumbnail.

    Parameters
    ----------
    hue_range : tuple[int, int]
        Hue range (0-180 in OpenCV). Default: (90, 180).
    sat_range : tuple[int, int]
        Saturation range (0-255). Default: (8, 255).
    val_range : tuple[int, int]
        Value range (0-255). Default: (103, 255).
    """

    hue_range: tuple[int, int] = (90, 180)
    sat_range: tuple[int, int] = (8, 255)
    val_range: tuple[int, int] = (103, 255)

    def detect(
        self, thumbnail: np.ndarray, downsample: tuple[float, float] = (1.0, 1.0)
    ) -> np.ndarray:
        hsv = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)
        lower = np.array([self.hue_range[0], self.sat_range[0], self.val_range[0]])
        upper = np.array([self.hue_range[1], self.sat_range[1], self.val_range[1]])
        return cv2.inRange(hsv, lower, upper) > 0