"""Combine multiple tissue detectors via logical AND."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wsistream.tissue.base import TissueDetector


@dataclass
class CombinedTissueDetector(TissueDetector):
    """
    Intersection of multiple detectors.

    Useful for chaining broad detection (e.g., Otsu or CLAM) with
    quality filtering (e.g., HSV to exclude adipose tissue).
    """

    detectors: list[TissueDetector] = field(default_factory=list)

    def detect(
        self, thumbnail: np.ndarray, downsample: tuple[float, float] = (1.0, 1.0)
    ) -> np.ndarray:
        if not self.detectors:
            raise ValueError("CombinedTissueDetector requires at least one detector.")
        masks = [d.detect(thumbnail, downsample=downsample) for d in self.detectors]
        result = masks[0]
        for m in masks[1:]:
            result = result & m
        return result

    def __repr__(self) -> str:
        inner = ", ".join(repr(d) for d in self.detectors)
        return f"CombinedTissueDetector([{inner}])"