"""Resize patches to a target size."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from wsistream.transforms.base import PatchTransform


@dataclass
class ResizeTransform(PatchTransform):
    """Resize a patch to target_size x target_size."""

    target_size: int = 224
    interpolation: int = cv2.INTER_LINEAR

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image.shape[0] == self.target_size and image.shape[1] == self.target_size:
            return image
        return cv2.resize(
            image, (self.target_size, self.target_size), interpolation=self.interpolation
        )
