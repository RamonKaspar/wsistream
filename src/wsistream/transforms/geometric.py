"""Geometric augmentations (flips, rotations)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from wsistream.transforms.base import PatchTransform

# np.rot90(image, k) rotates counterclockwise.
_ROTATIONS = {
    1: cv2.ROTATE_90_COUNTERCLOCKWISE,
    2: cv2.ROTATE_180,
    3: cv2.ROTATE_90_CLOCKWISE,
}


@dataclass
class RandomFlipRotate(PatchTransform):
    """
    Random horizontal/vertical flips and 90-degree rotations.

    Standard for pathology since tissue orientation is arbitrary.
    """

    p_hflip: float = 0.5
    p_vflip: float = 0.5
    p_rot90: float = 0.5
    seed: int | None = None

    def __post_init__(self) -> None:
        for name in ("p_hflip", "p_vflip", "p_rot90"):
            probability = getattr(self, name)
            if not 0 <= probability <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {probability}")
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        hflip = self._rng.random() < self.p_hflip
        vflip = self._rng.random() < self.p_vflip
        k = int(self._rng.integers(1, 4)) if self._rng.random() < self.p_rot90 else 0

        # OpenCV flips and rotates in one pass each, while copying a flipped
        # or rotated numpy view walks the image in a cache-unfriendly order.
        out = image
        if hflip or vflip:
            out = cv2.flip(out, -1 if hflip and vflip else 1 if hflip else 0)
        if k:
            out = cv2.rotate(out, _ROTATIONS[k])
        if out.ndim < image.ndim:
            out = out[:, :, np.newaxis]  # OpenCV drops a single channel axis
        return out
