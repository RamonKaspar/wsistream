"""Geometric augmentations (flips, rotations)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wsistream.transforms.base import PatchTransform


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
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self._rng.random() < self.p_hflip:
            image = np.flip(image, axis=1).copy()
        if self._rng.random() < self.p_vflip:
            image = np.flip(image, axis=0).copy()
        if self._rng.random() < self.p_rot90:
            k = int(self._rng.integers(1, 4))
            image = np.rot90(image, k=k).copy()
        return image
