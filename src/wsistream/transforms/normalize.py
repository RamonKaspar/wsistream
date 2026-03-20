"""Per-channel mean/std normalization."""

from __future__ import annotations

import numpy as np

from wsistream.transforms.base import PatchTransform


class NormalizeTransform(PatchTransform):
    """
    Convert uint8 -> float32 and normalize per-channel.

    Should be the LAST transform in a chain, since it changes dtype.

    Specify mean and std explicitly to match your model's expected
    normalization (e.g., ImageNet, (0.5,0.5,0.5), or dataset-specific statistics).

    Parameters
    ----------
    mean : tuple[float, float, float]
        Per-channel mean.
    std : tuple[float, float, float]
        Per-channel standard deviation.
    """

    def __init__(
        self,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        if len(mean) != 3 or len(std) != 3:
            raise ValueError(
                f"mean and std must be 3-element tuples (RGB), "
                f"got mean={mean}, std={std}"
            )
        if any(s == 0 for s in std):
            raise ValueError(f"std contains zero which would cause division by zero: {std}")
        self.mean = mean
        self.std = std

    def __call__(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32) / 255.0
        mean = np.array(self.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(self.std, dtype=np.float32).reshape(1, 1, 3)
        return (img - mean) / std

    def __repr__(self) -> str:
        return f"NormalizeTransform(mean={self.mean}, std={self.std})"
