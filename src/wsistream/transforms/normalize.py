"""Per-channel mean/std normalization."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from wsistream.transforms.base import PatchTransform

# Sentinel to detect missing arguments
_MISSING = object()


@dataclass
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

    mean: tuple[float, float, float] = _MISSING  # type: ignore[assignment]
    std: tuple[float, float, float] = _MISSING  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.mean is _MISSING or self.std is _MISSING:
            raise TypeError(
                "NormalizeTransform requires explicit mean and std. "
                "Example: NormalizeTransform(mean=(0.485, 0.456, 0.406), "
                "std=(0.229, 0.224, 0.225))"
            )
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError(
                f"mean and std must be 3-element tuples (RGB), "
                f"got mean={self.mean}, std={self.std}"
            )
        if any(s == 0 for s in self.std):
            raise ValueError(f"std contains zero which would cause division by zero: {self.std}")

    def __call__(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32) / 255.0
        mean = np.array(self.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(self.std, dtype=np.float32).reshape(1, 1, 3)
        return (img - mean) / std
