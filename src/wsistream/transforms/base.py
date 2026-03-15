"""Abstract base class and composition for patch transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


class PatchTransform(ABC):
    """
    Transform applied to a patch image.

    All transforms operate on numpy arrays (H, W, 3), uint8.
    Convention: transforms preserve uint8 unless they are explicitly
    a normalization step (which outputs float32 and should be last).
    """

    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@dataclass
class ComposeTransforms(PatchTransform):
    """Apply a list of transforms sequentially."""

    transforms: list[PatchTransform] = field(default_factory=list)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            image = t(image)
        return image

    def __repr__(self) -> str:
        inner = ", ".join(repr(t) for t in self.transforms)
        return f"ComposeTransforms([{inner}])"
