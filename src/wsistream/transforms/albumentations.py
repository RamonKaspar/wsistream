"""Wrapper for albumentations augmentation pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from wsistream.transforms.base import PatchTransform


@dataclass
class AlbumentationsWrapper(PatchTransform):
    """
    Wrap any albumentations.Compose as a PatchTransform.

    Example
    -------
    >>> import albumentations as A
    >>> aug = AlbumentationsWrapper(A.Compose([
    ...     A.ColorJitter(brightness=0.2, contrast=0.2),
    ...     A.GaussianBlur(blur_limit=3, p=0.3),
    ... ]))
    """

    transform: Any = None

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.transform is None:
            return image
        return self.transform(image=image)["image"]

    def __repr__(self) -> str:
        return f"AlbumentationsWrapper({self.transform!r})"
