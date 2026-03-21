"""Abstract base class for post-extraction patch filters."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PatchFilter(ABC):
    """
    Accept or reject a patch after extraction, before transforms.

    This is where per-tile quality checks belong — checks that need
    the actual patch pixels, not just the thumbnail.
    """

    @abstractmethod
    def accept(self, patch: np.ndarray) -> bool:
        """
        Return True to keep the patch, False to discard it.

        Parameters
        ----------
        patch : np.ndarray
            RGB image, shape (H, W, 3), dtype uint8.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
