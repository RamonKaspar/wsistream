"""Abstract base for WSI reader backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from wsistream.types import SlideProperties


class SlideBackend(ABC):
    """
    Abstract base class for WSI reader backends.

    Implement this to add support for a new slide reading library
    (e.g., cucim, bioformats, etc.)
    """

    @abstractmethod
    def open(self, path: str) -> None:
        """Open a slide file."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources."""
        ...

    @abstractmethod
    def read_region(self, x: int, y: int, level: int, width: int, height: int) -> np.ndarray:
        """Read a region as an RGB uint8 numpy array of shape (H, W, 3)."""
        ...

    @abstractmethod
    def get_thumbnail(self, size: tuple[int, int]) -> np.ndarray:
        """Return a low-resolution RGB thumbnail."""
        ...

    @abstractmethod
    def get_properties(self) -> SlideProperties:
        """Extract slide metadata."""
        ...

    @staticmethod
    def _safe_float(val: str | None) -> float | None:
        """Parse a string to float, returning None on failure."""
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
