"""Unified slide handle wrapping an explicit backend."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from wsistream.backends.base import SlideBackend
from wsistream.types import SlideProperties

logger = logging.getLogger(__name__)


class SlideHandle:
    """
    Handle for reading regions from a whole-slide image.

    Parameters
    ----------
    path : str or Path
        Path to the WSI file.
    backend : SlideBackend
        An instantiated backend. No fallback, no auto-detection.

    Examples
    --------
    >>> from wsistream.backends import OpenSlideBackend
    >>> slide = SlideHandle("tumor_001.svs", backend=OpenSlideBackend())
    >>> patch = slide.read_region(x=1000, y=2000, width=256, height=256)
    """

    def __init__(self, path: str | Path, backend: SlideBackend) -> None:
        self._path = str(path)
        self._backend = backend
        self._backend.open(self._path)
        try:
            self._properties = self._backend.get_properties()
        except Exception:
            self._backend.close()
            raise
        logger.debug("Opened %s (%dx%d, %d levels, mpp=%s)",
                     self._path, *self._properties.dimensions,
                     self._properties.level_count, self._properties.mpp)

    @property
    def properties(self) -> SlideProperties:
        return self._properties

    def read_region(
        self, x: int, y: int, width: int, height: int, level: int = 0
    ) -> np.ndarray:
        """Read a region as an RGB numpy array (H, W, 3), uint8."""
        return self._backend.read_region(x, y, level, width, height)

    def get_thumbnail(self, size: tuple[int, int] = (512, 512)) -> np.ndarray:
        """Return a low-resolution RGB thumbnail."""
        return self._backend.get_thumbnail(size)

    def best_level_for_mpp(self, target_mpp: float) -> int:
        """Find the pyramid level closest to the desired microns-per-pixel."""
        if self._properties.mpp is None:
            return 0
        best_level, best_diff = 0, float("inf")
        for lvl in range(self._properties.level_count):
            lvl_mpp = self._properties.mpp * self._properties.level_downsamples[lvl]
            diff = abs(lvl_mpp - target_mpp)
            if diff < best_diff:
                best_diff = diff
                best_level = lvl
        return best_level

    def close(self) -> None:
        logger.debug("Closing %s", self._path)
        self._backend.close()

    def __enter__(self) -> SlideHandle:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __repr__(self) -> str:
        w, h = self._properties.dimensions
        return (
            f"SlideHandle({self._path!r}, {w}x{h}, "
            f"levels={self._properties.level_count}, "
            f"mpp={self._properties.mpp}, backend={self._backend!r})"
        )
