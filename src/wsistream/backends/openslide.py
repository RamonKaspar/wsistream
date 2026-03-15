"""OpenSlide backend for reading whole-slide images."""

from __future__ import annotations

import numpy as np

from wsistream.backends.base import SlideBackend
from wsistream.types import SlideProperties


class OpenSlideBackend(SlideBackend):
    """
    Backend using the openslide-python library.

    Requires: pip install openslide-python
    Plus the OpenSlide C library installed on the system.
    """

    def __init__(self) -> None:
        self._slide = None
        self._path: str | None = None

    def open(self, path: str) -> None:
        from openslide import OpenSlide

        self._path = path
        self._slide = OpenSlide(path)

    def close(self) -> None:
        if self._slide is not None:
            self._slide.close()
            self._slide = None

    def read_region(self, x: int, y: int, level: int, width: int, height: int) -> np.ndarray:
        region = self._slide.read_region((x, y), level, (width, height))
        arr = np.asarray(region)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr

    def get_thumbnail(self, size: tuple[int, int]) -> np.ndarray:
        thumb = self._slide.get_thumbnail(size)
        arr = np.asarray(thumb)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr

    def get_properties(self) -> SlideProperties:
        s = self._slide
        mpp = self._safe_float(s.properties.get("openslide.mpp-x"))
        return SlideProperties(
            path=self._path,
            dimensions=s.dimensions,
            level_count=s.level_count,
            level_dimensions=tuple(s.level_dimensions),
            level_downsamples=tuple(s.level_downsamples),
            mpp=mpp,
            vendor=s.properties.get("openslide.vendor"),
        )

    @staticmethod
    def _safe_float(val: str | None) -> float | None:
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def __repr__(self) -> str:
        return "OpenSlideBackend()"
