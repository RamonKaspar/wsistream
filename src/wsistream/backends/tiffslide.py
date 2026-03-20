"""TiffSlide backend for reading whole-slide images."""

from __future__ import annotations

import numpy as np

from wsistream.backends.base import SlideBackend
from wsistream.types import SlideProperties


class TiffSlideBackend(SlideBackend):
    """
    Backend using the tiffslide library.

    Pure Python, no C dependencies. Supports cloud storage (S3/GCS) via fsspec.
    Requires: pip install tiffslide
    """

    def __init__(self) -> None:
        self._slide = None
        self._path: str | None = None

    def open(self, path: str) -> None:
        from tiffslide import TiffSlide

        self._path = path
        self._slide = TiffSlide(path)

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
        # TiffSlide v3+ uses "tiffslide.*" property keys, not "openslide.*".
        # Fall back to openslide keys for older versions or slides opened
        # via openslide-compatible property dicts.
        mpp = (
            self._safe_float(s.properties.get("tiffslide.mpp-x"))
            or self._safe_float(s.properties.get("openslide.mpp-x"))
        )
        vendor = (
            s.properties.get("tiffslide.vendor")
            or s.properties.get("openslide.vendor")
        )
        return SlideProperties(
            path=self._path,
            dimensions=s.dimensions,
            level_count=s.level_count,
            level_dimensions=tuple(s.level_dimensions),
            level_downsamples=tuple(s.level_downsamples),
            mpp=mpp,
            vendor=vendor,
        )

    def __repr__(self) -> str:
        return "TiffSlideBackend()"
