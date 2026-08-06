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
        slide = self._require_open_slide(self._slide)
        region = slide.read_region((x, y), level, (width, height))
        return self._to_rgb_array(region)

    def get_thumbnail(self, size: tuple[int, int]) -> np.ndarray:
        slide = self._require_open_slide(self._slide)
        return self._to_rgb_array(slide.get_thumbnail(size))

    def get_properties(self) -> SlideProperties:
        s = self._require_open_slide(self._slide)
        assert self._path is not None
        # TiffSlide v3+ uses "tiffslide.*" property keys, not "openslide.*".
        # Fall back to openslide keys for older versions or slides opened
        # via openslide-compatible property dicts.
        mpp = self._safe_float(s.properties.get("tiffslide.mpp-x")) or self._safe_float(
            s.properties.get("openslide.mpp-x")
        )
        vendor = s.properties.get("tiffslide.vendor") or s.properties.get("openslide.vendor")
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
