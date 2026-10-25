"""TiffSlide backend for reading whole-slide images."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from wsistream.backends.base import SlideBackend
from wsistream.types import SlideProperties

if TYPE_CHECKING:
    from tiffslide import TiffSlide


class TiffSlideBackend(SlideBackend):
    """
    Backend using the tiffslide library.

    Pure Python, no C dependencies. Supports cloud storage (S3/GCS) via fsspec.
    Requires: pip install tiffslide
    """

    def __init__(self) -> None:
        self._slide: TiffSlide | None = None
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
        # TiffSlide pads out-of-bounds pixels with zeros. Read only the part
        # inside the slide, from non-negative coordinates, onto a white canvas.
        ds = slide.level_downsamples[level]
        left = min(width, max(0, math.ceil(-x / ds)))
        top = min(height, max(0, math.ceil(-y / ds)))
        region = self._to_rgb_array(
            slide.read_region(
                (max(x, 0), max(y, 0)),
                level,
                (width - left, height - top),
                as_array=True,
                padding=False,
            )
        )
        if region.shape[:2] == (height, width):
            return region
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        canvas[top : top + region.shape[0], left : left + region.shape[1]] = region
        return canvas

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
