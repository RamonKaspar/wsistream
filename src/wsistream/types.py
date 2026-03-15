"""Shared data types used across the wsistream package."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SlideProperties:
    """Immutable metadata about a whole-slide image."""

    path: str
    dimensions: tuple[int, int]  # (width, height) at level 0
    level_count: int
    level_dimensions: tuple[tuple[int, int], ...]
    level_downsamples: tuple[float, ...]
    mpp: float | None  # microns per pixel at level 0
    vendor: str | None

    @property
    def width(self) -> int:
        return self.dimensions[0]

    @property
    def height(self) -> int:
        return self.dimensions[1]

    def mpp_at_level(self, level: int) -> float | None:
        if self.mpp is None:
            return None
        return self.mpp * self.level_downsamples[level]


@dataclass(frozen=True)
class TissueMask:
    """Binary tissue mask at a downsampled resolution."""

    mask: np.ndarray
    downsample: float
    slide_dimensions: tuple[int, int]

    @property
    def tissue_fraction(self) -> float:
        return float(self.mask.sum()) / self.mask.size

    def contains_tissue(
        self, x: int, y: int, width: int, height: int, threshold: float = 0.4
    ) -> bool:
        return self.tissue_fraction_at(x, y, width, height) >= threshold

    def tissue_fraction_at(self, x: int, y: int, width: int, height: int) -> float:
        ds = self.downsample
        mx, my = int(x / ds), int(y / ds)
        mw, mh = max(1, int(width / ds)), max(1, int(height / ds))

        h_mask, w_mask = self.mask.shape
        mx, my = min(mx, w_mask - 1), min(my, h_mask - 1)
        mx2, my2 = min(mx + mw, w_mask), min(my + mh, h_mask)

        region = self.mask[my:my2, mx:mx2]
        if region.size == 0:
            return 0.0
        return float(region.mean())


@dataclass(frozen=True)
class PatchCoordinate:
    """Location of a single patch within a slide."""

    x: int
    y: int
    level: int
    patch_size: int
    mpp: float | None
    slide_path: str


@dataclass(frozen=True)
class SlideMetadata:
    """
    Dataset-specific metadata for a slide.

    Populated by a DatasetAdapter. Generic fields are defined here;
    additional fields go in ``extra``.
    """

    slide_path: str
    dataset_name: str = "unknown"
    patient_id: str | None = None
    tissue_type: str | None = None
    cancer_type: str | None = None
    sample_type: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PatchResult:
    """A single extracted patch with all its metadata."""

    image: np.ndarray
    coordinate: PatchCoordinate
    tissue_fraction: float
    slide_metadata: SlideMetadata | None = None
