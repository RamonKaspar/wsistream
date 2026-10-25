"""Shared data types used across the wsistream package."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Sequence

import numpy as np

WSI_EXTENSIONS = {".svs", ".tiff", ".tif", ".ndpi", ".mrxs", ".vms", ".scn", ".bif"}


def resolve_slide_paths(slide_paths: str | Path | Sequence[str | Path]) -> list[str]:
    """Resolve slide paths from a directory, a single file, or a list.

    Accepts a directory path, a single file path, or a list of file/directory
    paths.  Directories are expanded recursively to all WSI files inside them.
    URI strings are preserved for backends that support remote storage.
    """
    if isinstance(slide_paths, (str, Path)):
        slide_paths = [slide_paths]

    resolved: list[str] = []
    for entry in slide_paths:
        # Path normalisation corrupts URI schemes (``s3://`` becomes ``s3:/``),
        # and remote existence is the responsibility of the selected backend.
        if isinstance(entry, str) and "://" in entry:
            resolved.append(entry)
            continue

        p = Path(entry)
        if p.is_dir():
            found = sorted(
                str(f) for f in p.rglob("*") if f.is_file() and f.suffix.lower() in WSI_EXTENSIONS
            )
            if not found:
                raise FileNotFoundError(f"No WSI files found in {p}")
            resolved.extend(found)
        elif p.is_file():
            resolved.append(str(p))
        else:
            raise FileNotFoundError(f"Slide path does not exist or is not a file: {p}")
    return resolved


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

    def tissue_fraction_grid(
        self,
        width: int,
        height: int,
        cells_x: np.ndarray | None = None,
        cells_y: np.ndarray | None = None,
    ) -> np.ndarray:
        """Tissue fraction of a ``width x height`` footprint anchored in every mask cell.

        :meth:`tissue_fraction_at` depends on a level-0 position only through
        the mask cell that contains it. Entry ``[my, mx]`` of the returned
        ``(H, W)`` array equals ``tissue_fraction_at(x, y, width, height)`` for
        every position whose top-left corner falls in cell ``(mx, my)``, with
        positions past the mask edge assigned to the last cell, as there.
        ``cells_x`` and ``cells_y`` restrict the result to those cell columns
        and rows, which is much cheaper than the full grid for a few cells.
        """
        ds = self.downsample
        mw, mh = max(1, int(width / ds)), max(1, int(height / ds))
        h_mask, w_mask = self.mask.shape
        x1 = np.arange(w_mask) if cells_x is None else np.asarray(cells_x)
        y1 = np.arange(h_mask) if cells_y is None else np.asarray(cells_y)
        x2, y2 = np.minimum(x1 + mw, w_mask), np.minimum(y1 + mh, h_mask)

        # Box sums from cumulative sums, one axis at a time.
        cumulative = np.zeros((h_mask, w_mask + 1), dtype=np.int64)
        np.cumsum(self.mask, axis=1, dtype=np.int64, out=cumulative[:, 1:])
        row_sums = cumulative[:, x2] - cumulative[:, x1]
        cumulative = np.zeros((h_mask + 1, len(x1)), dtype=np.int64)
        np.cumsum(row_sums, axis=0, out=cumulative[1:])
        sums = cumulative[y2] - cumulative[y1]
        return sums / np.outer(y2 - y1, x2 - x1)


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

    def to_flat_dict(self) -> dict[str, str]:
        """Return a flat dict of strings suitable for batched collation."""
        return {
            f.name: json.dumps(getattr(self, f.name))
            if f.name == "extra"
            else (getattr(self, f.name) or "")
            for f in fields(self)
        }

    @classmethod
    def empty_dict(cls) -> dict[str, str]:
        """Return a flat dict with empty defaults for every field."""
        return {f.name: "{}" if f.name == "extra" else "" for f in fields(cls)}


@dataclass(frozen=True)
class PatchResult:
    """A single extracted patch with all its metadata.

    Exactly one of ``image`` or ``views`` must be non-None.

    Attributes
    ----------
    image : np.ndarray or None
        The extracted patch in single-view mode.  Shape ``(H, W, 3)``,
        dtype ``uint8`` unless a ``NormalizeTransform`` was applied (then
        ``float32``).  ``None`` when ``views`` is configured.
    coordinate : PatchCoordinate
        Level-0 location of the patch in the slide.
    tissue_fraction : float
        Fraction of the patch region covered by tissue, in ``[0, 1]``.
    slide_metadata : SlideMetadata or None
        Dataset-specific metadata when a ``DatasetAdapter`` is configured.
    views : dict[str, np.ndarray] or None
        Named view outputs in multi-view mode.  Keys are the view names
        after ``count`` expansion (e.g. ``"global_0"``, ``"local_0"``).
        Each value has shape ``(H_view, W_view, 3)`` where the spatial
        dimensions are determined by each view's ``RandomResizedCrop.size``
        (or the primary patch size if no crop is applied).  dtype is
        ``uint8`` unless ``NormalizeTransform`` is used.  ``None`` in
        single-view mode.
    """

    image: np.ndarray | None
    coordinate: PatchCoordinate
    tissue_fraction: float
    slide_metadata: SlideMetadata | None = None
    views: dict[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        if self.image is None and self.views is None:
            raise ValueError("PatchResult must have either image or views set, not neither")
        if self.image is not None and self.views is not None:
            raise ValueError("PatchResult cannot have both image and views set simultaneously")
        if self.views is not None and not self.views:
            raise ValueError("PatchResult.views must contain at least one view")
