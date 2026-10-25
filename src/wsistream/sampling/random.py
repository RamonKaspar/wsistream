"""Uniform random sampling from tissue regions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from wsistream.sampling.base import (
    CoordinatePool,
    PatchSampler,
    enumerate_grid_coordinates,
    resolve_mpp_read,
)
from wsistream.slide import SlideHandle
from wsistream.types import PatchCoordinate, TissueMask


@dataclass
class RandomSampler(PatchSampler):
    """
    Rejection-sample random patches from tissue regions.

    This is the core online patching approach: draw random (x, y),
    reject if outside tissue, yield if inside. Gives virtually
    infinite data diversity without saving patches to disk.

    Parameters
    ----------
    patch_size : int
        Width and height at the target level.
    num_patches : int
        Patches per slide. -1 (default) for infinite streaming;
        the pipeline's ``patches_per_slide`` controls the budget.
    level : int
        Pyramid level to sample from. Ignored when ``target_mpp`` is set.
    target_mpp : float or None
        Desired µm/px. When set, each slide is read at the pyramid level
        within ``mpp_tolerance`` of this value, or, when no level is that
        close, from a finer level that the pipeline shrinks to
        ``patch_size`` (see :func:`~wsistream.sampling.resolve_mpp_read`).
        ``level`` is ignored. Slides without MPP metadata are read at level 0.
    tissue_threshold : float
        Minimum tissue fraction to accept a candidate.
    max_retries : int
        Rejection attempts before giving up on one patch.
    seed : int or None
        Random seed.
    mpp_tolerance : float
        Maximum relative deviation between a pyramid level's mpp and
        ``target_mpp`` for reading that level directly. Default: 0.1.
        ``float("inf")`` always reads the closest level directly.
    """

    patch_size: int = 256
    num_patches: int = -1
    level: int = 0
    target_mpp: float | None = None
    tissue_threshold: float = 0.4
    max_retries: int = 50
    seed: int | None = None
    mpp_tolerance: float = 0.1

    def __post_init__(self) -> None:
        if self.patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {self.patch_size}")
        if self.num_patches < -1 or self.num_patches == 0:
            raise ValueError(f"num_patches must be -1 (infinite) or >= 1, got {self.num_patches}")
        if self.target_mpp is not None and self.target_mpp <= 0:
            raise ValueError(f"target_mpp must be > 0, got {self.target_mpp}")
        if self.max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {self.max_retries}")
        if math.isnan(self.mpp_tolerance) or self.mpp_tolerance < 0:
            raise ValueError(f"mpp_tolerance must be >= 0, got {self.mpp_tolerance}")
        self._rng = np.random.default_rng(self.seed)

    @property
    def output_size(self) -> int:
        """Side length of yielded patches; the pipeline resizes reads of another size to it."""
        return self.patch_size

    def sample(self, slide: SlideHandle, tissue_mask: TissueMask) -> Iterator[PatchCoordinate]:
        rng = self._rng
        props = slide.properties

        level, read_size, mpp = self._resolve_read(slide)
        ds = props.level_downsamples[level]
        patch_size_l0 = int(read_size * ds)
        max_x = props.width - patch_size_l0
        max_y = props.height - patch_size_l0

        if max_x < 0 or max_y < 0:
            return

        count = 0
        infinite = self.num_patches == -1

        while infinite or count < self.num_patches:
            found = False
            for _ in range(self.max_retries):
                x = int(rng.integers(0, max_x + 1))
                y = int(rng.integers(0, max_y + 1))
                if tissue_mask.contains_tissue(
                    x, y, patch_size_l0, patch_size_l0, self.tissue_threshold
                ):
                    found = True
                    break

            if not found:
                break

            yield PatchCoordinate(
                x=x,
                y=y,
                level=level,
                patch_size=read_size,
                mpp=mpp,
                slide_path=props.path,
            )
            count += 1

    def build_coordinate_pool(
        self,
        slide: SlideHandle,
        tissue_mask: TissueMask,
        rng: np.random.Generator,
    ) -> CoordinatePool:
        """Build a shuffled pool of all valid grid coordinates for this slide.

        Used by :class:`~wsistream.pipeline.PatchPipeline` when
        ``replacement="without_replacement"``.
        """
        level, read_size, mpp = self._resolve_read(slide)
        coordinates = enumerate_grid_coordinates(
            slide, tissue_mask, level, read_size, self.tissue_threshold, mpp=mpp
        )
        max_size = None if self.num_patches == -1 else self.num_patches
        return CoordinatePool(coordinates, rng, max_size=max_size)

    def _resolve_read(self, slide: SlideHandle) -> tuple[int, int, float | None]:
        """Return the ``(level, read_size, mpp)`` used for every patch of a slide."""
        props = slide.properties
        if self.target_mpp is not None and props.mpp is not None:
            return resolve_mpp_read(props, self.target_mpp, self.patch_size, self.mpp_tolerance)

        # target_mpp cannot be honoured without MPP metadata, so read level 0.
        level = self.level if self.target_mpp is None else 0
        if level < 0 or level >= props.level_count:
            raise ValueError(
                f"level={level} is out of range for slide with "
                f"{props.level_count} levels (path={props.path!r})"
            )
        return level, self.patch_size, props.mpp_at_level(level)
