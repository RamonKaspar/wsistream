"""Uniform random sampling from tissue regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from wsistream.sampling.base import CoordinatePool, PatchSampler, enumerate_grid_coordinates
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
        Desired µm/px. When set, the sampler picks the pyramid level
        closest to this value for each slide (via
        ``SlideHandle.best_level_for_mpp``), ignoring ``level``.
    tissue_threshold : float
        Minimum tissue fraction to accept a candidate.
    max_retries : int
        Rejection attempts before giving up on one patch.
    seed : int or None
        Random seed.
    """

    patch_size: int = 256
    num_patches: int = -1
    level: int = 0
    target_mpp: float | None = None
    tissue_threshold: float = 0.4
    max_retries: int = 50
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {self.patch_size}")
        if self.num_patches < -1 or self.num_patches == 0:
            raise ValueError(f"num_patches must be -1 (infinite) or >= 1, got {self.num_patches}")
        if self.target_mpp is not None and self.target_mpp <= 0:
            raise ValueError(f"target_mpp must be > 0, got {self.target_mpp}")
        if self.max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {self.max_retries}")
        self._rng = np.random.default_rng(self.seed)

    def sample(self, slide: SlideHandle, tissue_mask: TissueMask) -> Iterator[PatchCoordinate]:
        rng = self._rng
        props = slide.properties

        # Resolve level: target_mpp takes priority over fixed level
        if self.target_mpp is not None:
            level = slide.best_level_for_mpp(self.target_mpp)
        else:
            level = self.level

        if level < 0 or level >= props.level_count:
            raise ValueError(
                f"level={level} is out of range for slide with "
                f"{props.level_count} levels (path={props.path!r})"
            )

        ds = props.level_downsamples[level]
        patch_size_l0 = int(self.patch_size * ds)
        max_x = props.width - patch_size_l0
        max_y = props.height - patch_size_l0

        if max_x < 0 or max_y < 0:
            return

        mpp = props.mpp_at_level(level)
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
                patch_size=self.patch_size,
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
        if self.target_mpp is not None:
            level = slide.best_level_for_mpp(self.target_mpp)
        else:
            level = self.level

        props = slide.properties
        if level < 0 or level >= props.level_count:
            raise ValueError(
                f"level={level} is out of range for slide with "
                f"{props.level_count} levels (path={props.path!r})"
            )

        coordinates = enumerate_grid_coordinates(
            slide, tissue_mask, level, self.patch_size, self.tissue_threshold
        )
        max_size = None if self.num_patches == -1 else self.num_patches
        return CoordinatePool(coordinates, rng, max_size=max_size)
