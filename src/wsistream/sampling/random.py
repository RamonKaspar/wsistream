"""Uniform random sampling from tissue regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from wsistream.sampling.base import PatchSampler
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
        Patches per slide. -1 for infinite streaming.
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
    num_patches: int = 100
    level: int = 0
    target_mpp: float | None = None
    tissue_threshold: float = 0.4
    max_retries: int = 50
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def sample(
        self, slide: SlideHandle, tissue_mask: TissueMask
    ) -> Iterator[PatchCoordinate]:
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
                x=x, y=y, level=level,
                patch_size=self.patch_size, mpp=mpp, slide_path=props.path,
            )
            count += 1
