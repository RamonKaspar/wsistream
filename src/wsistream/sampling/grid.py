"""Non-overlapping grid sampling with tissue filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from wsistream.sampling.base import PatchSampler
from wsistream.slide import SlideHandle
from wsistream.types import PatchCoordinate, TissueMask


@dataclass
class GridSampler(PatchSampler):
    """
    Exhaustive grid sampling, keeping only patches with enough tissue.

    Useful for deterministic feature extraction (CLAM-style).
    """

    patch_size: int = 256
    level: int = 0
    stride: int | None = None  # defaults to patch_size (non-overlapping)
    tissue_threshold: float = 0.4

    def sample(
        self, slide: SlideHandle, tissue_mask: TissueMask
    ) -> Iterator[PatchCoordinate]:
        props = slide.properties
        if self.level < 0 or self.level >= props.level_count:
            raise ValueError(
                f"level={self.level} is out of range for slide with "
                f"{props.level_count} levels (path={props.path!r})"
            )
        stride = self.stride or self.patch_size
        ds = props.level_downsamples[self.level]
        patch_l0 = int(self.patch_size * ds)
        stride_l0 = int(stride * ds)
        mpp = props.mpp_at_level(self.level)

        for y in range(0, props.height - patch_l0 + 1, stride_l0):
            for x in range(0, props.width - patch_l0 + 1, stride_l0):
                if tissue_mask.contains_tissue(x, y, patch_l0, patch_l0, self.tissue_threshold):
                    yield PatchCoordinate(
                        x=x, y=y, level=self.level,
                        patch_size=self.patch_size, mpp=mpp, slide_path=props.path,
                    )
