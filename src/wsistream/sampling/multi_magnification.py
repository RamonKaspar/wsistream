"""Sample patches at multiple magnification levels."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from wsistream.sampling.base import PatchSampler
from wsistream.sampling.random import RandomSampler
from wsistream.slide import SlideHandle
from wsistream.types import PatchCoordinate, TissueMask

logger = logging.getLogger(__name__)


@dataclass
class MultiMagnificationSampler(PatchSampler):
    """
    Randomly pick a magnification level, then sample a random patch at it.

    Midnight trains at 0.25, 0.5, 1.0, and 2.0 µm/px. Each iteration
    randomly selects one of these levels.

    Parameters
    ----------
    target_mpps : list[float]
        Desired µm/px values. Default matches Midnight.
    mpp_weights : list[float] or None
        Sampling probability per level. None = uniform.
    patch_size : int
        Patch width/height at the target level.
    num_patches : int
        Total patches to yield. -1 for infinite.
    tissue_threshold : float
        Minimum tissue fraction.
    max_consecutive_failures : int
        Stop after this many consecutive failed attempts to find a
        tissue patch across all levels. Prevents infinite loops on
        slides with very little tissue.
    seed : int or None
        Random seed.
    """

    target_mpps: list[float] = field(default_factory=lambda: [0.25, 0.5, 1.0, 2.0])
    mpp_weights: list[float] | None = None
    patch_size: int = 256
    num_patches: int = 100
    tissue_threshold: float = 0.4
    max_consecutive_failures: int = 100
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def sample(
        self, slide: SlideHandle, tissue_mask: TissueMask
    ) -> Iterator[PatchCoordinate]:
        rng = self._rng

        if slide.properties.mpp is None:
            # No MPP metadata: fall back to single-level random sampling.
            # Use a seed derived from the persistent RNG so that repeated
            # calls (e.g. cycle mode) produce different coordinates.
            logger.warning(
                "Slide %s has no MPP metadata; falling back to level-0 random sampling",
                slide.properties.path if hasattr(slide.properties, 'path') else "unknown",
            )
            inner = RandomSampler(
                patch_size=self.patch_size, num_patches=self.num_patches,
                level=0, tissue_threshold=self.tissue_threshold,
                seed=int(rng.integers(0, 2**31)),
            )
            yield from inner.sample(slide, tissue_mask)
            return

        weights = self.mpp_weights
        if weights is None:
            weights = [1.0 / len(self.target_mpps)] * len(self.target_mpps)
        else:
            total = sum(weights)
            weights = [w / total for w in weights]

        count = 0
        consecutive_failures = 0
        infinite = self.num_patches == -1

        while infinite or count < self.num_patches:
            idx = rng.choice(len(self.target_mpps), p=weights)
            level = slide.best_level_for_mpp(self.target_mpps[idx])

            inner = RandomSampler(
                patch_size=self.patch_size, num_patches=1, level=level,
                tissue_threshold=self.tissue_threshold,
                seed=int(rng.integers(0, 2**31)),
            )

            found = False
            for coord in inner.sample(slide, tissue_mask):
                yield coord
                count += 1
                consecutive_failures = 0
                found = True
                break

            if not found:
                consecutive_failures += 1
                if consecutive_failures >= self.max_consecutive_failures:
                    break
