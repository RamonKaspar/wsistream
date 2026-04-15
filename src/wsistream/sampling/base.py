"""Abstract base class for patch sampling strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Iterator

import numpy as np

from wsistream.slide import SlideHandle
from wsistream.types import PatchCoordinate, TissueMask


class CoordinatePool:
    """Pre-shuffled pool of patch coordinates for without-replacement sampling.

    Coordinates are shuffled once at construction time.  When *max_size*
    is set, a random subset of that size is selected (shuffle first,
    then truncate — so the subset is unbiased).  Calling :meth:`pop`
    consumes them in that order until the pool is exhausted.
    """

    __slots__ = ("_coords", "_total")

    def __init__(
        self,
        coordinates: list[PatchCoordinate],
        rng: np.random.Generator,
        max_size: int | None = None,
    ) -> None:
        indices = rng.permutation(len(coordinates))
        if max_size is not None and max_size < len(indices):
            indices = indices[:max_size]
        self._coords: deque[PatchCoordinate] = deque(coordinates[int(i)] for i in indices)
        self._total = len(self._coords)

    def pop(self, rng: np.random.Generator) -> PatchCoordinate | None:
        """Remove and return the next coordinate, or ``None`` if exhausted."""
        return self._coords.popleft() if self._coords else None

    @property
    def remaining(self) -> int:
        return len(self._coords)

    @property
    def exhausted(self) -> bool:
        return not self._coords

    @property
    def total(self) -> int:
        """Number of coordinates the pool was initialised with."""
        return self._total


class MultiLevelCoordinatePool:
    """Per-level coordinate pools with weighted random level selection.

    Each pyramid level maintains its own shuffled pool.  On every
    :meth:`pop`, a level is chosen from the non-empty levels that have
    positive weight, using the original weights renormalised over those
    active levels.  One coordinate is then consumed from the chosen level.

    Levels with zero weight are excluded entirely: their coordinates are
    not stored, and they do not contribute to :attr:`total` or
    :attr:`remaining`.

    Parameters
    ----------
    level_pools : dict[int, list[PatchCoordinate]]
        Coordinates per pyramid level.
    level_weights : dict[int, float]
        Sampling weight per level.  Levels with zero weight are dropped.
    rng : numpy.random.Generator
        Used to shuffle each level's pool.
    max_total : int or None
        If set, caps both stored coordinates (per level) and total pops.
        Each level is truncated to *max_total* after shuffling (since no
        single level can contribute more than *max_total* pops), and
        :meth:`pop` returns ``None`` after *max_total* coordinates have
        been consumed across all levels.
    """

    __slots__ = ("_pools", "_levels", "_base_weights", "_total", "_max_total", "_pop_count")

    def __init__(
        self,
        level_pools: dict[int, list[PatchCoordinate]],
        level_weights: dict[int, float],
        rng: np.random.Generator,
        max_total: int | None = None,
    ) -> None:
        self._pools: dict[int, deque[PatchCoordinate]] = {}
        self._total = 0
        for level, coords in level_pools.items():
            if level_weights.get(level, 0) <= 0:
                continue
            indices = rng.permutation(len(coords))
            if max_total is not None and max_total < len(indices):
                indices = indices[:max_total]
            self._pools[level] = deque(coords[int(i)] for i in indices)
            self._total += len(self._pools[level])
        self._levels = sorted(self._pools.keys())
        self._base_weights = {lvl: level_weights[lvl] for lvl in self._levels}
        self._max_total = max_total
        self._pop_count = 0

    def pop(self, rng: np.random.Generator) -> PatchCoordinate | None:
        """Choose an active level using renormalised weights, pop one coordinate."""
        if self._max_total is not None and self._pop_count >= self._max_total:
            return None

        active = [lvl for lvl in self._levels if self._pools[lvl]]
        if not active:
            return None

        weights = np.array([self._base_weights[lvl] for lvl in active])
        weights /= weights.sum()

        idx = int(rng.choice(len(active), p=weights))
        self._pop_count += 1
        return self._pools[active[idx]].popleft()

    @property
    def remaining(self) -> int:
        reachable = sum(len(self._pools[lvl]) for lvl in self._levels)
        if self._max_total is not None:
            return min(self._max_total - self._pop_count, reachable)
        return reachable

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    @property
    def total(self) -> int:
        """Reachable coordinates at construction time (excludes zero-weight levels)."""
        return self._total


def enumerate_grid_coordinates(
    slide: SlideHandle,
    tissue_mask: TissueMask,
    level: int,
    patch_size: int,
    tissue_threshold: float,
) -> list[PatchCoordinate]:
    """Enumerate all non-overlapping grid positions that pass the tissue threshold."""
    props = slide.properties
    ds = props.level_downsamples[level]
    patch_l0 = int(patch_size * ds)
    mpp = props.mpp_at_level(level)

    coordinates: list[PatchCoordinate] = []
    for y in range(0, props.height - patch_l0 + 1, patch_l0):
        for x in range(0, props.width - patch_l0 + 1, patch_l0):
            if tissue_mask.contains_tissue(x, y, patch_l0, patch_l0, tissue_threshold):
                coordinates.append(
                    PatchCoordinate(
                        x=x,
                        y=y,
                        level=level,
                        patch_size=patch_size,
                        mpp=mpp,
                        slide_path=props.path,
                    )
                )
    return coordinates


class PatchSampler(ABC):
    """
    Sample patch coordinates from a slide.
    """

    @abstractmethod
    def sample(self, slide: SlideHandle, tissue_mask: TissueMask) -> Iterator[PatchCoordinate]:
        """
        Yield patch coordinates from a slide.

        Parameters
        ----------
        slide : SlideHandle
            The opened whole-slide image.
        tissue_mask : TissueMask
            Binary mask indicating tissue regions.

        Yields
        ------
        PatchCoordinate
        """
        ...

    def build_coordinate_pool(
        self,
        slide: SlideHandle,
        tissue_mask: TissueMask,
        rng: np.random.Generator,
    ) -> CoordinatePool | MultiLevelCoordinatePool:
        """Build a finite pool of coordinates for without-replacement sampling.

        Override this method in subclasses that support
        ``replacement="without_replacement"`` in :class:`PatchPipeline`.
        The default implementation raises :class:`NotImplementedError`;
        the pipeline detects this and raises :class:`TypeError` at
        construction time.

        Parameters
        ----------
        slide : SlideHandle
            The opened whole-slide image.
        tissue_mask : TissueMask
            Binary mask indicating tissue regions.
        rng : numpy.random.Generator
            RNG used to shuffle the pool.

        Returns
        -------
        CoordinatePool or MultiLevelCoordinatePool
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support without-replacement sampling"
        )
