"""Abstract base class for patch sampling strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Generic, Iterator, TypeVar

import numpy as np

from wsistream.slide import SlideHandle
from wsistream.types import PatchCoordinate, SlideProperties, TissueMask

_PoolKeyT = TypeVar("_PoolKeyT", int, tuple[int, int])


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


class _GridPool(CoordinatePool):
    """Pool of the grid positions that pass the tissue threshold, kept as arrays.

    Pops the same coordinates in the same order, for the same draws from
    *rng*, as ``CoordinatePool(enumerate_grid_coordinates(...), rng, max_size)``,
    but stores 16 bytes per coordinate instead of a :class:`PatchCoordinate`
    object and builds each coordinate when it is popped.  The pipeline keeps
    the pool of every slide it opens, so this matters for many slides.
    """

    __slots__ = ("_x", "_y", "_fields", "_next")

    def __init__(
        self,
        slide: SlideHandle,
        tissue_mask: TissueMask,
        level: int,
        patch_size: int,
        tissue_threshold: float,
        rng: np.random.Generator,
        max_size: int | None = None,
        mpp: float | None = None,
    ) -> None:
        props = slide.properties
        if mpp is None:
            mpp = props.mpp_at_level(level)
        xs, ys, rows, cols = _tissue_grid(slide, tissue_mask, level, patch_size, tissue_threshold)
        indices = rng.permutation(len(rows))
        if max_size is not None and max_size < len(indices):
            indices = indices[:max_size]
        self._x = xs[cols[indices]]
        self._y = ys[rows[indices]]
        self._fields = (level, patch_size, mpp, props.path)
        self._next = 0
        self._total = len(indices)

    def pop(self, rng: np.random.Generator) -> PatchCoordinate | None:
        """Remove and return the next coordinate, or ``None`` if exhausted."""
        i = self._next
        if i >= self._total:
            return None
        self._next = i + 1
        level, patch_size, mpp, slide_path = self._fields
        return PatchCoordinate(
            x=int(self._x[i]),
            y=int(self._y[i]),
            level=level,
            patch_size=patch_size,
            mpp=mpp,
            slide_path=slide_path,
        )

    @property
    def remaining(self) -> int:
        return self._total - self._next

    @property
    def exhausted(self) -> bool:
        return self._next >= self._total


class MultiLevelCoordinatePool(Generic[_PoolKeyT]):
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
    level_pools : dict[int, list[PatchCoordinate]] or dict[tuple[int, int], list[PatchCoordinate]]
        Coordinates per pyramid level.  Keys may instead be
        ``(level, read_size)`` pairs that keep several reads of one level
        apart, as in :meth:`MultiMagnificationSampler.build_coordinate_pool`.
    level_weights : dict[int, float] or dict[tuple[int, int], float]
        Sampling weight per key of *level_pools*.  Levels with zero weight
        are dropped.
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
        level_pools: dict[_PoolKeyT, list[PatchCoordinate]],
        level_weights: dict[_PoolKeyT, float],
        rng: np.random.Generator,
        max_total: int | None = None,
    ) -> None:
        pools = {
            level: CoordinatePool(coords, rng, max_size=max_total)
            for level, coords in level_pools.items()
            if level_weights.get(level, 0) > 0
        }
        self._setup(pools, level_weights, max_total)

    @classmethod
    def _from_pools(
        cls,
        pools: dict[_PoolKeyT, CoordinatePool],
        level_weights: dict[_PoolKeyT, float],
        max_total: int | None,
    ) -> MultiLevelCoordinatePool[_PoolKeyT]:
        """Combine pools of the keys with positive weight, each capped at *max_total*."""
        pool = cls.__new__(cls)
        pool._setup(pools, level_weights, max_total)
        return pool

    def _setup(
        self,
        pools: dict[_PoolKeyT, CoordinatePool],
        level_weights: dict[_PoolKeyT, float],
        max_total: int | None,
    ) -> None:
        self._pools: dict[_PoolKeyT, CoordinatePool] = pools
        self._total = sum(pool.total for pool in pools.values())
        self._levels: list[_PoolKeyT] = sorted(pools.keys())
        self._base_weights: dict[_PoolKeyT, float] = {
            lvl: level_weights[lvl] for lvl in self._levels
        }
        self._max_total = max_total
        self._pop_count = 0

    def pop(self, rng: np.random.Generator) -> PatchCoordinate | None:
        """Choose an active level using renormalised weights, pop one coordinate."""
        if self._max_total is not None and self._pop_count >= self._max_total:
            return None

        active = [lvl for lvl in self._levels if not self._pools[lvl].exhausted]
        if not active:
            return None

        weights = np.array([self._base_weights[lvl] for lvl in active])
        weights /= weights.sum()

        idx = int(rng.choice(len(active), p=weights))
        self._pop_count += 1
        return self._pools[active[idx]].pop(rng)

    @property
    def remaining(self) -> int:
        reachable = sum(self._pools[lvl].remaining for lvl in self._levels)
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
    mpp: float | None = None,
) -> list[PatchCoordinate]:
    """Enumerate all non-overlapping grid positions that pass the tissue threshold.

    ``patch_size`` is the side length read at ``level``.  ``mpp`` is recorded
    in every coordinate and defaults to the mpp of ``level``; pass the
    effective mpp when the read region will be resized.
    """
    props = slide.properties
    if mpp is None:
        mpp = props.mpp_at_level(level)
    xs, ys, rows, cols = _tissue_grid(slide, tissue_mask, level, patch_size, tissue_threshold)

    # Coordinates in one row or column share their Python ints, which keeps
    # large lists small.
    x_values, y_values = xs.tolist(), ys.tolist()
    return [
        PatchCoordinate(
            x=x_values[col],
            y=y_values[row],
            level=level,
            patch_size=patch_size,
            mpp=mpp,
            slide_path=props.path,
        )
        for row, col in zip(rows.tolist(), cols.tolist())
    ]


def _tissue_grid(
    slide: SlideHandle,
    tissue_mask: TissueMask,
    level: int,
    patch_size: int,
    tissue_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Grid columns ``xs`` and rows ``ys``, and the indices of positions with tissue.

    The returned ``rows`` and ``cols`` index into ``ys`` and ``xs`` in
    row-major order, so they follow a y-then-x scan of the grid.
    """
    props = slide.properties
    patch_l0 = int(patch_size * props.level_downsamples[level])
    xs = np.arange(0, props.width - patch_l0 + 1, patch_l0)
    ys = np.arange(0, props.height - patch_l0 + 1, patch_l0)
    if len(xs) == 0 or len(ys) == 0:
        none = np.empty(0, dtype=np.intp)
        return xs, ys, none, none

    # Look up every grid position at once, with the cell assignment of
    # TissueMask.contains_tissue, instead of checking positions one by one.
    h_mask, w_mask = tissue_mask.mask.shape
    cells_x = np.minimum((xs / tissue_mask.downsample).astype(np.int64), w_mask - 1)
    cells_y = np.minimum((ys / tissue_mask.downsample).astype(np.int64), h_mask - 1)
    fractions = tissue_mask.tissue_fraction_grid(patch_l0, patch_l0, cells_x, cells_y)
    rows, cols = np.nonzero(fractions >= tissue_threshold)
    return xs, ys, rows, cols


def resolve_mpp_read(
    props: SlideProperties,
    target_mpp: float,
    patch_size: int,
    mpp_tolerance: float,
) -> tuple[int, int, float]:
    """Choose how to read a ``patch_size`` patch at ``target_mpp``.

    A pyramid level whose mpp is within ``mpp_tolerance`` (relative) of the
    target is read directly at ``patch_size``.  Otherwise a proportionally
    larger region is read from the coarsest level finer than the target, to be
    shrunk to ``patch_size``.  When every level is coarser than the target,
    level 0 is read at its native resolution, because enlarging cannot add
    detail.

    Parameters
    ----------
    props : SlideProperties
        Slide metadata with a known ``mpp``.
    target_mpp : float
        Desired µm/px of the output patch.
    patch_size : int
        Output side length in pixels.
    mpp_tolerance : float
        Maximum relative deviation between a level's mpp and ``target_mpp``
        for reading that level directly.  ``0.1`` reads a 0.2325 µm/px level
        directly for a 0.25 µm/px target.

    Returns
    -------
    tuple[int, int, float]
        ``(level, read_size, mpp)``: the pyramid level, the side length to read
        at that level, and the mpp of the patch once the read region is resized
        to ``patch_size``.
    """
    if props.mpp is None:
        raise ValueError(f"slide {props.path!r} has no MPP metadata")
    level_mpps = [props.mpp * ds for ds in props.level_downsamples]

    direct = [
        lvl
        for lvl, lvl_mpp in enumerate(level_mpps)
        if abs(lvl_mpp - target_mpp) <= mpp_tolerance * target_mpp
    ]
    if direct:
        level = min(direct, key=lambda lvl: abs(level_mpps[lvl] - target_mpp))
        return level, patch_size, level_mpps[level]

    finer = [lvl for lvl, lvl_mpp in enumerate(level_mpps) if lvl_mpp < target_mpp]
    if not finer:
        return 0, patch_size, level_mpps[0]

    level = max(finer, key=lambda lvl: level_mpps[lvl])
    read_size = round(patch_size * target_mpp / level_mpps[level])
    return level, read_size, level_mpps[level] * read_size / patch_size


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
