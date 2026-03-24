"""
PatchPipeline (the main orchestrator).

Maintains a **pool** of simultaneously open slides and round-robins
across them so that patches from different slides are interleaved.

Flow per slide: open → thumbnail → tissue mask → sample coords → read patch → filter → transform → yield
"""

from __future__ import annotations

import copy
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from wsistream.backends.base import SlideBackend
from wsistream.datasets.base import DatasetAdapter
from wsistream.filters.base import PatchFilter
from wsistream.sampling.base import PatchSampler
from wsistream.sampling.random import RandomSampler
from wsistream.slide import SlideHandle
from wsistream.tissue.base import TissueDetector
from wsistream.tissue.otsu import OtsuTissueDetector
from wsistream.transforms.base import PatchTransform
from wsistream.types import (
    PatchCoordinate,
    PatchResult,
    SlideMetadata,
    TissueMask,
    resolve_slide_paths,
)

logger = logging.getLogger(__name__)


@dataclass
class _TissueFractionStats:
    """Running min/max/mean for tissue fractions without storing every value."""

    count: int = 0
    total: float = 0.0
    min_val: float = float("inf")
    max_val: float = float("-inf")

    def update(self, value: float) -> None:
        self.count += 1
        self.total += value
        if value < self.min_val:
            self.min_val = value
        if value > self.max_val:
            self.max_val = value

    @property
    def mean(self) -> float | None:
        return self.total / self.count if self.count > 0 else None


@dataclass
class PipelineStats:
    """Accumulates statistics during pipeline execution for logging."""

    slides_processed: int = 0
    slides_failed: int = 0
    patches_extracted: int = 0
    patches_filtered: int = 0
    tissue_fractions: _TissueFractionStats = field(default_factory=_TissueFractionStats)
    magnification_counts: dict[float | None, int] = field(default_factory=dict)
    cancer_type_counts: dict[str, int] = field(default_factory=dict)
    sample_type_counts: dict[str, int] = field(default_factory=dict)
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0

    def record_error(self, slide_path: str, message: str) -> None:
        """Record an error, keeping only the most recent 100."""
        self.recent_errors.append((slide_path, message))
        self.error_count += 1

    def to_dict(self) -> dict:
        """Flat dict suitable for wandb.log() or similar."""
        result = {
            "pipeline/slides_processed": self.slides_processed,
            "pipeline/slides_failed": self.slides_failed,
            "pipeline/patches_extracted": self.patches_extracted,
            "pipeline/patches_filtered": self.patches_filtered,
        }
        if self.tissue_fractions.count > 0:
            result["pipeline/mean_tissue_fraction"] = self.tissue_fractions.mean
            result["pipeline/min_tissue_fraction"] = self.tissue_fractions.min_val
            result["pipeline/max_tissue_fraction"] = self.tissue_fractions.max_val
        for mpp, count in self.magnification_counts.items():
            key = f"pipeline/mpp_{mpp:.2f}" if mpp is not None else "pipeline/mpp_unknown"
            result[key] = count
        for ct, count in self.cancer_type_counts.items():
            result[f"pipeline/cancer_type/{ct}"] = count
        for st, count in self.sample_type_counts.items():
            safe = st.replace(" ", "_").lower()
            result[f"pipeline/sample_type/{safe}"] = count
        if self.error_count > 0:
            result["pipeline/error_count"] = self.error_count
        return result


class _PoolEntry:
    """One active slide in the pool (internal)."""

    __slots__ = (
        "slide",
        "sampler_iter",
        "tissue_mask",
        "metadata",
        "patch_count",
        "successful_reads",
    )

    def __init__(
        self,
        slide: SlideHandle,
        sampler_iter: Iterator[PatchCoordinate],
        tissue_mask: TissueMask,
        metadata: SlideMetadata | None,
    ) -> None:
        self.slide = slide
        self.sampler_iter = sampler_iter
        self.tissue_mask = tissue_mask
        self.metadata = metadata
        self.patch_count = 0
        self.successful_reads = 0


@dataclass
class PatchPipeline:
    """
    Pool-based online patch extraction pipeline.

    Maintains up to ``pool_size`` slides open simultaneously and
    round-robins across them.  By default, one patch is read per slide
    before advancing (``patches_per_visit=1``); set higher for better
    I/O locality on network filesystems.  This ensures patches from
    different slides are interleaved even when samplers produce many
    patches per slide (including ``num_patches=-1`` / infinite mode).

    Parameters
    ----------
    slide_paths : str, Path, or list
        A directory path (all WSI files are collected automatically),
        a single file path, or an explicit list of file paths.
    backend : SlideBackend
        Explicit backend instance used as a prototype.  A deep copy is
        created for each slide, preserving any constructor configuration.
    tissue_detector : TissueDetector
        Strategy for detecting tissue regions.
    sampler : PatchSampler
        Strategy for sampling patch coordinates.
    patch_filter : PatchFilter or None
        Per-tile quality filter applied after extraction, before transforms.
        A rejected patch is discarded and the pipeline moves on to the next
        sample.  Rejected patches still count towards ``patches_per_slide``
        to prevent infinite loops when a filter rejects everything.
    transforms : PatchTransform or None
        Transform pipeline applied to each patch.
    dataset_adapter : DatasetAdapter or None
        Extracts dataset-specific metadata from slide paths.
    thumbnail_size : tuple[int, int]
        Thumbnail resolution for tissue detection.
    slide_sampling : str
        ``"sequential"`` or ``"random"`` slide iteration order.
    pool_size : int
        Number of slides kept open simultaneously.  A larger pool gives
        better interleaving but uses more file handles and memory.
    patches_per_slide : int
        Maximum patch reads from one slide before closing it and
        opening the next.  Counts every extraction attempt, including
        patches rejected by ``patch_filter``.  Essential for infinite
        samplers (``num_patches=-1``) where the sampler itself never stops.
    patches_per_visit : int
        Number of patches to read from the current slide before
        round-robining to the next slide in the pool.  Higher values
        improve I/O locality (the OS file cache stays warm for
        consecutive reads from the same slide) at the cost of reduced
        interleaving.  Default ``1`` gives maximum interleaving.
        Values like ``8``--``16`` can significantly improve throughput
        on network filesystems.
    cycle : bool
        When ``True``, re-queue all slides once the queue is exhausted,
        producing an infinite stream that cycles over the whole corpus.
    seed : int or None
        Random seed for slide-level shuffling.
    """

    slide_paths: str | Path | list[str | Path] = field(default_factory=list)
    backend: SlideBackend = field(default_factory=lambda: _missing_backend())
    tissue_detector: TissueDetector = field(default_factory=OtsuTissueDetector)
    sampler: PatchSampler = field(default_factory=RandomSampler)
    patch_filter: PatchFilter | None = None
    transforms: PatchTransform | None = None
    dataset_adapter: DatasetAdapter | None = None
    thumbnail_size: tuple[int, int] = (2048, 2048)
    slide_sampling: str = "sequential"
    pool_size: int = 8
    patches_per_slide: int = 100
    patches_per_visit: int = 1
    cycle: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.slide_sampling not in ("sequential", "random"):
            raise ValueError(
                f"slide_sampling must be 'sequential' or 'random', got {self.slide_sampling!r}"
            )
        if self.pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, got {self.pool_size}")
        if self.patches_per_slide < 1:
            raise ValueError(f"patches_per_slide must be >= 1, got {self.patches_per_slide}")
        if self.patches_per_visit < 1:
            raise ValueError(f"patches_per_visit must be >= 1, got {self.patches_per_visit}")
        self.slide_paths = resolve_slide_paths(self.slide_paths)
        self._stats = PipelineStats()
        self._failed_slides: set[str] = set()
        # Mix PID into all seeds so workers (spawn or fork) diverge.
        self._pid_at_init: int = os.getpid()
        base = (self.seed or 0, self._pid_at_init)
        self._rng = np.random.default_rng(base)
        # Reseed sampler and transforms with pipeline-controlled seeds
        # so that externally-created RNGs don't collide across workers.
        if hasattr(self.sampler, "_rng"):
            self.sampler._rng = np.random.default_rng((*base, 1))
        self._reseed_transform(self.transforms, base)

    # ── public API ──

    def __iter__(self) -> Iterator[PatchResult]:
        return self._iterate()

    @property
    def stats(self) -> PipelineStats:
        return self._stats

    def stats_dict(self) -> dict:
        """Flat dict for wandb.log() or similar."""
        return self._stats.to_dict()

    def reset_stats(self) -> None:
        """Reset all accumulated statistics (e.g. between epochs)."""
        self._stats = PipelineStats()
        self._failed_slides.clear()

    # ── core iteration ──

    def _iterate(self) -> Iterator[PatchResult]:
        self._reseed_for_worker()
        slide_queue: deque[str] = deque(self._get_slide_order())
        pool: list[_PoolEntry] = []

        self._fill_pool(pool, slide_queue)
        if not pool:
            return

        pool_idx = 0
        visit_count = 0  # patches read from current slide in this visit
        try:
            while pool:
                entry = pool[pool_idx]

                # Try to get the next patch coordinate from this slide's sampler
                try:
                    coord = next(entry.sampler_iter)
                except StopIteration:
                    # Sampler exhausted for this slide
                    self._close_entry(entry)
                    pool.pop(pool_idx)
                    self._fill_pool(pool, slide_queue)
                    if not pool:
                        break
                    pool_idx = pool_idx % len(pool)
                    visit_count = 0
                    continue

                # Read patch from slide
                try:
                    patch = entry.slide.read_region(
                        x=coord.x,
                        y=coord.y,
                        width=coord.patch_size,
                        height=coord.patch_size,
                        level=coord.level,
                    )
                except Exception as exc:
                    logger.warning("Error reading patch from %s: %s", coord.slide_path, exc)
                    self._stats.record_error(coord.slide_path, str(exc))
                    entry.patch_count += 1
                    if entry.patch_count >= self.patches_per_slide:
                        self._close_entry(entry)
                        pool.pop(pool_idx)
                        self._fill_pool(pool, slide_queue)
                        if not pool:
                            break
                        pool_idx = pool_idx % len(pool)
                        visit_count = 0
                    else:
                        visit_count += 1
                        if visit_count >= self.patches_per_visit:
                            pool_idx = (pool_idx + 1) % len(pool)
                            visit_count = 0
                    continue

                # Every successful read counts toward the per-slide budget,
                # whether the patch is ultimately yielded or filtered out.
                # This prevents infinite loops when a filter rejects everything.
                entry.patch_count += 1
                entry.successful_reads += 1

                # Per-tile quality filter (e.g., HSV pixel check)
                if self.patch_filter is not None and not self.patch_filter.accept(patch):
                    self._stats.patches_filtered += 1
                    if entry.patch_count >= self.patches_per_slide:
                        self._close_entry(entry)
                        pool.pop(pool_idx)
                        self._fill_pool(pool, slide_queue)
                        if not pool:
                            break
                        pool_idx = pool_idx % len(pool)
                        visit_count = 0
                    else:
                        visit_count += 1
                        if visit_count >= self.patches_per_visit:
                            pool_idx = (pool_idx + 1) % len(pool)
                            visit_count = 0
                    continue

                if self.transforms is not None:
                    patch = self.transforms(patch)

                # Tissue fraction for this patch
                ds = entry.slide.properties.level_downsamples[coord.level]
                patch_l0 = int(coord.patch_size * ds)
                tf = entry.tissue_mask.tissue_fraction_at(
                    coord.x,
                    coord.y,
                    patch_l0,
                    patch_l0,
                )

                # Update stats
                self._stats.patches_extracted += 1
                mpp = round(coord.mpp, 2) if coord.mpp is not None else None
                self._stats.magnification_counts[mpp] = (
                    self._stats.magnification_counts.get(mpp, 0) + 1
                )

                yield PatchResult(
                    image=patch,
                    coordinate=coord,
                    tissue_fraction=tf,
                    slide_metadata=entry.metadata,
                )

                # Reached per-slide patch limit --> rotate this slide out
                if entry.patch_count >= self.patches_per_slide:
                    self._close_entry(entry)
                    pool.pop(pool_idx)
                    self._fill_pool(pool, slide_queue)
                    if not pool:
                        break
                    pool_idx = pool_idx % len(pool)
                    visit_count = 0
                else:
                    visit_count += 1
                    if visit_count >= self.patches_per_visit:
                        pool_idx = (pool_idx + 1) % len(pool)
                        visit_count = 0
        finally:
            # Cleanup: close any remaining slides (runs even on early break /
            # GeneratorExit, preventing file-handle leaks).
            for entry in pool:
                self._close_entry(entry)
            pool.clear()

    # ── helpers ──

    def _reseed_for_worker(self) -> None:
        """Reseed all RNGs if running in a forked worker process.

        For spawn-based workers, ``__post_init__`` re-runs with a new PID
        so the RNGs already diverge.  For fork-based workers, the pipeline
        object is copied in-memory and ``__post_init__`` does NOT re-run,
        so all workers share the same RNG state.  This method detects the
        fork case (PID changed since init) and reseeds once.
        """
        pid = os.getpid()
        if pid == self._pid_at_init:
            return

        # Fork detected — reseed everything and update the stored PID
        # so subsequent iterations in this worker don't reseed again.
        self._pid_at_init = pid
        base = (self.seed or 0, pid)
        self._rng = np.random.default_rng(base)

        if hasattr(self.sampler, "_rng"):
            self.sampler._rng = np.random.default_rng((*base, 1))

        self._reseed_transform(self.transforms, base)

    @staticmethod
    def _reseed_transform(transform: PatchTransform | None, base_seed) -> None:
        if transform is None:
            return
        if hasattr(transform, "_rng"):
            transform._rng = np.random.default_rng((*base_seed, 2))
        if hasattr(transform, "transforms"):
            for i, t in enumerate(transform.transforms):
                if hasattr(t, "_rng"):
                    t._rng = np.random.default_rng((*base_seed, 3, i))

    def _fill_pool(self, pool: list[_PoolEntry], slide_queue: deque[str]) -> None:
        """Open slides from the queue until the pool is full.

        When cycling, the queue is refilled transparently.  Two guards
        prevent pathological behaviour:
        - Never open more concurrent copies than ``len(slide_paths)``
          (avoids duplicate file handles for the same slide).
        - Stop if an entire pass over slide_paths fails consecutively
          (avoids infinite retries when every slide is broken).
        """
        consecutive_failures = 0

        while len(pool) < self.pool_size:
            if not slide_queue:
                if not self.cycle:
                    break
                # Never open more concurrent copies than there are unique slides
                if len(pool) >= len(set(self.slide_paths)):
                    break
                slide_queue.extend(self._get_slide_order())
                if not slide_queue:
                    break  # no slides at all

            slide_path = slide_queue.popleft()
            try:
                entry = self._open_slide(slide_path)
                pool.append(entry)
                consecutive_failures = 0
            except Exception as exc:
                self._stats.slides_failed += 1
                self._stats.record_error(slide_path, str(exc))
                logger.warning("Failed to open slide %s: %s", slide_path, exc)
                consecutive_failures += 1
                if consecutive_failures >= len(self.slide_paths):
                    break

    def _open_slide(self, slide_path: str) -> _PoolEntry:
        """Open a slide, detect tissue, create sampler iterator."""
        # Parse dataset-specific metadata (but don't update stats yet —
        # stats are only updated after the slide is fully set up).
        metadata: SlideMetadata | None = None
        if self.dataset_adapter is not None:
            metadata = self.dataset_adapter.parse_metadata(slide_path)

        # Fresh backend per slide (deep-copy preserves any constructor config)
        backend = copy.deepcopy(self.backend)
        slide = SlideHandle(slide_path, backend=backend)

        # Everything after open() must be wrapped so that a failure
        # closes the slide — including stats update and PoolEntry creation.
        try:
            thumbnail = slide.get_thumbnail(self.thumbnail_size)
            th, tw = thumbnail.shape[:2]
            sw, sh = slide.properties.dimensions
            downsample_xy = (sw / tw, sh / th)
            downsample_scalar = max(downsample_xy)

            mask_arr = self.tissue_detector.detect(thumbnail, downsample=downsample_xy)
            tissue_mask = TissueMask(
                mask=mask_arr,
                downsample=downsample_scalar,
                slide_dimensions=slide.properties.dimensions,
            )
            sampler_iter = iter(self.sampler.sample(slide, tissue_mask))

            # Only count as processed after full setup succeeds
            self._stats.slides_processed += 1
            self._stats.tissue_fractions.update(tissue_mask.tissue_fraction)
            if metadata is not None:
                if metadata.cancer_type:
                    ct = metadata.cancer_type
                    self._stats.cancer_type_counts[ct] = (
                        self._stats.cancer_type_counts.get(ct, 0) + 1
                    )
                if metadata.sample_type:
                    st = metadata.sample_type
                    self._stats.sample_type_counts[st] = (
                        self._stats.sample_type_counts.get(st, 0) + 1
                    )

            return _PoolEntry(
                slide=slide,
                sampler_iter=sampler_iter,
                tissue_mask=tissue_mask,
                metadata=metadata,
            )
        except Exception:
            slide.close()
            raise

    def _close_entry(self, entry: _PoolEntry) -> None:
        try:
            entry.slide.close()
        except Exception:
            pass
        # If a slide used its full budget but had zero successful reads,
        # it is effectively broken. Blacklist it so cycle mode doesn't
        # reopen it endlessly.
        if entry.patch_count > 0 and entry.successful_reads == 0:
            self._failed_slides.add(entry.slide.properties.path)
            self._stats.slides_failed += 1

    def _get_slide_order(self) -> list[str]:
        paths = [p for p in self.slide_paths if p not in self._failed_slides]
        if self.slide_sampling == "random":
            indices = self._rng.permutation(len(paths))
            return [paths[i] for i in indices]
        return list(paths)


def _missing_backend():
    """Raise immediately if no backend was specified."""
    raise TypeError(
        "PatchPipeline requires an explicit backend. "
        "Use backend=OpenSlideBackend() or backend=TiffSlideBackend(). "
        "Example: PatchPipeline(slide_paths=[...], backend=OpenSlideBackend())"
    )
