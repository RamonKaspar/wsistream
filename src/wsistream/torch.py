"""PyTorch integration: IterableDataset wrapper with DDP support."""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset

from wsistream.backends.base import SlideBackend
from wsistream.datasets.base import DatasetAdapter
from wsistream.filters.base import PatchFilter
from wsistream.pipeline import PatchPipeline
from wsistream.sampling.base import PatchSampler
from wsistream.tissue.base import TissueDetector
from wsistream.transforms.base import PatchTransform
from wsistream.types import PatchResult, SlideMetadata, resolve_slide_paths

logger = logging.getLogger(__name__)


def partition_slides_by_rank(
    slide_paths: str | Path | list[str | Path],
    rank: int | None = None,
    world_size: int | None = None,
) -> list[str]:
    """Partition slides across DDP ranks using round-robin assignment.

    Parameters
    ----------
    slide_paths : str, Path, or list
        A directory path, a single file, or a list of file paths.
    rank, world_size : int or None
        When ``None``, auto-detected from environment variables
        (``RANK``, ``WORLD_SIZE``).  Falls back to returning all
        slides if DDP is not active.
    """
    import os

    slide_paths = resolve_slide_paths(slide_paths)

    if rank is None:
        rank = int(os.environ.get("RANK", 0))
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    slides = slide_paths[rank::world_size]
    if not slides:
        raise RuntimeError(
            f"Rank {rank} got 0 slides ({len(slide_paths)} slides / {world_size} ranks). "
            f"Need at least as many slides as ranks."
        )
    logger.info("Rank %d/%d: %d/%d slides", rank, world_size, len(slides), len(slide_paths))
    return slides


@dataclass
class _StatsDelta:
    """Raw mergeable stats delta pushed from a worker to the aggregator."""

    slides_processed: int = 0
    slides_failed: int = 0
    patches_extracted: int = 0
    patches_filtered: int = 0
    error_count: int = 0
    tf_count: int = 0
    tf_total: float = 0.0
    tf_min: float = float("inf")
    tf_max: float = float("-inf")
    mpp_counts: dict = field(default_factory=dict)
    cancer_type_counts: dict = field(default_factory=dict)
    sample_type_counts: dict = field(default_factory=dict)


class _StatsAggregator:
    """Aggregates stats deltas from multiple workers.

    For ``num_workers=0`` (in-process), workers call ``merge()`` directly.
    For ``num_workers>0``, workers ``push()`` deltas into a queue and the
    main process drains it in ``to_dict()``.
    """

    def __init__(self) -> None:
        self._queue: mp.Queue = mp.Queue()
        self._agg = _StatsDelta()

    def merge(self, delta: _StatsDelta) -> None:
        """Merge a delta directly (used for num_workers=0)."""
        self._merge_into(delta)

    def push(self, delta: _StatsDelta) -> None:
        """Push a delta into the queue (used for num_workers>0)."""
        self._queue.put(delta)

    def _drain(self) -> None:
        """Drain all pending items from the queue. Non-blocking."""
        import queue as _queue_mod

        while True:
            try:
                delta = self._queue.get_nowait()
            except (_queue_mod.Empty, EOFError):
                break
            self._merge_into(delta)

    def _merge_into(self, delta: _StatsDelta) -> None:
        a = self._agg
        a.slides_processed += delta.slides_processed
        a.slides_failed += delta.slides_failed
        a.patches_extracted += delta.patches_extracted
        a.patches_filtered += delta.patches_filtered
        a.error_count += delta.error_count
        a.tf_count += delta.tf_count
        a.tf_total += delta.tf_total
        if delta.tf_min < a.tf_min:
            a.tf_min = delta.tf_min
        if delta.tf_max > a.tf_max:
            a.tf_max = delta.tf_max
        for k, v in delta.mpp_counts.items():
            a.mpp_counts[k] = a.mpp_counts.get(k, 0) + v
        for k, v in delta.cancer_type_counts.items():
            a.cancer_type_counts[k] = a.cancer_type_counts.get(k, 0) + v
        for k, v in delta.sample_type_counts.items():
            a.sample_type_counts[k] = a.sample_type_counts.get(k, 0) + v

    def to_dict(self) -> dict:
        """Drain pending items and build the public flat stats dict."""
        self._drain()
        a = self._agg
        result: dict = {
            "pipeline/slides_processed": a.slides_processed,
            "pipeline/slides_failed": a.slides_failed,
            "pipeline/patches_extracted": a.patches_extracted,
            "pipeline/patches_filtered": a.patches_filtered,
        }
        if a.tf_count > 0:
            result["pipeline/mean_tissue_fraction"] = a.tf_total / a.tf_count
            result["pipeline/min_tissue_fraction"] = a.tf_min
            result["pipeline/max_tissue_fraction"] = a.tf_max
        for mpp, count in a.mpp_counts.items():
            key = f"pipeline/mpp_{mpp:.2f}" if mpp is not None else "pipeline/mpp_unknown"
            result[key] = count
        for ct, count in a.cancer_type_counts.items():
            result[f"pipeline/cancer_type/{ct}"] = count
        for st, count in a.sample_type_counts.items():
            safe = st.replace(" ", "_").lower()
            result[f"pipeline/sample_type/{safe}"] = count
        if a.error_count > 0:
            result["pipeline/error_count"] = a.error_count
        return result

    def reset(self) -> None:
        """Clear all aggregated stats."""
        self._drain()
        self._agg = _StatsDelta()


class WsiStreamDataset(IterableDataset):
    """Wraps a ``PatchPipeline`` as a PyTorch ``IterableDataset``.

    Handles multi-worker ``DataLoader`` by partitioning slides across
    workers.  DDP rank partitioning should be done **before** creating
    this dataset (see :func:`partition_slides_by_rank`).

    Each yielded item is a ``dict`` with keys: ``"image"`` (a
    ``(C, H, W)`` float32 tensor in ``[0, 1]``), coordinate fields
    (``"x"``, ``"y"``, ``"level"``, ``"patch_size"``, ``"slide_path"``,
    ``"mpp"``), ``"tissue_fraction"``, and when a ``dataset_adapter``
    is configured: ``"patient_id"``, ``"cancer_type"``, ``"tissue_type"``,
    ``"sample_type"``, ``"dataset_name"``, ``"extra"`` (JSON string).

    Parameters
    ----------
    slide_paths : str, Path, or list
        A directory path (all WSI files inside are collected), a single
        file path, or an explicit list of file paths.
    backend : SlideBackend
        Backend instance (deep-copied per slide internally).
    tissue_detector : TissueDetector
        Tissue detection strategy.
    sampler : PatchSampler
        Patch sampling strategy.
    patch_filter : PatchFilter or None
        Optional per-tile quality filter.
    transforms : PatchTransform or None
        Optional numpy-level augmentations applied before tensor conversion.
    dataset_adapter : DatasetAdapter or None
        Optional dataset-specific metadata extractor.
    pool_size : int
        Number of slides kept open simultaneously.
    patches_per_slide : int
        Patches drawn from one slide before rotating to the next.
    cycle : bool
        When ``True``, re-queue slides after exhaustion for infinite
        streaming (the typical mode for FM pretraining).
    slide_sampling : str
        ``"sequential"`` or ``"random"`` slide iteration order.
    seed : int or None
        Random seed for slide-level shuffling.
    """

    def __init__(
        self,
        slide_paths: str | Path | list[str | Path],
        backend: SlideBackend,
        tissue_detector: TissueDetector,
        sampler: PatchSampler,
        patch_filter: PatchFilter | None = None,
        transforms: PatchTransform | None = None,
        dataset_adapter: DatasetAdapter | None = None,
        pool_size: int = 8,
        patches_per_slide: int = 100,
        cycle: bool = True,
        slide_sampling: str = "random",
        seed: int | None = None,
    ):
        self._slide_paths = resolve_slide_paths(slide_paths)
        self._backend = backend
        self._tissue_detector = tissue_detector
        self._sampler = sampler
        self._patch_filter = patch_filter
        self._transforms = transforms
        self._dataset_adapter = dataset_adapter
        self._pool_size = pool_size
        self._patches_per_slide = patches_per_slide
        self._slide_sampling = slide_sampling
        self._cycle = cycle
        self._seed = seed
        self._iter_count = 0
        self._shared_stats = _StatsAggregator()

    def stats_dict(self) -> dict:
        """Aggregated stats from all workers. Safe to call from the main process."""
        return self._shared_stats.to_dict()

    def reset_stats(self) -> None:
        """Reset all shared counters (e.g., between epochs)."""
        self._shared_stats.reset()

    def __iter__(self) -> Iterator[dict]:
        self._iter_count += 1

        # Partition slides across DataLoader workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            slides = self._slide_paths[worker_info.id :: worker_info.num_workers]
            base = (self._seed or 0) + worker_info.id
            worker_seed = base + self._iter_count
            logger.debug(
                "Worker %d/%d: %d slides",
                worker_info.id,
                worker_info.num_workers,
                len(slides),
            )
        else:
            slides = self._slide_paths
            worker_seed = (
                (self._seed or 0) + self._iter_count if self._seed is not None else self._iter_count
            )

        if not slides:
            logger.warning("Worker got 0 slides, yielding nothing")
            return

        pipeline = PatchPipeline(
            slide_paths=slides,
            backend=self._backend,
            tissue_detector=self._tissue_detector,
            sampler=self._sampler,
            patch_filter=self._patch_filter,
            transforms=self._transforms,
            dataset_adapter=self._dataset_adapter,
            slide_sampling=self._slide_sampling,
            pool_size=self._pool_size,
            patches_per_slide=self._patches_per_slide,
            cycle=self._cycle,
            seed=worker_seed,
        )

        use_queue = worker_info is not None
        flush_interval = 16 if use_queue else 1

        prev = _StatsDelta()
        patch_count = 0
        try:
            for result in pipeline:
                patch_count += 1
                if patch_count % flush_interval == 0:
                    self._flush_stats(pipeline, prev, use_queue)
                yield self._result_to_dict(result)
        finally:
            self._flush_stats(pipeline, prev, use_queue)

    def _flush_stats(
        self,
        pipeline: PatchPipeline,
        prev: _StatsDelta,
        use_queue: bool,
    ) -> None:
        """Compute delta from raw pipeline stats and send to aggregator."""
        s = pipeline.stats
        tf = s.tissue_fractions

        delta = _StatsDelta(
            slides_processed=s.slides_processed - prev.slides_processed,
            slides_failed=s.slides_failed - prev.slides_failed,
            patches_extracted=s.patches_extracted - prev.patches_extracted,
            patches_filtered=s.patches_filtered - prev.patches_filtered,
            error_count=s.error_count - prev.error_count,
            tf_count=tf.count - prev.tf_count,
            tf_total=tf.total - prev.tf_total,
            tf_min=tf.min_val,
            tf_max=tf.max_val,
        )

        # Diff the histogram dicts
        for mpp, count in s.magnification_counts.items():
            d = count - prev.mpp_counts.get(mpp, 0)
            if d:
                delta.mpp_counts[mpp] = d
        for ct, count in s.cancer_type_counts.items():
            d = count - prev.cancer_type_counts.get(ct, 0)
            if d:
                delta.cancer_type_counts[ct] = d
        for st, count in s.sample_type_counts.items():
            d = count - prev.sample_type_counts.get(st, 0)
            if d:
                delta.sample_type_counts[st] = d

        if use_queue:
            self._shared_stats.push(delta)
        else:
            self._shared_stats.merge(delta)

        # Update prev snapshot
        prev.slides_processed = s.slides_processed
        prev.slides_failed = s.slides_failed
        prev.patches_extracted = s.patches_extracted
        prev.patches_filtered = s.patches_filtered
        prev.error_count = s.error_count
        prev.tf_count = tf.count
        prev.tf_total = tf.total
        prev.tf_min = tf.min_val
        prev.tf_max = tf.max_val
        prev.mpp_counts = dict(s.magnification_counts)
        prev.cancer_type_counts = dict(s.cancer_type_counts)
        prev.sample_type_counts = dict(s.sample_type_counts)

    @staticmethod
    def _result_to_dict(result: PatchResult) -> dict:
        """Convert a PatchResult to a dict of collate-safe types."""
        image = np.ascontiguousarray(result.image)
        if image.dtype == np.uint8:
            tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        else:
            tensor = torch.from_numpy(image).permute(2, 0, 1).float()

        coord = result.coordinate

        # Metadata first (includes slide_path from adapter)
        meta = result.slide_metadata
        item = meta.to_flat_dict() if meta else SlideMetadata.empty_dict()

        # Coordinate fields overwrite — these are authoritative
        item.update(
            {
                "image": tensor,
                "x": coord.x,
                "y": coord.y,
                "level": coord.level,
                "patch_size": coord.patch_size,
                "slide_path": coord.slide_path,
                "mpp": coord.mpp if coord.mpp is not None else -1.0,
                "tissue_fraction": result.tissue_fraction,
            }
        )
        return item


# Re-export for convenience: `from wsistream.torch import MonitoredLoader`
from wsistream.torch_monitor import MonitoredLoader  # noqa: E402, F401
