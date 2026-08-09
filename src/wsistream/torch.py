"""PyTorch integration: IterableDataset wrapper with DDP support."""

from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path
from typing import Iterator, Literal

import cv2
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
from wsistream.views import ViewConfig

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


class _StatsAggregator:
    """Aggregates pipeline stats from multiple workers.

    Uses a hybrid approach:

    - **Core counters** (slides, patches, tissue fractions): ``mp.Value``
      shared memory objects.  Workers atomically add deltas.  Survives
      worker shutdown, no pipe buffer, no Manager process.
    - **Sparse histograms** (mpp counts, cancer/sample types):
      ``mp.SimpleQueue``, flushed only on slide rotation (very infrequent).
    - **num_workers=0**: direct Python operations, no IPC.
    """

    def __init__(self) -> None:
        # Fork-context synchronization primitives cannot be sent to spawn or
        # forkserver workers.  Spawn-context primitives remain usable when
        # inherited by fork workers, so use the portable context explicitly.
        context = mp.get_context("spawn")
        # Shared-memory counters (additive)
        self._slides_processed = context.Value("i", 0)
        self._slides_failed = context.Value("i", 0)
        self._patches_extracted = context.Value("i", 0)
        self._patches_filtered = context.Value("i", 0)
        self._error_count = context.Value("i", 0)
        self._tf_count = context.Value("i", 0)
        self._tf_total = context.Value("d", 0.0)
        self._tf_min = context.Value("d", float("inf"))
        self._tf_max = context.Value("d", float("-inf"))
        # Sparse histogram queue (flushed per-slide, not per-patch)
        self._histogram_queue = context.SimpleQueue()
        # Main-process-only accumulators
        self._slides_seen: set[str] = set()
        self._mpp_counts: dict = {}
        self._cancer_type_counts: dict = {}
        self._sample_type_counts: dict = {}

    def add_counters(
        self,
        slides_processed: int,
        slides_failed: int,
        patches_extracted: int,
        patches_filtered: int,
        error_count: int,
        tf_count: int,
        tf_total: float,
        tf_min: float,
        tf_max: float,
    ) -> None:
        """Atomically add counter deltas from a worker."""
        with self._slides_processed.get_lock():
            self._slides_processed.value += slides_processed
        with self._slides_failed.get_lock():
            self._slides_failed.value += slides_failed
        with self._patches_extracted.get_lock():
            self._patches_extracted.value += patches_extracted
        with self._patches_filtered.get_lock():
            self._patches_filtered.value += patches_filtered
        with self._error_count.get_lock():
            self._error_count.value += error_count
        with self._tf_count.get_lock():
            self._tf_count.value += tf_count
        with self._tf_total.get_lock():
            self._tf_total.value += tf_total
        with self._tf_min.get_lock():
            if tf_min < self._tf_min.value:
                self._tf_min.value = tf_min
        with self._tf_max.get_lock():
            if tf_max > self._tf_max.value:
                self._tf_max.value = tf_max

    def push_histograms(
        self, slide_paths: list[str], mpp: dict, cancer: dict, sample: dict
    ) -> None:
        """Push sparse histogram deltas and slide paths (called per-slide, not per-patch)."""
        self._histogram_queue.put((slide_paths, mpp, cancer, sample))

    def _drain_histograms(self) -> None:
        """Drain sparse histogram queue into local accumulators."""
        while not self._histogram_queue.empty():
            try:
                slide_paths, mpp, cancer, sample = self._histogram_queue.get()
            except EOFError:
                break
            self._slides_seen.update(slide_paths)
            for k, v in mpp.items():
                self._mpp_counts[k] = self._mpp_counts.get(k, 0) + v
            for k, v in cancer.items():
                self._cancer_type_counts[k] = self._cancer_type_counts.get(k, 0) + v
            for k, v in sample.items():
                self._sample_type_counts[k] = self._sample_type_counts.get(k, 0) + v

    def to_dict(self) -> dict:
        """Build the public flat stats dict."""
        self._drain_histograms()

        result: dict = {
            "pipeline/slides_processed": self._slides_processed.value,
            "pipeline/slides_failed": self._slides_failed.value,
            "pipeline/slides_unique": len(self._slides_seen),
            "pipeline/patches_extracted": self._patches_extracted.value,
            "pipeline/patches_filtered": self._patches_filtered.value,
        }
        tf_count = self._tf_count.value
        if tf_count > 0:
            result["pipeline/mean_tissue_fraction"] = self._tf_total.value / tf_count
            result["pipeline/min_tissue_fraction"] = self._tf_min.value
            result["pipeline/max_tissue_fraction"] = self._tf_max.value
        for mpp, count in self._mpp_counts.items():
            key = f"pipeline/mpp_{mpp:.2f}" if mpp is not None else "pipeline/mpp_unknown"
            result[key] = count
        for ct, count in self._cancer_type_counts.items():
            result[f"pipeline/cancer_type/{ct}"] = count
        for st, count in self._sample_type_counts.items():
            safe = st.replace(" ", "_").lower()
            result[f"pipeline/sample_type/{safe}"] = count
        error_count = self._error_count.value
        if error_count > 0:
            result["pipeline/error_count"] = error_count
        return result

    def reset(self) -> None:
        """Clear all aggregated stats."""
        for v in (
            self._slides_processed,
            self._slides_failed,
            self._patches_extracted,
            self._patches_filtered,
            self._error_count,
            self._tf_count,
            self._tf_total,
        ):
            with v.get_lock():
                v.value = 0
        with self._tf_min.get_lock():
            self._tf_min.value = float("inf")
        with self._tf_max.get_lock():
            self._tf_max.value = float("-inf")
        self._drain_histograms()
        self._slides_seen.clear()
        self._mpp_counts.clear()
        self._cancer_type_counts.clear()
        self._sample_type_counts.clear()


class WsiStreamDataset(IterableDataset):
    """Wraps a ``PatchPipeline`` as a PyTorch ``IterableDataset``.

    Handles multi-worker ``DataLoader`` by partitioning slides across
    workers.  DDP rank partitioning should be done **before** creating
    this dataset (see :func:`partition_slides_by_rank`).

    Each yielded item is a ``dict`` with keys: ``"image"`` (single-view
    mode) or one tensor per configured view, coordinate fields (``"x"``,
    ``"y"``, ``"level"``, ``"patch_size"``, ``"slide_path"``, ``"mpp"``),
    ``"tissue_fraction"``, and when a ``dataset_adapter`` is configured:
    ``"patient_id"``, ``"cancer_type"``, ``"tissue_type"``, ``"sample_type"``,
    ``"dataset_name"``, ``"extra"`` (JSON string).

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
    patches_per_visit : int
        Patches read from one slide before round-robining to the next
        in the pool.  Higher values improve I/O locality on network
        filesystems.  Default ``1``.
    cycle : bool
        When ``True``, re-queue slides after exhaustion for infinite
        streaming (the typical mode for FM pretraining).
    replacement : str
        ``"with_replacement"`` (default) or ``"without_replacement"``.
        See :class:`~wsistream.pipeline.PatchPipeline` for details.
    slide_sampling : str
        ``"sequential"`` or ``"random"`` slide iteration order.
    seed : int or None
        Seed for all internal RNGs: slide-queue order, sampler, transforms,
        and crops.  Set this instead of seeds on individual components.
        ``None`` (default) draws fresh entropy for every iteration.
    views : list[ViewConfig] or None
        Optional multi-view configuration.  Mutually exclusive with
        ``transforms``; see :class:`~wsistream.pipeline.PatchPipeline`.
    shared_transforms : PatchTransform or None
        Optional transform chain applied once to the primary extracted patch
        before per-view crop and transform processing.  Requires ``views``.
    tissue_mask_cache_size : int
        Maximum tissue masks cached per worker and iterator. Uses
        least-recently-used eviction. ``0`` (default) disables caching.
    output_dtype : {"float32", "uint8"}
        Tensor dtype for images and views. ``"float32"`` (default) converts
        uint8 arrays to float32 and scales them to ``[0, 1]``. ``"uint8"``
        preserves uint8 values so workers transfer one quarter as many image
        bytes; conversion and normalization must then happen after device
        transfer.
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
        patches_per_visit: int = 1,
        cycle: bool = True,
        replacement: str = "with_replacement",
        slide_sampling: str = "random",
        seed: int | None = None,
        views: list[ViewConfig] | None = None,
        shared_transforms: PatchTransform | None = None,
        tissue_mask_cache_size: int = 0,
        output_dtype: Literal["float32", "uint8"] = "float32",
    ):
        if views is not None and transforms is not None:
            raise ValueError("transforms and views are mutually exclusive")
        if views is None and shared_transforms is not None:
            raise ValueError("shared_transforms requires views")
        if tissue_mask_cache_size < 0:
            raise ValueError(f"tissue_mask_cache_size must be >= 0, got {tissue_mask_cache_size}")
        if output_dtype not in ("float32", "uint8"):
            raise ValueError(f"output_dtype must be 'float32' or 'uint8', got {output_dtype!r}")
        self._slide_paths = resolve_slide_paths(slide_paths)
        self._backend = backend
        self._tissue_detector = tissue_detector
        self._sampler = sampler
        self._patch_filter = patch_filter
        self._transforms = transforms
        self._views = views
        self._shared_transforms = shared_transforms
        self._dataset_adapter = dataset_adapter
        self._pool_size = pool_size
        self._patches_per_slide = patches_per_slide
        self._patches_per_visit = patches_per_visit
        self._slide_sampling = slide_sampling
        self._cycle = cycle
        self._replacement = replacement
        self._seed = seed
        self._tissue_mask_cache_size = tissue_mask_cache_size
        self._output_dtype = output_dtype
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
            # Disable OpenCV's internal threadpool inside each forked worker. The patch
            # pipeline (tissue detection, resizing) is OpenCV-heavy; left at its default a
            # worker can inherit OpenCV's pthread mutexes in a locked state and deadlock on
            # its first call, and N workers each spawning a CPU-count threadpool only
            # oversubscribes. Parallelism comes from the workers themselves. Scoped to the
            # worker branch so the main process (num_workers=0) keeps the user's cv2 setting.
            cv2.setNumThreads(0)
            slides = self._slide_paths[worker_info.id :: worker_info.num_workers]
            # Map each (iteration, worker) pair to a distinct deterministic seed.
            worker_seed = (
                self._seed + self._iter_count * worker_info.num_workers + worker_info.id
                if self._seed is not None
                else None
            )
            logger.debug(
                "Worker %d/%d: %d slides",
                worker_info.id,
                worker_info.num_workers,
                len(slides),
            )
        else:
            slides = self._slide_paths
            worker_seed = self._seed + self._iter_count if self._seed is not None else None

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
            views=self._views,
            shared_transforms=self._shared_transforms,
            dataset_adapter=self._dataset_adapter,
            slide_sampling=self._slide_sampling,
            pool_size=self._pool_size,
            patches_per_slide=self._patches_per_slide,
            patches_per_visit=self._patches_per_visit,
            cycle=self._cycle,
            replacement=self._replacement,
            seed=worker_seed,
            tissue_mask_cache_size=self._tissue_mask_cache_size,
        )

        flush_interval = 16 if worker_info is not None else 1

        prev_counters = [0, 0, 0, 0, 0, 0, 0.0]  # sp, sf, pe, pf, ec, tfc, tft
        prev_slides_total = 0
        prev_slides_seen: set[str] = set()
        prev_mpp: dict = {}
        prev_cancer: dict = {}
        prev_sample: dict = {}
        patch_count = 0

        try:
            for result in pipeline:
                patch_count += 1
                if patch_count % flush_interval == 0:
                    (
                        prev_counters,
                        prev_slides_total,
                        prev_slides_seen,
                        prev_mpp,
                        prev_cancer,
                        prev_sample,
                    ) = self._flush_stats(
                        pipeline,
                        prev_counters,
                        prev_slides_total,
                        prev_slides_seen,
                        prev_mpp,
                        prev_cancer,
                        prev_sample,
                    )
                yield self._result_to_dict(result)
        finally:
            prev_counters, _, prev_slides_seen, prev_mpp, prev_cancer, prev_sample = (
                self._flush_stats(
                    pipeline,
                    prev_counters,
                    prev_slides_total,
                    prev_slides_seen,
                    prev_mpp,
                    prev_cancer,
                    prev_sample,
                )
            )
            # Force-push any remaining histogram deltas (mpp from last patches)
            s = pipeline.stats
            new_slides = list(s.slides_seen - prev_slides_seen)
            mpp_delta = {}
            for k, v in s.magnification_counts.items():
                d = v - prev_mpp.get(k, 0)
                if d:
                    mpp_delta[k] = d
            cancer_delta = {}
            for k, v in s.cancer_type_counts.items():
                d = v - prev_cancer.get(k, 0)
                if d:
                    cancer_delta[k] = d
            sample_delta = {}
            for k, v in s.sample_type_counts.items():
                d = v - prev_sample.get(k, 0)
                if d:
                    sample_delta[k] = d
            if new_slides or mpp_delta or cancer_delta or sample_delta:
                self._shared_stats.push_histograms(
                    new_slides, mpp_delta, cancer_delta, sample_delta
                )

    def _flush_stats(
        self,
        pipeline: PatchPipeline,
        prev_counters: list,
        prev_slides_total: int,
        prev_slides_seen: set,
        prev_mpp: dict,
        prev_cancer: dict,
        prev_sample: dict,
    ) -> tuple:
        """Compute deltas from pipeline stats and push to shared aggregator."""
        s = pipeline.stats
        tf = s.tissue_fractions

        # Current counter values
        cur = [
            s.slides_processed,
            s.slides_failed,
            s.patches_extracted,
            s.patches_filtered,
            s.error_count,
            tf.count,
            tf.total,
        ]

        # Compute deltas
        deltas = [c - p for c, p in zip(cur, prev_counters)]
        d_sp, d_sf, d_pe, d_pf, d_ec, d_tfc, d_tft = deltas

        if any(d != 0 for d in deltas):
            self._shared_stats.add_counters(
                slides_processed=d_sp,
                slides_failed=d_sf,
                patches_extracted=d_pe,
                patches_filtered=d_pf,
                error_count=d_ec,
                tf_count=d_tfc,
                tf_total=d_tft,
                tf_min=tf.min_val,
                tf_max=tf.max_val,
            )

        # Push histogram deltas and new slide paths only when slide count changes
        cur_slides_total = s.slides_processed + s.slides_failed
        if cur_slides_total > prev_slides_total:
            new_slides = list(s.slides_seen - prev_slides_seen)
            mpp_delta = {}
            for k, v in s.magnification_counts.items():
                d = v - prev_mpp.get(k, 0)
                if d:
                    mpp_delta[k] = d
            cancer_delta = {}
            for k, v in s.cancer_type_counts.items():
                d = v - prev_cancer.get(k, 0)
                if d:
                    cancer_delta[k] = d
            sample_delta = {}
            for k, v in s.sample_type_counts.items():
                d = v - prev_sample.get(k, 0)
                if d:
                    sample_delta[k] = d
            if new_slides or mpp_delta or cancer_delta or sample_delta:
                self._shared_stats.push_histograms(
                    new_slides, mpp_delta, cancer_delta, sample_delta
                )
            prev_slides_seen = set(s.slides_seen)
            prev_mpp = dict(s.magnification_counts)
            prev_cancer = dict(s.cancer_type_counts)
            prev_sample = dict(s.sample_type_counts)

        return list(cur), cur_slides_total, prev_slides_seen, prev_mpp, prev_cancer, prev_sample

    def _result_to_dict(self, result: PatchResult) -> dict:
        """Convert a PatchResult to a dict of collate-safe types."""
        coord = result.coordinate

        # Metadata first (includes slide_path from adapter)
        meta = result.slide_metadata
        item = meta.to_flat_dict() if meta else SlideMetadata.empty_dict()

        coord_fields = {
            "x": coord.x,
            "y": coord.y,
            "level": coord.level,
            "patch_size": coord.patch_size,
            "slide_path": coord.slide_path,
            "mpp": coord.mpp if coord.mpp is not None else -1.0,
            "tissue_fraction": result.tissue_fraction,
        }
        if result.views is not None:
            for name, view in result.views.items():
                item[name] = self._image_to_tensor(view)
            # Coordinate fields overwrite; these are authoritative.
            item.update(coord_fields)
        else:
            item.update({"image": self._image_to_tensor(result.image), **coord_fields})
        return item

    def _image_to_tensor(self, image: np.ndarray | None) -> torch.Tensor:
        """Convert an HWC numpy image to the configured CHW tensor dtype."""
        if image is None:
            raise ValueError("image array is None")
        if self._output_dtype == "uint8" and image.dtype != np.uint8:
            raise ValueError(
                "output_dtype='uint8' requires every pipeline image to have dtype uint8, "
                f"got dtype {image.dtype}. Remove NormalizeTransform or other dtype-changing "
                "transforms and convert or normalize the batch after device transfer."
            )
        image = np.ascontiguousarray(image)
        if not image.flags.writeable:
            image = image.copy()
        tensor = torch.from_numpy(image).permute(2, 0, 1)
        if self._output_dtype == "uint8":
            return tensor.contiguous()
        if image.dtype == np.uint8:
            return tensor.float() / 255.0
        return tensor.float()


# Re-export for convenience: `from wsistream.torch import MonitoredLoader`
from wsistream.torch_monitor import MonitoredLoader  # noqa: E402, F401
