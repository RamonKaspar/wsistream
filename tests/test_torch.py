"""Tests for wsistream.torch (WsiStreamDataset + DDP utilities)."""

from __future__ import annotations

import gc
import multiprocessing as mp
import threading
import warnings

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from tests.conftest import FakeBackend, fake_slide_paths
from wsistream.datasets.base import DatasetAdapter
from wsistream.filters.base import PatchFilter
from wsistream.sampling.random import RandomSampler
from wsistream.tissue.otsu import OtsuTissueDetector
from wsistream.torch import WsiStreamDataset, _StatsAggregator, partition_slides_by_rank
from wsistream.transforms import NormalizeTransform
from wsistream.transforms.base import PatchTransform
from wsistream.types import SlideMetadata


# ── helpers ──


def _make_dataset(n_slides=3, patches_per_slide=5, **kwargs) -> WsiStreamDataset:
    defaults = dict(
        slide_paths=fake_slide_paths(n_slides),
        backend=FakeBackend(),
        tissue_detector=OtsuTissueDetector(),
        sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
        pool_size=min(2, n_slides),
        patches_per_slide=patches_per_slide,
        cycle=False,
    )
    defaults.update(kwargs)
    return WsiStreamDataset(**defaults)


class _RejectAll(PatchFilter):
    def accept(self, patch: np.ndarray) -> bool:
        return False


class _ReadOnlyTransform(PatchTransform):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        result = image.copy()
        result.flags.writeable = False
        return result


class _FailBackend(FakeBackend):
    def open(self, path: str) -> None:
        raise RuntimeError("disk on fire")


def _push_histograms(aggregator: _StatsAggregator, count: int) -> None:
    """Send ``count`` histogram messages, each naming a distinct long slide path."""
    for i in range(count):
        aggregator.push_histograms([f"/data/{'x' * 200}/slide_{i}.svs"], {0.5: 1}, {}, {})


class _FakeAdapter(DatasetAdapter):
    def parse_metadata(self, slide_path: str) -> SlideMetadata:
        return SlideMetadata(
            slide_path=slide_path,
            dataset_name="test",
            patient_id="P001",
            cancer_type="BRCA",
            tissue_type="breast",
            sample_type="primary",
        )


# ── partition_slides_by_rank ──


class TestPartitionSlidesByRank:
    def test_basic_partitioning(self):
        slides = fake_slide_paths(4)
        assert partition_slides_by_rank(slides, rank=0, world_size=2) == slides[::2]
        assert partition_slides_by_rank(slides, rank=1, world_size=2) == slides[1::2]

    def test_single_rank(self):
        slides = fake_slide_paths(3)
        assert partition_slides_by_rank(slides, rank=0, world_size=1) == slides

    def test_world_size_zero_raises(self):
        with pytest.raises(ValueError, match="world_size must be >= 1"):
            partition_slides_by_rank(fake_slide_paths(1), rank=0, world_size=0)

    def test_negative_rank_raises(self):
        with pytest.raises(ValueError, match="rank must be in"):
            partition_slides_by_rank(fake_slide_paths(2), rank=-1, world_size=2)

    def test_rank_too_large_raises(self):
        with pytest.raises(ValueError, match="rank must be in"):
            partition_slides_by_rank(fake_slide_paths(2), rank=2, world_size=2)

    def test_too_few_slides_raises(self):
        with pytest.raises(RuntimeError, match="got 0 slides"):
            partition_slides_by_rank(fake_slide_paths(1), rank=1, world_size=3)

    def test_auto_detection_without_env(self, monkeypatch):
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        slides = fake_slide_paths(3)
        assert partition_slides_by_rank(slides) == slides


# ── WsiStreamDataset basic iteration ──


class TestWsiStreamDatasetIteration:
    def test_produces_dicts(self):
        dataset = _make_dataset(n_slides=2, patches_per_slide=3)
        items = list(dataset)
        assert len(items) == 6
        assert all(isinstance(item, dict) for item in items)

    def test_image_tensor_shape(self):
        dataset = _make_dataset(n_slides=1, patches_per_slide=3)
        for item in dataset:
            assert item["image"].shape == (3, 256, 256)
            assert item["image"].dtype == torch.float32
            assert item["image"].min() >= 0.0
            assert item["image"].max() <= 1.0

    def test_uint8_output_preserves_pixels_and_reduces_tensor_bytes(self):
        float_items = list(_make_dataset(n_slides=1, patches_per_slide=1))
        uint8_items = list(
            _make_dataset(
                n_slides=1,
                patches_per_slide=1,
                output_dtype="uint8",
            )
        )
        float_image = float_items[0]["image"]
        uint8_image = uint8_items[0]["image"]

        assert len(float_items) == 1
        assert len(uint8_items) == 1
        assert uint8_image.dtype == torch.uint8
        assert uint8_image.is_contiguous()
        uint8_bytes = uint8_image.numel() * uint8_image.element_size()
        float_bytes = float_image.numel() * float_image.element_size()
        assert uint8_bytes * 4 == float_bytes
        torch.testing.assert_close(uint8_image.float() / 255.0, float_image)

    def test_uint8_output_rejects_float_transform_results(self):
        dataset = _make_dataset(
            n_slides=1,
            patches_per_slide=1,
            transforms=NormalizeTransform(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            output_dtype="uint8",
        )

        with pytest.raises(ValueError, match="output_dtype='uint8'.*dtype float32"):
            next(iter(dataset))

    def test_float32_output_does_not_rescale_normalized_results(self):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        dataset = _make_dataset(
            n_slides=1,
            patches_per_slide=1,
            transforms=NormalizeTransform(mean=mean, std=std),
        )

        items = list(dataset)
        image = items[0]["image"]
        expected = torch.tensor([(128 / 255.0 - m) / s for m, s in zip(mean, std)])

        assert len(items) == 1
        assert image.dtype == torch.float32
        torch.testing.assert_close(image[:, 0, 0], expected)

    def test_uint8_output_copies_non_writable_arrays(self):
        dataset = _make_dataset(
            n_slides=1,
            patches_per_slide=1,
            transforms=_ReadOnlyTransform(),
            output_dtype="uint8",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            items = list(dataset)

        assert len(items) == 1
        image = items[0]["image"]
        image[0, 0, 0] = 0
        assert image.dtype == torch.uint8
        assert image[0, 0, 0] == 0

    def test_coordinate_fields_present(self):
        dataset = _make_dataset(n_slides=1, patches_per_slide=2)
        for item in dataset:
            assert isinstance(item["x"], int)
            assert isinstance(item["y"], int)
            assert isinstance(item["level"], int)
            assert isinstance(item["patch_size"], int)
            assert isinstance(item["slide_path"], str)
            assert item["slide_path"] != ""  # must not be overwritten by empty metadata

    def test_mpp_always_present(self):
        dataset = _make_dataset(n_slides=1, patches_per_slide=2)
        for item in dataset:
            assert "mpp" in item
            assert isinstance(item["mpp"], float)

    def test_metadata_fields_always_present(self):
        dataset = _make_dataset(n_slides=1, patches_per_slide=2)
        for item in dataset:
            assert "patient_id" in item
            assert "cancer_type" in item
            assert "tissue_type" in item
            assert "sample_type" in item
            assert "dataset_name" in item
            assert "extra" in item

    def test_metadata_empty_without_adapter(self):
        dataset = _make_dataset(n_slides=1, patches_per_slide=2, dataset_adapter=None)
        for item in dataset:
            assert item["patient_id"] == ""
            assert item["cancer_type"] == ""
            assert item["extra"] == "{}"

    def test_metadata_populated_with_adapter(self):
        dataset = _make_dataset(
            n_slides=1,
            patches_per_slide=2,
            dataset_adapter=_FakeAdapter(),
        )
        for item in dataset:
            assert item["patient_id"] == "P001"
            assert item["cancer_type"] == "BRCA"
            assert item["dataset_name"] == "test"

    def test_tissue_mask_cache_forwarded_to_pipeline(self, monkeypatch):
        detector = OtsuTissueDetector()
        call_count = 0
        original_detect = detector.detect

        def count_detect(thumbnail, downsample=(1.0, 1.0)):
            nonlocal call_count
            call_count += 1
            return original_detect(thumbnail, downsample=downsample)

        monkeypatch.setattr(detector, "detect", count_detect)
        dataset = _make_dataset(
            n_slides=1,
            patches_per_slide=1,
            cycle=True,
            tissue_detector=detector,
            tissue_mask_cache_size=1,
        )
        iterator = iter(dataset)
        try:
            for _ in range(3):
                next(iterator)
        finally:
            iterator.close()

        assert call_count == 1


# ── DataLoader collation ──


class TestDataLoaderCollation:
    def test_batched_collation_works(self):
        dataset = _make_dataset(n_slides=2, patches_per_slide=4)
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        batch = next(iter(loader))
        assert batch["image"].shape == (4, 3, 256, 256)
        assert len(batch["slide_path"]) == 4

    def test_collation_with_adapter(self):
        dataset = _make_dataset(
            n_slides=2,
            patches_per_slide=4,
            dataset_adapter=_FakeAdapter(),
        )
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        batch = next(iter(loader))
        assert len(batch["patient_id"]) == 4
        assert all(pid == "P001" for pid in batch["patient_id"])


# ── Shared stats ──


class TestSharedStats:
    def test_stats_dict_after_iteration(self):
        dataset = _make_dataset(n_slides=2, patches_per_slide=5)
        list(dataset)
        stats = dataset.stats_dict()
        assert stats["pipeline/patches_extracted"] == 10
        assert stats["pipeline/slides_processed"] == 2
        assert stats["pipeline/slides_failed"] == 0

    def test_stats_with_filter_rejection(self):
        dataset = _make_dataset(
            n_slides=1,
            patches_per_slide=10,
            patch_filter=_RejectAll(),
        )
        list(dataset)
        stats = dataset.stats_dict()
        assert stats["pipeline/patches_extracted"] == 0
        assert stats["pipeline/patches_filtered"] > 0

    def test_stats_all_slides_fail(self):
        dataset = _make_dataset(n_slides=3, patches_per_slide=5, backend=_FailBackend())
        list(dataset)
        stats = dataset.stats_dict()
        assert stats["pipeline/slides_failed"] == 3
        assert stats["pipeline/patches_extracted"] == 0

    def test_stats_reset(self):
        dataset = _make_dataset(n_slides=1, patches_per_slide=5)
        list(dataset)
        assert dataset.stats_dict()["pipeline/patches_extracted"] == 5

        dataset.reset_stats()
        assert dataset.stats_dict()["pipeline/patches_extracted"] == 0

    def test_stats_reset_during_iteration(self):
        """After a reset, slides_unique and tissue fractions cover only later slides."""
        paths = fake_slide_paths(2)

        class _TissuePerSlide(FakeBackend):
            def get_thumbnail(self, size):
                thumb = np.full((size[1], size[0], 3), 240, dtype=np.uint8)
                side = size[0] // (4 if self._path == paths[0] else 2)
                thumb[:side, :side] = 80
                return thumb

        dataset = _make_dataset(
            n_slides=2,
            patches_per_slide=5,
            backend=_TissuePerSlide(),
            sampler=RandomSampler(patch_size=32, num_patches=-1, tissue_threshold=0.0),
            pool_size=1,
            cycle=True,
        )
        iterator = iter(dataset)
        for _ in range(10):
            next(iterator)  # slide 0, then slide 1
        dataset.reset_stats()
        for _ in range(5):
            next(iterator)  # slide 0 again

        stats = dataset.stats_dict()
        assert stats["pipeline/patches_extracted"] == 5
        assert stats["pipeline/slides_processed"] == 1
        assert stats["pipeline/slides_unique"] == 1
        assert stats["pipeline/min_tissue_fraction"] == stats["pipeline/max_tissue_fraction"]

    def test_stats_accumulate_across_iterations(self):
        dataset = _make_dataset(n_slides=1, patches_per_slide=5)
        list(dataset)
        list(dataset)
        assert dataset.stats_dict()["pipeline/patches_extracted"] == 10

    def test_slides_unique_stable_across_iterations(self):
        """Repeated iteration over the same slides must not inflate slides_unique."""
        dataset = _make_dataset(n_slides=3, patches_per_slide=5)
        list(dataset)
        stats1 = dataset.stats_dict()
        assert stats1["pipeline/slides_unique"] == 3
        assert stats1["pipeline/slides_processed"] == 3

        list(dataset)
        stats2 = dataset.stats_dict()
        assert stats2["pipeline/slides_unique"] == 3  # still 3, not 6
        assert stats2["pipeline/slides_processed"] == 6  # 3 + 3

    def test_unread_histograms_do_not_block_workers(self):
        """A worker must not block when stats_dict() is not called; 2000 messages overfill a pipe."""
        aggregator = _StatsAggregator()
        worker = mp.get_context("spawn").Process(target=_push_histograms, args=(aggregator, 2000))
        worker.start()
        worker.join(timeout=60)
        blocked = worker.is_alive()
        if blocked:
            worker.terminate()
        assert not blocked

        stats = aggregator.to_dict()
        assert stats["pipeline/mpp_0.50"] == 2000
        assert stats["pipeline/slides_unique"] == 2000

    def test_unread_histograms_do_not_block_the_main_process(self):
        """With num_workers=0, the main process writes the histogram messages itself."""
        aggregator = _StatsAggregator()
        writer = threading.Thread(target=_push_histograms, args=(aggregator, 2000), daemon=True)
        writer.start()
        writer.join(timeout=60)
        assert not writer.is_alive()
        assert aggregator.to_dict()["pipeline/mpp_0.50"] == 2000

    def test_drain_thread_ends_with_its_aggregator(self):
        aggregator = _StatsAggregator()
        drainer = aggregator._drainer
        del aggregator
        gc.collect()
        drainer.join(timeout=10)
        assert not drainer.is_alive()


# ── Re-iteration seed diversity ──


class TestSeedDiversity:
    def test_re_iteration_produces_different_coordinates(self):
        dataset = _make_dataset(n_slides=1, patches_per_slide=5, seed=42)
        coords1 = [(item["x"], item["y"]) for item in dataset]
        coords2 = [(item["x"], item["y"]) for item in dataset]
        assert coords1 != coords2

    def test_unseeded_instances_use_fresh_entropy(self):
        first = _make_dataset(n_slides=2, patches_per_slide=5, seed=None)
        second = _make_dataset(n_slides=2, patches_per_slide=5, seed=None)

        first_coords = [(item["x"], item["y"]) for item in first]
        second_coords = [(item["x"], item["y"]) for item in second]
        assert first_coords != second_coords

    @staticmethod
    def _epoch_coordinates(loader: DataLoader, epochs: int = 3) -> list[tuple]:
        return [
            tuple(
                sorted(
                    (path, int(x), int(y))
                    for batch in loader
                    for path, x, y in zip(batch["slide_path"], batch["x"], batch["y"])
                )
            )
            for _ in range(epochs)
        ]

    @pytest.mark.parametrize("worker_context", mp.get_all_start_methods())
    def test_non_persistent_workers_draw_distinct_epochs(self, worker_context):
        """Workers receive a fresh copy of the dataset every epoch; streams must still differ."""
        dataset = _make_dataset(n_slides=4, patches_per_slide=3, seed=42)
        loader = DataLoader(
            dataset,
            batch_size=4,
            num_workers=2,
            multiprocessing_context=worker_context,
        )

        epochs = self._epoch_coordinates(loader)

        assert len(set(epochs)) == 3

    def test_non_persistent_workers_match_persistent_workers(self):
        def epochs(persistent_workers: bool) -> list[tuple]:
            dataset = _make_dataset(n_slides=4, patches_per_slide=3, seed=42)
            loader = DataLoader(
                dataset,
                batch_size=4,
                num_workers=2,
                persistent_workers=persistent_workers,
            )
            return self._epoch_coordinates(loader)

        assert epochs(persistent_workers=False) == epochs(persistent_workers=True)

    def test_adjacent_seeds_draw_distinct_worker_streams(self):
        """seed=base_seed + rank must not give workers on two ranks the same stream."""
        paths = fake_slide_paths(4)

        def worker_coordinates(seed: int) -> list[list[tuple[int, int]]]:
            dataset = _make_dataset(n_slides=4, patches_per_slide=3, seed=seed)
            loader = DataLoader(dataset, batch_size=None, num_workers=2)
            coordinates: list[list[tuple[int, int]]] = [[], []]
            for item in loader:
                worker_id = paths.index(item["slide_path"]) % 2
                coordinates[worker_id].append((int(item["x"]), int(item["y"])))
            return coordinates

        streams = worker_coordinates(42) + worker_coordinates(43)
        assert len({tuple(stream) for stream in streams}) == 4


# ── Parameter validation ──


class TestParameterValidation:
    def test_invalid_output_dtype_raises(self):
        with pytest.raises(ValueError, match="output_dtype must be 'float32' or 'uint8'"):
            _make_dataset(output_dtype="float16")

    def test_negative_tissue_mask_cache_size_raises(self):
        with pytest.raises(ValueError, match="tissue_mask_cache_size must be >= 0"):
            _make_dataset(tissue_mask_cache_size=-1)

    def test_invalid_slide_sampling_raises(self):
        dataset = _make_dataset(slide_sampling="bogus")
        with pytest.raises(ValueError, match="slide_sampling"):
            list(dataset)

    def test_valid_slide_sampling_sequential(self):
        dataset = _make_dataset(slide_sampling="sequential")
        items = list(dataset)
        assert len(items) > 0

    def test_valid_slide_sampling_random(self):
        dataset = _make_dataset(slide_sampling="random")
        items = list(dataset)
        assert len(items) > 0

    def test_cycle_true(self):
        dataset = _make_dataset(n_slides=1, patches_per_slide=3, cycle=True)
        count = 0
        for _ in dataset:
            count += 1
            if count >= 10:
                break
        assert count == 10

    def test_cycle_false(self):
        dataset = _make_dataset(n_slides=2, patches_per_slide=3, cycle=False)
        items = list(dataset)
        assert len(items) == 6


# ── Multi-worker ──


class TestMultiWorker:
    def test_uint8_output_crosses_worker_boundary_without_conversion(self):
        dataset = _make_dataset(
            n_slides=4,
            patches_per_slide=2,
            output_dtype="uint8",
        )
        loader = DataLoader(dataset, batch_size=4, num_workers=2)

        batches = list(loader)
        batch = batches[0]

        assert len(batches) == 2
        assert batch["image"].dtype == torch.uint8
        assert batch["image"].shape == (4, 3, 256, 256)

    def test_multi_worker_produces_patches(self):
        dataset = _make_dataset(n_slides=4, patches_per_slide=3)
        loader = DataLoader(dataset, batch_size=2, num_workers=2)
        items = []
        for batch in loader:
            items.append(batch)
            if len(items) >= 3:
                break
        assert len(items) >= 1

    def test_multi_worker_stats_aggregated(self):
        dataset = _make_dataset(n_slides=4, patches_per_slide=3)
        loader = DataLoader(dataset, batch_size=2, num_workers=2)
        for batch in loader:
            pass  # exhaust
        stats = dataset.stats_dict()
        assert stats["pipeline/patches_extracted"] == 12
        assert stats["pipeline/slides_processed"] == 4

    @pytest.mark.parametrize("worker_context", mp.get_all_start_methods())
    def test_all_worker_contexts_aggregate_stats(self, worker_context, monkeypatch):
        import wsistream.torch as torch_module

        if "fork" in mp.get_all_start_methods():
            # Simulate constructing the dataset under Linux's fork-default context
            # without changing the process-wide start method for other tests.
            monkeypatch.setattr(torch_module, "mp", mp.get_context("fork"))
        dataset = _make_dataset(n_slides=4, patches_per_slide=3)
        loader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=2,
            multiprocessing_context=worker_context,
        )

        list(loader)

        stats = dataset.stats_dict()
        assert stats["pipeline/patches_extracted"] == 12
        assert stats["pipeline/slides_processed"] == 4
        assert stats["pipeline/slides_unique"] == 4


class TestWorkerOpenCVThreads:
    """The patch pipeline is OpenCV-heavy, and forked DataLoader workers can inherit a
    locked OpenCV mutex and deadlock. Iterating inside a worker must disable OpenCV's
    threadpool; iterating in the main process must leave the user's setting untouched."""

    def test_worker_iteration_disables_opencv_threads(self, monkeypatch):
        from types import SimpleNamespace

        original = cv2.getNumThreads()
        try:
            cv2.setNumThreads(4)
            monkeypatch.setattr(
                torch.utils.data,
                "get_worker_info",
                lambda: SimpleNamespace(id=0, num_workers=1),
            )
            next(iter(_make_dataset(n_slides=2)))
            # setNumThreads(0) disables the pool; getNumThreads() then reports 1.
            assert cv2.getNumThreads() == 1
        finally:
            cv2.setNumThreads(original)

    def test_main_process_iteration_leaves_opencv_threads(self, monkeypatch):
        original = cv2.getNumThreads()
        try:
            cv2.setNumThreads(4)
            expected = cv2.getNumThreads()  # whatever OpenCV actually set (may clamp)
            monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: None)
            next(iter(_make_dataset(n_slides=2)))
            assert cv2.getNumThreads() == expected
        finally:
            cv2.setNumThreads(original)
