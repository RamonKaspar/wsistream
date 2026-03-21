"""Tests for wsistream.torch (WsiStreamDataset + DDP utilities)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from tests.conftest import FakeBackend, fake_slide_paths
from wsistream.datasets.base import DatasetAdapter
from wsistream.filters.base import PatchFilter
from wsistream.sampling.random import RandomSampler
from wsistream.tissue.otsu import OtsuTissueDetector
from wsistream.torch import WsiStreamDataset, partition_slides_by_rank
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


class _FailBackend(FakeBackend):
    def open(self, path: str) -> None:
        raise RuntimeError("disk on fire")


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
        slides = ["a", "b", "c", "d"]
        assert partition_slides_by_rank(slides, rank=0, world_size=2) == ["a", "c"]
        assert partition_slides_by_rank(slides, rank=1, world_size=2) == ["b", "d"]

    def test_single_rank(self):
        slides = ["a", "b", "c"]
        assert partition_slides_by_rank(slides, rank=0, world_size=1) == slides

    def test_world_size_zero_raises(self):
        with pytest.raises(ValueError, match="world_size must be >= 1"):
            partition_slides_by_rank(["a"], rank=0, world_size=0)

    def test_negative_rank_raises(self):
        with pytest.raises(ValueError, match="rank must be in"):
            partition_slides_by_rank(["a", "b"], rank=-1, world_size=2)

    def test_rank_too_large_raises(self):
        with pytest.raises(ValueError, match="rank must be in"):
            partition_slides_by_rank(["a", "b"], rank=2, world_size=2)

    def test_too_few_slides_raises(self):
        with pytest.raises(RuntimeError, match="got 0 slides"):
            partition_slides_by_rank(["a"], rank=1, world_size=3)

    def test_auto_detection_without_env(self, monkeypatch):
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        slides = ["a", "b", "c"]
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
            n_slides=1, patches_per_slide=2, dataset_adapter=_FakeAdapter(),
        )
        for item in dataset:
            assert item["patient_id"] == "P001"
            assert item["cancer_type"] == "BRCA"
            assert item["dataset_name"] == "test"


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
            n_slides=2, patches_per_slide=4, dataset_adapter=_FakeAdapter(),
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
            n_slides=1, patches_per_slide=10, patch_filter=_RejectAll(),
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

    def test_stats_accumulate_across_iterations(self):
        dataset = _make_dataset(n_slides=1, patches_per_slide=5)
        list(dataset)
        list(dataset)
        assert dataset.stats_dict()["pipeline/patches_extracted"] == 10


# ── Re-iteration seed diversity ──


class TestSeedDiversity:
    def test_re_iteration_produces_different_coordinates(self):
        dataset = _make_dataset(n_slides=1, patches_per_slide=5, seed=42)
        coords1 = [(item["x"], item["y"]) for item in dataset]
        coords2 = [(item["x"], item["y"]) for item in dataset]
        assert coords1 != coords2


# ── Parameter validation ──


class TestParameterValidation:
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
