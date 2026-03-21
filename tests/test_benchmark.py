"""Tests for wsistream.benchmark (throughput benchmarking)."""

from __future__ import annotations

import pytest

from tests.conftest import FakeBackend, fake_slide_paths
from wsistream.benchmark import BenchmarkResult, benchmark_throughput
from wsistream.sampling.random import RandomSampler
from wsistream.tissue.otsu import OtsuTissueDetector
from wsistream.torch import WsiStreamDataset


def _can_use_gloo() -> bool:
    try:
        import torch.distributed as dist

        return dist.is_available() and dist.is_gloo_available()
    except ImportError:
        return False


def _make_test_dataset(slide_paths, pool_size, patches_per_slide, seed):
    """Top-level factory for benchmark tests. Must be picklable."""
    return WsiStreamDataset(
        slide_paths=slide_paths,
        backend=FakeBackend(),
        tissue_detector=OtsuTissueDetector(),
        sampler=RandomSampler(patch_size=64, num_patches=-1, seed=42),
        pool_size=pool_size,
        patches_per_slide=patches_per_slide,
        cycle=True,  # benchmark measures steady-state; finite iteration would stop early
        seed=seed,
    )


def _make_cycle_false_dataset(slide_paths, pool_size, patches_per_slide, seed):
    return WsiStreamDataset(
        slide_paths=slide_paths,
        backend=FakeBackend(),
        tissue_detector=OtsuTissueDetector(),
        sampler=RandomSampler(patch_size=64, num_patches=-1, seed=42),
        pool_size=pool_size,
        patches_per_slide=patches_per_slide,
        cycle=False,
        seed=seed,
    )


class TestBenchmarkSingleRank:
    def test_returns_results(self):
        results = benchmark_throughput(
            make_dataset=_make_test_dataset,
            slide_paths=fake_slide_paths(4),
            num_workers=0,
            world_size=1,
            pool_size=2,
            patches_per_slide=10,
            batch_size=4,
            warmup_batches=2,
            measure_batches=5,
            verbose=False,
        )
        assert len(results) == 1
        assert isinstance(results[0], BenchmarkResult)

    def test_positive_throughput(self):
        results = benchmark_throughput(
            make_dataset=_make_test_dataset,
            slide_paths=fake_slide_paths(4),
            num_workers=0,
            world_size=1,
            pool_size=2,
            patches_per_slide=10,
            batch_size=4,
            warmup_batches=2,
            measure_batches=5,
            verbose=False,
        )
        assert results[0].effective_sync_throughput > 0
        assert results[0].aggregate_throughput > 0
        assert results[0].total_patches > 0

    def test_result_fields(self):
        results = benchmark_throughput(
            make_dataset=_make_test_dataset,
            slide_paths=fake_slide_paths(4),
            num_workers=0,
            world_size=1,
            pool_size=2,
            patches_per_slide=10,
            batch_size=4,
            warmup_batches=2,
            measure_batches=5,
            verbose=False,
        )
        r = results[0]
        assert r.num_workers == 0
        assert r.world_size == 1
        assert r.pool_size == 2
        assert r.patches_per_slide == 10
        assert r.batch_size == 4
        assert len(r.per_rank_patches_per_sec) == 1
        assert len(r.per_rank_batch_times_ms) == 1
        assert len(r.per_rank_batch_times_ms[0]) == 5

    def test_with_workers(self):
        results = benchmark_throughput(
            make_dataset=_make_test_dataset,
            slide_paths=fake_slide_paths(4),
            num_workers=[0, 1],
            world_size=1,
            pool_size=2,
            patches_per_slide=10,
            batch_size=4,
            warmup_batches=2,
            measure_batches=3,
            verbose=False,
        )
        assert len(results) == 2
        assert results[0].num_workers == 0
        assert results[1].num_workers == 1


class TestBenchmarkGridSearch:
    def test_sweep_multiple_params(self):
        results = benchmark_throughput(
            make_dataset=_make_test_dataset,
            slide_paths=fake_slide_paths(4),
            num_workers=[0, 1],
            world_size=1,
            pool_size=[1, 2],
            patches_per_slide=10,
            batch_size=4,
            warmup_batches=2,
            measure_batches=3,
            verbose=False,
        )
        # 2 num_workers x 2 pool_sizes = 4 configs
        assert len(results) == 4

    def test_skip_insufficient_slides(self):
        results = benchmark_throughput(
            make_dataset=_make_test_dataset,
            slide_paths=fake_slide_paths(2),
            num_workers=0,
            world_size=[1, 4],
            pool_size=2,
            patches_per_slide=10,
            batch_size=4,
            warmup_batches=2,
            measure_batches=3,
            verbose=False,
        )
        # world_size=4 should be skipped (only 2 slides)
        assert len(results) == 1
        assert results[0].world_size == 1


class TestBenchmarkMultiRank:
    @pytest.mark.skipif(
        not _can_use_gloo(),
        reason="torch distributed gloo backend not available",
    )
    def test_two_ranks(self):
        results = benchmark_throughput(
            make_dataset=_make_test_dataset,
            slide_paths=fake_slide_paths(4),
            num_workers=0,
            world_size=2,
            pool_size=2,
            patches_per_slide=10,
            batch_size=4,
            warmup_batches=2,
            measure_batches=3,
            verbose=False,
        )
        assert len(results) == 1
        r = results[0]
        assert r.world_size == 2
        assert len(r.per_rank_patches_per_sec) == 2
        assert r.effective_sync_throughput > 0
        assert all(pps > 0 for pps in r.per_rank_patches_per_sec)


class TestBenchmarkValidation:
    def test_empty_slides_raises(self):
        with pytest.raises(ValueError, match="slide_paths is empty"):
            benchmark_throughput(
                make_dataset=_make_test_dataset,
                slide_paths=[],
                verbose=False,
            )

    def test_cycle_false_raises(self):
        with pytest.raises(ValueError, match="cycle=True"):
            benchmark_throughput(
                make_dataset=_make_cycle_false_dataset,
                slide_paths=fake_slide_paths(4),
                num_workers=0,
                world_size=1,
                pool_size=2,
                patches_per_slide=10,
                batch_size=4,
                warmup_batches=1,
                measure_batches=1,
                verbose=False,
            )

    @pytest.mark.skipif(
        not _can_use_gloo(),
        reason="torch distributed gloo backend not available",
    )
    def test_unpicklable_factory_raises(self):
        # Nested functions are not picklable — DDP spawn will fail
        def bad_factory(sp, ps, pps, s):
            return _make_test_dataset(sp, ps, pps, s)

        with pytest.raises(TypeError, match="not picklable"):
            benchmark_throughput(
                make_dataset=bad_factory,
                slide_paths=fake_slide_paths(4),
                num_workers=0,
                world_size=2,
                verbose=False,
            )
