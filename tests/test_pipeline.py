"""Tests for PatchPipeline (using FakeBackend, no WSI files needed)."""

from __future__ import annotations

import copy
import dataclasses
import multiprocessing as mp
import os
import pickle
from collections import Counter

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset

from tests.conftest import FakeBackend, fake_slide_paths
from wsistream.filters.base import PatchFilter
from wsistream.pipeline import PatchPipeline, PipelineStats
from wsistream.sampling.base import PatchSampler
from wsistream.sampling.random import RandomSampler
from wsistream.tissue.base import TissueDetector
from wsistream.tissue.otsu import OtsuTissueDetector
from wsistream.transforms import (
    ComposeTransforms,
    NormalizeTransform,
    ResizeTransform,
)
from wsistream.transforms.base import PatchTransform
from wsistream.types import PatchCoordinate
from wsistream.views import RandomResizedCrop, ViewConfig


# ── helpers ──


def _make_pipeline(n_slides=3, patches_per_slide=5, **kwargs) -> PatchPipeline:
    defaults = dict(
        slide_paths=fake_slide_paths(n_slides),
        backend=FakeBackend(),
        tissue_detector=OtsuTissueDetector(),
        sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
        pool_size=max(1, min(2, n_slides)),
        patches_per_slide=patches_per_slide,
    )
    defaults.update(kwargs)
    return PatchPipeline(**defaults)


class _RejectAll(PatchFilter):
    """Filter that rejects every patch."""

    def accept(self, patch: np.ndarray) -> bool:
        return False


class _CountTransform(PatchTransform):
    """Transform that tracks how many times it was called."""

    def __init__(self):
        self.call_count = 0

    def __call__(self, image: np.ndarray) -> np.ndarray:
        self.call_count += 1
        return image


class _RandomOffsetTransform(PatchTransform):
    """Transform with internal RNG for worker-isolation tests."""

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        delta = int(self._rng.integers(1, 8))
        out = np.clip(image.astype(np.int16) + delta, 0, 255)
        return out.astype(np.uint8)


class _NonCopyableTransform(PatchTransform):
    def __deepcopy__(self, memo):
        raise RuntimeError("cannot copy")

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return image


class _CountingDetector(TissueDetector):
    """Return an all-tissue mask while tracking calls and optional failures."""

    def __init__(self, failures: int = 0) -> None:
        self.call_count = 0
        self.failures = failures

    def detect(self, thumbnail, downsample=(1.0, 1.0)):
        self.call_count += 1
        if self.call_count <= self.failures:
            raise RuntimeError("tissue detection failed")
        return np.ones(thumbnail.shape[:2], dtype=bool)


class _CountingThumbnailBackend(FakeBackend):
    """Track thumbnail generation across deep-copied slide handles."""

    call_count = 0

    def get_thumbnail(self, size: tuple[int, int]) -> np.ndarray:
        type(self).call_count += 1
        return super().get_thumbnail(size)


class _MaskMutatingSampler(PatchSampler):
    """Mutate active masks to verify cached arrays remain isolated."""

    def __init__(self) -> None:
        self.initial_fractions = []

    def sample(self, slide, tissue_mask):
        self.initial_fractions.append(tissue_mask.tissue_fraction)
        tissue_mask.mask[:] = False
        yield PatchCoordinate(
            x=0,
            y=0,
            level=0,
            patch_size=256,
            mpp=slide.properties.mpp,
            slide_path=slide.properties.path,
        )


def _take(pipeline: PatchPipeline, count: int):
    """Take a finite prefix from a cycling pipeline and close its generator."""
    iterator = iter(pipeline)
    try:
        return [next(iterator) for _ in range(count)]
    finally:
        iterator.close()


def _pipeline_seed_probe(connection, seed: int) -> None:
    """Return RNG samples from a fresh spawned process."""
    pipeline = _make_pipeline(
        n_slides=2,
        seed=seed,
        transforms=ComposeTransforms([_RandomOffsetTransform()]),
    )
    transform = pipeline.transforms.transforms[0]
    state = (
        pipeline._rng.integers(0, 2**63, size=8).tolist(),
        pipeline.sampler._rng.integers(0, 2**63, size=8).tolist(),
        transform._rng.integers(0, 2**63, size=8).tolist(),
    )
    connection.send((os.getpid(), state))
    connection.close()


def _pipeline_cache_fork_probe(connection, pipeline: PatchPipeline) -> None:
    """Return detector and cache state after using an inherited pipeline."""
    list(pipeline)
    detector = pipeline.tissue_detector
    assert isinstance(detector, _CountingDetector)
    connection.send(
        (
            detector.call_count,
            len(pipeline._tissue_mask_cache),
        )
    )
    connection.close()


class _WorkerProbeDataset(IterableDataset):
    """Yield worker id, coordinates, and transformed patch checksum."""

    def __init__(self, slide_paths: list[str]) -> None:
        self.slide_paths = slide_paths

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        slides = self.slide_paths
        if worker_info is not None:
            slides = slides[worker_info.id :: worker_info.num_workers]

        pipeline = PatchPipeline(
            slide_paths=slides,
            backend=FakeBackend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            transforms=ComposeTransforms(transforms=[_RandomOffsetTransform(seed=123)]),
            pool_size=1,
            patches_per_slide=2,
            cycle=False,
        )

        for result in pipeline:
            yield worker_id, result.coordinate.x, result.coordinate.y, int(result.image.sum())


# ── PipelineCreation ──


class TestPipelineCreation:
    def test_missing_backend_raises(self):
        with pytest.raises(TypeError, match="explicit backend"):
            PatchPipeline(slide_paths=["test.svs"])

    def test_default_pool_params(self):
        fields = {
            f.name: f.default
            for f in dataclasses.fields(PatchPipeline)
            if f.default is not dataclasses.MISSING
        }
        assert fields["pool_size"] == 8
        assert fields["patches_per_slide"] == 100
        assert fields["cycle"] is False


# ── PipelineStats ──


class TestPipelineStats:
    def test_to_dict(self):
        stats = PipelineStats()
        stats.slides_processed = 5
        stats.patches_extracted = 100
        for v in [0.3, 0.5, 0.7]:
            stats.tissue_fractions.update(v)
        stats.magnification_counts = {0.5: 60, 1.0: 40}
        stats.cancer_type_counts = {"TCGA-BRCA": 3, "TCGA-LUAD": 2}
        stats.sample_type_counts = {"Primary Solid Tumor": 5}

        d = stats.to_dict()
        assert d["pipeline/slides_processed"] == 5
        assert d["pipeline/patches_extracted"] == 100
        assert abs(d["pipeline/mean_tissue_fraction"] - 0.5) < 1e-6
        assert d["pipeline/mpp_0.50"] == 60
        assert d["pipeline/cancer_type/TCGA-BRCA"] == 3
        assert "pipeline/sample_type/primary_solid_tumor" in d

    def test_patches_filtered_in_dict(self):
        stats = PipelineStats()
        stats.patches_filtered = 42
        d = stats.to_dict()
        assert d["pipeline/patches_filtered"] == 42

    def test_empty_stats(self):
        d = PipelineStats().to_dict()
        assert d["pipeline/slides_processed"] == 0
        assert d["pipeline/patches_extracted"] == 0
        assert d["pipeline/patches_filtered"] == 0
        assert "pipeline/mean_tissue_fraction" not in d
        assert "pipeline/error_count" not in d

    def test_error_count(self):
        stats = PipelineStats()
        stats.record_error("slide.svs", "corrupt file")
        d = stats.to_dict()
        assert d["pipeline/error_count"] == 1

    def test_reset_stats(self):
        pipeline = _make_pipeline(n_slides=1, patches_per_slide=3)
        list(pipeline)
        assert pipeline.stats.slides_processed > 0
        assert pipeline.stats.patches_extracted > 0

        pipeline.reset_stats()

        assert pipeline.stats.slides_processed == 0
        assert pipeline.stats.patches_extracted == 0
        assert pipeline.stats.tissue_fractions.count == 0

    def test_mpp_none_in_dict(self):
        stats = PipelineStats()
        stats.magnification_counts = {None: 10}
        d = stats.to_dict()
        assert d["pipeline/mpp_unknown"] == 10


# ── Pipeline Iteration ──


class TestPipelineIteration:
    def test_produces_patches(self):
        pipeline = _make_pipeline(n_slides=2, patches_per_slide=5)
        patches = list(pipeline)
        assert len(patches) == 10  # 2 slides × 5 each

    def test_patches_have_correct_fields(self):
        pipeline = _make_pipeline(n_slides=1, patches_per_slide=3)
        for result in pipeline:
            assert result.image.shape == (256, 256, 3)
            assert result.image.dtype == np.uint8
            assert result.coordinate.patch_size == 256
            assert 0.0 <= result.tissue_fraction <= 1.0

    def test_stats_updated(self):
        pipeline = _make_pipeline(n_slides=2, patches_per_slide=5)
        list(pipeline)
        assert pipeline.stats.slides_processed == 2
        assert pipeline.stats.patches_extracted == 10

    def test_slides_unique_without_cycle(self):
        """Without cycling, unique == processed."""
        pipeline = _make_pipeline(n_slides=3, patches_per_slide=5, cycle=False)
        list(pipeline)
        assert len(pipeline.stats.slides_seen) == 3
        assert pipeline.stats.slides_processed == 3
        d = pipeline.stats.to_dict()
        assert d["pipeline/slides_unique"] == 3

    def test_slides_unique_with_cycle(self):
        """With cycling, unique < processed because slides are revisited."""
        pipeline = _make_pipeline(n_slides=2, patches_per_slide=3, cycle=True)
        count = 0
        for _ in pipeline:
            count += 1
            if count >= 18:  # 3 passes × 2 slides × 3 patches
                break
        assert len(pipeline.stats.slides_seen) == 2
        assert pipeline.stats.slides_processed >= 6  # 2 slides × 3 passes
        d = pipeline.stats.to_dict()
        assert d["pipeline/slides_unique"] == 2
        assert d["pipeline/slides_processed"] >= 6

    def test_empty_slide_paths(self):
        pipeline = _make_pipeline(n_slides=0)
        patches = list(pipeline)
        assert len(patches) == 0


class TestRoundRobin:
    def test_interleaves_slides(self):
        """Patches should alternate between slides, not come in blocks."""
        pipeline = _make_pipeline(n_slides=3, patches_per_slide=6, pool_size=3)
        slide_order = [r.coordinate.slide_path for r in pipeline]

        # First 3 patches should come from 3 different slides
        first_three = set(slide_order[:3])
        assert len(first_three) == 3

    def test_patches_per_slide_respected(self):
        cap = 4
        pipeline = _make_pipeline(n_slides=3, patches_per_slide=cap)
        per_slide = Counter()
        for result in pipeline:
            per_slide[result.coordinate.slide_path] += 1
        for slide, count in per_slide.items():
            assert count == cap, f"{slide} got {count}, expected {cap}"


class TestSlideSampling:
    @staticmethod
    def _seed_state_in_spawned_process(seed: int):
        context = mp.get_context("spawn")
        receive, send = context.Pipe(duplex=False)
        process = context.Process(target=_pipeline_seed_probe, args=(send, seed))
        process.start()
        send.close()
        try:
            assert receive.poll(20), "spawned seed probe did not return"
            result = receive.recv()
        finally:
            receive.close()
            process.join(20)
            if process.is_alive():
                process.terminate()
                process.join()
        assert process.exitcode == 0
        return result

    def test_seed_is_reproducible_across_processes(self):
        first_pid, first_state = self._seed_state_in_spawned_process(42)
        second_pid, second_state = self._seed_state_in_spawned_process(42)

        assert first_pid != second_pid
        assert first_state == second_state

    def test_shared_components_do_not_change_an_existing_pipeline(self):
        sampler = RandomSampler(patch_size=256, num_patches=-1)
        transform = _RandomOffsetTransform()
        sampler_state = pickle.dumps(sampler._rng.bit_generator.state)
        transform_state = pickle.dumps(transform._rng.bit_generator.state)
        pipeline1 = _make_pipeline(
            n_slides=1,
            sampler=sampler,
            transforms=transform,
            seed=11,
        )
        pipeline2 = _make_pipeline(
            n_slides=1,
            sampler=sampler,
            transforms=transform,
            seed=22,
        )
        reference = _make_pipeline(
            n_slides=1,
            sampler=RandomSampler(patch_size=256, num_patches=-1),
            transforms=_RandomOffsetTransform(),
            seed=11,
        )

        def _outputs(pipeline):
            return [
                (result.coordinate.x, result.coordinate.y, int(result.image[0, 0, 0]))
                for result in pipeline
            ]

        assert pipeline1.sampler is not sampler
        assert pipeline2.sampler is not sampler
        assert pipeline1.sampler is not pipeline2.sampler
        assert pipeline1.transforms is not transform
        assert pipeline2.transforms is not transform
        assert pipeline1.transforms is not pipeline2.transforms
        assert pickle.dumps(sampler._rng.bit_generator.state) == sampler_state
        assert pickle.dumps(transform._rng.bit_generator.state) == transform_state
        assert _outputs(pipeline1) == _outputs(reference)

    def test_views_and_crops_are_pipeline_owned(self):
        views = [
            ViewConfig(
                name="view",
                crop=RandomResizedCrop(size=64, scale=(0.2, 1.0)),
                transforms=_RandomOffsetTransform(),
            )
        ]
        shared_transform = _RandomOffsetTransform()
        pipeline1 = _make_pipeline(
            n_slides=1,
            views=views,
            shared_transforms=shared_transform,
            seed=11,
        )
        pipeline2 = _make_pipeline(
            n_slides=1,
            views=views,
            shared_transforms=shared_transform,
            seed=22,
        )

        assert pipeline1.views is not views
        assert pipeline2.views is not views
        assert pipeline1.views is not None
        assert pipeline2.views is not None
        assert pipeline1.views[0] is not pipeline2.views[0]
        assert pipeline1.views[0].crop is not views[0].crop
        assert pipeline2.views[0].crop is not views[0].crop
        assert pipeline1.views[0].transforms is not views[0].transforms
        assert pipeline2.views[0].transforms is not views[0].transforms
        assert pipeline1.shared_transforms is not shared_transform
        assert pipeline2.shared_transforms is not shared_transform
        assert pipeline1.shared_transforms is not pipeline2.shared_transforms

    def test_non_copyable_component_raises(self):
        with pytest.raises(TypeError, match="transforms.*must support copy.deepcopy"):
            _make_pipeline(transforms=_NonCopyableTransform())

    def test_random_slide_sampling_is_seeded(self):
        # Use 20 slides so that a random permutation equalling alphabetical
        # order is astronomically unlikely (probability 1/20! ≈ 2e-19).
        slide_paths = fake_slide_paths(20)

        pipeline1 = _make_pipeline(
            n_slides=20,
            slide_paths=slide_paths,
            patches_per_slide=1,
            pool_size=20,
            slide_sampling="random",
            seed=7,
        )
        pipeline2 = _make_pipeline(
            n_slides=20,
            slide_paths=slide_paths,
            patches_per_slide=1,
            pool_size=20,
            slide_sampling="random",
            seed=7,
        )

        order1 = [result.coordinate.slide_path for result in pipeline1]
        order2 = [result.coordinate.slide_path for result in pipeline2]

        assert order1 == order2
        assert order1 != slide_paths

    def test_random_slide_sampling_changes_with_seed(self):
        pipeline1 = _make_pipeline(
            n_slides=6,
            patches_per_slide=1,
            pool_size=6,
            slide_sampling="random",
            seed=7,
        )
        pipeline2 = _make_pipeline(
            n_slides=6,
            patches_per_slide=1,
            pool_size=6,
            slide_sampling="random",
            seed=99,
        )

        order1 = [result.coordinate.slide_path for result in pipeline1]
        order2 = [result.coordinate.slide_path for result in pipeline2]

        assert order1 != order2


class TestCycleMode:
    def test_cycle_produces_more_than_one_pass(self):
        n_slides, pps = 2, 5
        one_pass = n_slides * pps
        target = one_pass * 3

        pipeline = _make_pipeline(
            n_slides=n_slides,
            patches_per_slide=pps,
            cycle=True,
        )

        count = 0
        for _ in pipeline:
            count += 1
            if count >= target:
                break

        assert count >= target

    def test_cycle_revisits_slides(self):
        n_slides, pps = 2, 3
        pipeline = _make_pipeline(
            n_slides=n_slides,
            patches_per_slide=pps,
            cycle=True,
        )

        per_slide = Counter()
        count = 0
        for result in pipeline:
            per_slide[result.coordinate.slide_path] += 1
            count += 1
            if count >= n_slides * pps * 3:
                break

        # Each slide should have been visited more than one pass
        for slide, total in per_slide.items():
            assert total > pps, f"{slide} only got {total} patches (one pass = {pps})"

    def test_cycle_no_duplicate_pool_entries(self):
        """With pool_size > n_slides, should NOT open the same slide twice."""
        pipeline = _make_pipeline(
            n_slides=2,
            patches_per_slide=5,
            pool_size=8,
            cycle=True,
        )

        count = 0
        for _ in pipeline:
            count += 1
            if count >= 30:
                break
        assert count == 30

        # Verify: slides_processed tracks how many times slides were opened.
        # Over 30 patches with pps=5, we need 6 opens (3 passes × 2 slides).
        # Without duplicate guard, pool_size=8 would open 8 copies immediately.
        assert pipeline.stats.slides_processed <= count // 5 + 2

    def test_cycle_false_stops_after_one_pass(self):
        pipeline = _make_pipeline(n_slides=3, patches_per_slide=5, cycle=False)
        patches = list(pipeline)
        assert len(patches) == 15  # exactly one pass


class TestTissueMaskCache:
    def test_negative_size_raises(self):
        with pytest.raises(ValueError, match="tissue_mask_cache_size must be >= 0"):
            _make_pipeline(tissue_mask_cache_size=-1)

    def test_disabled_cache_recomputes_revisited_slide(self):
        detector = _CountingDetector()
        pipeline = _make_pipeline(
            n_slides=1,
            patches_per_slide=1,
            pool_size=1,
            cycle=True,
            tissue_detector=detector,
        )

        _take(pipeline, 3)

        assert detector.call_count == 3

    def test_enabled_cache_reuses_revisited_slide(self):
        _CountingThumbnailBackend.call_count = 0
        detector = _CountingDetector()
        pipeline = _make_pipeline(
            n_slides=1,
            backend=_CountingThumbnailBackend(),
            patches_per_slide=1,
            pool_size=1,
            cycle=True,
            tissue_detector=detector,
            tissue_mask_cache_size=1,
        )

        _take(pipeline, 3)

        assert detector.call_count == 1
        assert _CountingThumbnailBackend.call_count == 1

    def test_lru_cache_evicts_oldest_slide(self):
        detector = _CountingDetector()
        pipeline = _make_pipeline(
            n_slides=3,
            patches_per_slide=1,
            pool_size=1,
            cycle=True,
            slide_sampling="sequential",
            tissue_detector=detector,
            tissue_mask_cache_size=2,
        )

        _take(pipeline, 4)

        assert detector.call_count == 4
        assert len(pipeline._tissue_mask_cache) == 2

    def test_clear_forces_redetection(self):
        detector = _CountingDetector()
        pipeline = _make_pipeline(
            n_slides=1,
            patches_per_slide=1,
            pool_size=1,
            cycle=True,
            tissue_detector=detector,
            tissue_mask_cache_size=1,
        )
        _take(pipeline, 2)

        pipeline.clear_tissue_mask_cache()
        _take(pipeline, 1)

        assert detector.call_count == 2

    def test_thumbnail_size_change_invalidates_cache(self):
        detector = _CountingDetector()
        pipeline = _make_pipeline(
            n_slides=1,
            patches_per_slide=1,
            pool_size=1,
            cycle=True,
            tissue_detector=detector,
            tissue_mask_cache_size=1,
        )
        _take(pipeline, 1)

        pipeline.thumbnail_size = (1024, 1024)
        _take(pipeline, 1)

        assert detector.call_count == 2

    def test_component_replacement_invalidates_cache(self):
        detector = _CountingDetector()
        pipeline = _make_pipeline(
            n_slides=1,
            patches_per_slide=1,
            pool_size=1,
            tissue_detector=detector,
            tissue_mask_cache_size=1,
        )
        list(pipeline)

        pipeline.backend = FakeBackend(token="replacement")
        list(pipeline)
        assert detector.call_count == 2

        replacement_detector = _CountingDetector()
        pipeline.tissue_detector = replacement_detector
        list(pipeline)
        assert replacement_detector.call_count == 1

    def test_cached_mask_is_isolated_from_sampler_mutation(self):
        detector = _CountingDetector()
        sampler = _MaskMutatingSampler()
        pipeline = _make_pipeline(
            n_slides=1,
            patches_per_slide=1,
            pool_size=1,
            cycle=True,
            tissue_detector=detector,
            sampler=sampler,
            tissue_mask_cache_size=1,
        )

        _take(pipeline, 2)

        assert detector.call_count == 1
        assert pipeline.sampler.initial_fractions == [1.0, 1.0]

    def test_negative_size_after_construction_raises(self):
        pipeline = _make_pipeline(tissue_mask_cache_size=1)
        pipeline.tissue_mask_cache_size = -1

        with pytest.raises(ValueError, match="tissue_mask_cache_size must be >= 0"):
            list(pipeline)

    def test_failed_detection_is_not_cached(self):
        detector = _CountingDetector(failures=1)
        pipeline = _make_pipeline(
            n_slides=1,
            patches_per_slide=1,
            pool_size=1,
            tissue_detector=detector,
            tissue_mask_cache_size=1,
        )

        assert list(pipeline) == []
        assert len(list(pipeline)) == 1
        assert detector.call_count == 2

    def test_cache_entries_are_not_pickled(self):
        detector = _CountingDetector()
        pipeline = _make_pipeline(
            n_slides=1,
            patches_per_slide=1,
            pool_size=1,
            tissue_detector=detector,
            tissue_mask_cache_size=1,
        )
        list(pipeline)

        restored = pickle.loads(pickle.dumps(pipeline))
        list(restored)

        assert restored.tissue_detector.call_count == 2

    def test_cache_entries_are_not_inherited_after_fork(self):
        if "fork" not in mp.get_all_start_methods():
            pytest.skip("fork start method unavailable")

        detector = _CountingDetector()
        pipeline = _make_pipeline(
            n_slides=1,
            patches_per_slide=1,
            pool_size=1,
            tissue_detector=detector,
            tissue_mask_cache_size=1,
        )
        list(pipeline)

        context = mp.get_context("fork")
        receive, send = context.Pipe(duplex=False)
        process = context.Process(target=_pipeline_cache_fork_probe, args=(send, pipeline))
        process.start()
        send.close()
        try:
            assert receive.poll(20), "forked cache probe did not return"
            child_detector_calls, child_cache_size = receive.recv()
        finally:
            receive.close()
            process.join(20)
            if process.is_alive():
                process.terminate()
                process.join()

        assert process.exitcode == 0
        assert child_detector_calls == 2
        assert child_cache_size == 1


class TestPatchFilter:
    def test_filter_rejects(self):
        pipeline = _make_pipeline(
            n_slides=1,
            patches_per_slide=10,
            patch_filter=_RejectAll(),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
        )
        patches = list(pipeline)
        assert len(patches) == 0

    def test_filter_stats(self):
        pipeline = _make_pipeline(
            n_slides=1,
            patches_per_slide=20,
            patch_filter=_RejectAll(),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
        )
        list(pipeline)
        assert pipeline.stats.patches_filtered > 0
        assert pipeline.stats.patches_extracted == 0

    def test_reject_all_with_infinite_sampler_terminates(self):
        """Regression: all-reject filter + infinite sampler must NOT hang."""
        pipeline = _make_pipeline(
            n_slides=2,
            patches_per_slide=10,
            cycle=False,
            patch_filter=_RejectAll(),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
        )
        patches = list(pipeline)  # must terminate
        assert len(patches) == 0
        # All reads counted as filtered
        assert pipeline.stats.patches_filtered == 20  # 2 slides × 10 attempts


class TestTransformIntegration:
    def test_transforms_applied(self):
        counter = _CountTransform()
        pipeline = _make_pipeline(
            n_slides=1,
            patches_per_slide=5,
            transforms=counter,
        )
        list(pipeline)
        assert pipeline.transforms.call_count == 5

    def test_normalize_changes_dtype(self):
        norm = NormalizeTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        pipeline = _make_pipeline(
            n_slides=1,
            patches_per_slide=3,
            transforms=norm,
        )
        for result in pipeline:
            assert result.image.dtype == np.float32

    def test_resize_changes_shape(self):
        pipeline = _make_pipeline(
            n_slides=1,
            patches_per_slide=3,
            transforms=ResizeTransform(target_size=224),
        )
        for result in pipeline:
            assert result.image.shape == (224, 224, 3)


class TestCleanup:
    def test_early_break_closes_slides(self):
        """Breaking out of iteration must close all open slides."""
        backend = FakeBackend()
        pipeline = PatchPipeline(
            slide_paths=fake_slide_paths(5),
            backend=backend,
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            pool_size=3,
            patches_per_slide=100,
            cycle=True,
        )

        count = 0
        for _ in pipeline:
            count += 1
            if count >= 5:
                break

        # The generator's finally block should have closed everything.
        # We can't easily inspect the pool, but we verify no exception
        # was raised and stats are consistent.
        assert count == 5
        assert pipeline.stats.patches_extracted == 5

    def test_full_consumption_closes_slides(self):
        pipeline = _make_pipeline(n_slides=2, patches_per_slide=3)
        list(pipeline)
        assert pipeline.stats.slides_processed == 2


class TestBackendCloning:
    def test_deepcopy_preserves_config(self):
        """Backend constructor args must survive the prototype cloning."""
        backend = FakeBackend(token="secret")
        pipeline = PatchPipeline(
            slide_paths=fake_slide_paths(1),
            backend=backend,
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=256, num_patches=1, seed=42),
            pool_size=1,
            patches_per_slide=1,
        )

        # Iterate to trigger _open_slide (which clones the backend)
        results = list(pipeline)
        assert len(results) == 1

        # The original backend should NOT have been opened
        assert not backend._opened

        # Verify via deepcopy directly that token survives
        cloned = copy.deepcopy(backend)
        assert cloned.token == "secret"


class TestErrorHandling:
    def test_all_slides_fail_gracefully(self):
        """If every slide fails to open, pipeline yields nothing."""

        class _FailBackend(FakeBackend):
            def open(self, path: str) -> None:
                raise RuntimeError("disk on fire")

        pipeline = PatchPipeline(
            slide_paths=fake_slide_paths(3),
            backend=_FailBackend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=256, num_patches=5, seed=42),
            pool_size=2,
            patches_per_slide=5,
        )
        patches = list(pipeline)
        assert len(patches) == 0
        assert pipeline.stats.slides_failed == 3

    def test_all_slides_fail_cycle_no_infinite_loop(self):
        """cycle=True + all slides broken must NOT loop forever."""

        class _FailBackend(FakeBackend):
            def open(self, path: str) -> None:
                raise RuntimeError("broken")

        pipeline = PatchPipeline(
            slide_paths=fake_slide_paths(3),
            backend=_FailBackend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            pool_size=2,
            patches_per_slide=5,
            cycle=True,
        )
        # This must terminate (not RecursionError or infinite loop)
        patches = list(pipeline)
        assert len(patches) == 0

    def test_partial_slide_failure(self):
        """Some slides fail, others succeed — pipeline keeps going."""

        class _FailOnSecond(FakeBackend):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._call_count = 0

            def open(self, path: str) -> None:
                self._call_count += 1
                if "slide_1" in path:
                    raise RuntimeError("corrupt")
                super().open(path)

        pipeline = PatchPipeline(
            slide_paths=fake_slide_paths(3),
            backend=_FailOnSecond(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            pool_size=3,
            patches_per_slide=5,
        )
        patches = list(pipeline)
        # slide_1 fails, slide_0 and slide_2 succeed → 10 patches
        assert len(patches) == 10
        assert pipeline.stats.slides_failed == 1
        assert pipeline.stats.slides_processed == 2

    def test_post_open_failure_closes_slide(self):
        """If tissue detection fails after slide is opened, the slide is closed."""
        from wsistream.tissue.base import TissueDetector

        class _FailDetector(TissueDetector):
            def detect(self, thumbnail, downsample=(1.0, 1.0)):
                raise RuntimeError("detector crashed")

        closed_slides = []
        original_close = FakeBackend.close

        class _TrackClose(FakeBackend):
            def close(self):
                closed_slides.append(self._path)
                original_close(self)

        pipeline = PatchPipeline(
            slide_paths=fake_slide_paths(2),
            backend=_TrackClose(),
            tissue_detector=_FailDetector(),
            sampler=RandomSampler(patch_size=256, num_patches=5, seed=42),
            pool_size=2,
            patches_per_slide=5,
        )
        patches = list(pipeline)
        assert len(patches) == 0
        assert pipeline.stats.slides_failed == 2
        # slides_processed should be 0 — failure happened before setup completed
        assert pipeline.stats.slides_processed == 0
        # Both slides should have been closed despite the failure
        assert len(closed_slides) == 2


class TestCycleRngDiversity:
    def test_revisited_slide_gets_different_patches(self):
        """Regression: cycling must NOT replay identical coordinates."""
        pipeline = _make_pipeline(
            n_slides=1,
            patches_per_slide=5,
            cycle=True,
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
        )

        coords_by_pass: list[list[tuple[int, int]]] = [[], []]
        current_pass = 0
        count = 0
        for result in pipeline:
            coords_by_pass[current_pass].append((result.coordinate.x, result.coordinate.y))
            count += 1
            if count == 5:
                current_pass = 1
            if count == 10:
                break

        assert (
            coords_by_pass[0] != coords_by_pass[1]
        ), "Second pass produced identical coordinates — RNG is being re-seeded"


class TestWorkerRngIsolation:
    def test_multi_worker_dataloader_uses_independent_rng_streams(self):
        dataset = _WorkerProbeDataset(fake_slide_paths(4))
        loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=2,
            multiprocessing_context="spawn",
        )

        per_worker: dict[int, list[tuple[int, int, int]]] = {}
        for item in loader:
            worker_id, x, y, checksum = (int(value) for value in item)
            per_worker.setdefault(worker_id, []).append((x, y, checksum))
            if all(len(values) >= 2 for values in per_worker.values()) and len(per_worker) == 2:
                break

        assert set(per_worker) == {0, 1}
        assert len(per_worker[0]) >= 2
        assert len(per_worker[1]) >= 2
        assert [(x, y) for x, y, _ in per_worker[0][:2]] != [
            (x, y) for x, y, _ in per_worker[1][:2]
        ]
        assert [checksum for _, _, checksum in per_worker[0][:2]] != [
            checksum for _, _, checksum in per_worker[1][:2]
        ]
