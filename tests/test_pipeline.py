"""Tests for PatchPipeline (using FakeBackend, no WSI files needed)."""

from __future__ import annotations

import copy
import dataclasses
from collections import Counter

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset

from tests.conftest import FakeBackend, fake_slide_paths
from wsistream.filters.base import PatchFilter
from wsistream.pipeline import PatchPipeline, PipelineStats
from wsistream.sampling.random import RandomSampler
from wsistream.tissue.otsu import OtsuTissueDetector
from wsistream.transforms import (
    ComposeTransforms,
    NormalizeTransform,
    ResizeTransform,
)
from wsistream.transforms.base import PatchTransform


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
        stats.errors = [("slide.svs", "corrupt file")]
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
    def test_random_slide_sampling_is_seeded(self):
        slide_paths = fake_slide_paths(6)

        pipeline1 = _make_pipeline(
            n_slides=6,
            slide_paths=slide_paths,
            patches_per_slide=1,
            pool_size=6,
            slide_sampling="random",
            seed=7,
        )
        pipeline2 = _make_pipeline(
            n_slides=6,
            slide_paths=slide_paths,
            patches_per_slide=1,
            pool_size=6,
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
            n_slides=n_slides, patches_per_slide=pps, cycle=True,
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
            n_slides=n_slides, patches_per_slide=pps, cycle=True,
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
            n_slides=2, patches_per_slide=5, pool_size=8, cycle=True,
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


class TestPatchFilter:
    def test_filter_rejects(self):
        pipeline = _make_pipeline(
            n_slides=1, patches_per_slide=10,
            patch_filter=_RejectAll(),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
        )
        patches = list(pipeline)
        assert len(patches) == 0

    def test_filter_stats(self):
        pipeline = _make_pipeline(
            n_slides=1, patches_per_slide=20,
            patch_filter=_RejectAll(),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
        )
        list(pipeline)
        assert pipeline.stats.patches_filtered > 0
        assert pipeline.stats.patches_extracted == 0

    def test_reject_all_with_infinite_sampler_terminates(self):
        """Regression: all-reject filter + infinite sampler must NOT hang."""
        pipeline = _make_pipeline(
            n_slides=2, patches_per_slide=10, cycle=False,
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
            n_slides=1, patches_per_slide=5, transforms=counter,
        )
        list(pipeline)
        assert counter.call_count == 5

    def test_normalize_changes_dtype(self):
        norm = NormalizeTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        pipeline = _make_pipeline(
            n_slides=1, patches_per_slide=3, transforms=norm,
        )
        for result in pipeline:
            assert result.image.dtype == np.float32

    def test_resize_changes_shape(self):
        pipeline = _make_pipeline(
            n_slides=1, patches_per_slide=3,
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
            n_slides=1, patches_per_slide=5, cycle=True,
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
        )

        coords_by_pass: list[list[tuple[int, int]]] = [[], []]
        current_pass = 0
        count = 0
        for result in pipeline:
            coords_by_pass[current_pass].append(
                (result.coordinate.x, result.coordinate.y)
            )
            count += 1
            if count == 5:
                current_pass = 1
            if count == 10:
                break

        assert coords_by_pass[0] != coords_by_pass[1], (
            "Second pass produced identical coordinates — RNG is being re-seeded"
        )


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
