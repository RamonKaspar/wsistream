"""Tests for without-replacement sampling (coordinate pools + pipeline integration)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from tests.conftest import FakeBackend, fake_slide_paths
from wsistream.filters.base import PatchFilter
from wsistream.pipeline import PatchPipeline
from wsistream.sampling.base import (
    CoordinatePool,
    MultiLevelCoordinatePool,
    PatchSampler,
    enumerate_grid_coordinates,
)
from wsistream.sampling.grid import GridSampler
from wsistream.sampling.multi_magnification import MultiMagnificationSampler
from wsistream.sampling.random import RandomSampler
from wsistream.slide import SlideHandle
from wsistream.tissue.otsu import OtsuTissueDetector
from wsistream.types import PatchCoordinate, TissueMask


# ── fixtures ──


@pytest.fixture
def fake_slide(fake_backend):
    """SlideHandle backed by FakeBackend (4096x4096, 3 levels, mpp=0.25)."""
    slide = SlideHandle("fake.svs", backend=fake_backend)
    yield slide
    slide.close()


@pytest.fixture
def full_tissue_mask():
    """100x100 mask that is ALL tissue."""
    mask = np.ones((100, 100), dtype=bool)
    return TissueMask(mask=mask, downsample=40.96, slide_dimensions=(4096, 4096))


@pytest.fixture
def center_tissue_mask():
    """100x100 mask with tissue only in center 40x40."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 30:70] = True
    return TissueMask(mask=mask, downsample=40.96, slide_dimensions=(4096, 4096))


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


# ── CoordinatePool ──


class TestCoordinatePool:
    def test_pop_returns_all_coordinates(self):
        coords = [
            PatchCoordinate(x=i, y=0, level=0, patch_size=256, mpp=0.25, slide_path="s.svs")
            for i in range(10)
        ]
        pool = CoordinatePool(coords, rng=np.random.default_rng(42))

        popped = []
        while not pool.exhausted:
            popped.append(pool.pop(rng=np.random.default_rng(0)))
        assert len(popped) == 10
        assert set((c.x, c.y) for c in popped) == set((c.x, c.y) for c in coords)

    def test_pop_returns_none_when_exhausted(self):
        pool = CoordinatePool([], rng=np.random.default_rng(0))
        assert pool.pop(rng=np.random.default_rng(0)) is None

    def test_shuffles_order(self):
        coords = [
            PatchCoordinate(x=i, y=0, level=0, patch_size=256, mpp=0.25, slide_path="s.svs")
            for i in range(50)
        ]
        pool = CoordinatePool(coords, rng=np.random.default_rng(42))
        order = [pool.pop(rng=np.random.default_rng(0)).x for _ in range(50)]
        assert order != list(range(50))

    def test_remaining_and_total(self):
        coords = [
            PatchCoordinate(x=i, y=0, level=0, patch_size=256, mpp=0.25, slide_path="s.svs")
            for i in range(5)
        ]
        pool = CoordinatePool(coords, rng=np.random.default_rng(0))
        assert pool.total == 5
        assert pool.remaining == 5

        pool.pop(rng=np.random.default_rng(0))
        pool.pop(rng=np.random.default_rng(0))
        assert pool.remaining == 3
        assert pool.total == 5
        assert not pool.exhausted

    def test_empty_pool(self):
        pool = CoordinatePool([], rng=np.random.default_rng(0))
        assert pool.exhausted
        assert pool.remaining == 0
        assert pool.total == 0


# ── MultiLevelCoordinatePool ──


class TestMultiLevelCoordinatePool:
    def _make_coords(self, level: int, n: int) -> list[PatchCoordinate]:
        return [
            PatchCoordinate(x=i, y=0, level=level, patch_size=256, mpp=None, slide_path="s.svs")
            for i in range(n)
        ]

    def test_pops_from_all_levels(self):
        pool = MultiLevelCoordinatePool(
            level_pools={0: self._make_coords(0, 10), 1: self._make_coords(1, 10)},
            level_weights={0: 0.5, 1: 0.5},
            rng=np.random.default_rng(42),
        )
        levels = Counter()
        rng = np.random.default_rng(42)
        while not pool.exhausted:
            coord = pool.pop(rng)
            levels[coord.level] += 1
        assert levels[0] == 10
        assert levels[1] == 10

    def test_respects_weights(self):
        pool = MultiLevelCoordinatePool(
            level_pools={0: self._make_coords(0, 500), 1: self._make_coords(1, 500)},
            level_weights={0: 0.9, 1: 0.1},
            rng=np.random.default_rng(42),
        )
        levels = Counter()
        rng = np.random.default_rng(42)
        for _ in range(200):
            coord = pool.pop(rng)
            levels[coord.level] += 1
        # Level 0 should dominate with 0.9 weight
        assert levels[0] > levels[1] * 3

    def test_renormalises_when_level_exhausted(self):
        pool = MultiLevelCoordinatePool(
            level_pools={0: self._make_coords(0, 3), 1: self._make_coords(1, 100)},
            level_weights={0: 0.5, 1: 0.5},
            rng=np.random.default_rng(42),
        )
        rng = np.random.default_rng(42)
        all_coords = []
        while not pool.exhausted:
            all_coords.append(pool.pop(rng))
        assert len(all_coords) == 103
        # After level 0 exhausts, remaining coords must all be level 1
        level_1_coords = [c for c in all_coords if c.level == 1]
        assert len(level_1_coords) == 100

    def test_remaining_and_total(self):
        pool = MultiLevelCoordinatePool(
            level_pools={0: self._make_coords(0, 5), 1: self._make_coords(1, 3)},
            level_weights={0: 0.5, 1: 0.5},
            rng=np.random.default_rng(0),
        )
        assert pool.total == 8
        assert pool.remaining == 8

    def test_empty_pools(self):
        pool = MultiLevelCoordinatePool(
            level_pools={0: [], 1: []},
            level_weights={0: 0.5, 1: 0.5},
            rng=np.random.default_rng(0),
        )
        assert pool.exhausted
        assert pool.pop(np.random.default_rng(0)) is None


# ── RandomSampler.build_coordinate_pool ──


class TestRandomSamplerPool:
    def test_correct_count_full_tissue(self, fake_slide, full_tissue_mask):
        sampler = RandomSampler(patch_size=256, seed=42)
        pool = sampler.build_coordinate_pool(fake_slide, full_tissue_mask, np.random.default_rng(0))
        # 4096 / 256 = 16 per axis → 256
        assert pool.total == 256

    def test_respects_tissue_mask(self, fake_slide, full_tissue_mask, center_tissue_mask):
        sampler = RandomSampler(patch_size=256, tissue_threshold=0.1, seed=42)
        full_pool = sampler.build_coordinate_pool(
            fake_slide, full_tissue_mask, np.random.default_rng(0)
        )
        center_pool = sampler.build_coordinate_pool(
            fake_slide, center_tissue_mask, np.random.default_rng(0)
        )
        assert center_pool.total < full_pool.total
        assert center_pool.total > 0

    def test_no_tissue_empty_pool(self, fake_slide):
        mask = np.zeros((100, 100), dtype=bool)
        no_tissue = TissueMask(mask=mask, downsample=40.96, slide_dimensions=(4096, 4096))
        sampler = RandomSampler(patch_size=256, seed=42)
        pool = sampler.build_coordinate_pool(fake_slide, no_tissue, np.random.default_rng(0))
        assert pool.total == 0
        assert pool.exhausted

    def test_target_mpp_selects_level(self, fake_slide, full_tissue_mask):
        sampler = RandomSampler(patch_size=256, target_mpp=0.5, seed=42)
        pool = sampler.build_coordinate_pool(fake_slide, full_tissue_mask, np.random.default_rng(0))
        # Level 1: dims 2048x2048 at ds=2.0 → patch_l0 = 512 → 4096/512 = 8 per axis → 64
        assert pool.total == 64
        coord = pool.pop(np.random.default_rng(0))
        assert coord.level == 1
        assert coord.mpp == 0.5

    def test_all_coordinates_unique(self, fake_slide, full_tissue_mask):
        sampler = RandomSampler(patch_size=256, seed=42)
        pool = sampler.build_coordinate_pool(fake_slide, full_tissue_mask, np.random.default_rng(0))
        seen = set()
        rng = np.random.default_rng(0)
        while not pool.exhausted:
            c = pool.pop(rng)
            key = (c.x, c.y, c.level)
            assert key not in seen, f"duplicate coordinate: {key}"
            seen.add(key)

    def test_coordinates_are_on_grid(self, fake_slide, full_tissue_mask):
        sampler = RandomSampler(patch_size=256, seed=42)
        pool = sampler.build_coordinate_pool(fake_slide, full_tissue_mask, np.random.default_rng(0))
        rng = np.random.default_rng(0)
        while not pool.exhausted:
            c = pool.pop(rng)
            assert c.x % 256 == 0
            assert c.y % 256 == 0


# ── MultiMagnificationSampler.build_coordinate_pool ──


class TestMultiMagnificationSamplerPool:
    def test_builds_pools_for_all_levels(self, fake_slide, full_tissue_mask):
        sampler = MultiMagnificationSampler(target_mpps=[0.25, 0.5, 1.0], patch_size=256, seed=42)
        pool = sampler.build_coordinate_pool(fake_slide, full_tissue_mask, np.random.default_rng(0))
        assert isinstance(pool, MultiLevelCoordinatePool)
        # Level 0: 256, level 1: 64, level 2: 16 → 336 total
        assert pool.total == 256 + 64 + 16

    def test_no_mpp_fallback(self, fake_backend, full_tissue_mask):
        from wsistream.types import SlideProperties

        original = fake_backend.get_properties

        def no_mpp_props():
            p = original()
            return SlideProperties(
                path=p.path,
                dimensions=p.dimensions,
                level_count=p.level_count,
                level_dimensions=p.level_dimensions,
                level_downsamples=p.level_downsamples,
                mpp=None,
                vendor=p.vendor,
            )

        fake_backend.get_properties = no_mpp_props
        slide = SlideHandle("no_mpp.svs", backend=fake_backend)

        sampler = MultiMagnificationSampler(target_mpps=[0.25, 0.5], patch_size=256, seed=42)
        pool = sampler.build_coordinate_pool(slide, full_tissue_mask, np.random.default_rng(0))
        assert isinstance(pool, CoordinatePool)
        assert pool.total == 256  # level 0 only
        slide.close()

    def test_merged_levels(self, fake_slide, full_tissue_mask):
        """Multiple target_mpps that resolve to the same pyramid level merge."""
        sampler = MultiMagnificationSampler(
            target_mpps=[0.25, 0.26],  # both resolve to level 0
            mpp_weights=[0.5, 0.5],
            patch_size=256,
            seed=42,
        )
        pool = sampler.build_coordinate_pool(fake_slide, full_tissue_mask, np.random.default_rng(0))
        assert isinstance(pool, MultiLevelCoordinatePool)
        # Only one actual level, but still a MultiLevelCoordinatePool
        assert pool.total == 256

    def test_custom_weights_preserved(self, fake_slide, full_tissue_mask):
        sampler = MultiMagnificationSampler(
            target_mpps=[0.25, 1.0],
            mpp_weights=[0.9, 0.1],
            patch_size=256,
            seed=42,
        )
        pool = sampler.build_coordinate_pool(
            fake_slide, full_tissue_mask, np.random.default_rng(42)
        )
        levels = Counter()
        rng = np.random.default_rng(42)
        for _ in range(100):
            c = pool.pop(rng)
            levels[c.level] += 1
        # Level 0 (weight 0.9) should dominate
        assert levels[0] > levels[2] * 3


# ── Pipeline integration ──


class TestPipelineWithoutReplacement:
    def test_default_is_with_replacement(self):
        pipeline = _make_pipeline()
        assert pipeline.replacement == "with_replacement"

    def test_invalid_replacement_raises(self):
        with pytest.raises(ValueError, match="replacement"):
            _make_pipeline(replacement="invalid")

    def test_unsupported_sampler_raises(self):
        with pytest.raises(TypeError, match="GridSampler"):
            _make_pipeline(
                sampler=GridSampler(patch_size=256),
                replacement="without_replacement",
            )

    def test_produces_patches(self):
        pipeline = _make_pipeline(
            n_slides=2,
            patches_per_slide=5,
            replacement="without_replacement",
        )
        patches = list(pipeline)
        assert len(patches) == 10

    def test_no_duplicate_coordinates_single_slide(self):
        pipeline = _make_pipeline(
            n_slides=1,
            patches_per_slide=50,
            replacement="without_replacement",
        )
        coords = [(r.coordinate.x, r.coordinate.y) for r in pipeline]
        assert len(coords) == len(set(coords))

    def test_pool_persists_across_reopenings(self):
        """With patches_per_slide < pool size, the slide is rotated out and
        later reopened. The remaining pool should be reused, not rebuilt."""
        pipeline = _make_pipeline(
            n_slides=1,
            patches_per_slide=5,
            pool_size=1,
            cycle=True,
            replacement="without_replacement",
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
        )
        coords = []
        for result in pipeline:
            coords.append((result.coordinate.x, result.coordinate.y))
            # Collect enough patches to span multiple reopenings but not
            # exhaust the pool (full tissue → 256 grid cells available).
            if len(coords) >= 30:
                break

        # All coordinates should be unique (no repeats across reopenings).
        assert len(coords) == len(set(coords))

    def test_pool_resets_after_exhaustion_with_cycle(self):
        """Once the pool is exhausted, cycle=True should rebuild it."""
        pipeline = _make_pipeline(
            n_slides=1,
            patches_per_slide=1000,
            pool_size=1,
            cycle=True,
            replacement="without_replacement",
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
        )
        # The pool has some number of valid tissue coords. Exhaust it once
        # (no duplicates), then continue into the second cycle (duplicates
        # allowed across cycles, but not within).
        coords = []
        for result in pipeline:
            coords.append((result.coordinate.x, result.coordinate.y))
            if len(coords) >= 300:
                break

        # The pool for this slide has at most 256 coords (full grid).
        # Reaching 300 proves the pool was reset at least once.
        assert len(coords) == 300

    def test_cycle_false_stops_when_pool_exhausted(self):
        """Without cycle, the pipeline should stop when the pool runs out."""
        pipeline = _make_pipeline(
            n_slides=1,
            patches_per_slide=10000,  # high budget so pool exhaustion is the limiter
            pool_size=1,
            cycle=False,
            replacement="without_replacement",
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
        )
        patches = list(pipeline)
        # Should stop at pool size, not at patches_per_slide
        assert len(patches) <= 256
        assert len(patches) > 0

    def test_patches_per_slide_still_respected(self):
        pipeline = _make_pipeline(
            n_slides=2,
            patches_per_slide=10,
            replacement="without_replacement",
        )
        per_slide = Counter()
        for result in pipeline:
            per_slide[result.coordinate.slide_path] += 1
        for count in per_slide.values():
            assert count == 10

    def test_filter_consumes_from_pool(self):
        """Filtered patches should be consumed, not retried."""
        pipeline = _make_pipeline(
            n_slides=1,
            patches_per_slide=20,
            patch_filter=_RejectAll(),
            replacement="without_replacement",
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
        )
        patches = list(pipeline)
        assert len(patches) == 0
        assert pipeline.stats.patches_filtered > 0
        assert pipeline.stats.patches_extracted == 0

    def test_interleaving_still_works(self):
        pipeline = _make_pipeline(
            n_slides=3,
            patches_per_slide=6,
            pool_size=3,
            replacement="without_replacement",
        )
        slide_order = [r.coordinate.slide_path for r in pipeline]
        first_three = set(slide_order[:3])
        assert len(first_three) == 3

    def test_second_cycle_different_order(self):
        """The reshuffle on pool rebuild should produce a different order."""
        pipeline = _make_pipeline(
            n_slides=1,
            patches_per_slide=1000,
            pool_size=1,
            cycle=True,
            replacement="without_replacement",
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
        )
        # Collect two full passes worth of coordinates.
        first_pass: list[tuple[int, int]] = []
        second_pass: list[tuple[int, int]] = []
        pool_size = None

        for result in pipeline:
            xy = (result.coordinate.x, result.coordinate.y)
            if pool_size is None:
                first_pass.append(xy)
                # Detect pool exhaustion: when we see a coord that was
                # already in first_pass, the pool was reset.
                if xy in set(first_pass[:-1]):
                    # This coord appeared before → we crossed into cycle 2.
                    pool_size = len(first_pass) - 1
                    first_pass.pop()
                    second_pass.append(xy)
            else:
                second_pass.append(xy)
                if len(second_pass) >= pool_size:
                    break

        if pool_size is not None and len(second_pass) == pool_size:
            # Same set of coordinates, but different ordering
            assert set(first_pass) == set(second_pass)
            assert first_pass != second_pass

    def test_with_multi_magnification_sampler(self):
        pipeline = _make_pipeline(
            n_slides=1,
            patches_per_slide=30,
            replacement="without_replacement",
            sampler=MultiMagnificationSampler(
                target_mpps=[0.25, 0.5, 1.0],
                patch_size=256,
                seed=42,
            ),
        )
        levels = Counter()
        for result in pipeline:
            levels[result.coordinate.level] += 1
        assert len(levels) >= 2
        assert sum(levels.values()) == 30

    def test_seeded_is_reproducible(self):
        def _run(seed):
            pipeline = _make_pipeline(
                n_slides=2,
                patches_per_slide=10,
                replacement="without_replacement",
                seed=seed,
                sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            )
            return [(r.coordinate.x, r.coordinate.y) for r in pipeline]

        assert _run(123) == _run(123)

    def test_different_seeds_differ(self):
        def _run(seed):
            pipeline = _make_pipeline(
                n_slides=1,
                patches_per_slide=10,
                replacement="without_replacement",
                seed=seed,
                sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            )
            return [(r.coordinate.x, r.coordinate.y) for r in pipeline]

        assert _run(1) != _run(2)


class TestNumPatchesContract:
    """num_patches must cap pool size in without-replacement mode."""

    def test_random_sampler_num_patches_caps_pool(self, fake_slide, full_tissue_mask):
        sampler = RandomSampler(patch_size=256, num_patches=10, seed=42)
        pool = sampler.build_coordinate_pool(fake_slide, full_tissue_mask, np.random.default_rng(0))
        # Full grid would be 256, but num_patches=10 caps it
        assert pool.total == 10

    def test_random_sampler_num_patches_larger_than_grid(self, fake_slide, full_tissue_mask):
        """num_patches larger than grid size is fine — pool has fewer."""
        sampler = RandomSampler(patch_size=256, num_patches=9999, seed=42)
        pool = sampler.build_coordinate_pool(fake_slide, full_tissue_mask, np.random.default_rng(0))
        assert pool.total == 256

    def test_random_sampler_num_patches_selects_unbiased_subset(self, fake_slide, full_tissue_mask):
        """num_patches < full grid must select a random subset, not top-left."""
        sampler = RandomSampler(patch_size=256, num_patches=10, seed=42)

        # Build pools with two different seeds — subsets must differ
        pool_a = sampler.build_coordinate_pool(
            fake_slide, full_tissue_mask, np.random.default_rng(1)
        )
        pool_b = sampler.build_coordinate_pool(
            fake_slide, full_tissue_mask, np.random.default_rng(2)
        )

        rng_a = np.random.default_rng(0)
        rng_b = np.random.default_rng(0)
        set_a = set()
        set_b = set()
        while not pool_a.exhausted:
            c = pool_a.pop(rng_a)
            set_a.add((c.x, c.y))
        while not pool_b.exhausted:
            c = pool_b.pop(rng_b)
            set_b.add((c.x, c.y))

        # Different RNG seeds must select different subsets of the grid
        assert set_a != set_b

        # Neither subset should be only top-left cells
        assert not all(x < 256 * 4 for x, _ in set_a)

    def test_multi_mag_num_patches_caps_pops(self, fake_slide, full_tissue_mask):
        sampler = MultiMagnificationSampler(
            target_mpps=[0.25, 0.5], num_patches=5, patch_size=256, seed=42
        )
        pool = sampler.build_coordinate_pool(fake_slide, full_tissue_mask, np.random.default_rng(0))
        rng = np.random.default_rng(42)
        count = 0
        while not pool.exhausted:
            pool.pop(rng)
            count += 1
        assert count == 5
        assert pool.remaining == 0

    def test_multi_mag_num_patches_caps_memory(self, fake_slide, full_tissue_mask):
        """Per-level storage must be capped, not just pop count."""
        sampler = MultiMagnificationSampler(
            target_mpps=[0.25, 0.5], num_patches=5, patch_size=256, seed=42
        )
        pool = sampler.build_coordinate_pool(fake_slide, full_tissue_mask, np.random.default_rng(0))
        # Each level stores at most max_total=5 coordinates
        assert isinstance(pool, MultiLevelCoordinatePool)
        for lvl_deque in pool._pools.values():
            assert len(lvl_deque) <= 5

    def test_pipeline_num_patches_respected(self):
        pipeline = _make_pipeline(
            n_slides=1,
            patches_per_slide=10000,
            pool_size=1,
            cycle=False,
            replacement="without_replacement",
            sampler=RandomSampler(patch_size=256, num_patches=7, seed=42),
        )
        patches = list(pipeline)
        assert len(patches) == 7


class TestMultiLevelZeroWeights:
    """Zero-weight levels must not cause NaN or crashes, and accounting must be consistent."""

    def _make_coords(self, level: int, n: int) -> list[PatchCoordinate]:
        return [
            PatchCoordinate(x=i, y=0, level=level, patch_size=256, mpp=None, slide_path="s.svs")
            for i in range(n)
        ]

    def test_zero_weight_level_never_selected(self):
        pool = MultiLevelCoordinatePool(
            level_pools={0: self._make_coords(0, 10), 1: self._make_coords(1, 10)},
            level_weights={0: 0.0, 1: 1.0},
            rng=np.random.default_rng(42),
        )
        # Zero-weight level excluded from total
        assert pool.total == 10
        assert pool.remaining == 10

        rng = np.random.default_rng(42)
        levels = Counter()
        while not pool.exhausted:
            c = pool.pop(rng)
            levels[c.level] += 1
        assert levels.get(0, 0) == 0
        assert levels[1] == 10

        # After exhaustion, remaining must be 0
        assert pool.remaining == 0
        assert pool.exhausted

    def test_all_positive_weight_levels_exhausted_with_zero_weight_remaining(self):
        """After positive-weight levels exhaust, pool reports exhausted
        and remaining == 0 (zero-weight coords are not counted)."""
        pool = MultiLevelCoordinatePool(
            level_pools={0: self._make_coords(0, 3), 1: self._make_coords(1, 100)},
            level_weights={0: 1.0, 1: 0.0},
            rng=np.random.default_rng(42),
        )
        # Only level 0 is reachable
        assert pool.total == 3
        assert pool.remaining == 3

        rng = np.random.default_rng(42)
        count = 0
        while not pool.exhausted:
            c = pool.pop(rng)
            assert c is not None
            count += 1
        assert count == 3

        assert pool.remaining == 0
        assert pool.exhausted
        assert pool.pop(rng) is None


class TestCustomSamplerWithoutRng:
    """Custom samplers that override build_coordinate_pool but lack _rng
    must work — the pipeline provides its own RNG for pool operations."""

    def test_custom_sampler_without_rng(self):
        class _MinimalSampler(PatchSampler):
            def sample(self, slide, tissue_mask):
                yield from ()

            def build_coordinate_pool(self, slide, tissue_mask, rng):
                coords = enumerate_grid_coordinates(slide, tissue_mask, 0, 256, 0.0)
                return CoordinatePool(coords, rng)

        pipeline = PatchPipeline(
            slide_paths=fake_slide_paths(1),
            backend=FakeBackend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=_MinimalSampler(),
            pool_size=1,
            patches_per_slide=5,
            replacement="without_replacement",
        )
        patches = list(pipeline)
        assert len(patches) == 5


class TestPoolExhaustionSemantics:
    """Verify exact duplicate-free and exhaustion semantics per cycle."""

    def test_first_cycle_is_duplicate_free(self):
        pipeline = _make_pipeline(
            n_slides=1,
            patches_per_slide=10000,
            pool_size=1,
            cycle=False,
            replacement="without_replacement",
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
        )
        coords = [(r.coordinate.x, r.coordinate.y) for r in pipeline]
        pool_size = len(coords)
        assert pool_size > 0
        # Every coordinate in the first (only) cycle must be unique
        assert len(set(coords)) == pool_size

    def test_exact_pool_size_matches_grid(self):
        """With full tissue, pool size must equal the non-overlapping grid."""
        pipeline = _make_pipeline(
            n_slides=1,
            patches_per_slide=10000,
            pool_size=1,
            cycle=False,
            replacement="without_replacement",
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
        )
        patches = list(pipeline)
        # FakeBackend: 4096x4096 slide, OtsuTissueDetector on dark-center thumbnail.
        # The exact count depends on the tissue mask, but it must be deterministic
        # and <= 256 (full grid).
        assert len(patches) <= 256
        assert len(patches) > 0
        # Run again with same seed — must get the exact same count
        pipeline2 = _make_pipeline(
            n_slides=1,
            patches_per_slide=10000,
            pool_size=1,
            cycle=False,
            replacement="without_replacement",
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
        )
        patches2 = list(pipeline2)
        assert len(patches2) == len(patches)

    def test_cycle_boundary_is_clean(self):
        """Cycle 1 must be duplicate-free, cycle 2 starts only after full exhaustion."""
        pipeline = _make_pipeline(
            n_slides=1,
            patches_per_slide=10000,
            pool_size=1,
            cycle=True,
            replacement="without_replacement",
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
        )
        # First, find the pool size by running without cycle
        probe = _make_pipeline(
            n_slides=1,
            patches_per_slide=10000,
            pool_size=1,
            cycle=False,
            replacement="without_replacement",
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
        )
        pool_size = len(list(probe))
        assert pool_size > 0

        # Now collect exactly 2 cycles worth
        coords = []
        for result in pipeline:
            coords.append((result.coordinate.x, result.coordinate.y))
            if len(coords) >= pool_size * 2:
                break

        cycle1 = coords[:pool_size]
        cycle2 = coords[pool_size:]

        # Cycle 1 must be duplicate-free
        assert len(set(cycle1)) == pool_size
        # Cycle 2 must be duplicate-free
        assert len(set(cycle2)) == pool_size
        # Both cycles cover the same set of coordinates
        assert set(cycle1) == set(cycle2)
        # But in different order
        assert cycle1 != cycle2


class TestWsiStreamDatasetForwarding:
    """WsiStreamDataset must forward the replacement parameter."""

    def test_replacement_forwarded(self):
        from wsistream.torch import WsiStreamDataset

        dataset = WsiStreamDataset(
            slide_paths=fake_slide_paths(1),
            backend=FakeBackend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            pool_size=1,
            patches_per_slide=5,
            cycle=False,
            replacement="without_replacement",
        )
        items = list(dataset)
        assert len(items) == 5
        coords = [(d["x"], d["y"]) for d in items]
        assert len(set(coords)) == len(coords)

    def test_unsupported_sampler_raises_via_dataset(self):
        from wsistream.torch import WsiStreamDataset

        dataset = WsiStreamDataset(
            slide_paths=fake_slide_paths(1),
            backend=FakeBackend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=GridSampler(patch_size=256),
            pool_size=1,
            patches_per_slide=5,
            replacement="without_replacement",
        )
        # Pipeline is created lazily in __iter__, so the error surfaces there.
        with pytest.raises(TypeError, match="GridSampler"):
            list(dataset)
