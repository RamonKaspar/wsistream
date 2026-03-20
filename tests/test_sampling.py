"""Tests for all samplers (using FakeBackend, no WSI files needed)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from wsistream.sampling.grid import GridSampler
from wsistream.sampling.multi_magnification import MultiMagnificationSampler
from wsistream.sampling.random import RandomSampler
from wsistream.slide import SlideHandle
from wsistream.types import TissueMask


@pytest.fixture
def fake_slide(fake_backend):
    """A SlideHandle backed by FakeBackend (4096x4096, 3 levels, mpp=0.25)."""
    slide = SlideHandle("fake.svs", backend=fake_backend)
    yield slide
    slide.close()


@pytest.fixture
def full_tissue_mask():
    """100x100 mask that is ALL tissue — ensures no rejection."""
    mask = np.ones((100, 100), dtype=bool)
    return TissueMask(mask=mask, downsample=40.96, slide_dimensions=(4096, 4096))


@pytest.fixture
def no_tissue_mask():
    """100x100 mask with NO tissue."""
    mask = np.zeros((100, 100), dtype=bool)
    return TissueMask(mask=mask, downsample=40.96, slide_dimensions=(4096, 4096))


@pytest.fixture
def center_tissue_mask():
    """100x100 mask with tissue only in center 40x40."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 30:70] = True
    return TissueMask(mask=mask, downsample=40.96, slide_dimensions=(4096, 4096))


# ── RandomSampler ──


class TestRandomSampler:
    def test_produces_correct_count(self, fake_slide, full_tissue_mask):
        sampler = RandomSampler(patch_size=256, num_patches=10, seed=42)
        coords = list(sampler.sample(fake_slide, full_tissue_mask))
        assert len(coords) == 10

    def test_coordinates_within_bounds(self, fake_slide, full_tissue_mask):
        sampler = RandomSampler(patch_size=256, num_patches=50, seed=42)
        for coord in sampler.sample(fake_slide, full_tissue_mask):
            assert 0 <= coord.x <= 4096 - 256
            assert 0 <= coord.y <= 4096 - 256
            assert coord.level == 0
            assert coord.patch_size == 256

    def test_respects_tissue_mask(self, fake_slide, no_tissue_mask):
        """No tissue → sampler gives up after max_retries."""
        sampler = RandomSampler(
            patch_size=256, num_patches=10, max_retries=5, seed=42
        )
        coords = list(sampler.sample(fake_slide, no_tissue_mask))
        assert len(coords) == 0

    def test_slide_too_small(self, fake_backend):
        """Slide smaller than patch_size should yield nothing."""
        # Monkey-patch to return tiny dimensions
        from wsistream.types import SlideProperties

        original = fake_backend.get_properties

        def tiny_props():
            p = original()
            return SlideProperties(
                path=p.path, dimensions=(100, 100), level_count=1,
                level_dimensions=((100, 100),), level_downsamples=(1.0,),
                mpp=0.25, vendor="fake",
            )

        fake_backend.get_properties = tiny_props
        slide = SlideHandle("tiny.svs", backend=fake_backend)
        mask = TissueMask(
            mask=np.ones((10, 10), dtype=bool), downsample=10.0,
            slide_dimensions=(100, 100),
        )

        sampler = RandomSampler(patch_size=256, num_patches=10)
        coords = list(sampler.sample(slide, mask))
        assert len(coords) == 0
        slide.close()

    def test_single_patch_slide(self, fake_backend):
        """Slide exactly one patch wide should yield patches (boundary inclusive)."""
        from wsistream.types import SlideProperties

        original = fake_backend.get_properties

        def exact_props():
            p = original()
            return SlideProperties(
                path=p.path, dimensions=(256, 256), level_count=1,
                level_dimensions=((256, 256),), level_downsamples=(1.0,),
                mpp=0.25, vendor="fake",
            )

        fake_backend.get_properties = exact_props
        slide = SlideHandle("exact.svs", backend=fake_backend)
        mask = TissueMask(
            mask=np.ones((10, 10), dtype=bool), downsample=25.6,
            slide_dimensions=(256, 256),
        )

        sampler = RandomSampler(patch_size=256, num_patches=5, seed=42)
        coords = list(sampler.sample(slide, mask))
        assert len(coords) == 5
        # Only valid position is (0, 0)
        for c in coords:
            assert c.x == 0
            assert c.y == 0
        slide.close()

    def test_infinite_mode_can_be_stopped(self, fake_slide, full_tissue_mask):
        """num_patches=-1 produces patches indefinitely; verify we can cap externally."""
        sampler = RandomSampler(patch_size=256, num_patches=-1, seed=42)
        count = 0
        for _ in sampler.sample(fake_slide, full_tissue_mask):
            count += 1
            if count >= 50:
                break
        assert count == 50

    def test_seeded_is_reproducible(self, fake_slide, full_tissue_mask):
        sampler1 = RandomSampler(patch_size=256, num_patches=5, seed=123)
        sampler2 = RandomSampler(patch_size=256, num_patches=5, seed=123)
        coords1 = [(c.x, c.y) for c in sampler1.sample(fake_slide, full_tissue_mask)]
        coords2 = [(c.x, c.y) for c in sampler2.sample(fake_slide, full_tissue_mask)]
        assert coords1 == coords2

    def test_mpp_in_coordinates(self, fake_slide, full_tissue_mask):
        sampler = RandomSampler(patch_size=256, num_patches=1, level=0, seed=42)
        coord = next(iter(sampler.sample(fake_slide, full_tissue_mask)))
        assert coord.mpp == 0.25  # base mpp * downsample(0) = 0.25 * 1.0

    def test_higher_level(self, fake_slide, full_tissue_mask):
        sampler = RandomSampler(patch_size=256, num_patches=5, level=1, seed=42)
        coords = list(sampler.sample(fake_slide, full_tissue_mask))
        assert len(coords) == 5
        for c in coords:
            assert c.level == 1
            assert c.mpp == 0.5  # 0.25 * 2.0

    def test_slide_path_in_coordinates(self, fake_slide, full_tissue_mask):
        sampler = RandomSampler(patch_size=256, num_patches=1, seed=42)
        coord = next(iter(sampler.sample(fake_slide, full_tissue_mask)))
        assert coord.slide_path == "fake.svs"

    def test_target_mpp_selects_correct_level(self, fake_slide, full_tissue_mask):
        """target_mpp=0.5 should select level 1 (mpp=0.25*2.0=0.5)."""
        sampler = RandomSampler(
            patch_size=256, num_patches=5, target_mpp=0.5, seed=42,
        )
        coords = list(sampler.sample(fake_slide, full_tissue_mask))
        assert len(coords) == 5
        for c in coords:
            assert c.level == 1
            assert c.mpp == 0.5

    def test_target_mpp_selects_highest_level(self, fake_slide, full_tissue_mask):
        """target_mpp=1.0 should select level 2 (mpp=0.25*4.0=1.0)."""
        sampler = RandomSampler(
            patch_size=256, num_patches=5, target_mpp=1.0, seed=42,
        )
        coords = list(sampler.sample(fake_slide, full_tissue_mask))
        assert len(coords) == 5
        for c in coords:
            assert c.level == 2
            assert c.mpp == 1.0

    def test_target_mpp_overrides_level(self, fake_slide, full_tissue_mask):
        """When target_mpp is set, the level parameter is ignored."""
        sampler = RandomSampler(
            patch_size=256, num_patches=3, level=0, target_mpp=0.5, seed=42,
        )
        coords = list(sampler.sample(fake_slide, full_tissue_mask))
        for c in coords:
            assert c.level == 1  # target_mpp=0.5 → level 1, not level 0

    def test_target_mpp_closest_match(self, fake_slide, full_tissue_mask):
        """target_mpp=0.6 should pick level 1 (mpp=0.5), not level 2 (mpp=1.0)."""
        sampler = RandomSampler(
            patch_size=256, num_patches=3, target_mpp=0.6, seed=42,
        )
        coords = list(sampler.sample(fake_slide, full_tissue_mask))
        for c in coords:
            assert c.level == 1

    def test_target_mpp_no_mpp_fallback(self, fake_backend, full_tissue_mask):
        """Slide without MPP metadata → target_mpp falls back to level 0."""
        from wsistream.types import SlideProperties

        original = fake_backend.get_properties

        def no_mpp_props():
            p = original()
            return SlideProperties(
                path=p.path, dimensions=p.dimensions,
                level_count=p.level_count,
                level_dimensions=p.level_dimensions,
                level_downsamples=p.level_downsamples,
                mpp=None, vendor=p.vendor,
            )

        fake_backend.get_properties = no_mpp_props
        slide = SlideHandle("no_mpp.svs", backend=fake_backend)

        sampler = RandomSampler(
            patch_size=256, num_patches=3, target_mpp=0.5, seed=42,
        )
        coords = list(sampler.sample(slide, full_tissue_mask))
        assert len(coords) == 3
        for c in coords:
            assert c.level == 0  # best_level_for_mpp returns 0 when mpp is None
        slide.close()


# ── GridSampler ──


class TestGridSampler:
    def test_grid_covers_slide(self, fake_slide, full_tissue_mask):
        sampler = GridSampler(patch_size=256, level=0, tissue_threshold=0.0)
        coords = list(sampler.sample(fake_slide, full_tissue_mask))
        # 4096 / 256 = 16 patches per axis → 256 total
        assert len(coords) == 16 * 16

    def test_grid_coordinates_are_on_grid(self, fake_slide, full_tissue_mask):
        sampler = GridSampler(patch_size=256, level=0, tissue_threshold=0.0)
        for coord in sampler.sample(fake_slide, full_tissue_mask):
            assert coord.x % 256 == 0
            assert coord.y % 256 == 0

    def test_custom_stride(self, fake_slide, full_tissue_mask):
        sampler = GridSampler(patch_size=256, stride=128, level=0, tissue_threshold=0.0)
        coords = list(sampler.sample(fake_slide, full_tissue_mask))
        # stride=128 → (4096-256)/128 + 1 = 31 per axis → 961
        assert len(coords) == 31 * 31

    def test_tissue_filtering(self, fake_slide, no_tissue_mask):
        sampler = GridSampler(patch_size=256, level=0, tissue_threshold=0.5)
        coords = list(sampler.sample(fake_slide, no_tissue_mask))
        assert len(coords) == 0

    def test_partial_tissue(self, fake_slide, center_tissue_mask):
        sampler = GridSampler(patch_size=256, level=0, tissue_threshold=0.1)
        filtered = list(sampler.sample(fake_slide, center_tissue_mask))
        # Center tissue should yield fewer patches than full grid
        assert 0 < len(filtered) < 16 * 16

    def test_higher_level(self, fake_slide, full_tissue_mask):
        sampler = GridSampler(patch_size=256, level=1, tissue_threshold=0.0)
        coords = list(sampler.sample(fake_slide, full_tissue_mask))
        for c in coords:
            assert c.level == 1
            assert c.mpp == 0.5

    def test_deterministic(self, fake_slide, full_tissue_mask):
        sampler = GridSampler(patch_size=256, level=0, tissue_threshold=0.0)
        coords1 = [(c.x, c.y) for c in sampler.sample(fake_slide, full_tissue_mask)]
        coords2 = [(c.x, c.y) for c in sampler.sample(fake_slide, full_tissue_mask)]
        assert coords1 == coords2


# ── MultiMagnificationSampler ──


class TestMultiMagnificationSampler:
    def test_samples_from_multiple_levels(self, fake_slide, full_tissue_mask):
        sampler = MultiMagnificationSampler(
            target_mpps=[0.25, 0.5, 1.0],
            patch_size=256, num_patches=30, seed=42,
        )
        levels = Counter()
        for coord in sampler.sample(fake_slide, full_tissue_mask):
            levels[coord.level] += 1
        # With 30 patches and 3 levels, each should get some
        assert len(levels) >= 2

    def test_respects_count(self, fake_slide, full_tissue_mask):
        sampler = MultiMagnificationSampler(
            target_mpps=[0.25, 0.5], num_patches=15, seed=42,
        )
        coords = list(sampler.sample(fake_slide, full_tissue_mask))
        assert len(coords) == 15

    def test_no_tissue_stops(self, fake_slide, no_tissue_mask):
        sampler = MultiMagnificationSampler(
            target_mpps=[0.25], num_patches=10,
            max_consecutive_failures=5, seed=42,
        )
        coords = list(sampler.sample(fake_slide, no_tissue_mask))
        assert len(coords) == 0

    def test_no_mpp_falls_back(self, fake_backend, full_tissue_mask):
        """Slide without MPP → falls back to single-level RandomSampler."""
        from wsistream.types import SlideProperties

        original = fake_backend.get_properties

        def no_mpp_props():
            p = original()
            return SlideProperties(
                path=p.path, dimensions=p.dimensions,
                level_count=p.level_count,
                level_dimensions=p.level_dimensions,
                level_downsamples=p.level_downsamples,
                mpp=None, vendor=p.vendor,
            )

        fake_backend.get_properties = no_mpp_props
        slide = SlideHandle("no_mpp.svs", backend=fake_backend)

        sampler = MultiMagnificationSampler(
            target_mpps=[0.25, 0.5], num_patches=5, seed=42,
        )
        coords = list(sampler.sample(slide, full_tissue_mask))
        assert len(coords) == 5
        # All at level 0 since no MPP info
        assert all(c.level == 0 for c in coords)
        slide.close()

    def test_custom_weights(self, fake_slide, full_tissue_mask):
        """Heavily weighted toward one level should bias sampling."""
        sampler = MultiMagnificationSampler(
            target_mpps=[0.25, 1.0],
            mpp_weights=[0.99, 0.01],
            num_patches=50, seed=42,
        )
        levels = Counter()
        for coord in sampler.sample(fake_slide, full_tissue_mask):
            levels[coord.level] += 1
        # Level 0 (mpp=0.25) should dominate
        assert levels[0] > levels.get(2, 0)

    def test_weight_distribution_matches_bias(self, fake_slide, full_tissue_mask):
        sampler = MultiMagnificationSampler(
            target_mpps=[0.25, 1.0],
            mpp_weights=[0.9, 0.1],
            num_patches=500,
            seed=42,
        )
        levels = Counter(coord.level for coord in sampler.sample(fake_slide, full_tissue_mask))
        assert levels[0] > levels[2] * 4
