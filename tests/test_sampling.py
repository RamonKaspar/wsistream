"""Tests for all samplers (using FakeBackend, no WSI files needed)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from wsistream.sampling import resolve_mpp_read
from wsistream.sampling.grid import GridSampler
from wsistream.sampling.multi_magnification import MultiMagnificationSampler
from wsistream.sampling.random import RandomSampler, _TissuePositions
from wsistream.slide import SlideHandle
from wsistream.types import SlideProperties, TissueMask


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


def _scanner_properties(mpp: float | None, level_downsamples: tuple[float, ...]) -> SlideProperties:
    """Properties of a 100k x 50k slide with the given pyramid, as scanners write them."""
    return SlideProperties(
        path="scanner.svs",
        dimensions=(100_000, 50_000),
        level_count=len(level_downsamples),
        level_dimensions=tuple((int(100_000 / ds), int(50_000 / ds)) for ds in level_downsamples),
        level_downsamples=level_downsamples,
        mpp=mpp,
        vendor="aperio",
    )


# Pyramids of TCGA slides: 40x (TCGA-E9-A1N4 DX1), 40x with a 0.2325 µm/px
# scan (TCGA-AA-3949 DX1), and 20x (TCGA-12-1600 TS1).
_TCGA_40X = _scanner_properties(0.2485, (1.0, 4.0, 16.001, 32.005))
_TCGA_40X_COARSE = _scanner_properties(0.2325, (1.0, 4.0, 16.0, 64.0))
_TCGA_20X = _scanner_properties(0.5015, (1.0, 4.001))


class TestResolveMppRead:
    def test_level_within_tolerance_is_read_directly(self):
        assert resolve_mpp_read(_TCGA_40X, 0.25, 256, 0.1) == (0, 256, 0.2485)
        assert resolve_mpp_read(_TCGA_40X, 1.0, 256, 0.1) == (1, 256, 0.2485 * 4.0)

    def test_tolerance_is_relative_to_target(self):
        """0.2325 µm/px is 7% from 0.25, so it is read directly."""
        assert resolve_mpp_read(_TCGA_40X_COARSE, 0.25, 256, 0.1) == (0, 256, 0.2325)
        level, read_size, mpp = resolve_mpp_read(_TCGA_40X_COARSE, 0.25, 256, 0.05)
        assert (level, read_size) == (0, 275)
        assert mpp == pytest.approx(0.25, rel=1e-2)

    def test_missing_level_is_read_from_coarsest_finer_level(self):
        """0.5 µm/px on a 40x slide is read at 0.2485 µm/px, 2x larger."""
        level, read_size, mpp = resolve_mpp_read(_TCGA_40X, 0.5, 256, 0.1)
        assert (level, read_size) == (0, 515)
        assert mpp == pytest.approx(0.2485 * 515 / 256)

        level, read_size, mpp = resolve_mpp_read(_TCGA_40X, 2.0, 256, 0.1)
        assert (level, read_size) == (1, 515)
        assert mpp == pytest.approx(0.994 * 515 / 256)

    def test_target_finer_than_slide_reads_native_level_0(self):
        """A 20x slide cannot provide 0.25 µm/px; enlarging would not add detail."""
        assert resolve_mpp_read(_TCGA_20X, 0.25, 256, 0.1) == (0, 256, 0.5015)

    def test_infinite_tolerance_reads_closest_level(self):
        props = _TCGA_40X
        for target in (0.25, 0.5, 1.0, 2.0, 8.0):
            level, read_size, _ = resolve_mpp_read(props, target, 256, float("inf"))
            closest = min(
                range(props.level_count),
                key=lambda lvl: abs(props.mpp_at_level(lvl) - target),
            )
            assert (level, read_size) == (closest, 256)

    def test_requires_mpp(self):
        with pytest.raises(ValueError, match="no MPP metadata"):
            resolve_mpp_read(_scanner_properties(None, (1.0, 4.0)), 0.5, 256, 0.1)


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
        sampler = RandomSampler(patch_size=256, num_patches=10, max_retries=5, seed=42)
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
                path=p.path,
                dimensions=(100, 100),
                level_count=1,
                level_dimensions=((100, 100),),
                level_downsamples=(1.0,),
                mpp=0.25,
                vendor="fake",
            )

        fake_backend.get_properties = tiny_props
        slide = SlideHandle("tiny.svs", backend=fake_backend)
        mask = TissueMask(
            mask=np.ones((10, 10), dtype=bool),
            downsample=10.0,
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
                path=p.path,
                dimensions=(256, 256),
                level_count=1,
                level_dimensions=((256, 256),),
                level_downsamples=(1.0,),
                mpp=0.25,
                vendor="fake",
            )

        fake_backend.get_properties = exact_props
        slide = SlideHandle("exact.svs", backend=fake_backend)
        mask = TissueMask(
            mask=np.ones((10, 10), dtype=bool),
            downsample=25.6,
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
            patch_size=256,
            num_patches=5,
            target_mpp=0.5,
            seed=42,
        )
        coords = list(sampler.sample(fake_slide, full_tissue_mask))
        assert len(coords) == 5
        for c in coords:
            assert c.level == 1
            assert c.mpp == 0.5

    def test_target_mpp_selects_highest_level(self, fake_slide, full_tissue_mask):
        """target_mpp=1.0 should select level 2 (mpp=0.25*4.0=1.0)."""
        sampler = RandomSampler(
            patch_size=256,
            num_patches=5,
            target_mpp=1.0,
            seed=42,
        )
        coords = list(sampler.sample(fake_slide, full_tissue_mask))
        assert len(coords) == 5
        for c in coords:
            assert c.level == 2
            assert c.mpp == 1.0

    def test_target_mpp_overrides_level(self, fake_slide, full_tissue_mask):
        """When target_mpp is set, the level parameter is ignored."""
        sampler = RandomSampler(
            patch_size=256,
            num_patches=3,
            level=0,
            target_mpp=0.5,
            seed=42,
        )
        coords = list(sampler.sample(fake_slide, full_tissue_mask))
        for c in coords:
            assert c.level == 1  # target_mpp=0.5 → level 1, not level 0

    def test_target_mpp_resamples_without_matching_level(self, fake_slide, full_tissue_mask):
        """No level is within 10% of 0.6, so 307 px are read at level 1 (mpp=0.5)."""
        sampler = RandomSampler(
            patch_size=256,
            num_patches=3,
            target_mpp=0.6,
            seed=42,
        )
        coords = list(sampler.sample(fake_slide, full_tissue_mask))
        assert len(coords) == 3
        for c in coords:
            assert c.level == 1
            assert c.patch_size == 307
            assert c.mpp == pytest.approx(0.6, rel=1e-2)

    def test_target_mpp_within_tolerance_reads_level_directly(self, fake_slide, full_tissue_mask):
        """0.53 is 6% from level 1 (mpp=0.5), inside the default 10% tolerance."""
        sampler = RandomSampler(patch_size=256, num_patches=3, target_mpp=0.53, seed=42)
        for c in sampler.sample(fake_slide, full_tissue_mask):
            assert (c.level, c.patch_size, c.mpp) == (1, 256, 0.5)

    def test_target_mpp_finer_than_slide_reads_native_level_0(self, fake_slide, full_tissue_mask):
        sampler = RandomSampler(patch_size=256, num_patches=3, target_mpp=0.1, seed=42)
        for c in sampler.sample(fake_slide, full_tissue_mask):
            assert (c.level, c.patch_size, c.mpp) == (0, 256, 0.25)

    def test_infinite_mpp_tolerance_reads_closest_level(self, fake_slide, full_tissue_mask):
        sampler = RandomSampler(
            patch_size=256,
            num_patches=3,
            target_mpp=0.6,
            seed=42,
            mpp_tolerance=float("inf"),
        )
        for c in sampler.sample(fake_slide, full_tissue_mask):
            assert (c.level, c.patch_size, c.mpp) == (1, 256, 0.5)

    @pytest.mark.parametrize("mpp_tolerance", [-0.1, float("nan")])
    def test_invalid_mpp_tolerance(self, mpp_tolerance):
        with pytest.raises(ValueError, match="mpp_tolerance"):
            RandomSampler(mpp_tolerance=mpp_tolerance)

    def test_output_size_is_patch_size(self):
        assert RandomSampler(patch_size=224, target_mpp=0.5).output_size == 224

    def test_target_mpp_no_mpp_fallback(self, fake_backend, full_tissue_mask):
        """Slide without MPP metadata → target_mpp falls back to level 0."""
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

        sampler = RandomSampler(
            patch_size=256,
            num_patches=3,
            target_mpp=0.5,
            seed=42,
        )
        coords = list(sampler.sample(slide, full_tissue_mask))
        assert len(coords) == 3
        for c in coords:
            assert c.level == 0  # best_level_for_mpp returns 0 when mpp is None
        slide.close()

    def test_low_tissue_slide_yields_every_patch(self, fake_slide):
        """A slide with little tissue must not end early when rejection sampling misses."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:14, 70:74] = True  # 0.16% of the slide
        tissue_mask = TissueMask(mask=mask, downsample=40.96, slide_dimensions=(4096, 4096))
        sampler = RandomSampler(patch_size=64, num_patches=200, max_retries=1, seed=42)

        coords = list(sampler.sample(fake_slide, tissue_mask))

        assert len(coords) == 200
        for c in coords:
            assert tissue_mask.contains_tissue(c.x, c.y, 64, 64, 0.4)

    def test_fallback_keeps_the_rejection_sampling_distribution(self, fake_slide):
        """Patches split between two tissue regions in proportion to their accepted area."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:20, 10:20] = True
        mask[60:90, 60:90] = True  # 9x the area of the first region
        tissue_mask = TissueMask(mask=mask, downsample=40.96, slide_dimensions=(4096, 4096))

        def fraction_in_small_region(max_retries):
            sampler = RandomSampler(
                patch_size=64, num_patches=4000, max_retries=max_retries, seed=42
            )
            coords = list(sampler.sample(fake_slide, tissue_mask))
            assert len(coords) == 4000
            return np.mean([c.x < 2048 for c in coords])

        # max_retries=1 draws most patches through the fallback, 10_000 none.
        assert fraction_in_small_region(1) == pytest.approx(
            fraction_in_small_region(10_000), abs=0.02
        )


class _IndexRng:
    """Stand-in generator whose ``integers`` returns a fixed value."""

    def __init__(self, value: int) -> None:
        self.value = value

    def integers(self, low: int, high: int) -> int:
        return self.value


class TestTissuePositions:
    def test_draws_are_a_bijection_onto_accepted_positions(self):
        """Every random index maps to a distinct position that rejection sampling accepts."""
        rng = np.random.default_rng(0)
        # Non-integer downsample and a slide wider than the mask exercise edge clamping.
        tissue_mask = TissueMask(
            mask=rng.random((15, 20)) < 0.4, downsample=7.3, slide_dimensions=(150, 110)
        )
        footprint = 40
        max_x, max_y = 150 - footprint, 110 - footprint
        accepted = {
            (x, y)
            for x in range(max_x + 1)
            for y in range(max_y + 1)
            if tissue_mask.contains_tissue(x, y, footprint, footprint, 0.4)
        }
        positions = _TissuePositions(tissue_mask, max_x, max_y, footprint, 0.4)

        drawn = [positions.sample(_IndexRng(i)) for i in range(len(accepted))]

        assert accepted
        assert not positions.empty
        assert int(positions._cumulative[-1]) == len(accepted)
        assert len(set(drawn)) == len(accepted)
        assert set(drawn) == accepted

    def test_no_accepted_position_is_empty(self):
        tissue_mask = TissueMask(
            mask=np.zeros((10, 10), dtype=bool), downsample=10.0, slide_dimensions=(100, 100)
        )
        assert _TissuePositions(tissue_mask, 50, 50, 50, 0.4).empty


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
            patch_size=256,
            num_patches=30,
            seed=42,
        )
        levels = Counter()
        for coord in sampler.sample(fake_slide, full_tissue_mask):
            levels[coord.level] += 1
        # With 30 patches and 3 levels, each should get some
        assert len(levels) >= 2

    def test_resamples_targets_without_matching_level(self, fake_slide, full_tissue_mask):
        """2.0 µm/px has no level, so 512 px are read at level 2 (mpp=1.0)."""
        sampler = MultiMagnificationSampler(
            target_mpps=[2.0],
            patch_size=256,
            num_patches=5,
            seed=42,
        )
        coords = list(sampler.sample(fake_slide, full_tissue_mask))
        assert len(coords) == 5
        for c in coords:
            assert (c.level, c.patch_size, c.mpp) == (2, 512, 2.0)

    def test_mpp_tolerance_reaches_inner_sampler(self, fake_slide, full_tissue_mask):
        sampler = MultiMagnificationSampler(
            target_mpps=[2.0],
            patch_size=256,
            num_patches=5,
            seed=42,
            mpp_tolerance=float("inf"),
        )
        for c in sampler.sample(fake_slide, full_tissue_mask):
            assert (c.level, c.patch_size, c.mpp) == (2, 256, 1.0)

    def test_respects_count(self, fake_slide, full_tissue_mask):
        sampler = MultiMagnificationSampler(
            target_mpps=[0.25, 0.5],
            num_patches=15,
            seed=42,
        )
        coords = list(sampler.sample(fake_slide, full_tissue_mask))
        assert len(coords) == 15

    def test_no_tissue_stops(self, fake_slide, no_tissue_mask):
        sampler = MultiMagnificationSampler(
            target_mpps=[0.25],
            num_patches=10,
            max_consecutive_failures=5,
            seed=42,
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

        sampler = MultiMagnificationSampler(
            target_mpps=[0.25, 0.5],
            num_patches=5,
            seed=42,
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
            num_patches=50,
            seed=42,
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


# ── Parameter validation ──


@pytest.mark.parametrize("sampler_class", [RandomSampler, GridSampler, MultiMagnificationSampler])
class TestSamplerParameterValidation:
    @pytest.mark.parametrize("threshold", [-0.1, 1.5, float("nan")])
    def test_tissue_threshold_outside_unit_interval_raises(self, sampler_class, threshold):
        with pytest.raises(ValueError, match="tissue_threshold must be in"):
            sampler_class(tissue_threshold=threshold)

    @pytest.mark.parametrize("threshold", [0.0, 1.0])
    def test_tissue_threshold_bounds_are_accepted(self, sampler_class, threshold):
        assert sampler_class(tissue_threshold=threshold).tissue_threshold == threshold


class TestMultiMagnificationWeightValidation:
    def test_negative_weight_raises(self):
        """A negative weight used to pass construction and fail on the first draw."""
        with pytest.raises(ValueError, match="mpp_weights must be >= 0"):
            MultiMagnificationSampler(mpp_weights=[1.0, -0.5, 1.0, 1.0])

    def test_zero_weight_is_accepted(self):
        sampler = MultiMagnificationSampler(mpp_weights=[1.0, 0.0, 1.0, 1.0])
        assert sampler.mpp_weights == [1.0, 0.0, 1.0, 1.0]
