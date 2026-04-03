"""Tests for ContinuousMagnificationSampler."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import FakeBackend, fake_slide_paths
from wsistream.pipeline import PatchPipeline
from wsistream.sampling.continuous_magnification import (
    ContinuousMagnificationSampler,
    _best_level_for_downsample,
    _compute_maxavg_weights,
    _compute_minmax_weights,
    _transfer_potential,
)
from wsistream.slide import SlideHandle
from wsistream.tissue import OtsuTissueDetector
from wsistream.types import SlideProperties, TissueMask


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_slide():
    """SlideHandle backed by FakeBackend (4096x4096, 3 levels, mpp=0.25)."""
    backend = FakeBackend()
    return SlideHandle("fake.svs", backend=backend)


@pytest.fixture
def full_tissue_mask():
    """100 % tissue mask matching FakeBackend dimensions."""
    mask = np.ones((100, 100), dtype=bool)
    return TissueMask(mask=mask, downsample=40.96, slide_dimensions=(4096, 4096))


@pytest.fixture
def no_tissue_mask():
    """0 % tissue."""
    mask = np.zeros((100, 100), dtype=bool)
    return TissueMask(mask=mask, downsample=40.96, slide_dimensions=(4096, 4096))


# ---------------------------------------------------------------------------
# Transfer potential & distribution helpers
# ---------------------------------------------------------------------------


class TestTransferPotential:
    def test_closed_form_matches_numerical(self):
        """K̄(x) = 4x/3 - a³/(3x²) - x²/b matches numerical integration."""
        a, b = 0.25, 2.0
        grid = np.array([0.5, 1.0, 1.5])
        closed = _transfer_potential(grid, a, b)

        # Numerical integration for verification
        from scipy.integrate import quad

        def k_info(x, y):
            return (min(x, y) / max(x, y)) ** 2

        for i, x in enumerate(grid):
            numerical, _ = quad(lambda y: k_info(x, y), a, b)
            assert abs(closed[i] - numerical) < 1e-10

    def test_peak_is_interior(self):
        """K̄ peaks in the interior, not at the boundaries."""
        a, b = 0.25, 2.0
        grid = np.linspace(a, b, 1000)
        k_bar = _transfer_potential(grid, a, b)
        peak_idx = np.argmax(k_bar)
        # Peak should be away from the edges
        assert 100 < peak_idx < 900
        # Boundaries should have lower potential
        assert k_bar[0] < k_bar[peak_idx]
        assert k_bar[-1] < k_bar[peak_idx]


class TestMaxAvgDistribution:
    def test_sums_to_one(self):
        grid = np.linspace(0.25, 2.0, 1000)
        w = _compute_maxavg_weights(grid, 0.25, 2.0, lam=1.0)
        assert abs(w.sum() - 1.0) < 1e-12

    def test_concentrates_at_high_potential(self):
        """MaxAvg with small lambda should peak near the K̄ maximum."""
        grid = np.linspace(0.25, 2.0, 1000)
        k_bar = _transfer_potential(grid, 0.25, 2.0)
        k_bar_peak = grid[np.argmax(k_bar)]
        w = _compute_maxavg_weights(grid, 0.25, 2.0, lam=0.1)
        w_peak = grid[np.argmax(w)]
        assert abs(w_peak - k_bar_peak) < 0.05

    def test_high_lambda_approaches_uniform(self):
        """Large lambda should give near-uniform weights."""
        grid = np.linspace(0.25, 2.0, 1000)
        w = _compute_maxavg_weights(grid, 0.25, 2.0, lam=100.0)
        assert w.max() / w.min() < 1.5  # close to uniform


class TestMinMaxDistribution:
    def test_sums_to_one(self):
        grid = np.linspace(0.25, 2.0, 100)  # small grid for speed
        w = _compute_minmax_weights(grid)
        assert abs(w.sum() - 1.0) < 1e-8

    def test_oversamples_boundaries(self):
        """MinMax should place more mass at boundaries than center."""
        grid = np.linspace(0.25, 2.0, 100)
        w = _compute_minmax_weights(grid)
        boundary_mass = w[:10].sum() + w[-10:].sum()
        center_mass = w[40:60].sum()
        assert boundary_mass > center_mass


# ---------------------------------------------------------------------------
# Level selection
# ---------------------------------------------------------------------------


class TestLevelSelection:
    def test_picks_coarsest_valid_level(self):
        props = SlideProperties(
            path="s.svs",
            dimensions=(10000, 10000),
            level_count=3,
            level_dimensions=((10000, 10000), (5000, 5000), (2500, 2500)),
            level_downsamples=(1.0, 2.0, 4.0),
            mpp=0.25,
            vendor="test",
        )
        # target=0.75 → level 1 has mpp=0.50 ≤ 0.75, level 2 has 1.0 > 0.75
        assert _best_level_for_downsample(props, 0.75) == 1

    def test_picks_level_2_for_high_target(self):
        props = SlideProperties(
            path="s.svs",
            dimensions=(10000, 10000),
            level_count=3,
            level_dimensions=((10000, 10000), (5000, 5000), (2500, 2500)),
            level_downsamples=(1.0, 2.0, 4.0),
            mpp=0.25,
            vendor="test",
        )
        # target=1.5 → level 2 has mpp=1.0 ≤ 1.5
        assert _best_level_for_downsample(props, 1.5) == 2

    def test_falls_back_to_level_0_if_target_finer_than_native(self):
        props = SlideProperties(
            path="s.svs",
            dimensions=(10000, 10000),
            level_count=3,
            level_dimensions=((10000, 10000), (5000, 5000), (2500, 2500)),
            level_downsamples=(1.0, 2.0, 4.0),
            mpp=0.5,
            vendor="test",
        )
        # target=0.3 < native 0.5 → must use level 0
        assert _best_level_for_downsample(props, 0.3) == 0

    def test_no_mpp_returns_level_0(self):
        props = SlideProperties(
            path="s.svs",
            dimensions=(10000, 10000),
            level_count=3,
            level_dimensions=((10000, 10000), (5000, 5000), (2500, 2500)),
            level_downsamples=(1.0, 2.0, 4.0),
            mpp=None,
            vendor="test",
        )
        assert _best_level_for_downsample(props, 1.0) == 0


# ---------------------------------------------------------------------------
# Sampler validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_mpp_range(self):
        with pytest.raises(ValueError, match="mpp_range"):
            ContinuousMagnificationSampler(mpp_range=(2.0, 0.25))

    def test_invalid_distribution(self):
        with pytest.raises(ValueError, match="distribution"):
            ContinuousMagnificationSampler(distribution="bad")

    def test_invalid_output_size(self):
        with pytest.raises(ValueError, match="output_size"):
            ContinuousMagnificationSampler(output_size=0)

    def test_invalid_lambda(self):
        with pytest.raises(ValueError, match="lambda_maxavg"):
            ContinuousMagnificationSampler(lambda_maxavg=-1.0)

    def test_invalid_num_patches(self):
        with pytest.raises(ValueError, match="num_patches"):
            ContinuousMagnificationSampler(num_patches=0)

    def test_invalid_max_retries(self):
        with pytest.raises(ValueError, match="max_retries"):
            ContinuousMagnificationSampler(max_retries=0)


# ---------------------------------------------------------------------------
# Sampling behaviour
# ---------------------------------------------------------------------------


class TestSampling:
    def test_yields_correct_count(self, fake_slide, full_tissue_mask):
        sampler = ContinuousMagnificationSampler(
            num_patches=10,
            output_size=64,
            seed=42,
        )
        coords = list(sampler.sample(fake_slide, full_tissue_mask))
        assert len(coords) == 10

    def test_mpp_within_range(self, fake_slide, full_tissue_mask):
        sampler = ContinuousMagnificationSampler(
            num_patches=50,
            output_size=64,
            seed=42,
        )
        for coord in sampler.sample(fake_slide, full_tissue_mask):
            assert 0.25 <= coord.mpp <= 2.0

    def test_coordinates_within_bounds(self, fake_slide, full_tissue_mask):
        sampler = ContinuousMagnificationSampler(
            num_patches=50,
            output_size=64,
            seed=42,
        )
        props = fake_slide.properties
        for coord in sampler.sample(fake_slide, full_tissue_mask):
            ds = props.level_downsamples[coord.level]
            crop_l0 = round(coord.patch_size * ds)
            assert coord.x >= 0
            assert coord.y >= 0
            assert coord.x + crop_l0 <= props.width
            assert coord.y + crop_l0 <= props.height

    def test_no_tissue_breaks(self, fake_slide, no_tissue_mask):
        sampler = ContinuousMagnificationSampler(
            num_patches=10,
            output_size=64,
            seed=42,
        )
        coords = list(sampler.sample(fake_slide, no_tissue_mask))
        assert len(coords) == 0

    def test_tiny_slide_does_not_loop_forever(self):
        """A slide too small for most mpp values should terminate."""

        class TinyBackend(FakeBackend):
            def get_properties(self):
                return SlideProperties(
                    path="tiny.svs",
                    dimensions=(200, 200),
                    level_count=1,
                    level_dimensions=((200, 200),),
                    level_downsamples=(1.0,),
                    mpp=0.25,
                    vendor="fake",
                )

        slide = SlideHandle("tiny.svs", backend=TinyBackend())
        mask = TissueMask(
            mask=np.ones((10, 10), dtype=bool),
            downsample=20.0,
            slide_dimensions=(200, 200),
        )
        sampler = ContinuousMagnificationSampler(
            num_patches=10,
            output_size=224,
            max_retries=20,
            seed=42,
        )
        # Should not hang; at most a few patches (or zero) from narrow mpp range
        coords = list(sampler.sample(slide, mask))
        assert len(coords) <= 10

    def test_seed_reproducibility(self, fake_slide, full_tissue_mask):
        s1 = ContinuousMagnificationSampler(num_patches=20, output_size=64, seed=99)
        s2 = ContinuousMagnificationSampler(num_patches=20, output_size=64, seed=99)
        c1 = [(c.x, c.y, c.mpp) for c in s1.sample(fake_slide, full_tissue_mask)]
        c2 = [(c.x, c.y, c.mpp) for c in s2.sample(fake_slide, full_tissue_mask)]
        assert c1 == c2

    def test_output_size_attribute_exposed(self):
        sampler = ContinuousMagnificationSampler(output_size=224)
        assert sampler.output_size == 224

    def test_maxavg_samples(self, fake_slide, full_tissue_mask):
        sampler = ContinuousMagnificationSampler(
            num_patches=20,
            output_size=64,
            distribution="maxavg",
            seed=42,
        )
        coords = list(sampler.sample(fake_slide, full_tissue_mask))
        assert len(coords) == 20

    def test_minmax_samples(self, fake_slide, full_tissue_mask):
        sampler = ContinuousMagnificationSampler(
            num_patches=20,
            output_size=64,
            distribution="minmax",
            seed=42,
        )
        coords = list(sampler.sample(fake_slide, full_tissue_mask))
        assert len(coords) == 20

    def test_crop_size_formula(self, fake_slide, full_tissue_mask):
        """crop_size = round(output_size * target_mpp / level_mpp)."""
        sampler = ContinuousMagnificationSampler(
            num_patches=30,
            output_size=224,
            seed=42,
        )
        props = fake_slide.properties
        for coord in sampler.sample(fake_slide, full_tissue_mask):
            level_mpp = props.mpp * props.level_downsamples[coord.level]
            expected = round(224 * coord.mpp / level_mpp)
            expected = max(expected, 1)
            assert coord.patch_size == expected

    def test_infinite_mode(self, fake_slide, full_tissue_mask):
        sampler = ContinuousMagnificationSampler(
            num_patches=-1,
            output_size=64,
            seed=42,
        )
        count = 0
        for _ in sampler.sample(fake_slide, full_tissue_mask):
            count += 1
            if count >= 50:
                break
        assert count == 50


# ---------------------------------------------------------------------------
# No-MPP fallback
# ---------------------------------------------------------------------------


class TestNoMppFallback:
    def test_falls_back_to_random_sampler(self):
        """Slides without MPP metadata should still yield patches."""

        class NoMppBackend(FakeBackend):
            def get_properties(self):
                p = super().get_properties()
                return SlideProperties(
                    path=p.path,
                    dimensions=p.dimensions,
                    level_count=p.level_count,
                    level_dimensions=p.level_dimensions,
                    level_downsamples=p.level_downsamples,
                    mpp=None,
                    vendor=p.vendor,
                )

        slide = SlideHandle("no_mpp.svs", backend=NoMppBackend())
        mask = TissueMask(
            mask=np.ones((100, 100), dtype=bool),
            downsample=40.96,
            slide_dimensions=(4096, 4096),
        )
        sampler = ContinuousMagnificationSampler(
            num_patches=5,
            output_size=64,
            seed=42,
        )
        coords = list(sampler.sample(slide, mask))
        assert len(coords) == 5
        # Fallback uses level 0 with output_size as patch_size
        for c in coords:
            assert c.level == 0
            assert c.patch_size == 64


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    def test_patches_are_output_size(self):
        """Pipeline should resize patches to sampler.output_size."""
        paths = fake_slide_paths(2)
        pipeline = PatchPipeline(
            slide_paths=paths,
            backend=FakeBackend(),
            sampler=ContinuousMagnificationSampler(
                output_size=128,
                num_patches=-1,
                seed=42,
            ),
            patches_per_slide=5,
        )
        for result in pipeline:
            assert result.image.shape == (128, 128, 3)

    def test_mpp_varies_across_patches(self):
        """Continuous sampler should produce varying mpp values."""
        paths = fake_slide_paths(2)
        pipeline = PatchPipeline(
            slide_paths=paths,
            backend=FakeBackend(),
            sampler=ContinuousMagnificationSampler(
                output_size=128,
                num_patches=-1,
                seed=42,
            ),
            patches_per_slide=10,
        )
        mpps = set()
        for result in pipeline:
            mpps.add(round(result.coordinate.mpp, 4))
        # Should have many distinct mpp values
        assert len(mpps) > 5

    def test_no_resize_for_standard_samplers(self):
        """Standard samplers (no output_size) should not trigger resize."""
        from wsistream.sampling import RandomSampler

        paths = fake_slide_paths(1)
        pipeline = PatchPipeline(
            slide_paths=paths,
            backend=FakeBackend(),
            sampler=RandomSampler(patch_size=256, num_patches=5, seed=42),
            patches_per_slide=5,
        )
        for result in pipeline:
            assert result.image.shape == (256, 256, 3)

    def test_stats_track_continuous_mpp(self):
        """Pipeline stats should contain continuous mpp keys."""
        paths = fake_slide_paths(1)
        pipeline = PatchPipeline(
            slide_paths=paths,
            backend=FakeBackend(),
            sampler=ContinuousMagnificationSampler(
                output_size=64,
                num_patches=-1,
                seed=42,
            ),
            patches_per_slide=10,
        )
        list(pipeline)
        stats = pipeline.stats_dict()
        mpp_keys = [k for k in stats if k.startswith("pipeline/mpp_")]
        assert len(mpp_keys) > 1

    def test_filter_sees_resized_image(self):
        """Filter should receive the output_size image, not the raw crop."""
        from wsistream.filters.base import PatchFilter

        class SizeCheckFilter(PatchFilter):
            """Accept patch and record its shape."""

            def __init__(self):
                self.seen_shapes = []

            def accept(self, patch):
                self.seen_shapes.append(patch.shape[:2])
                return True

        size_filter = SizeCheckFilter()
        paths = fake_slide_paths(1)
        pipeline = PatchPipeline(
            slide_paths=paths,
            backend=FakeBackend(),
            sampler=ContinuousMagnificationSampler(
                output_size=128,
                num_patches=-1,
                seed=42,
            ),
            patch_filter=size_filter,
            patches_per_slide=5,
        )
        list(pipeline)
        # Filter must see (128, 128) — the output size, not the variable crop
        for h, w in size_filter.seen_shapes:
            assert h == 128
            assert w == 128

    def test_patch_size_differs_from_image_shape(self):
        """coord.patch_size is the crop size, image is output_size."""
        paths = fake_slide_paths(1)
        pipeline = PatchPipeline(
            slide_paths=paths,
            backend=FakeBackend(),
            sampler=ContinuousMagnificationSampler(
                output_size=128,
                num_patches=-1,
                seed=42,
            ),
            patches_per_slide=10,
        )
        crop_sizes = set()
        for result in pipeline:
            assert result.image.shape == (128, 128, 3)
            crop_sizes.add(result.coordinate.patch_size)
        # Crop sizes should vary and generally differ from output_size
        assert len(crop_sizes) > 1


class TestWsiStreamDataset:
    def test_dict_output_shape(self):
        """WsiStreamDataset yields fixed-size tensors with variable patch_size."""
        from torch.utils.data import DataLoader

        from wsistream.torch import WsiStreamDataset

        dataset = WsiStreamDataset(
            slide_paths=fake_slide_paths(2),
            backend=FakeBackend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=ContinuousMagnificationSampler(
                output_size=64,
                num_patches=-1,
                seed=42,
            ),
            patches_per_slide=5,
            cycle=False,
        )
        loader = DataLoader(dataset, batch_size=2)
        batch = next(iter(loader))
        # Image is always (B, 3, output_size, output_size)
        assert batch["image"].shape == (2, 3, 64, 64)
        # mpp should be continuous values (not -1.0 sentinel)
        assert all(m > 0 for m in batch["mpp"].tolist())
