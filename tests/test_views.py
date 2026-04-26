"""Tests for multi-view patch generation."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from tests.conftest import FakeBackend, fake_slide_paths
from wsistream.datasets.base import DatasetAdapter
from wsistream.pipeline import PatchPipeline
from wsistream.sampling.base import PatchSampler
from wsistream.sampling import RandomSampler
from wsistream.tissue import OtsuTissueDetector
from wsistream.torch import WsiStreamDataset
from wsistream.transforms import ComposeTransforms, ResizeTransform
from wsistream.transforms.base import PatchTransform
from wsistream.types import PatchCoordinate, PatchResult, SlideMetadata, SlideProperties
from wsistream.views import RandomResizedCrop, ViewConfig, expand_view_names


class _AddValue(PatchTransform):
    """Add a constant value to the patch."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __call__(self, image: np.ndarray) -> np.ndarray:
        out = np.clip(image.astype(np.int16) + self.value, 0, 255)
        return out.astype(np.uint8)


class _RecordingCrop(RandomResizedCrop):
    """RandomResizedCrop that records sampled parameters for tests."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.params: list[tuple[int, int, int, int]] = []

    def __call__(self, image: np.ndarray) -> np.ndarray:
        params = self.sample_params(image)
        self.params.append((params.x, params.y, params.width, params.height))
        return self.apply_params(image, params)


class _TrackBackend(FakeBackend):
    calls: list[tuple[int, int, int, int, int]] = []

    def read_region(self, x: int, y: int, level: int, width: int, height: int) -> np.ndarray:
        type(self).calls.append((x, y, level, width, height))
        value = 80 + level * 40
        return np.full((height, width, 3), value, dtype=np.uint8)


class _PaddingTrackBackend(_TrackBackend):
    def read_region(self, x: int, y: int, level: int, width: int, height: int) -> np.ndarray:
        type(self).calls.append((x, y, level, width, height))
        return np.full((height, width, 3), 128, dtype=np.uint8)


class _SmallSlideBackend(FakeBackend):
    """Backend whose slide is tiny so that mpp_override views exceed its dimensions."""

    def get_properties(self) -> SlideProperties:
        props = super().get_properties()
        return SlideProperties(
            path=props.path,
            dimensions=(128, 128),
            level_count=props.level_count,
            level_dimensions=((128, 128), (64, 64), (32, 32)),
            level_downsamples=props.level_downsamples,
            mpp=props.mpp,
            vendor=props.vendor,
        )

    def read_region(self, x: int, y: int, level: int, width: int, height: int) -> np.ndarray:
        return np.full((height, width, 3), 128, dtype=np.uint8)

    def get_thumbnail(self, size: tuple[int, int]) -> np.ndarray:
        w, h = size
        thumb = np.ones((h, w, 3), dtype=np.uint8) * 240
        cy, cx = h // 2, w // 2
        r = min(h, w) // 4
        thumb[cy - r : cy + r, cx - r : cx + r] = 80
        return thumb


class _NoMppBackend(FakeBackend):
    def get_properties(self) -> SlideProperties:
        props = super().get_properties()
        return SlideProperties(
            path=props.path,
            dimensions=props.dimensions,
            level_count=props.level_count,
            level_dimensions=props.level_dimensions,
            level_downsamples=props.level_downsamples,
            mpp=None,
            vendor=props.vendor,
        )


class _Adapter(DatasetAdapter):
    def parse_metadata(self, slide_path: str) -> SlideMetadata:
        return SlideMetadata(slide_path=slide_path, dataset_name="test", patient_id="P001")


class _FixedSampler(PatchSampler):
    def __init__(self, coord: PatchCoordinate) -> None:
        self.coord = coord

    def sample(self, slide, tissue_mask):
        yield self.coord


class TestRandomResizedCrop:
    def test_output_shape_and_dtype(self, random_patch_256):
        crop = RandomResizedCrop(size=96, scale=(0.2, 0.5), seed=42)
        out = crop(random_patch_256)
        assert out.shape == (96, 96, 3)
        assert out.dtype == np.uint8

    def test_seeded_is_reproducible(self, random_patch_256):
        c1 = RandomResizedCrop(size=128, scale=(0.2, 0.8), seed=7)
        c2 = RandomResizedCrop(size=128, scale=(0.2, 0.8), seed=7)
        np.testing.assert_array_equal(c1(random_patch_256), c2(random_patch_256))

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError, match="size"):
            RandomResizedCrop(size=0, scale=(0.2, 0.8))

    def test_sample_params_fit_inside_image(self, random_patch_256):
        crop = RandomResizedCrop(size=96, scale=(0.2, 0.5), seed=42)
        params = crop.sample_params(random_patch_256)
        assert 0 <= params.x < 256
        assert 0 <= params.y < 256
        assert params.x + params.width <= 256
        assert params.y + params.height <= 256

    def test_sample_params_respect_scale_range(self, random_patch_256):
        # Integer rounding means the realized area fraction can fall just outside
        # the requested scale range by at most 1 pixel per dimension.  Use a
        # tolerance that accounts for this instead of asserting exact bounds.
        crop = RandomResizedCrop(size=96, scale=(0.2, 0.5), ratio=(1.0, 1.0), seed=42)
        params = crop.sample_params(random_patch_256)
        area = 256 * 256
        area_fraction = (params.width * params.height) / area
        # Max rounding error is about 2*sqrt(area * scale_min) / area = 0.018 here.
        tol = 0.02
        assert area_fraction >= 0.2 - tol, f"area_fraction {area_fraction:.4f} too small"
        assert area_fraction <= 0.5 + tol, f"area_fraction {area_fraction:.4f} too large"

    def test_invalid_scale_above_one_raises(self):
        with pytest.raises(ValueError, match="scale"):
            RandomResizedCrop(size=96, scale=(0.2, 1.2))

    def test_output_size_larger_than_patch_upsamples(self):
        image = np.full((128, 128, 3), 100, dtype=np.uint8)
        crop = RandomResizedCrop(size=224, scale=(0.5, 1.0), seed=42)
        out = crop(image)
        assert out.shape == (224, 224, 3)
        assert out.dtype == np.uint8

    def test_successive_calls_produce_different_crops(self, random_patch_256):
        crop = RandomResizedCrop(size=96, scale=(0.05, 0.5), seed=0)
        params_a = crop.sample_params(random_patch_256)
        params_b = crop.sample_params(random_patch_256)
        # With different RNG draws the crops should (almost certainly) differ.
        assert (params_a.x, params_a.y) != (params_b.x, params_b.y) or (
            params_a.width != params_b.width or params_a.height != params_b.height
        ), "Two successive calls to sample_params returned identical crops"


class TestViewConfig:
    def test_expand_counted_names(self):
        with pytest.warns(UserWarning, match="identical copies"):
            local_view = ViewConfig(name="local", count=3)
        names = expand_view_names([ViewConfig(name="global"), local_view])
        assert names == ["global", "local_0", "local_1", "local_2"]

    def test_invalid_count_raises(self):
        with pytest.raises(ValueError, match="count"):
            ViewConfig(name="local", count=0)

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            ViewConfig(name="")

    def test_invalid_mpp_override_raises(self):
        with pytest.raises(ValueError, match="mpp_override"):
            ViewConfig(name="v", mpp_override=-1.0)

    def test_patch_size_override_without_mpp_override_raises(self):
        with pytest.raises(ValueError, match="patch_size_override requires mpp_override"):
            ViewConfig(name="v", patch_size_override=64)

    def test_patch_size_override_with_mpp_override_is_valid(self):
        v = ViewConfig(name="v", mpp_override=1.0, patch_size_override=64)
        assert v.patch_size_override == 64


class TestPatchResult:
    def test_requires_image_or_views(self):
        coord = PatchCoordinate(
            x=0,
            y=0,
            level=0,
            patch_size=64,
            mpp=None,
            slide_path="slide.svs",
        )
        with pytest.raises(ValueError, match="either image or views"):
            PatchResult(image=None, views=None, coordinate=coord, tissue_fraction=1.0)

    def test_rejects_empty_views(self):
        coord = PatchCoordinate(
            x=0,
            y=0,
            level=0,
            patch_size=64,
            mpp=None,
            slide_path="slide.svs",
        )
        with pytest.raises(ValueError, match="at least one view"):
            PatchResult(image=None, views={}, coordinate=coord, tissue_fraction=1.0)

    def test_rejects_both_image_and_views(self):
        coord = PatchCoordinate(
            x=0,
            y=0,
            level=0,
            patch_size=64,
            mpp=None,
            slide_path="slide.svs",
        )
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="both image and views"):
            PatchResult(
                image=img,
                views={"v": img},
                coordinate=coord,
                tissue_fraction=1.0,
            )


class TestPatchPipelineViews:
    def test_single_view_result_is_unchanged(self):
        pipeline = PatchPipeline(
            slide_paths=fake_slide_paths(1),
            backend=FakeBackend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=128, num_patches=1, seed=42),
            pool_size=1,
            patches_per_slide=1,
        )
        result = next(iter(pipeline))
        assert result.image is not None
        assert result.views is None
        assert result.image.shape == (128, 128, 3)

    def test_stats_not_updated_when_transform_raises(self):
        class FailingTransform(PatchTransform):
            def __call__(self, image: np.ndarray) -> np.ndarray:
                raise RuntimeError("boom")

        pipeline = PatchPipeline(
            slide_paths=fake_slide_paths(1),
            backend=FakeBackend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=128, num_patches=1, seed=42),
            transforms=FailingTransform(),
            pool_size=1,
            patches_per_slide=1,
        )
        with pytest.raises(RuntimeError, match="boom"):
            next(iter(pipeline))
        assert pipeline.stats.patches_extracted == 0

    def test_old_positional_pipeline_call_keeps_dataset_adapter_slot(self):
        pipeline = PatchPipeline(
            fake_slide_paths(1),
            FakeBackend(),
            OtsuTissueDetector(),
            RandomSampler(patch_size=128, num_patches=1, seed=42),
            None,
            None,
            _Adapter(),
            (2048, 2048),
            "sequential",
            1,
            1,
            1,
            False,
            "with_replacement",
            42,
        )
        result = next(iter(pipeline))
        assert result.slide_metadata.patient_id == "P001"
        assert result.image is not None

    def test_views_and_transforms_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            PatchPipeline(
                slide_paths=fake_slide_paths(1),
                backend=FakeBackend(),
                tissue_detector=OtsuTissueDetector(),
                sampler=RandomSampler(patch_size=256, num_patches=1),
                transforms=ResizeTransform(224),
                views=[ViewConfig(name="view")],
            )

    def test_empty_views_raise(self):
        with pytest.raises(ValueError, match="at least one"):
            PatchPipeline(
                slide_paths=fake_slide_paths(1),
                backend=FakeBackend(),
                tissue_detector=OtsuTissueDetector(),
                sampler=RandomSampler(patch_size=256, num_patches=1),
                views=[],
            )

    def test_shared_transforms_without_views_raises(self):
        with pytest.raises(ValueError, match="requires views"):
            PatchPipeline(
                slide_paths=fake_slide_paths(1),
                backend=FakeBackend(),
                tissue_detector=OtsuTissueDetector(),
                sampler=RandomSampler(patch_size=256, num_patches=1),
                shared_transforms=_AddValue(1),
            )

    def test_shared_transforms_with_mpp_override_raises(self):
        with pytest.raises(ValueError, match="shared_transforms"):
            PatchPipeline(
                slide_paths=fake_slide_paths(1),
                backend=FakeBackend(),
                tissue_detector=OtsuTissueDetector(),
                sampler=RandomSampler(patch_size=256, num_patches=1),
                views=[ViewConfig(name="context", mpp_override=1.0)],
                shared_transforms=_AddValue(1),
            )

    def test_duplicate_expanded_names_raise(self):
        with pytest.warns(UserWarning, match="identical copies"):
            counted = ViewConfig(name="local", count=2)
        with pytest.raises(ValueError, match="view names"):
            PatchPipeline(
                slide_paths=fake_slide_paths(1),
                backend=FakeBackend(),
                tissue_detector=OtsuTissueDetector(),
                sampler=RandomSampler(patch_size=256, num_patches=1),
                views=[counted, ViewConfig(name="local_0")],
            )

    def test_view_name_colliding_with_reserved_key_raises(self):
        for reserved in ("image", "x", "y", "mpp", "tissue_fraction", "patient_id", "slide_path"):
            with pytest.raises(ValueError, match="reserved batch key"):
                PatchPipeline(
                    slide_paths=fake_slide_paths(1),
                    backend=FakeBackend(),
                    tissue_detector=OtsuTissueDetector(),
                    sampler=RandomSampler(patch_size=256, num_patches=1),
                    views=[ViewConfig(name=reserved)],
                )

    def test_produces_named_views(self):
        pipeline = PatchPipeline(
            slide_paths=fake_slide_paths(1),
            backend=FakeBackend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=256, num_patches=1, seed=42),
            views=[
                ViewConfig(
                    name="global",
                    crop=RandomResizedCrop(size=224, scale=(0.8, 1.0), seed=1),
                ),
                ViewConfig(
                    name="local",
                    crop=RandomResizedCrop(size=96, scale=(0.1, 0.4), seed=2),
                    count=2,
                ),
            ],
            pool_size=1,
            patches_per_slide=1,
        )
        result = next(iter(pipeline))
        assert result.image is None
        assert set(result.views) == {"global", "local_0", "local_1"}
        assert result.views["global"].shape == (224, 224, 3)
        assert result.views["local_0"].shape == (96, 96, 3)

    def test_shared_transform_applies_before_views(self):
        pipeline = PatchPipeline(
            slide_paths=fake_slide_paths(1),
            backend=FakeBackend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=64, num_patches=1, seed=42),
            views=[
                ViewConfig(name="mild", transforms=_AddValue(1)),
                ViewConfig(name="strong", transforms=_AddValue(2)),
            ],
            shared_transforms=_AddValue(10),
            pool_size=1,
            patches_per_slide=1,
        )
        result = next(iter(pipeline))
        assert int(result.views["mild"][0, 0, 0]) == 139
        assert int(result.views["strong"][0, 0, 0]) == 140

    def test_mpp_override_reads_second_region(self):
        _TrackBackend.calls.clear()
        pipeline = PatchPipeline(
            slide_paths=fake_slide_paths(1),
            backend=_TrackBackend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=128, num_patches=1, target_mpp=0.25, seed=42),
            views=[
                ViewConfig(name="detail"),
                ViewConfig(name="context", mpp_override=1.0, patch_size_override=64),
            ],
            pool_size=1,
            patches_per_slide=1,
        )
        result = next(iter(pipeline))
        assert result.views["detail"].shape == (128, 128, 3)
        assert result.views["context"].shape == (64, 64, 3)
        assert len(_TrackBackend.calls) == 2
        assert _TrackBackend.calls[0][2] == 0
        assert _TrackBackend.calls[1][2] == 2

    def test_mpp_override_count_reads_slide_once_not_per_count(self):
        """count > 1 with mpp_override must read the slide exactly once."""
        _TrackBackend.calls.clear()
        with pytest.warns(UserWarning, match="identical copies"):
            ctx_view = ViewConfig(name="ctx", mpp_override=1.0, patch_size_override=64, count=3)
        pipeline = PatchPipeline(
            slide_paths=fake_slide_paths(1),
            backend=_TrackBackend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=128, num_patches=1, target_mpp=0.25, seed=42),
            views=[ctx_view],
            pool_size=1,
            patches_per_slide=1,
        )
        result = next(iter(pipeline))
        assert set(result.views) == {"ctx_0", "ctx_1", "ctx_2"}
        # One primary read + one override read (not three override reads).
        assert len(_TrackBackend.calls) == 2

    def test_mpp_override_requires_mpp_metadata(self):
        pipeline = PatchPipeline(
            slide_paths=fake_slide_paths(1),
            backend=_NoMppBackend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=128, num_patches=1, seed=42),
            views=[ViewConfig(name="context", mpp_override=1.0)],
            pool_size=1,
            patches_per_slide=1,
        )
        with pytest.raises(ValueError, match="no MPP metadata"):
            next(iter(pipeline))

    def test_mpp_override_exceeding_slide_dimensions_warns(self):
        # _SmallSlideBackend is 128x128; mpp_override=1.0 at patch_size=128
        # yields a view_l0 = 128 * 4.0 = 512 px > 128 px slide width.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline = PatchPipeline(
                slide_paths=fake_slide_paths(1),
                backend=_SmallSlideBackend(),
                tissue_detector=OtsuTissueDetector(),
                sampler=RandomSampler(patch_size=32, num_patches=1, target_mpp=0.25, seed=42),
                views=[ViewConfig(name="ctx", mpp_override=1.0, patch_size_override=128)],
                pool_size=1,
                patches_per_slide=1,
            )
            next(iter(pipeline))
        assert any(
            "extends outside slide dimensions" in str(w.message) for w in caught
        ), "Expected a UserWarning about same-center read extending outside slide dimensions"

    def test_mpp_override_preserves_center_near_border(self):
        _PaddingTrackBackend.calls.clear()
        coord = PatchCoordinate(
            x=0,
            y=0,
            level=0,
            patch_size=64,
            mpp=0.25,
            slide_path=fake_slide_paths(1)[0],
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline = PatchPipeline(
                slide_paths=fake_slide_paths(1),
                backend=_PaddingTrackBackend(),
                tissue_detector=OtsuTissueDetector(),
                sampler=_FixedSampler(coord),
                views=[ViewConfig(name="ctx", mpp_override=1.0, patch_size_override=64)],
                pool_size=1,
                patches_per_slide=1,
            )
            next(iter(pipeline))

        # Primary read at (0, 0), override read centered on the same level-0
        # center.  At level 2, patch_size=64 spans 256 level-0 pixels, so the
        # same-center origin is 32 - 128 = -96.  Clamping would have changed
        # this to 0 and shifted the biological center.
        assert _PaddingTrackBackend.calls[0][:3] == (0, 0, 0)
        assert _PaddingTrackBackend.calls[1][:3] == (-96, -96, 2)
        assert any("extends outside slide dimensions" in str(w.message) for w in caught)

    def test_successive_patches_get_different_random_crops(self):
        crop = _RecordingCrop(size=96, scale=(0.05, 0.5))
        pipeline = PatchPipeline(
            slide_paths=fake_slide_paths(1),
            backend=FakeBackend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=0),
            views=[
                ViewConfig(
                    name="v",
                    crop=crop,
                ),
            ],
            pool_size=1,
            patches_per_slide=4,
            cycle=False,
        )
        results = list(pipeline)
        assert len(results) == 4
        assert len(crop.params) == 4
        assert len(set(crop.params)) > 1


class TestWsiStreamDatasetViews:
    def test_single_view_dataset_item_keys_match_existing_order(self):
        dataset = WsiStreamDataset(
            fake_slide_paths(1),
            FakeBackend(),
            OtsuTissueDetector(),
            RandomSampler(patch_size=64, num_patches=1, seed=42),
            None,
            None,
            _Adapter(),
            1,
            1,
            1,
            False,
            "with_replacement",
            "sequential",
            42,
        )
        item = next(iter(dataset))
        assert list(item) == [
            "slide_path",
            "dataset_name",
            "patient_id",
            "tissue_type",
            "cancer_type",
            "sample_type",
            "extra",
            "image",
            "x",
            "y",
            "level",
            "patch_size",
            "mpp",
            "tissue_fraction",
        ]
        assert item["image"].shape == (3, 64, 64)

    def test_old_positional_dataset_call_keeps_dataset_adapter_slot(self):
        dataset = WsiStreamDataset(
            fake_slide_paths(1),
            FakeBackend(),
            OtsuTissueDetector(),
            RandomSampler(patch_size=128, num_patches=1, seed=42),
            None,
            None,
            _Adapter(),
            1,
            1,
            1,
            False,
            "with_replacement",
            "sequential",
            42,
        )
        item = next(iter(dataset))
        assert item["image"].shape == (3, 128, 128)
        assert item["patient_id"] == "P001"

    def test_dataloader_collates_named_views(self):
        dataset = WsiStreamDataset(
            slide_paths=fake_slide_paths(2),
            backend=FakeBackend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=128, num_patches=-1, seed=42),
            views=[
                ViewConfig(
                    name="global",
                    transforms=ComposeTransforms([ResizeTransform(64)]),
                ),
                ViewConfig(
                    name="local",
                    crop=RandomResizedCrop(size=32, scale=(0.2, 0.5), seed=1),
                    count=2,
                ),
            ],
            pool_size=1,
            patches_per_slide=2,
            cycle=False,
        )
        batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
        assert "image" not in batch
        assert batch["global"].shape == (2, 3, 64, 64)
        assert batch["local_0"].shape == (2, 3, 32, 32)
        assert batch["local_1"].shape == (2, 3, 32, 32)
        assert batch["global"].dtype == torch.float32
        assert "x" in batch and "slide_path" in batch

    def test_views_and_transforms_mutually_exclusive_at_construction(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            WsiStreamDataset(
                slide_paths=fake_slide_paths(1),
                backend=FakeBackend(),
                tissue_detector=OtsuTissueDetector(),
                sampler=RandomSampler(patch_size=64, num_patches=1),
                transforms=ResizeTransform(32),
                views=[ViewConfig(name="v", crop=RandomResizedCrop(size=32, scale=(0.5, 1.0)))],
            )

    def test_shared_transforms_without_views_raises_at_construction(self):
        from wsistream.transforms import RandomFlipRotate

        with pytest.raises(ValueError, match="requires views"):
            WsiStreamDataset(
                slide_paths=fake_slide_paths(1),
                backend=FakeBackend(),
                tissue_detector=OtsuTissueDetector(),
                sampler=RandomSampler(patch_size=64, num_patches=1),
                shared_transforms=RandomFlipRotate(),
            )

    def test_expand_view_names_importable_from_package_root(self):
        import wsistream

        assert hasattr(wsistream, "expand_view_names")
        with pytest.warns(UserWarning, match="identical copies"):
            local_view = wsistream.ViewConfig(name="local", count=2)
        names = wsistream.expand_view_names([wsistream.ViewConfig(name="g"), local_view])
        assert names == ["g", "local_0", "local_1"]


class TestMppOverrideFallback:
    def test_mpp_override_without_patch_size_override_uses_primary_patch_size(self):
        """When patch_size_override is omitted, the primary patch_size is reused."""
        _TrackBackend.calls.clear()
        primary_patch_size = 64
        pipeline = PatchPipeline(
            slide_paths=fake_slide_paths(1),
            backend=_TrackBackend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(
                patch_size=primary_patch_size, num_patches=1, target_mpp=0.25, seed=42
            ),
            views=[
                ViewConfig(name="detail"),
                ViewConfig(name="context", mpp_override=1.0),  # no patch_size_override
            ],
            pool_size=1,
            patches_per_slide=1,
        )
        result = next(iter(pipeline))
        # context view falls back to primary patch_size
        assert result.views["context"].shape == (primary_patch_size, primary_patch_size, 3)
        # Two reads: one primary, one override
        assert len(_TrackBackend.calls) == 2
        # The override read used primary_patch_size as width/height
        _, _, _, override_w, override_h = _TrackBackend.calls[1]
        assert override_w == primary_patch_size
        assert override_h == primary_patch_size

    def test_mpp_override_count_with_crop_produces_distinct_crops(self):
        """count > 1 with mpp_override + crop must advance the RNG independently per iteration."""
        _TrackBackend.calls.clear()
        crop = _RecordingCrop(size=32, scale=(0.2, 0.8))
        pipeline = PatchPipeline(
            slide_paths=fake_slide_paths(1),
            backend=_TrackBackend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=128, num_patches=1, target_mpp=0.25, seed=42),
            views=[
                ViewConfig(
                    name="ctx",
                    mpp_override=1.0,
                    patch_size_override=64,
                    crop=crop,
                    count=3,
                )
            ],
            pool_size=1,
            patches_per_slide=1,
        )
        result = next(iter(pipeline))
        assert set(result.views) == {"ctx_0", "ctx_1", "ctx_2"}
        # Slide is read exactly once for the override (not once per count).
        assert len(_TrackBackend.calls) == 2
        # Each iteration draws a different random crop from the same override image.
        assert len(crop.params) == 3
        assert len(set(crop.params)) > 1

    def test_count_without_crop_or_transforms_warns(self):
        with pytest.warns(UserWarning, match="identical copies"):
            ViewConfig(name="v", count=3)
