"""Tests for core data types."""

from pathlib import Path

import numpy as np
import pytest

from wsistream.types import PatchCoordinate, TissueMask, resolve_slide_paths


class TestResolveSlidePaths:
    def test_single_file(self, tmp_path: Path):
        slide = tmp_path / "slide.svs"
        slide.touch()

        assert resolve_slide_paths(slide) == [str(slide)]

    def test_directory_is_expanded_recursively(self, tmp_path: Path):
        first = tmp_path / "a.svs"
        second = tmp_path / "nested" / "b.tiff"
        ignored = tmp_path / "notes.txt"
        first.touch()
        second.parent.mkdir()
        second.touch()
        ignored.touch()

        assert resolve_slide_paths(tmp_path) == sorted((str(first), str(second)))

    def test_missing_path_raises(self, tmp_path: Path):
        missing = tmp_path / "missing.svs"

        with pytest.raises(FileNotFoundError, match="does not exist"):
            resolve_slide_paths(missing)

    def test_directory_without_slides_raises(self, tmp_path: Path):
        (tmp_path / "notes.txt").touch()

        with pytest.raises(FileNotFoundError, match="No WSI files"):
            resolve_slide_paths(tmp_path)

    def test_uri_is_preserved_for_remote_backends(self):
        uri = "s3://bucket/path/slide.svs"

        assert resolve_slide_paths(uri) == [uri]


class TestTissueMask:
    def test_contains_tissue(self, sample_tissue_mask):
        assert sample_tissue_mask.contains_tissue(300, 300, 100, 100, threshold=0.5)
        assert not sample_tissue_mask.contains_tissue(0, 0, 100, 100, threshold=0.5)

    def test_tissue_fraction_at(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[0:50, 0:50] = True
        tm = TissueMask(mask=mask, downsample=1.0, slide_dimensions=(100, 100))
        assert 0.2 < tm.tissue_fraction_at(0, 0, 100, 100) < 0.3
        assert tm.tissue_fraction_at(0, 0, 50, 50) > 0.9

    def test_tissue_fraction_property(self):
        mask = np.ones((100, 100), dtype=bool)
        tm = TissueMask(mask=mask, downsample=1.0, slide_dimensions=(100, 100))
        assert tm.tissue_fraction == 1.0

    def test_empty_region_returns_zero(self):
        mask = np.zeros((10, 10), dtype=bool)
        tm = TissueMask(mask=mask, downsample=1.0, slide_dimensions=(10, 10))
        assert tm.tissue_fraction_at(0, 0, 10, 10) == 0.0

    def test_out_of_bounds_clamps(self):
        mask = np.ones((10, 10), dtype=bool)
        tm = TissueMask(mask=mask, downsample=1.0, slide_dimensions=(10, 10))
        frac = tm.tissue_fraction_at(8, 8, 100, 100)
        assert 0.0 < frac <= 1.0


class TestSlideProperties:
    def test_mpp_at_level(self, sample_properties):
        assert sample_properties.mpp_at_level(0) == 0.25
        assert sample_properties.mpp_at_level(1) == 0.5
        assert sample_properties.mpp_at_level(2) == 1.0

    def test_mpp_none(self):
        from wsistream.types import SlideProperties

        props = SlideProperties(
            path="test.svs",
            dimensions=(1000, 1000),
            level_count=1,
            level_dimensions=((1000, 1000),),
            level_downsamples=(1.0,),
            mpp=None,
            vendor=None,
        )
        assert props.mpp_at_level(0) is None

    def test_width_height(self, sample_properties):
        assert sample_properties.width == 10000
        assert sample_properties.height == 10000


class TestPatchCoordinate:
    def test_fields(self):
        pc = PatchCoordinate(x=100, y=200, level=0, patch_size=256, mpp=0.5, slide_path="t.svs")
        assert pc.x == 100
        assert pc.y == 200
        assert pc.mpp == 0.5

    def test_frozen(self):
        import pytest

        pc = PatchCoordinate(x=100, y=200, level=0, patch_size=256, mpp=0.5, slide_path="t.svs")
        with pytest.raises(AttributeError):
            pc.x = 999
