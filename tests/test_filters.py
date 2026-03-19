"""Tests for post-extraction patch filters."""

import numpy as np
import pytest

from wsistream.filters import HSVPatchFilter, PatchFilter


class TestHSVPatchFilterAccept:
    """Core accept/reject logic."""

    def test_accepts_tissue_like_patch(self):
        """A patch with mostly stained-tissue-like pixels should pass."""
        # H&E-stained tissue: pinkish-purple, maps to high saturation in HSV
        patch = np.full((256, 256, 3), [180, 120, 180], dtype=np.uint8)
        filt = HSVPatchFilter()
        assert filt.accept(patch)

    def test_rejects_white_patch(self):
        """A pure white patch (background) should be rejected."""
        patch = np.full((256, 256, 3), 255, dtype=np.uint8)
        filt = HSVPatchFilter()
        assert not filt.accept(patch)

    def test_rejects_black_patch(self):
        """A pure black patch (pen mark / artifact) should be rejected."""
        patch = np.zeros((256, 256, 3), dtype=np.uint8)
        filt = HSVPatchFilter()
        assert not filt.accept(patch)

    def test_borderline_fraction(self):
        """Exactly at the threshold boundary."""
        # 60% tissue, 40% white
        patch = np.full((100, 100, 3), 255, dtype=np.uint8)
        # Fill 60 rows with tissue-like color
        patch[:60, :] = [180, 120, 180]
        filt = HSVPatchFilter(min_pixel_fraction=0.6)
        assert filt.accept(patch)

    def test_just_below_threshold(self):
        """Just below the threshold should reject."""
        patch = np.full((100, 100, 3), 255, dtype=np.uint8)
        # Fill 59 rows with tissue-like color (59%)
        patch[:59, :] = [180, 120, 180]
        filt = HSVPatchFilter(min_pixel_fraction=0.6)
        assert not filt.accept(patch)


class TestHSVPatchFilterParams:
    """Custom parameter configurations."""

    def test_custom_hue_range(self):
        patch = np.full((64, 64, 3), [180, 120, 180], dtype=np.uint8)
        # Very narrow hue range that excludes the patch
        filt = HSVPatchFilter(hue_range=(0, 10))
        assert not filt.accept(patch)

    def test_custom_saturation_range(self):
        patch = np.full((64, 64, 3), [180, 120, 180], dtype=np.uint8)
        # Very high saturation threshold
        filt = HSVPatchFilter(sat_range=(250, 255))
        assert not filt.accept(patch)

    def test_custom_value_range(self):
        patch = np.full((64, 64, 3), [180, 120, 180], dtype=np.uint8)
        # Very high value threshold
        filt = HSVPatchFilter(val_range=(250, 255))
        assert not filt.accept(patch)

    def test_zero_threshold_accepts_all(self):
        """With min_pixel_fraction=0, everything passes."""
        patch = np.full((64, 64, 3), 255, dtype=np.uint8)
        filt = HSVPatchFilter(min_pixel_fraction=0.0)
        assert filt.accept(patch)

    def test_midnight_defaults(self):
        """Verify Midnight defaults are what we expect."""
        filt = HSVPatchFilter()
        assert filt.hue_range == (90, 180)
        assert filt.sat_range == (8, 255)
        assert filt.val_range == (103, 255)
        assert filt.min_pixel_fraction == 0.6


class TestPatchFilterInterface:
    """The ABC contract."""

    def test_is_abstract(self):
        with pytest.raises(TypeError):
            PatchFilter()

    def test_hsv_is_subclass(self):
        assert issubclass(HSVPatchFilter, PatchFilter)

    def test_repr(self):
        filt = HSVPatchFilter()
        assert "HSVPatchFilter" in repr(filt)
