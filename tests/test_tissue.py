"""Tests for tissue detection strategies."""

import numpy as np
import pytest

from wsistream.tissue import (
    CLAMTissueDetector,
    CombinedTissueDetector,
    HSVTissueDetector,
    OtsuTissueDetector,
)


class TestOtsuDetector:
    def test_detects_dark_region(self, white_thumbnail):
        mask = OtsuTissueDetector().detect(white_thumbnail)
        assert mask.shape == (100, 100)
        assert mask.dtype == bool
        assert mask[50, 50]  # center is tissue
        assert not mask[5, 5]  # corner is background

    def test_all_white_has_no_tissue(self):
        thumb = np.ones((100, 100, 3), dtype=np.uint8) * 255
        mask = OtsuTissueDetector().detect(thumb)
        assert mask.sum() == 0

    def test_custom_blur_ksize(self, white_thumbnail):
        mask = OtsuTissueDetector(blur_ksize=3).detect(white_thumbnail)
        assert mask.shape == (100, 100)
        assert mask[50, 50]

    def test_no_morph_close(self, white_thumbnail):
        mask = OtsuTissueDetector(morph_close_ksize=0).detect(white_thumbnail)
        assert mask[50, 50]

    def test_no_min_area_filter(self, white_thumbnail):
        mask = OtsuTissueDetector(min_area_ratio=0).detect(white_thumbnail)
        assert mask[50, 50]


class TestHSVDetector:
    def test_detects_colored_region(self):
        thumb = np.ones((100, 100, 3), dtype=np.uint8) * 240
        thumb[30:70, 30:70] = [180, 120, 180]
        mask = HSVTissueDetector().detect(thumb)
        assert mask.shape == (100, 100)
        assert mask.dtype == bool

    def test_custom_ranges(self):
        thumb = np.ones((50, 50, 3), dtype=np.uint8) * 240
        thumb[10:40, 10:40] = [180, 120, 180]
        mask = HSVTissueDetector(hue_range=(0, 179), sat_range=(1, 255), val_range=(1, 255)).detect(
            thumb
        )
        assert mask.dtype == bool


class TestCLAMDetector:
    def test_detects_tissue(self, white_thumbnail):
        mask = CLAMTissueDetector().detect(white_thumbnail)
        assert mask.shape == (100, 100)
        assert mask.dtype == bool

    def test_all_white_has_no_tissue(self):
        thumb = np.ones((100, 100, 3), dtype=np.uint8) * 255
        mask = CLAMTissueDetector().detect(thumb)
        assert mask.sum() == 0

    def test_accepts_downsample(self, white_thumbnail):
        mask = CLAMTissueDetector().detect(white_thumbnail, downsample=(4.0, 4.0))
        assert mask.shape == (100, 100)
        assert mask.dtype == bool

    def test_custom_params(self, white_thumbnail):
        det = CLAMTissueDetector(sthresh=20, close=0, mthresh=5)
        mask = det.detect(white_thumbnail)
        assert mask.shape == (100, 100)


class TestCombinedDetector:
    def test_intersection(self, white_thumbnail):
        combined = CombinedTissueDetector(detectors=[OtsuTissueDetector(), OtsuTissueDetector()])
        mask = combined.detect(white_thumbnail)
        assert mask[50, 50]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            CombinedTissueDetector(detectors=[]).detect(np.zeros((10, 10, 3), dtype=np.uint8))

    def test_repr(self):
        combined = CombinedTissueDetector(detectors=[OtsuTissueDetector(), HSVTissueDetector()])
        r = repr(combined)
        assert "OtsuTissueDetector" in r
        assert "HSVTissueDetector" in r
