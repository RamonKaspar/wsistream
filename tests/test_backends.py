"""Tests shared by the built-in slide backends."""

from __future__ import annotations

import numpy as np
import pytest

from wsistream.backends import OpenSlideBackend, SlideBackend, TiffSlideBackend

BACKEND_CLASSES = (OpenSlideBackend, TiffSlideBackend)


class _NativeSlide:
    level_downsamples = (1.0,)

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def read_region(self, location, level, size, **kwargs):
        width, height = size
        return np.full((height, width, 4), 128, dtype=np.uint8)

    def get_thumbnail(self, size):
        width, height = size
        return np.full((height, width, 4), 128, dtype=np.uint8)


@pytest.mark.parametrize("backend_class", BACKEND_CLASSES)
class TestBackendLifecycle:
    @pytest.mark.parametrize(
        ("method_name", "args"),
        (
            ("read_region", (0, 0, 0, 1, 1)),
            ("get_thumbnail", ((1, 1),)),
            ("get_properties", ()),
        ),
    )
    def test_use_before_open_raises_clear_error(self, backend_class, method_name, args):
        backend = backend_class()

        with pytest.raises(RuntimeError, match=r"not open; call open\(\) first"):
            getattr(backend, method_name)(*args)

    def test_use_after_close_raises_clear_error(self, backend_class):
        backend = backend_class()
        native_slide = _NativeSlide()
        backend._slide = native_slide
        backend.close()

        assert native_slide.closed
        with pytest.raises(RuntimeError, match=r"not open; call open\(\) first"):
            backend.get_thumbnail((1, 1))


class TestTiffSlideStorageOptions:
    @staticmethod
    def _record_opens(monkeypatch) -> list[tuple[str, object]]:
        import tiffslide

        opened = []

        def fake_tiffslide(path, *, storage_options=None):
            opened.append((path, storage_options))
            return _NativeSlide()

        monkeypatch.setattr(tiffslide, "TiffSlide", fake_tiffslide)
        return opened

    def test_storage_options_reach_tiffslide(self, monkeypatch):
        opened = self._record_opens(monkeypatch)

        TiffSlideBackend(storage_options={"anon": True}).open("s3://bucket/slide.svs")

        assert opened == [("s3://bucket/slide.svs", {"anon": True})]

    def test_default_passes_no_storage_options(self, monkeypatch):
        opened = self._record_opens(monkeypatch)

        TiffSlideBackend().open("slide.svs")

        assert opened == [("slide.svs", None)]

    def test_deep_copy_keeps_storage_options(self, monkeypatch):
        """The pipeline opens every slide with a deep copy of the backend."""
        import copy

        opened = self._record_opens(monkeypatch)
        backend = TiffSlideBackend(storage_options={"anon": True})

        copy.deepcopy(backend).open("s3://bucket/slide.svs")

        assert opened == [("s3://bucket/slide.svs", {"anon": True})]

    def test_repr_hides_option_values(self):
        backend = TiffSlideBackend(storage_options={"secret": "abc123", "key": "AKIA"})

        assert repr(backend) == "TiffSlideBackend(storage_options={'key': ..., 'secret': ...})"
        assert repr(TiffSlideBackend()) == "TiffSlideBackend()"


@pytest.mark.parametrize("backend_class", BACKEND_CLASSES)
class TestBackendImageConversion:
    def test_read_region_discards_alpha_channel(self, backend_class):
        backend = backend_class()
        backend._slide = _NativeSlide()

        region = backend.read_region(0, 0, 0, 8, 6)

        assert region.shape == (6, 8, 3)

    def test_thumbnail_discards_alpha_channel(self, backend_class):
        backend = backend_class()
        backend._slide = _NativeSlide()

        thumbnail = backend.get_thumbnail((8, 6))

        assert thumbnail.shape == (6, 8, 3)


class TestAlphaCompositing:
    def test_opaque_pixels_are_unchanged(self):
        rgba = np.random.default_rng(0).integers(0, 256, (4, 5, 4), dtype=np.uint8)
        rgba[:, :, 3] = 255

        np.testing.assert_array_equal(SlideBackend._to_rgb_array(rgba), rgba[:, :, :3])

    def test_opaque_pixels_are_returned_writable_and_contiguous(self):
        """In-place transforms must work on OpenSlide reads as on TiffSlide reads."""
        rgba = np.random.default_rng(0).integers(0, 256, (4, 5, 4), dtype=np.uint8)
        rgba[:, :, 3] = 255
        rgba.flags.writeable = False  # like np.asarray of a PIL image

        rgb = SlideBackend._to_rgb_array(rgba)

        assert rgb.flags.writeable
        assert rgb.flags.c_contiguous

    def test_transparent_pixels_become_the_background(self):
        rgba = np.zeros((4, 5, 4), dtype=np.uint8)

        white = SlideBackend._to_rgb_array(rgba)
        colored = SlideBackend._to_rgb_array(rgba, (16, 32, 48))

        np.testing.assert_array_equal(white, np.full((4, 5, 3), 255))
        np.testing.assert_array_equal(colored, np.broadcast_to([16, 32, 48], (4, 5, 3)))

    def test_partial_alpha_is_blended_over_the_background(self):
        rgba = np.array([[[0, 100, 200, 128]]], dtype=np.uint8)

        blended = SlideBackend._to_rgb_array(rgba)

        expected = np.rint((np.array([0, 100, 200]) * 128 + 255 * 127) / 255)
        np.testing.assert_array_equal(blended[0, 0], expected)

    def test_openslide_uses_the_slide_background_color(self, monkeypatch):
        import openslide

        class _TransparentSlide(_NativeSlide):
            properties = {"openslide.background-color": "102030"}

            def read_region(self, location, level, size, **kwargs):
                width, height = size
                return np.zeros((height, width, 4), dtype=np.uint8)

        monkeypatch.setattr(openslide, "OpenSlide", lambda path: _TransparentSlide())
        backend = OpenSlideBackend()
        backend.open("slide.mrxs")

        region = backend.read_region(0, 0, 0, 3, 2)

        np.testing.assert_array_equal(region, np.broadcast_to([16, 32, 48], (2, 3, 3)))


class _ArrayTiffSlide:
    """TiffSlide stand-in serving in-memory levels with TiffSlide's unpadded reads."""

    def __init__(self, levels: list[np.ndarray], level_downsamples: tuple[float, ...]) -> None:
        self.levels = levels
        self.level_downsamples = level_downsamples

    def read_region(self, location, level, size, *, as_array=False, padding=True):
        assert as_array and not padding
        image = self.levels[level]
        level_h, level_w = image.shape[:2]
        rx0 = int(location[0] / self.level_downsamples[level])
        ry0 = int(location[1] / self.level_downsamples[level])
        rx1, ry1 = rx0 + size[0], ry0 + size[1]
        rx0, rx1 = np.clip([rx0, rx1], 0, level_w)
        ry0, ry1 = np.clip([ry0, ry1], 0, level_h)
        return image[ry0:ry1, rx0:rx1]


class TestTiffSlidePadding:
    @staticmethod
    def _backend() -> tuple[TiffSlideBackend, list[np.ndarray]]:
        rng = np.random.default_rng(0)
        levels = [
            rng.integers(0, 200, (40, 48, 3), dtype=np.uint8),
            rng.integers(0, 200, (10, 12, 3), dtype=np.uint8),
        ]
        backend = TiffSlideBackend()
        backend._slide = _ArrayTiffSlide(levels, (1.0, 4.0))
        return backend, levels

    def test_in_bounds_region_is_unchanged(self):
        backend, levels = self._backend()

        region = backend.read_region(8, 4, 0, 16, 12)

        np.testing.assert_array_equal(region, levels[0][4:16, 8:24])

    def test_region_before_the_slide_origin_is_white(self):
        backend, levels = self._backend()

        region = backend.read_region(-3, -2, 0, 6, 5)

        assert (region[:2] == 255).all() and (region[:, :3] == 255).all()
        np.testing.assert_array_equal(region[2:, 3:], levels[0][0:3, 0:3])

    def test_region_past_the_slide_end_is_white(self):
        backend, levels = self._backend()

        region = backend.read_region(40, 36, 1, 5, 3)

        # Level 1 starts at pixel (10, 9) of a 12x10 image: 2 columns and 1 row are inside.
        np.testing.assert_array_equal(region[:1, :2], levels[1][9:10, 10:12])
        assert (region[1:] == 255).all() and (region[:, 2:] == 255).all()

    def test_partially_covered_columns_follow_the_level_grid(self):
        """At level 1, x=-6 starts 1.5 pixels left of the slide: 2 columns are white."""
        backend, levels = self._backend()

        region = backend.read_region(-6, 0, 1, 4, 3)

        assert (region[:, :2] == 255).all()
        np.testing.assert_array_equal(region[:, 2:], levels[1][0:3, 0:2])
