"""Tests shared by the built-in slide backends."""

from __future__ import annotations

import numpy as np
import pytest

from wsistream.backends import OpenSlideBackend, TiffSlideBackend

BACKEND_CLASSES = (OpenSlideBackend, TiffSlideBackend)


class _NativeSlide:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def read_region(self, location, level, size):
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
