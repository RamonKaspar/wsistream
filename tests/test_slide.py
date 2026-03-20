"""Tests for SlideHandle (using FakeBackend, no WSI files needed)."""

from __future__ import annotations

import numpy as np

from wsistream.slide import SlideHandle


class TestSlideHandle:
    def test_opens_and_reads(self, fake_backend):
        slide = SlideHandle("fake.svs", backend=fake_backend)
        patch = slide.read_region(x=0, y=0, width=256, height=256)
        assert patch.shape == (256, 256, 3)
        assert patch.dtype == np.uint8
        slide.close()

    def test_read_region_passes_level(self, fake_backend):
        """Verify level is forwarded (public API has level last, backend has it 3rd)."""
        calls = []
        original = fake_backend.read_region

        def spy(x, y, level, width, height):
            calls.append((x, y, level, width, height))
            return original(x, y, level, width, height)

        fake_backend.read_region = spy

        slide = SlideHandle("fake.svs", backend=fake_backend)
        slide.read_region(x=10, y=20, width=64, height=64, level=2)
        slide.close()

        assert calls == [(10, 20, 2, 64, 64)]

    def test_context_manager(self, fake_backend):
        with SlideHandle("fake.svs", backend=fake_backend) as slide:
            assert slide.properties.dimensions == (4096, 4096)
        assert fake_backend._closed

    def test_thumbnail(self, fake_backend):
        slide = SlideHandle("fake.svs", backend=fake_backend)
        thumb = slide.get_thumbnail((512, 512))
        assert thumb.shape == (512, 512, 3)
        slide.close()

    def test_properties(self, fake_backend):
        slide = SlideHandle("fake.svs", backend=fake_backend)
        p = slide.properties
        assert p.path == "fake.svs"
        assert p.level_count == 3
        assert p.mpp == 0.25
        assert p.width == 4096
        assert p.height == 4096
        slide.close()

    def test_repr(self, fake_backend):
        slide = SlideHandle("fake.svs", backend=fake_backend)
        r = repr(slide)
        assert "fake.svs" in r
        assert "4096x4096" in r
        assert "FakeBackend" in r
        slide.close()


class TestBestLevelForMpp:
    def test_exact_match(self, fake_backend):
        # FakeBackend: mpp=0.25, downsamples=(1.0, 2.0, 4.0)
        # level 0 → 0.25, level 1 → 0.50, level 2 → 1.0
        slide = SlideHandle("fake.svs", backend=fake_backend)
        assert slide.best_level_for_mpp(0.25) == 0
        assert slide.best_level_for_mpp(0.50) == 1
        assert slide.best_level_for_mpp(1.00) == 2
        slide.close()

    def test_closest_match(self, fake_backend):
        slide = SlideHandle("fake.svs", backend=fake_backend)
        # 0.4 is closer to 0.5 (level 1) than 0.25 (level 0)
        assert slide.best_level_for_mpp(0.4) == 1
        # 0.3 is closer to 0.25 (level 0) than 0.5 (level 1)
        assert slide.best_level_for_mpp(0.3) == 0
        slide.close()

    def test_returns_zero_when_no_mpp(self):
        """Slides without MPP metadata should default to level 0."""
        from tests.conftest import FakeBackend

        backend = FakeBackend()
        # Monkey-patch to return mpp=None
        original_get_props = backend.get_properties

        def no_mpp_props():
            props = original_get_props()
            from wsistream.types import SlideProperties

            return SlideProperties(
                path=props.path,
                dimensions=props.dimensions,
                level_count=props.level_count,
                level_dimensions=props.level_dimensions,
                level_downsamples=props.level_downsamples,
                mpp=None,
                vendor=props.vendor,
            )

        backend.get_properties = no_mpp_props
        slide = SlideHandle("fake.svs", backend=backend)
        assert slide.best_level_for_mpp(0.5) == 0
        slide.close()

    def test_very_high_mpp_picks_coarsest(self, fake_backend):
        slide = SlideHandle("fake.svs", backend=fake_backend)
        assert slide.best_level_for_mpp(100.0) == 2
        slide.close()
