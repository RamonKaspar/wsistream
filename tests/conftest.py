"""Shared fixtures and CLI options for wsistream tests."""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pytest

from wsistream.backends.base import SlideBackend
from wsistream.types import SlideProperties, TissueMask


# ── pytest CLI options ──


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--slide-dir",
        type=str,
        default=None,
        help="Directory containing WSI files (.svs/.tiff/.ndpi) for e2e tests.",
    )
    parser.addoption(
        "--backend",
        type=str,
        default="openslide",
        choices=["openslide", "tiffslide"],
        help="WSI backend for e2e tests (default: openslide).",
    )


# ── e2e / benchmark fixtures ──


def _find_slides(slide_dir: str) -> list[str]:
    slides = []
    for ext in ("*.svs", "*.tiff", "*.tif", "*.ndpi"):
        slides.extend(glob.glob(str(Path(slide_dir) / ext)))
    return sorted(slides)


@pytest.fixture(scope="session")
def slide_dir(request: pytest.FixtureRequest) -> str:
    path = request.config.getoption("--slide-dir")
    if path is None:
        pytest.skip("No --slide-dir provided; skipping e2e tests.")
    return path


@pytest.fixture(scope="session")
def slides(slide_dir: str) -> list[str]:
    found = _find_slides(slide_dir)
    if not found:
        pytest.skip(f"No WSI files in {slide_dir}")
    return found


@pytest.fixture(scope="session")
def backend_name(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--backend")


@pytest.fixture
def make_backend(backend_name: str):
    """Factory that returns a fresh backend instance each call."""

    def _make():
        if backend_name == "openslide":
            from wsistream.backends import OpenSlideBackend

            return OpenSlideBackend()
        elif backend_name == "tiffslide":
            from wsistream.backends import TiffSlideBackend

            return TiffSlideBackend()
        raise ValueError(f"Unknown backend: {backend_name}")

    return _make


# ── fake backend for unit tests (no WSI files needed) ──


class FakeBackend(SlideBackend):
    """In-memory backend that returns synthetic data.

    Produces a thumbnail with a dark center (tissue-like) and white
    surround so that tissue detectors produce a usable mask.

    Accepts an optional ``token`` kwarg to verify that deepcopy
    preserves constructor configuration.
    """

    def __init__(self, *, token: str = "default") -> None:
        self.token = token
        self._path: str | None = None
        self._opened = False
        self._closed = False

    def open(self, path: str) -> None:
        self._path = path
        self._opened = True
        self._closed = False

    def close(self) -> None:
        self._closed = True

    def read_region(
        self, x: int, y: int, level: int, width: int, height: int
    ) -> np.ndarray:
        return np.full((height, width, 3), 128, dtype=np.uint8)

    def get_thumbnail(self, size: tuple[int, int]) -> np.ndarray:
        w, h = size
        thumb = np.ones((h, w, 3), dtype=np.uint8) * 240
        # Dark center square → tissue region
        cy, cx = h // 2, w // 2
        r = min(h, w) // 4
        thumb[cy - r : cy + r, cx - r : cx + r] = 80
        return thumb

    def get_properties(self) -> SlideProperties:
        return SlideProperties(
            path=self._path or "fake.svs",
            dimensions=(4096, 4096),
            level_count=3,
            level_dimensions=((4096, 4096), (2048, 2048), (1024, 1024)),
            level_downsamples=(1.0, 2.0, 4.0),
            mpp=0.25,
            vendor="fake",
        )

    def __repr__(self) -> str:
        return f"FakeBackend(token={self.token!r})"


@pytest.fixture
def fake_backend():
    """Return a fresh FakeBackend instance."""
    return FakeBackend()


# ── unit test fixtures (no slides needed) ──


@pytest.fixture
def white_thumbnail():
    """100x100 mostly-white thumbnail with a dark square in the center."""
    thumb = np.ones((100, 100, 3), dtype=np.uint8) * 240
    thumb[30:70, 30:70] = 80
    return thumb


@pytest.fixture
def sample_mask():
    """100x100 boolean mask with True in center square."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 30:70] = True
    return mask


@pytest.fixture
def sample_tissue_mask(sample_mask):
    return TissueMask(mask=sample_mask, downsample=10.0, slide_dimensions=(1000, 1000))


@pytest.fixture
def sample_properties():
    return SlideProperties(
        path="test.svs",
        dimensions=(10000, 10000),
        level_count=3,
        level_dimensions=((10000, 10000), (5000, 5000), (2500, 2500)),
        level_downsamples=(1.0, 2.0, 4.0),
        mpp=0.25,
        vendor="aperio",
    )


@pytest.fixture
def random_patch():
    """Random 64x64 RGB uint8 patch."""
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def random_patch_256():
    """Random 256x256 RGB uint8 patch."""
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
