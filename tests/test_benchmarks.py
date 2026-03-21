"""
Performance benchmarks against real WSI files.

Skipped automatically when ``--slide-dir`` is not provided.

Run:
    pytest tests/test_benchmarks.py --slide-dir /path/to/slides --backend openslide -v -s
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset

from wsistream.filters import HSVPatchFilter
from wsistream.pipeline import PatchPipeline
from wsistream.sampling import RandomSampler
from wsistream.tissue import HSVTissueDetector, OtsuTissueDetector
from wsistream.transforms import (
    ComposeTransforms,
    HEDColorAugmentation,
    RandomFlipRotate,
    ResizeTransform,
)

pytestmark = pytest.mark.benchmark

NUM_PATCHES = 100


def _run_pipeline(slides, make_backend, transforms=None) -> tuple[int, float]:
    """Return (count, elapsed_seconds)."""
    pipeline = PatchPipeline(
        slide_paths=slides,
        backend=make_backend(),
        tissue_detector=HSVTissueDetector(),
        sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
        transforms=transforms,
        pool_size=min(4, len(slides)),
        patches_per_slide=max(NUM_PATCHES // len(slides), 10),
        cycle=True,
    )

    t0 = time.perf_counter()
    count = 0
    for _ in pipeline:
        count += 1
        if count >= NUM_PATCHES:
            break
    elapsed = time.perf_counter() - t0
    return count, elapsed


class TestThroughput:
    """Measure patches/sec under different transform configurations."""

    def test_no_transforms(self, slides, make_backend):
        count, elapsed = _run_pipeline(slides, make_backend, transforms=None)
        rate = count / elapsed
        print(f"\n  no transforms: {count} patches in {elapsed:.2f}s " f"-> {rate:.1f} patches/sec")
        assert count == NUM_PATCHES

    def test_resize_only(self, slides, make_backend):
        transforms = ComposeTransforms([ResizeTransform(target_size=224)])
        count, elapsed = _run_pipeline(slides, make_backend, transforms=transforms)
        rate = count / elapsed
        print(f"\n  resize only:   {count} patches in {elapsed:.2f}s " f"-> {rate:.1f} patches/sec")
        assert count == NUM_PATCHES

    def test_full_midnight(self, slides, make_backend):
        transforms = ComposeTransforms(
            [
                HEDColorAugmentation(sigma=0.05),
                RandomFlipRotate(),
                ResizeTransform(target_size=224),
            ]
        )
        count, elapsed = _run_pipeline(slides, make_backend, transforms=transforms)
        rate = count / elapsed
        print(f"\n  full Midnight: {count} patches in {elapsed:.2f}s " f"-> {rate:.1f} patches/sec")
        assert count == NUM_PATCHES

    def test_with_hsv_filter(self, slides, make_backend):
        """Measure throughput with HSV patch filter enabled."""
        pipeline = PatchPipeline(
            slide_paths=slides,
            backend=make_backend(),
            tissue_detector=HSVTissueDetector(),
            patch_filter=HSVPatchFilter(min_pixel_fraction=0.6),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            transforms=ComposeTransforms(
                [
                    HEDColorAugmentation(sigma=0.05),
                    RandomFlipRotate(),
                    ResizeTransform(target_size=224),
                ]
            ),
            pool_size=min(4, len(slides)),
            patches_per_slide=max(NUM_PATCHES // len(slides), 10),
            cycle=True,
        )

        t0 = time.perf_counter()
        count = 0
        for _ in pipeline:
            count += 1
            if count >= NUM_PATCHES:
                break
        elapsed = time.perf_counter() - t0
        rate = count / elapsed
        filtered = pipeline.stats.patches_filtered
        print(
            f"\n  with HSV filter: {count} patches in {elapsed:.2f}s "
            f"-> {rate:.1f} patches/sec ({filtered} filtered)"
        )
        assert count == NUM_PATCHES


class TestTissueDetectionLatency:
    """Measure how long tissue detection takes per slide."""

    def test_otsu_latency(self, slides, make_backend):
        from wsistream.slide import SlideHandle

        detector = OtsuTissueDetector()
        times = []
        for path in slides[:5]:
            backend = make_backend()
            with SlideHandle(path, backend=backend) as slide:
                thumb = slide.get_thumbnail((2048, 2048))
                t0 = time.perf_counter()
                detector.detect(thumb)
                times.append(time.perf_counter() - t0)

        avg_ms = np.mean(times) * 1000
        print(f"\n  Otsu tissue detection: {avg_ms:.1f} ms/slide (n={len(times)})")
        assert avg_ms < 5000, f"Tissue detection too slow: {avg_ms:.0f}ms"

    def test_hsv_latency(self, slides, make_backend):
        from wsistream.slide import SlideHandle

        detector = HSVTissueDetector()
        times = []
        for path in slides[:5]:
            backend = make_backend()
            with SlideHandle(path, backend=backend) as slide:
                thumb = slide.get_thumbnail((2048, 2048))
                t0 = time.perf_counter()
                detector.detect(thumb)
                times.append(time.perf_counter() - t0)

        avg_ms = np.mean(times) * 1000
        print(f"\n  HSV tissue detection: {avg_ms:.1f} ms/slide (n={len(times)})")
        assert avg_ms < 5000, f"Tissue detection too slow: {avg_ms:.0f}ms"


class TestSlideOpenLatency:
    """Measure how long it takes to open a slide and read a thumbnail."""

    def test_open_and_thumbnail(self, slides, make_backend):
        from wsistream.slide import SlideHandle

        times = []
        for path in slides[:5]:
            backend = make_backend()
            t0 = time.perf_counter()
            with SlideHandle(path, backend=backend) as slide:
                _ = slide.get_thumbnail((2048, 2048))
            times.append(time.perf_counter() - t0)

        avg_ms = np.mean(times) * 1000
        print(f"\n  Open + thumbnail: {avg_ms:.1f} ms/slide (n={len(times)})")
        assert avg_ms < 10000, f"Slide open too slow: {avg_ms:.0f}ms"

    def test_read_region_latency(self, slides, make_backend):
        from wsistream.slide import SlideHandle

        backend = make_backend()
        with SlideHandle(slides[0], backend=backend) as slide:
            props = slide.properties
            rng = np.random.default_rng(42)

            times = []
            for _ in range(20):
                x = int(rng.integers(0, max(1, props.width - 256)))
                y = int(rng.integers(0, max(1, props.height - 256)))
                t0 = time.perf_counter()
                _ = slide.read_region(x=x, y=y, width=256, height=256, level=0)
                times.append(time.perf_counter() - t0)

        avg_ms = np.mean(times) * 1000
        p95_ms = np.percentile(times, 95) * 1000
        print(f"\n  read_region 256x256: avg={avg_ms:.1f}ms, p95={p95_ms:.1f}ms (n={len(times)})")
        assert avg_ms < 5000, f"read_region too slow: {avg_ms:.0f}ms"


def _make_backend(bname: str):
    if bname == "openslide":
        from wsistream.backends import OpenSlideBackend

        return OpenSlideBackend()
    else:
        from wsistream.backends import TiffSlideBackend

        return TiffSlideBackend()


class _BenchmarkDataset(IterableDataset):
    """Module-level so it's picklable for multi-worker DataLoader."""

    def __init__(self, slide_paths: list[str], bname: str, partition: bool = False):
        self.slide_paths = slide_paths
        self.bname = bname
        self.partition = partition

    def __iter__(self):
        slides = self.slide_paths
        if self.partition:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                slides = slides[worker_info.id :: worker_info.num_workers]
            if not slides:
                return

        pipe = PatchPipeline(
            slide_paths=slides,
            backend=_make_backend(self.bname),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            transforms=ComposeTransforms([ResizeTransform(target_size=224)]),
            pool_size=min(2, len(slides)),
            patches_per_slide=5,
            cycle=False,
        )
        for result in pipe:
            img = np.ascontiguousarray(result.image)
            yield torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


class TestDataLoaderIntegration:
    """Verify patches flow through a real PyTorch DataLoader."""

    def test_single_worker(self, slides, backend_name):
        ds = _BenchmarkDataset(slides, backend_name, partition=False)
        loader = DataLoader(ds, batch_size=4, num_workers=0)

        batches = 0
        for batch in loader:
            assert batch.shape[1:] == (3, 224, 224)
            assert batch.dtype == torch.float32
            batches += 1
            if batches >= 5:
                break

        assert batches > 0

    def test_multi_worker(self, slides, backend_name):
        ds = _BenchmarkDataset(slides, backend_name, partition=True)
        loader = DataLoader(ds, batch_size=4, num_workers=2)

        batches = 0
        for batch in loader:
            assert batch.shape[1:] == (3, 224, 224)
            batches += 1
            if batches >= 5:
                break

        assert batches > 0
