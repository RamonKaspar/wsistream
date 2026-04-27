"""
End-to-end stress tests for wsistream.torch.

FakeBackend tests run unconditionally (no WSI files needed).
The real-WSI DDP test requires ``--slide-dir``.

Run all:
    pytest tests/test_torch_e2e.py -v

Run with real slides:
    pytest tests/test_torch_e2e.py --slide-dir /path/to/slides --backend openslide -v
"""

from __future__ import annotations

import json
import socket
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as torch_mp
import torch.nn as nn
from torch.utils.data import DataLoader

from tests.conftest import FakeBackend, fake_slide_paths
from wsistream.datasets import TCGAAdapter
from wsistream.datasets.base import DatasetAdapter
from wsistream.filters.base import PatchFilter
from wsistream.sampling import RandomSampler
from wsistream.tissue import OtsuTissueDetector
from wsistream.tissue.base import TissueDetector
from wsistream.torch import WsiStreamDataset, partition_slides_by_rank
from wsistream.types import SlideMetadata, SlideProperties


# ── test backends and filters ──


class _FailBackend(FakeBackend):
    def open(self, path: str) -> None:
        raise RuntimeError("disk on fire")


class _FailOnSpecificSlides(FakeBackend):
    def __init__(self, fail_substrings: list[str], **kwargs):
        super().__init__(**kwargs)
        self._fail_substrings = fail_substrings

    def open(self, path: str) -> None:
        for sub in self._fail_substrings:
            if sub in path:
                raise RuntimeError(f"simulated failure for {path}")
        super().open(path)


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


class _FailDetector(TissueDetector):
    def detect(self, thumbnail, downsample=(1.0, 1.0)):
        raise RuntimeError("detector crashed")


class _RejectAll(PatchFilter):
    def accept(self, patch: np.ndarray) -> bool:
        return False


class _FakeAdapter(DatasetAdapter):
    def parse_metadata(self, slide_path: str) -> SlideMetadata:
        idx = slide_path.split("_")[1].split(".")[0] if "_" in slide_path else "0"
        return SlideMetadata(
            slide_path=slide_path,
            dataset_name="test_dataset",
            patient_id=f"P-{idx}",
            cancer_type="BRCA",
            tissue_type="breast",
            sample_type="primary",
            extra={"source": "fake", "slide": slide_path},
        )


def _make_dataset(n_slides=3, patches_per_slide=5, **kwargs) -> WsiStreamDataset:
    defaults = dict(
        slide_paths=fake_slide_paths(n_slides),
        backend=FakeBackend(),
        tissue_detector=OtsuTissueDetector(),
        sampler=RandomSampler(patch_size=64, num_patches=-1, seed=42),
        pool_size=min(2, max(n_slides, 1)),
        patches_per_slide=patches_per_slide,
        cycle=False,
        seed=123,
    )
    defaults.update(kwargs)
    return WsiStreamDataset(**defaults)


# ── Stats flush correctness (FakeBackend, no WSIs needed) ──


class TestStatsFlushing:
    def test_break_from_infinite_stream(self):
        """The finally block must flush stats when consumer breaks."""
        dataset = _make_dataset(n_slides=2, patches_per_slide=10, cycle=True)
        count = 0
        for _ in dataset:
            count += 1
            if count >= 7:
                break
        stats = dataset.stats_dict()
        assert stats["pipeline/patches_extracted"] >= 7

    def test_all_slides_fail(self):
        """Zero yielded patches, but slides_failed must still be flushed."""
        dataset = _make_dataset(n_slides=3, patches_per_slide=5, backend=_FailBackend())
        list(dataset)
        stats = dataset.stats_dict()
        assert stats["pipeline/slides_failed"] == 3
        assert stats["pipeline/patches_extracted"] == 0
        assert stats["pipeline/slides_processed"] == 0

    def test_all_patches_filtered(self):
        """Zero yielded patches, but patches_filtered must still be flushed."""
        dataset = _make_dataset(n_slides=2, patches_per_slide=10, patch_filter=_RejectAll())
        list(dataset)
        stats = dataset.stats_dict()
        assert stats["pipeline/patches_extracted"] == 0
        assert stats["pipeline/patches_filtered"] == 20

    def test_partial_slide_failure(self):
        dataset = _make_dataset(
            n_slides=4,
            patches_per_slide=3,
            backend=_FailOnSpecificSlides(["slide_1", "slide_3"]),
        )
        list(dataset)
        stats = dataset.stats_dict()
        assert stats["pipeline/slides_failed"] == 2
        assert stats["pipeline/slides_processed"] == 2
        assert stats["pipeline/patches_extracted"] == 6

    def test_tissue_detection_failure(self):
        dataset = _make_dataset(n_slides=2, patches_per_slide=5, tissue_detector=_FailDetector())
        list(dataset)
        stats = dataset.stats_dict()
        assert stats["pipeline/slides_failed"] == 2
        assert stats["pipeline/patches_extracted"] == 0


# ── Collation correctness (FakeBackend) ──


class TestCollationStress:
    def test_no_mpp_sentinel_collates(self):
        dataset = _make_dataset(n_slides=2, patches_per_slide=4, backend=_NoMppBackend())
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        batch = next(iter(loader))
        assert (batch["mpp"] == -1.0).all()

    def test_consistent_keys_with_adapter(self):
        dataset = _make_dataset(n_slides=3, patches_per_slide=5, dataset_adapter=_FakeAdapter())
        items = list(dataset)
        keys_set = {frozenset(item.keys()) for item in items}
        assert len(keys_set) == 1

    def test_consistent_keys_without_adapter(self):
        dataset = _make_dataset(n_slides=3, patches_per_slide=5, dataset_adapter=None)
        items = list(dataset)
        keys_set = {frozenset(item.keys()) for item in items}
        assert len(keys_set) == 1

    def test_adapter_extra_deserializes(self):
        dataset = _make_dataset(n_slides=2, patches_per_slide=4, dataset_adapter=_FakeAdapter())
        loader = DataLoader(dataset, batch_size=4, num_workers=0)
        batch = next(iter(loader))
        for extra_str in batch["extra"]:
            parsed = json.loads(extra_str)
            assert parsed["source"] == "fake"


# ── Multi-worker stress (FakeBackend) ──


class TestMultiWorkerStress:
    def test_more_workers_than_slides(self):
        dataset = _make_dataset(n_slides=2, patches_per_slide=3)
        loader = DataLoader(dataset, batch_size=1, num_workers=4)
        total = sum(b["image"].shape[0] for b in loader)
        assert total == 6

    def test_all_slides_fail_multi_worker(self):
        dataset = _make_dataset(n_slides=4, patches_per_slide=5, backend=_FailBackend())
        loader = DataLoader(dataset, batch_size=2, num_workers=2)
        total = sum(b["image"].shape[0] for b in loader)
        assert total == 0
        assert dataset.stats_dict()["pipeline/slides_failed"] == 4

    def test_all_filtered_multi_worker(self):
        dataset = _make_dataset(n_slides=4, patches_per_slide=10, patch_filter=_RejectAll())
        loader = DataLoader(dataset, batch_size=2, num_workers=2)
        total = sum(b["image"].shape[0] for b in loader)
        assert total == 0
        assert dataset.stats_dict()["pipeline/patches_filtered"] == 40

    def test_break_from_cycle_multi_worker(self):
        dataset = _make_dataset(n_slides=4, patches_per_slide=5, cycle=True)
        loader = DataLoader(dataset, batch_size=4, num_workers=2)
        count = 0
        for batch in loader:
            count += batch["image"].shape[0]
            if count >= 40:
                break
        assert count >= 40
        assert dataset.stats_dict()["pipeline/patches_extracted"] >= 40

    def test_disjoint_slide_partitioning(self):
        dataset = _make_dataset(n_slides=8, patches_per_slide=3)
        loader = DataLoader(dataset, batch_size=1, num_workers=2)
        all_paths = []
        for batch in loader:
            all_paths.extend(batch["slide_path"])
        per_slide = Counter(all_paths)
        for path, count in per_slide.items():
            assert count == 3, f"{path} appeared {count} times"
        assert len(per_slide) == 8

    def test_step_based_with_workers(self):
        dataset = _make_dataset(n_slides=4, patches_per_slide=10, cycle=True)
        loader = DataLoader(dataset, batch_size=4, num_workers=2)
        loader_iter = iter(loader)
        seen = 0
        try:
            for _ in range(5):
                batch = next(loader_iter)
                assert batch["image"].shape == (4, 3, 64, 64)
                seen += batch["image"].shape[0]
        finally:
            shutdown = getattr(loader_iter, "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()
        assert dataset.stats_dict()["pipeline/patches_extracted"] >= seen


# ── Seed diversity ──


class TestSeedStress:
    def test_same_seed_same_stream(self):
        ds1 = _make_dataset(n_slides=2, patches_per_slide=5, seed=42)
        ds2 = _make_dataset(n_slides=2, patches_per_slide=5, seed=42)
        c1 = [(item["x"], item["y"]) for item in ds1]
        c2 = [(item["x"], item["y"]) for item in ds2]
        assert c1 == c2


# ── Edge cases ──


class TestEdgeCases:
    def test_empty_slide_list(self):
        dataset = WsiStreamDataset(
            slide_paths=[],
            backend=FakeBackend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=64, num_patches=-1, seed=42),
            pool_size=1,
            patches_per_slide=5,
            cycle=False,
        )
        assert len(list(dataset)) == 0

    def test_cycle_all_failures_terminates(self):
        dataset = _make_dataset(n_slides=3, patches_per_slide=5, backend=_FailBackend(), cycle=True)
        assert len(list(dataset)) == 0
        assert dataset.stats_dict()["pipeline/slides_failed"] == 3

    def test_all_expected_slides_appear(self):
        paths = fake_slide_paths(4)
        dataset = WsiStreamDataset(
            slide_paths=paths,
            backend=FakeBackend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=64, num_patches=-1, seed=42),
            pool_size=4,
            patches_per_slide=2,
            cycle=False,
            seed=1,
        )
        seen = {item["slide_path"] for item in dataset}
        assert seen == set(paths)


# ── helpers for DDP tests ──


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_backend(backend_name: str):
    if backend_name == "openslide":
        from wsistream.backends import OpenSlideBackend

        return OpenSlideBackend()
    if backend_name == "tiffslide":
        from wsistream.backends import TiffSlideBackend

        return TiffSlideBackend()
    raise ValueError(f"Unknown backend: {backend_name}")


class _TinyDdpModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(8, 1)

    def forward(self, x):
        x = self.conv(x).relu()
        x = self.pool(x).flatten(1)
        return self.head(x)


def _run_rank(
    rank: int,
    world_size: int,
    port: int,
    result_dir: str,
    slide_paths: list[str],
    backend_name: str,
    batch_size: int,
    num_workers: int,
    patches_per_slide: int,
) -> None:
    torch.manual_seed(123)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    try:
        my_slides = partition_slides_by_rank(slide_paths, rank=rank, world_size=world_size)
        dataset = WsiStreamDataset(
            slide_paths=my_slides,
            backend=_make_backend(backend_name),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42 + rank),
            dataset_adapter=TCGAAdapter(),
            pool_size=min(2, len(my_slides)),
            patches_per_slide=patches_per_slide,
            cycle=False,
            slide_sampling="sequential",
            seed=100,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            multiprocessing_context="spawn",
        )

        model = _TinyDdpModel()
        model = nn.parallel.DistributedDataParallel(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

        seen_paths: set[str] = set()
        coords: list[tuple[str, int, int, int]] = []
        losses: list[float] = []
        batches = 0
        t0 = time.perf_counter()
        for batch in loader:
            seen_paths.update(batch["slide_path"])
            coords.extend(
                [
                    (
                        str(slide_path),
                        int(x),
                        int(y),
                        int(level),
                    )
                    for slide_path, x, y, level in zip(
                        batch["slide_path"],
                        batch["x"],
                        batch["y"],
                        batch["level"],
                    )
                ]
            )

            logits = model(batch["image"])
            loss = logits.square().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            time.sleep(0.01)
            losses.append(float(loss.item()))
            batches += 1

        payload = {
            "rank": rank,
            "elapsed": time.perf_counter() - t0,
            "seen_paths": sorted(seen_paths),
            "coords": coords,
            "losses": losses,
            "batches": batches,
            "stats": dataset.stats_dict(),
            "param_checksum": float(
                sum(param.detach().sum().item() for param in model.module.parameters())
            ),
            "batch_size": batch_size,
            "consumed_patches": len(coords),
        }
        Path(result_dir, f"rank_{rank}.json").write_text(json.dumps(payload))
    finally:
        dist.destroy_process_group()


# ── DDP training loop ──


@pytest.mark.e2e
class TestDdpTrainingLoop:
    @pytest.mark.skipif(
        not dist.is_available() or not dist.is_gloo_available(),
        reason="torch distributed gloo backend not available",
    )
    def test_multi_rank_multi_worker_training_loop(
        self,
        slides,
        backend_name,
        tmp_path: Path,
    ):
        if len(slides) < 10:
            pytest.skip("Need at least 10 slides for the DDP e2e test.")

        world_size = 5
        batch_size = 10
        num_workers = 2
        patches_per_slide = 100
        slide_paths = slides[:10]
        assert len(slide_paths) % world_size == 0
        port = _find_free_port()

        torch_mp.spawn(
            _run_rank,
            args=(
                world_size,
                port,
                str(tmp_path),
                slide_paths,
                backend_name,
                batch_size,
                num_workers,
                patches_per_slide,
            ),
            nprocs=world_size,
            join=True,
        )

        results = [
            json.loads((tmp_path / f"rank_{rank}.json").read_text()) for rank in range(world_size)
        ]
        expected = {
            rank: set(partition_slides_by_rank(slide_paths, rank=rank, world_size=world_size))
            for rank in range(world_size)
        }

        total_consumed = 0
        total_extracted = 0
        total_processed = 0
        print("\nDDP e2e summary:")
        for result in results:
            rank = int(result["rank"])
            slide_counts = Counter(coord[0] for coord in result["coords"])
            unique_coords = len({tuple(coord) for coord in result["coords"]})
            expected_patches = len(expected[rank]) * patches_per_slide
            print(
                f"  rank={rank} elapsed={result['elapsed']:.2f}s "
                f"batches={result['batches']} "
                f"consumed={result['consumed_patches']} "
                f"extracted={result['stats']['pipeline/patches_extracted']} "
                f"processed={result['stats']['pipeline/slides_processed']} "
                f"failed={result['stats']['pipeline/slides_failed']} "
                f"seen={len(result['seen_paths'])}/{len(expected[rank])} "
                f"unique_coords={unique_coords}"
            )
            print(f"    expected: {sorted(expected[rank])}")
            print(f"    seen:     {result['seen_paths']}")
            print(f"    slide_counts: {dict(slide_counts)}")
            print(f"    stats: {result['stats']}")

            assert result["batch_size"] == batch_size
            assert result["consumed_patches"] == expected_patches
            assert result["batches"] == expected_patches // batch_size
            assert len(result["losses"]) == result["batches"]
            assert set(result["seen_paths"]) == expected[rank]
            assert all(np.isfinite(loss) for loss in result["losses"])
            assert result["elapsed"] > 0.0
            assert len(result["coords"]) == expected_patches
            assert unique_coords >= int(expected_patches * 0.8)
            assert slide_counts == {slide: patches_per_slide for slide in expected[rank]}

            stats = result["stats"]
            assert stats["pipeline/slides_processed"] == len(expected[rank])
            assert stats["pipeline/slides_failed"] == 0
            assert stats["pipeline/patches_extracted"] == expected_patches
            assert stats["pipeline/patches_filtered"] == 0

            total_consumed += result["consumed_patches"]
            total_extracted += stats["pipeline/patches_extracted"]
            total_processed += stats["pipeline/slides_processed"]

        checksums = [round(float(result["param_checksum"]), 6) for result in results]
        assert len(set(checksums)) == 1

        print(
            f"  aggregate consumed={total_consumed} extracted={total_extracted} "
            f"processed={total_processed}"
        )
        assert total_consumed == len(slide_paths) * patches_per_slide
        assert total_extracted == len(slide_paths) * patches_per_slide
        assert total_processed == len(slide_paths)
