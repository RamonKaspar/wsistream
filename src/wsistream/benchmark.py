"""
Throughput benchmarking for WsiStreamDataset.

Sweeps over DataLoader and pipeline configurations to find the
settings that maximize patch throughput.  Supports multi-rank DDP
simulation to measure realistic filesystem contention.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import pickle
import socket
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from torch.utils.data import IterableDataset

from wsistream.types import resolve_slide_paths

logger = logging.getLogger(__name__)


class _MemoryMonitor:
    """Sample peak RSS of the current process tree in a background thread."""

    def __init__(self, interval: float = 0.5) -> None:
        import threading

        import psutil

        self._process = psutil.Process()
        self._interval = interval
        self._peak_bytes: int = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _tree_rss(self) -> int:
        """Sum RSS of the process and all its children."""
        import psutil

        try:
            rss = self._process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
            return 0

        try:
            for child in self._process.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
                    pass
            return rss
        except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
            return rss

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = self._tree_rss()
            if rss > self._peak_bytes:
                self._peak_bytes = rss
            self._stop.wait(self._interval)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> float:
        """Stop monitoring and return peak RSS in MB."""
        self._stop.set()
        self._thread.join(timeout=2.0)
        return self._peak_bytes / (1024 * 1024)


# Factory: (slide_paths, pool_size, patches_per_slide, patches_per_visit, seed) -> IterableDataset
MakeDatasetFn = Callable[["list[str]", int, int, int, int], IterableDataset]


@dataclass
class BenchmarkResult:
    """Results from a single benchmark configuration."""

    num_workers: int
    world_size: int
    pool_size: int
    patches_per_slide: int
    patches_per_visit: int
    batch_size: int
    per_rank_patches_per_sec: list[float]
    per_rank_batch_times_ms: list[list[float]]
    effective_sync_throughput: float
    aggregate_throughput: float
    total_patches: int
    total_time_sec: float
    per_rank_peak_rss_mb: list[float]
    peak_rss_mb: float


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _configure_threads() -> dict[str, str]:
    """Pin thread pools to 1 thread for consistent, comparable measurements."""
    import torch

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        import cv2

        cv2.setNumThreads(1)
    except ImportError:
        pass

    return {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "?"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "?"),
        "torch.num_threads": str(torch.get_num_threads()),
    }


def _infer_batch_size(batch) -> int:
    """Infer batch size from the first tensor-like field."""
    if isinstance(batch, dict):
        if "image" in batch:
            try:
                return int(batch["image"].shape[0])
            except (AttributeError, IndexError):
                return 1
        for value in batch.values():
            try:
                return int(value.shape[0])
            except (AttributeError, IndexError):
                continue
        for value in batch.values():
            if isinstance(value, (list, tuple)):
                return len(value)
    try:
        return int(batch.shape[0])
    except (AttributeError, IndexError):
        return 1


def _measure_rank(
    make_dataset: MakeDatasetFn,
    slide_paths: list[str],
    num_workers: int,
    pool_size: int,
    patches_per_slide: int,
    patches_per_visit: int,
    batch_size: int,
    warmup_batches: int,
    measure_batches: int,
    prefetch_factor: int | None,
    persistent_workers: bool,
    pin_memory: bool,
    multiprocessing_context: str | None,
    seed: int,
) -> dict:
    """Run measurement for a single rank. Returns raw results dict."""
    from torch.utils.data import DataLoader

    dataset = make_dataset(slide_paths, pool_size, patches_per_slide, patches_per_visit, seed)

    loader_kwargs: dict = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        if multiprocessing_context is not None:
            loader_kwargs["multiprocessing_context"] = multiprocessing_context
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor

    loader = DataLoader(dataset, **loader_kwargs)
    loader_iter = iter(loader)

    # Start memory monitoring before warmup (captures slide opens + worker startup)
    mem_monitor = _MemoryMonitor(interval=0.5)
    mem_monitor.start()

    for _ in range(warmup_batches):
        next(loader_iter)

    batch_times: list[float] = []
    total_patches = 0
    t_start = time.perf_counter()
    for _ in range(measure_batches):
        t_batch = time.perf_counter()
        batch = next(loader_iter)
        batch_times.append(time.perf_counter() - t_batch)
        total_patches += _infer_batch_size(batch)
    elapsed = time.perf_counter() - t_start

    peak_rss_mb = mem_monitor.stop()

    return {
        "total_patches": total_patches,
        "elapsed_sec": elapsed,
        "batch_times": batch_times,
        "patches_per_sec": total_patches / elapsed,
        "peak_rss_mb": peak_rss_mb,
    }


def _benchmark_worker(
    rank: int,
    world_size: int,
    port: int,
    result_dir: str,
    make_dataset: MakeDatasetFn,
    slide_paths: list[str],
    num_workers: int,
    pool_size: int,
    patches_per_slide: int,
    patches_per_visit: int,
    batch_size: int,
    warmup_batches: int,
    measure_batches: int,
    prefetch_factor: int | None,
    persistent_workers: bool,
    pin_memory: bool,
    multiprocessing_context: str | None,
    seed: int,
) -> None:
    """DDP worker function. Must be module-level for torch_mp.spawn pickling."""
    import torch.distributed as dist

    from wsistream.torch import partition_slides_by_rank

    _configure_threads()

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        my_slides = partition_slides_by_rank(slide_paths, rank=rank, world_size=world_size)
        result = _measure_rank(
            make_dataset=make_dataset,
            slide_paths=my_slides,
            num_workers=num_workers,
            pool_size=pool_size,
            patches_per_slide=patches_per_slide,
            patches_per_visit=patches_per_visit,
            batch_size=batch_size,
            warmup_batches=warmup_batches,
            measure_batches=measure_batches,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            multiprocessing_context=multiprocessing_context,
            seed=seed + rank,
        )
        result["rank"] = rank
        Path(result_dir, f"rank_{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


def _run_config(
    make_dataset: MakeDatasetFn,
    slide_paths: list[str],
    num_workers: int,
    world_size: int,
    pool_size: int,
    patches_per_slide: int,
    patches_per_visit: int,
    batch_size: int,
    warmup_batches: int,
    measure_batches: int,
    prefetch_factor: int | None,
    persistent_workers: bool,
    pin_memory: bool,
    multiprocessing_context: str | None,
    seed: int,
) -> BenchmarkResult:
    """Run a single benchmark configuration and return results."""

    if world_size == 1:
        # Fast path: no DDP overhead, no pickling requirement
        result = _measure_rank(
            make_dataset=make_dataset,
            slide_paths=slide_paths,
            num_workers=num_workers,
            pool_size=pool_size,
            patches_per_slide=patches_per_slide,
            patches_per_visit=patches_per_visit,
            batch_size=batch_size,
            warmup_batches=warmup_batches,
            measure_batches=measure_batches,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            multiprocessing_context=multiprocessing_context,
            seed=seed,
        )
        rank_results = [result]
    else:
        import torch.multiprocessing as torch_mp

        with tempfile.TemporaryDirectory(prefix="wsistream_bench_") as tmp_dir:
            port = _find_free_port()
            torch_mp.spawn(
                _benchmark_worker,
                args=(
                    world_size,
                    port,
                    tmp_dir,
                    make_dataset,
                    slide_paths,
                    num_workers,
                    pool_size,
                    patches_per_slide,
                    patches_per_visit,
                    batch_size,
                    warmup_batches,
                    measure_batches,
                    prefetch_factor,
                    persistent_workers,
                    pin_memory,
                    multiprocessing_context,
                    seed,
                ),
                nprocs=world_size,
                join=True,
            )
            rank_results = [
                json.loads((Path(tmp_dir) / f"rank_{r}.json").read_text())
                for r in range(world_size)
            ]

    per_rank_pps = [r["patches_per_sec"] for r in rank_results]
    per_rank_batch_times = [
        [t * 1000 for t in r["batch_times"]]  # convert to ms
        for r in rank_results
    ]
    per_rank_rss = [r.get("peak_rss_mb", 0.0) for r in rank_results]

    # In DDP, training rate is limited by the slowest rank.
    # Use the slowest rank's total throughput (not median batch time,
    # which is misleading for bursty/bimodal workloads).
    slowest_pps = min(per_rank_pps)
    effective_sync = slowest_pps * world_size

    return BenchmarkResult(
        num_workers=num_workers,
        world_size=world_size,
        pool_size=pool_size,
        patches_per_slide=patches_per_slide,
        patches_per_visit=patches_per_visit,
        batch_size=batch_size,
        per_rank_patches_per_sec=per_rank_pps,
        per_rank_batch_times_ms=per_rank_batch_times,
        effective_sync_throughput=effective_sync,
        aggregate_throughput=sum(per_rank_pps),
        total_patches=sum(r["total_patches"] for r in rank_results),
        total_time_sec=max(r["elapsed_sec"] for r in rank_results),
        per_rank_peak_rss_mb=per_rank_rss,
        peak_rss_mb=max(per_rank_rss),
    )


def _ensure_list(x: int | list[int]) -> list[int]:
    return x if isinstance(x, list) else [x]


def benchmark_throughput(
    make_dataset: MakeDatasetFn,
    slide_paths: str | Path | list[str | Path],
    num_workers: list[int] | int = 4,
    world_size: int = 1,
    pool_size: list[int] | int = 8,
    patches_per_slide: list[int] | int = 100,
    patches_per_visit: list[int] | int = 1,
    batch_size: int = 64,
    warmup_batches: int = 10,
    measure_batches: int = 50,
    prefetch_factor: int | None = 2,
    persistent_workers: bool = False,
    pin_memory: bool = False,
    multiprocessing_context: str | None = "spawn",
    seed: int = 42,
    verbose: bool = True,
) -> list[BenchmarkResult]:
    """Benchmark DataLoader throughput across pipeline configurations.

    Sweeps over ``num_workers``, ``pool_size``, ``patches_per_slide``,
    and ``patches_per_visit`` to find the configuration that maximizes
    patch throughput.  For ``world_size > 1``, launches actual DDP
    processes via ``torch.multiprocessing.spawn`` to measure realistic
    filesystem contention.

    Parameters
    ----------
    make_dataset : callable
        Factory ``(slide_paths, pool_size, patches_per_slide,
        patches_per_visit, seed) -> WsiStreamDataset``.
        Must return a dataset with ``cycle=True``.  Must be a top-level
        function (not a lambda or closure) when ``world_size > 1``.
    slide_paths : str, Path, or list
        A directory path, a single file, or an explicit list of files.
        For multi-rank configs, slides are partitioned internally via
        ``partition_slides_by_rank``.
    num_workers : int or list[int]
        DataLoader worker counts to sweep.
    world_size : int
        Number of DDP ranks (determined by your GPU count).
    pool_size : int or list[int]
        Pipeline pool sizes to sweep.
    patches_per_slide : int or list[int]
        Patches-per-slide values to sweep.
    patches_per_visit : int or list[int]
        Patches-per-visit values to sweep. Higher values improve I/O
        locality on network filesystems.
    batch_size : int
        Fixed batch size for all configs.
    warmup_batches : int
        Batches to discard before measuring (covers slide open + tissue
        detection + DataLoader prefetch warm-up).
    measure_batches : int
        Batches to time.
    prefetch_factor, persistent_workers, pin_memory, multiprocessing_context
        Passed through to ``DataLoader``.
    seed : int
        Base random seed.
    verbose : bool
        Print progress table to stdout.

    Returns
    -------
    list[BenchmarkResult]
        One result per configuration tested.
    """
    slide_paths = resolve_slide_paths(slide_paths)
    if not slide_paths:
        raise ValueError("slide_paths is empty")

    num_workers_list = _ensure_list(num_workers)
    pool_size_list = _ensure_list(pool_size)
    pps_list = _ensure_list(patches_per_slide)
    ppv_list = _ensure_list(patches_per_visit)

    configs = list(
        itertools.product(
            num_workers_list,
            pool_size_list,
            pps_list,
            ppv_list,
        )
    )

    # Pin thread pools before any measurement
    thread_settings = _configure_threads()

    if verbose:
        print(f"Slides: {len(slide_paths)} | World size: {world_size}")
        print(f"Thread settings: {thread_settings}")
        print(
            f"Batch size: {batch_size}, "
            f"Warmup: {warmup_batches} batches, Measure: {measure_batches} batches"
        )
        print(f"Testing {len(configs)} configuration(s)\n")

    # Check for DDP pickle safety upfront
    if world_size > 1:
        try:
            pickle.loads(pickle.dumps(make_dataset))
        except (pickle.PicklingError, AttributeError, TypeError) as e:
            raise TypeError(
                f"make_dataset factory is not picklable: {e}\n"
                "DDP benchmark uses torch.multiprocessing.spawn which requires "
                "picklable arguments. Define make_dataset as a top-level function "
                "(not a lambda, closure, or nested function)."
            ) from e

    # Validate cycle=True with a probe dataset before running the full sweep
    probe = make_dataset(slide_paths[:1], pool_size_list[0], pps_list[0], ppv_list[0], seed)
    if hasattr(probe, "_cycle") and not probe._cycle:
        raise ValueError(
            "benchmark_throughput requires cycle=True in the dataset factory. "
            "Finite iteration would terminate early for some configs, making "
            "throughput measurements incomparable."
        )
    del probe

    # Table header
    if verbose:
        header = (
            f"{'num_workers':>11}  {'pool_size':>9}  "
            f"{'patches/slide':>13}  {'patches/visit':>13}  "
            f"{'effective':>11}  {'aggregate':>11}  {'slowest':>9}  "
            f"{'peak_rss':>10}"
        )
        print(header)
        print("-" * len(header))

    cpu_count = os.cpu_count() or 1
    results: list[BenchmarkResult] = []

    for nw, ps, pps, ppv in configs:
        if len(slide_paths) < world_size:
            if verbose:
                print(
                    f"{nw:>11}  {ps:>9}  {pps:>13}  {ppv:>13}  "
                    f"SKIP: need >= {world_size} slides (have {len(slide_paths)})"
                )
            continue

        total_workers = world_size * nw
        if total_workers > cpu_count:
            logger.warning(
                "Oversubscription: world_size=%d x num_workers=%d = %d worker "
                "processes but only %d CPU cores. Results may be bottlenecked "
                "by context switching.",
                world_size,
                nw,
                total_workers,
                cpu_count,
            )

        try:
            result = _run_config(
                make_dataset=make_dataset,
                slide_paths=slide_paths,
                num_workers=nw,
                world_size=world_size,
                pool_size=ps,
                patches_per_slide=pps,
                patches_per_visit=ppv,
                batch_size=batch_size,
                warmup_batches=warmup_batches,
                measure_batches=measure_batches,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
                pin_memory=pin_memory,
                multiprocessing_context=multiprocessing_context,
                seed=seed,
            )
            results.append(result)

            slowest = min(result.per_rank_patches_per_sec)

            if verbose:
                print(
                    f"{nw:>11}  {ps:>9}  {pps:>13}  {ppv:>13}  "
                    f"{result.effective_sync_throughput:>11.0f}  "
                    f"{result.aggregate_throughput:>11.0f}  "
                    f"{slowest:>9.0f}  "
                    f"{result.peak_rss_mb:>8.0f} MB"
                )

        except Exception as e:
            if verbose:
                print(f"{nw:>11}  {ps:>9}  {pps:>13}  {ppv:>13}  ERROR: {e}")
            logger.warning("Config nw=%d ps=%d pps=%d ppv=%d failed: %s", nw, ps, pps, ppv, e)

    if not results:
        raise RuntimeError(
            f"All {len(configs)} benchmark configuration(s) failed. " f"Check the logs for details."
        )

    if verbose:
        best = max(results, key=lambda r: r.effective_sync_throughput)
        print(
            f"\nBest: num_workers={best.num_workers}, "
            f"pool_size={best.pool_size}, "
            f"patches_per_slide={best.patches_per_slide}, "
            f"patches_per_visit={best.patches_per_visit} "
            f"-> {best.effective_sync_throughput:.0f} effective patches/sec"
        )
        print("\nPer-rank detail (best config):")
        for i, pps_val in enumerate(best.per_rank_patches_per_sec):
            p50 = float(np.median(best.per_rank_batch_times_ms[i]))
            p95 = float(np.percentile(best.per_rank_batch_times_ms[i], 95))
            rss = best.per_rank_peak_rss_mb[i]
            print(
                f"  rank {i}: {pps_val:.0f} patches/sec, "
                f"batch_time p50={p50:.1f}ms p95={p95:.1f}ms, "
                f"peak_rss={rss:.0f} MB"
            )

    return results
