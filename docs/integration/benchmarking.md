# Benchmarking

Before training, you want to know: can my data pipeline keep the GPU fed? `benchmark_throughput` answers this by measuring actual patch throughput across different configurations.

Typically, you fix `world_size` (determined by your GPU count and model size) and sweep `num_workers` and `pool_size` to find where throughput saturates. The benchmark uses `world_size` to simulate realistic filesystem contention — all ranks hit storage simultaneously, just like in real training.

## Quick start

```python
from wsistream.backends import OpenSlideBackend
from wsistream.benchmark import benchmark_throughput
from wsistream.sampling import RandomSampler
from wsistream.tissue import OtsuTissueDetector
from wsistream.torch import WsiStreamDataset


def make_dataset(slide_paths, pool_size, patches_per_slide, seed):
    """Factory that creates a dataset for each benchmark config.

    Must be a top-level function (not a lambda or closure) so that
    it can be pickled for DDP multi-rank benchmarks.
    """
    return WsiStreamDataset(
        slide_paths=slide_paths,
        backend=OpenSlideBackend(),
        tissue_detector=OtsuTissueDetector(),
        sampler=RandomSampler(patch_size=256, num_patches=1000, target_mpp=0.5),
        pool_size=pool_size,
        patches_per_slide=patches_per_slide,
        cycle=True,  # benchmark measures steady-state; finite iteration would stop early
        seed=seed,
    )


results = benchmark_throughput(
    make_dataset=make_dataset,
    slide_paths="/data/tcga",  # directory or list of files
    world_size=4,              # fixed: determined by your GPU count / model size
    num_workers=[1, 2, 4, 8],  # sweep: find where throughput saturates
    pool_size=[4, 8],          # sweep: trade-off between interleaving and file handles
    batch_size=64,
)
```

Output:

```
Slides: 100
Thread settings: {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'torch.num_threads': '1'}
Batch size: 64, Warmup: 10 batches, Measure: 50 batches
Testing 8 configuration(s)

world_size  num_workers  pool_size  patches/slide    effective    aggregate    slowest
         4            1          4            100          312          312         78
         4            2          4            100         1103         1103        276
         4            4          4            100         2100         2180        525
         4            4          8            100         2150         2300        537

Best: world_size=4, num_workers=4, pool_size=8, patches_per_slide=100 -> 2150 effective patches/sec
```

## Metrics

The output includes three throughput metrics:

- **effective**: `world_size * batch_size / max(rank_batch_p50)` — the actual DDP training step rate. In DDP, all ranks must synchronize at each backward pass, so throughput is limited by the slowest rank. This is the number that matters for training.
- **aggregate**: sum of per-rank throughputs. The theoretical maximum if ranks never waited for each other.
- **slowest**: the throughput of the slowest rank. If this is much lower than the fastest, you have a straggler problem (likely caused by uneven slide sizes or storage hotspots).

## The factory pattern

`benchmark_throughput` takes a factory function instead of a dataset object. The factory receives `(slide_paths, pool_size, patches_per_slide, seed)` and returns a `WsiStreamDataset`:

- **`slide_paths`**: the slides assigned to this rank (already partitioned for multi-rank configs)
- **`pool_size`** and **`patches_per_slide`**: the values being swept by the benchmark
- **`seed`**: a per-rank seed for reproducibility

Everything else (backend, tissue detector, sampler, transforms, etc.) is fixed by the factory. This keeps the benchmark generic without introspecting dataset internals.

## Multi-rank benchmarks

When `world_size > 1`, the benchmark spawns actual processes via `torch.multiprocessing.spawn` with the gloo backend. Each rank partitions slides via `partition_slides_by_rank` and creates its own DataLoader. This measures realistic filesystem contention — single-rank benchmarks can be misleadingly optimistic if your storage can't handle concurrent readers.

!!! note "Pickling requirement"
    `torch.multiprocessing.spawn` pickles the factory function to send it to worker processes. The factory must be a **top-level function** (not a lambda, closure, or nested function). If pickling fails, the benchmark raises a `TypeError` with an actionable message.

## Thread control

The benchmark pins `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`, and `cv2.setNumThreads(1)` before running. This ensures consistent, comparable results across configs — without pinning, OpenCV and NumPy may spawn varying numbers of internal threads that interfere with measurements.

## Oversubscription

When `world_size * num_workers` exceeds the number of available CPU cores, the benchmark emits a warning. Results under oversubscription are dominated by context switching, not I/O throughput.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `make_dataset` | callable | required | Factory `(slide_paths, pool_size, patches_per_slide, seed) -> WsiStreamDataset` |
| `slide_paths` | str, Path, or list | required | Directory path or list of slide files |
| `num_workers` | int or list | `4` | DataLoader worker counts to sweep |
| `world_size` | int or list | `1` | DDP rank counts to sweep |
| `pool_size` | int or list | `8` | Pipeline pool sizes to sweep |
| `patches_per_slide` | int or list | `100` | Patches-per-slide values to sweep |
| `batch_size` | int | `64` | Batch size (fixed, not swept) |
| `warmup_batches` | int | `10` | Batches discarded before measuring (slide warm-up) |
| `measure_batches` | int | `50` | Batches to time |
| `prefetch_factor` | int or None | `2` | DataLoader prefetch factor |
| `persistent_workers` | bool | `False` | Keep workers alive between iterations |
| `pin_memory` | bool | `False` | Pin tensors to CUDA memory |
| `multiprocessing_context` | str or None | `"spawn"` | Worker process start method |
| `seed` | int | `42` | Base random seed |
| `verbose` | bool | `True` | Print progress table |
