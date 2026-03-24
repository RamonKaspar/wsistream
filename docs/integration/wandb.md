# Weights & Biases

Both `PatchPipeline` and `WsiStreamDataset` provide a `stats_dict()` method that returns a flat dictionary ready for `wandb.log()`. For PyTorch training loops, `MonitoredLoader` adds automatic throughput and timing metrics.

## With PatchPipeline

For non-PyTorch workflows or custom training loops:

```python
import wandb
from wsistream.pipeline import PatchPipeline

wandb.init(project="pathology-fm")

pipeline = PatchPipeline(
    slide_paths="/data/tcga",
    dataset_adapter=TCGAAdapter(),
    ...
)

for step, result in enumerate(pipeline):
    # ... training step ...

    if step % 500 == 0:
        wandb.log(pipeline.stats_dict(), step=step)
```

`pipeline.stats_dict()` includes detailed per-slide statistics:

```python
pipeline.stats_dict()
# {
#     "pipeline/slides_processed": 42,
#     "pipeline/slides_failed": 1,
#     "pipeline/patches_extracted": 4200,
#     "pipeline/patches_filtered": 380,
#     "pipeline/mean_tissue_fraction": 0.63,
#     "pipeline/min_tissue_fraction": 0.21,
#     "pipeline/max_tissue_fraction": 0.91,
#     "pipeline/mpp_0.25": 1050,      # ~40x
#     "pipeline/mpp_0.50": 1050,      # ~20x
#     "pipeline/mpp_1.00": 1050,      # ~10x
#     "pipeline/mpp_2.00": 1050,      # ~5x
#     "pipeline/cancer_type/TCGA-BRCA": 15,
#     "pipeline/cancer_type/TCGA-LUAD": 12,
#     "pipeline/sample_type/primary_solid_tumor": 38,
#     "pipeline/sample_type/solid_tissue_normal": 4,
#     "pipeline/error_count": 1,
# }
```

The magnification keys (`mpp_*`) are only present when patches are extracted at known MPP values. The cancer/sample type keys are only present when a `DatasetAdapter` is configured. `error_count` is only present when errors have occurred.

For programmatic access to raw data, use the `stats` property:

```python
stats = pipeline.stats

stats.slides_processed            # int
stats.slides_failed               # int
stats.patches_extracted           # int
stats.patches_filtered            # int (rejected by PatchFilter)
stats.tissue_fractions.count      # int — number of slides
stats.tissue_fractions.mean       # float | None
stats.tissue_fractions.min_val    # float
stats.tissue_fractions.max_val    # float
stats.error_count                 # int — total errors recorded
stats.recent_errors               # deque — most recent 100 (slide_path, message) pairs
```

Use `pipeline.reset_stats()` to clear accumulated counters at any point.

## With WsiStreamDataset + MonitoredLoader (PyTorch)

`MonitoredLoader` wraps the DataLoader and automatically tracks data wait time, compute time, and throughput. It also includes `dataset.stats_dict()` in every payload:

```python
import wandb
from torch.utils.data import DataLoader
from wsistream.torch import MonitoredLoader, WsiStreamDataset

wandb.init(project="pathology-fm")

dataset = WsiStreamDataset(
    slide_paths="/data/tcga",
    ...
)

loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)
mon = MonitoredLoader(loader, dataset=dataset, device=device, log_every=100)

for step, batch in enumerate(mon):
    images = batch["image"].to(device, non_blocking=True)
    loss = model(images).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    payload = mon.mark_step(extra={"train/loss": float(loss)})
    if payload is not None:
        wandb.log(payload, step=step)
```

Each payload includes:

| Key | Description |
|-----|-------------|
| `loader/data_wait_ms` | Average time waiting for the next batch (per step) |
| `loader/compute_ms` | Average time in the training step (per step) |
| `loader/step_ms` | Total step time (data + compute) |
| `loader/data_fraction` | Fraction of step time spent waiting for data |
| `loader/batches_per_sec` | Training step throughput |
| `loader/patches_per_sec` | Patch throughput |
| `pipeline/*` | All keys from `dataset.stats_dict()` |

If `loader/data_fraction` is consistently above 0.5, the data pipeline is the bottleneck — increase `num_workers` or check your storage I/O. See [Benchmarking](benchmarking.md) for how to find optimal settings before training.

!!! note "Live stats accuracy"
    With `num_workers=0`, pipeline stats are exact on every step. With `num_workers>0`, stats are aggregated from worker processes and may lag by up to 16 patches per worker between flushes. Final totals at the end of iteration are always correct.

## DDP

In DDP, each rank has its own `MonitoredLoader`. All ranks call `mark_step()` every step, but only rank 0 logs:

```python
mon = MonitoredLoader(loader, dataset=dataset, device=device, log_every=100)

for step, batch in enumerate(mon):
    # ... training step ...

    payload = mon.mark_step(extra={"train/loss": float(loss)})
    if payload is not None and rank == 0:
        wandb.log(payload, step=step)
```

For per-rank straggler detection, log from every rank with a prefix:

```python
    if payload is not None:
        wandb.log({
            f"rank_{rank}/{k}": v for k, v in payload.items()
        }, step=step)
```
