# PyTorch DataLoader

`wsistream` provides `WsiStreamDataset`, an `IterableDataset` that wraps `PatchPipeline` and handles multi-worker slide partitioning automatically.

```bash
pip install -e ".[torch]"
```

## Basic usage

```python
from torch.utils.data import DataLoader

from wsistream.backends import OpenSlideBackend
from wsistream.sampling import RandomSampler
from wsistream.tissue import OtsuTissueDetector
from wsistream.torch import WsiStreamDataset

dataset = WsiStreamDataset(
    slide_paths=slide_paths,
    backend=OpenSlideBackend(),
    tissue_detector=OtsuTissueDetector(),
    sampler=RandomSampler(patch_size=256, target_mpp=0.5),
    pool_size=8,
    patches_per_slide=100,
    # replacement="without_replacement",  # optional: no repeated coords per slide
)

loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)

for batch in loader:
    images = batch["image"]            # (B, 3, H, W) float32
    x = batch["x"]                     # (B,) int — level-0 x coordinates
    y = batch["y"]                     # (B,) int — level-0 y coordinates
    mpp = batch["mpp"]                 # (B,) float — microns/px, -1.0 if unavailable
    tf = batch["tissue_fraction"]      # (B,) float
    paths = batch["slide_path"]        # list[str], length B
    patient = batch["patient_id"]      # list[str], length B (empty if no adapter)
```

Each batch is a dict of primitives and tensors. Image conversion (HWC uint8 → CHW float32, divided by 255) is handled internally. If a `NormalizeTransform` is included in the transforms chain, the image is already float32 and is passed through without re-scaling — values will reflect the normalization (e.g., roughly `[-2, 3]` for ImageNet stats), not `[0, 1]`.

With multi-view datasets, each view is collated under its configured name:

```python
batch["global_0"]  # (B, 3, 224, 224)
batch["global_1"]  # (B, 3, 224, 224)
batch["local_0"]  # (B, 3, 96, 96)
```

Coordinate and metadata fields follow the same schema. See [Views](../components/views.md) for multi-view configuration examples.

!!! note "Default slide ordering"
    `WsiStreamDataset` defaults to `slide_sampling="random"` (better for training diversity), while `PatchPipeline` defaults to `"sequential"`. If you need deterministic slide order through the dataset wrapper, pass `slide_sampling="sequential"` explicitly.

!!! note "Deterministic validation"
    If you want the same validation patches every time, use a fixed `seed`, `slide_sampling="sequential"`, `cycle=False`, and `num_workers=0` on the validation `DataLoader`. With `num_workers>0`, `PatchPipeline` mixes the worker PID into RNG seeds so repeated validation runs are not bit-exact across calls.

## Why IterableDataset, not Dataset?

A map-style [`Dataset`](https://pytorch.org/docs/stable/data.html#map-style-datasets) requires `__len__` and `__getitem__`. Online patching is inherently stochastic -- there is no fixed set of patches to index. [`IterableDataset`](https://pytorch.org/docs/stable/data.html#iterable-style-datasets) streams lazily, which is what online patching needs. See the [PyTorch data loading docs](https://pytorch.org/docs/stable/data.html) for background on the two dataset styles.

## Step-based training

With `cycle=True` (the default in `WsiStreamDataset`), the pipeline produces an infinite stream. Since patches are randomly sampled from tissue regions, there is no guarantee of seeing the same patches twice — a traditional "epoch" is not meaningful. Training is defined by a number of steps:

```python
loader_iter = iter(loader)

for step in range(total_steps):
    batch = next(loader_iter)
    images = batch["image"].to(device, non_blocking=True)

    loss = model(images)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

See [Online Patching](../concepts/online-patching.md) for why there are no epochs.

## Logging and throughput monitoring

Wrap the DataLoader with `MonitoredLoader` to automatically track data wait time, compute time, and throughput. It also merges `dataset.stats_dict()` into each payload:

```python
from wsistream.torch import MonitoredLoader

mon = MonitoredLoader(loader, dataset=dataset, device=device, log_every=100)

for step, batch in enumerate(mon):
    images = batch["image"].to(device, non_blocking=True)
    loss = model(images).mean()  # placeholder — replace with your actual loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    payload = mon.mark_step(extra={"train/loss": float(loss.detach())})
    if payload is not None:
        wandb.log(payload, step=step)
```

See [Weights & Biases](wandb.md) for details on what metrics are included.

## Contiguous arrays

Numpy arrays from `np.flip` or `np.rot90` (used by `RandomFlipRotate`) may not be contiguous in memory, which causes `torch.from_numpy` to fail. `WsiStreamDataset` handles this internally with `np.ascontiguousarray()`.

## Full example

See `examples/train_single_gpu.py` and `examples/train_ddp.py` in the repository for complete working examples.
