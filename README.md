# `wsistream`

<p align="center">
    <a href="https://pypi.org/project/wsistream/"><img alt="PyPI" src="https://img.shields.io/pypi/v/wsistream"></a>
    <a href="https://pypi.org/project/wsistream/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/wsistream"></a>
    <a href="https://github.com/RamonKaspar/wsistream/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/RamonKaspar/wsistream"></a>
    <a href="https://ramonkaspar.github.io/wsistream"><img alt="Docs" src="https://img.shields.io/badge/docs-GitHub%20Pages-blue"></a>
</p>

Modular online patch streaming from whole-slide images for computational pathology. Stream patches directly from WSIs during training (no disk pre-extraction, no storage overhead).

Every component is pluggable: backends, tissue detectors, samplers, filters, transforms, dataset adapters.

## Install

```bash
pip install "wsistream[openslide]"   # with OpenSlide
pip install "wsistream[tiffslide]"   # with TiffSlide (pure Python)
pip install "wsistream[torch]"       # add PyTorch integration (WsiStreamDataset, DDP)
pip install "wsistream[all]"         # everything (OpenSlide + TiffSlide + PyTorch + albumentations + matplotlib)
```

For development:

```bash
git clone https://github.com/RamonKaspar/wsistream.git
cd wsistream
pip install -e ".[dev]"
```

## Documentation

Full documentation: [ramonkaspar.github.io/wsistream](https://ramonkaspar.github.io/wsistream)

To build locally:

```bash
pip install mkdocs-material
mkdocs serve          # local preview at http://127.0.0.1:8000
```

## How it works

Each slide goes through a fixed pipeline:

1. **Open slide**: via an explicit backend (`OpenSlideBackend` or `TiffSlideBackend`)
2. **Detect tissue**: run a `TissueDetector` on a low-res thumbnail to get a binary mask
3. **Sample coordinates**: a `PatchSampler` proposes (x, y) locations within tissue regions
4. **Extract patch**: read the pixel data from the slide at each coordinate
5. **Filter patch**: a `PatchFilter` accepts or rejects the tile based on its pixels
6. **Transform patch**: apply augmentations (`HEDColorAugmentation`, `RandomFlipRotate`, etc.)
7. **Yield result**: `PatchResult` with image, coordinates, tissue fraction, and metadata

## Quick start

```python
from wsistream.pipeline import PatchPipeline
from wsistream.backends import OpenSlideBackend
from wsistream.tissue import CLAMTissueDetector
from wsistream.sampling import RandomSampler
from wsistream.filters import HSVPatchFilter
from wsistream.transforms import ComposeTransforms, HEDColorAugmentation, RandomFlipRotate, ResizeTransform
from wsistream.datasets import TCGAAdapter

pipeline = PatchPipeline(
    slide_paths="/data/tcga",  # directory or list of files
    backend=OpenSlideBackend(),
    tissue_detector=CLAMTissueDetector(),
    sampler=RandomSampler(patch_size=256, num_patches=-1, target_mpp=0.5),
    patch_filter=HSVPatchFilter(min_pixel_fraction=0.6),
    transforms=ComposeTransforms(transforms=[
        HEDColorAugmentation(sigma=0.05),
        RandomFlipRotate(),
        ResizeTransform(target_size=224),
    ]),
    dataset_adapter=TCGAAdapter(),
    pool_size=8,
    patches_per_slide=100,
    cycle=True,
)

for result in pipeline:
    print(result.image.shape)                # (224, 224, 3) uint8
    print(result.coordinate.mpp)             # ~0.5
    print(result.tissue_fraction)            # 0.87
    print(result.slide_metadata.patient_id)  # TCGA-3L-AA1B
```

## Pool-based slide interleaving

The pipeline keeps `pool_size` slides open simultaneously and takes `patches_per_slide` patches from each before closing it and opening the next. With `cycle=True`, slides are re-queued for infinite streaming.

## PyTorch integration

`wsistream.torch` provides `WsiStreamDataset` (an `IterableDataset`), `MonitoredLoader` for throughput tracking, and `partition_slides_by_rank` for DDP. Worker-level slide partitioning is handled automatically.

```python
from torch.utils.data import DataLoader
from wsistream.backends import OpenSlideBackend
from wsistream.sampling import RandomSampler
from wsistream.tissue import OtsuTissueDetector
from wsistream.torch import WsiStreamDataset, partition_slides_by_rank

my_slides = partition_slides_by_rank("/data/tcga", rank=rank, world_size=world_size)

dataset = WsiStreamDataset(
    slide_paths=my_slides,
    backend=OpenSlideBackend(),
    tissue_detector=OtsuTissueDetector(),
    sampler=RandomSampler(patch_size=256, num_patches=-1, target_mpp=0.5),
)

loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)
loader_iter = iter(loader)

for step in range(total_steps):
    batch = next(loader_iter)
    images = batch["image"].to(device, non_blocking=True)  # (B, 3, H, W) float32
```
