# wsistream

**Modular online patch streaming from whole-slide images for training pathology foundation models.**

`wsistream` delivers patches directly from WSIs during training -- no pre-extraction to disk, no storage overhead. Every component is pluggable: backends, tissue detectors, samplers, filters, transforms, views, and dataset adapters.

## Why online patching?

Traditional pathology FM training pre-extracts millions of patches to disk before training begins. For example, UNI extracted ~100 million patches from 100K slides ([Chen et al., 2024](https://arxiv.org/abs/2308.15474)), requiring substantial storage and a preprocessing phase.

Online patching, proposed for FM pre-training by Kaiko ([Aben et al., 2024](https://arxiv.org/abs/2404.15217)) and refined in Midnight ([Karasikov et al., 2025](https://arxiv.org/abs/2504.05186)), eliminates this by sampling patches on-the-fly from the original slide files. Patches are read, filtered for tissue, augmented, and fed to the model during training.

See [Online Patching](concepts/online-patching.md) for a detailed discussion.

## What's included

- **Two WSI backends**: OpenSlide (C-based) and TiffSlide (pure Python, cloud-compatible)
- **Four tissue detectors**: Otsu, HSV, CLAM, and a combined detector
- **Three samplers**: random (with rejection sampling), grid, and multi-magnification
- **Per-tile quality filtering**: HSV pixel-based patch acceptance (Midnight-style)
- **Augmentations**: HED stain augmentation, random flip/rotate, resize, normalize, and an albumentations wrapper
- **Multi-view outputs**: multi-view augmentation (SimCLR/BYOL/MoCo-style), DINO-style multi-crop, and same-location magnification views
- **Pool-based slide interleaving**: multiple slides open simultaneously with round-robin patch delivery
- **Infinite streaming**: `cycle=True` for step-based training without epochs
- **Dataset adapters**: TCGA barcode parsing with automatic metadata extraction
- **Pipeline statistics**: tissue fractions, magnification counts, error tracking -- ready for logging (e.g., Weights & Biases)
- **PyTorch compatible**: built-in `WsiStreamDataset` (`IterableDataset`), `MonitoredLoader` for throughput tracking, and DDP slide partitioning

## Quick example

```python
from wsistream.pipeline import PatchPipeline
from wsistream.backends import OpenSlideBackend
from wsistream.tissue import OtsuTissueDetector
from wsistream.filters import HSVPatchFilter
from wsistream.sampling import RandomSampler
from wsistream.transforms import (
    ComposeTransforms, HEDColorAugmentation, RandomFlipRotate, ResizeTransform,
)

pipeline = PatchPipeline(
    slide_paths="/path/to/slides",  # directory or list of files
    backend=OpenSlideBackend(),
    tissue_detector=OtsuTissueDetector(),
    sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
    patch_filter=HSVPatchFilter(),
    transforms=ComposeTransforms(transforms=[
        HEDColorAugmentation(sigma=0.05),
        RandomFlipRotate(),
        ResizeTransform(target_size=224),
    ]),
    pool_size=1,
    patches_per_slide=100,
)

for result in pipeline:
    image = result.image              # numpy array, (224, 224, 3), uint8
    coord = result.coordinate         # PatchCoordinate (x, y, level, mpp, ...)
    tissue = result.tissue_fraction   # float in [0, 1]
```

## Next steps

- [Getting Started](getting-started.md) -- installation and first pipeline
- [Online Patching](concepts/online-patching.md) -- the core concept
- [Architecture](concepts/architecture.md) -- how the pipeline works
