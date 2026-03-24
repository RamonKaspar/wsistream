# Getting Started

## Installation

```bash
git clone https://github.com/RamonKaspar/wsistream.git
cd wsistream
pip install -e ".[openslide]"   # or [tiffslide], or [all], or [dev]
```

Available extras:

| Extra | What it adds |
|-------|-------------|
| `openslide` | OpenSlide backend (requires system-level OpenSlide library) |
| `tiffslide` | TiffSlide backend (pure Python, no C dependencies) |
| `torch` | PyTorch integration (`WsiStreamDataset`, `MonitoredLoader`, DDP utilities) |
| `all` | Both backends + torch + albumentations + matplotlib |
| `dev` | Everything in `all` + pytest, ruff, mypy, mkdocs-material |

!!! note "System dependency for OpenSlide"
    If using the OpenSlide backend, you need the C library installed:

    - Ubuntu/Debian: `apt-get install openslide-tools`
    - macOS: `brew install openslide`

## Minimal example

```python
from wsistream.pipeline import PatchPipeline
from wsistream.backends import OpenSlideBackend
from wsistream.tissue import OtsuTissueDetector
from wsistream.sampling import RandomSampler

pipeline = PatchPipeline(
    slide_paths="/path/to/slides",  # directory or list of files
    backend=OpenSlideBackend(),
    tissue_detector=OtsuTissueDetector(),
    sampler=RandomSampler(patch_size=256, num_patches=10),
    pool_size=1,
    patches_per_slide=10,
)

results = list(pipeline)
for result in results:
    print(result.image.shape, result.coordinate.x, result.coordinate.y)
```

Each iteration yields a `PatchResult` with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `image` | `np.ndarray` | The patch pixels. Shape `(H, W, 3)`, dtype `uint8` (or `float32` after normalization). |
| `coordinate` | `PatchCoordinate` | Location in the slide: `x`, `y`, `level`, `patch_size`, `mpp`, `slide_path`. |
| `tissue_fraction` | `float` | Fraction of the patch region covered by tissue (from the tissue mask), in `[0, 1]`. |
| `slide_metadata` | `SlideMetadata` or `None` | Dataset-specific metadata (populated when a `DatasetAdapter` is configured). |

Key pipeline parameters:

| Parameter | Description |
|-----------|-------------|
| `pool_size` | Number of slides kept open simultaneously. Patches are interleaved across the pool via round-robin. |
| `patches_per_slide` | Maximum patches extracted from one slide before it is closed and replaced by the next. |
| `patches_per_visit` | Patches read from one slide before advancing to the next in the pool. Higher values improve I/O throughput on network filesystems. Default `1`. |
| `cycle` | When `True`, slides are re-queued after processing, producing an infinite stream for step-based training. |
| `seed` | Random seed for slide-level shuffling. |

See [Architecture](concepts/architecture.md) for a full explanation of the pipeline flow.

## Visualizing results

```python
from wsistream.viz import plot_patch_grid

patches = [r.image for r in results]
plot_patch_grid(patches, ncols=5, save_path="my_patches.png")
```

## Next steps

- [Online Patching](concepts/online-patching.md) -- understand the core concept
- [Architecture](concepts/architecture.md) -- how the pipeline works internally
