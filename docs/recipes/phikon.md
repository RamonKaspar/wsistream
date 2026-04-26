# Phikon

> Filiot et al., "Phikon-v2, A large and public feature extractor for biomarker prediction", NeurIPS, 2024.
>
> Filiot et al., "Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling", 2023. [medRxiv:2023.07.21.23292757](https://www.medrxiv.org/content/10.1101/2023.07.21.23292757)

## What they do

Phikon-v2 uses **offline** tissue segmentation with a proprietary U-Net, followed by tile extraction at a single magnification.

- **Tile size**: 224×224 at 20× (0.5 µm/px)
- **Tissue detection**: an **in-house bi-directional U-Net** segments tissue and discards background and artifacts at 2.5× magnification. This model is **not open-sourced**.
- **Tissue threshold**: 60% — tiles must have a minimal tissue matter proportion of 60%
- **Normalization**: ImageNet — mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
- **DINOv2 multi-crop**: tiles are 224×224 and fed to DINOv2. From Extended Table 5 in the paper: **2 global crops** (size 224, scale 0.32–1.0) and **8 local crops** (size 96, scale 0.05–0.32). Local 96px = 6 × 16 for ViT-L/16. No registers.
- **Architecture**: ViT-L/16 (307M params), no registers
- **Training**: DINOv2 (DINO + iBOT + KoLeo). 250K iterations, batch size 4096, 32 nodes × 4 V100 GPUs (128 V100s total), 11K GPU hours.  Released model taken at iteration 100K (400M tiles seen, ~93% of dataset).
- **Data**: PANCAN-XL — 456M tiles from 58K publicly available WSIs across 132 public datasets + 4 internal (TCGA, CPTAC, GTEx, and others), covering 30+ cancer sites

## wsistream approximation

Phikon-v2's tissue detection uses a proprietary U-Net at 2.5× magnification that is not publicly available. We substitute `OtsuTissueDetector` as a heuristic. Note that Phikon-v2 extracts tiles at 224×224 directly (not 256 with a resize), matching the DINOv2 global crop size.

```python
from wsistream.pipeline import PatchPipeline
from wsistream.backends import OpenSlideBackend
from wsistream.tissue import OtsuTissueDetector
from wsistream.sampling import RandomSampler

pipeline = PatchPipeline(
    slide_paths=slide_paths,
    backend=OpenSlideBackend(),
    tissue_detector=OtsuTissueDetector(),
    sampler=RandomSampler(
        patch_size=224,  # 224x224 at 20x (no separate resize step)
        num_patches=-1,
        target_mpp=0.5,  # 20x magnification (0.5 µm/px)
        tissue_threshold=0.6,  # 60% tissue required
    ),
    pool_size=8,
    patches_per_slide=100,
    cycle=True,
)
```

### With multi-crop views

Phikon-v2 feeds 224×224 tiles to DINOv2's multi-crop. Extended Table 5 of the paper states the crop configuration. Because the source tile (224px) equals the global crop output size (224px), global crops at scale < 1.0 involve upsampling — this is intentional and matches the paper's setup.

```python
from wsistream.views import ViewConfig, RandomResizedCrop

pipeline = PatchPipeline(
    slide_paths=slide_paths,
    backend=OpenSlideBackend(),
    tissue_detector=OtsuTissueDetector(),
    sampler=RandomSampler(patch_size=224, num_patches=-1, target_mpp=0.5,
                          tissue_threshold=0.6),
    views=[
        ViewConfig(
            name="global",
            crop=RandomResizedCrop(size=224, scale=(0.32, 1.0)),
            count=2,  # global_0, global_1 — paper: 2 global crops (Extended Table 5)
        ),
        ViewConfig(
            name="local",
            crop=RandomResizedCrop(size=96, scale=(0.05, 0.32)),  # paper: 8 local, 96px = 6×16 for ViT-L/16
            count=8,  # local_0 … local_7 — paper: 8 local crops (Extended Table 5)
        ),
    ],
    pool_size=8,
    patches_per_slide=100,
    cycle=True,
)
```

!!! note "Per-crop augmentations"
    Phikon-v2 uses standard DINOv2 photometric augmentations (color jitter, Gaussian blur, grayscale, solarization, horizontal flip) applied per crop. To add them with view-asymmetric probabilities matching DINOv2 defaults, see the [DINOv2-style multi-crop example](../components/views.md#dinov2-style-multi-crop-for-pathology).

## Deviations from paper

| Step | Paper | wsistream | Match |
|------|-------|-----------|-------|
| Tissue detection | In-house bi-directional U-Net at 2.5× (not open-sourced) | `OtsuTissueDetector` | **Approximate** — heuristic substitute for a learned model |
| Tissue threshold | 60% | `tissue_threshold=0.6` | Exact |
| Tile size | 224×224 | `patch_size=224` | Exact |
| Magnification | 20× (0.5 µm/px) | `target_mpp=0.5` | Exact |
| Extraction | Offline, all non-overlapping tiles | Online random sampling (with replacement) | **Different** — random sampling does not guarantee full coverage |
| DINOv2 global crop | size 224, scale (0.32, 1.0), count 2 (Extended Table 5) | `size=224, scale=(0.32, 1.0), count=2` | Exact |
| DINOv2 local crop | size 96, scale (0.05, 0.32), count 8 (Extended Table 5) | `size=96, scale=(0.05, 0.32), count=8` | Exact |
| Normalization | ImageNet mean/std | Training code | Exact (not part of wsistream) |

## Original Phikon (Filiot et al., 2023)

The original Phikon uses a simpler pipeline: tiles are extracted at 224×224 and 20× using OpenSlide's `DeepZoomGenerator` (tile_size=224, overlap=0) with basic foreground filtering ("all matter tiles"). It was trained with **iBOT** (not DINOv2) on a ViT-B/16 using ~40M tiles from TCGA (16 cancer types). Phikon-v2 scales this up in both data (456M tiles from 58K WSIs) and model (ViT-L) while switching from iBOT to DINOv2 and adding a U-Net tissue segmenter.

The same wsistream approximation above applies to Phikon, except that Phikon's simpler foreground filtering is likely closer to `OtsuTissueDetector` than Phikon-v2's U-Net.
