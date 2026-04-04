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
- **DINOv2 multi-crop**: tiles are 224×224, matching the global crop size. DINOv2 produces global crops (224×224, scale 0.32–1.0) and local crops (96×96, scale 0.05–0.32). Standard DINOv2 augmentations apply (color jitter, grayscale, blur, solarization, horizontal flip).
- **Architecture**: ViT-L/16 (307M params), no registers
- **Training**: DINOv2 (DINO + iBOT + KoLeo), 250K iterations, batch size 4096, fp16, 128× V100 GPUs. Released model taken at iteration 100K (400M tiles seen, ~93% of dataset).
- **Data**: PANCAN-XL — 456M tiles from 58K publicly available WSIs across 132+ datasets (TCGA, CPTAC, GTEx, and others), covering 30+ cancer sites

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

## Deviations from paper

| Step | Paper | wsistream | Match |
|------|-------|-----------|-------|
| Tissue detection | In-house bi-directional U-Net at 2.5× (not open-sourced) | `OtsuTissueDetector` | **Approximate** — heuristic substitute for a learned model |
| Tissue threshold | 60% | `tissue_threshold=0.6` | Exact |
| Tile size | 224×224 | `patch_size=224` | Exact |
| Magnification | 20× (0.5 µm/px) | `target_mpp=0.5` | Exact |
| Extraction | Offline, all non-overlapping tiles | Online random sampling (with replacement) | **Different** — random sampling does not guarantee full coverage |
| Normalization | ImageNet mean/std | Training code | Exact (not part of wsistream) |

## Original Phikon (Filiot et al., 2023)

The original Phikon uses a simpler pipeline: tiles are extracted at 224×224 and 20× using OpenSlide's `DeepZoomGenerator` (tile_size=224, overlap=0) with basic foreground filtering ("all matter tiles"). It was trained with **iBOT** (not DINOv2) on a ViT-B/16 using 43M tiles from 6K TCGA WSIs. Phikon-v2 scales this up in both data (456M tiles from 58K WSIs) and model (ViT-L) while switching from iBOT to DINOv2 and adding a U-Net tissue segmenter.

The same wsistream approximation above applies to Phikon, except that Phikon's simpler foreground filtering is likely closer to `OtsuTissueDetector` than Phikon-v2's U-Net.
