# Virchow

> Vorontsov et al., "A foundation model for clinical-grade computational pathology and rare cancers detection", Nature Medicine, 2024. [DOI: 10.1038/s41591-024-03141-0](https://doi.org/10.1038/s41591-024-03141-0)

## What they do

Virchow uses **offline** HSV-based foreground detection followed by non-overlapping tile extraction at a single magnification.

- **Patch size**: 224x224 at 20x (0.5 um/px)
- **Tissue detection**: Each WSI is downsampled 16x, then per-pixel HSV filtering with hue in [90, 180], saturation in [8, 255], value in [103, 255]
- **Tissue threshold**: 25% (tiles with at least 25% tissue area are kept)
- **Sampling**: All non-overlapping foreground tiles are collected; during training, one WSI is selected per GPU and 256 tiles are randomly sampled from that WSI
- **Augmentation**: Not detailed in the paper for the data pipeline; DINOv2 training framework handles augmentations
- **Architecture**: ViT-H/14 (632M params)
- **Training**: DINOv2
- **Data**: 1.5M H&E WSIs from diverse institutions

## wsistream approximation

Virchow's HSV ranges and 25% tissue threshold map naturally to `HSVTissueDetector` + `tissue_threshold=0.25`, but the exact paper pipeline still differs: the paper thresholds a fixed 16x downsample, precomputes all non-overlapping foreground tiles, and then samples 256 from one WSI per GPU.

```python
from wsistream.pipeline import PatchPipeline
from wsistream.backends import OpenSlideBackend
from wsistream.tissue import HSVTissueDetector
from wsistream.sampling import RandomSampler

pipeline = PatchPipeline(
    slide_paths=slide_paths,
    backend=OpenSlideBackend(),
    tissue_detector=HSVTissueDetector(
        hue_range=(90, 180),
        sat_range=(8, 255),
        val_range=(103, 255),
    ),
    sampler=RandomSampler(
        patch_size=224,  # 224x224 at 20x
        num_patches=-1,  # stream random patches indefinitely (with replacement)
        target_mpp=0.5,  # 20x magnification (0.5 µm/px)
        tissue_threshold=0.25,  # 25% tissue required (lower than most papers)
    ),
    pool_size=1,  # 1 WSI per GPU (matches paper's batch construction)
    patches_per_slide=256,  # 256 tiles per WSI
    slide_sampling="random",
    cycle=True,
)
```

## Deviations

| Step | Paper | wsistream | Match |
|------|-------|-----------|-------|
| Tissue detection | HSV per-pixel filtering at 16x downsample | `HSVTissueDetector` with same ranges | Approximate (HSV ranges match; downsample is determined by `thumbnail_size`, not a fixed 16x factor) |
| Tissue threshold | 25% | `tissue_threshold=0.25` | Exact |
| Magnification | 20x (0.5 um/px) | `target_mpp=0.5` | Exact |
| Patch size | 224x224 | `patch_size=224` | Exact |
| Batch construction | 1 WSI per GPU, 256 tiles | `pool_size=1`, `patches_per_slide=256` | Exact (with DDP slide partitioning), but tiles are drawn online rather than from a precomputed foreground set |
| Extraction | Offline, all non-overlapping tiles | Online random sampling | Different strategy |
