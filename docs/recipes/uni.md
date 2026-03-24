# UNI

> Chen et al., "Towards a General-Purpose Foundation Model for Computational Pathology", Nature Medicine, 2024. [DOI: 10.1038/s41591-024-02857-3](https://doi.org/10.1038/s41591-024-02857-3)

## What they do

UNI uses **offline** CLAM-based tissue segmentation followed by non-overlapping patch coordinate extraction and sampling of approximately 800 patches per WSI.

- **Patch size**: 256x256 at 20x (0.5 um/px)
- **Tissue detection**: [CLAM](https://github.com/mahmoodlab/CLAM) toolbox -- binary thresholding of the saturation channel in HSV, median blur, morphological closing, contour filtering with area thresholds
- **Sampling**: ~800 patches sampled per WSI from the extracted non-overlapping tissue patch set
- **Resize**: 256 -> 224 for model input
- **Augmentation**: Standard DINOv2 augmentations (multi-crop, color jitter, grayscale, blur, solarization) -- handled inside DINOv2 training code, not the data pipeline
- **Normalization**: ImageNet -- mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
- **Architecture**: ViT-L/16
- **Training**: DINOv2 with fp16 and PyTorch-FSDP
- **Data**: Mass-100K -- ~100M patches from 100K H&E WSIs (private: MGH, BWH, GTEx)

## wsistream approximation

UNI extracts a non-overlapping grid of all tissue patches, then samples ~800 per WSI from that grid. We use `RandomSampler` rather than `GridSampler` here because `GridSampler` yields patches in fixed scan order (top-left to bottom-right) -- with `patches_per_slide=800`, it would always select the same spatially biased subset. `RandomSampler` gives uniform coverage across the whole tissue region, which better approximates sampling ~800 patches from the full grid.

```python
from wsistream.pipeline import PatchPipeline
from wsistream.backends import OpenSlideBackend
from wsistream.tissue import CLAMTissueDetector
from wsistream.sampling import RandomSampler
from wsistream.transforms import ResizeTransform

pipeline = PatchPipeline(
    slide_paths=slide_paths,
    backend=OpenSlideBackend(),
    tissue_detector=CLAMTissueDetector(),
    sampler=RandomSampler(
        patch_size=256,  # 256x256 at 20x
        num_patches=-1,  # stream random patches indefinitely (with replacement)
        target_mpp=0.5,  # 20x magnification (0.5 µm/px)
        tissue_threshold=0.4,
    ),
    transforms=ResizeTransform(target_size=224),  # 256 -> 224 for ViT-L/16 input
    pool_size=8,
    patches_per_slide=800,
    cycle=True,
)
```

!!! note "Augmentations"
    UNI uses standard DINOv2 augmentations, which are applied inside the DINOv2 training framework (the `DataAugmentationDINO` class), not in the data pipeline. wsistream delivers raw patches; augmentations are handled downstream.

## Deviations

| Step | Paper | wsistream | Match |
|------|-------|-----------|-------|
| Tissue detection | [CLAM](https://github.com/mahmoodlab/CLAM) saturation thresholding | `CLAMTissueDetector` | Exact |
| Patch extraction | Offline non-overlapping grid, ~800/WSI | Online random, `patches_per_slide=800` |  Different strategy; same spatial coverage per slide |
| Magnification | 20x (0.5 um/px) | `target_mpp=0.5` | Exact |
| Resize to 224 | Yes | `ResizeTransform(target_size=224)` | Exact |
| DINOv2 augmentations | In training code | Not in wsistream | N/A (training code) |
