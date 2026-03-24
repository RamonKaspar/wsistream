# Prov-GigaPath

> Xu et al., "A whole-slide foundation model for digital pathology from real-world data", Nature, 2024. [DOI: 10.1038/s41586-024-07441-w](https://doi.org/10.1038/s41586-024-07441-w)

## What they do

Prov-GigaPath uses **offline** Otsu-based tissue detection, resizes all slides to a standard resolution, then extracts tiles with a low tissue occupancy threshold.

- **Patch size**: 256x256 at 20x (0.5 um/px)
- **Tissue detection**: Otsu thresholding at a downsampled resolution (e.g., 1024 pixels)
- **Preprocessing**: All WSIs are resized to 0.5 um/px (20x) using pyvips
- **Tissue threshold**: Tiles with occupancy < 0.1 (10%) are discarded
- **Sampling**: All remaining tiles are used for pretraining
- **Architecture**: GigaPath -- a ViT-g/14 tile encoder + LongNet slide encoder
- **Training**: DINOv2 for tile encoder, LongNet for slide-level modeling
- **Data**: 1.38B tiles from 171K H&E and IHC slides (Prov-Path, proprietary)

## wsistream approximation

Prov-GigaPath extracts all non-overlapping tissue tiles at 20x. We use `RandomSampler` rather than `GridSampler` because `GridSampler` yields patches in fixed scan order -- with a finite `patches_per_slide`, it would always select the same spatially biased subset. `RandomSampler` gives uniform coverage across the tissue, better approximating sampling from the full grid. The 10% tissue threshold is notably lower than most other papers.

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
        patch_size=256,  # 256x256 at 20x
        num_patches=-1,  # stream random patches indefinitely (with replacement)
        target_mpp=0.5,  # 20x magnification (0.5 µm/px)
        tissue_threshold=0.1,  # 10% tissue (notably lower than other papers)
    ),
    pool_size=8,
    patches_per_slide=100,
    cycle=True,
)
```

## Deviations

| Step | Paper | wsistream | Match |
|------|-------|-----------|-------|
| Tissue detection | Otsu thresholding at downsampled resolution | `OtsuTissueDetector` | Approximate — same Otsu-based approach, but the paper does not specify enough preprocessing detail to verify exact parity. |
| Tissue threshold | 10% occupancy | `tissue_threshold=0.1` | Exact |
| Magnification | 20x (0.5 um/px) | `target_mpp=0.5` | Exact |
| Patch size | 256x256 | `patch_size=256` | Exact |
| Resolution normalization | pyvips resize to 0.5 um/px before tiling | `target_mpp` selects closest pyramid level | Approximate -- wsistream reads from the closest existing pyramid level rather than resizing the entire slide |
| Extraction | Offline, all tiles | Online random sampling | Different strategy |
