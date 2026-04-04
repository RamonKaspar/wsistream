# Prov-GigaPath

> Xu et al., "A whole-slide foundation model for digital pathology from real-world data", Nature, 2024. [DOI: 10.1038/s41586-024-07441-w](https://doi.org/10.1038/s41586-024-07441-w)

## What they do

Prov-GigaPath uses **offline** Otsu-based tissue segmentation, resolution normalization via pyvips, and tile extraction with a notably low occupancy threshold.

- **Tile size**: 256×256 at 0.5 µm/px (20×)
- **Resolution normalization**: all WSIs are resized to 0.5 µm/px using **pyvips** before tiling. "This step is necessary because some slides have higher resolution depending on the scanner settings."
- **Tissue detection**: Otsu thresholding on **luminance** (simple RGB channel mean) at a downsampled resolution (e.g. 1,024 pixels). Foreground = pixels with luminance below the Otsu threshold. Implemented with `skimage.filters.threshold_otsu()` in the [source code](https://github.com/prov-gigapath/prov-gigapath).
- **Tissue threshold**: 10% — tiles with occupancy < 0.1 are discarded. Occupancy is computed as the fraction of foreground pixels per tile.
- **Sampling**: all remaining non-overlapping tiles are used for pretraining
- **Normalization**: ImageNet statistics — mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225). At inference, tiles are resized to 256, center-cropped to 224.
- **Architecture**: GigaPath — ViT-g/14 tile encoder + LongNet slide encoder
- **Training**: DINOv2 for tile encoder (standard settings, batch size 12/GPU, effective 384), LongNet for slide-level modeling
- **Data**: 1.38B tiles from 171K H&E and IHC slides (Providence health network, proprietary)

## wsistream approximation

Prov-GigaPath extracts **all** non-overlapping tissue tiles at 20× offline. This is fundamentally a `GridSampler` workflow. However, `GridSampler` yields patches in fixed scan order, so with a finite `patches_per_slide` it always selects the same top-left-biased subset. `RandomSampler` avoids that spatial bias, but samples with replacement and does not guarantee full coverage. Neither is a perfect online substitute for offline full-grid extraction.

The paper normalizes all WSIs to 0.5 µm/px with pyvips before tiling. wsistream instead selects the closest existing pyramid level via `target_mpp`, which avoids a full-slide resize but means the effective resolution may not be exactly 0.5 µm/px.

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
        num_patches=-1,
        target_mpp=0.5,  # 20x magnification (0.5 µm/px)
        tissue_threshold=0.1,  # 10% occupancy (notably lower than other papers)
    ),
    pool_size=8,
    patches_per_slide=100,
    cycle=True,
)
```

## Deviations from paper

| Step | Paper | wsistream | Match |
|------|-------|-----------|-------|
| Tissue detection | Otsu on luminance (RGB channel mean) via `skimage.filters.threshold_otsu` | `OtsuTissueDetector` (Otsu on weighted grayscale via `cv2.cvtColor`) | **Approximate** — both are Otsu-based, but the grayscale conversion differs (simple RGB mean vs. weighted 0.299R+0.587G+0.114B) |
| Tissue threshold | 10% occupancy | `tissue_threshold=0.1` | Exact |
| Resolution normalization | pyvips resize of entire WSI to 0.5 µm/px before tiling | `target_mpp=0.5` selects closest pyramid level | **Approximate** — wsistream reads from the nearest existing level rather than resampling the full slide |
| Tile size | 256×256 | `patch_size=256` | Exact |
| Extraction | Offline, all non-overlapping tiles | Online random sampling (with replacement) | **Different** — random sampling does not guarantee full coverage |
| Normalization | ImageNet mean/std | Training code | Exact (not part of wsistream) |
