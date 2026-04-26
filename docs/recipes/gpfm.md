# GPFM

> Ma et al., "Towards a generalizable pathology foundation model via unified knowledge distillation", 2024. [arXiv:2407.18449](https://arxiv.org/abs/2407.18449)

## What they do

GPFM uses **offline** CLAM-based tissue segmentation and patch extraction. A key design choice is that patches are extracted at **native resolution** (level 0 of each WSI), intentionally not normalized to a fixed magnification — the paper states this was "implemented to increase the robustness of the FMs to varying resolutions."

- **Patch size**: 512×512 at level 0 (native resolution of each WSI)
- **Resolution**: explicitly NOT normalized. "We did not scale the patches to a uniform resolution, opting instead to use the original resolution of each WSI."
- **Tissue detection**: [CLAM](https://github.com/mahmoodlab/CLAM) toolkit with **dataset-specific presets**. The GPFM repo (`Patching/presets/`) uses different CLAM parameters for different source datasets. For TCGA: `use_otsu=True`, `sthresh=8`, `mthresh=7`, `close=4`, `a_t=16`, `a_h=4`, `max_n_holes=8`. Other datasets differ (e.g. BCNB uses `sthresh=10`, DHMC uses `sthresh=4, mthresh=6`).
- **Sampling**: all non-overlapping tissue patches at level 0 (offline grid extraction)
- **Normalization**: ImageNet statistics — mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
- **Local-global self-distillation**: 512×512 patches are saved to disk, then used in the unified knowledge distillation framework. The paper does not enumerate crop sizes, crop scale ranges, crop counts, or per-crop augmentations. The multi-crop example below is therefore a DINOv2-style approximation, not an exact paper reproduction.
- **Architecture**: ViT-L/14
- **Training**: unified knowledge distillation — DINOv2 self-distillation + expert distillation from UNI, CONCH, and Phikon simultaneously
- **Data**: 190M patches from 72K publicly available WSIs across 33 datasets and 34 tissue types

## wsistream approximation

GPFM extracts **all** non-overlapping tissue patches at level 0 offline. This is fundamentally a `GridSampler` workflow. However, `GridSampler` yields patches in fixed scan order, so with a finite `patches_per_slide` it always selects the same top-left-biased subset. `RandomSampler` avoids that spatial bias, but does not guarantee full coverage — it samples with replacement, so some locations may be visited multiple times while others are missed. Neither is a perfect online substitute for offline full-grid extraction.

The GPFM repo uses dataset-specific CLAM presets. For TCGA data, the preset uses `use_otsu=True` and smaller area thresholds (`a_t=16`, `a_h=4`) compared to `CLAMTissueDetector` defaults.

```python
from wsistream.pipeline import PatchPipeline
from wsistream.backends import OpenSlideBackend
from wsistream.tissue import CLAMTissueDetector
from wsistream.sampling import RandomSampler

pipeline = PatchPipeline(
    slide_paths=slide_paths,
    backend=OpenSlideBackend(),
    tissue_detector=CLAMTissueDetector(
        use_otsu=True,  # GPFM TCGA preset enables Otsu
        a_t=16,         # GPFM TCGA preset (wsistream default: 100)
        a_h=4,          # GPFM TCGA preset (wsistream default: 16)
    ),
    sampler=RandomSampler(
        patch_size=512,  # 512x512 at native resolution
        num_patches=-1,
        level=0,  # native resolution (no magnification normalization)
    ),
    pool_size=8,
    patches_per_slide=100,
    cycle=True,
)
```

!!! note "No `target_mpp`"
    GPFM intentionally uses `level=0` (not `target_mpp`), because they do not normalize to a fixed physical resolution. Different slides may have different µm/px values at level 0.

### With multi-crop views

GPFM uses local-global self-distillation, but the paper does not publish the exact crop configuration. The view configuration below is a DINOv2-style approximation for ViT-L/14: DINOv2 default crop scales and counts are used, and the 98px local crop size follows the DINOv2 ViT-L/14 training config. Do not treat this block as an exact reproduction of the released training run.

```python
from wsistream.views import ViewConfig, RandomResizedCrop

pipeline = PatchPipeline(
    slide_paths=slide_paths,
    backend=OpenSlideBackend(),
    tissue_detector=CLAMTissueDetector(use_otsu=True, a_t=16, a_h=4),
    sampler=RandomSampler(patch_size=512, num_patches=-1, level=0),
    views=[
        ViewConfig(
            name="global",
            crop=RandomResizedCrop(size=224, scale=(0.32, 1.0)),
            count=2,   # global_0, global_1 — DINOv2 default: 2 global crops
        ),
        ViewConfig(
            name="local",
            crop=RandomResizedCrop(size=98, scale=(0.05, 0.32)),  # 7×14 for ViT-L/14
            count=8,   # local_0 … local_7 — DINOv2 default: 8 local crops
        ),
    ],
    pool_size=8,
    patches_per_slide=100,
    cycle=True,
)
```

!!! note "Crop and augmentation details"
    The paper does not enumerate per-crop augmentations or crop hyperparameters for the local-global branch. The code above uses DINOv2-style defaults only as a practical approximation.

## Deviations from paper

| Step | Paper | wsistream | Match |
|------|-------|-----------|-------|
| Tissue detection | CLAM with dataset-specific presets | `CLAMTissueDetector` (TCGA preset shown) | **Approximate** — paper uses per-dataset presets; we show TCGA preset only |
| Tissue threshold | CLAM contour-based extraction (patches inside tissue contours) | `RandomSampler` tissue_threshold (default 0.4) | **Approximate** — different mechanism; CLAM checks contour containment, wsistream checks mask overlap fraction |
| Patch size | 512×512 | `patch_size=512` | Exact |
| Resolution | Level 0, native (not normalized) | `level=0` | Exact |
| Extraction | Offline non-overlapping grid (all tissue patches) | Online random sampling (with replacement) | **Different** — random sampling does not guarantee full coverage; see note above |
| Normalization | ImageNet mean/std | Training code | Exact (not part of wsistream) |
| Local-global crop sizes / scales | Not enumerated in paper | DINOv2-style scales/counts; local size 98 from DINOv2 ViT-L/14 config | **Unverified** — paper does not state crop sizes or scale ranges |
| Per-crop augmentations | Not enumerated in paper | Training code | **Unverified** — paper does not state the exact augmentation chain |
| Knowledge distillation | UNI + CONCH + Phikon experts | N/A | Training code, not data pipeline |
