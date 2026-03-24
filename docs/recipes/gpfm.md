# GPFM

> Ma et al., "Towards a generalizable pathology foundation model via unified knowledge distillation", 2024. [arXiv:2407.18449](https://arxiv.org/abs/2407.18449)

## What they do

GPFM uses **offline** CLAM-based tissue segmentation with a key distinction: patches are extracted at **native resolution** (level 0 of each WSI), not normalized to a fixed magnification.

- **Patch size**: 512x512 at level 0 (native resolution of each WSI)
- **Tissue detection**: [CLAM](https://github.com/mahmoodlab/CLAM) toolbox (the GPFM repo forks CLAM's `wsi_core/` and `Patching/` directories)
- **Sampling**: All non-overlapping tissue patches at level 0
- **Resolution**: Explicitly NOT normalized to a uniform um/px. The paper states: "we did not scale the patches to a uniform resolution, opting instead to use the original resolution of each WSI"
- **Architecture**: ViT-L/14
- **Training**: Unified Knowledge Distillation -- DINOv2 self-distillation + expert distillation from [UNI](https://doi.org/10.1038/s41591-024-02857-3), [CONCH](https://arxiv.org/abs/2307.12914), and [Phikon](https://www.medrxiv.org/content/10.1101/2023.07.21.23292757) simultaneously
- **Data**: 190M patches from 72K WSIs (TCGA + private centers)

## wsistream approximation

GPFM extracts all non-overlapping tissue patches at level 0. We use `RandomSampler` rather than `GridSampler` because `GridSampler` yields patches in fixed scan order -- with a finite `patches_per_slide`, it would always select the same spatially biased subset. `RandomSampler` gives uniform coverage across the tissue, better approximating sampling from the full grid. `CLAMTissueDetector` matches their tissue detection exactly.

```python
from wsistream.pipeline import PatchPipeline
from wsistream.backends import OpenSlideBackend
from wsistream.tissue import CLAMTissueDetector
from wsistream.sampling import RandomSampler

pipeline = PatchPipeline(
    slide_paths=slide_paths,
    backend=OpenSlideBackend(),
    tissue_detector=CLAMTissueDetector(),
    sampler=RandomSampler(
        patch_size=512,  # 512x512 at native resolution
        num_patches=-1,  # stream random patches indefinitely (with replacement)
        level=0,  # native resolution (no magnification normalization)
        tissue_threshold=0.4,
    ),
    pool_size=8,
    patches_per_slide=100,
    cycle=True,
)
```

!!! note "No `target_mpp`"
    GPFM intentionally uses `level=0` (not `target_mpp`), because they do not normalize to a fixed physical resolution. Different slides may have different um/px values at level 0.

## Deviations

| Step | Paper | wsistream | Match |
|------|-------|-----------|-------|
| Tissue detection | [CLAM](https://github.com/mahmoodlab/CLAM) | `CLAMTissueDetector` | Exact |
| Patch size | 512x512 | `patch_size=512` | Exact |
| Resolution | Level 0, native (not normalized) | `level=0` | Exact |
| Extraction | Offline non-overlapping grid | Online random sampling | Different strategy |
| Knowledge distillation | UNI + CONCH + Phikon experts | Not in wsistream | N/A (training code) |
