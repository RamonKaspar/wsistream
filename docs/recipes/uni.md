# UNI

> Chen et al., "A General-Purpose Self-Supervised Model for Computational Pathology", Nature Medicine, 2024. [DOI: 10.1038/s41591-024-02857-3](https://doi.org/10.1038/s41591-024-02857-3)

## What they do

UNI uses **offline** CLAM-based tissue segmentation followed by non-overlapping patch extraction at 20× magnification (0.5 µm/px).

- **Patch sizes**: 256×256 (75.8M patches) and 512×512 (24.3M patches) at 20× (0.5 µm/px)
- **Tissue detection**: [CLAM](https://github.com/mahmoodlab/CLAM) toolkit — saturation-channel thresholding in HSV, median blur, morphological closing, contour filtering with area thresholds. Exact CLAM parameters are not specified in the paper.
- **Sampling**: non-overlapping tissue patches extracted offline; ~1000 patches per WSI on average (100M patches from 100K WSIs)
- **Normalization**: ImageNet — mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
- **DINOv2 multi-crop**: extracted tiles (256 or 512) are fed to DINOv2, which internally produces global crops (224×224) and local crops (96×96). Standard DINOv2 augmentations apply (color jitter, grayscale, blur, solarization, horizontal flip).
- **Architecture**: ViT-L/16 (0.3B params)
- **Training**: DINOv2 (DINO + iBOT + KoLeo), 125K iterations, batch size 3072, fp16, PyTorch-FSDP on 32× A100 GPUs
- **Data**: Mass-100K — ~100M patches from 100K H&E WSIs across 20 tissue types (MGH, BWH, GTEx; no TCGA)

## wsistream approximation

UNI extracts all non-overlapping tissue patches at 20× offline. This is fundamentally a `GridSampler` workflow. However, `GridSampler` yields patches in fixed scan order, so with a finite `patches_per_slide` it always selects the same top-left-biased subset. `RandomSampler` avoids that spatial bias but samples with replacement and does not guarantee full coverage. Neither is a perfect online substitute for offline full-grid extraction.

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
        patch_size=256,  # 256x256 at 20x (paper also uses 512x512 for some slides)
        num_patches=-1,
        target_mpp=0.5,  # 20x magnification (0.5 µm/px)
    ),
    pool_size=8,
    patches_per_slide=100,
    cycle=True,
)
```

## Deviations from paper

| Step | Paper | wsistream | Match |
|------|-------|-----------|-------|
| Tissue detection | CLAM (exact parameters not specified) | `CLAMTissueDetector` (wsistream defaults) | **Approximate** — same algorithm, but paper does not specify CLAM parameters to verify parity |
| Patch sizes | 256×256 and 512×512 | `patch_size=256` only | **Partial** — paper uses both sizes; wsistream uses one |
| Magnification | 20× (0.5 µm/px) | `target_mpp=0.5` | Exact |
| Extraction | Offline non-overlapping grid (all tissue patches) | Online random sampling (with replacement) | **Different** — random sampling does not guarantee full coverage |
| Normalization | ImageNet mean/std | Training code | Exact (not part of wsistream) |
| DINOv2 augmentations | Standard DINOv2 multi-crop + augmentations | Training code | N/A (not part of wsistream) |

## UNI2-h (Chen et al., 2025)

UNI2-h scales up the original UNI approach to 200M+ image tiles sampled from 350K+ diverse H&E and IHC slides (Mass General Brigham), using a ViT-H/14 with 8 register tokens (681M params). Training uses the same DINOv2 recipe (DINO + iBOT + KoLeo) with bf16 mixed-precision on A100 GPUs. Inference uses 224×224 input with ImageNet normalization, producing 1536-dimensional embeddings.

Detailed preprocessing methodology for UNI2-h (tissue detection parameters, patch extraction specifics, tissue thresholds) is not publicly documented at the time of writing. The key known differences from UNI are the larger model (ViT-H/14 vs ViT-L/16), the addition of IHC slides alongside H&E, register tokens, and the substantially larger training set (350K+ vs 100K WSIs). The same wsistream approximation above applies, with the caveat that UNI2-h's exact preprocessing details cannot be verified.
