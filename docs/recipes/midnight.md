# Midnight (Kaiko)

> Karasikov et al., "Training state-of-the-art pathology foundation models with orders of magnitude less data", 2025. [arXiv:2504.05186](https://arxiv.org/abs/2504.05186)
>
> Aben et al., "Towards Large-Scale Training of Pathology Foundation Models", 2024. [arXiv:2404.15217](https://arxiv.org/abs/2404.15217)

## What they do

Midnight uses **online patching** — tiles are sampled uniformly at random from arbitrary positions of the WSIs during training, with no pre-extraction to disk. The online patching system was introduced in the earlier Kaiko-FM paper ([Aben et al., 2024](https://arxiv.org/abs/2404.15217)); Midnight adds per-tile HSV filtering (from Virchow) and HED color augmentation on top.

- **Tile size**: 256×256 pixels
- **Magnifications**: 0.25, 0.5, 1.0, 2.0 µm/px (≈40×, 20×, 10×, 5×). The paper does not state how magnifications are selected; we assume uniform.
- **Foreground mask**: the online patching pipeline uses a **U-Net-based foreground segmentation model** at thumbnail scale, trained on annotations provided by the Netherlands Cancer Institute. This model is **not open-sourced**.
- **Foreground threshold**: 40% — a candidate tile must have ≥40% overlap with the foreground mask
- **Per-tile HSV filter**: adopted from Virchow ([Vorontsov et al., 2024](https://arxiv.org/abs/2309.07778)). A tile is accepted only if ≥60% of its pixels have hue in [90, 180], saturation in [8, 255], and value in [103, 255].
- **HED augmentation**: color augmentations in the HED space ([Tellez et al., 2019](https://doi.org/10.1016/j.media.2019.101544)). The paper does **not** state the sigma value. Tellez et al. define "light" as σ=0.05 and "strong" as σ=0.2.
- **Normalization**: mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), scaling pixel values to [−1, 1]. Applied in the DINOv2 data transforms, not the patching pipeline.
- **Training**: DINOv2 with a KDE regularizer replacing KoLeo, ViT-g/14 (1.1B params), initialized from ImageNet-pretrained DINOv2 checkpoints. 1M iterations on 32× H100 GPUs, effective batch size 768.
- **DINOv2 multi-crop**: tiles are fed as 256×256 to DINOv2. The paper states local and global crop output sizes of **98px and 224px** for standard training (scaled to 168px and 392px for high-resolution post-training). Scale ranges, number of crops, and whether standard DINOv2 color augmentations (color jitter, Gaussian blur, solarize) are used alongside HED are not specified in the paper.

## wsistream approximation

The exact paper-matched parts are `tissue_threshold=0.4` for the 40% foreground-overlap requirement and `HSVPatchFilter` for the per-tile ≥60% HSV acceptance rule. We use `HSVTissueDetector` as a coarse thumbnail-stage heuristic to build the foreground mask; the paper instead uses a U-Net-based model that is not publicly available.

The paper uses random sampling (not grid extraction), which `MultiMagnificationSampler` captures at the level of random online patch draws across the four target MPPs. Because wsistream selects the closest existing pyramid level per slide, this is exact only when those MPPs are present in the slide pyramid.

```python
from wsistream.pipeline import PatchPipeline
from wsistream.backends import OpenSlideBackend
from wsistream.tissue import HSVTissueDetector
from wsistream.filters import HSVPatchFilter
from wsistream.sampling import MultiMagnificationSampler
from wsistream.transforms import ComposeTransforms, HEDColorAugmentation

pipeline = PatchPipeline(
    slide_paths=slide_paths,
    backend=OpenSlideBackend(),
    tissue_detector=HSVTissueDetector(
        hue_range=(90, 180),
        sat_range=(8, 255),
        val_range=(103, 255),
    ),
    patch_filter=HSVPatchFilter(
        hue_range=(90, 180),
        sat_range=(8, 255),
        val_range=(103, 255),
        min_pixel_fraction=0.6,  # >=60% of pixels must pass HSV check
    ),
    sampler=MultiMagnificationSampler(
        target_mpps=[0.25, 0.5, 1.0, 2.0],  # ~40x, ~20x, ~10x, ~5x
        patch_size=256,
        num_patches=-1,  # infinite random with replacement
        tissue_threshold=0.4,  # 40% foreground required per patch
    ),
    transforms=ComposeTransforms(transforms=[
        HEDColorAugmentation(sigma=0.05),  # sigma not stated; 0.05 = Tellez "light"
    ]),
    slide_sampling="random",
    pool_size=8,
    patches_per_slide=100,
    cycle=True,
)
```

!!! note "`num_patches=-1` means infinite streaming"
    The sampler generates random coordinates indefinitely (with replacement). This matches Midnight's online patching: each training step draws fresh random crops. The pipeline's `patches_per_slide` controls how many patches are drawn before rotating to the next slide.

### With multi-crop views

To replicate DINOv2's internal multi-crop within wsistream, move HED augmentation to `shared_transforms` and add `views`. The paper confirms local crops are 98px and global crops are 224px for standard training. Scale ranges and crop counts are not stated in the paper — DINOv2 default scales and counts are used below. Whether standard DINOv2 color augmentations (color jitter, Gaussian blur) apply alongside HED is also not specified.

```python
from wsistream.views import ViewConfig, RandomResizedCrop

pipeline = PatchPipeline(
    slide_paths=slide_paths,
    backend=OpenSlideBackend(),
    tissue_detector=HSVTissueDetector(...),
    patch_filter=HSVPatchFilter(...),
    sampler=MultiMagnificationSampler(
        target_mpps=[0.25, 0.5, 1.0, 2.0],
        patch_size=256,
        num_patches=-1,
        tissue_threshold=0.4,
    ),
    shared_transforms=HEDColorAugmentation(sigma=0.05),  # applied once to the 256x256 tile
    views=[
        ViewConfig(
            name="global",
            crop=RandomResizedCrop(size=224, scale=(0.32, 1.0)),
            count=2,   # global_0, global_1 — DINOv2 default: 2 global crops
        ),
        ViewConfig(
            name="local",
            crop=RandomResizedCrop(size=98, scale=(0.05, 0.32)),  # 7×14 for ViT-g/14
            count=8,   # local_0 … local_7 — DINOv2 default: 8 local crops
        ),
    ],
    slide_sampling="random",
    pool_size=8,
    patches_per_slide=100,
    cycle=True,
)
```

!!! note "Per-crop augmentations"
    The paper does not state whether standard DINOv2 photometric augmentations (Gaussian blur, grayscale, solarization) are applied alongside HED. To add them with view-asymmetric probabilities matching DINOv2 defaults, see the [DINOv2-style multi-crop example](../components/views.md#dinov2-style-multi-crop-for-pathology).

## Deviations from paper

| Step | Paper | wsistream | Match |
|------|-------|-----------|-------|
| Foreground mask | U-Net segmentation model (trained on NKI annotations, not open-sourced) | `HSVTissueDetector` | **Approximate** — heuristic substitute for a learned model |
| Foreground threshold | 40% | `tissue_threshold=0.4` | Exact |
| Per-tile HSV filter | ≥60% pixels in hue [90, 180], sat [8, 255], val [103, 255] | `HSVPatchFilter(min_pixel_fraction=0.6)` | Exact |
| Magnifications | 0.25, 0.5, 1.0, 2.0 µm/px | `target_mpps=[0.25, 0.5, 1.0, 2.0]` | Exact (nearest level used when pyramid lacks a match) |
| Magnification weights | Not specified in paper | Uniform (default) | Unknown |
| Sampling strategy | Online random from arbitrary positions | `num_patches=-1` (infinite random with replacement) | Exact |
| Tile size | 256×256 | `patch_size=256` | Exact |
| HED augmentation | HED perturbation, sigma not stated | `HEDColorAugmentation(sigma=0.05)` | **Approximate** — paper does not specify sigma |
| DINOv2 crop output sizes | local 98px, global 224px (paper-stated for standard training) | `size=98` (local), `size=224` (global) | Exact |
| DINOv2 scale ranges | Not specified in paper | DINOv2 defaults (0.32–1.0 global, 0.05–0.32 local) | **Unverified** |
| DINOv2 crop counts | Not specified in paper | 2 global + 8 local (DINOv2 default) | **Unverified** |
| Per-crop color augmentations | Not specified — unknown if used alongside HED | Not included in wsistream config | **Unverified** |
| Normalization | mean/std = 0.5 | Training code | Exact for Kaiko-FM (Aben et al.); not explicitly stated for Midnight |

## Earlier Kaiko-FM pipeline (Aben et al., 2024)

The Kaiko-FM paper introduced the online patching technique that Midnight builds upon. The patching core is identical — same U-Net foreground mask, same 40% threshold, same 256×256 tiles, same four magnification levels, same mean/std=0.5 normalization. The differences are all on the filtering and augmentation side: Kaiko-FM has **no per-tile HSV filter** and **no HED augmentation**. The only augmentations are the standard ones built into DINO/DINOv2 (color jitter, Gaussian blur, solarization, horizontal flip). Kaiko-FM also trained on all 29k TCGA slides (FFPE + Flash-Frozen) with DINO or DINOv2, whereas Midnight uses only the ~12k FFPE subset (plus optionally 80k proprietary NKI slides) with DINOv2 + KDE.

To reproduce the Kaiko-FM pipeline with wsistream, use the same configuration as above but drop `patch_filter` and `transforms`. The same foreground-mask deviation applies.
