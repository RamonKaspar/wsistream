# Midnight (kaiko)

> Karasikov et al., "Training state-of-the-art pathology foundation models with orders of magnitude less data", 2025. [arXiv:2504.05186](https://arxiv.org/abs/2504.05186)
>
> Aben et al., "Towards Large-Scale Training of Pathology Foundation Models", 2024. [arXiv:2404.15217](https://arxiv.org/abs/2404.15217)

## What they do

Midnight uses **online patching** -- patches are sampled uniformly at random from arbitrary positions of the WSIs during training, with no pre-extraction to disk. The earlier Kaiko.ai paper ([Aben et al., 2024](https://arxiv.org/abs/2404.15217)) describes the coarse foreground mask with a U-Net-based foreground segmentation model at thumbnail scale (instead of HSV filtering).

- **Patch size**: 256x256, resized to 224x224 for model input
- **Magnifications**: 0.25, 0.5, 1.0, 2.0 um/px (~40x, ~20x, ~10x, ~5x)
- **Foreground mask**: online patching foreground mask; the cited Kaiko.ai pipeline ([Aben et al., 2024](https://arxiv.org/abs/2404.15217)) uses a U-Net-based foreground segmentation model, which is not open-sourced. Instead we use HSV color-space filtering.
- **Foreground threshold**: 40% -- a candidate patch must have at least 40% foreground in the coarse foreground mask
- **Per-tile filter**: a tile is accepted only if >=60% of its pixels have hue in [90, 180], saturation in [8, 255], value in [103, 255]
- **Augmentation**: HED color augmentation ([Tellez et al., 2019](https://arxiv.org/pdf/1902.06543); exact sigma not stated -- Tellez defines "light" as 0.05 and "strong" as 0.2)
- **Normalization**: mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
- **Architecture**: ViT-g/14
- **Training**: DINOv2 with a KDE regularizer replacing KoLeo

## wsistream approximation

The exact paper-matched parts are `tissue_threshold=0.4` for the 40% foreground-overlap requirement and `HSVPatchFilter` for the per-tile >=60% HSV acceptance rule. We additionally use `HSVTissueDetector` as a coarse thumbnail-stage heuristic to build the foreground mask inside wsistream; the Kaiko.ai pipeline instead describes a U-Net-based foreground segmentation model.

The paper uses random sampling (not grid extraction), which `MultiMagnificationSampler` captures at the level of random online patch draws across the four target MPPs. Because wsistream selects the closest existing pyramid level per slide, this is exact only when those MPPs are present in the slide pyramid.

```python
from wsistream.pipeline import PatchPipeline
from wsistream.backends import OpenSlideBackend
from wsistream.tissue import HSVTissueDetector
from wsistream.filters import HSVPatchFilter
from wsistream.sampling import MultiMagnificationSampler
from wsistream.transforms import ComposeTransforms, HEDColorAugmentation, ResizeTransform

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
        num_patches=-1,  # stream random patches indefinitely (with replacement)
        tissue_threshold=0.4,  # 40% foreground required per patch
    ),
    transforms=ComposeTransforms(transforms=[
        HEDColorAugmentation(sigma=0.05),  # sigma not stated in paper; 0.05 = Tellez "light"
        ResizeTransform(target_size=224),  # 256 -> 224 for model input
    ]),
    slide_sampling="random",
    pool_size=8,
    patches_per_slide=100,
    cycle=True,
)
```

!!! note "`num_patches=-1` means infinite streaming"
    With `num_patches=-1`, the sampler generates random patch coordinates indefinitely (with replacement). It does **not** mean "extract all unique patches." This matches Midnight's online patching: each training step draws fresh random crops. The pipeline's `patches_per_slide` controls how many patches are drawn before rotating to the next slide.

## Deviations

| Step | Paper | wsistream | Match |
|------|-------|-----------|-------|
| Foreground mask | Coarse foreground mask for online patching ([Aben et al., 2024](https://arxiv.org/abs/2404.15217) uses a U-Net-based foreground segmenter) | `HSVTissueDetector` | Approximate -- heuristic thumbnail-stage substitute for the learned foreground mask |
| Foreground threshold | 40% per patch | `tissue_threshold=0.4` | Exact |
| Per-tile HSV filter | >=60% pixels in HSV ranges | `HSVPatchFilter` | Exact |
| Multi-magnification | 0.25, 0.5, 1.0, 2.0 um/px | `MultiMagnificationSampler` | Exact (only when the slide pyramid contains matching MPP levels though) |
| Sampling strategy | Online random from arbitrary positions | `num_patches=-1` (infinite random, with replacement) | Exact |
| Resize | 256 -> 224 | `ResizeTransform(target_size=224)` | Exact |
| Normalization | mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) | `NormalizeTransform` or training code | Exact |
| HED augmentation | HED perturbation (sigma not stated) | `HEDColorAugmentation(sigma=0.05)` | Approximate -- sigma 0.05 is Tellez "light" default; paper does not specify |
