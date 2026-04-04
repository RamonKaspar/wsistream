# Virchow

> Zimmermann, Vorontsov et al., "Virchow2: Scaling Self-Supervised Mixed-Modality Foundation Models in Pathology", 2025. [arXiv:2408.00738](https://arxiv.org/abs/2408.00738)
>
> Vorontsov et al., "A foundation model for clinical-grade computational pathology and rare cancers detection", Nature Medicine, 2024. [DOI: 10.1038/s41591-024-03141-0](https://doi.org/10.1038/s41591-024-03141-0)

## What they do

Virchow2 extends the original Virchow with domain-specific modifications to DINOv2: extended-context translation (ECT) replacing standard crop-and-resize, KDE regularization replacing KoLeo, vertical flips, and optionally removing solarization. Tissue detection also changed from a simple HSV filter (Virchow 1) to a trained fully-convolutional network (Virchow 2).

- **Tile size**: 392×392 pixel source regions. DINOv2 with ECT extracts global views (224×224) and local views (98×98) from within these regions, using minimal resizing to avoid distorting cell morphology.
- **Magnifications**: tiles drawn from 5×, 10×, 20×, 40× with online balancing (approximately 20%, 40%, 20%, 20% for 40×, 20×, 10×, 5× respectively)
- **Tissue detection**: a **trained fully-convolutional network** with Otsu thresholding post-processing at thresholds (0.4, 0.5). This is **not open-sourced**.
- **Tissue threshold**: 65% — tiles must contain at least 65% tissue by area
- **Augmentations**: ECT (extended-context translation), horizontal and vertical flips, grayscale, color jitter. Solarization included for Virchow2, removed for Virchow2G. No stain-specific augmentation.
- **Architecture**: ViT-H/14 with 4 registers (Virchow2), ViT-G/14 with 8 registers (Virchow2G, 1.85B params)
- **Training**: DINOv2 with KDE replacing KoLeo, 512× V100 GPUs, batch size 4096 (Virchow2) / 3072 (Virchow2G), 2B tiles total
- **Data**: 3.1M WSIs (H&E + IHC) from 225K patients, including 15% external consultation slides

## wsistream approximation

Virchow2 uses a trained FCN for tissue detection that is not publicly available. We substitute `HSVTissueDetector` as a heuristic, using the same HSV ranges that Virchow 1 used and that were also adopted for the Virchow 2 ablation experiments. Virchow2's ECT augmentation reads from 392×392 source regions — to approximate this, we extract 392×392 tiles and let the training framework handle the cropping.

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
        patch_size=392,  # 392x392 source regions for ECT
        num_patches=-1,
        target_mpp=0.5,  # 20x magnification (primary resolution)
        tissue_threshold=0.65,  # 65% tissue required
    ),
    slide_sampling="random",
    pool_size=8,
    patches_per_slide=100,
    cycle=True,
)
```

## Deviations from paper

| Step | Paper | wsistream | Match |
|------|-------|-----------|-------|
| Tissue detection | Trained FCN + Otsu post-processing (not open-sourced) | `HSVTissueDetector` with Virchow 1 HSV ranges | **Approximate** — heuristic substitute for a learned model |
| Tissue threshold | 65% | `tissue_threshold=0.65` | Exact |
| Tile size | 392×392 source regions | `patch_size=392` | Exact |
| Magnifications | Mixed 5×/10×/20×/40× with online balancing | Single `target_mpp=0.5` (20×) | **Approximate** — wsistream samples at one magnification; paper balances across four |
| ECT augmentation | Extended-context translation (crops from 392 region, minimal resize) | Not in wsistream — handled by training code | N/A |
| Batch construction | 4096 total batch, balanced across metadata | Standard DataLoader batching | **Different** — no metadata-based balancing in wsistream |
| Extraction | Offline, all non-overlapping tiles | Online random sampling (with replacement) | Different strategy |

## Original Virchow pipeline (Vorontsov et al., 2024)

Virchow 1 shares the same HSV ranges but differs from Virchow 2 in several ways: tiles are 224×224 (not 392), tissue detection uses a simple HSV per-pixel filter at a fixed 16× downsample (not a trained FCN), the tissue threshold is 25% (not 65%), training is at a single magnification of 20× (0.5 µm/px), and batch construction selects 1 WSI per GPU with 256 tiles per WSI. Training used standard DINOv2 (no ECT, no KDE, no vertical flips) on 1.5M H&E-only WSIs from MSKCC with a ViT-H/14 (632M params).

To approximate the Virchow 1 pipeline, change the sampler to `patch_size=224, target_mpp=0.5, tissue_threshold=0.25` and optionally set `pool_size=1, patches_per_slide=256` to match the 1-WSI-per-GPU batch construction.
