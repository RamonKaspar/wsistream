# Online Patching

## The problem with offline patching

WSIs are large -- typically tens of thousands of pixels in each dimension, sometimes exceeding 100,000. Training pathology foundation models requires extracting patches from these slides, and most approaches do this as a preprocessing step before training begins:

- **UNI** extracted ~100M patches from 100K WSIs ([Chen et al., 2024](https://arxiv.org/abs/2308.15474))
- **Prov-GigaPath** pre-tiled 171K WSIs into 1.3 billion tiles ([Xu et al., 2024](https://arxiv.org/abs/2407.18886))

Pre-extraction requires storage proportional to the number of patches extracted, and a preprocessing phase before training can start.

## The online alternative

Online patching skips pre-extraction entirely. Patches are read directly from the original WSI files during training, on demand. This approach was proposed for FM pre-training by Kaiko ([Aben et al., 2024](https://arxiv.org/abs/2404.15217)) and refined in Midnight ([Karasikov et al., 2025](https://arxiv.org/abs/2504.05186)).

The general approach:

1. Detect tissue regions on a low-resolution thumbnail of the slide
2. Propose random (x, y) coordinates and check whether they fall within tissue (rejection sampling)
3. Read the patch at the accepted coordinate directly from the WSI file
4. Apply augmentations and feed it to the model

No patches are saved to disk. Because coordinates are sampled stochastically, the model sees different patches in every training run (assuming no fixed seed).

## How wsistream implements this

`wsistream` implements online patching through its `PatchPipeline`, which adds several features on top of the basic approach described above:

- **Pool-based slide interleaving**: rather than processing one slide at a time, the pipeline keeps multiple slides open simultaneously and round-robins across them. This ensures patches from different slides are interleaved in the output stream.
- **Per-slide budgets**: each slide has a `patches_per_slide` limit. Once reached, the slide is closed and replaced by the next one from the queue. This is essential when using infinite samplers (`num_patches=-1`).
- **Cycle mode**: when `cycle=True`, slides are re-queued after all have been processed, producing an infinite stream suitable for step-based training.
- **Two-stage tissue filtering**: a coarse `TissueDetector` runs once per slide on a thumbnail, followed by a fine-grained `PatchFilter` that checks every extracted patch at the sampled resolution.

See [Architecture](architecture.md) for a full breakdown of the pipeline flow.

## Step-based training

With offline patching, one epoch means one pass through the entire pre-extracted dataset. With online patching, patches are randomly sampled from tissue regions — there is no guarantee of seeing the same patch twice, so a traditional "epoch" (one complete pass over every sample) is not meaningful. Training is therefore defined by a number of **steps** rather than epochs. For example, the Midnight model was trained for 1M iterations with a batch size of 12 per GPU across 32 GPUs ([Karasikov et al., 2025](https://arxiv.org/abs/2504.05186)).

Learning rate schedules, checkpointing, and logging are all defined in terms of steps, not epochs.

## Tradeoffs

| | Offline | Online |
|---|---------|--------|
| **Storage** | Proportional to the number of patches extracted | Only the original WSI files |
| **Preprocessing** | Required before training | None |
| **Data diversity** | Fixed set of patches | Stochastic -- different patches each run |
| **I/O during training** | Reading pre-extracted images | Reading from WSI files via slide-reading libraries |
| **Flexibility** | Re-extraction needed when changing patch size or magnification | Change configuration and retrain |

The Midnight paper demonstrated that online patching does not compromise model performance and can even improve it through increased data diversity ([Karasikov et al., 2025](https://arxiv.org/abs/2504.05186)).
