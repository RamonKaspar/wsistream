# Paper Recipes

wsistream can approximate the data pipelines of several pathology foundation models. This is a selected (not exhaustive) collection -- each recipe shows how to configure the pipeline to match a specific paper's preprocessing as closely as possible.

| Paper | Patch size | Resolution | Tissue detection | Tissue threshold | Training data |
|-------|-----------|------------|-----------------|-----------------|---------------|
| [Midnight](midnight.md) | 256x256 | 0.25, 0.5, 1.0, 2.0 mpp | HSV filtering | 40% | 3B+ tiles from 423K WSIs |
| [UNI](uni.md) | 256x256 | 20x (0.5 mpp) | CLAM | 40% | ~100M tiles from 100K WSIs |
| [Virchow](virchow.md) | 224x224 | 20x (0.5 mpp) | HSV filtering | 25% | 1.5M WSIs |
| [GPFM](gpfm.md) | 512x512 | Native (level 0) | CLAM | 40% | 190M tiles from 72K WSIs |
| [Prov-GigaPath](prov-gigapath.md) | 256x256 | 20x (0.5 mpp) | Otsu | 10% | 1.38B tiles from 171K WSIs |

<figure markdown="span">
  ![Recipe comparison](../assets/recipe_comparison.svg)
  <figcaption>Same WSI processed with each recipe's tissue detection and sampling configuration. Top: binary tissue masks. Middle: mask overlay. Bottom: 64 sampled patch locations (colored by pyramid level).</figcaption>
</figure>
