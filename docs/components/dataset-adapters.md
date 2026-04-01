# Dataset Adapters

Dataset adapters extract metadata from slide paths. This metadata flows through the pipeline into `PatchResult.slide_metadata` and into `pipeline.stats_dict()` for logging.

## TCGAAdapter

Parses [TCGA barcodes](https://docs.gdc.cancer.gov/Encyclopedia/pages/TCGA_Barcode/) from filenames. TCGA slide filenames encode patient ID, tissue source site, sample type (tumor vs. normal), and more.

Example filename: `TCGA-3L-AA1B-01Z-00-DX1.8923A151-A690-40B7-9E5A-FCBEDFC2394F.svs`

```python
from wsistream.datasets import TCGAAdapter

adapter = TCGAAdapter()
meta = adapter.parse_metadata("/data/TCGA-BRCA/TCGA-3L-AA1B-01Z-00-DX1.svs")

meta.dataset_name   # "TCGA"
meta.patient_id     # "TCGA-3L-AA1B"
meta.cancer_type    # "TCGA-BRCA" (inferred from parent directory)
meta.sample_type    # "Primary Solid Tumor" (sample code 01)
meta.tissue_type    # "3L" (tissue source site code)
meta.extra          # {"tissue_source_site": "3L", "vial": "Z", "portion": "00",
                    #  "slide_id": "DX1", "slide_section": "DX", "is_frozen": False,
                    #  "sample_code": "01", "barcode": "TCGA-3L-AA1B-01Z-00-DX1"}
```

The cancer type is inferred from the parent directory name (e.g., `TCGA-BRCA/`). You can override it:

```python
adapter = TCGAAdapter(cancer_type="LUAD")
```

The adapter also distinguishes diagnostic slides (`DX`) from frozen sections (`TS`, `BS`, `MS`) via the `is_frozen` field in `extra`.

## Downloading TCGA slides

wsistream includes helpers to download TCGA slides directly from the [GDC Data Portal](https://portal.gdc.cancer.gov/). This is useful when setting up a new machine or VM.

### Query available slides

```python
from wsistream.datasets import query_tcga_slides

# See what's available for two cancer types
manifest = query_tcga_slides(
    cancer_types=["TCGA-BRCA", "TCGA-LUAD"],
    slide_type="diagnostic",       # "diagnostic" (FFPE/DX), "frozen" (TS/BS/MS), or "all"
    max_per_cancer_type=10,        # stratified cap per cancer type (None = all)
    seed=42,                       # reproducible subsampling
)
# Found 20 slides (18.3 GB):
#   TCGA-BRCA: 10 slides (10.2 GB)
#   TCGA-LUAD: 10 slides (8.1 GB)
```

`query_tcga_slides` returns a list of file records (dicts) without downloading anything. Each record contains `file_id`, `filename`, `file_size`, `cancer_type`, `md5sum`, and `state`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cancer_types` | str, list, or None | `None` | TCGA project IDs (e.g., `"TCGA-BRCA"`). `None` = all projects. |
| `slide_type` | str | `"diagnostic"` | `"diagnostic"` (FFPE), `"frozen"` (tissue sections), or `"all"`. |
| `max_per_cancer_type` | int or None | `None` | Stratified cap. `None` = return all matching slides. |
| `seed` | int or None | `42` | Random seed for reproducible subsampling. |

### Download slides

```python
from wsistream.datasets import download_tcga_slides

paths = download_tcga_slides(
    manifest,
    output_dir="/data/tcga",       # saves as /data/tcga/TCGA-BRCA/file.svs
    organize_by="cancer_type",     # or "flat" for all files in one directory
    skip_existing=True,            # skip already-downloaded files
    max_workers=4,                 # parallel download threads
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `manifest` | list[dict] | required | File records returned by `query_tcga_slides`. |
| `output_dir` | str or Path | required | Root directory to save slides into. |
| `organize_by` | str | `"cancer_type"` | `"cancer_type"` creates subdirectories; `"flat"` puts everything in `output_dir/`. |
| `skip_existing` | bool | `True` | Skip files that already exist with matching size. |
| `max_workers` | int | `4` | Number of parallel download threads. |

Downloads run in parallel via a thread pool with a `tqdm` progress bar. For very large-scale downloads (thousands of slides), consider exporting a manifest and using the GDC Data Transfer Tool -- see below.

### Export a GDC manifest

For downloading many slides, the [GDC Data Transfer Tool](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool) (`gdc-client`) is faster than HTTPS because it supports parallel connections. Export a manifest and use `gdc-client`:

```python
from wsistream.datasets import save_manifest

save_manifest(manifest, "my_manifest.tsv")
```

Then from the command line:

```bash
gdc-client download -m my_manifest.tsv -d /data/tcga
```

### End-to-end example

Set up a fresh VM with 10 diagnostic slides per cancer type from BRCA and LUAD, then start training:

```python
from wsistream.datasets import query_tcga_slides, download_tcga_slides, TCGAAdapter
from wsistream.pipeline import PatchPipeline
from wsistream.backends import OpenSlideBackend
from wsistream.tissue import OtsuTissueDetector
from wsistream.sampling import RandomSampler

# Step 1: Download slides
manifest = query_tcga_slides(
    cancer_types=["TCGA-BRCA", "TCGA-LUAD"],
    slide_type="diagnostic",
    max_per_cancer_type=10,
)
download_tcga_slides(manifest, output_dir="/data/tcga")

# Step 2: Stream patches
pipeline = PatchPipeline(
    slide_paths="/data/tcga",
    backend=OpenSlideBackend(),
    tissue_detector=OtsuTissueDetector(),
    sampler=RandomSampler(patch_size=256, num_patches=-1, target_mpp=0.5),
    dataset_adapter=TCGAAdapter(),
    pool_size=4,
    patches_per_slide=100,
    cycle=True,
)

for result in pipeline:
    print(result.image.shape, result.slide_metadata.cancer_type)
```

## What gets logged

When a `DatasetAdapter` is configured, `pipeline.stats_dict()` includes dataset-specific counts:

```python
{
    "pipeline/cancer_type/TCGA-BRCA": 150,
    "pipeline/cancer_type/TCGA-LUAD": 120,
    "pipeline/sample_type/primary_solid_tumor": 250,
    "pipeline/sample_type/solid_tissue_normal": 20,
    ...
}
```

These are ready for logging to Weights & Biases or similar tools.

## Writing your own

```python
from pathlib import Path
from wsistream.datasets.base import DatasetAdapter
from wsistream.types import SlideMetadata

class CamelyonAdapter(DatasetAdapter):
    def parse_metadata(self, slide_path: str) -> SlideMetadata:
        filename = Path(slide_path).stem
        is_tumor = "tumor" in filename.lower()
        return SlideMetadata(
            slide_path=slide_path,
            dataset_name="Camelyon16",
            sample_type="tumor" if is_tumor else "normal",
        )
```
