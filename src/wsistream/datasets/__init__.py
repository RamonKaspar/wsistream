"""Dataset adapters and download helpers."""

from wsistream.datasets.base import DatasetAdapter
from wsistream.datasets.tcga import (
    TCGAAdapter,
    download_tcga_slides,
    query_tcga_slides,
    save_manifest,
)

__all__ = [
    "DatasetAdapter",
    "TCGAAdapter",
    "query_tcga_slides",
    "download_tcga_slides",
    "save_manifest",
]
