"""Dataset adapters for extracting slide-level metadata."""

from wsistream.datasets.base import DatasetAdapter
from wsistream.datasets.tcga import TCGAAdapter

__all__ = ["DatasetAdapter", "TCGAAdapter"]
