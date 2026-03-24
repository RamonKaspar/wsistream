"""
wsistream; Modular online patch streaming from whole-slide images.

Inspired by the online patching approach from Kaiko/Midnight.
"""

from wsistream.backends import OpenSlideBackend, SlideBackend, TiffSlideBackend
from wsistream.datasets import DatasetAdapter, TCGAAdapter
from wsistream.filters import HSVPatchFilter, PatchFilter
from wsistream.pipeline import PatchPipeline
from wsistream.sampling import GridSampler, MultiMagnificationSampler, PatchSampler, RandomSampler
from wsistream.slide import SlideHandle
from wsistream.tissue import (
    CLAMTissueDetector,
    CombinedTissueDetector,
    HSVTissueDetector,
    OtsuTissueDetector,
    TissueDetector,
)
from wsistream.transforms import (
    AlbumentationsWrapper,
    ComposeTransforms,
    HEDColorAugmentation,
    NormalizeTransform,
    PatchTransform,
    RandomFlipRotate,
    ResizeTransform,
)
from wsistream.types import PatchCoordinate, PatchResult, SlideMetadata, TissueMask

__version__ = "0.1.2"
__all__ = [
    # Core
    "PatchPipeline",
    "SlideHandle",
    # Types
    "PatchCoordinate",
    "PatchResult",
    "SlideMetadata",
    "TissueMask",
    # Backends
    "SlideBackend",
    "OpenSlideBackend",
    "TiffSlideBackend",
    # Tissue detection
    "TissueDetector",
    "OtsuTissueDetector",
    "HSVTissueDetector",
    "CLAMTissueDetector",
    "CombinedTissueDetector",
    # Sampling
    "PatchSampler",
    "RandomSampler",
    "GridSampler",
    "MultiMagnificationSampler",
    # Filters
    "PatchFilter",
    "HSVPatchFilter",
    # Transforms
    "PatchTransform",
    "ComposeTransforms",
    "HEDColorAugmentation",
    "RandomFlipRotate",
    "ResizeTransform",
    "NormalizeTransform",
    "AlbumentationsWrapper",
    # Datasets
    "DatasetAdapter",
    "TCGAAdapter",
]
