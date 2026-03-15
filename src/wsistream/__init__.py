"""
wsistream; Modular online patch streaming from whole-slide images.

Inspired by the online patching approach from Kaiko/Midnight.
"""

from wsistream.filters import HSVPatchFilter, PatchFilter
from wsistream.pipeline import PatchPipeline
from wsistream.slide import SlideHandle
from wsistream.types import PatchCoordinate, PatchResult, SlideMetadata, TissueMask

__version__ = "0.1.0"
__all__ = [
    "PatchPipeline", "SlideHandle",
    "PatchCoordinate", "PatchResult", "SlideMetadata", "TissueMask",
    "PatchFilter", "HSVPatchFilter",
]
