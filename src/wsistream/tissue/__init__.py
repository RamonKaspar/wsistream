"""Tissue detection strategies."""

from wsistream.tissue.base import TissueDetector
from wsistream.tissue.clam import CLAMTissueDetector
from wsistream.tissue.combined import CombinedTissueDetector
from wsistream.tissue.hsv import HSVTissueDetector
from wsistream.tissue.otsu import OtsuTissueDetector

__all__ = [
    "TissueDetector",
    "OtsuTissueDetector",
    "HSVTissueDetector",
    "CombinedTissueDetector",
    "CLAMTissueDetector",
]
