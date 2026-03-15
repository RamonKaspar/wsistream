"""Patch transforms and augmentations."""

from wsistream.transforms.albumentations import AlbumentationsWrapper
from wsistream.transforms.base import ComposeTransforms, PatchTransform
from wsistream.transforms.geometric import RandomFlipRotate
from wsistream.transforms.hed import HEDColorAugmentation
from wsistream.transforms.normalize import NormalizeTransform
from wsistream.transforms.resize import ResizeTransform

__all__ = [
    "PatchTransform",
    "ComposeTransforms",
    "HEDColorAugmentation",
    "RandomFlipRotate",
    "ResizeTransform",
    "NormalizeTransform",
    "AlbumentationsWrapper",
]
