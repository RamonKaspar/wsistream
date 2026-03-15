"""Patch sampling strategies."""

from wsistream.sampling.base import PatchSampler
from wsistream.sampling.grid import GridSampler
from wsistream.sampling.multi_magnification import MultiMagnificationSampler
from wsistream.sampling.random import RandomSampler

__all__ = ["PatchSampler", "RandomSampler", "GridSampler", "MultiMagnificationSampler"]
