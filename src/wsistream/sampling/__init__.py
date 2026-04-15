"""Patch sampling strategies."""

from wsistream.sampling.base import (
    CoordinatePool,
    MultiLevelCoordinatePool,
    PatchSampler,
    enumerate_grid_coordinates,
)
from wsistream.sampling.continuous_magnification import ContinuousMagnificationSampler
from wsistream.sampling.grid import GridSampler
from wsistream.sampling.multi_magnification import MultiMagnificationSampler
from wsistream.sampling.random import RandomSampler

__all__ = [
    "PatchSampler",
    "RandomSampler",
    "GridSampler",
    "MultiMagnificationSampler",
    "ContinuousMagnificationSampler",
    "CoordinatePool",
    "MultiLevelCoordinatePool",
    "enumerate_grid_coordinates",
]
