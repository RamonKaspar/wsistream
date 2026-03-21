"""Abstract base class for patch sampling strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from wsistream.slide import SlideHandle
from wsistream.types import PatchCoordinate, TissueMask


class PatchSampler(ABC):
    """
    Sample patch coordinates from a slide.
    """

    @abstractmethod
    def sample(self, slide: SlideHandle, tissue_mask: TissueMask) -> Iterator[PatchCoordinate]:
        """
        Yield patch coordinates from a slide.

        Parameters
        ----------
        slide : SlideHandle
            The opened whole-slide image.
        tissue_mask : TissueMask
            Binary mask indicating tissue regions.

        Yields
        ------
        PatchCoordinate
        """
        ...
