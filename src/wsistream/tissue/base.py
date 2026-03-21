"""Abstract base class for tissue detection strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class TissueDetector(ABC):
    """
    Detect tissue regions in a low-resolution thumbnail.

    Subclass this and implement ``detect`` to add new detection methods.
    """

    @abstractmethod
    def detect(
        self, thumbnail: np.ndarray, downsample: tuple[float, float] = (1.0, 1.0)
    ) -> np.ndarray:
        """
        Produce a binary tissue mask from a thumbnail.

        Parameters
        ----------
        thumbnail : np.ndarray
            RGB image, shape (H, W, 3), dtype uint8.
        downsample : tuple[float, float]
            (scale_x, scale_y) of the thumbnail relative to level 0.
            Required by CLAMTissueDetector for area threshold scaling.
            Provided automatically by the pipeline.

        Returns
        -------
        np.ndarray
            Boolean mask, shape (H, W), True = tissue.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
