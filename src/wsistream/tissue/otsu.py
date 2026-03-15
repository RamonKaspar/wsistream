"""
Otsu thresholding on grayscale intensity.

Grayscale conversion, Gaussian blur, Otsu binarization (inverted, since
tissue is darker than background), morphological closing, and small-
component removal.

Converts the thumbnail to grayscale, applies Gaussian blur, and uses
Otsu's method to separate tissue (dark) from background (white); inverted, since
tissue is darker than background. Optionally applies morphological closing and 
removes small connected components. 

Not the same as CLAM's tissue detection, which thresholds HSV saturation.
For CLAM-matching behavior, use ``CLAMTissueDetector``.

References
----------
Otsu N., "A Threshold Selection Method from Gray-Level Histograms",
IEEE Trans. Systems, Man, and Cybernetics, 1979.
DOI: 10.1109/TSMC.1979.4310076
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from wsistream.tissue.base import TissueDetector


@dataclass
class OtsuTissueDetector(TissueDetector):
    """
    Grayscale Otsu thresholding with morphological cleanup.

    Parameters
    ----------
    blur_ksize : int
        Gaussian blur kernel size before thresholding. Default: 7.
    morph_close_ksize : int
        Elliptical kernel for morphological closing. 0 disables. Default: 5.
    min_area_ratio : float
        Connected components smaller than this fraction of the total
        thumbnail area are removed. 0 disables. Default: 0.001.
    """

    blur_ksize: int = 7
    morph_close_ksize: int = 5
    min_area_ratio: float = 0.001

    def detect(
        self, thumbnail: np.ndarray, downsample: tuple[float, float] = (1.0, 1.0)
    ) -> np.ndarray:
        gray = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)

        # BINARY_INV because tissue is darker than background
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = mask.astype(bool)

        if self.morph_close_ksize > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.morph_close_ksize, self.morph_close_ksize)
            )
            mask_uint8 = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
            mask = mask_uint8 > 0

        if self.min_area_ratio > 0:
            min_area = int(mask.size * self.min_area_ratio)
            mask_uint8 = mask.astype(np.uint8)
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8)
            for i in range(1, n_labels):
                if stats[i, cv2.CC_STAT_AREA] < min_area:
                    mask[labels == i] = False

        return mask