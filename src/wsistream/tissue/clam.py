"""
CLAM tissue segmentation.

Reimplements the segmentation algorithm from mahmoodlab/CLAM, producing a
boolean mask instead of CLAM's polygon lists.

Source: github.com/mahmoodlab/CLAM

Algorithm: HSV saturation extraction, median blur,
binary threshold, morphological close (square kernel), contour extraction
with RETR_CCOMP, then area-based filtering with hole subtraction and
sorting.

Output format: the only difference from CLAM. CLAM stores filtered
contour/hole polygon lists (self.contours_tissue, self.holes_tissue)
for downstream isInContour coordinate checks. We rasterize those same
filtered contours onto a boolean mask. Given identical inputs, both
representations define the same tissue regions.

References
----------
Lu et al., "Data Efficient and Weakly Supervised Computational Pathology 
on Whole Slide Images", Nature BME, 2021. DOI: 10.1038/s41551-020-00682-w
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from wsistream.tissue.base import TissueDetector


@dataclass
class CLAMTissueDetector(TissueDetector):
    """
    CLAM tissue segmentation with contour-based area filtering.

    Defaults match create_patches_fp.py, not the segmentTissue() function
    signature (which has sthresh=20, close=0). The CLAM README documents
    max_n_holes=10 but create_patches_fp.py passes 8.

    Parameters
    ----------
    sthresh : int
        Saturation threshold. Default: 8.
    sthresh_up : int
        Upper bound for cv2.threshold. Default: 255.
    use_otsu : bool
        Otsu on saturation instead of fixed threshold. Default: False.
    mthresh : int
        Median blur kernel applied to saturation before thresholding. Default: 7.
    close : int
        Square kernel for morphological closing. 0 disables. Default: 4.
    a_t : int
        Min net foreground area threshold, scaled internally by
        ref_patch_size^2 / (downsample_x * downsample_y). Default: 100.
    a_h : int
        Min hole area threshold, same scaling. Default: 16.
    max_n_holes : int
        Keep N largest holes per contour. Default: 8.
    ref_patch_size : int
        Reference patch size for area scaling. Default: 512.
    """

    sthresh: int = 8
    sthresh_up: int = 255
    use_otsu: bool = False
    mthresh: int = 7
    close: int = 4
    a_t: int = 100
    a_h: int = 16
    max_n_holes: int = 8
    ref_patch_size: int = 512

    def detect(
        self, thumbnail: np.ndarray, downsample: tuple[float, float] = (1.0, 1.0)
    ) -> np.ndarray:
        img_hsv = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)
        img_med = cv2.medianBlur(img_hsv[:, :, 1], self.mthresh)

        if self.use_otsu:
            _, img_otsu = cv2.threshold(
                img_med, 0, self.sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY
            )
        else:
            _, img_otsu = cv2.threshold(
                img_med, self.sthresh, self.sthresh_up, cv2.THRESH_BINARY
            )

        if self.close > 0:
            kernel = np.ones((self.close, self.close), np.uint8)
            img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)

        # Scale area thresholds by reference patch area at this downsample
        scaled_ref_patch_area = int(
            self.ref_patch_size ** 2 / (downsample[0] * downsample[1])
        )
        scaled_a_t = self.a_t * scaled_ref_patch_area
        scaled_a_h = self.a_h * scaled_ref_patch_area

        contours, hierarchy = cv2.findContours(
            img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )

        if not contours or hierarchy is None:
            return img_otsu > 0

        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]

        # _filter_contours logic

        hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
        filtered = []
        all_holes = []

        for cont_idx in hierarchy_1:
            cont = contours[cont_idx]
            holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)

            a = cv2.contourArea(cont)
            hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
            a = a - np.array(hole_areas).sum()

            if a == 0:
                continue
            if a > scaled_a_t:
                filtered.append(cont_idx)
                all_holes.append(holes)

        foreground_contours = [contours[cont_idx] for cont_idx in filtered]

        hole_contours = []
        for hole_ids in all_holes:
            unfiltered_holes = [contours[idx] for idx in hole_ids]
            unfiltered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
            unfiltered_holes = unfiltered_holes[:self.max_n_holes]

            filtered_holes = []
            for hole in unfiltered_holes:
                if cv2.contourArea(hole) > scaled_a_h:
                    filtered_holes.append(hole)

            hole_contours.append(filtered_holes)

        # Rasterize filtered contours to mask
        filtered_mask = np.zeros_like(img_otsu)
        for fg_contour, fg_holes in zip(foreground_contours, hole_contours):
            cv2.drawContours(filtered_mask, [fg_contour], 0, 255, cv2.FILLED)
            for hole in fg_holes:
                cv2.drawContours(filtered_mask, [hole], 0, 0, cv2.FILLED)

        return filtered_mask > 0