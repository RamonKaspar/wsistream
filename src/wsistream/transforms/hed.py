"""Color augmentation in the Hematoxylin-Eosin-DAB color space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wsistream.transforms.base import PatchTransform


@dataclass
class HEDColorAugmentation(PatchTransform):
    """
    Random perturbation of HED stain channels.

    Decomposes the image into Hematoxylin, Eosin, and DAB channels,
    applies random multiplicative noise, and converts back to RGB.

    References
    ----------
    Karasikov et al., "Training state-of-the-art pathology foundation models
    with orders of magnitude less data", 2025.
    """

    sigma: float = 0.05
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        from skimage.color import combine_stains, hed_from_rgb, rgb_from_hed, separate_stains

        img_float = np.clip(image.astype(np.float64) / 255.0, 1e-6, 1.0)

        hed = separate_stains(img_float, hed_from_rgb)
        for ch in range(3):
            hed[:, :, ch] *= 1.0 + self._rng.normal(0, self.sigma)

        rgb = combine_stains(hed, rgb_from_hed)
        return np.clip(rgb * 255, 0, 255).astype(np.uint8)
