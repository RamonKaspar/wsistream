"""Color augmentation in the Hematoxylin-Eosin-DAB color space."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from wsistream.transforms.base import PatchTransform


@dataclass
class HEDColorAugmentation(PatchTransform):
    """
    Random perturbation of HED stain channels, following Tellez et al.

    Decomposes the image into Hematoxylin, Eosin and DAB channels and perturbs
    each channel ``i`` independently in optical-density space as
    ``s_i' = alpha_i * s_i + beta_i``, with both parameters drawn per channel
    per image from uniform distributions:

    - ``alpha_i ~ U(1 - sigma, 1 + sigma)`` scales stain concentration
    - ``beta_i ~ U(-sigma_bias, sigma_bias)`` shifts the stain baseline

    ``sigma`` is in the same units the papers quote, so ``sigma=0.05``
    reproduces Tellez "HED-light" and ``sigma=0.2`` "HED-strong".

    Parameters
    ----------
    sigma : float
        Half-width of the multiplicative range. Default: 0.05 (Tellez "light").
    seed : int or None
        Optional seed. Overridden by ``PatchPipeline`` seeding.
    sigma_bias : float or None
        Half-width of the additive range. ``None`` (default) reuses ``sigma``,
        matching the StainTools and HistomicsTK convention. Pass ``0.0`` to
        disable the additive term.

    Notes
    -----
    Tellez et al. (2019) report intensity ranges of ``[-0.05, 0.05]``
    ("HED-light") and ``[-0.2, 0.2]`` ("HED-strong") but do not say whether the
    range covers both terms.  StainTools (``sigma1``, ``sigma2``) and
    HistomicsTK both default to one value for both, which is the convention
    followed here.

    Those published values are in optical-density units, and ``sigma`` is
    applied in the same space, so it needs no rescaling.

    See Also
    --------
    ``albumentations.HEStain`` : stain augmentation wrappable via
        :class:`~wsistream.transforms.AlbumentationsWrapper`.  This class always
        uses skimage's fixed HED matrix; ``HEStain`` can estimate the matrix
        from the image with ``method="macenko"`` or ``"vahadane"``, though its
        default ``method="random_preset"`` picks from predefined matrices
        instead.

    References
    ----------
    Tellez et al., "Whole-Slide Mitosis Detection in H&E Breast Histology Using
    PHH3 as a Reference to Train Distilled Stain-Invariant Convolutional
    Networks", IEEE TMI, 2018. https://arxiv.org/abs/1808.05896
    (defines the alpha/beta perturbation used here)

    Tellez et al., "Quantifying the effects of data augmentation and stain
    color normalization in convolutional neural networks for computational
    pathology", Medical Image Analysis, 2019.
    https://doi.org/10.1016/j.media.2019.101544
    (reports the light/strong intensity ranges)

    Karasikov et al., "Training state-of-the-art pathology foundation models
    with orders of magnitude less data", 2025. https://arxiv.org/abs/2504.05186
    """

    sigma: float = 0.05
    seed: int | None = None
    sigma_bias: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.sigma) or self.sigma < 0:
            raise ValueError(f"sigma must be a finite number >= 0, got {self.sigma}")
        if self.sigma_bias is not None:
            if not math.isfinite(self.sigma_bias) or self.sigma_bias < 0:
                raise ValueError(f"sigma_bias must be a finite number >= 0, got {self.sigma_bias}")
        self._rng = np.random.default_rng(self.seed)

    @property
    def effective_sigma_bias(self) -> float:
        """The additive half-width actually used, in optical-density units."""
        return self.sigma if self.sigma_bias is None else self.sigma_bias

    def __call__(self, image: np.ndarray) -> np.ndarray:
        from skimage.color import hed_from_rgb, rgb_from_hed

        sigma_bias = self.effective_sigma_bias
        conv = np.asarray(hed_from_rgb)
        inv = np.asarray(rgb_from_hed)

        rgb = np.clip(image.astype(np.float64) / 255.0, 1e-6, 1.0)
        concentrations = -np.log(rgb) @ conv

        for ch in range(3):
            alpha = self._rng.uniform(1.0 - self.sigma, 1.0 + self.sigma)
            beta = self._rng.uniform(-sigma_bias, sigma_bias)
            concentrations[:, :, ch] = concentrations[:, :, ch] * alpha + beta

        rgb_out = np.exp(-(concentrations @ inv))
        return np.clip(rgb_out * 255, 0, 255).astype(np.uint8)
