"""Color augmentation in the Hematoxylin-Eosin-DAB color space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wsistream.transforms.base import PatchTransform


@dataclass
class HEDColorAugmentation(PatchTransform):
    """
    Random perturbation of HED stain channels, following Tellez et al.

    Decomposes the image into Hematoxylin, Eosin, and DAB channels, perturbs
    each channel ``i`` independently as ``s_i' = alpha_i * s_i + beta_i``, and
    converts back to RGB.  Both parameters are drawn per channel per image:

    - ``alpha_i ~ U(1 - sigma, 1 + sigma)`` scales stain concentration
    - ``beta_i ~ U(-sigma_bias, sigma_bias)`` shifts the stain baseline

    Tellez et al. (2019) report intensity ratios of ``[-0.05, 0.05]`` for
    "HED-light" and ``[-0.2, 0.2]`` for "HED-strong", which map onto
    ``sigma=0.05`` and ``sigma=0.2`` here.

    Parameters
    ----------
    sigma : float
        Half-width of the multiplicative range. Default: 0.05 (Tellez "light").
    sigma_bias : float
        Half-width of the additive range. Default: ``0.0`` (disabled). See the
        scale warning below before raising this.
    seed : int or None
        Optional seed. Overridden by ``PatchPipeline`` seeding.

    Warnings
    --------
    ``sigma`` and ``sigma_bias`` are **not** on the same scale, and the paper's
    sigma does not transfer to ``sigma_bias``.

    ``alpha`` is a ratio, so it is invariant to the stain-space convention and
    the published values port directly.  ``beta`` is an absolute offset in
    stain space, so its meaning depends entirely on that convention.  The
    reference implementations that use the published sigma for beta (HistomicsTK,
    StainTools) work in SDA space scaled to roughly ``[0, 255]``.  This class
    uses ``skimage.color.separate_stains``, whose output for typical H&E tissue
    has channel means around ``0.02`` and maxima around ``0.25``.  A beta of
    +/-0.05 there is several times the channel mean: it swamps the stain signal
    and produces non-physical yellow, cyan and purple tissue rather than a
    plausible stain shift.

    ``sigma_bias`` therefore defaults to ``0.0``.  If you enable it, scale it to
    your data's stain-channel magnitude (order ``1e-3`` for skimage HED output)
    and inspect the result visually.  For reference, OpenMidnight applies
    ``s_i + U(-0.05, 0.05)`` to ``skimage.rgb2hed`` output with no multiplicative
    term, which lands in this aggressive regime by design
    (https://github.com/MedARC-AI/OpenMidnight,
    ``dinov2/data/augmentations.py``, class ``hed_mod``).

    See Also
    --------
    ``albumentations.HEStain`` : Macenko/Vahadane stain augmentation, wrappable
        via :class:`~wsistream.transforms.AlbumentationsWrapper`.  It estimates
        the stain matrix per image instead of using the fixed HED matrix used
        here, and applies both a multiplicative and an additive term
        (``intensity_scale_range``, ``intensity_shift_range``) on its own
        concentration scale.  Prefer it when you want stain augmentation matched
        to each slide's actual stain vectors.

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
    sigma_bias: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {self.sigma}")
        if self.sigma_bias < 0:
            raise ValueError(f"sigma_bias must be >= 0, got {self.sigma_bias}")
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        from skimage.color import combine_stains, hed_from_rgb, rgb_from_hed, separate_stains

        img_float = np.clip(image.astype(np.float64) / 255.0, 1e-6, 1.0)

        hed = separate_stains(img_float, hed_from_rgb)
        for ch in range(3):
            alpha = self._rng.uniform(1.0 - self.sigma, 1.0 + self.sigma)
            hed[:, :, ch] = hed[:, :, ch] * alpha
            if self.sigma_bias > 0:
                hed[:, :, ch] += self._rng.uniform(-self.sigma_bias, self.sigma_bias)

        rgb = combine_stains(hed, rgb_from_hed)
        return np.clip(rgb * 255, 0, 255).astype(np.uint8)
