"""Multi-view patch generation primitives."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import cv2
import numpy as np

from wsistream.transforms.base import PatchTransform


@dataclass(frozen=True)
class CropParams:
    """Spatial crop parameters in image coordinates."""

    x: int
    y: int
    width: int
    height: int


@dataclass
class RandomResizedCrop:
    """Random crop followed by resize, operating on numpy RGB arrays.

    Mirrors torchvision's ``RandomResizedCrop`` at the patch level.  The
    source window area is sampled uniformly from ``scale``, its aspect ratio
    is sampled log-uniformly from ``ratio``, and the result is resized to
    ``size × size``.

    ``scale`` has **no default** because no single value is appropriate for
    all use cases.  Common choices:

    - DINO v1 global crops: ``(0.4, 1.0)``
      (``facebookresearch/dino/main_dino.py``, ``global_crops_scale``)
    - DINO v1 local crops: ``(0.05, 0.4)``
      (``facebookresearch/dino/main_dino.py``, ``local_crops_scale``)
    - DINOv2 global crops: ``(0.32, 1.0)``
      (``facebookresearch/dinov2/dinov2/configs/ssl_default_config.yaml``)
    - DINOv2 local crops: ``(0.05, 0.32)``
      (``facebookresearch/dinov2/dinov2/configs/ssl_default_config.yaml``)
      with architecture-specific ``local_crops_size=98`` in the ViT-L/14
      and ViT-g/14 training configs.
    - SimCLR / torchvision default: ``(0.08, 1.0)`` (very aggressive for pathology)

    Parameters
    ----------
    size : int
        Output side length in pixels after resizing.  The output is always
        ``size × size`` regardless of the sampled crop shape.
    scale : tuple[float, float]
        ``(min, max)`` range for the source-window area as a fraction of the
        extracted patch area.  Must satisfy ``0 < min <= max <= 1``.
        There is no default — always pass this explicitly.
    ratio : tuple[float, float]
        ``(min, max)`` aspect-ratio range for the sampled source window.
        Log-uniformly sampled, matching torchvision's ``RandomResizedCrop``.
        Default: ``(3/4, 4/3)``.
    interpolation : int
        OpenCV interpolation flag used when resizing the cropped window to
        ``size × size``.  Default: ``cv2.INTER_LINEAR`` (bilinear).
        Use ``cv2.INTER_AREA`` when the crop is larger than ``size × size``
        (downscaling); use ``cv2.INTER_CUBIC`` or ``cv2.INTER_LANCZOS4``
        for higher-quality upscaling.
    seed : int or None
        Optional seed for the internal RNG.  Default: ``None`` (random).

    .. note::
        Any ``seed`` passed here is overridden by the pipeline's own
        seeding.  Set ``seed`` on ``PatchPipeline`` instead.
    """

    size: int
    scale: tuple[float, float]
    ratio: tuple[float, float] = (3 / 4, 4 / 3)
    interpolation: int = cv2.INTER_LINEAR
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError(f"size must be >= 1, got {self.size}")
        if (
            self.scale[0] <= 0
            or self.scale[1] <= 0
            or self.scale[0] > self.scale[1]
            or self.scale[1] > 1
        ):
            raise ValueError(f"scale must be (a, b) with 0 < a <= b <= 1, got {self.scale}")
        if self.ratio[0] <= 0 or self.ratio[1] <= 0 or self.ratio[0] > self.ratio[1]:
            raise ValueError(f"ratio must be (a, b) with 0 < a <= b, got {self.ratio}")
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        params = self.sample_params(image)
        return self.apply_params(image, params)

    def sample_params(self, image: np.ndarray) -> CropParams:
        """Sample crop parameters for an extracted patch.

        Attempts up to 10 times to find a valid (w, h) pair that fits inside
        the image.  Falls back to a central crop with the closest valid aspect
        ratio (matching torchvision's fallback).
        """
        h, w = image.shape[:2]
        if h < 1 or w < 1:
            raise ValueError(f"image must have positive spatial dimensions, got {image.shape}")
        area = h * w
        log_ratio = np.log(self.ratio)

        for _ in range(10):
            target_area = float(self._rng.uniform(self.scale[0], self.scale[1]) * area)
            aspect = float(np.exp(self._rng.uniform(log_ratio[0], log_ratio[1])))

            crop_w = int(round(np.sqrt(target_area * aspect)))
            crop_h = int(round(np.sqrt(target_area / aspect)))

            if 0 < crop_w <= w and 0 < crop_h <= h:
                x = int(self._rng.integers(0, w - crop_w + 1))
                y = int(self._rng.integers(0, h - crop_h + 1))
                return CropParams(x=x, y=y, width=crop_w, height=crop_h)

        # Fallback from torchvision: central crop with closest valid aspect ratio.
        in_ratio = w / h
        if in_ratio < self.ratio[0]:
            crop_w = w
            crop_h = int(round(crop_w / self.ratio[0]))
        elif in_ratio > self.ratio[1]:
            crop_h = h
            crop_w = int(round(crop_h * self.ratio[1]))
        else:
            crop_w = w
            crop_h = h

        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)
        x = (w - crop_w) // 2
        y = (h - crop_h) // 2
        return CropParams(x=x, y=y, width=crop_w, height=crop_h)

    def apply_params(self, image: np.ndarray, params: CropParams) -> np.ndarray:
        """Apply pre-sampled crop parameters and resize to ``size × size``."""
        crop = image[params.y : params.y + params.height, params.x : params.x + params.width]
        return cv2.resize(crop, (self.size, self.size), interpolation=self.interpolation)

    def __repr__(self) -> str:
        return f"RandomResizedCrop(size={self.size}, scale={self.scale}, " f"ratio={self.ratio})"


@dataclass
class ViewConfig:
    """Configuration for one named output view.

    Parameters
    ----------
    name : str
        Output key in ``PatchResult.views`` and PyTorch batch dicts.
        Must not collide with reserved batch keys (``"image"``, ``"x"``,
        ``"y"``, ``"level"``, ``"patch_size"``, ``"slide_path"``,
        ``"mpp"``, ``"tissue_fraction"``, or any ``SlideMetadata`` field).
        Collisions are detected at ``PatchPipeline`` construction time.
    transforms : PatchTransform or None
        Per-view augmentation chain applied after optional cropping.
        Default: ``None`` (no per-view transforms — the view is a copy of
        the shared-transformed patch, or the raw patch if no
        ``shared_transforms`` are configured).
    crop : RandomResizedCrop or None
        Optional spatial crop applied before ``transforms``.
        Default: ``None`` (no crop — full extracted patch is used).
    count : int
        Number of independent views generated from this config.
        ``count=1`` (default) keeps the name as-is.
        ``count > 1`` produces ``name_0``, ``name_1``, … — each draw
        is independent (separate crop and transform calls).
    mpp_override : float or None
        Target MPP for a same-center slide re-read at a different pyramid
        level.  Requires slide MPP metadata.  ``shared_transforms`` cannot
        be combined with ``mpp_override`` views.
        Default: ``None`` (use the primary extracted patch).
    patch_size_override : int or None
        Number of pixels to request from the slide when ``mpp_override``
        triggers a re-read.  Requires ``mpp_override`` to be set.
        Default: ``None`` (reuse the primary sampler's ``patch_size``).

    .. note::
        Any ``seed`` set on ``transforms`` or ``crop`` is overridden by
        the pipeline's seeding at construction time.  Set ``seed`` on
        ``PatchPipeline`` to control reproducibility.
    """

    name: str
    transforms: PatchTransform | None = None
    crop: RandomResizedCrop | None = None
    count: int = 1
    mpp_override: float | None = None
    patch_size_override: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ViewConfig.name must be non-empty")
        if self.count < 1:
            raise ValueError(f"count must be >= 1, got {self.count}")
        if self.mpp_override is not None and self.mpp_override <= 0:
            raise ValueError(f"mpp_override must be > 0, got {self.mpp_override}")
        if self.patch_size_override is not None and self.patch_size_override < 1:
            raise ValueError(f"patch_size_override must be >= 1, got {self.patch_size_override}")
        if self.patch_size_override is not None and self.mpp_override is None:
            raise ValueError("patch_size_override requires mpp_override to be set")
        if self.count > 1 and self.crop is None and self.transforms is None:
            warnings.warn(
                f"ViewConfig(name={self.name!r}, count={self.count}) has no crop or transforms"
                f" — all {self.count} views will be identical copies.",
                UserWarning,
                stacklevel=2,
            )


def expand_view_names(views: list[ViewConfig]) -> list[str]:
    """Return the concrete output names produced by a view list after expanding ``count``.

    For a ``ViewConfig`` with ``count=1``, the name is returned as-is.
    For ``count > 1``, names ``name_0``, ``name_1``, … are returned.

    Example
    -------
    >>> expand_view_names([ViewConfig("global"), ViewConfig("local", count=3)])
    ['global', 'local_0', 'local_1', 'local_2']
    """
    names: list[str] = []
    for view in views:
        if view.count == 1:
            names.append(view.name)
        else:
            names.extend(f"{view.name}_{i}" for i in range(view.count))
    return names
