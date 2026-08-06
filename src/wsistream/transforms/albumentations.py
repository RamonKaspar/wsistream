"""Wrapper for albumentations augmentation pipelines."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from wsistream.transforms.base import PatchTransform


@dataclass
class AlbumentationsWrapper(PatchTransform):
    """
    Wrap any albumentations.Compose as a PatchTransform.

    Albumentations initialises its RNG state at construction time.  Under
    fork-based DataLoader workers the constructed object is copied in memory,
    so all workers share the same initial RNG state and replay identical
    augmentation sequences.  (PyTorch does reseed the numpy global per worker,
    but albumentations is not fully governed by that global even after
    ``set_random_seed``.)  To prevent this, the wrapper owns an RNG that
    ``PatchPipeline`` reseeds per worker, and pushes a derived seed into
    albumentations on the first call after every reseed.

    Needs ``set_random_seed``, added in albumentations 1.4.21 (absent in
    1.4.18). With anything older the wrapper warns once and augmentations stay
    tied to the numpy global RNG.

    Parameters
    ----------
    transform : albumentations.Compose or albumentations.BasicTransform
        The albumentations pipeline to apply. ``None`` makes this a no-op.
    seed : int or None
        Optional seed for the internal RNG. Default: ``None`` (random).

    .. note::
        Any ``seed`` passed here is overridden by the pipeline's own seeding.
        Set ``seed`` on ``PatchPipeline`` instead.

    Example
    -------
    >>> import albumentations as A
    >>> aug = AlbumentationsWrapper(A.Compose([
    ...     A.ColorJitter(brightness=0.2, contrast=0.2),
    ...     A.GaussianBlur(blur_limit=3, p=0.3),
    ... ]))
    """

    transform: Any = None
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        # The generator the wrapped transform was last seeded from. The
        # pipeline reseeds by *replacing* ``_rng``, so an identity mismatch is
        # exactly the signal that a fresh seed has to be pushed down.
        self._seeded_from: np.random.Generator | None = None

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if self.transform is None:
            return image
        if self._rng is not self._seeded_from:
            self._sync_seed()
        return self.transform(image=image)["image"]

    def _sync_seed(self) -> None:
        """Push a seed derived from the wrapper's RNG into albumentations."""
        self._seeded_from = self._rng
        set_random_seed = getattr(self.transform, "set_random_seed", None)
        if set_random_seed is None:
            warnings.warn(
                "The wrapped transform has no set_random_seed(); albumentations "
                "augmentations fall back to the numpy global RNG and will repeat "
                "identically across forked DataLoader workers. Upgrade to "
                "albumentations >= 1.4.21.",
                UserWarning,
                stacklevel=3,
            )
            return
        set_random_seed(int(self._rng.integers(0, 2**63 - 1)))

    def __repr__(self) -> str:
        return f"AlbumentationsWrapper({self.transform!r})"
