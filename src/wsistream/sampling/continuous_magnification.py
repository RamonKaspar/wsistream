"""Continuous magnification sampling via crop-and-resize."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from wsistream.sampling.base import PatchSampler
from wsistream.sampling.random import RandomSampler
from wsistream.slide import SlideHandle
from wsistream.types import PatchCoordinate, SlideProperties, TissueMask

logger = logging.getLogger(__name__)

_GRID_SIZE = 1000


def _info_kernel_matrix(grid: np.ndarray) -> np.ndarray:
    """K_info(x, y) = (min(x, y) / max(x, y))^2  (vectorised)."""
    X, Y = np.meshgrid(grid, grid)
    return (np.minimum(X, Y) / np.maximum(X, Y)) ** 2


def _transfer_potential(grid: np.ndarray, a: float, b: float) -> np.ndarray:
    r"""Closed-form transfer potential for the information-based kernel.

    K̄(x) = \int_a^b K_info(x, y) dy = 4x/3 - a^3/(3x^2) - x^2/b
    """
    return 4 * grid / 3 - a**3 / (3 * grid**2) - grid**2 / b


def _compute_maxavg_weights(grid: np.ndarray, a: float, b: float, lam: float) -> np.ndarray:
    """p*(x) = exp(K̄(x) / λ) / Z   (Eq. 10, Moellers et al. 2026)."""
    k_bar = _transfer_potential(grid, a, b)
    log_p = k_bar / lam
    log_p -= log_p.max()  # numerical stability
    p = np.exp(log_p)
    p /= p.sum()
    return p


def _compute_minmax_weights(grid: np.ndarray) -> np.ndarray:
    """Solve the max-min LP  (Eq. 11, Moellers et al. 2026).

    max_{p, t}  t   s.t.  Kp >= t·1,  p >= 0,  1^T p = 1

    Requires scipy (lazy-imported).
    """
    try:
        from scipy.optimize import linprog
    except ImportError:
        raise ImportError(
            "scipy is required for distribution='minmax'. " "Install it with: pip install scipy"
        ) from None

    n = len(grid)
    K = _info_kernel_matrix(grid)

    # Variables: x = [p_1, ..., p_n, t]
    c = np.zeros(n + 1)
    c[-1] = -1.0  # minimise -t  ≡  maximise t

    # Inequality: -Kp + t <= 0
    A_ub = np.column_stack([-K, np.ones(n)])
    b_ub = np.zeros(n)

    # Equality: sum(p) = 1
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, None)] * n + [(None, None)]

    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"MinMax LP did not converge: {result.message}")

    p = result.x[:n]
    p = np.maximum(p, 0.0)  # clean numerical noise
    p /= p.sum()
    return p


def _best_level_for_downsample(props: SlideProperties, target_mpp: float) -> int:
    """Pick the coarsest level whose mpp <= *target_mpp*.

    Ensures we always downsample (crop > output), preserving quality.
    Falls back to level 0 when *target_mpp* is finer than the native
    resolution (unavoidable mild upsample).
    """
    if props.mpp is None:
        return 0
    best = 0
    for lvl in range(props.level_count):
        lvl_mpp = props.mpp * props.level_downsamples[lvl]
        if lvl_mpp <= target_mpp:
            best = lvl  # levels get coarser, keep updating
    return best


@dataclass
class ContinuousMagnificationSampler(PatchSampler):
    """
    Sample patches at continuously varying magnification via crop-and-resize.

    Instead of restricting training to fixed scanner resolutions, this
    sampler synthesises patches at arbitrary magnifications by reading a
    larger crop from the WSI and resizing it to ``output_size``.  Three
    sampling distributions are supported:

    * ``"uniform"`` -- Uniform over *mpp_range*.
    * ``"maxavg"`` -- Maximises average representation quality across
      magnifications (entropy-regularised; Eq. 5-6 / 10 of paper).
      Concentrates on central magnifications.
    * ``"minmax"`` -- Maximises worst-case representation quality (LP;
      Eq. 7 / 11 of paper).  Oversamples boundary magnifications.
      Requires *scipy*.

    The ``output_size`` attribute is read by
    :class:`~wsistream.pipeline.PatchPipeline`, which automatically
    resizes patches after reading -- no ``ResizeTransform`` needed.

    Parameters
    ----------
    output_size : int
        Final patch size after resize (e.g. 224 for ViT).
    mpp_range : tuple[float, float]
        Continuous um/px range to sample from.
    distribution : str
        ``"uniform"``, ``"maxavg"``, or ``"minmax"``.
    lambda_maxavg : float
        Entropy regularisation for ``"maxavg"`` (higher = more uniform).
    num_patches : int
        Patches per slide.  ``-1`` for infinite streaming;
        the pipeline's ``patches_per_slide`` controls the budget.
    tissue_threshold : float
        Minimum tissue fraction to accept a candidate.
    max_retries : int
        Rejection attempts per patch before giving up.
    seed : int or None
        Random seed.

    References
    ----------
    Moellers, A. et al. (2026). "Mind the Gap: Continuous Magnification
    Sampling for Pathology Foundation Models." arXiv:2601.02198.
    """

    output_size: int = 224
    mpp_range: tuple[float, float] = (0.25, 2.0)
    distribution: str = "uniform"
    lambda_maxavg: float = 1.0
    num_patches: int = -1
    tissue_threshold: float = 0.4
    max_retries: int = 50
    seed: int | None = None

    def __post_init__(self) -> None:
        a, b = self.mpp_range
        if a <= 0 or b <= 0 or a >= b:
            raise ValueError(f"mpp_range must be (a, b) with 0 < a < b, got {self.mpp_range}")
        if self.output_size < 1:
            raise ValueError(f"output_size must be >= 1, got {self.output_size}")
        if self.distribution not in ("uniform", "maxavg", "minmax"):
            raise ValueError(
                f"distribution must be 'uniform', 'maxavg', or 'minmax', "
                f"got {self.distribution!r}"
            )
        if self.lambda_maxavg <= 0:
            raise ValueError(f"lambda_maxavg must be > 0, got {self.lambda_maxavg}")
        if self.num_patches < -1 or self.num_patches == 0:
            raise ValueError(f"num_patches must be -1 (infinite) or >= 1, got {self.num_patches}")
        if self.max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {self.max_retries}")

        self._rng = np.random.default_rng(self.seed)

        # Pre-compute sampling distribution on a fine grid
        grid = np.linspace(a, b, _GRID_SIZE)
        self._mpp_grid = grid

        if self.distribution == "uniform":
            self._mpp_weights = np.ones(_GRID_SIZE) / _GRID_SIZE
        elif self.distribution == "maxavg":
            self._mpp_weights = _compute_maxavg_weights(grid, a, b, self.lambda_maxavg)
        else:  # minmax
            self._mpp_weights = _compute_minmax_weights(grid)

    def sample(self, slide: SlideHandle, tissue_mask: TissueMask) -> Iterator[PatchCoordinate]:
        rng = self._rng
        props = slide.properties

        # No MPP metadata: fall back to single-level random sampling.
        if props.mpp is None:
            logger.warning(
                "Slide %s has no MPP metadata; falling back to level-0 "
                "random sampling with patch_size=%d",
                props.path,
                self.output_size,
            )
            inner = RandomSampler(
                patch_size=self.output_size,
                num_patches=self.num_patches,
                level=0,
                tissue_threshold=self.tissue_threshold,
                seed=int(rng.integers(0, 2**31)),
            )
            yield from inner.sample(slide, tissue_mask)
            return

        count = 0
        consecutive_failures = 0
        infinite = self.num_patches == -1

        while infinite or count < self.num_patches:
            # Sample target magnification from the chosen distribution
            if self.distribution == "uniform":
                target_mpp = float(rng.uniform(self.mpp_range[0], self.mpp_range[1]))
            else:
                idx = rng.choice(len(self._mpp_grid), p=self._mpp_weights)
                target_mpp = float(self._mpp_grid[idx])

            # Pick best level (coarsest with level_mpp <= target_mpp)
            level = _best_level_for_downsample(props, target_mpp)
            level_mpp = props.mpp * props.level_downsamples[level]

            # Compute crop size at that level
            crop_size = max(1, round(self.output_size * target_mpp / level_mpp))

            # Footprint in level-0 coordinates for bounds / tissue check
            ds = props.level_downsamples[level]
            crop_size_l0 = round(crop_size * ds)
            max_x = props.width - crop_size_l0
            max_y = props.height - crop_size_l0

            if max_x < 0 or max_y < 0:
                consecutive_failures += 1
                if consecutive_failures >= self.max_retries:
                    break  # slide too small for this mpp range
                continue

            # Rejection-sample a tissue location
            found = False
            for _ in range(self.max_retries):
                x = int(rng.integers(0, max_x + 1))
                y = int(rng.integers(0, max_y + 1))
                if tissue_mask.contains_tissue(
                    x, y, crop_size_l0, crop_size_l0, self.tissue_threshold
                ):
                    found = True
                    break

            if not found:
                # Different mpp → different crop size → might find tissue.
                # Don't hard-break; count as failure and try another mpp.
                consecutive_failures += 1
                if consecutive_failures >= self.max_retries:
                    break
                continue

            consecutive_failures = 0

            yield PatchCoordinate(
                x=x,
                y=y,
                level=level,
                patch_size=crop_size,
                mpp=target_mpp,
                slide_path=props.path,
            )
            count += 1
