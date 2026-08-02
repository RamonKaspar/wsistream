"""Tests for patch transforms."""

import numpy as np
import pytest

from wsistream.pipeline import PatchPipeline
from wsistream.transforms import (
    AlbumentationsWrapper,
    ComposeTransforms,
    HEDColorAugmentation,
    NormalizeTransform,
    RandomFlipRotate,
    ResizeTransform,
)


class TestRandomFlipRotate:
    def test_preserves_shape_and_dtype(self, random_patch):
        out = RandomFlipRotate(seed=42)(random_patch)
        assert out.shape == random_patch.shape
        assert out.dtype == np.uint8

    def test_no_ops_with_zero_probability(self, random_patch):
        out = RandomFlipRotate(p_hflip=0, p_vflip=0, p_rot90=0)(random_patch)
        np.testing.assert_array_equal(out, random_patch)

    def test_always_flips(self, random_patch):
        out = RandomFlipRotate(p_hflip=1.0, p_vflip=0, p_rot90=0)(random_patch)
        np.testing.assert_array_equal(out, np.flip(random_patch, axis=1))


class TestResizeTransform:
    def test_resizes(self, random_patch_256):
        out = ResizeTransform(target_size=224)(random_patch_256)
        assert out.shape == (224, 224, 3)

    def test_noop_if_already_correct(self):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        out = ResizeTransform(target_size=224)(img)
        np.testing.assert_array_equal(out, img)

    def test_upsizes(self, random_patch):
        out = ResizeTransform(target_size=128)(random_patch)
        assert out.shape == (128, 128, 3)


class TestNormalizeTransform:
    def test_missing_args_raises(self):
        with pytest.raises(TypeError):
            NormalizeTransform()

    def test_output_float32(self):
        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        out = NormalizeTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(img)
        assert out.dtype == np.float32

    def test_normalization_value(self):
        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        out = NormalizeTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(img)
        # (128/255 - 0.5) / 0.5 ~= 0.004
        assert abs(out[0, 0, 0]) < 0.05

    def test_zero_mean_unit_std(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        out = NormalizeTransform(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))(img)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_imagenet_normalization(self):
        img = np.ones((10, 10, 3), dtype=np.uint8) * 128
        out = NormalizeTransform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(img)
        assert out.dtype == np.float32
        assert out.shape == (10, 10, 3)


class TestComposeTransforms:
    def test_chain(self, random_patch_256):
        norm = NormalizeTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        out = ComposeTransforms([ResizeTransform(224), norm])(random_patch_256)
        assert out.shape == (224, 224, 3)
        assert out.dtype == np.float32

    def test_empty_compose(self, random_patch):
        out = ComposeTransforms([])(random_patch)
        np.testing.assert_array_equal(out, random_patch)

    def test_repr(self):
        norm = NormalizeTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        c = ComposeTransforms([ResizeTransform(224), norm])
        r = repr(c)
        assert "ResizeTransform" in r
        assert "NormalizeTransform" in r


class _SeededFake:
    """Stand-in for an albumentations Compose that honours set_random_seed."""

    def __init__(self):
        self.seeds = []
        self._rng = np.random.default_rng(0)

    def set_random_seed(self, seed):
        self.seeds.append(seed)
        self._rng = np.random.default_rng(seed)

    def __call__(self, *, image):
        shift = int(self._rng.integers(0, 60))
        return {"image": np.clip(image.astype(np.int16) + shift, 0, 255).astype(np.uint8)}

    def __repr__(self):
        return "SeededFake()"


class TestAlbumentationsWrapper:
    def test_none_is_noop(self, random_patch):
        out = AlbumentationsWrapper()(random_patch)
        np.testing.assert_array_equal(out, random_patch)

    def test_applies_wrapped_transform(self, random_patch):
        class _AddOne:
            def set_random_seed(self, seed):
                pass

            def __call__(self, *, image):
                out = np.clip(image.astype(np.int16) + 1, 0, 255).astype(np.uint8)
                return {"image": out}

            def __repr__(self):
                return "AddOne()"

        out = AlbumentationsWrapper(_AddOne())(random_patch)
        expected = np.clip(random_patch.astype(np.int16) + 1, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(out, expected)

    def test_repr(self):
        class _Identity:
            def set_random_seed(self, seed):
                pass

            def __call__(self, *, image):
                return {"image": image}

            def __repr__(self):
                return "Identity()"

        wrapper = AlbumentationsWrapper(_Identity())
        assert repr(wrapper) == "AlbumentationsWrapper(Identity())"


class TestAlbumentationsSeeding:
    """The wrapper must carry pipeline-controlled seeds into albumentations.

    Albumentations draws from the numpy global RNG, which PyTorch does not
    reseed per DataLoader worker, so without this plumbing every forked worker
    replays an identical augmentation sequence.
    """

    def test_seeds_wrapped_transform_once_per_reseed(self, random_patch):
        fake = _SeededFake()
        wrapper = AlbumentationsWrapper(fake, seed=1)

        for _ in range(5):
            wrapper(random_patch)
        assert fake.seeds == [fake.seeds[0]], "seed pushed more than once without a reseed"

        PatchPipeline._reseed_transform(wrapper, (0, 4321))
        wrapper(random_patch)
        assert len(fake.seeds) == 2, "pipeline reseed did not reach the wrapper"
        assert fake.seeds[0] != fake.seeds[1]

    def test_pipeline_reseed_is_reachable(self):
        """_reseed_transform must replace the wrapper's RNG, not skip it."""
        wrapper = AlbumentationsWrapper(_SeededFake(), seed=1)
        before = wrapper._rng
        PatchPipeline._reseed_transform(wrapper, (0, 99))
        assert wrapper._rng is not before

    def test_nested_in_compose_is_reached(self, random_patch):
        fake = _SeededFake()
        chain = ComposeTransforms([ResizeTransform(32), AlbumentationsWrapper(fake, seed=1)])

        chain(random_patch)
        first = list(fake.seeds)

        PatchPipeline._reseed_transform(chain, (0, 777))
        chain(random_patch)
        assert len(fake.seeds) == len(first) + 1
        assert fake.seeds[-1] != first[-1]

    def test_distinct_workers_diverge(self, random_patch):
        """Two workers (distinct pipeline seeds) must not share a sequence."""
        outputs = []
        for pid in (1000, 2000):
            wrapper = AlbumentationsWrapper(_SeededFake(), seed=1)
            PatchPipeline._reseed_transform(wrapper, (0, pid))
            outputs.append([wrapper(random_patch).mean() for _ in range(5)])

        assert outputs[0] != outputs[1], "forked workers produced identical augmentations"

    def test_same_base_seed_is_reproducible(self, random_patch):
        runs = []
        for _ in range(2):
            wrapper = AlbumentationsWrapper(_SeededFake(), seed=1)
            PatchPipeline._reseed_transform(wrapper, (0, 55))
            runs.append([wrapper(random_patch).mean() for _ in range(5)])

        assert runs[0] == runs[1]

    def test_warns_when_set_random_seed_missing(self, random_patch):
        class _Legacy:
            def __call__(self, *, image):
                return {"image": image}

        wrapper = AlbumentationsWrapper(_Legacy())
        with pytest.warns(UserWarning, match="set_random_seed"):
            wrapper(random_patch)

    def test_real_albumentations_diverges_across_workers(self, random_patch):
        """End-to-end check against the actual albumentations API."""
        A = pytest.importorskip("albumentations")

        outputs = []
        for pid in (1000, 2000):
            wrapper = AlbumentationsWrapper(
                A.Compose([A.ColorJitter(brightness=0.5, contrast=0.5, p=1.0)])
            )
            PatchPipeline._reseed_transform(wrapper, (0, pid))
            outputs.append([float(wrapper(random_patch).mean()) for _ in range(5)])

        assert outputs[0] != outputs[1]


class TestSeededRandomness:
    """Seeded transforms must produce a varied sequence, not the same output."""

    def test_flip_rotate_varies_across_calls(self):
        t = RandomFlipRotate(p_hflip=1.0, p_vflip=1.0, p_rot90=1.0, seed=42)
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        outputs = [t(img.copy()) for _ in range(10)]
        # Not all outputs should be identical
        unique = sum(
            1 for i in range(1, len(outputs)) if not np.array_equal(outputs[0], outputs[i])
        )
        assert unique > 0, "Seeded RandomFlipRotate produced identical output every call"

    def test_hed_varies_across_calls(self):
        t = HEDColorAugmentation(sigma=0.05, seed=42)
        img = np.random.randint(50, 200, (32, 32, 3), dtype=np.uint8)

        outputs = [t(img.copy()) for _ in range(5)]
        unique = sum(
            1 for i in range(1, len(outputs)) if not np.array_equal(outputs[0], outputs[i])
        )
        assert unique > 0, "Seeded HEDColorAugmentation produced identical output every call"

    def test_seeded_is_reproducible(self):
        """Two instances with the same seed should produce the same sequence."""
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        t1 = RandomFlipRotate(seed=99)
        t2 = RandomFlipRotate(seed=99)

        for _ in range(5):
            out1 = t1(img.copy())
            out2 = t2(img.copy())
            np.testing.assert_array_equal(out1, out2)


class TestHEDColorAugmentation:
    def test_preserves_shape_and_dtype(self):
        img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        out = HEDColorAugmentation(sigma=0.05, seed=42)(img)
        assert out.shape == (64, 64, 3)
        assert out.dtype == np.uint8

    def test_zero_sigma_is_near_identity(self):
        img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        out = HEDColorAugmentation(sigma=0.0)(img)
        # With sigma=0, alpha is 1.0 and beta is 0.0 — but the RGB->HED->RGB
        # round trip has inherent numerical loss from the color deconvolution.
        assert np.abs(out.astype(int) - img.astype(int)).mean() < 25

    def test_rejects_negative_sigma(self):
        with pytest.raises(ValueError, match="sigma must be >= 0"):
            HEDColorAugmentation(sigma=-0.1)
        with pytest.raises(ValueError, match="sigma_bias must be >= 0"):
            HEDColorAugmentation(sigma_bias=-0.1)


class TestHEDMatchesTellez:
    """The perturbation must be s' = alpha * s + beta with both terms uniform.

    Verified against Tellez et al. 2018 (arXiv:1808.05896, sec. IV-B) and the
    HistomicsTK reference implementation, which draws
    alpha ~ U(1 - sigma1, 1 + sigma1) and beta ~ U(-sigma2, sigma2).
    """

    @staticmethod
    def _draws(sigma, sigma_bias, n=4000):
        """Recover the (alpha, beta) pairs a transform would draw."""
        t = HEDColorAugmentation(sigma=sigma, sigma_bias=sigma_bias, seed=0)
        alphas, betas = [], []
        for _ in range(n):
            alphas.append(t._rng.uniform(1.0 - sigma, 1.0 + sigma))
            betas.append(t._rng.uniform(-sigma_bias, sigma_bias))
        return np.array(alphas), np.array(betas)

    def test_additive_term_is_applied(self):
        """sigma=0 must still perturb, because beta alone is non-zero."""
        img = np.full((32, 32, 3), 160, dtype=np.uint8)
        out = HEDColorAugmentation(sigma=0.0, sigma_bias=0.2, seed=0)(img)
        baseline = HEDColorAugmentation(sigma=0.0, sigma_bias=0.0, seed=0)(img)
        assert not np.array_equal(out, baseline), "additive beta term has no effect"

    def test_multiplicative_term_is_applied(self):
        """sigma_bias=0 must still perturb, because alpha alone is non-unit."""
        img = np.full((32, 32, 3), 160, dtype=np.uint8)
        out = HEDColorAugmentation(sigma=0.2, sigma_bias=0.0, seed=0)(img)
        baseline = HEDColorAugmentation(sigma=0.0, sigma_bias=0.0, seed=0)(img)
        assert not np.array_equal(out, baseline), "multiplicative alpha term has no effect"

    def test_draws_are_bounded_not_gaussian(self):
        """Uniform draws are strictly bounded; a Gaussian would have tails."""
        sigma = 0.05
        alphas, betas = self._draws(sigma, sigma)
        assert alphas.min() >= 1.0 - sigma and alphas.max() <= 1.0 + sigma
        assert betas.min() >= -sigma and betas.max() <= sigma
        # A Gaussian(0, sigma) would put ~0.3% of mass outside +/- 3 sigma.
        # Bounded support also means the extremes are actually reached.
        assert alphas.max() > 1.0 + 0.9 * sigma
        assert betas.min() < -0.9 * sigma

    def test_draws_are_uniform(self):
        """A uniform sample has mean at the centre and variance range^2/12."""
        sigma = 0.05
        alphas, betas = self._draws(sigma, sigma)
        assert abs(alphas.mean() - 1.0) < 0.005
        assert abs(betas.mean()) < 0.005
        expected_var = (2 * sigma) ** 2 / 12
        assert abs(alphas.var() - expected_var) < 0.2 * expected_var
        assert abs(betas.var() - expected_var) < 0.2 * expected_var

    def test_sigma_bias_defaults_to_disabled(self):
        """beta is off by default: its scale does not transfer from the paper."""
        img = np.full((32, 32, 3), 160, dtype=np.uint8)
        default = HEDColorAugmentation(sigma=0.05, seed=3)(img)
        explicit_off = HEDColorAugmentation(sigma=0.05, sigma_bias=0.0, seed=3)(img)
        np.testing.assert_array_equal(default, explicit_off)

    def test_default_stays_in_plausible_stain_range(self):
        """Guards the scale trap: beta at the paper's sigma shifts hue wildly.

        On skimage's stain scale a beta of +/-0.05 is several times the channel
        mean, which turns tissue yellow/cyan/purple. The default must stay in a
        regime where the mean channel order of an H&E patch is preserved.
        """
        rng = np.random.default_rng(0)
        img = np.stack(
            [
                rng.integers(170, 210, (48, 48), dtype=np.uint8),  # R high
                rng.integers(90, 130, (48, 48), dtype=np.uint8),  # G low
                rng.integers(150, 190, (48, 48), dtype=np.uint8),  # B mid
            ],
            axis=-1,
        )
        t = HEDColorAugmentation(sigma=0.05, seed=0)
        for _ in range(25):
            out = t(img).reshape(-1, 3).mean(0)
            assert out[0] > out[1] and out[2] > out[1], f"channel order broken: {out}"
