"""Tests for patch transforms."""

import numpy as np
import pytest

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
        with pytest.raises(TypeError, match="requires explicit mean and std"):
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
        out = NormalizeTransform(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )(img)
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


class TestAlbumentationsWrapper:
    def test_none_is_noop(self, random_patch):
        out = AlbumentationsWrapper()(random_patch)
        np.testing.assert_array_equal(out, random_patch)

    def test_applies_wrapped_transform(self, random_patch):
        class _AddOne:
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
            def __call__(self, *, image):
                return {"image": image}

            def __repr__(self):
                return "Identity()"

        wrapper = AlbumentationsWrapper(_Identity())
        assert repr(wrapper) == "AlbumentationsWrapper(Identity())"


class TestSeededRandomness:
    """Seeded transforms must produce a varied sequence, not the same output."""

    def test_flip_rotate_varies_across_calls(self):
        t = RandomFlipRotate(p_hflip=1.0, p_vflip=1.0, p_rot90=1.0, seed=42)
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        outputs = [t(img.copy()) for _ in range(10)]
        # Not all outputs should be identical
        unique = sum(
            1 for i in range(1, len(outputs))
            if not np.array_equal(outputs[0], outputs[i])
        )
        assert unique > 0, "Seeded RandomFlipRotate produced identical output every call"

    def test_hed_varies_across_calls(self):
        t = HEDColorAugmentation(sigma=0.05, seed=42)
        img = np.random.randint(50, 200, (32, 32, 3), dtype=np.uint8)

        outputs = [t(img.copy()) for _ in range(5)]
        unique = sum(
            1 for i in range(1, len(outputs))
            if not np.array_equal(outputs[0], outputs[i])
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
        # With sigma=0, perturbation is *1.0 — but the RGB->HED->RGB round
        # trip has inherent numerical loss from the color deconvolution.
        assert np.abs(out.astype(int) - img.astype(int)).mean() < 25
