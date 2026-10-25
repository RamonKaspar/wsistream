"""Tests for internal shared utilities."""

import cv2
import numpy as np

from wsistream._utils import infer_batch_size, resize_patch


class TestInferBatchSize:
    def test_prefers_image_field(self):
        batch = {
            "metadata": np.zeros((8, 2)),
            "image": np.zeros((4, 3, 16, 16)),
        }

        assert infer_batch_size(batch) == 4

    def test_uses_first_tensor_like_field_for_multi_view_batch(self):
        batch = {"patient_id": "P001", "global_view": np.zeros((6, 3, 16, 16))}

        assert infer_batch_size(batch) == 6

    def test_falls_back_to_sequence_field(self):
        batch = {"label": "tumour", "slide_path": ["a.svs", "b.svs"]}

        assert infer_batch_size(batch) == 2

    def test_supports_tensor_like_batch(self):
        assert infer_batch_size(np.zeros((5, 3))) == 5

    def test_uses_first_tensor_of_tuple_batch(self):
        assert infer_batch_size((np.zeros((7, 3, 16, 16)), np.zeros(7))) == 7

    def test_counts_list_of_samples(self):
        assert infer_batch_size([{"image": np.zeros((3, 16, 16))}] * 5) == 5

    def test_returns_one_when_size_cannot_be_inferred(self):
        assert infer_batch_size(object()) == 1


def _checkerboard(size: int) -> np.ndarray:
    """RGB image alternating black and white pixels, the worst case for aliasing."""
    board = (np.indices((size, size)).sum(axis=0) % 2 * 255).astype(np.uint8)
    return np.dstack([board] * 3)


class TestResizePatch:
    def test_shrinking_averages_fine_structure(self):
        """A non-integer shrink must average a checkerboard to grey instead of aliasing it."""
        patch = _checkerboard(515)

        resized = resize_patch(patch, 256)
        bilinear = cv2.resize(patch, (256, 256), interpolation=cv2.INTER_LINEAR)

        assert resized.shape == (256, 256, 3)
        assert resized.min() >= 126 and resized.max() <= 129
        assert bilinear.std() > 40

    def test_enlarging_uses_bilinear_interpolation(self):
        patch = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)

        resized = resize_patch(patch, 100)

        expected = cv2.resize(patch, (100, 100), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_array_equal(resized, expected)
