"""Tests for internal shared utilities."""

import numpy as np

from wsistream._utils import infer_batch_size


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

    def test_returns_one_when_size_cannot_be_inferred(self):
        assert infer_batch_size(object()) == 1
