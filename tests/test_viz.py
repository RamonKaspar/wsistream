"""Tests for visualization utilities."""

import numpy as np

from wsistream.viz import plot_patch_grid, plot_sampling_locations, plot_tissue_mask


class TestPlotTissueMask:
    def test_returns_correct_shape(self, white_thumbnail, sample_mask):
        overlay = plot_tissue_mask(white_thumbnail, sample_mask)
        assert overlay.shape == (100, 100, 3)
        assert overlay.dtype == np.uint8

    def test_custom_color(self, white_thumbnail, sample_mask):
        overlay = plot_tissue_mask(white_thumbnail, sample_mask, color=(255, 0, 0))
        assert overlay.shape == (100, 100, 3)

    def test_tissue_overlay_leaves_background_unchanged(self, white_thumbnail, sample_mask):
        overlay = plot_tissue_mask(white_thumbnail, sample_mask, alpha=0.5, color=(0, 255, 0))
        np.testing.assert_array_equal(overlay[5, 5], white_thumbnail[5, 5])
        np.testing.assert_array_equal(overlay[50, 50], [40, 168, 40])  # (80 + color) / 2


class TestPlotPatchGrid:
    def test_grid_layout(self):
        patches = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(6)]
        grid = plot_patch_grid(patches, ncols=3)
        assert grid.ndim == 3

    def test_empty_returns_placeholder(self):
        grid = plot_patch_grid([])
        assert grid.shape == (128, 128, 3)

    def test_single_patch(self):
        patches = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)]
        grid = plot_patch_grid(patches, ncols=4)
        assert grid.ndim == 3

    def test_with_titles(self):
        patches = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
        grid = plot_patch_grid(patches, ncols=3, titles=["a", "b", "c"])
        assert grid.ndim == 3


class TestPlotSamplingLocations:
    def test_returns_correct_shape(self):
        from wsistream.types import PatchCoordinate

        thumb = np.ones((100, 100, 3), dtype=np.uint8) * 200
        coords = [
            PatchCoordinate(x=500, y=500, level=0, patch_size=256, mpp=0.5, slide_path="t.svs"),
        ]
        vis = plot_sampling_locations(thumb, coords, slide_dimensions=(1000, 1000))
        assert vis.shape == (100, 100, 3)
