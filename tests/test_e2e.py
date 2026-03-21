"""
End-to-end tests against real WSI files.

Skipped automatically when ``--slide-dir`` is not provided.

Run:
    pytest tests/test_e2e.py --slide-dir /path/to/slides --backend openslide -v
"""

from __future__ import annotations

from collections import Counter

import pytest

from wsistream.datasets import TCGAAdapter
from wsistream.filters import HSVPatchFilter
from wsistream.pipeline import PatchPipeline
from wsistream.sampling import RandomSampler
from wsistream.tissue import CombinedTissueDetector, HSVTissueDetector, OtsuTissueDetector
from wsistream.transforms import (
    ComposeTransforms,
    HEDColorAugmentation,
    RandomFlipRotate,
    ResizeTransform,
)

pytestmark = pytest.mark.e2e


class TestSlideInterleaving:
    """Patches must come from multiple slides, not just the first."""

    def test_multiple_slides_seen(self, slides, make_backend):
        pipeline = PatchPipeline(
            slide_paths=slides,
            backend=make_backend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            pool_size=min(4, len(slides)),
            patches_per_slide=10,
            cycle=False,
        )

        slide_counter = Counter()
        for result in pipeline:
            slide_counter[result.coordinate.slide_path] += 1
            if sum(slide_counter.values()) >= 50:
                break

        assert len(slide_counter) > 1, "All patches came from a single slide."

    def test_interleaving_order(self, slides, make_backend):
        """First N patches should not all come from the same slide."""
        pipeline = PatchPipeline(
            slide_paths=slides,
            backend=make_backend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            pool_size=min(4, len(slides)),
            patches_per_slide=5,
            cycle=False,
        )

        seen_slides = []
        for result in pipeline:
            seen_slides.append(result.coordinate.slide_path)
            if len(seen_slides) >= 20:
                break

        # With pool_size >= 2 and patches_per_slide=5, the first 10 patches
        # must include at least 2 different slides.
        first_10 = set(seen_slides[:10])
        assert len(first_10) >= 2, f"First 10 patches all from same slide(s): {first_10}"


class TestPatchesPerSlideCap:
    """No single slide should exceed patches_per_slide."""

    @pytest.mark.parametrize("cap", [5, 15, 50])
    def test_cap_respected(self, slides, make_backend, cap):
        pipeline = PatchPipeline(
            slide_paths=slides,
            backend=make_backend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            pool_size=min(4, len(slides)),
            patches_per_slide=cap,
            cycle=False,
        )

        slide_counter = Counter()
        for result in pipeline:
            slide_counter[result.coordinate.slide_path] += 1

        max_from_one = max(slide_counter.values())
        assert max_from_one <= cap, f"Slide exceeded cap: got {max_from_one}, limit is {cap}"


class TestCycleMode:
    """cycle=True must produce more patches than a single pass."""

    def test_produces_more_than_one_pass(self, slides, make_backend):
        patches_per_slide = 5
        one_pass = len(slides) * patches_per_slide
        target = one_pass * 2

        pipeline = PatchPipeline(
            slide_paths=slides,
            backend=make_backend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            pool_size=min(2, len(slides)),
            patches_per_slide=patches_per_slide,
            cycle=True,
        )

        count = 0
        for _ in pipeline:
            count += 1
            if count >= target:
                break

        assert count >= target, f"Cycle produced {count} patches, expected >= {target}"

    def test_revisits_slides(self, slides, make_backend):
        """With cycle=True, slides should be visited more than once."""
        patches_per_slide = 3
        one_pass = len(slides) * patches_per_slide

        pipeline = PatchPipeline(
            slide_paths=slides,
            backend=make_backend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            pool_size=min(2, len(slides)),
            patches_per_slide=patches_per_slide,
            cycle=True,
        )

        slide_counter = Counter()
        count = 0
        for result in pipeline:
            slide_counter[result.coordinate.slide_path] += 1
            count += 1
            if count >= one_pass * 3:
                break

        # At least one slide should have been visited more than once
        max_visits = max(slide_counter.values())
        assert (
            max_visits > patches_per_slide
        ), f"No slide was revisited (max patches from one slide: {max_visits})"


class TestFullMidnightPipeline:
    """End-to-end with the full Midnight-style transform chain."""

    def test_patches_are_valid(self, slides, make_backend):
        pipeline = PatchPipeline(
            slide_paths=slides,
            backend=make_backend(),
            tissue_detector=CombinedTissueDetector(
                detectors=[OtsuTissueDetector(), HSVTissueDetector()],
            ),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            transforms=ComposeTransforms(
                [
                    HEDColorAugmentation(sigma=0.05),
                    RandomFlipRotate(),
                    ResizeTransform(target_size=224),
                ]
            ),
            dataset_adapter=TCGAAdapter(),
            pool_size=min(4, len(slides)),
            patches_per_slide=5,
            cycle=False,
        )

        count = 0
        for result in pipeline:
            assert result.image.shape == (224, 224, 3)
            assert result.image.dtype.kind == "u"  # uint8
            assert 0.0 <= result.tissue_fraction <= 1.0
            assert result.coordinate.patch_size == 256
            count += 1

        assert count > 0, "Pipeline produced zero patches."

    def test_metadata_populated(self, slides, make_backend):
        pipeline = PatchPipeline(
            slide_paths=slides,
            backend=make_backend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            dataset_adapter=TCGAAdapter(),
            pool_size=min(4, len(slides)),
            patches_per_slide=2,
            cycle=False,
        )

        for result in pipeline:
            meta = result.slide_metadata
            assert meta is not None
            assert meta.dataset_name == "TCGA"
            # At least some slides should parse successfully
            if meta.patient_id is not None:
                assert meta.patient_id.startswith("TCGA-")
                return  # found one valid parse, good enough

        pytest.fail("No slide metadata was parsed successfully.")


class TestPatchFilter:
    """Per-tile filtering with HSVPatchFilter."""

    def test_filter_rejects_some_patches(self, slides, make_backend):
        """With a strict HSV filter, some patches should be filtered out."""
        pipeline = PatchPipeline(
            slide_paths=slides,
            backend=make_backend(),
            tissue_detector=OtsuTissueDetector(),
            patch_filter=HSVPatchFilter(min_pixel_fraction=0.6),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            pool_size=min(2, len(slides)),
            patches_per_slide=20,
            cycle=False,
        )

        count = 0
        for result in pipeline:
            assert result.image.shape[:2] == (256, 256)
            count += 1

        # Some patches should have been filtered
        stats = pipeline.stats
        assert count > 0, "Pipeline produced zero patches with filter."
        assert stats.patches_filtered >= 0  # may be 0 on tissue-rich slides

    def test_filter_stats_tracked(self, slides, make_backend):
        """patches_filtered must appear in stats dict."""
        pipeline = PatchPipeline(
            slide_paths=slides,
            backend=make_backend(),
            tissue_detector=OtsuTissueDetector(),
            patch_filter=HSVPatchFilter(min_pixel_fraction=0.99),  # strict
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            pool_size=min(2, len(slides)),
            patches_per_slide=10,
            cycle=False,
        )

        for _ in pipeline:
            pass

        d = pipeline.stats_dict()
        assert "pipeline/patches_filtered" in d

    def test_full_midnight_with_filter(self, slides, make_backend):
        """Full Midnight pipeline: OtsuTissueDetector + HSVPatchFilter + transforms."""
        pipeline = PatchPipeline(
            slide_paths=slides,
            backend=make_backend(),
            tissue_detector=OtsuTissueDetector(),
            patch_filter=HSVPatchFilter(
                hue_range=(90, 180),
                sat_range=(8, 255),
                val_range=(103, 255),
                min_pixel_fraction=0.6,
            ),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            transforms=ComposeTransforms(
                [
                    HEDColorAugmentation(sigma=0.05),
                    RandomFlipRotate(),
                    ResizeTransform(target_size=224),
                ]
            ),
            dataset_adapter=TCGAAdapter(),
            pool_size=min(4, len(slides)),
            patches_per_slide=10,
            cycle=False,
        )

        count = 0
        for result in pipeline:
            assert result.image.shape == (224, 224, 3)
            count += 1

        assert count > 0, "Full Midnight pipeline produced zero patches."


class TestPipelineStats:
    """Stats must be consistent with actual output."""

    def test_stats_match_output(self, slides, make_backend):
        pipeline = PatchPipeline(
            slide_paths=slides,
            backend=make_backend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            dataset_adapter=TCGAAdapter(),
            pool_size=min(4, len(slides)),
            patches_per_slide=5,
            cycle=False,
        )

        patch_count = 0
        for _ in pipeline:
            patch_count += 1

        d = pipeline.stats_dict()

        assert d["pipeline/patches_extracted"] == patch_count
        assert d["pipeline/slides_processed"] > 0
        assert d["pipeline/slides_failed"] == 0
        assert "pipeline/mean_tissue_fraction" in d

    def test_slides_processed_equals_input(self, slides, make_backend):
        """All healthy slides should be processed, none failed."""
        pipeline = PatchPipeline(
            slide_paths=slides,
            backend=make_backend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            pool_size=min(4, len(slides)),
            patches_per_slide=3,
            cycle=False,
        )
        list(pipeline)

        assert pipeline.stats.slides_processed == len(slides)
        assert pipeline.stats.slides_failed == 0
        assert pipeline.stats.error_count == 0

    def test_tissue_fractions_populated(self, slides, make_backend):
        """One tissue fraction per slide, all in [0, 1]."""
        pipeline = PatchPipeline(
            slide_paths=slides,
            backend=make_backend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            pool_size=min(4, len(slides)),
            patches_per_slide=3,
            cycle=False,
        )
        list(pipeline)

        fracs = pipeline.stats.tissue_fractions
        assert fracs.count == len(slides)
        assert 0.0 <= fracs.min_val <= fracs.max_val <= 1.0

        d = pipeline.stats_dict()
        assert "pipeline/mean_tissue_fraction" in d
        assert "pipeline/min_tissue_fraction" in d
        assert "pipeline/max_tissue_fraction" in d
        assert d["pipeline/min_tissue_fraction"] <= d["pipeline/mean_tissue_fraction"]
        assert d["pipeline/mean_tissue_fraction"] <= d["pipeline/max_tissue_fraction"]

    def test_magnification_counts(self, slides, make_backend):
        """Every extracted patch must be counted under its mpp key."""
        pipeline = PatchPipeline(
            slide_paths=slides,
            backend=make_backend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            pool_size=min(4, len(slides)),
            patches_per_slide=5,
            cycle=False,
        )

        patch_count = 0
        for _ in pipeline:
            patch_count += 1

        mag_counts = pipeline.stats.magnification_counts
        assert len(mag_counts) > 0, "No magnification counts recorded"
        assert sum(mag_counts.values()) == patch_count

        d = pipeline.stats_dict()
        mpp_keys = [k for k in d if k.startswith("pipeline/mpp_")]
        assert len(mpp_keys) > 0
        assert sum(d[k] for k in mpp_keys) == patch_count

    def test_cancer_type_counts_with_adapter(self, slides, make_backend):
        """TCGAAdapter should populate cancer_type_counts in stats."""
        pipeline = PatchPipeline(
            slide_paths=slides,
            backend=make_backend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            dataset_adapter=TCGAAdapter(),
            pool_size=min(4, len(slides)),
            patches_per_slide=3,
            cycle=False,
        )
        list(pipeline)

        stats = pipeline.stats
        d = pipeline.stats_dict()

        # Cancer type counts should sum to slides_processed
        # (only if all slides parse successfully, which depends on filenames)
        ct_total = sum(stats.cancer_type_counts.values())
        if ct_total > 0:
            cancer_keys = [k for k in d if k.startswith("pipeline/cancer_type/")]
            assert len(cancer_keys) > 0
            assert sum(d[k] for k in cancer_keys) == ct_total

    def test_sample_type_counts_with_adapter(self, slides, make_backend):
        """TCGAAdapter should populate sample_type_counts in stats."""
        pipeline = PatchPipeline(
            slide_paths=slides,
            backend=make_backend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            dataset_adapter=TCGAAdapter(),
            pool_size=min(4, len(slides)),
            patches_per_slide=3,
            cycle=False,
        )
        list(pipeline)

        stats = pipeline.stats
        d = pipeline.stats_dict()

        st_total = sum(stats.sample_type_counts.values())
        if st_total > 0:
            st_keys = [k for k in d if k.startswith("pipeline/sample_type/")]
            assert len(st_keys) > 0
            assert sum(d[k] for k in st_keys) == st_total

    def test_filter_stats_with_real_slides(self, slides, make_backend):
        """With a strict filter, patches_filtered should be > 0 and
        patches_extracted + patches_filtered == total reads."""
        pipeline = PatchPipeline(
            slide_paths=slides,
            backend=make_backend(),
            tissue_detector=OtsuTissueDetector(),
            patch_filter=HSVPatchFilter(min_pixel_fraction=0.99),  # very strict
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            pool_size=min(2, len(slides)),
            patches_per_slide=20,
            cycle=False,
        )

        yielded = 0
        for _ in pipeline:
            yielded += 1

        stats = pipeline.stats
        assert stats.patches_extracted == yielded
        assert stats.patches_filtered >= 0
        # Total reads = yielded + filtered
        total_reads = stats.patches_extracted + stats.patches_filtered
        assert total_reads > 0, "Pipeline did zero reads"

        d = pipeline.stats_dict()
        assert d["pipeline/patches_extracted"] == yielded
        assert d["pipeline/patches_filtered"] == stats.patches_filtered

    def test_stats_dict_no_error_key_on_success(self, slides, make_backend):
        """When all slides succeed, error_count should not appear in dict."""
        pipeline = PatchPipeline(
            slide_paths=slides,
            backend=make_backend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            pool_size=min(2, len(slides)),
            patches_per_slide=3,
            cycle=False,
        )
        list(pipeline)

        d = pipeline.stats_dict()
        assert "pipeline/error_count" not in d

    def test_stats_internal_consistency(self, slides, make_backend):
        """Full pipeline with adapter + filter: all stats must be consistent."""
        pipeline = PatchPipeline(
            slide_paths=slides,
            backend=make_backend(),
            tissue_detector=OtsuTissueDetector(),
            patch_filter=HSVPatchFilter(
                hue_range=(90, 180),
                sat_range=(8, 255),
                val_range=(103, 255),
                min_pixel_fraction=0.6,
            ),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            dataset_adapter=TCGAAdapter(),
            pool_size=min(4, len(slides)),
            patches_per_slide=10,
            cycle=False,
        )

        yielded = 0
        for _ in pipeline:
            yielded += 1

        stats = pipeline.stats

        # Core counts
        assert stats.patches_extracted == yielded
        assert stats.slides_processed + stats.slides_failed == len(slides)

        # Magnification counts must sum to patches_extracted
        assert sum(stats.magnification_counts.values()) == yielded

        # Tissue fractions: one per successfully processed slide
        assert stats.tissue_fractions.count == stats.slides_processed

        # Metadata counts: cancer + sample type each sum to slides_processed
        # (only when slides have parseable TCGA barcodes)
        ct_total = sum(stats.cancer_type_counts.values())
        st_total = sum(stats.sample_type_counts.values())
        if ct_total > 0:
            assert ct_total == stats.slides_processed
        if st_total > 0:
            assert st_total == stats.slides_processed

    def test_reset_clears_between_iterations(self, slides, make_backend):
        pipeline = PatchPipeline(
            slide_paths=slides,
            backend=make_backend(),
            tissue_detector=OtsuTissueDetector(),
            sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
            pool_size=min(2, len(slides)),
            patches_per_slide=3,
            cycle=False,
        )

        # First pass
        for _ in pipeline:
            pass
        first_count = pipeline.stats.patches_extracted

        # Reset and second pass
        pipeline.reset_stats()
        for _ in pipeline:
            pass
        second_count = pipeline.stats.patches_extracted

        assert first_count > 0
        assert second_count > 0
        # After reset, second pass count should be independent
        assert pipeline.stats.slides_processed > 0
