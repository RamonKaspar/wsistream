"""
Demo: test the full wsistream pipeline on a single slide.

Usage:
    python examples/demo_pipeline.py --slide path/to/slide.svs --backend openslide
    python examples/demo_pipeline.py --slide path/to/slide.svs --backend tiffslide --compare-augmentations
"""

from __future__ import annotations

import argparse
from pathlib import Path


def get_backend(name: str):
    """Instantiate the requested backend."""
    if name == "openslide":
        from wsistream.backends import OpenSlideBackend

        return OpenSlideBackend()
    elif name == "tiffslide":
        from wsistream.backends import TiffSlideBackend

        return TiffSlideBackend()
    else:
        raise ValueError(f"Unknown backend: {name!r}")


def demo_tissue_detection(slide_path: str, backend_name: str, output_dir: Path) -> None:
    """Compare tissue detection strategies."""
    from wsistream.slide import SlideHandle
    from wsistream.tissue import (
        CLAMTissueDetector,
        CombinedTissueDetector,
        HSVTissueDetector,
        OtsuTissueDetector,
    )
    from wsistream.viz import plot_tissue_mask

    print("\n--- Tissue Detection ---")
    with SlideHandle(slide_path, backend=get_backend(backend_name)) as slide:
        print(f"  Slide: {slide}")
        thumbnail = slide.get_thumbnail((1024, 1024))
        th, tw = thumbnail.shape[:2]
        sw, sh = slide.properties.dimensions
        downsample = (sw / tw, sh / th)

        for name, det in [
            ("Otsu", OtsuTissueDetector()),
            ("HSV", HSVTissueDetector()),
            ("CLAM", CLAMTissueDetector()),
            (
                "Combined",
                CombinedTissueDetector(detectors=[OtsuTissueDetector(), HSVTissueDetector()]),
            ),
        ]:
            mask = det.detect(thumbnail, downsample=downsample)
            frac = mask.sum() / mask.size
            print(f"  {name}: tissue fraction = {frac:.3f}")
            plot_tissue_mask(thumbnail, mask, save_path=output_dir / f"tissue_{name.lower()}.png")

    print(f"  Saved to {output_dir}")


def demo_patch_sampling(slide_path: str, backend_name: str, output_dir: Path) -> None:
    """Random sampling with filtering, metadata, and location visualization."""
    from wsistream.datasets import TCGAAdapter
    from wsistream.filters import HSVPatchFilter
    from wsistream.pipeline import PatchPipeline
    from wsistream.sampling import RandomSampler
    from wsistream.slide import SlideHandle
    from wsistream.tissue import OtsuTissueDetector
    from wsistream.viz import plot_patch_grid, plot_sampling_locations

    print("\n--- Patch Sampling ---")
    pipeline = PatchPipeline(
        slide_paths=[slide_path],
        backend=get_backend(backend_name),
        tissue_detector=OtsuTissueDetector(),
        patch_filter=HSVPatchFilter(),
        sampler=RandomSampler(patch_size=256, num_patches=-1, seed=42),
        dataset_adapter=TCGAAdapter(),
        pool_size=1,
        patches_per_slide=16,
    )

    patches, coords = [], []
    metadata_printed = False
    for result in pipeline:
        patches.append(result.image)
        coords.append(result.coordinate)
        if result.slide_metadata and not metadata_printed:
            print(
                f"  Metadata: patient={result.slide_metadata.patient_id}, "
                f"cancer={result.slide_metadata.cancer_type}, "
                f"sample={result.slide_metadata.sample_type}"
            )
            metadata_printed = True

    print(f"  Extracted {len(patches)} patches")
    print(f"  Stats: {pipeline.stats_dict()}")

    if patches:
        plot_patch_grid(patches, ncols=4, save_path=output_dir / "patches_random.png")

        # Show where patches were sampled on the thumbnail
        with SlideHandle(slide_path, backend=get_backend(backend_name)) as slide:
            thumbnail = slide.get_thumbnail((1024, 1024))
            dims = slide.properties.dimensions
        plot_sampling_locations(
            thumbnail,
            coords,
            slide_dimensions=dims,
            save_path=output_dir / "sampling_locations.png",
        )

    print(f"  Saved to {output_dir}")


def demo_augmentations(slide_path: str, backend_name: str, output_dir: Path) -> None:
    """Compare augmentation strategies."""
    from wsistream.pipeline import PatchPipeline
    from wsistream.sampling import RandomSampler
    from wsistream.tissue import OtsuTissueDetector
    from wsistream.transforms import HEDColorAugmentation, RandomFlipRotate
    from wsistream.viz import compare_transforms

    print("\n--- Augmentation Comparison ---")
    pipeline = PatchPipeline(
        slide_paths=[slide_path],
        backend=get_backend(backend_name),
        tissue_detector=OtsuTissueDetector(),
        sampler=RandomSampler(patch_size=256, num_patches=1, seed=42),
        pool_size=1,
        patches_per_slide=1,
    )

    original = None
    for result in pipeline:
        original = result.image
        break

    if original is None:
        print("  Could not extract a patch.")
        return

    compare_transforms(
        original,
        {
            "HED s=0.03": HEDColorAugmentation(sigma=0.03),
            "HED s=0.05": HEDColorAugmentation(sigma=0.05),
            "HED s=0.10": HEDColorAugmentation(sigma=0.10),
            "Flip+Rotate": RandomFlipRotate(),
        },
        n_samples=5,
        save_path=output_dir / "augmentation_comparison.png",
    )
    print(f"  Saved to {output_dir}/augmentation_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="wsistream demo")
    parser.add_argument("--slide", required=True, help="Path to a WSI file")
    parser.add_argument("--backend", required=True, choices=["openslide", "tiffslide"])
    parser.add_argument("--output-dir", default="./wsistream_demo_output")
    parser.add_argument("--compare-augmentations", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    demo_tissue_detection(args.slide, args.backend, output_dir)
    demo_patch_sampling(args.slide, args.backend, output_dir)

    if args.compare_augmentations:
        demo_augmentations(args.slide, args.backend, output_dir)

    print(f"\nDone! Outputs in: {output_dir}")


if __name__ == "__main__":
    main()
