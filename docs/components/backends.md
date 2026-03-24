# Backends

Backends handle reading pixels from WSI files. You must explicitly choose one -- there is no auto-detection.

## OpenSlideBackend

Uses the [OpenSlide](https://openslide.org/) C library. Supports the widest range of vendor formats: SVS (Aperio), NDPI (Hamamatsu), MRXS (3DHISTECH), SCN (Leica), BIF (Ventana), and generic tiled TIFF.

```python
from wsistream.backends import OpenSlideBackend

backend = OpenSlideBackend()
```

Requires the OpenSlide C library installed on the system:

- Ubuntu/Debian: `apt-get install openslide-tools`
- macOS: `brew install openslide`

## TiffSlideBackend

Uses [TiffSlide](https://github.com/Bayer-Group/tiffslide), a pure-Python drop-in replacement for OpenSlide based on `tifffile`. No C dependencies. Supports cloud storage (S3, GCS) via `fsspec`.

```python
from wsistream.backends import TiffSlideBackend

backend = TiffSlideBackend()
```

## Which to use?

| | OpenSlide | TiffSlide |
|---|-----------|-----------|
| **Format support** | Widest (SVS, NDPI, MRXS, SCN, BIF, ...) | SVS, generic TIFF, NDPI (partial) |
| **Installation** | Needs C library | Pure Python, `pip install` only |
| **Cloud storage** | No | Yes (S3, GCS via fsspec) |

For TCGA data (predominantly SVS format), both work. If you need broad vendor format support, use `OpenSlideBackend`. If you need cloud storage or want to avoid C dependencies, use `TiffSlideBackend`.

## SlideHandle

`SlideHandle` wraps a backend and provides a unified interface for reading slides. The pipeline uses it internally, but you can also use it directly to explore slides or build custom workflows.

```python
from wsistream.backends import OpenSlideBackend
from wsistream.slide import SlideHandle

slide = SlideHandle("path/to/slide.svs", backend=OpenSlideBackend())

# Slide metadata
props = slide.properties
print(props.dimensions)        # (width, height) at level 0
print(props.level_count)       # number of pyramid levels
print(props.level_downsamples) # downsample factor per level
print(props.mpp)               # microns per pixel at level 0 (or None)
print(props.vendor)            # scanner vendor (or None)

# Read a 256x256 patch at level 0
patch = slide.read_region(x=1000, y=2000, width=256, height=256, level=0)

# Low-resolution thumbnail for visualization or tissue detection
thumbnail = slide.get_thumbnail(size=(1024, 1024))

# Find the pyramid level closest to a target microns-per-pixel
level = slide.best_level_for_mpp(target_mpp=0.5)

slide.close()
```

`SlideHandle` supports context managers:

```python
with SlideHandle("slide.svs", backend=OpenSlideBackend()) as slide:
    patch = slide.read_region(x=0, y=0, width=256, height=256)
```
