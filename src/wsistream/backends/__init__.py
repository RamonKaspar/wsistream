"""WSI reader backends."""

from wsistream.backends.base import SlideBackend
from wsistream.backends.openslide import OpenSlideBackend
from wsistream.backends.tiffslide import TiffSlideBackend

__all__ = ["SlideBackend", "OpenSlideBackend", "TiffSlideBackend"]
