"""Shared helpers for visualization scripts."""

from __future__ import annotations


def get_backend(name: str):
    """Instantiate the requested WSI backend."""
    if name == "openslide":
        from wsistream.backends import OpenSlideBackend

        return OpenSlideBackend()
    elif name == "tiffslide":
        from wsistream.backends import TiffSlideBackend

        return TiffSlideBackend()
    else:
        raise ValueError(f"Unknown backend: {name!r}")
