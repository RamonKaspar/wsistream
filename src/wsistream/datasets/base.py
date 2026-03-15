"""Abstract base class for dataset adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from wsistream.types import SlideMetadata


class DatasetAdapter(ABC):
    """
    Extract dataset-specific metadata from a slide path.

    Subclass this for each dataset (TCGA, CPTAC, Camelyon, etc.)
    so that the pipeline can log tissue type, cancer type, patient ID,
    and other dataset-specific information.
    """

    @abstractmethod
    def parse_metadata(self, slide_path: str) -> SlideMetadata:
        """
        Parse metadata from a slide path.

        Parameters
        ----------
        slide_path : str
            Full path to the WSI file.

        Returns
        -------
        SlideMetadata
            Parsed metadata for this slide.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
