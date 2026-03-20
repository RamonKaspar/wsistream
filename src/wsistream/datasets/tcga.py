"""TCGA dataset adapter; parses metadata from TCGA slide filenames."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from wsistream.datasets.base import DatasetAdapter
from wsistream.types import SlideMetadata

# TCGA sample type codes (complete list)
# https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/sample-type-codes
TCGA_SAMPLE_TYPES = {
    "01": "Primary Solid Tumor",
    "02": "Recurrent Solid Tumor",
    "03": "Primary Blood Derived Cancer - Peripheral Blood",
    "04": "Recurrent Blood Derived Cancer - Bone Marrow",
    "05": "Additional - New Primary",
    "06": "Metastatic",
    "07": "Additional Metastatic",
    "08": "Human Tumor Original Cells",
    "09": "Primary Blood Derived Cancer - Bone Marrow",
    "10": "Blood Derived Normal",
    "11": "Solid Tissue Normal",
    "12": "Buccal Cell Normal",
    "13": "EBV Immortalized Normal",
    "14": "Bone Marrow Normal",
    "20": "Control Analyte",
    "40": "Recurrent Blood Derived Cancer - Peripheral Blood",
    "50": "Cell Lines",
    "60": "Primary Xenograft Tissue",
    "61": "Cell Line Derived Xenograft Tissue",
}

# TCGA slide barcode pattern:
# TCGA-{TSS}-{Participant}-{SampleType}{Vial}-{Portion}-{SlideID}
# Example: TCGA-3L-AA1B-01Z-00-DX1.8923A151-A690-40B7-9E5A-FCBEDFC2394F.svs
#
# Slide section types: DX = diagnostic (FFPE), TS = top section (frozen),
#                      BS = bottom section (frozen), MS = middle section (frozen)
# Slide order: number or letter (per GDC spec)
_TCGA_PATTERN = re.compile(
    r"TCGA-([A-Z0-9]{2})-([A-Z0-9]{4})-(\d{2})([A-Z])-(\d{2})-([A-Z]{2}[A-Z0-9])"
)


@dataclass
class TCGAAdapter(DatasetAdapter):
    """
    Parse TCGA slide barcodes into structured metadata.

    Extracts tissue source site, patient ID, sample type,
    and infers cancer type from the parent directory name
    (e.g., TCGA-BRCA, TCGA-LUAD).

    Parameters
    ----------
    cancer_type : str or None
        Override cancer type. If None, inferred from parent dir name.
    """

    cancer_type: str | None = None

    def parse_metadata(self, slide_path: str) -> SlideMetadata:
        filename = Path(slide_path).stem
        match = _TCGA_PATTERN.search(filename)

        if match is None:
            return SlideMetadata(
                slide_path=slide_path,
                dataset_name="TCGA",
                extra={"parse_error": f"Could not parse barcode from: {filename}"},
            )

        tss = match.group(1)
        participant = match.group(2)
        sample_code = match.group(3)
        vial = match.group(4)
        portion = match.group(5)
        slide_id = match.group(6)

        patient_id = f"TCGA-{tss}-{participant}"
        barcode = f"TCGA-{tss}-{participant}-{sample_code}{vial}-{portion}-{slide_id}"
        sample_type = TCGA_SAMPLE_TYPES.get(sample_code, f"Unknown ({sample_code})")

        # DX = diagnostic (FFPE), TS/BS/MS = frozen sections
        slide_section = slide_id[:2]
        is_frozen = slide_section in ("TS", "BS", "MS")

        # Try to infer cancer type from parent directory
        cancer = self.cancer_type
        if cancer is None:
            parent = Path(slide_path).parent.name
            if parent.startswith("TCGA-"):
                cancer = parent

        return SlideMetadata(
            slide_path=slide_path,
            dataset_name="TCGA",
            patient_id=patient_id,
            tissue_type=tss,
            cancer_type=cancer,
            sample_type=sample_type,
            extra={
                "tissue_source_site": tss,
                "vial": vial,
                "portion": portion,
                "slide_id": slide_id,
                "slide_section": slide_section,
                "is_frozen": is_frozen,
                "sample_code": sample_code,
                "barcode": barcode,
            },
        )
