"""TCGA dataset adapter and slide download helpers.

Provides:
- ``TCGAAdapter``: parse TCGA barcodes from slide filenames into structured metadata.
- ``query_tcga_slides``: query the GDC API for available TCGA slide images.
- ``download_tcga_slides``: download slides returned by ``query_tcga_slides``.
- ``save_manifest``: export a query result as a GDC manifest TSV for use with ``gdc-client``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from wsistream.datasets.base import DatasetAdapter
from wsistream.types import SlideMetadata

logger = logging.getLogger(__name__)

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
# Anchored at start of filename: TCGA barcode optionally followed by .UUID or end
_TCGA_PATTERN = re.compile(
    r"^TCGA-([A-Z0-9]{2})-([A-Z0-9]{4})-(\d{2})([A-Z])-(\d{2})-([A-Z]{2}[A-Z0-9])(?:\.|$)"
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
        match = _TCGA_PATTERN.match(filename)

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


# GDC API helpers for downloading TCGA slides

_GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
_GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"

_SLIDE_TYPE_MAP = {
    "diagnostic": "Diagnostic Slide",
    "frozen": "Tissue Slide",
}


def _gdc_query(
    filters: dict,
    fields: list[str],
    size: int = 10000,
) -> list[dict]:
    """Query the GDC /files endpoint with pagination."""
    import requests

    results: list[dict] = []
    offset = 0

    while True:
        params = {
            "filters": json.dumps(filters),
            "fields": ",".join(fields),
            "format": "JSON",
            "size": size,
            "from": offset,
        }
        resp = requests.get(_GDC_FILES_ENDPOINT, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()["data"]
        hits = data["hits"]
        if not hits:
            break
        results.extend(hits)
        if len(results) >= data["pagination"]["total"]:
            break
        offset += len(hits)

    return results


def query_tcga_slides(
    cancer_types: list[str] | str | None = None,
    slide_type: str = "diagnostic",
    max_per_cancer_type: int | None = None,
    seed: int | None = 42,
) -> list[dict]:
    """Query the GDC API for TCGA whole-slide images.

    Returns a list of file records (dicts) that can be passed to
    :func:`download_tcga_slides` or :func:`save_manifest`.

    Parameters
    ----------
    cancer_types : str, list[str], or None
        TCGA project IDs to include (e.g., ``"TCGA-BRCA"`` or
        ``["TCGA-BRCA", "TCGA-LUAD"]``).  ``None`` queries all
        TCGA projects.
    slide_type : str
        ``"diagnostic"`` (FFPE / DX slides), ``"frozen"`` (TS/BS/MS
        tissue slides), or ``"all"`` for both.
    max_per_cancer_type : int or None
        When set, subsample up to this many slides per cancer type
        (stratified).  ``None`` returns all matching slides.
    seed : int or None
        Random seed for reproducible subsampling.

    Returns
    -------
    list[dict]
        Each dict contains: ``file_id``, ``filename``, ``file_size``,
        ``cancer_type``, ``md5sum``, and ``state``.

    Examples
    --------
    >>> manifest = query_tcga_slides("TCGA-BRCA", max_per_cancer_type=5)
    >>> print(len(manifest), "slides")
    5 slides
    >>> download_tcga_slides(manifest, output_dir="/data/tcga")
    """
    if slide_type not in ("diagnostic", "frozen", "all"):
        raise ValueError(
            f"slide_type must be 'diagnostic', 'frozen', or 'all', got {slide_type!r}"
        )

    # Build GDC filter
    filter_content: list[dict] = [
        {"op": "=", "content": {"field": "files.data_type", "value": "Slide Image"}},
        {"op": "=", "content": {"field": "files.data_format", "value": "SVS"}},
    ]

    if cancer_types is not None:
        if isinstance(cancer_types, str):
            cancer_types = [cancer_types]
        if len(cancer_types) == 1:
            filter_content.append(
                {"op": "=", "content": {"field": "cases.project.project_id", "value": cancer_types[0]}}
            )
        else:
            filter_content.append(
                {"op": "in", "content": {"field": "cases.project.project_id", "value": cancer_types}}
            )

    if slide_type != "all":
        strategy = _SLIDE_TYPE_MAP[slide_type]
        filter_content.append(
            {"op": "=", "content": {"field": "files.experimental_strategy", "value": strategy}}
        )

    filters = {"op": "and", "content": filter_content}
    fields = [
        "file_id", "file_name", "file_size", "md5sum", "state",
        "cases.project.project_id",
    ]

    logger.info("Querying GDC API for TCGA slides...")
    raw_hits = _gdc_query(filters, fields)

    if not raw_hits:
        logger.warning("No slides found matching the query.")
        return []

    # Flatten into clean records
    records: list[dict] = []
    for hit in raw_hits:
        cases = hit.get("cases", [{}])
        project = cases[0].get("project", {}).get("project_id", "unknown") if cases else "unknown"
        records.append({
            "file_id": hit["file_id"],
            "filename": hit["file_name"],
            "file_size": hit["file_size"],
            "cancer_type": project,
            "md5sum": hit.get("md5sum", ""),
            "state": hit.get("state", ""),
        })

    # Stratified subsampling
    if max_per_cancer_type is not None:
        import numpy as np

        rng = np.random.default_rng(seed)
        by_type: dict[str, list[dict]] = {}
        for rec in records:
            by_type.setdefault(rec["cancer_type"], []).append(rec)

        sampled: list[dict] = []
        for ct in sorted(by_type):
            group = by_type[ct]
            if len(group) <= max_per_cancer_type:
                sampled.extend(group)
            else:
                indices = rng.choice(len(group), size=max_per_cancer_type, replace=False)
                sampled.extend(group[i] for i in indices)
        records = sampled

    # Print summary
    by_type: dict[str, list[dict]] = {}
    for rec in records:
        by_type.setdefault(rec["cancer_type"], []).append(rec)
    total_bytes = sum(r["file_size"] for r in records)
    total_gb = total_bytes / (1024 ** 3)

    summary_lines = [f"Found {len(records)} slides ({total_gb:.1f} GB):"]
    for ct in sorted(by_type):
        group = by_type[ct]
        gb = sum(r["file_size"] for r in group) / (1024 ** 3)
        summary_lines.append(f"  {ct}: {len(group)} slides ({gb:.1f} GB)")
    logger.info("\n".join(summary_lines))

    return records


def _resolve_dest_path(rec: dict, output_dir: Path, organize_by: str) -> Path:
    """Compute destination path for a single file record."""
    if organize_by == "cancer_type":
        dest_dir = output_dir / rec["cancer_type"]
    else:
        dest_dir = output_dir
    return dest_dir / rec["filename"]


def _download_one(
    rec: dict, dest_path: Path, chunk_size: int,
) -> Path:
    """Download a single file from GDC. Returns the final path."""
    import requests

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{_GDC_DATA_ENDPOINT}/{rec['file_id']}"

    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    tmp_path = dest_path.with_suffix(".svs.partial")
    try:
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
        tmp_path.rename(dest_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return dest_path


def download_tcga_slides(
    manifest: list[dict],
    output_dir: str | Path,
    organize_by: str = "cancer_type",
    skip_existing: bool = True,
    max_workers: int = 4,
    chunk_size: int = 8192,
) -> list[Path]:
    """Download TCGA slides from the GDC data portal.

    Downloads run in parallel using a thread pool with a ``tqdm``
    progress bar.

    Parameters
    ----------
    manifest : list[dict]
        File records returned by :func:`query_tcga_slides`.
    output_dir : str or Path
        Root directory to save slides into.
    organize_by : str
        ``"cancer_type"`` saves as ``output_dir/TCGA-BRCA/file.svs``.
        ``"flat"`` saves all files directly in ``output_dir/``.
    skip_existing : bool
        Skip files that already exist (matched by filename and size).
    max_workers : int
        Number of parallel download threads.
    chunk_size : int
        Download chunk size in bytes.

    Returns
    -------
    list[Path]
        Paths to all downloaded (or already existing) slide files.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if organize_by not in ("cancer_type", "flat"):
        raise ValueError(f"organize_by must be 'cancer_type' or 'flat', got {organize_by!r}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Separate into already-existing and to-download
    existing: list[Path] = []
    to_download: list[tuple[dict, Path]] = []

    for rec in manifest:
        dest_path = _resolve_dest_path(rec, output_dir, organize_by)
        if skip_existing and dest_path.exists() and dest_path.stat().st_size == rec["file_size"]:
            existing.append(dest_path)
        else:
            to_download.append((rec, dest_path))

    if existing:
        logger.info("Skipping %d slides (already downloaded)", len(existing))

    if not to_download:
        logger.info("All %d slides already present in %s", len(manifest), output_dir)
        return existing

    total_bytes = sum(rec["file_size"] for rec, _ in to_download)
    logger.info(
        "Downloading %d slides (%.1f GB) with %d threads...",
        len(to_download), total_bytes / (1024 ** 3), max_workers,
    )

    from tqdm import tqdm

    pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Downloading")
    downloaded: list[Path] = list(existing)
    errors: list[tuple[str, str]] = []

    def _task(rec: dict, dest_path: Path) -> Path:
        result = _download_one(rec, dest_path, chunk_size)
        pbar.update(rec["file_size"])
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_rec = {
            pool.submit(_task, rec, dest_path): rec
            for rec, dest_path in to_download
        }
        for future in as_completed(future_to_rec):
            rec = future_to_rec[future]
            try:
                path = future.result()
                downloaded.append(path)
            except Exception as exc:
                errors.append((rec["filename"], str(exc)))
                logger.warning("Failed to download %s: %s", rec["filename"], exc)

    pbar.close()

    if errors:
        failed_names = [name for name, _ in errors]
        raise RuntimeError(
            f"{len(errors)}/{len(to_download)} downloads failed: {failed_names}. "
            f"{len(downloaded)} slides were saved successfully to {output_dir}."
        )
    logger.info("Done. %d slides in %s", len(downloaded), output_dir)
    return downloaded


def save_manifest(manifest: list[dict], path: str | Path) -> Path:
    """Save a query result as a GDC-compatible manifest TSV.

    The output file can be used with ``gdc-client download -m manifest.tsv``
    for faster parallel downloads of large datasets.

    Parameters
    ----------
    manifest : list[dict]
        File records returned by :func:`query_tcga_slides`.
    path : str or Path
        Output file path.

    Returns
    -------
    Path
        The written manifest file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("id\tfilename\tmd5\tsize\tstate\n")
        for rec in manifest:
            f.write(
                f"{rec['file_id']}\t{rec['filename']}\t"
                f"{rec['md5sum']}\t{rec['file_size']}\t{rec['state']}\n"
            )
    logger.info("Manifest written to %s (%d files)", path, len(manifest))
    return path
