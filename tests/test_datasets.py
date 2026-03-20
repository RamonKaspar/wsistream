"""Tests for dataset adapters and TCGA download helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from wsistream.datasets import TCGAAdapter
from wsistream.datasets.tcga import (
    download_tcga_slides,
    query_tcga_slides,
    save_manifest,
)


# ── TCGAAdapter ──


class TestTCGAAdapter:
    def test_parses_standard_barcode(self):
        meta = TCGAAdapter().parse_metadata(
            "/data/TCGA-BRCA/TCGA-3L-AA1B-01Z-00-DX1.svs"
        )
        assert meta.dataset_name == "TCGA"
        assert meta.patient_id == "TCGA-3L-AA1B"
        assert meta.sample_type == "Primary Solid Tumor"
        assert meta.tissue_type == "3L"
        assert meta.cancer_type == "TCGA-BRCA"
        assert meta.extra["sample_code"] == "01"
        assert meta.extra["vial"] == "Z"
        assert meta.extra["portion"] == "00"
        assert meta.extra["slide_id"] == "DX1"
        assert meta.extra["slide_section"] == "DX"
        assert meta.extra["is_frozen"] is False
        assert meta.extra["barcode"] == "TCGA-3L-AA1B-01Z-00-DX1"

    def test_recurrent_tumor_type(self):
        meta = TCGAAdapter().parse_metadata(
            "/data/TCGA-GBM/TCGA-06-0878-02A-01-TS1.svs"
        )
        assert meta.sample_type == "Recurrent Solid Tumor"
        assert meta.extra["sample_code"] == "02"
        assert meta.extra["slide_section"] == "TS"
        assert meta.extra["is_frozen"] is True

    def test_normal_tissue_type(self):
        meta = TCGAAdapter().parse_metadata(
            "/data/TCGA-KIRC/TCGA-B0-4718-11A-01-TS1.svs"
        )
        assert meta.sample_type == "Solid Tissue Normal"
        assert meta.extra["is_frozen"] is True

    def test_bottom_section_frozen(self):
        meta = TCGAAdapter().parse_metadata(
            "/data/TCGA-LUAD/TCGA-55-6969-01A-01-BS1.svs"
        )
        assert meta.extra["slide_section"] == "BS"
        assert meta.extra["is_frozen"] is True

    def test_diagnostic_ffpe(self):
        meta = TCGAAdapter().parse_metadata(
            "/data/TCGA-BRCA/TCGA-A8-A082-01A-01-DX2.svs"
        )
        assert meta.extra["slide_section"] == "DX"
        assert meta.extra["slide_id"] == "DX2"
        assert meta.extra["is_frozen"] is False

    def test_handles_bad_filename(self):
        meta = TCGAAdapter().parse_metadata("/data/random_file.svs")
        assert meta.dataset_name == "TCGA"
        assert meta.patient_id is None
        assert "parse_error" in meta.extra

    def test_cancer_type_override(self):
        adapter = TCGAAdapter(cancer_type="LUAD")
        meta = adapter.parse_metadata("/whatever/TCGA-3L-AA1B-01Z-00-DX1.svs")
        assert meta.cancer_type == "LUAD"

    def test_cancer_type_from_non_tcga_parent(self):
        meta = TCGAAdapter().parse_metadata(
            "/data/slides/TCGA-3L-AA1B-01Z-00-DX1.svs"
        )
        # Parent dir "slides" doesn't start with "TCGA-" so cancer_type is None
        assert meta.cancer_type is None

    def test_rejects_garbage_prefix(self):
        """Barcode must start at the beginning of the filename."""
        meta = TCGAAdapter().parse_metadata(
            "/data/TCGA-BRCA/garbage-TCGA-3L-AA1B-01Z-00-DX1.svs"
        )
        assert meta.patient_id is None
        assert "parse_error" in meta.extra

    def test_parses_barcode_with_uuid_suffix(self):
        """Full GDC filename with UUID after the barcode."""
        meta = TCGAAdapter().parse_metadata(
            "/data/TCGA-BRCA/TCGA-3L-AA1B-01Z-00-DX1.8923A151-A690-40B7-9E5A-FCBEDFC2394F.svs"
        )
        assert meta.patient_id == "TCGA-3L-AA1B"
        assert meta.extra["barcode"] == "TCGA-3L-AA1B-01Z-00-DX1"


# ── GDC download helpers (mocked) ──


def _make_gdc_hits(n: int, project: str = "TCGA-BRCA") -> list[dict]:
    """Create fake GDC API response hits."""
    return [
        {
            "file_id": f"uuid-{i}",
            "file_name": f"TCGA-XX-{i:04d}-01Z-00-DX1.svs",
            "file_size": 1000 * (i + 1),
            "md5sum": f"md5-{i}",
            "state": "released",
            "cases": [{"project": {"project_id": project}}],
        }
        for i in range(n)
    ]


class TestQueryTcgaSlides:
    @patch("wsistream.datasets.tcga._gdc_query")
    def test_returns_records(self, mock_query):
        mock_query.return_value = _make_gdc_hits(5)
        result = query_tcga_slides(cancer_types="TCGA-BRCA")
        assert len(result) == 5
        assert all(r["cancer_type"] == "TCGA-BRCA" for r in result)
        assert all("file_id" in r for r in result)

    @patch("wsistream.datasets.tcga._gdc_query")
    def test_stratified_subsampling(self, mock_query):
        hits = _make_gdc_hits(10, "TCGA-BRCA") + _make_gdc_hits(10, "TCGA-LUAD")
        mock_query.return_value = hits
        result = query_tcga_slides(
            cancer_types=["TCGA-BRCA", "TCGA-LUAD"],
            max_per_cancer_type=3,
            seed=42,
        )
        assert len(result) == 6
        brca = [r for r in result if r["cancer_type"] == "TCGA-BRCA"]
        luad = [r for r in result if r["cancer_type"] == "TCGA-LUAD"]
        assert len(brca) == 3
        assert len(luad) == 3

    @patch("wsistream.datasets.tcga._gdc_query")
    def test_empty_result(self, mock_query):
        mock_query.return_value = []
        result = query_tcga_slides(cancer_types="TCGA-BRCA")
        assert result == []

    def test_invalid_slide_type_raises(self):
        with pytest.raises(ValueError, match="slide_type"):
            query_tcga_slides(slide_type="bogus")

    @patch("wsistream.datasets.tcga._gdc_query")
    def test_seeded_is_reproducible(self, mock_query):
        mock_query.return_value = _make_gdc_hits(20)
        r1 = query_tcga_slides(max_per_cancer_type=5, seed=99)
        mock_query.return_value = _make_gdc_hits(20)
        r2 = query_tcga_slides(max_per_cancer_type=5, seed=99)
        assert [r["file_id"] for r in r1] == [r["file_id"] for r in r2]


class TestDownloadTcgaSlides:
    def _make_manifest(self, n: int = 3) -> list[dict]:
        return [
            {
                "file_id": f"uuid-{i}",
                "filename": f"slide_{i}.svs",
                "file_size": 100,
                "cancer_type": "TCGA-BRCA",
                "md5sum": f"md5-{i}",
                "state": "released",
            }
            for i in range(n)
        ]

    @patch("wsistream.datasets.tcga._download_one")
    def test_downloads_files(self, mock_dl, tmp_path):
        manifest = self._make_manifest(3)
        mock_dl.side_effect = lambda rec, dest, cs: dest

        paths = download_tcga_slides(manifest, output_dir=tmp_path)
        assert len(paths) == 3
        assert mock_dl.call_count == 3

    @patch("wsistream.datasets.tcga._download_one")
    def test_skips_existing(self, mock_dl, tmp_path):
        manifest = self._make_manifest(2)
        # Pre-create one file with matching size
        dest = tmp_path / "TCGA-BRCA" / "slide_0.svs"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(b"x" * 100)

        mock_dl.side_effect = lambda rec, dest, cs: dest
        paths = download_tcga_slides(manifest, output_dir=tmp_path)
        assert len(paths) == 2
        assert mock_dl.call_count == 1  # only slide_1 downloaded

    @patch("wsistream.datasets.tcga._download_one")
    def test_raises_on_partial_failure(self, mock_dl, tmp_path):
        manifest = self._make_manifest(3)

        def fail_on_second(rec, dest, cs):
            if "uuid-1" in rec["file_id"]:
                raise ConnectionError("network down")
            return dest

        mock_dl.side_effect = fail_on_second
        with pytest.raises(RuntimeError, match="1/3 downloads failed"):
            download_tcga_slides(manifest, output_dir=tmp_path)

    def test_invalid_organize_by_raises(self, tmp_path):
        with pytest.raises(ValueError, match="organize_by"):
            download_tcga_slides([], output_dir=tmp_path, organize_by="bogus")

    @patch("wsistream.datasets.tcga._download_one")
    def test_flat_organization(self, mock_dl, tmp_path):
        manifest = self._make_manifest(2)
        mock_dl.side_effect = lambda rec, dest, cs: dest

        paths = download_tcga_slides(manifest, output_dir=tmp_path, organize_by="flat")
        assert all(p.parent == tmp_path for p in paths)


class TestSaveManifest:
    def test_writes_tsv(self, tmp_path):
        manifest = [
            {
                "file_id": "uuid-1",
                "filename": "slide.svs",
                "file_size": 1000,
                "cancer_type": "TCGA-BRCA",
                "md5sum": "abc123",
                "state": "released",
            }
        ]
        path = save_manifest(manifest, tmp_path / "manifest.tsv")
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2  # header + 1 record
        assert lines[0] == "id\tfilename\tmd5\tsize\tstate"
        assert "uuid-1" in lines[1]
        assert "slide.svs" in lines[1]

    def test_empty_manifest(self, tmp_path):
        path = save_manifest([], tmp_path / "empty.tsv")
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1  # header only
