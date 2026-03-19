"""Tests for dataset adapters."""

from wsistream.datasets import TCGAAdapter


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
