import pytest
from pathlib import Path
from main import Converter
import json

mimetypes_table = json.load(open("mimetypes.json", "r"))

test_files = {
    "csv": "test_files/GSE164073_raw_counts_GRCh38.p13_NCBI.csv",
    "h5ad": "data/pbmc3k_raw.h5ad",
    "h5": "data/500_PBMC/500_PBMC_3p_LT_Chromium_X_raw_feature_bc_matrix.h5",
    "tsv": "test_files/10x/features.tsv",
    "tsv.gz": "test_files/10x_gz/features.tsv.gz",
    "mtx": "test_files/10x/matrix.mtx",
    "mtx.gz": "test_files/10x_gz/matrix.mtx.gz",
}

fake_files = {
    "csv": "test_files/fake_csv.csv",
}


def test_converter_decide_filetype():
    c = Converter("data/pbmc3k_raw.h5ad", "h5ad", "csv")
    assert c.filename == "data/pbmc3k_raw.h5ad"
    assert c.from_format == "h5ad"
    assert c.to_format == "csv"
    assert c.output_dir == "."
    assert c.lookup_table == mimetypes_table
    for key, value in test_files.items():
        c = Converter(value, key, "csv")
        assert c.filename == value
        assert c.from_format == key
        assert c.to_format == "csv"
        assert c.decide_filetype() == key


def test_converter_decide_filetype_fake_files():
    c = Converter("test_files/fake_csv.csv", "csv", "csv")
    assert c.decide_filetype() is None