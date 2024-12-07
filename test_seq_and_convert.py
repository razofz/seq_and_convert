import getpass
import gzip
import mimetypes

import json
import os
from pathlib import Path

import magic
import pandas as pd
import pytest
import scanpy as sc
from scipy.io import mmread, mmwrite
from scipy.sparse import coo_matrix

from seq_and_convert import Converter


@pytest.fixture(scope="module")
def mimetypes_table():
    mimetypes_table = json.load(open("mimetypes.json", "r"))
    return mimetypes_table


def test_converter_decide_filetype(mimetypes_table, datadir):
    test_files = {
        "csv": datadir / "pbmc3k_subset.csv",
        # "h5ad": "data/pbmc3k_raw.h5ad",
        # "h5": "data/500_PBMC/500_PBMC_3p_LT_Chromium_X_raw_feature_bc_matrix.h5",
        # "tsv": "test_files/10x/features.tsv",
        # "tsv.gz": "test_files/10x_gz/features.tsv.gz",
        # "mtx": datadir / "pbmc3k_subset/matrix.mtx",
        "mtx.gz": datadir / "pbmc3k_subset/matrix.mtx.gz",
        # "mtx.gz": "test_files/10x_gz/matrix.mtx.gz",
    }

    c = Converter((datadir / "pbmc3k_subset.csv"), "csv", "mtx")
    if c.filename is Path:
        assert c.filename == datadir / "pbmc3k_subset.csv"
    elif c.filename is str:
        assert c.filename == f"{datadir}/pbmc3k_subset.csv"
    assert c.from_format == "csv"
    assert c.to_format == "mtx"
    assert c.output_dir == "."
    assert c.lookup_table == mimetypes_table
    for key, value in test_files.items():
        if key != "csv":
            c = Converter(value, key, "csv")
            assert c.filename == value
            assert c.from_format == key
            assert c.to_format == "csv"
            assert c.decide_filetype() == key


def test_converter_decide_filetype_fake_files(datadir):
    c = Converter(datadir / "fake_csv.csv", "csv", "mtx")
    assert c.decide_filetype() is None


def test_converter_convert_csv_to_mtx(tmp_path, datadir):
    with pytest.raises(ValueError):
        Converter(
            datadir / "pbmc3k_subset_identical_colnames.csv",
            from_format="csv",
            to_format="mtx",
            output_dir=tmp_path,
        ).convert()
    assert (
        Converter(
            datadir / "pbmc3k_subset.csv",
            from_format="csv",
            to_format="mtx",
            output_dir=tmp_path,
        ).convert()
        is True
    )
    with pytest.raises(FileExistsError):
        Converter(
            datadir / "pbmc3k_subset.csv",
            from_format="csv",
            to_format="mtx",
            output_dir=tmp_path,
        ).convert()
    assert (
        Converter(
            datadir / "pbmc3k_subset.csv",
            from_format="csv",
            to_format="mtx",
            output_dir=tmp_path,
            force=True,
        ).convert()
        is True
    )


def test_converter_convert_mtx_to_csv(tmp_path, datadir):
    assert (
        Converter(
            datadir / "pbmc3k_subset",
            from_format="mtx",
            to_format="csv",
            output_dir=tmp_path,
        ).convert()
        is True
    )
    with pytest.raises(FileExistsError):
        Converter(
            datadir / "pbmc3k_subset",
            from_format="mtx",
            to_format="csv",
            output_dir=tmp_path,
        ).convert()
    assert (
        Converter(
            datadir / "pbmc3k_subset",
            from_format="mtx",
            to_format="csv",
            output_dir=tmp_path,
            force=True,
        ).convert()
        is True
    )


def test_converter_convert_csv_to_h5(tmp_path, datadir):
    with pytest.raises(ValueError):
        Converter(
            datadir / "pbmc3k_subset_identical_colnames.csv",
            from_format="csv",
            to_format="h5",
            output_dir=tmp_path,
        ).convert()
    assert (
        Converter(
            datadir / "pbmc3k_subset.csv",
            from_format="csv",
            to_format="h5",
            output_dir=tmp_path,
        ).convert()
        is True
    )
    with pytest.raises(FileExistsError):
        Converter(
            datadir / "pbmc3k_subset.csv",
            from_format="csv",
            to_format="h5",
            output_dir=tmp_path,
        ).convert()
    c = Converter(
        datadir / "pbmc3k_subset.csv",
        from_format="csv",
        to_format="h5",
        output_dir=tmp_path,
        force=True,
    )
    assert c.convert() is True


def test_converter_convert_mtx_to_h5(tmp_path, datadir):
    assert (
        Converter(
            datadir / "pbmc3k_subset",
            from_format="mtx",
            to_format="h5",
            output_dir=tmp_path,
        ).convert()
        is True
    )
    with pytest.raises(FileExistsError):
        Converter(
            datadir / "pbmc3k_subset",
            from_format="mtx",
            to_format="h5",
            output_dir=tmp_path,
        ).convert()
    c = Converter(
        datadir / "pbmc3k_subset",
        from_format="mtx",
        to_format="h5",
        output_dir=tmp_path,
        force=True,
    )
    assert c.convert() is True


@pytest.fixture
def csv_to_h5_converter(tmp_path, datadir):
    return Converter(
        datadir / "pbmc3k_subset.csv",
        from_format="csv",
        to_format="h5",
        output_dir=tmp_path,
        force=True,
    )


@pytest.fixture
def mtx_to_h5_converter(tmp_path, datadir):
    return Converter(
        datadir / "pbmc3k_subset",
        from_format="mtx",
        to_format="h5",
        output_dir=tmp_path,
        force=True,
    )


@pytest.fixture
def h5_to_mtx_converter(tmp_path, datadir):
    return Converter(
        datadir / "pbmc3k_subset.h5",
        from_format="h5",
        to_format="mtx",
        output_dir=tmp_path,
        force=True,
    )


def test_seurat_readin(
    tmp_path, csv_to_h5_converter, mtx_to_h5_converter, h5_to_mtx_converter, datadir
):
    assert csv_to_h5_converter.convert() is True
    assert h5_to_mtx_converter.convert() is True

    if getpass.getuser() == "mambauser":
        try:
            from rpy2.robjects.packages import importr

            seurat = importr("Seurat")
            seuratobject = importr("SeuratObject")
            mat = seurat.Read10X(str(datadir / "pbmc3k_subset"))
            sobj = seuratobject.CreateSeuratObject(mat)
        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")
    else:
        try:
            assert (
                os.system(
                    "/usr/local/bin/R -e 'library(Seurat); "
                    + f"Read10X_h5(\"{tmp_path / 'pbmc3k_subset.h5'}\")' --quiet"
                )
                == 0
            )
        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")
        assert mtx_to_h5_converter.convert() is True
        try:
            assert (
                os.system(
                    "/usr/local/bin/R -e 'library(Seurat); "
                    + f"Read10X_h5(\"{tmp_path / 'pbmc3k_subset.h5'}\")' --quiet"
                )
                == 0
            )
        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")


def test_scanpy_readin(tmp_path, csv_to_h5_converter, mtx_to_h5_converter):
    assert csv_to_h5_converter.convert() is True
    try:
        sc.read_10x_h5(tmp_path / "pbmc3k_subset.h5")
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")
    assert mtx_to_h5_converter.convert() is True
    try:
        sc.read_10x_h5(tmp_path / "pbmc3k_subset.h5")
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")
