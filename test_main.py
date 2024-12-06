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

from main import Converter


# mimetypes_table = json.load(open("mimetypes.json", "r"))
@pytest.fixture(scope="module")
def mimetypes_table():
    formats = [
        "mtx",
        "mtx.gz",
        "csv",
        "tsv",
        "tsv.gz",
        # "json",
        "h5ad",
        "h5",
        # "zarr",
        # "gz",
        # "Rdata",
    ]
    test_files = {
        "csv": "output/pbmc3k/var.csv",
        "h5ad": "data/pbmc3k_raw.h5ad",
        "h5": "data/500_PBMC/500_PBMC_3p_LT_Chromium_X_raw_feature_bc_matrix.h5",
        "tsv.gz": "raw_feature_bc_matrix/features.tsv.gz",
        "mtx.gz": "raw_feature_bc_matrix/matrix.mtx.gz",
        "mtx": "raw_feature_bc_matrix/unzipped/matrix.mtx",
        "tsv": "raw_feature_bc_matrix/unzipped/features.tsv",
    }
    table = dict()
    for format in formats:
        file = test_files[format]
        table[format] = {
            "magic": magic.from_file(file),
            "magic_mime": magic.from_file(file, mime=True),
            "mimetype_guess_type": mimetypes.guess_type(file)[0],
            "mimetype_guess_encoding": mimetypes.guess_type(file)[1],
            "Path_suffix": Path(file).suffix,
            "splitext": os.path.splitext(file)[1],
        }
    return table


@pytest.fixture(scope="module")
def setup_test_files(tmp_path, mimetypes_table):
    from scanpy.datasets import pbmc3k

    pbmc = pbmc3k()
    subset_sc = pbmc[:100, :100]
    del pbmc
    df = subset_sc.to_df()
    del subset_sc

    sanity_check = df.apply(sum, axis=1)
    # make sure there aren't only zeroes in the df
    assert sanity_check[(sanity_check != 0)].shape != (0,)
    df.to_csv(tmp_path + "/test_files/pbmc1k_subset.csv")
    df.transpose().to_csv(tmp_path + "/test_files/pbmc1k_subset_transposed.csv")
    df.iloc[:0].to_csv(tmp_path + "/test_files/pbmc1k_empty.csv")
    tamper = df.copy()
    tamper.columns = list(tamper.columns[:10]) * 10
    tamper.to_csv(tmp_path + "/test_files/pbmc1k_subset_identical_colnames.csv")

    Path.mkdir(
        Path(tmp_path + "/test_files/pbmc1k_subset"), parents=True, exist_ok=True
    )
    mmwrite(
        target=tmp_path + "/test_files/pbmc1k_subset" / Path("matrix.mtx"),
        a=coo_matrix(df),
    )
    with open(tmp_path + "/test_files/pbmc1k_subset" / Path("features.tsv"), "w") as f:
        f.write("\n".join(df.index))
    with open(tmp_path + "/test_files/pbmc1k_subset" / Path("barcodes.tsv"), "w") as f:
        f.write("\n".join(df.columns))

    Path.mkdir(
        Path(tmp_path + "/test_files/pbmc1k_subset_gzipped"),
        parents=True,
        exist_ok=True,
    )
    for file in ["features.tsv", "barcodes.tsv", "matrix.mtx"]:
        with open(tmp_path + "/test_files/pbmc1k_subset" / Path(file), "rb") as f:
            with gzip.open(
                tmp_path + "/test_files/pbmc1k_subset_gzipped" / Path(file + ".gz"),
                "wb",
            ) as g:
                g.write(f.read())

    Path.mkdir(
        Path(tmp_path + "/test_files/false_gzipped_mtx"), parents=True, exist_ok=True
    )
    for file in ["features.tsv", "barcodes.tsv", "matrix.mtx"]:
        with gzip.open(
            tmp_path + "/test_files/false_gzipped_mtx" / Path(file + ".gz"), "wb"
        ) as g:
            g.write(json.dump(mimetypes_table))


test_files = {
    "csv": "test_files/GSE164073_raw_counts_GRCh38.p13_NCBI.csv",
    # "h5ad": "data/pbmc3k_raw.h5ad",
    # "h5": "data/500_PBMC/500_PBMC_3p_LT_Chromium_X_raw_feature_bc_matrix.h5",
    # "tsv": "test_files/10x/features.tsv",
    # "tsv.gz": "test_files/10x_gz/features.tsv.gz",
    "mtx": "test_files/pbmc1k_subset/matrix.mtx",
    # "mtx.gz": "test_files/10x_gz/matrix.mtx.gz",
}

fake_files = {
    "csv": "test_files/fake_csv.csv",
}


def test_converter_decide_filetype(mimetypes_table):
    c = Converter("test_files/pbmc1k_subset.csv", "csv", "mtx")
    assert c.filename == "test_files/pbmc1k_subset.csv"
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


def test_converter_decide_filetype_fake_files():
    c = Converter("test_files/fake_csv.csv", "csv", "mtx")
    assert c.decide_filetype() is None


def test_converter_convert_csv_to_mtx(tmp_path):
    with pytest.raises(ValueError):
        Converter(
            "test_files/pbmc1k_subset_identical_colnames.csv",
            from_format="csv",
            to_format="mtx",
            output_dir=tmp_path,
        ).convert()
    assert (
        Converter(
            "test_files/pbmc1k_subset.csv",
            from_format="csv",
            to_format="mtx",
            output_dir=tmp_path,
        ).convert()
        is True
    )
    with pytest.raises(FileExistsError):
        Converter(
            "test_files/pbmc1k_subset.csv",
            from_format="csv",
            to_format="mtx",
            output_dir=tmp_path,
        ).convert()
    assert (
        Converter(
            "test_files/pbmc1k_subset.csv",
            from_format="csv",
            to_format="mtx",
            output_dir=tmp_path,
            force=True,
        ).convert()
        is True
    )


def test_converter_convert_mtx_to_csv(tmp_path):
    # unzipped
    assert (
        Converter(
            "test_files/pbmc1k_subset",
            from_format="mtx",
            to_format="csv",
            output_dir=tmp_path,
        ).convert()
        is True
    )
    with pytest.raises(FileExistsError):
        Converter(
            "test_files/pbmc1k_subset",
            from_format="mtx",
            to_format="csv",
            output_dir=tmp_path,
        ).convert()
    assert (
        Converter(
            "test_files/pbmc1k_subset",
            from_format="mtx",
            to_format="csv",
            output_dir=tmp_path,
            force=True,
        ).convert()
        is True
    )
    # zipped
    assert (
        Converter(
            "test_files/pbmc1k_subset_gzipped",
            from_format="mtx",
            to_format="csv",
            output_dir=tmp_path,
        ).convert()
        is True
    )
    with pytest.raises(FileExistsError):
        Converter(
            "test_files/pbmc1k_subset_gzipped",
            from_format="mtx",
            to_format="csv",
            output_dir=tmp_path,
        ).convert()
    assert (
        Converter(
            "test_files/pbmc1k_subset_gzipped",
            from_format="mtx",
            to_format="csv",
            output_dir=tmp_path,
            force=True,
        ).convert()
        is True
    )
    with pytest.raises(ValueError):
        Converter(
            "test_files/false_gzipped_mtx",
            from_format="mtx",
            to_format="csv",
            output_dir=tmp_path,
            force=True,
        ).convert()


def test_converter_convert_csv_to_h5(tmp_path):
    with pytest.raises(ValueError):
        Converter(
            "test_files/pbmc1k_subset_identical_colnames.csv",
            from_format="csv",
            to_format="h5",
            output_dir=tmp_path,
        ).convert()
    assert (
        Converter(
            "test_files/pbmc1k_subset.csv",
            from_format="csv",
            to_format="h5",
            output_dir=tmp_path,
        ).convert()
        is True
    )
    with pytest.raises(FileExistsError):
        Converter(
            "test_files/pbmc1k_subset.csv",
            from_format="csv",
            to_format="h5",
            output_dir=tmp_path,
        ).convert()
    c = Converter(
        "test_files/pbmc1k_subset.csv",
        from_format="csv",
        to_format="h5",
        output_dir=tmp_path,
        force=True,
    )
    assert c.convert() is True


def test_converter_convert_mtx_to_h5(tmp_path):
    assert (
        Converter(
            "test_files/pbmc1k_subset",
            from_format="mtx",
            to_format="h5",
            output_dir=tmp_path,
        ).convert()
        is True
    )
    with pytest.raises(FileExistsError):
        Converter(
            "test_files/pbmc1k_subset",
            from_format="mtx",
            to_format="h5",
            output_dir=tmp_path,
        ).convert()
    c = Converter(
        "test_files/pbmc1k_subset",
        from_format="mtx",
        to_format="h5",
        output_dir=tmp_path,
        force=True,
    )
    assert c.convert() is True


@pytest.fixture
def csv_to_h5_converter(tmp_path):
    return Converter(
        "test_files/pbmc1k_subset.csv",
        from_format="csv",
        to_format="h5",
        output_dir=tmp_path,
        force=True,
    )


@pytest.fixture
def mtx_to_h5_converter(tmp_path):
    return Converter(
        "test_files/pbmc1k_subset",
        from_format="mtx",
        to_format="h5",
        output_dir=tmp_path,
        force=True,
    )


def test_seurat_readin(tmp_path, csv_to_h5_converter, mtx_to_h5_converter):
    assert csv_to_h5_converter.convert() is True

    if getpass.getuser() == "mambauser":
        try:
            from rpy2.robjects.packages import importr

            seurat = importr("Seurat")
            seuratobject = importr("SeuratObject")
            mat = seurat.Read10X("sandbox/pbmc1k_subset")
            sobj = seuratobject.CreateSeuratObject(mat)
        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")
    else:
        try:
            assert (
                os.system(
                    "/usr/local/bin/R -e 'library(Seurat); "
                    + f"Read10X_h5(\"{tmp_path / 'pbmc1k_subset.h5'}\")' --quiet"
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
                    + f"Read10X_h5(\"{tmp_path / 'pbmc1k_subset.h5'}\")' --quiet"
                )
                == 0
            )
        except Exception as e:
            pytest.fail(f"Unexpected exception raised: {e}")


def test_scanpy_readin(tmp_path, csv_to_h5_converter, mtx_to_h5_converter):
    assert csv_to_h5_converter.convert() is True
    try:
        sc.read_10x_h5(tmp_path / "pbmc1k_subset.h5")
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")
    assert mtx_to_h5_converter.convert() is True
    try:
        sc.read_10x_h5(tmp_path / "pbmc1k_subset.h5")
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")
