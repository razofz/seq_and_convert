import gzip
import os
import shutil
from pathlib import Path

import h5py
import pandas as pd
from scanpy.datasets import pbmc3k
from scipy.io import mmwrite
from scipy.sparse import coo_matrix, csc_matrix
from yaml import safe_load

out_dir = Path(safe_load(open("conf.yaml"))["datadir"])

if not Path(out_dir).exists():
    Path(out_dir).mkdir()

pbmc = pbmc3k()
subset_sc = pbmc[:100, :100]
del pbmc
df = subset_sc.to_df()

sanity_check = df.apply(sum, axis=1)
# make sure there aren't only zeroes in the df
assert sanity_check[(sanity_check != 0)].shape != (0,)
df.to_csv(Path(out_dir / "pbmc3k_subset.csv"))
df.transpose().to_csv(Path(out_dir / "pbmc3k_subset_transposed.csv"))
df.iloc[:0].to_csv(Path(out_dir / "pbmc3k_empty.csv"))
tamper = df.copy()
tamper.columns = list(tamper.columns[:10]) * 10
tamper.to_csv(Path(out_dir / "pbmc3k_subset_identical_colnames.csv"))

Path.mkdir(Path(out_dir / "pbmc3k_subset"), parents=True, exist_ok=True)
mtx_path = Path(out_dir / "pbmc3k_subset" / "matrix.mtx")
mmwrite(
    target=mtx_path,
    a=coo_matrix(df),
)
with open(mtx_path, "rb") as f_in:
    with gzip.open(Path(str(mtx_path) + ".gz"), "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
os.remove(mtx_path)

features = pd.DataFrame(
    {
        "gene_ids": subset_sc.var.gene_ids.tolist(),
        "gene_names": subset_sc.var.index.tolist(),
    }
)
features.to_csv(
    Path(out_dir / "pbmc3k_subset" / "features.tsv.gz"),
    header=None,
    index=None,
    sep="\t",
)
pd.DataFrame(df.index).to_csv(
    Path(out_dir / "pbmc3k_subset" / "barcodes.tsv.gz"), header=None, index=None, sep="\t"
)

# Path.mkdir(
#     Path(out_dir / "pbmc3k_subset_gzipped"),
#     parents=True,
#     exist_ok=True,
# )
# for file in ["features.tsv", "barcodes.tsv", "matrix.mtx"]:
#     with open(Path(out_dir / "pbmc3k_subset" / file), "rb") as f:
#         with gzip.open(
#             Path(out_dir / "pbmc3k_subset_gzipped" / (file + ".gz")),
#             "wb",
#         ) as g:
#             g.write(f.read())

Path(out_dir / "fake_csv.csv").touch()

Path.mkdir(Path(out_dir / "false_gzipped_mtx"), parents=True, exist_ok=True)
for file in ["features.tsv", "barcodes.tsv", "matrix.mtx"]:
    with open("mimetypes.json", "rb") as f:
        with gzip.open(Path(out_dir / "false_gzipped_mtx" / (file + ".gz")), "wb") as g:
            g.write(f.read())


def create_h5_dataset(
    h5_file,
    name,
    data,
    dtype,
    maxshape={
        None,
    },
    compression="gzip",
    compression_opts=1,
):
    h5_file.create_dataset(
        name,
        data=data,
        dtype=dtype,
        maxshape=maxshape,
        compression=compression,
        compression_opts=compression_opts,
    )


h5_path = Path(out_dir / "pbmc3k_subset.h5")
h5_file = h5py.File(h5_path, "w")
matrix = csc_matrix(df)
h5_file.create_group("matrix")
h5_file.create_group("matrix/features")
dset_dict = {
    "matrix/barcodes": {
        "data": df.index.tolist(),
        "dtype": "S18",
    },
    "matrix/indices": {
        "data": matrix.indices,
        "dtype": "int64",
    },
    "matrix/indptr": {
        "data": matrix.indptr,
        "dtype": "int64",
    },
    "matrix/shape": {
        "data": matrix.shape,
        "dtype": "int32",
    },
    "matrix/data": {
        "data": matrix.data,
        "dtype": "int32",
    },
    "matrix/features/feature_type": {
        "data": ["Gene Expression"] * len(df.columns),
        "dtype": "S15",
    },
    "matrix/features/id": {
        "data": subset_sc.var.gene_ids.tolist(),
        "dtype": "S15",
    },
    "matrix/features/name": {
        "data": subset_sc.var.index.tolist(),
        "dtype": "S17",
    },
    "matrix/features/genome": {
        "data": [""] * len(df.columns),
        "dtype": "S16",
    },
    "matrix/features/_all_tag_keys": {
        "data": ["genome"],
        "dtype": "S6",
    },
}
for key, value in dset_dict.items():
    create_h5_dataset(
        h5_file=h5_file,
        name=key,
        data=dset_dict[key]["data"],
        dtype=dset_dict[key]["dtype"],
    )
h5_file.close()
