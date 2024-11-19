import mimetypes
from pathlib import Path
import magic
from os import path
import json


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

# for format in formats:
#     file = test_files[format]
#     print("#########")
#     print(file)
#     print(f"{magic.from_file(file)=}")
#     print(f"{magic.from_file(file, mime=True)=}")
#     print(f"{mimetypes.guess_type(file)=}")
#     print(f"{Path(file).suffix=}")
#     print(f"{path.splitext(file)=}")

table = dict()
for format in formats:
    file = test_files[format]
    table[format] = {
        "magic": magic.from_file(file),
        "magic_mime": magic.from_file(file, mime=True),
        "mimetype_guess_type": mimetypes.guess_type(file)[0],
        "mimetype_guess_encoding": mimetypes.guess_type(file)[1],
        "Path_suffix": Path(file).suffix,
        "splitext": path.splitext(file)[1],
    }

print(table)
json.dump(table, open("mimetypes.json", "w"), indent=4)
