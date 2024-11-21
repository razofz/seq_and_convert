import pandas as pd
from scipy.io import mmread

# standard csv format of counts
pbmc1k = mmread("data/pbmc_1k/filtered_feature_bc_matrix/matrix.mtx.gz")
feats = pd.read_table(
    "data/pbmc_1k/filtered_feature_bc_matrix/features.tsv.gz", header=None
)
barcodes = pd.read_table(
    "data/pbmc_1k/filtered_feature_bc_matrix/barcodes.tsv.gz", header=None
)
df = pd.DataFrame(
    pbmc1k.todense(), columns=barcodes[0].tolist(), index=feats[1].tolist()
)

sanity_check = df.iloc[:100, :100].apply(sum, axis=1)
assert sanity_check[(sanity_check != 0)].shape != (0,)
df.iloc[:100, :100].to_csv("test_files/pbmc1k_subset.csv")
