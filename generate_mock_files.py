import pandas as pd
from scipy.io import mmread

# this code can(/should?) probably be included in the test files,
# but writing it here now to get the logic out, can alway refactor later

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

subset = df.iloc[:100, :100]
sanity_check = subset.apply(sum, axis=1)
# make sure there aren't only zeroes in the subset
assert sanity_check[(sanity_check != 0)].shape != (0,)
subset.to_csv("test_files/pbmc1k_subset.csv")
subset.transpose().to_csv("test_files/pbmc1k_subset_transposed.csv")
df.iloc[:0].to_csv("test_files/pbmc1k_empty.csv")
tamper = subset.copy()
tamper.columns = list(tamper.columns[:10])*10
tamper.to_csv("test_files/pbmc1k_subset_identical_colnames.csv")
