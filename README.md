# seq and convert

Prototype of a CLI program to convert between different single cell RNA-seq-related formats.
Both between "simply a matrix" formats like mtx (MM), csv etc, but the vision is also between 
file types for different scRNA-seq toolkits: Seurat (SeuratH5, Rdata), scanpy (h5ad), loom, zarr etc.

Vision is for usage in the style of:

```bash
╭─> ~/code/seq_and_convert
╰─ saq data/GSE164073_raw_counts_GRCh38.p13_NCBI.tsv -d output --from tsv --to mtx
```

This could then be used directly or utilised by toolkits, pipelines etc.

_(The working title for the tool is loosely based, I guess, on Seek and Destroy by Metallica. No guarantees for keeping it.)_
