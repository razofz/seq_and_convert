import json
import mimetypes
from pathlib import Path
import re

import h5py
import shutil
import gzip
import magic
import pandas as pd
import anndata as ad
import typer
import os
from scipy.io import mmread, mmwrite
from scipy.sparse import coo_matrix, csc_matrix
from typing_extensions import Annotated

app = typer.Typer()


class Converter:
    """
    Converter class for converting between different file formats.

        filename (str | Path): The input file or directory path.
        from_format (str): The format of the input file.
        to_format (str): The desired format of the output file.
        output_dir (str): The directory where the output file will be saved.
        force (bool): Whether to overwrite existing files.
        lookup_table (dict): A lookup table for file type properties.
        filetype_checks (list): A list of file type checks to perform.
        matrix (scipy.sparse.coo_matrix): The matrix data.
        mtx (str): The path to the matrix file.
        barcodes (str): The path to the barcodes file.
        features (str): The path to the features file.
        genome (str): The genome identifier.

    Methods:
        __init__(filename, from_format, to_format, genome=None, force=False, output_dir="."):
            Initializes the Converter instance.
        __str__():
            Returns a string representation of the Converter instance.
        decide_filetype():
            Determines the file type of the given file or files in a directory.
        convert():
            Converts data from one format to another.
        read_in_mtx():
        read_in_csv():
        read_in_h5():
        create_output_dir(directory=None):
        csv_to_h5():
            Converts a CSV file to an HDF5 file.
        extract_features(features, matrix):
            Extracts features from the given features DataFrame and matrix.
        mtx_to_h5():
            Converts a matrix market (MTX) file to an HDF5 file.
        mtx_to_anndata():
            Converts a matrix market (MTX) file to an AnnData (H5AD) file.
        mtx_to_csv():
            Converts a matrix market (MTX) file to a CSV file.
        csv_to_mtx():
            Converts a CSV file to a matrix market (MTX) file.
        h5_to_mtx():
            Converts an HDF5 file to a matrix market (MTX) file.
    """

    def __init__(
        self,
        filename: str | Path,
        from_format: str,
        to_format: str,
        genome: str = None,
        force: bool = False,
        output_dir: str = ".",
    ):
        self.filename = filename
        self.from_format = from_format
        self.to_format = to_format
        self.output_dir = output_dir
        self.force = force
        self.lookup_table = json.load(open("mimetypes.json", "r"))
        self.filetype_checks = [
            "magic",
            "magic_mime",
            "mimetype_guess_type",
            "mimetype_guess_encoding",
            "Path_suffix",
        ]
        self.matrix = None
        self.mtx = None
        self.barcodes = None
        self.features = None
        self.genome = genome
        if not self.filename:
            raise ValueError("No filename provided")
        if not Path(self.filename).exists():
            raise FileNotFoundError(f"File {self.filename} not found")
        try:
            ft = self.decide_filetype()
        except Exception as e:
            print(f"Error: {e}")
            raise e
        if ft is not None:
            print(f"File {self.filename} seems to be a {ft} file")

    def __str__(self):
        return f"{self.filename}"

    def decide_filetype(self):
        """
        Determines the file type of the given file or files in a directory by comparing their properties against a lookup table.

        If the input is a directory, it expects exactly three files in the directory, corresponding to a '10X style' format
        (matrix.mtx, features.tsv, barcodes.tsv), which can also be compressed (e.g., matrix.mtx.gz). It identifies the file
        types and assigns them to the appropriate attributes (mtx, features, barcodes).

        If the input is a single file, it determines the file type based on its properties.

        Returns:
            str or None: The determined file type key from the lookup table, or None if no match is found.

        Raises:
            FileNotFoundError: If the input is a directory and no files are found.
            ValueError: If the input is a directory and the number of files is not exactly three.
        """

        def control_filetype(filename):
            """
            Determines the file type of the given file by comparing its properties against a lookup table.

            Args:
                filename (str): The path to the file whose type is to be determined.

            Returns:
                str or None: The key from the lookup table that matches the file's properties, or None if no match is found.

            The function checks the following properties of the file against the lookup table:
                - MIME type from the file's magic number
                - Guessed MIME encoding type
                - Guessed MIME type
                - File extension (suffix)
                - Magic number (with special handling for ".gz" files)
            """
            for key in self.lookup_table:
                eligible = True
                if not self.lookup_table[key]["magic_mime"] == magic.from_file(
                    filename, mime=True
                ):
                    eligible = False
                if (
                    not self.lookup_table[key]["mimetype_guess_encoding"]
                    == mimetypes.guess_type(filename)[1]
                ):
                    eligible = False
                if (
                    not self.lookup_table[key]["mimetype_guess_type"]
                    == mimetypes.guess_type(filename)[0]
                ):
                    eligible = False
                if not self.lookup_table[key]["Path_suffix"] == Path(filename).suffix:
                    eligible = False
                if self.lookup_table[key]["splitext"] == ".gz":
                    if (
                        not self.lookup_table[key]["magic"].split(",")[0]
                        == magic.from_file(filename).split(",")[0]
                    ):
                        eligible = False
                else:
                    if not self.lookup_table[key]["magic"] == magic.from_file(filename):
                        eligible = False
                if eligible:
                    return key
            return None

        if Path(self.filename).is_dir():
            dir_files = [f for f in Path(self.filename).iterdir() if f.is_file()]
            if len(dir_files) == 0:
                raise FileNotFoundError(f"No files found in directory {self.filename}")
            elif len(dir_files) != 3:
                raise ValueError(
                    f"Wrong number of files in directory {self.filename}, "
                    + "expected 3 files '10X style' "
                    + "(matrix.mtx, features.tsv, barcodes.tsv) "
                    + "(could also be compressed, e.g. matrix.mtx.gz)"
                )
            filetypes = [control_filetype(f) for f in dir_files]
            print(filetypes)
            mtx = None
            features = None
            barcodes = None
            if sum([f.endswith(".gz") for f in filetypes]) > 0:
                for i in range(len(filetypes)):
                    guessed_type = mimetypes.guess_type(dir_files[i])[0]
                    if guessed_type is None:
                        mtx = dir_files[i]
                    elif guessed_type == "text/tab-separated-values":
                        if (
                            Path(Path(dir_files[i]).stem).stem == "features"
                            or Path(Path(dir_files[i]).stem).stem == "genes"
                        ):
                            features = dir_files[i]
                        elif Path(Path(dir_files[i]).stem).stem == "barcodes":
                            barcodes = dir_files[i]
            else:
                for i in range(len(filetypes)):
                    if filetypes[i] == "mtx":
                        mtx = dir_files[i]
                    elif filetypes[i] == "tsv":
                        if (
                            Path(dir_files[i]).stem == "features"
                            or Path(dir_files[i]).stem == "genes"
                        ):
                            features = dir_files[i]
                        elif Path(dir_files[i]).stem == "barcodes":
                            barcodes = dir_files[i]
            self.mtx = mtx
            self.barcodes = barcodes
            self.features = features
            return "mtx"
        else:
            key = control_filetype(self.filename)
            if key is not None:
                return key
        return None

    def convert(self):
        """
        The function responsible for converting data from one format to another.

        Supported conversions:
        - CSV to MTX
        - MTX to CSV
        - CSV to H5
        - MTX to H5
        - H5 to MTX
        - MTX to H5AD

        Raises:
            NotImplementedError: If the conversion from `self.from_format` to `self.to_format` is not implemented.

        Returns:
            The result of the conversion method corresponding to the specified formats.
        """
        if self.from_format == "csv" and self.to_format == "mtx":
            return self.csv_to_mtx()
        elif self.from_format == "mtx" and self.to_format == "csv":
            return self.mtx_to_csv()
        elif self.from_format == "csv" and self.to_format == "h5":
            return self.csv_to_h5()
        elif self.from_format == "mtx" and self.to_format == "h5":
            return self.mtx_to_h5()
        elif self.from_format == "h5" and self.to_format == "mtx":
            return self.h5_to_mtx()
        elif self.from_format == "mtx" and self.to_format == "h5ad":
            return self.mtx_to_anndata()
        else:
            raise NotImplementedError(
                f"Conversion from {self.from_format} to {self.to_format} "
                + "is not yet implemented"
            )

    def read_in_mtx(self):
        """
        Reads in a matrix market (MTX) file along with associated barcodes and features files.

        This function reads the MTX file specified by `self.mtx`, the barcodes file specified by `self.barcodes`,
        and the features file specified by `self.features`. It performs several checks to ensure the integrity
        and consistency of the data, such as verifying that the number of barcodes matches the number of columns
        in the matrix and that the number of features matches the number of rows in the matrix.

        Returns:
            tuple: A tuple containing the following elements:
                - features (pd.DataFrame): The features data frame.
                - barcodes (pd.DataFrame): The barcodes data frame.
                - matrix (scipy.sparse.coo_matrix): The matrix market file read into a sparse matrix.
                - mtx_feats (pd.Series or None): A series containing the selected feature identifiers, or None if not found.

        Raises:
            AssertionError: If the number of barcodes does not match the number of columns in the matrix,
                            or if the number of features does not match the number of rows in the matrix.
        """
        matrix = mmread(self.mtx)
        barcodes = pd.read_table(self.barcodes, header=None)
        assert barcodes.shape[0] == matrix.shape[1]
        assert barcodes.shape[1] == 1
        features = pd.read_table(self.features, header=None)
        # lots of things to potentially check for features.
        # could be a single column, could be lots of columns,
        # could have the gene_id as the first column and gene_name as second,
        # the other way around, or in another order entirely.
        # To solve this we could have a sample set of gene ids and names,
        # and check if they exist in one of the columns.
        # Feels a bit clunky, but a better heuristic than nothing, I guess.
        # Let's do some simplification for now, let's assume the id and name
        # are in one of the first two columns and select one of them.
        # edit: if a 10x file, should be id, name, feat_type:
        # https://www.10xgenomics.com/support/software/cell-ranger/latest/analysis/outputs/cr-outputs-mex-matrices
        assert features.shape[0] == matrix.shape[0]
        mtx_feats = None
        if features.shape[1] == 1:
            mtx_feats = features[0]
        else:
            # using a very bad heuristic for now..
            if features.iloc[0, 0].startswith("ENS"):
                mtx_feats = features[1]
            elif features.iloc[0, 1].startswith("ENS"):
                mtx_feats = features[0]
        return features, barcodes, matrix, mtx_feats

    def read_in_csv(self):
        """
        Reads a CSV file into a pandas DataFrame and performs necessary checks and adjustments.

        This method attempts to read a CSV file specified by `self.filename` into `self.matrix`.
        It handles potential issues such as:
        - The entire DataFrame being read as object type, indicating a possible header row issue.
        - The first column being read as row names/index, indicating a possible index column issue.
        - Non-unique column names, which could indicate issues with cell barcodes.

        Raises:
            ValueError: If non-unique column names are detected, indicating potential issues with cell barcodes.
        """
        try:
            self.matrix = pd.read_csv(self.filename)
        except Exception as e:
            print(f"Error: {e}")
        # pot. unnecessary all if case, let's see
        if (self.matrix.dtypes == object).any():
            if (self.matrix.dtypes == object).all():
                # probably a header row read in as first row, redo
                self.matrix = pd.read_csv(self.filename, header=0)
            elif pd.api.types.is_dtype_equal(
                self.matrix.dtypes.iloc[0], pd.api.types.pandas_dtype("object")
            ):
                # probably rownames/index read in as first column, redo
                self.matrix = pd.read_csv(self.filename, index_col=0)
        for i in range(len(self.matrix.columns)):
            for j in range(len(self.matrix.columns)):
                if i != j:
                    if self.matrix.columns[i] == self.matrix.columns[j].split(".")[0]:
                        raise ValueError(
                            "Column names are not unique, please check cell barcodes, e.g.\n"
                            + f"Column {i}: {self.matrix.columns[i]}\n"
                            + f"Column {j}: {self.matrix.columns[j]}"
                        )
                        # just the first case of identical barcodes for now

    def read_in_h5(self):
        """
        Reads an HDF5 file specified by the instance's filename attribute.

        This method attempts to open the HDF5 file in read mode. If the file cannot be opened,
        an error message is printed. The method then checks for the presence of specific keys
        within the HDF5 file, assuming it follows the 10x Genomics format. If all keys are
        present, the HDF5 file is assigned to the instance's matrix attribute.

        Raises:
            Exception: If there is an error opening the HDF5 file.

        Attributes:
            filename (str): The path to the HDF5 file.
            matrix (h5py.File): The opened HDF5 file if all required keys are present.
        """
        try:
            h5_file = h5py.File(self.filename, "r")
        except Exception as e:
            print(f"Error: {e}")
        # for now let's treat it as a 10x h5 file,
        # and later we can extend the definition
        print(h5_file)
        keys = [
            "matrix",
            "matrix/barcodes",
            "matrix/data",
            "matrix/indices",
            "matrix/indptr",
            "matrix/shape",
            "matrix/features",
            "matrix/features/_all_tag_keys",
            "matrix/features/feature_type",
            "matrix/features/genome",
            "matrix/features/id",
            "matrix/features/name",
        ]
        for key in keys:
            assert key in h5_file
        self.matrix = h5_file

    def create_output_dir(self, directory: str = None):
        """
        Creates the output directory if it does not exist.

        Args:
            directory (str, optional): The path to the directory to create.
                                       If None, uses the instance's output_dir attribute.

        Raises:
            FileExistsError: If the directory already exists and self.force is not set to True.
        """
        if directory is None:
            directory = self.output_dir
        if not Path(directory).exists():
            try:
                Path.mkdir(directory, parents=True, exist_ok=self.force)
            except FileExistsError as e:
                print(
                    f"Directory {directory} already exists."
                    + " Select a different output directory or use --force/-f."
                )
                raise e

    def csv_to_h5(self):
        """
        Converts a CSV file to an HDF5 file.

        This method reads a CSV file, converts its contents to a sparse matrix,
        and saves the matrix and its metadata to an HDF5 file. The output file
        will be saved in the specified output directory with the same name as
        the input file but with an .h5 extensionnstead of a .csv one.

        Raises:
            FileExistsError: If the output HDF5 file already exists and the force
                             flag is not set.
            Exception: If any error occurs during the conversion process.

        Returns:
            bool: True if the conversion is successful.
        """
        self.read_in_csv()

        h5_path = Path(self.output_dir) / Path(Path(self.filename).stem + ".h5")
        if not self.force and Path(h5_path).exists():
            raise FileExistsError(
                f"File {h5_path} already exists. Use --force/-f to overwrite."
            )
        self.create_output_dir()
        try:
            h5_file = h5py.File(h5_path, "w")
            matrix = csc_matrix(self.matrix)
            h5_file.create_group("matrix")
            h5_file.create_dataset(
                "matrix/barcodes", data=list(self.matrix.columns), dtype="S18"
            )
            h5_file.create_dataset("matrix/indices", data=matrix.indices, dtype="int64")
            h5_file.create_dataset("matrix/indptr", data=matrix.indptr, dtype="int64")
            h5_file.create_dataset("matrix/shape", data=matrix.shape, dtype="int32")
            h5_file.create_dataset("matrix/data", data=matrix.data, dtype="int32")
            h5_file.create_group("matrix/features")
            # TODO: check if all features are gene exp
            h5_file.create_dataset(
                "matrix/features/feature_type",
                data=["Gene Expression"] * self.matrix.shape[1],
                dtype=f"S{len('Gene Expression')}",
            )
            if self.genome is not None:
                gnome = [self.genome] * self.matrix.shape[0]
                dtype = f"S{len(self.genome)}"
            else:
                gnome = [""] * self.matrix.shape[0]
                dtype = "S16"
            h5_file.create_dataset("matrix/features/genome", data=gnome, dtype=dtype)

            gene_ids = [""] * self.matrix.shape[0]
            gene_names = [""] * self.matrix.shape[0]
            if pd.api.types.is_dtype_equal(
                self.matrix.index.dtype, pd.api.types.pandas_dtype("object")
            ):
                if self.matrix.index[0].startswith("ENS"):
                    gene_ids = self.matrix.index
                else:
                    gene_names = self.matrix.index
            h5_file.create_dataset("matrix/features/id", data=gene_ids, dtype="S16")
            h5_file.create_dataset("matrix/features/name", data=gene_names, dtype="S16")

            h5_file.create_dataset(
                "matrix/features/_all_tag_keys", data=["genome"], dtype="S6"
            )
        except Exception as e:
            print(f"Error: {e}")
            raise e
        return True

    def extract_features(self, features, matrix):
        """
        Extracts gene IDs, gene names, and feature types from the given features DataFrame.

        Parameters:
        features (pd.DataFrame): A DataFrame containing feature information. The structure of the DataFrame can vary:
                                 - If it has 1 column, it contains either gene IDs or gene names.
                                 - If it has 2 columns, it contains gene IDs and gene names.
                                 - If it has 3 columns, it contains gene IDs, gene names, and feature types.
                                 The index of the DataFrame can also be used to determine gene IDs or gene names.
        matrix (pd.DataFrame): A DataFrame representing the gene expression matrix. Used to determine the number of features.

        Returns:
        tuple: A tuple containing three lists:
               - gene_ids (list): A list of gene IDs.
               - gene_names (list): A list of gene names.
               - features_type (list): A list of feature types, defaulting to "Gene Expression" if not provided.
        """
        features_type = ["Gene Expression"] * matrix.shape[0]
        gene_ids = [""] * matrix.shape[0]
        gene_names = [""] * matrix.shape[0]
        if features.shape[1] == 1:
            if pd.api.types.is_dtype_equal(
                features.index.dtype, pd.api.types.pandas_dtype("object")
            ):
                if features.index[0].startswith("ENS"):
                    gene_ids = features.index.to_list()
                    gene_names = features[features.columns[0]].to_list()
                else:
                    gene_names = features.index.to_list()
                    gene_ids = features[features.columns[0]].to_list()
            else:
                if features.iloc[0, 0].startswith("ENS"):
                    gene_ids = features[features.columns[0]].to_list()
                else:
                    gene_names = features[features.columns[0]].to_list()
        elif features.shape[1] == 2:
            if pd.api.types.is_dtype_equal(
                features.index.dtype, pd.api.types.pandas_dtype("object")
            ):
                if features.index[0].startswith("ENS"):
                    gene_ids = features.index.to_list()
                    gene_names = features[features.columns[0]].to_list()
                    features_type = features[features.columns[1]].to_list()
                else:
                    gene_names = features.index.to_list()
                    gene_ids = features[features.columns[0]].to_list()
                    features_type = features[features.columns[1]].to_list()
            else:
                if features.iloc[0, 0].startswith("ENS"):
                    gene_ids = features[features.columns[0]].to_list()
                    gene_names = features[features.columns[1]].to_list()
                else:
                    gene_names = features[features.columns[0]].to_list()
                    gene_ids = features[features.columns[1]].to_list()
        elif features.shape[1] == 3:
            if pd.api.types.is_dtype_equal(
                features.index.dtype, pd.api.types.pandas_dtype("object")
            ):
                if features.index[0].startswith("ENS"):
                    gene_ids = features.index.to_list()
                    gene_names = features[features.columns[0]].to_list()
                    features_type = features[features.columns[1]].to_list()
                else:
                    gene_names = features.index.to_list()
                    gene_ids = features[features.columns[0]].to_list()
                    features_type = features[features.columns[1]].to_list()
            else:
                if features.iloc[0, 0].startswith("ENS"):
                    gene_ids = features[features.columns[0]].to_list()
                    gene_names = features[features.columns[1]].to_list()
                    features_type = features[features.columns[2]].to_list()
                else:
                    gene_names = features[features.columns[0]].to_list()
                    gene_ids = features[features.columns[1]].to_list()
                    features_type = features[features.columns[2]].to_list()
        return gene_ids, gene_names, features_type

    def mtx_to_h5(self):
        """
        Converts a Matrix Market file to an HDF5 file.

        This method reads a Matrix Market file, processes its contents, and writes the data to an HDF5 file. The HDF5 file will contain the matrix data, barcodes, and features.

        Raises:
            FileExistsError: If the HDF5 file already exists and the `force` flag is not set.
            Exception: If any error occurs during the file processing or writing.

        Returns:
            bool: True if the conversion is successful.

        Notes:
            - The method will create the output directory if it does not exist.
            - If the `force` flag is set and the HDF5 file already exists, the existing file will be overwritten.
            - The method uses gzip compression for the HDF5 datasets.
        """

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
            """
            Helper function to create a dataset in an HDF5 file.

            Parameters:
                h5_file (h5py.File): The HDF5 file object where the dataset will be created.
                name (str): The name of the dataset to create.
                data (array-like): The data to store in the dataset.
                dtype (str or numpy.dtype): The data type of the dataset.
                maxshape (tuple or None, optional): The maximum shape of the dataset. Default is (None,).
                compression (str, optional): The compression strategy. Default is "gzip".
                compression_opts (int, optional): The compression level. Default is 1.

            Returns:
                None
            """
            h5_file.create_dataset(
                name,
                data=data,
                dtype=dtype,
                maxshape=maxshape,
                compression=compression,
                compression_opts=compression_opts,
            )

        features, barcodes, matrix, _ = self.read_in_mtx()

        h5_path = Path(self.output_dir) / Path(Path(self.filename).stem + ".h5")
        if not self.force and Path(h5_path).exists():
            raise FileExistsError(
                f"File {h5_path} already exists. Use --force/-f to overwrite."
            )
        elif self.force and Path(h5_path).exists():
            Path.unlink(h5_path)
        self.create_output_dir()

        try:
            h5_file = h5py.File(h5_path, "w")
            matrix = csc_matrix(matrix)
            h5_file.create_group("matrix")
            dset_dict = {
                "matrix/barcodes": {
                    "data": barcodes[0],
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
            }
            for key, value in dset_dict.items():
                create_h5_dataset(
                    h5_file=h5_file,
                    name=key,
                    data=dset_dict[key]["data"],
                    dtype=dset_dict[key]["dtype"],
                )

            h5_file.create_group("matrix/features")

            # TODO: check if all features are gene exp

            # features_type = ["Gene Expression"] * matrix.shape[0]
            # gene_ids = [""] * matrix.shape[0]
            # gene_names = [""] * matrix.shape[0]

            gene_ids, gene_names, features_type = self.extract_features(
                features=features, matrix=matrix
            )

            if features_type is None or len(features_type) == 0:
                features_type = ["Gene Expression"] * matrix.shape[0]
            if gene_ids is None or len(gene_ids) == 0:
                gene_ids = gene_names
            elif gene_names is None or len(gene_names) == 0:
                gene_names = gene_ids
            if self.genome is not None:
                gnome = [self.genome] * matrix.shape[0]
                dtype = f"S{len(self.genome)}"
            else:
                gnome = ["unknown"] * matrix.shape[0]
                dtype = "S16"

            # this is just necessary because of scanpy's only importing
            # gene_ids
            # if gene_ids == [""] * matrix.shape[0]:
            #     gene_ids = gene_names
            # gene_ids = list(range(matrix.shape[0]))

            dset_dict = {
                "matrix/features/feature_type": {
                    "data": features_type,
                    "dtype": "S15",
                },
                "matrix/features/id": {
                    "data": gene_ids,
                    "dtype": "S15",
                },
                "matrix/features/name": {
                    "data": gene_names,
                    "dtype": "S17",
                },
                "matrix/features/genome": {
                    "data": gnome,
                    "dtype": dtype,
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

        except Exception as e:
            print(f"Error: {e}")
            raise e
        return True

    def mtx_to_anndata(self):
        """
        Converts a matrix market file to an AnnData object and saves it as a .h5ad file.

        This method reads in a matrix market file, extracts features, and converts the data into an AnnData object.
        The resulting AnnData object is then saved to the specified output directory in .h5ad format.

        Raises:
            FileExistsError: If the output file already exists and the force flag is not set.
            Exception: If any other error occurs during the process.

        Returns:
            bool: True if the conversion and saving process is successful.
        """
        feats, barcodes, matrix, _ = self.read_in_mtx()
        self.create_output_dir()
        try:
            ad_path = Path(self.output_dir) / Path(Path(self.filename).stem + ".h5ad")
            if Path(ad_path).exists():
                if self.force:
                    os.remove(ad_path)
                else:
                    raise FileExistsError
            adata = ad.AnnData(csc_matrix(matrix), filename=str(ad_path), filemode="a")
        except FileExistsError as e:
            print(
                f"File {ad_path} already exists."
                + " Select a different output directory or use --force/-f."
            )
            raise e
        except Exception as e:
            print(f"Error: {e}")
            raise e
        gene_ids, gene_names, features_type = self.extract_features(
            features=feats, matrix=matrix
        )
        adata.obs_names = barcodes[0].tolist()
        if gene_ids == [""] * matrix.shape[0]:
            adata.var_names = gene_names
        else:
            adata.var_names = gene_ids
        try:
            adata.write(compression="gzip")
        except Exception as e:
            print(f"Error: {e}")
            raise e
        return True

    def mtx_to_csv(self):
        """
        Converts a matrix market file to CSV format and saves it to the output directory.

        This method reads in a matrix market file, converts it to a dense pandas DataFrame,
        and then saves the DataFrame to a CSV file. It also saves the features to a separate
        CSV file. If the output files already exist and the `force` attribute is not set to True,
        a FileExistsError is raised.

        Returns:
            bool: True if the conversion and saving were successful.

        Raises:
            FileExistsError: If the output CSV files already exist and `force` is not set to True.
        """
        features, barcodes, matrix, mtx_feats = self.read_in_mtx()
        # borrowing heuristic from 10X for scaling of memory usage from matrix size:
        # https://github.com/10XGenomics/cellranger/blob/main/lib/python/cellranger/h5_constants.py
        # 2.6 times matrix file size in memory usage. Can issue a warning if the system memory
        # is less than the calculated memory usage.

        df = pd.DataFrame(
            matrix.todense(), columns=barcodes[0].tolist(), index=mtx_feats
        )
        self.create_output_dir()
        csv_path = Path(self.output_dir) / Path(Path(self.filename).stem + ".csv")
        features_path = Path(self.output_dir) / Path(
            Path(self.filename).stem + "_" + "features.csv"
        )
        if not self.force and Path(csv_path).exists():
            raise FileExistsError(
                f"File {csv_path} already exists. Use --force/-f to overwrite."
            )
        if not self.force and Path(features_path).exists():
            raise FileExistsError(
                f"File {features_path} already exists. Use --force/-f to overwrite."
            )
        df.to_csv(csv_path)
        features.to_csv(
            features_path,
            index=False,
        )
        return True

    # could handle both tsv and csv here. xsv?
    def csv_to_mtx(self):
        """
        Converts a CSV file to Matrix Market (MTX) format along with features and barcodes files.

        This method reads a CSV file, processes it, and writes the data into three files:
        - matrix.mtx.gz: The matrix in Matrix Market format, compressed with gzip.
        - features.tsv.gz: A TSV file containing gene IDs, gene names, and feature types, compressed with gzip.
        - barcodes.tsv.gz: A TSV file containing barcodes, compressed with gzip.

        Raises:
            FileExistsError: If any of the output files already exist and the `force` flag is not set.

        Returns:
            bool: True if the conversion is successful.
        """
        self.read_in_csv()

        mtx_output_dir = Path(self.output_dir) / Path(Path(self.filename).stem)
        self.create_output_dir(mtx_output_dir)
        mtx_path = mtx_output_dir / Path("matrix.mtx")
        feats_path = mtx_output_dir / Path("features.tsv.gz")
        barcodes_path = mtx_output_dir / Path("barcodes.tsv.gz")

        for p in [mtx_path, feats_path, barcodes_path]:
            if not self.force and p.exists():
                raise FileExistsError(
                    f"File {p} already exists. Use --force/-f to overwrite."
                )

        pattern = re.compile(r"^[AGCT\d-]+$")
        if all(pattern.fullmatch(s) for s in self.matrix.index):
            # the indices contain the barcodes, so we need to transpose
            self.matrix = self.matrix.T
        features_type = ["Gene Expression"] * self.matrix.shape[0]
        gene_ids = self.matrix.index.tolist()
        gene_names = self.matrix.index.tolist()

        mmwrite(target=mtx_path, a=coo_matrix(self.matrix))
        with open(mtx_path, "rb") as f_in:
            with gzip.open(Path(str(mtx_path) + ".gz"), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(mtx_path)
        features = pd.DataFrame(
            {
                "gene_id": gene_ids,
                "gene_name": gene_names,
                "feature_type": features_type,
            }
        )
        features.to_csv(feats_path, index=False, header=False, sep="\t")
        # with gzip.open(feats_path, "wb") as f:
        #     f.write(b"\n".join([x.encode() for x in self.matrix.index]))
        #     f.write(b"\n")
        with gzip.open(barcodes_path, "wb") as f:
            f.write(b"\n".join([x.encode() for x in self.matrix.columns]))
            f.write(b"\n")
        return True

    def h5_to_mtx(self):
        """
        Converts an HDF5 file to Matrix Market (MTX) format along with features and barcodes TSV files.

        This method reads data from an HDF5 file, processes it, and writes the output to Matrix Market (MTX) format.
        It also generates corresponding features and barcodes TSV files.

        Raises:
            FileExistsError: If the output files already exist and the `force` flag is not set.

        Returns:
            bool: True if the conversion is successful.
        """
        self.read_in_h5()

        mtx_output_dir = Path(self.output_dir) / Path(Path(self.filename).stem)
        self.create_output_dir(mtx_output_dir)
        mtx_path = mtx_output_dir / Path("matrix.mtx")
        feats_path = mtx_output_dir / Path("features.tsv")
        barcodes_path = mtx_output_dir / Path("barcodes.tsv")
        for p in [mtx_path, feats_path, barcodes_path]:
            if not self.force and p.exists():
                raise FileExistsError(
                    f"File {p} already exists. Use --force/-f to overwrite."
                )
        mmwrite(target=mtx_path, a=csc_matrix(self.matrix["matrix/data"][:]))
        # construct features dataframe
        # should check basic stuff, e.g. that it's not empty etc, but later
        feat_type = None
        genome = None
        gene_names = None
        gene_ids = None
        if max([len(x) for x in self.matrix["matrix/features/feature_type"][:10]]) > 0:
            feat_type = [
                str(j)
                for j in self.matrix["matrix/features/feature_type"][:].astype(str)
            ]
        else:
            feat_type = ["Gene Expression"] * self.matrix[
                "matrix/features/feature_type"
            ].shape[0]
        if max([len(x) for x in self.matrix["matrix/features/genome"][:10]]) > 0:
            genome = [
                str(j) for j in self.matrix["matrix/features/genome"][:].astype(str)
            ]
        if max([len(x) for x in self.matrix["matrix/features/name"][:10]]) > 0:
            gene_names = [
                str(j) for j in self.matrix["matrix/features/name"][:].astype(str)
            ]
        if max(
            [len(x) for x in self.matrix["matrix/features/id"][:10]]
        ) > 0 and self.matrix["matrix/features/id"].dtype not in ["int32", "int64"]:
            gene_ids = [
                str(j) for j in self.matrix["matrix/features/id"][:].astype(str)
            ]
        feature_dict = {}
        if gene_ids is not None:
            feature_dict["id"] = gene_ids
        if gene_names is not None:
            feature_dict["name"] = gene_names
        if feat_type is not None:
            feature_dict["feature_type"] = feat_type
        if genome is not None:
            feature_dict["genome"] = genome

        if feature_dict["id"] is None:
            feature_dict["id"] = feature_dict["name"]
        if feature_dict["name"] is None:
            feature_dict["name"] = feature_dict["id"]

        features = pd.DataFrame(feature_dict)
        assert features.shape[1] >= 2
        features.to_csv(feats_path, index=False, header=False, sep="\t")
        with open(barcodes_path, "w") as f:
            f.write(
                "\n".join([bc for bc in self.matrix["matrix/barcodes"][:].astype(str)])
            )
        return True


@app.command()
def convert(
    filename: str,
    from_format: Annotated[
        str, typer.Option("--from", help="The format to convert from")
    ],
    to_format: Annotated[str, typer.Option("--to", help="The format to convert to")],
    output_dir: Annotated[
        str,
        typer.Option(
            "--output-dir", "-d", help="Which directory/folder to output files into."
        ),
    ] = ".",
    genome: Annotated[
        str, typer.Option("--genome", "-g", help="What genome to use")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Overwrite existing files")
    ] = False,
) -> bool:
    """
    Convert files from one format to another and save them to the specified output directory.

    Args:
        filename (list[str]): List of filenames to be converted.
        from_format (str): The format to convert from.
        to_format (str): The format to convert to.
        output_dir (str, optional): The directory to output files into. Defaults to the current directory.

    Returns:
        bool: True if the conversion was successful, False otherwise.
    """
    # print(f"{filename}, {len(filename)=}, {type(filename)=}")
    # print(f"{filename=}, {from_format=}, {to_format=}, {output_dir=}, {force=}")
    c = Converter(
        filename=filename,
        from_format=from_format,
        to_format=to_format,
        output_dir=output_dir,
        genome=genome,
        force=force,
    )
    return c.convert()


if __name__ == "__main__":
    app()
