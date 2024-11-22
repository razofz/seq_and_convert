import typer
from typing_extensions import Annotated
import pandas as pd
from pathlib import Path
import magic
import mimetypes
import json
from scipy.io import mmread, mmwrite
from scipy.sparse import coo_matrix

app = typer.Typer()


class Converter:
    def __init__(
        self,
        filename: str,
        from_format: str,
        to_format: str,
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
        if not self.filename:
            raise ValueError("No filename provided")
        if not Path(self.filename).exists():
            raise FileNotFoundError(f"File {self.filename} not found")
        try:
            ft = self.decide_filetype()
        except Exception as e:
            print(f"Error: {e}")
        if ft is not None:
            print(f"File {self.filename} seems to be a {ft} file")
            # return self.convert()

    def __str__(self):
        return f"{self.filename}"

    def decide_filetype(self):
        if Path(self.filename).is_dir():
            dir_files = [f for f in Path(self.filename).iterdir() if f.is_file()]
            print(dir_files)
        else:
            for key in self.lookup_table:
                eligible = True
                if not self.lookup_table[key]["magic_mime"] == magic.from_file(
                    self.filename, mime=True
                ):
                    eligible = False
                if (
                    not self.lookup_table[key]["mimetype_guess_encoding"]
                    == mimetypes.guess_type(self.filename)[1]
                ):
                    eligible = False
                if (
                    not self.lookup_table[key]["mimetype_guess_type"]
                    == mimetypes.guess_type(self.filename)[0]
                ):
                    eligible = False
                if (
                    not self.lookup_table[key]["Path_suffix"]
                    == Path(self.filename).suffix
                ):
                    eligible = False
                if self.from_format == "gz":
                    if (
                        not self.lookup_table[key]["magic"].split(",")[0]
                        == magic.from_file(self.filename).split(",")[0]
                    ):
                        eligible = False
                else:
                    if not self.lookup_table[key]["magic"] == magic.from_file(
                        self.filename
                    ):
                        eligible = False
                if eligible:
                    return key
        return None

    def convert(self):
        if self.from_format == "csv" and self.to_format == "mtx":
            return self.csv_to_mtx()
        elif self.from_format == "mtx" and self.to_format == "csv":
            return self.mtx_to_csv()
        else:
            raise NotImplementedError(
                f"Conversion from {self.from_format} to {self.to_format} is not yet implemented"
            )

    def mtx_to_csv(self):
        pass

    # could handle both tsv and csv here. xsv?
    def csv_to_mtx(self):
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

        # print(self.matrix.head())
        mtx_output_dir = Path(self.output_dir) / Path(Path(self.filename).stem)
        try:
            Path.mkdir(Path(mtx_output_dir), parents=True, exist_ok=self.force)
        except FileExistsError as e:
            print(
                f"Directory {mtx_output_dir} already exists."
                + " Select a different output directory or use --force/-f."
            )
            raise e
        mmwrite(target=mtx_output_dir / Path("matrix.mtx"), a=coo_matrix(self.matrix))
        with open(mtx_output_dir / Path("features.tsv"), "w") as f:
            f.write("\n".join(self.matrix.index))
        with open(mtx_output_dir / Path("barcodes.tsv"), "w") as f:
            f.write("\n".join(self.matrix.columns))
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
        force=force,
    )
    return c.convert()


if __name__ == "__main__":
    app()
