import typer
from typing_extensions import Annotated
import pandas as pd
from pathlib import Path
import magic
import mimetypes
import json
from scipy.io import mmread

app = typer.Typer()


class Converter:
    def __init__(
        self, filename: str, from_format: str, to_format: str, output_dir: str = "."
    ):
        self.filename = filename
        self.from_format = from_format
        self.to_format = to_format
        self.output_dir = output_dir
        self.lookup_table = json.load(open("mimetypes.json", "r"))
        self.filetype_checks = [
            "magic",
            "magic_mime",
            "mimetype_guess_type",
            "mimetype_guess_encoding",
            "Path_suffix",
        ]
        if not self.filename:
            raise ValueError("No filename provided")
        if not Path(self.filename).exists():
            raise FileNotFoundError(f"File {self.filename} not found")
        try:
            ft = self.decide_filetype()
            if ft is not None:
                print(f"File {self.filename} seems to be a {ft} file")
            # self.matrix = pd.read_table(self.filename)
            # would be nice to compare the first row to other rows to see if it's probably a header or not.
            # compare types?
            # print(self.matrix.head())
        except Exception as e:
            print(f"Error: {e}")

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
            self.csv_to_mtx()
        elif self.from_format == "mtx" and self.to_format == "csv":
            self.mtx_to_csv()
        else:
            raise NotImplementedError(
                f"Conversion from {self.from_format} to {self.to_format} is not yet implemented"
            )

    def mtx_to_csv(self):
        pass

    def csv_to_mtx(self):
        pass


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
    Converter(filename, from_format, to_format, output_dir)


if __name__ == "__main__":
    app()
