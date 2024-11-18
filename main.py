import typer
from typing_extensions import Annotated
import pandas as pd
from pathlib import Path

app = typer.Typer()


class Converter:
    def __init__(
        self, filename: list[str], from_format, to_format, output_dir: str = "."
    ):
        self.filename = filename
        self.from_format = from_format
        self.to_format = to_format
        self.output_dir = output_dir
        if not self.filename:
            raise ValueError("No filename provided")
        elif len(self.filename) < 1:
            raise ValueError("No filename provided")
        elif len(self.filename) == 1:
            # potentially csv
            if not Path(self.filename[0]).exists():
                raise FileNotFoundError(f"File {self.filename[0]} not found")
            try:
                self.matrix = pd.read_table(self.filename[0], header=None)
                print(self.matrix.head())
            except Exception as e:
                print(f"Error: {e}")

    def __str__(self):
        return f"{self.filename}"


@app.command()
def hello(name: str):
    print(f"Hello {name}")


@app.command()
def goodbye(name: str, formal: bool = False):
    if formal:
        print(f"Goodbye Mx. {name}")
    else:
        print(f"Goodbye {name}")


@app.command()
def convert(
    filename: list[str],
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
    print(f"{filename}, {len(filename)=}, {type(filename)=}")
    print(Converter(filename, from_format, to_format, output_dir))


if __name__ == "__main__":
    app()
