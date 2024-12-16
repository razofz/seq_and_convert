import typer
from typing_extensions import Annotated
from .core import Converter

app = typer.Typer()


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
