import typer
from typing_extensions import Annotated

app = typer.Typer()


class Converter:
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return f"Hello {self.name}"


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
    output_dir: Annotated[str, typer.Option("--output-dir", "-d")],
    from_format: Annotated[str, typer.Option("--from")],
    to: Annotated[str, typer.Option()],
):
    print(f"{filename}, {len(filename)=}, {type(filename)=}")


if __name__ == "__main__":
    app()
