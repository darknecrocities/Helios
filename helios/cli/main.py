"""Helios CLI entry point."""

import typer

import helios

app = typer.Typer(
    name="helios",
    help="Helios Open-Weight LLM CLI: inspection, training, and memory management.",
    add_completion=False,
)


@app.command()
def version() -> None:
    """Print Helios ecosystem version."""
    typer.echo(f"Helios version {helios.__version__}")


@app.command()
def inspect(
    model_size: str = typer.Option(
        "nano-20m",
        "--model-size",
        "-m",
        help="Target model size preset (nano-15m, nano-20m, nano-25m, nano-35m)",
    ),
) -> None:
    """Inspect Helios model configuration and parameter counts."""
    typer.echo(f"Helios Model Inspector - Target: {model_size}")


if __name__ == "__main__":
    app()
