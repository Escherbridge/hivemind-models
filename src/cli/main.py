"""
CLI entry point for hivemind-models.

Provides commands for:
- convert: Convert HuggingFace models to sharded format
- upload: Upload shards to Cloudflare R2
- validate: Validate sharded models
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

app = typer.Typer(
    name="hivemind-models",
    help="Model sharding and CDN upload pipeline for Hivemind distributed inference",
    add_completion=False,
)

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@app.command()
def convert(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to model configuration YAML file",
        exists=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (overrides config)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be done without actually converting",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
) -> None:
    """
    Convert a HuggingFace model to sharded safetensors format.

    Example:
        python -m src.cli.main convert --config configs/tinyllama-1b.yaml
    """
    setup_logging(verbose)

    from src.convert.sharder import ShardConfig, ModelSharder

    console.print(f"[bold blue]Loading configuration from {config}[/bold blue]")

    try:
        shard_config = ShardConfig.from_yaml(config)

        if output:
            shard_config.output_dir = output

        console.print(f"Model: [green]{shard_config.model_id}[/green]")
        console.print(f"Output: [green]{shard_config.output_dir}[/green]")
        console.print(f"Layer groups: {shard_config.layer_groups}")
        console.print(f"Quantize: {shard_config.quantize} ({shard_config.quant_bits}-bit)")

        sharder = ModelSharder(shard_config)

        if dry_run:
            console.print("\n[yellow]DRY RUN - No actual conversion will be performed[/yellow]\n")
            plan = sharder.dry_run()

            table = Table(title="Planned Shards")
            table.add_column("Filename", style="cyan")
            table.add_column("Description", style="green")

            for shard in plan["planned_shards"]:
                table.add_row(shard["filename"], shard["description"])

            console.print(table)
            return

        console.print("\n[bold]Starting conversion...[/bold]\n")
        result = sharder.shard()

        console.print(f"\n[bold green]Conversion complete![/bold green]")
        console.print(f"Total shards: {len(result.shards)}")
        console.print(f"Total size: {result.total_size_bytes / (1024**2):.2f} MB")
        console.print(f"Manifest: {result.manifest_path}")

        # Show shard summary
        table = Table(title="Generated Shards")
        table.add_column("Filename", style="cyan")
        table.add_column("Size (MB)", justify="right")
        table.add_column("Tensors", justify="right")

        for shard in result.shards:
            table.add_row(
                shard.filename,
                f"{shard.size_bytes / (1024**2):.2f}",
                str(shard.tensor_count),
            )

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


@app.command()
def upload(
    model_dir: Path = typer.Argument(
        ...,
        help="Directory containing sharded model",
        exists=True,
    ),
    model_name: str = typer.Option(
        ...,
        "--name",
        "-n",
        help="Name for the model in CDN",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
) -> None:
    """
    Upload a sharded model to Cloudflare R2.

    Requires R2 environment variables to be set.

    Example:
        python -m src.cli.main upload ./output --name tinyllama-1b-q4
    """
    setup_logging(verbose)

    from src.upload.r2 import create_uploader_from_env

    try:
        uploader = create_uploader_from_env()
    except ValueError as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        console.print("\nSet the following environment variables:")
        console.print("  - R2_ACCOUNT_ID")
        console.print("  - R2_ACCESS_KEY")
        console.print("  - R2_SECRET_KEY")
        console.print("  - R2_BUCKET_NAME")
        console.print("  - R2_PUBLIC_URL (optional)")
        raise typer.Exit(code=1)

    console.print(f"[bold blue]Uploading model to R2[/bold blue]")
    console.print(f"Source: [green]{model_dir}[/green]")
    console.print(f"Destination: [green]models/{model_name}[/green]")

    try:
        result = uploader.upload_model(model_dir, model_name)

        console.print(f"\n[bold green]Upload complete![/bold green]")
        console.print(f"Files uploaded: {result.successful}/{result.total_files}")
        console.print(f"Total size: {result.total_bytes / (1024**2):.2f} MB")

        if result.cdn_base_url:
            console.print(f"CDN URL: [cyan]{result.cdn_base_url}[/cyan]")

        if result.failed > 0:
            console.print(f"\n[yellow]Warning: {result.failed} files failed to upload[/yellow]")
            for r in result.results:
                if not r.success:
                    console.print(f"  - {r.key}: {r.message}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


@app.command()
def validate(
    model_dir: Path = typer.Argument(
        ...,
        help="Directory containing sharded model",
        exists=True,
    ),
    verify_checksums: bool = typer.Option(
        True,
        "--checksums/--no-checksums",
        help="Verify file checksums",
    ),
    verify_tensors: bool = typer.Option(
        True,
        "--tensors/--no-tensors",
        help="Verify tensor contents",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
) -> None:
    """
    Validate a sharded model.

    Checks manifest, checksums, and tensor integrity.

    Example:
        python -m src.cli.main validate ./output
    """
    setup_logging(verbose)

    from src.convert.validation import validate_shards, print_validation_report

    console.print(f"[bold blue]Validating model at {model_dir}[/bold blue]\n")

    report = validate_shards(
        model_dir,
        verify_checksums=verify_checksums,
        verify_tensors=verify_tensors,
    )

    print_validation_report(report)

    if not report.passed:
        raise typer.Exit(code=1)


@app.command()
def info(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to model configuration YAML file",
        exists=True,
    ),
) -> None:
    """
    Show information about a model configuration.

    Example:
        python -m src.cli.main info --config configs/tinyllama-1b.yaml
    """
    from src.convert.sharder import ShardConfig

    try:
        shard_config = ShardConfig.from_yaml(config)

        table = Table(title="Model Configuration")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Model ID", shard_config.model_id)
        table.add_row("Output Directory", str(shard_config.output_dir))
        table.add_row("Data Type", shard_config.dtype)
        table.add_row("Quantize", str(shard_config.quantize))
        if shard_config.quantize:
            table.add_row("Quantization Bits", str(shard_config.quant_bits))

        console.print(table)

        # Layer groups table
        groups_table = Table(title="Layer Groups")
        groups_table.add_column("Group", justify="center")
        groups_table.add_column("Start Layer", justify="right")
        groups_table.add_column("End Layer", justify="right")
        groups_table.add_column("Count", justify="right")

        for i, (start, end) in enumerate(shard_config.layer_groups):
            groups_table.add_row(
                str(i),
                str(start),
                str(end),
                str(end - start + 1),
            )

        console.print(groups_table)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def list_models(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
) -> None:
    """
    List models available in R2 bucket.

    Requires R2 environment variables to be set.
    """
    setup_logging(verbose)

    from src.upload.r2 import create_uploader_from_env

    try:
        uploader = create_uploader_from_env()
    except ValueError as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        raise typer.Exit(code=1)

    console.print("[bold blue]Fetching models from R2...[/bold blue]\n")

    objects = uploader.list_objects(prefix="models/")

    # Group by model name
    models: dict[str, list[dict]] = {}
    for obj in objects:
        parts = obj["key"].split("/")
        if len(parts) >= 2:
            model_name = parts[1]
            if model_name not in models:
                models[model_name] = []
            models[model_name].append(obj)

    if not models:
        console.print("[yellow]No models found[/yellow]")
        return

    table = Table(title="Available Models")
    table.add_column("Model", style="cyan")
    table.add_column("Files", justify="right")
    table.add_column("Total Size (MB)", justify="right")

    for model_name, files in sorted(models.items()):
        total_size = sum(f["size"] for f in files)
        table.add_row(
            model_name,
            str(len(files)),
            f"{total_size / (1024**2):.2f}",
        )

    console.print(table)


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
