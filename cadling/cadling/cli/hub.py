"""Hub CLI commands for HuggingFace integration.

Provides commands for publishing CAD datasets to HuggingFace Hub
and previewing streamed data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click


@click.group()
def hub():
    """HuggingFace Hub operations for CAD datasets.

    Publish datasets to HuggingFace Hub or preview streamed data
    from hosted datasets.

    Examples:

        \b
        # Publish local Parquet files to Hub
        cadling hub publish ./data --repo-id username/my-cad-dataset

        \b
        # Preview a streamed dataset
        cadling hub preview --repo-id latticelabs/deepcad-sequences --max-batches 5
    """
    pass


@hub.command()
@click.argument("source_dir", type=click.Path(exists=True))
@click.option(
    "--repo-id",
    "-r",
    required=True,
    help="HuggingFace repository ID (e.g., username/dataset-name)",
)
@click.option(
    "--private",
    is_flag=True,
    help="Make the repository private",
)
@click.option(
    "--token",
    envvar="HF_TOKEN",
    help="HuggingFace API token (default: from HF_TOKEN env var)",
)
@click.option(
    "--pattern",
    default="*.parquet",
    help="Glob pattern for finding files (default: *.parquet)",
)
@click.option(
    "--shard-size",
    type=int,
    default=500,
    help="Target shard size in MB (default: 500)",
)
@click.option(
    "--compression",
    type=click.Choice(["zstd", "snappy", "gzip"]),
    default="zstd",
    help="Parquet compression codec (default: zstd)",
)
@click.option(
    "--schema",
    type=click.Choice(["command_sequences", "brep_graphs", "text_cad"]),
    default="command_sequences",
    help="Schema type for validation (default: command_sequences)",
)
@click.option(
    "--no-readme",
    is_flag=True,
    help="Skip generating dataset card",
)
def publish(
    source_dir: str,
    repo_id: str,
    private: bool,
    token: Optional[str],
    pattern: str,
    shard_size: int,
    compression: str,
    schema: str,
    no_readme: bool,
):
    """Publish CAD dataset to HuggingFace Hub.

    Uploads Parquet files from SOURCE_DIR to the HuggingFace Hub repository
    specified by --repo-id. Creates the repository if it doesn't exist.

    Examples:

        \b
        # Publish from current directory
        cadling hub publish . --repo-id myuser/cad-dataset

        \b
        # Publish private dataset with specific pattern
        cadling hub publish ./output --repo-id myorg/private-data --private --pattern "train*.parquet"

        \b
        # Publish with schema validation for text-CAD data
        cadling hub publish ./text2cad -r user/text2cad --schema text_cad
    """
    try:
        from cadling.data.hub_publisher import CADDatasetPublisher, PublishConfig

        click.echo(f"Publishing {source_dir} to {repo_id} (schema: {schema})...", err=True)

        config = PublishConfig(
            repo_id=repo_id,
            private=private,
            token=token,
            shard_size_mb=shard_size,
            compression=compression,
            create_readme=not no_readme,
            schema_type=schema,
        )

        publisher = CADDatasetPublisher(config)
        url = publisher.publish_from_directory(source_dir, pattern=pattern)

        click.echo(f"\nSuccessfully published dataset!", err=True)
        click.echo(f"View at: {url}", err=True)

    except ImportError as e:
        click.echo(
            f"Error: Missing dependencies. Install with:\n"
            f"  pip install huggingface-hub pyarrow",
            err=True,
        )
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@hub.command()
@click.option(
    "--repo-id",
    "-r",
    required=True,
    help="HuggingFace dataset repository ID",
)
@click.option(
    "--split",
    default="train",
    help="Dataset split to preview (default: train)",
)
@click.option(
    "--max-batches",
    "-n",
    type=int,
    default=5,
    help="Maximum number of batches to preview (default: 5)",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=4,
    help="Batch size (default: 4)",
)
@click.option(
    "--columns",
    "-c",
    multiple=True,
    help="Columns to load (can specify multiple, default: all)",
)
@click.option(
    "--token",
    envvar="HF_TOKEN",
    help="HuggingFace API token for private datasets",
)
@click.option(
    "--json-output",
    is_flag=True,
    help="Output as JSON instead of human-readable format",
)
def preview(
    repo_id: str,
    split: str,
    max_batches: int,
    batch_size: int,
    columns: tuple,
    token: Optional[str],
    json_output: bool,
):
    """Preview a streamed CAD dataset from HuggingFace Hub.

    Streams samples from a HuggingFace dataset and displays them.
    Useful for verifying dataset structure and content before training.

    Examples:

        \b
        # Preview first 5 batches
        cadling hub preview --repo-id latticelabs/deepcad-sequences

        \b
        # Preview specific columns as JSON
        cadling hub preview --repo-id myuser/dataset -c command_types -c mask --json-output
    """
    try:
        from cadling.data.streaming import CADStreamingDataset, CADStreamingConfig

        click.echo(f"Streaming from {repo_id} ({split} split)...\n", err=True)

        config = CADStreamingConfig(
            dataset_id=repo_id,
            split=split,
            streaming=True,
            batch_size=batch_size,
            shuffle=False,
            columns=list(columns) if columns else None,
            token=token,
        )

        dataset = CADStreamingDataset(config)

        batch_count = 0
        sample_count = 0

        for batch in dataset.batch_iter():
            batch_count += 1
            batch_size_actual = _get_batch_size(batch)
            sample_count += batch_size_actual

            if json_output:
                # Convert tensors to lists for JSON
                batch_json = _batch_to_json(batch)
                click.echo(json.dumps(batch_json, indent=2))
            else:
                click.echo(f"=== Batch {batch_count} ===")
                _print_batch_summary(batch)
                click.echo()

            if batch_count >= max_batches:
                break

        click.echo(
            f"\nPreviewed {sample_count} samples in {batch_count} batches",
            err=True,
        )

    except ImportError as e:
        click.echo(
            f"Error: Missing dependencies. Install with:\n"
            f"  pip install datasets torch",
            err=True,
        )
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@hub.command()
@click.argument("source_dir", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output directory for Parquet files",
)
@click.option(
    "--config",
    "-c",
    type=click.Choice(["default", "with_text", "with_renders", "full"]),
    default="default",
    help="Dataset configuration (default: default)",
)
@click.option(
    "--splits",
    default="train,val,test",
    help="Comma-separated list of splits to process",
)
@click.option(
    "--compression",
    type=click.Choice(["zstd", "snappy", "gzip"]),
    default="zstd",
    help="Parquet compression codec",
)
def build(
    source_dir: str,
    output: str,
    config: str,
    splits: str,
    compression: str,
):
    """Build Parquet dataset from source files.

    Converts DeepCAD-style JSON files or STEP files to Parquet format
    for efficient storage and streaming.

    Examples:

        \b
        # Build from DeepCAD JSON files
        cadling hub build ./deepcad_json -o ./output

        \b
        # Build with text descriptions
        cadling hub build ./text2cad -o ./output --config with_text
    """
    try:
        from cadling.data.hf_builders import CADCommandSequenceBuilder
        from cadling.data.hf_builders.cad_dataset_builder import (
            DEFAULT_CONFIG,
            WITH_TEXT_CONFIG,
            WITH_RENDERS_CONFIG,
            FULL_CONFIG,
        )

        click.echo(f"Building dataset from {source_dir}...", err=True)

        # Select configuration
        config_map = {
            "default": DEFAULT_CONFIG,
            "with_text": WITH_TEXT_CONFIG,
            "with_renders": WITH_RENDERS_CONFIG,
            "full": FULL_CONFIG,
        }
        builder_config = config_map[config]

        # Parse splits
        split_list = [s.strip() for s in splits.split(",")]

        builder = CADCommandSequenceBuilder(
            source_dir=source_dir,
            config=builder_config,
            splits=split_list,
        )

        output_files = builder.build(output, compression=compression)

        click.echo(f"\nBuilt {len(output_files)} files:", err=True)
        for split, path in output_files.items():
            size_mb = path.stat().st_size / (1024 * 1024)
            click.echo(f"  {split}: {path} ({size_mb:.1f} MB)", err=True)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _get_batch_size(batch: dict) -> int:
    """Get the batch size from a batch dict."""
    for key, value in batch.items():
        if hasattr(value, "shape"):
            return value.shape[0]
        if isinstance(value, (list, tuple)):
            return len(value)
    return 0


def _batch_to_json(batch: dict) -> dict:
    """Convert a batch with tensors to JSON-serializable dict."""
    result = {}
    for key, value in batch.items():
        if hasattr(value, "tolist"):
            result[key] = value.tolist()
        elif isinstance(value, (list, tuple)):
            result[key] = list(value)
        else:
            result[key] = str(value)
    return result


def _print_batch_summary(batch: dict) -> None:
    """Print a human-readable summary of a batch."""
    for key, value in batch.items():
        if hasattr(value, "shape"):
            click.echo(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        elif isinstance(value, (list, tuple)):
            click.echo(f"  {key}: list of {len(value)} items")
            if len(value) > 0 and isinstance(value[0], str):
                click.echo(f"    First: {value[0][:50]}...")
        else:
            click.echo(f"  {key}: {type(value).__name__}")


__all__ = ["hub"]
