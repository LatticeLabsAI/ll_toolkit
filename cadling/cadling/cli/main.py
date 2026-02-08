"""CADling CLI for CAD file processing.

This module provides a command-line interface for:
- Converting CAD files to structured formats
- Chunking CAD documents for RAG systems
- Generating synthetic Q&A pairs for training
- Generating CAD models from text/image descriptions

Commands:
    convert: Convert CAD file to JSON/Markdown
    chunk: Chunk CAD file for RAG
    generate-qa: Generate Q&A pairs from CAD file
    generate: Generate CAD model from text/image description
"""

import json
import sys
from pathlib import Path
from typing import Optional

import click


@click.group()
@click.version_option()
def cli():
    """CADling - CAD file processing toolkit.

    CADling converts CAD files (STEP, STL, BRep, IGES, DXF, PDF) to structured
    formats suitable for RAG systems and language models. Also supports generating
    CAD models from text descriptions via LLM code generation.

    Examples:

        \b
        # Convert STEP file to JSON
        cadling convert part.step --format json -o part.json

        \b
        # Chunk CAD file for RAG
        cadling chunk part.step --max-tokens 512

        \b
        # Generate Q&A pairs
        cadling generate-qa part.step --num-pairs 100 -o qa_pairs.jsonl

        \b
        # Generate CAD model from text
        cadling generate --from-text "A flanged bearing housing" -o bearing.step
    """
    pass


# Register the generate command from the generation CLI module
from cadling.cli.generate import generate_cmd

cli.add_command(generate_cmd, "generate")

# Register the hub command for HuggingFace Hub operations
from cadling.cli.hub import hub

cli.add_command(hub)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path (default: stdout)",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "markdown"], case_sensitive=False),
    default="json",
    help="Output format (default: json)",
)
@click.option(
    "--pretty",
    is_flag=True,
    help="Pretty-print JSON output",
)
def convert(input_file: str, output: Optional[str], format: str, pretty: bool):
    """Convert CAD file to structured format.

    Converts STEP, STL, BRep, or IGES files to JSON or Markdown format.
    The structured output includes entities, topology, and metadata.

    Examples:

        \b
        cadling convert part.step --format json
        cadling convert mesh.stl --format markdown -o output.md
        cadling convert assembly.step --format json --pretty -o assembly.json
    """
    try:
        from cadling.backend.document_converter import DocumentConverter
        from cadling.datamodel.base_models import ConversionStatus

        click.echo(f"Converting {input_file}...", err=True)

        # Convert file
        converter = DocumentConverter()
        result = converter.convert(input_file)

        # Check status
        if result.status != ConversionStatus.SUCCESS:
            click.echo(f"Error: Conversion failed", err=True)
            if result.errors:
                for error in result.errors:
                    click.echo(f"  - {error.error_message}", err=True)
            sys.exit(1)

        # Export to desired format
        if format.lower() == "json":
            output_data = result.document.export_to_json()

            # Pretty print if requested
            if pretty:
                parsed = json.loads(output_data)
                output_data = json.dumps(parsed, indent=2)
        else:
            output_data = result.document.export_to_markdown()

        # Write output
        if output:
            output_path = Path(output)
            output_path.write_text(output_data)
            click.echo(f"Saved to {output}", err=True)
        else:
            click.echo(output_data)

        click.echo(
            f"Successfully converted: {len(result.document.items)} items",
            err=True,
        )

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--max-tokens",
    "-t",
    type=int,
    default=512,
    help="Maximum tokens per chunk (default: 512)",
)
@click.option(
    "--overlap",
    type=int,
    default=50,
    help="Overlap tokens between chunks (default: 50)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file for chunks (JSONL format)",
)
def chunk(
    input_file: str,
    max_tokens: int,
    overlap: int,
    output: Optional[str],
):
    """Chunk CAD file for RAG systems.

    Splits a CAD document into semantically coherent chunks suitable for
    RAG (Retrieval-Augmented Generation) systems. Each chunk includes
    text, metadata, topology information, and embeddings (if available).

    Examples:

        \b
        cadling chunk part.step --max-tokens 512
        cadling chunk assembly.step --max-tokens 256 --overlap 25 -o chunks.jsonl
    """
    try:
        from cadling.chunker.hybrid_chunker import CADHybridChunker
        from cadling.backend.document_converter import DocumentConverter

        click.echo(f"Chunking {input_file}...", err=True)

        # Convert file
        converter = DocumentConverter()
        result = converter.convert(input_file)

        if not result.document:
            click.echo("Error: No document created", err=True)
            sys.exit(1)

        # Create chunker
        chunker = CADHybridChunker(
            max_tokens=max_tokens,
            overlap_tokens=overlap,
        )

        # Generate chunks
        chunks = list(chunker.chunk(result.document))

        click.echo(f"Generated {len(chunks)} chunks", err=True)

        # Output chunks
        if output:
            output_path = Path(output)
            with output_path.open("w") as f:
                for chunk in chunks:
                    f.write(chunk.model_dump_json() + "\n")
            click.echo(f"Saved chunks to {output}", err=True)
        else:
            for i, chunk in enumerate(chunks):
                click.echo(f"\n=== Chunk {i+1}/{len(chunks)} ===")
                click.echo(f"ID: {chunk.chunk_id}")
                click.echo(f"Doc: {chunk.doc_name}")
                click.echo(f"Entities: {len(chunk.meta.entity_ids)}")
                click.echo(f"Text preview: {chunk.text[:200]}...")
                click.echo()

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("generate-qa")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--num-pairs",
    "-n",
    type=int,
    default=100,
    help="Number of Q&A pairs to generate (default: 100)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output file for Q&A pairs (JSONL format)",
)
@click.option(
    "--model",
    "-m",
    default="gpt-4",
    help="LLM model for generation (default: gpt-4)",
)
@click.option(
    "--no-critique",
    is_flag=True,
    help="Disable critique step",
)
@click.option(
    "--api-key",
    envvar="OPENAI_API_KEY",
    help="API key (default: from OPENAI_API_KEY env var)",
)
def generate_qa(
    input_file: str,
    num_pairs: int,
    output: str,
    model: str,
    no_critique: bool,
    api_key: Optional[str],
):
    """Generate synthetic Q&A pairs from CAD file.

    Generates question-answer pairs for training language models to
    understand CAD data. Uses LLMs to create questions and answers
    based on CAD chunks, with optional critique and improvement.

    Examples:

        \b
        cadling generate-qa part.step -n 100 -o qa.jsonl
        cadling generate-qa assembly.step -n 50 -m claude-3-opus --no-critique -o qa.jsonl
    """
    try:
        from cadling.backend.document_converter import DocumentConverter
        from cadling.sdg.qa_generator import CADQAGenerator

        click.echo(f"Generating Q&A pairs from {input_file}...", err=True)

        # Convert file
        converter = DocumentConverter()
        result = converter.convert(input_file)

        if not result.document:
            click.echo("Error: No document created", err=True)
            sys.exit(1)

        # Create QA generator
        qa_gen = CADQAGenerator(
            llm_model=model,
            critique_enabled=not no_critique,
            api_key=api_key,
        )

        # Generate Q&A pairs
        click.echo(f"Generating {num_pairs} Q&A pairs...", err=True)

        qa_pairs = qa_gen.generate_qa_pairs(result.document, num_pairs=num_pairs)

        # Save to file
        qa_gen.save_qa_pairs(qa_pairs, output)

        click.echo(f"Successfully generated {len(qa_pairs)} Q&A pairs", err=True)
        click.echo(f"Saved to {output}", err=True)

        # Show sample
        if qa_pairs:
            click.echo("\nSample Q&A pair:", err=True)
            sample = qa_pairs[0]
            click.echo(f"Q: {sample.question}", err=True)
            click.echo(f"A: {sample.answer}", err=True)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
def info(input_file: str):
    """Show information about CAD file.

    Displays file format, size, and basic statistics without full conversion.

    Examples:

        \b
        cadling info part.step
        cadling info mesh.stl
    """
    try:
        from cadling.backend.document_converter import DocumentConverter

        file_path = Path(input_file)

        click.echo(f"File: {file_path.name}")
        click.echo(f"Size: {file_path.stat().st_size:,} bytes")

        # Detect format
        suffix = file_path.suffix.lower()
        format_map = {
            ".step": "STEP",
            ".stp": "STEP",
            ".stl": "STL",
            ".brep": "BRep",
            ".iges": "IGES",
            ".igs": "IGES",
            ".dxf": "DXF",
            ".pdf": "PDF Drawing",
        }

        detected_format = format_map.get(suffix, "Unknown")
        click.echo(f"Format: {detected_format}")

        # Try to convert and show stats
        converter = DocumentConverter()
        result = converter.convert(input_file)

        if result.document:
            click.echo(f"Items: {len(result.document.items)}")

            if result.document.topology:
                click.echo(f"Topology nodes: {result.document.topology.num_nodes}")
                click.echo(f"Topology edges: {result.document.topology.num_edges}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
