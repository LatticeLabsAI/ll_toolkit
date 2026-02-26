"""
Command-line interface for cadling.

Provides commands for:
- convert: Convert CAD files to CADlingDocument
- chunk: Split CAD documents into chunks for RAG
- export: Export documents to various formats
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)


def _configure_logging():
    """Configure logging for CLI use."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def convert_command(args):
    """Convert CAD file to CADlingDocument."""
    from cadling.backend.document_converter import DocumentConverter
    from cadling.datamodel.base_models import ConversionStatus
    from cadling.datamodel.pipeline_options import PipelineOptions

    input_path = Path(args.input)

    if not input_path.exists():
        _log.error(f"Input file not found: {input_path}")
        return 1

    _log.info(f"Converting {input_path}")

    try:
        # Create converter
        options = PipelineOptions(
            do_topology_analysis=not args.no_topology,
            device=args.device,
        )

        converter = DocumentConverter()

        # Convert with pipeline options
        result = converter.convert(input_path, pipeline_options=options)

        # Check status
        if result.status == ConversionStatus.FAILURE:
            _log.error("Conversion failed")
            for error in result.errors:
                _log.error(f"  {error.component}: {error.error_message}")
            return 1

        if result.status == ConversionStatus.PARTIAL:
            _log.warning(f"Conversion completed with warnings")
            for error in result.errors:
                _log.warning(f"  {error.component}: {error.error_message}")

        document = result.document

        _log.info(
            f"Conversion successful: {len(document.items)} items, "
            f"{document.topology.num_nodes if document.topology else 0} topology nodes"
        )

        # Export based on format
        output_path = Path(args.output) if args.output else input_path.with_suffix(".json")

        if args.format == "json":
            output_data = document.export_to_json()
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            _log.info(f"Exported to JSON: {output_path}")

        elif args.format == "markdown":
            markdown_text = document.export_to_markdown()
            with open(output_path, "w") as f:
                f.write(markdown_text)
            _log.info(f"Exported to Markdown: {output_path}")

        elif args.format == "text":
            # Simple text export
            with open(output_path, "w") as f:
                f.write(f"CAD Document: {document.name}\n")
                f.write(f"Items: {len(document.items)}\n\n")
                for i, item in enumerate(document.items):
                    f.write(f"Item {i + 1}:\n")
                    if item.label:
                        f.write(f"  Label: {item.label.get('text', 'N/A')}\n")
                    f.write(f"  Type: {item.item_type}\n")
                    if item.text:
                        f.write(f"  Text: {item.text}\n")
                    f.write("\n")
            _log.info(f"Exported to text: {output_path}")

        return 0

    except Exception as e:
        _log.exception(f"Conversion failed with exception: {e}")
        return 1


def chunk_command(args):
    """Chunk CAD document for RAG."""
    from cadling.backend.document_converter import DocumentConverter
    from cadling.chunking import (
        SequentialChunker,
        EntityTypeChunker,
        SpatialChunker,
    )

    input_path = Path(args.input)

    if not input_path.exists():
        _log.error(f"Input file not found: {input_path}")
        return 1

    _log.info(f"Chunking {input_path}")

    try:
        # First convert the document
        converter = DocumentConverter()
        result = converter.convert(input_path)

        if result.status.value == "failure":
            _log.error(f"Conversion failed, cannot chunk")
            return 1

        document = result.document

        # Select chunker based on strategy
        if args.strategy == "sequential":
            chunker = SequentialChunker(
                chunk_size=args.chunk_size,
                chunk_overlap=args.overlap,
            )
        elif args.strategy == "entity_type":
            chunker = EntityTypeChunker(
                chunk_size=args.chunk_size,
                chunk_overlap=args.overlap,
            )
        elif args.strategy == "spatial":
            chunker = SpatialChunker(
                chunk_size=args.chunk_size,
                chunk_overlap=args.overlap,
            )
        else:
            _log.error(f"Unknown chunking strategy: {args.strategy}")
            return 1

        # Chunk the document
        chunks = chunker.chunk(document)

        _log.info(f"Created {len(chunks)} chunks")

        # Export chunks
        output_path = Path(args.output) if args.output else input_path.with_suffix(".chunks.json")

        chunks_data = []
        for chunk in chunks:
            chunk_dict = chunk.to_dict()
            if args.include_text:
                chunk_dict["text"] = chunk.text
            chunks_data.append(chunk_dict)

        with open(output_path, "w") as f:
            json.dump(
                {
                    "source_document": document.name,
                    "num_chunks": len(chunks),
                    "chunk_size": args.chunk_size,
                    "strategy": args.strategy,
                    "chunks": chunks_data,
                },
                f,
                indent=2,
            )

        _log.info(f"Exported chunks to: {output_path}")

        # Print summary
        for chunk in chunks[:5]:  # First 5 chunks
            _log.info(f"  {chunk.chunk_id}: {len(chunk.items)} items")
        if len(chunks) > 5:
            _log.info(f"  ... and {len(chunks) - 5} more chunks")

        return 0

    except Exception as e:
        _log.exception(f"Chunking failed with exception: {e}")
        return 1


def export_command(args):
    """Export CAD document to various formats.

    Maps the export command's positional output argument to the convert
    command's optional --output flag, then delegates to convert_command.
    """
    _log.info(f"Export command: {args.input} -> {args.output}")

    # Ensure the convert command flags exist with defaults if not set
    if not hasattr(args, "no_topology"):
        args.no_topology = False
    if not hasattr(args, "device"):
        args.device = "cpu"

    return convert_command(args)


def main():
    """Main CLI entry point."""
    _configure_logging()
    parser = argparse.ArgumentParser(
        description="cadling - CAD document processing toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert STEP file to JSON
  cadling convert input.step -o output.json

  # Convert with markdown output
  cadling convert input.stl -f markdown -o output.md

  # Chunk document for RAG
  cadling chunk input.step --strategy entity_type --chunk-size 50

  # Chunk with spatial strategy
  cadling chunk input.stl --strategy spatial --chunk-size 100 --overlap 10
        """,
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Convert command
    convert_parser = subparsers.add_parser(
        "convert", help="Convert CAD file to CADlingDocument"
    )
    convert_parser.add_argument("input", type=str, help="Input CAD file path")
    convert_parser.add_argument(
        "-o", "--output", type=str, help="Output file path (default: input.json)"
    )
    convert_parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=["json", "markdown", "text"],
        default="json",
        help="Output format (default: json)",
    )
    convert_parser.add_argument(
        "--no-topology",
        action="store_true",
        help="Disable topology analysis",
    )
    convert_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for processing (default: cpu)",
    )

    # Chunk command
    chunk_parser = subparsers.add_parser(
        "chunk", help="Chunk CAD document for RAG"
    )
    chunk_parser.add_argument("input", type=str, help="Input CAD file path")
    chunk_parser.add_argument(
        "-o", "--output", type=str, help="Output file path (default: input.chunks.json)"
    )
    chunk_parser.add_argument(
        "--strategy",
        type=str,
        choices=["sequential", "entity_type", "spatial"],
        default="entity_type",
        help="Chunking strategy (default: entity_type)",
    )
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Target chunk size (number of items, default: 100)",
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=10,
        help="Chunk overlap (number of items, default: 10)",
    )
    chunk_parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include full text in chunk output",
    )

    # Export command
    export_parser = subparsers.add_parser(
        "export", help="Export CAD document to various formats"
    )
    export_parser.add_argument("input", type=str, help="Input CAD file path")
    export_parser.add_argument("output", type=str, help="Output file path")
    export_parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=["json", "markdown", "text"],
        default="json",
        help="Output format",
    )

    # Parse arguments
    args = parser.parse_args()

    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # Execute command
    if not args.command:
        parser.print_help()
        return 0

    if args.command == "convert":
        return convert_command(args)
    elif args.command == "chunk":
        return chunk_command(args)
    elif args.command == "export":
        return export_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
