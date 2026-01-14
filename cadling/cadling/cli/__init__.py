"""CADling command-line interface.

This package provides CLI commands for CAD file processing.

Commands:
    convert: Convert CAD file to JSON/Markdown
    chunk: Chunk CAD file for RAG
    generate-qa: Generate Q&A pairs from CAD file
    info: Show CAD file information
"""

from cadling.cli.main import cli

__all__ = ["cli"]
