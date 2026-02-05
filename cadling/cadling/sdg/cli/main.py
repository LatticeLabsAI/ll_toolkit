"""Main CLI entry point for CAD SDG.

This module provides the main CLI application for synthetic
data generation from CAD documents.

Usage:
    python -m cadling.sdg.cli.main --help
    python -m cadling.sdg.cli.main qa sample --help
    python -m cadling.sdg.cli.main qa generate --help
"""

from __future__ import annotations

import logging

import typer

from cadling.sdg.cli.qa import qa_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

_log = logging.getLogger(__name__)

# Main CLI application
app = typer.Typer(
    name="cadling-sdg",
    help="Synthetic Data Generation for CAD documents",
    add_completion=False,
)

# Add QA subcommands
app.add_typer(qa_app, name="qa", help="Q&A generation commands")


@app.command("version")
def version():
    """Show version information."""
    try:
        from cadling import __version__
        typer.echo(f"cadling-sdg version: {__version__}")
    except ImportError:
        typer.echo("cadling-sdg version: unknown")


@app.callback()
def main_callback(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
):
    """CADling Synthetic Data Generation CLI.

    Generate Q&A pairs from CAD documents for LLM training.
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        _log.debug("Debug logging enabled")
    elif verbose:
        logging.getLogger().setLevel(logging.INFO)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
