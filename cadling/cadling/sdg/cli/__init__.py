"""CLI interface for CAD SDG module.

This module provides command-line tools for synthetic data generation
from CAD documents.

Usage:
    python -m cadling.sdg.cli.main [COMMAND] [OPTIONS]

Commands:
    qa sample    - Sample passages from CAD files
    qa generate  - Generate Q&A pairs from passages
    qa critique  - Critique and improve Q&A pairs
    version      - Show version information
"""

from cadling.sdg.cli.main import app

__all__ = ["app"]
