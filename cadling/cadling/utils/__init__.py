"""Utility functions and helpers for CADling.

This module provides common utility functions used across the CADling toolkit,
including file handling, geometric calculations, and data transformations.

Functions:
    compute_file_hash: Compute SHA256 hash of a file
    normalize_path: Normalize file path
    ensure_dir: Ensure directory exists
    format_bytes: Format byte count to human-readable string
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Union

_log = logging.getLogger(__name__)


def compute_file_hash(file_path: Union[Path, str]) -> str:
    """Compute SHA256 hash of a file.

    Args:
        file_path: Path to file

    Returns:
        Hexadecimal hash string
    """
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)

    return sha256.hexdigest()


def normalize_path(path: Union[Path, str]) -> Path:
    """Normalize file path.

    Args:
        path: Path to normalize

    Returns:
        Normalized Path object
    """
    if isinstance(path, str):
        path = Path(path)

    return path.expanduser().resolve()


def ensure_dir(dir_path: Union[Path, str]) -> Path:
    """Ensure directory exists, create if needed.

    Args:
        dir_path: Directory path

    Returns:
        Path object
    """
    path = normalize_path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_bytes(num_bytes: int) -> str:
    """Format byte count to human-readable string.

    Args:
        num_bytes: Number of bytes

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0

    return f"{num_bytes:.1f} PB"


__all__ = [
    "compute_file_hash",
    "normalize_path",
    "ensure_dir",
    "format_bytes",
]
