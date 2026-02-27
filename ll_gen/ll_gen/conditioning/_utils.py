"""Shared utilities for conditioning encoders."""
from __future__ import annotations

import logging

_log = logging.getLogger(__name__)


def safe_no_grad():
    """Context manager for torch.no_grad() if torch is available.

    Returns:
        torch.no_grad() context manager or a no-op context manager.
    """
    try:
        import torch

        return torch.no_grad()
    except ImportError:

        class NoOp:
            """No-op context manager when torch unavailable."""

            def __enter__(self):
                return self

            def __exit__(self, *args):
                """Exit context, doing nothing."""
                return False

        return NoOp()
