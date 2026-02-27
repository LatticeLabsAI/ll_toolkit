"""Shared lazy-import helpers for dataset loaders.

Centralises the lazy-import pattern so that each loader does not need
to duplicate the same global + accessor boilerplate.
"""
from __future__ import annotations

import logging

_log = logging.getLogger(__name__)

# Lazy-import caches
_torch = None
_datasets = None
_numpy = None
_geotoken = None


def _get_torch():
    """Lazily import and return the ``torch`` module."""
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _get_datasets():
    """Lazily import and return the ``datasets`` (HuggingFace) module."""
    global _datasets
    if _datasets is None:
        import datasets
        _datasets = datasets
    return _datasets


def _get_numpy():
    """Lazily import and return ``numpy``."""
    global _numpy
    if _numpy is None:
        import numpy as np
        _numpy = np
    return _numpy


def _get_geotoken():
    """Lazily import and return the ``geotoken`` module."""
    global _geotoken
    if _geotoken is None:
        import geotoken
        _geotoken = geotoken
    return _geotoken
