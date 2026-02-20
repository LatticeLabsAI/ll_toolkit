"""Pytest configuration for ll_stepnet tests.

CRITICAL: This file imports torch FIRST before any other heavy dependencies.
On macOS, OpenMP library conflicts occur when numpy/scipy/sklearn/transformers
load before torch. By importing torch here (conftest.py is always loaded first
by pytest), we ensure torch's OpenMP runtime is initialized before any conflicts
can occur.

The root cause is that PyTorch bundles its own libomp.dylib while conda-forge
packages use llvm-openmp. Loading both causes:
    OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.

See:
    - https://github.com/pytorch/pytorch/issues/44282
    - https://github.com/pytorch/pytorch/issues/132372
    - https://discuss.huggingface.co/t/segfault-during-pytorch-transformers-inference-on-apple-silicon-m4
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

# =============================================================================
# CRITICAL: Set OpenMP environment variables BEFORE any imports
# =============================================================================
# On macOS, set OMP_NUM_THREADS=1 to disable OpenMP parallelism entirely.
# This is a belt-and-suspenders approach that prevents threading-related
# conflicts even if import order gets violated somehow.
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

# =============================================================================
# CRITICAL: Import torch FIRST to prevent OpenMP conflicts on macOS
# =============================================================================
# pytest loads conftest.py BEFORE any test modules. By importing torch here,
# we ensure torch's OpenMP runtime is the one that gets initialized first.
# This prevents the fatal "libomp already initialized" crash.
try:
    import torch  # noqa: F401 - imported for side effect (OpenMP init)
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# Now other imports are safe - torch's OpenMP is already initialized
import pytest

_log = logging.getLogger(__name__)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device() -> str:
    """Get the appropriate device for testing.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'.
    """
    if not _HAS_TORCH:
        return "cpu"

    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for output files.

    Yields:
        Path to the temporary directory, cleaned up after the test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def skip_if_no_torch():
    """Skip test if torch is not available."""
    if not _HAS_TORCH:
        pytest.skip("torch not installed")


@pytest.fixture
def sample_encoder_config():
    """Create a minimal encoder config for testing."""
    class EncoderConfig:
        vocab_size: int = 50000
        token_embed_dim: int = 128  # Smaller for faster tests
        num_transformer_layers: int = 2
        dropout: float = 0.1
    return EncoderConfig()


# =============================================================================
# Pytest Hooks
# =============================================================================

def pytest_configure(config):
    """Called after command line options have been parsed.

    Runs before test collection, ensuring torch is imported early.
    """
    if _HAS_TORCH:
        _log.debug(
            "torch imported successfully in conftest.py (OpenMP protection active)"
        )
    else:
        _log.warning("torch not available - some tests will be skipped")

    # Register custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_torch: marks tests that require torch"
    )
    config.addinivalue_line(
        "markers", "requires_transformers: marks tests that require transformers"
    )
