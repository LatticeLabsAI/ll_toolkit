"""Pytest configuration for ll_brepnet tests.

CRITICAL: This file imports torch FIRST before any other heavy dependencies.
On macOS, OpenMP library conflicts occur when numpy/scipy/sklearn/pythonocc
load before torch. By importing torch here (conftest.py is always loaded first
by pytest), we ensure torch's OpenMP runtime is initialized before any conflict
can occur.

The root cause is that PyTorch bundles its own libomp.dylib while conda-forge
packages use llvm-openmp. Loading both causes:
    OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.

See:
    - https://github.com/pytorch/pytorch/issues/44282
    - https://github.com/pytorch/pytorch/issues/132372
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
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

# =============================================================================
# CRITICAL: Import torch FIRST to prevent OpenMP conflicts on macOS
# =============================================================================
try:
    import torch  # noqa: F401 - imported for side effect (OpenMP init)

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import OCC  # noqa: F401
    import occwl  # noqa: F401

    _HAS_PYTHONOCC = True
except ImportError:
    _HAS_PYTHONOCC = False

try:
    import pytorch_lightning  # noqa: F401

    _HAS_LIGHTNING = True
except ImportError:
    _HAS_LIGHTNING = False

# Now other imports are safe - torch's OpenMP is already initialized.
import pytest

_log = logging.getLogger(__name__)

# Real STEP fixtures bundled with the BRepNet reference (used for genuine,
# behaviour-level acceptance tests rather than smoke imports). These are small
# Fusion 360 Gallery solids; the directory is resolved relative to the repo
# root (three levels up from this tests/ directory: ll_brepnet/tests -> repo).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_STEP_FIXTURE_DIRS = [
    _REPO_ROOT / "resources" / "BRepNet" / "tests" / "test_data",
    _REPO_ROOT / "resources" / "BRepNet" / "example_files" / "step_examples",
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def device() -> str:
    """Return the appropriate torch device string for testing."""
    if not _HAS_TORCH:
        return "cpu"
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture
def temp_output_dir():
    """Yield a temporary directory, cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def skip_if_no_torch():
    """Skip the test if torch is not available."""
    if not _HAS_TORCH:
        pytest.skip("torch not installed")


@pytest.fixture
def skip_if_no_pythonocc():
    """Skip the test if pythonocc-core / occwl are not available."""
    if not _HAS_PYTHONOCC:
        pytest.skip("pythonocc-core / occwl not installed")


@pytest.fixture
def step_fixture_files() -> list[Path]:
    """Return the list of bundled real STEP fixture files (.step/.stp).

    These are real Fusion 360 Gallery solids shipped under
    ``resources/BRepNet`` and are used for behaviour-level acceptance tests.
    """
    files: list[Path] = []
    for d in _STEP_FIXTURE_DIRS:
        if d.is_dir():
            files.extend(sorted(d.glob("*.step")))
            files.extend(sorted(d.glob("*.stp")))
    return files


@pytest.fixture
def one_step_fixture(step_fixture_files) -> Path:
    """Return a single real STEP fixture file, or skip if none are present."""
    if not step_fixture_files:
        pytest.skip("no bundled STEP fixtures found under resources/BRepNet")
    return step_fixture_files[0]


@pytest.fixture(scope="session")
def prepared_dataset(tmp_path_factory):
    """Extract a few real STEP fixtures + synthesize labels + a manifest once.

    Returns ``(npz_dir, manifest_path)``. Labels are the per-face surface-type
    argmax (a real, geometry-derived 7-class target) purely so the dataset /
    model tests have something to train against. Skips without pythonocc/torch.
    """
    if not (_HAS_PYTHONOCC and _HAS_TORCH):
        pytest.skip("pythonocc-core and torch are required")
    import numpy as np

    from ll_brepnet.pipelines.build_dataset_file import build_dataset_file
    from ll_brepnet.pipelines.extract_brepnet_data_from_step import extract_step_files

    files: list[Path] = []
    for d in _STEP_FIXTURE_DIRS:
        if d.is_dir():
            files.extend(sorted(d.glob("*.stp")))
            files.extend(sorted(d.glob("*.step")))
    files = files[:4]
    if len(files) < 2:
        pytest.skip("need at least 2 bundled STEP fixtures")

    out = tmp_path_factory.mktemp("ll_brepnet_ds")
    written = extract_step_files(files, out, num_workers=1)
    for npz in written:
        with np.load(npz) as data:
            labels = data["face_features"][:, :7].argmax(1)
        np.savetxt(out / f"{npz.stem}.seg", labels, fmt="%d")
    build_dataset_file(
        out,
        out / "dataset.json",
        validation_split=0.34,
        test_split=0.0,
        class_names=[f"c{i}" for i in range(7)],
    )
    return out, out / "dataset.json"


# =============================================================================
# Pytest hooks
# =============================================================================


def pytest_configure(config):
    """Register markers and confirm the OpenMP-protective torch import."""
    if _HAS_TORCH:
        _log.debug("torch imported in conftest.py (OpenMP protection active)")
    else:
        _log.warning("torch not available - some tests will be skipped")

    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "requires_torch: marks tests that require torch")
    config.addinivalue_line(
        "markers",
        "requires_pythonocc: marks tests that require pythonocc-core / occwl",
    )
    config.addinivalue_line(
        "markers",
        "requires_lightning: marks tests that require pytorch-lightning",
    )
