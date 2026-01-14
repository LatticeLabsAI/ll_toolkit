"""Fixtures for functional tests."""

import pytest
from pathlib import Path


@pytest.fixture
def test_data_path() -> Path:
    """Root test data directory.

    Returns:
        Path to cadling/data/test_data/
    """
    return Path(__file__).parent.parent.parent / "data" / "test_data"


@pytest.fixture
def test_data_step_path(test_data_path: Path) -> Path:
    """STEP test data directory.

    Args:
        test_data_path: Root test data path

    Returns:
        Path to STEP test files

    Raises:
        pytest.skip: If STEP test data not found
    """
    step_path = test_data_path / "step"
    if not step_path.exists():
        pytest.skip(f"STEP test data not found: {step_path}")
    return step_path


@pytest.fixture
def test_data_stl_path(test_data_path: Path) -> Path:
    """STL test data directory.

    Args:
        test_data_path: Root test data path

    Returns:
        Path to STL test files

    Raises:
        pytest.skip: If STL test data not found
    """
    stl_path = test_data_path / "stl"
    if not stl_path.exists():
        pytest.skip(f"STL test data not found: {stl_path}")
    return stl_path


@pytest.fixture
def functional_output_dir(test_data_path: Path) -> Path:
    """Base directory for functional test outputs.

    Args:
        test_data_path: Root test data path

    Returns:
        Path to functional_runs_outputs directory
    """
    output_dir = test_data_path / "functional_runs_outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir
