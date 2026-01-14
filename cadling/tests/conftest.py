"""Pytest configuration and fixtures for cadling tests."""

import pytest
from pathlib import Path


@pytest.fixture
def fixtures_path() -> Path:
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def test_data_path() -> Path:
    """Path to test data root directory."""
    return Path(__file__).parent.parent / "data" / "test_data"


@pytest.fixture
def test_data_stl_path(test_data_path: Path) -> Path:
    """Path to STL test data directory."""
    stl_path = test_data_path / "stl"
    if not stl_path.exists():
        pytest.skip(f"STL test data directory not found: {stl_path}")
    return stl_path


@pytest.fixture
def test_data_step_path(test_data_path: Path) -> Path:
    """Path to STEP test data directory."""
    step_path = test_data_path / "step"
    if not step_path.exists():
        pytest.skip(f"STEP test data directory not found: {step_path}")
    return step_path


@pytest.fixture
def temp_cad_file(tmp_path):
    """Create a temporary CAD file for testing."""
    def _create_file(filename: str, content: str) -> Path:
        file_path = tmp_path / filename
        file_path.write_text(content)
        return file_path
    return _create_file
