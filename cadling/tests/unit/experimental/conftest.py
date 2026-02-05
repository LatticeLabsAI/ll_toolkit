"""Shared fixtures for experimental tests.

Provides realistic fixtures that match actual CADling data structures.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock

from cadling.datamodel.base_models import (
    CADInputDocument,
    ConversionResult,
    InputFormat,
    CADlingDocument,
)


@pytest.fixture
def mock_backend():
    """Create a mock backend that returns a proper CADlingDocument."""
    backend = Mock()

    # Create a realistic document structure
    doc = Mock(spec=CADlingDocument)
    doc.name = "test_part.step"
    doc.properties = {}
    doc.topology = {
        "num_faces": 10,
        "num_edges": 20,
        "faces": [
            {"id": "face_0", "geometry_type": "cylinder"},
            {"id": "face_1", "geometry_type": "plane"},
        ],
    }

    # Create mock items with rendered images
    item = Mock()
    item.self_ref = "part_1"
    item.properties = {
        "rendered_images": {
            "front": Mock(),  # Mock PIL Image
            "top": Mock(),
            "isometric": Mock(),
        }
    }
    doc.items = [item]

    backend.convert = Mock(return_value=doc)
    return backend


@pytest.fixture
def mock_input_doc(mock_backend):
    """Create a proper CADInputDocument with backend."""
    input_doc = CADInputDocument(
        file=Path("/tmp/test.step"),
        format=InputFormat.STEP,
        document_hash="abc123",
    )
    # Attach backend (this is how pipelines get access to conversion)
    input_doc._backend = mock_backend
    return input_doc


@pytest.fixture
def mock_conversion_result(mock_input_doc):
    """Create a proper ConversionResult with all required fields."""
    return ConversionResult(input=mock_input_doc)


@pytest.fixture
def mock_converted_doc():
    """Create a mock converted CAD document."""
    doc = Mock()
    doc.properties = {}
    doc.topology = {
        "num_faces": 10,
        "num_edges": 20,
        "faces": [
            {"id": "face_0", "geometry_type": "cylinder"},
            {"id": "face_1", "geometry_type": "plane"},
        ],
    }

    item = Mock()
    item.self_ref = "part_1"
    item.properties = {
        "bounding_box": {"x": 100, "y": 50, "z": 30},
        "volume": 150000,
        "mass": 1.2,
    }

    doc.items = [item]
    return doc
