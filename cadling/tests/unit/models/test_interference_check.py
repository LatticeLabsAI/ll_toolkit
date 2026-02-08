"""Tests for interference check model.

Tests cover:
- InterferenceCheckModel initialization and configuration
- Interference detection (collision checking)
- Clearance computation
- Containment detection
- Severity calculation
- Integration with real STEP assembly files
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock

from cadling.models.interference_check import (
    InterferenceCheckModel,
    Interference,
    Clearance,
)


class TestInterferenceCheckModel:
    """Test InterferenceCheckModel initialization and basic functionality."""

    def test_model_initialization_default(self):
        """Test model initialization with default parameters."""
        model = InterferenceCheckModel()

        assert model.min_clearance == 0.1
        assert model.check_containment is True
        assert model.tolerance == 1e-6

    def test_model_initialization_custom(self):
        """Test model initialization with custom parameters."""
        model = InterferenceCheckModel(
            min_clearance=0.5,
            check_containment=False,
            tolerance=1e-5
        )

        assert model.min_clearance == 0.5
        assert model.check_containment is False
        assert model.tolerance == 1e-5

    def test_model_has_required_attributes(self):
        """Test model has required attributes."""
        model = InterferenceCheckModel()

        assert hasattr(model, 'has_pythonocc')
        assert isinstance(model.has_pythonocc, bool)

    def test_model_call_without_assembly_analysis(self):
        """Test model call without prior assembly analysis."""
        model = InterferenceCheckModel()

        # Create simple document without assembly_analysis
        class SimpleDoc:
            def __init__(self):
                self.properties = {}
                self.items = []

        doc = SimpleDoc()

        # Should not crash, will add empty results
        model(doc, [])

        # Should have interference_check even without assembly_analysis (empty results)
        assert "interference_check" in doc.properties
        assert doc.properties["interference_check"]["has_interferences"] is False


class TestInterference:
    """Test Interference data structure."""

    def test_interference_initialization(self):
        """Test creating an Interference object."""
        interference = Interference(
            interference_id="inter_1",
            part1_id="part_1",
            part2_id="part_2",
            interference_type="collision",
            volume=150.0,
            location=[10, 20, 30],
            severity=0.7,
            confidence=0.9
        )

        assert interference.interference_id == "inter_1"
        assert interference.part1_id == "part_1"
        assert interference.part2_id == "part_2"
        assert interference.interference_type == "collision"
        assert interference.volume == 150.0
        assert interference.severity == 0.7
        assert interference.confidence == 0.9

    def test_interference_to_dict(self):
        """Test interference serialization to dictionary."""
        interference = Interference(
            interference_id="inter_1",
            part1_id="part_1",
            part2_id="part_2",
            volume=100.0,
            location=[1.0, 2.0, 3.0],
        )

        inter_dict = interference.to_dict()

        assert inter_dict["interference_id"] == "inter_1"
        assert inter_dict["part1_id"] == "part_1"
        assert inter_dict["part2_id"] == "part_2"
        assert inter_dict["volume"] == 100.0
        assert inter_dict["location"] == [1.0, 2.0, 3.0]
        assert "severity" in inter_dict
        assert "confidence" in inter_dict


class TestClearance:
    """Test Clearance data structure."""

    def test_clearance_initialization(self):
        """Test creating a Clearance object."""
        clearance = Clearance(
            clearance_id="clear_1",
            part1_id="part_1",
            part2_id="part_2",
            distance=2.5,
            point1=[0, 0, 0],
            point2=[2.5, 0, 0],
            direction=[1, 0, 0],
            is_sufficient=True
        )

        assert clearance.clearance_id == "clear_1"
        assert clearance.part1_id == "part_1"
        assert clearance.part2_id == "part_2"
        assert clearance.distance == 2.5
        assert clearance.is_sufficient is True

    def test_clearance_to_dict(self):
        """Test clearance serialization to dictionary."""
        clearance = Clearance(
            clearance_id="clear_1",
            part1_id="part_1",
            part2_id="part_2",
            distance=1.5,
            point1=[0.0, 0.0, 0.0],
            point2=[1.5, 0.0, 0.0],
            is_sufficient=True
        )

        clear_dict = clearance.to_dict()

        assert clear_dict["clearance_id"] == "clear_1"
        assert clear_dict["distance"] == 1.5
        assert clear_dict["is_sufficient"] is True
        assert clear_dict["point1"] == [0.0, 0.0, 0.0]
        assert clear_dict["point2"] == [1.5, 0.0, 0.0]

    def test_clearance_insufficient(self):
        """Test clearance with insufficient distance."""
        clearance = Clearance(
            clearance_id="clear_1",
            part1_id="part_1",
            part2_id="part_2",
            distance=0.05,
            is_sufficient=False
        )

        assert clearance.is_sufficient is False


class TestInterferenceDetection:
    """Test interference detection functionality."""

    def test_check_interferences_no_parts(self):
        """Test interference check with no parts."""
        model = InterferenceCheckModel()

        class SimpleDoc:
            def __init__(self):
                self.properties = {}

        doc = SimpleDoc()
        interferences = model.check_interferences(doc, [])

        assert isinstance(interferences, list)
        assert len(interferences) == 0

    def test_check_interferences_single_part(self):
        """Test interference check with single part."""
        model = InterferenceCheckModel()

        class SimpleDoc:
            def __init__(self):
                self.properties = {}

        doc = SimpleDoc()

        # Create one mock part
        item = Mock()
        item.item_type = "brep_solid"
        item._occ_shape = None

        interferences = model.check_interferences(doc, [item])

        # Should not find any interferences with single part
        assert len(interferences) == 0

    def test_compute_severity_known_volumes(self):
        """Test severity computation with known part volumes."""
        model = InterferenceCheckModel()

        # Create mock items with geometry
        item1 = Mock()
        item1.properties = {
            "geometry_analysis": {"volume": 1000.0}
        }

        item2 = Mock()
        item2.properties = {
            "geometry_analysis": {"volume": 500.0}
        }

        # Intersection of 50 mm³ with min volume 500 mm³
        severity = model._compute_severity(50.0, item1, item2)

        assert 0.0 <= severity <= 1.0
        assert abs(severity - 0.1) < 0.01  # 50/500 = 0.1

    def test_compute_severity_unknown_volumes(self):
        """Test severity computation with unknown volumes."""
        model = InterferenceCheckModel()

        # Create mock items without geometry
        item1 = Mock()
        item1.properties = {}

        item2 = Mock()
        item2.properties = {}

        # Should return default severity
        severity = model._compute_severity(50.0, item1, item2)

        assert severity == 0.5  # Default


class TestClearanceComputation:
    """Test clearance computation functionality."""

    def test_compute_clearances_no_shapes(self):
        """Test clearance computation with no OCC shapes."""
        model = InterferenceCheckModel()

        # Create mock items without shapes
        item1 = Mock()
        item1.item_type = "brep_solid"
        item1._occ_shape = None

        item2 = Mock()
        item2.item_type = "brep_solid"
        item2._occ_shape = None

        clearance = model.compute_clearances(item1, item2)

        # Should return None if no shapes
        assert clearance is None


class TestContainmentDetection:
    """Test containment detection functionality."""

    def test_detect_containment_no_geometry(self):
        """Test containment detection without geometry properties."""
        model = InterferenceCheckModel()

        # Create mock items without geometry
        item1 = Mock()
        item1.properties = {}

        item2 = Mock()
        item2.properties = {}

        result = model.detect_containment(item1, item2)

        # Should return False if no geometry
        assert result is False

    def test_detect_containment_contained(self):
        """Test containment detection with contained part."""
        model = InterferenceCheckModel()

        # Create mock items with bounding boxes
        item1 = Mock()  # Outer part
        item1.properties = {
            "geometry_analysis": {
                "bounding_box": {
                    "min_x": 0, "max_x": 100,
                    "min_y": 0, "max_y": 100,
                    "min_z": 0, "max_z": 100,
                }
            }
        }

        item2 = Mock()  # Inner part
        item2.properties = {
            "geometry_analysis": {
                "bounding_box": {
                    "min_x": 25, "max_x": 75,
                    "min_y": 25, "max_y": 75,
                    "min_z": 25, "max_z": 75,
                }
            }
        }

        result = model.detect_containment(item1, item2)

        # item2 should be contained within item1
        assert result is True

    def test_detect_containment_not_contained(self):
        """Test containment detection with non-contained part."""
        model = InterferenceCheckModel()

        # Create mock items with non-overlapping bounding boxes
        item1 = Mock()
        item1.properties = {
            "geometry_analysis": {
                "bounding_box": {
                    "min_x": 0, "max_x": 50,
                    "min_y": 0, "max_y": 50,
                    "min_z": 0, "max_z": 50,
                }
            }
        }

        item2 = Mock()
        item2.properties = {
            "geometry_analysis": {
                "bounding_box": {
                    "min_x": 60, "max_x": 100,
                    "min_y": 60, "max_y": 100,
                    "min_z": 60, "max_z": 100,
                }
            }
        }

        result = model.detect_containment(item1, item2)

        # item2 should NOT be contained within item1
        assert result is False

    def test_detect_containment_partial_overlap(self):
        """Test containment detection with partial overlap."""
        model = InterferenceCheckModel()

        # Create mock items with partial overlap
        item1 = Mock()
        item1.properties = {
            "geometry_analysis": {
                "bounding_box": {
                    "min_x": 0, "max_x": 50,
                    "min_y": 0, "max_y": 50,
                    "min_z": 0, "max_z": 50,
                }
            }
        }

        item2 = Mock()
        item2.properties = {
            "geometry_analysis": {
                "bounding_box": {
                    "min_x": 25, "max_x": 75,  # Extends beyond
                    "min_y": 25, "max_y": 75,
                    "min_z": 25, "max_z": 75,
                }
            }
        }

        result = model.detect_containment(item1, item2)

        # item2 is NOT fully contained (extends beyond)
        assert result is False


class TestIntegration:
    """Integration tests with synthetic and real data."""

    def test_interference_check_with_synthetic_parts(self):
        """Test interference check with synthetic part data."""
        model = InterferenceCheckModel(min_clearance=1.0)

        # Create simple document with assembly analysis
        class SimpleDoc:
            def __init__(self):
                self.properties = {
                    "assembly_analysis": {
                        "graph": {
                            "nodes": {
                                "part_0": {"id": "part_0", "type": "part"},
                                "part_1": {"id": "part_1", "type": "part"}
                            }
                        }
                    }
                }
                self.items = []

        doc = SimpleDoc()

        # Create mock items with geometry
        for i in range(2):
            item = Mock()
            item.item_type = "brep_solid"
            item.properties = {
                "geometry_analysis": {
                    "volume": 1000.0,
                    "bounding_box": {
                        "min_x": i * 20 - 5,
                        "max_x": i * 20 + 5,
                        "min_y": -5,
                        "max_y": 5,
                        "min_z": -5,
                        "max_z": 5,
                    }
                }
            }
            item._occ_shape = None  # No actual OCC shape
            doc.items.append(item)

        # Run model (won't find interferences without real OCC shapes)
        model(doc, doc.items)

        # Should have interference_check in properties
        if model.has_pythonocc:
            assert "interference_check" in doc.properties
            result = doc.properties["interference_check"]
            assert "interferences" in result
            assert "clearances" in result
            assert "num_interferences" in result

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.parent.parent / "data" / "test_data" / "step",
        reason="Test data directory not found"
    )
    def test_interference_check_with_real_step_file(self):
        """Test interference check with real STEP assembly file."""
        pytest.importorskip("OCC")

        from cadling.backend.document_converter import DocumentConverter
        from cadling.models.geometry_analysis import GeometryAnalysisModel
        from cadling.models.assembly_analysis import AssemblyAnalysisModel

        # Find STEP assembly files
        test_data_dir = (
            Path(__file__).parent.parent.parent.parent / "data" / "test_data" / "step"
        )
        step_files = list(test_data_dir.glob("*.step")) + list(test_data_dir.glob("*.stp"))

        if not step_files:
            pytest.skip("No STEP files found in test data")

        # Use first file
        step_file = step_files[0]

        # Load using DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(str(step_file))
        doc = result.document

        # Run prerequisite models
        geom_model = GeometryAnalysisModel()
        geom_model(doc, doc.items)

        asm_model = AssemblyAnalysisModel()
        asm_model(doc, doc.items)

        # Run interference check
        interference_model = InterferenceCheckModel(min_clearance=0.5)
        interference_model(doc, doc.items)

        # Verify results
        if "interference_check" in doc.properties:
            result = doc.properties["interference_check"]
            assert "interferences" in result
            assert "clearances" in result
            assert "num_interferences" in result
            assert isinstance(result["num_interferences"], int)
            assert result["num_interferences"] >= 0
            assert "has_interferences" in result
            assert isinstance(result["has_interferences"], bool)
