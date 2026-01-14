"""Tests for constraint detection model.

Tests cover:
- ConstraintDetectionModel initialization and configuration
- Concentric mate detection
- Planar contact detection
- Fastener detection
- Constraint classification
- Integration with real STEP assembly files
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock

from cadling.models.constraint_detection import (
    ConstraintDetectionModel,
    Mate,
    Fastener,
    ConstraintType,
    FastenerType,
)


class TestConstraintDetectionModel:
    """Test ConstraintDetectionModel initialization and basic functionality."""

    def test_model_initialization_default(self):
        """Test model initialization with default parameters."""
        model = ConstraintDetectionModel()

        assert model.concentric_tolerance == 0.1
        assert model.planar_tolerance == 0.01
        assert model.min_contact_area == 1.0

    def test_model_initialization_custom(self):
        """Test model initialization with custom parameters."""
        model = ConstraintDetectionModel(
            concentric_tolerance=0.2,
            planar_tolerance=0.05,
            min_contact_area=5.0
        )

        assert model.concentric_tolerance == 0.2
        assert model.planar_tolerance == 0.05
        assert model.min_contact_area == 5.0

    def test_model_has_required_attributes(self):
        """Test model has required attributes."""
        model = ConstraintDetectionModel()

        assert hasattr(model, 'has_pythonocc')
        assert isinstance(model.has_pythonocc, bool)

    def test_model_call_without_assembly_analysis(self):
        """Test model call without prior assembly analysis."""
        model = ConstraintDetectionModel()

        # Create simple document without assembly_analysis
        class SimpleDoc:
            def __init__(self):
                self.properties = {}
                self.items = []

        doc = SimpleDoc()

        # Should not crash, but won't add results
        model(doc, [])

        # Should not have constraint_detection if no assembly_analysis
        assert "constraint_detection" not in doc.properties


class TestMate:
    """Test Mate data structure."""

    def test_mate_initialization(self):
        """Test creating a Mate object."""
        mate = Mate(
            mate_id="mate_1",
            part1_id="part_1",
            part2_id="part_2",
            constraint_type=ConstraintType.CONCENTRIC,
            parameters={"radius1": 10.0, "radius2": 9.8},
            location=[0, 0, 0],
            direction=[0, 0, 1],
            confidence=0.9
        )

        assert mate.mate_id == "mate_1"
        assert mate.part1_id == "part_1"
        assert mate.part2_id == "part_2"
        assert mate.constraint_type == ConstraintType.CONCENTRIC
        assert mate.parameters["radius1"] == 10.0
        assert mate.confidence == 0.9

    def test_mate_to_dict(self):
        """Test mate serialization to dictionary."""
        mate = Mate(
            mate_id="mate_1",
            part1_id="part_1",
            part2_id="part_2",
            constraint_type=ConstraintType.PLANAR,
            location=[1.0, 2.0, 3.0],
            direction=[0.0, 0.0, 1.0],
        )

        mate_dict = mate.to_dict()

        assert mate_dict["mate_id"] == "mate_1"
        assert mate_dict["constraint_type"] == "planar"
        assert mate_dict["location"] == [1.0, 2.0, 3.0]
        assert mate_dict["direction"] == [0.0, 0.0, 1.0]
        assert "confidence" in mate_dict


class TestFastener:
    """Test Fastener data structure."""

    def test_fastener_initialization(self):
        """Test creating a Fastener object."""
        fastener = Fastener(
            fastener_id="fastener_1",
            fastener_type=FastenerType.BOLT,
            connected_parts=["part_1", "part_2"],
            location=[0, 0, 0],
            axis=[0, 0, 1],
            diameter=8.0,
            length=40.0,
            confidence=0.8
        )

        assert fastener.fastener_id == "fastener_1"
        assert fastener.fastener_type == FastenerType.BOLT
        assert len(fastener.connected_parts) == 2
        assert fastener.diameter == 8.0
        assert fastener.length == 40.0

    def test_fastener_to_dict(self):
        """Test fastener serialization to dictionary."""
        fastener = Fastener(
            fastener_id="fastener_1",
            fastener_type=FastenerType.SCREW,
            diameter=4.0,
            length=20.0,
        )

        fastener_dict = fastener.to_dict()

        assert fastener_dict["fastener_id"] == "fastener_1"
        assert fastener_dict["fastener_type"] == "screw"
        assert fastener_dict["diameter"] == 4.0
        assert fastener_dict["length"] == 20.0


class TestConstraintType:
    """Test ConstraintType enum."""

    def test_constraint_types_exist(self):
        """Test all constraint types are defined."""
        assert ConstraintType.CONCENTRIC.value == "concentric"
        assert ConstraintType.PLANAR.value == "planar"
        assert ConstraintType.COINCIDENT.value == "coincident"
        assert ConstraintType.PARALLEL.value == "parallel"
        assert ConstraintType.PERPENDICULAR.value == "perpendicular"
        assert ConstraintType.TANGENT.value == "tangent"


class TestFastenerType:
    """Test FastenerType enum."""

    def test_fastener_types_exist(self):
        """Test all fastener types are defined."""
        assert FastenerType.BOLT.value == "bolt"
        assert FastenerType.SCREW.value == "screw"
        assert FastenerType.NUT.value == "nut"
        assert FastenerType.WASHER.value == "washer"
        assert FastenerType.PIN.value == "pin"
        assert FastenerType.UNKNOWN.value == "unknown"


class TestConcentricMateDetection:
    """Test concentric mate detection."""

    def test_detect_concentric_mates_no_parts(self):
        """Test concentric detection with no parts."""
        model = ConstraintDetectionModel()

        class SimpleDoc:
            def __init__(self):
                self.properties = {}

        doc = SimpleDoc()
        mates = model.detect_concentric_mates(doc, [])

        assert isinstance(mates, list)
        assert len(mates) == 0

    def test_are_concentric_parallel_axes(self):
        """Test _are_concentric with parallel axes."""
        model = ConstraintDetectionModel(concentric_tolerance=0.1)

        # Two cylinders with parallel axes
        cyl1 = {
            "radius": 10.0,
            "axis": [0, 0, 1],
            "center": [0, 0, 0]
        }
        cyl2 = {
            "radius": 9.8,
            "axis": [0, 0, 1],  # Parallel
            "center": [0, 0, 10]  # Offset along axis
        }

        result = model._are_concentric(cyl1, cyl2)
        assert result is True

    def test_are_concentric_non_parallel_axes(self):
        """Test _are_concentric with non-parallel axes."""
        model = ConstraintDetectionModel(concentric_tolerance=0.1)

        # Two cylinders with non-parallel axes
        cyl1 = {
            "radius": 10.0,
            "axis": [0, 0, 1],
            "center": [0, 0, 0]
        }
        cyl2 = {
            "radius": 10.0,
            "axis": [1, 0, 0],  # Perpendicular
            "center": [0, 0, 0]
        }

        result = model._are_concentric(cyl1, cyl2)
        assert result is False

    def test_are_concentric_offset_centers(self):
        """Test _are_concentric with radially offset centers."""
        model = ConstraintDetectionModel(concentric_tolerance=0.1)

        # Two cylinders with offset centers
        cyl1 = {
            "radius": 10.0,
            "axis": [0, 0, 1],
            "center": [0, 0, 0]
        }
        cyl2 = {
            "radius": 10.0,
            "axis": [0, 0, 1],
            "center": [5, 0, 0]  # Radial offset > tolerance
        }

        result = model._are_concentric(cyl1, cyl2)
        assert result is False


class TestPlanarContactDetection:
    """Test planar contact detection."""

    def test_detect_planar_contacts_no_parts(self):
        """Test planar detection with no parts."""
        model = ConstraintDetectionModel()

        class SimpleDoc:
            def __init__(self):
                self.properties = {}

        doc = SimpleDoc()
        mates = model.detect_planar_contacts(doc, [])

        assert isinstance(mates, list)
        assert len(mates) == 0

    def test_are_planar_contact_facing(self):
        """Test _are_planar_contact with facing planes."""
        model = ConstraintDetectionModel(
            planar_tolerance=0.01,
            min_contact_area=1.0
        )

        # Two planes facing each other
        plane1 = {
            "normal": [0, 0, 1],
            "point": [0, 0, 0],
            "area": 100.0
        }
        plane2 = {
            "normal": [0, 0, -1],  # Opposite direction
            "point": [0, 0, 0.005],  # Small gap
            "area": 100.0
        }

        result = model._are_planar_contact(plane1, plane2)
        assert result is True

    def test_are_planar_contact_parallel_not_facing(self):
        """Test _are_planar_contact with parallel but not facing planes."""
        model = ConstraintDetectionModel(
            planar_tolerance=0.01,
            min_contact_area=1.0
        )

        # Two planes parallel but facing same direction
        plane1 = {
            "normal": [0, 0, 1],
            "point": [0, 0, 0],
            "area": 100.0
        }
        plane2 = {
            "normal": [0, 0, 1],  # Same direction
            "point": [0, 0, 1],
            "area": 100.0
        }

        result = model._are_planar_contact(plane1, plane2)
        assert result is False

    def test_are_planar_contact_large_gap(self):
        """Test _are_planar_contact with large gap."""
        model = ConstraintDetectionModel(
            planar_tolerance=0.01,
            min_contact_area=1.0
        )

        # Two planes with large gap
        plane1 = {
            "normal": [0, 0, 1],
            "point": [0, 0, 0],
            "area": 100.0
        }
        plane2 = {
            "normal": [0, 0, -1],
            "point": [0, 0, 10],  # Large gap
            "area": 100.0
        }

        result = model._are_planar_contact(plane1, plane2)
        assert result is False

    def test_are_planar_contact_small_area(self):
        """Test _are_planar_contact with insufficient contact area."""
        model = ConstraintDetectionModel(
            planar_tolerance=0.01,
            min_contact_area=50.0  # Require at least 50 mm²
        )

        # Two planes with small area
        plane1 = {
            "normal": [0, 0, 1],
            "point": [0, 0, 0],
            "area": 10.0  # Too small
        }
        plane2 = {
            "normal": [0, 0, -1],
            "point": [0, 0, 0.005],
            "area": 10.0
        }

        result = model._are_planar_contact(plane1, plane2)
        assert result is False


class TestFastenerDetection:
    """Test fastener detection."""

    def test_detect_fasteners_no_parts(self):
        """Test fastener detection with no parts."""
        model = ConstraintDetectionModel()

        class SimpleDoc:
            def __init__(self):
                self.properties = {}

        doc = SimpleDoc()
        fasteners = model.detect_fasteners(doc, [])

        assert isinstance(fasteners, list)
        assert len(fasteners) == 0

    def test_classify_fastener_bolt(self):
        """Test fastener classification as bolt."""
        model = ConstraintDetectionModel()

        # Large diameter, long length
        dims = [5.0, 5.0, 30.0]  # Width, width, length
        fastener_type = model._classify_fastener(None, dims)

        assert fastener_type == FastenerType.BOLT

    def test_classify_fastener_screw(self):
        """Test fastener classification as screw."""
        model = ConstraintDetectionModel()

        # Small diameter, long length
        dims = [2.0, 2.0, 15.0]
        fastener_type = model._classify_fastener(None, dims)

        assert fastener_type == FastenerType.SCREW

    def test_classify_fastener_pin(self):
        """Test fastener classification as pin."""
        model = ConstraintDetectionModel()

        # Small diameter, moderate length
        dims = [3.0, 3.0, 10.0]
        fastener_type = model._classify_fastener(None, dims)

        assert fastener_type == FastenerType.PIN

    def test_classify_fastener_washer(self):
        """Test fastener classification as washer."""
        model = ConstraintDetectionModel()

        # Wide, thin
        dims = [8.0, 8.0, 2.0]
        fastener_type = model._classify_fastener(None, dims)

        assert fastener_type == FastenerType.WASHER

    def test_classify_fastener_nut(self):
        """Test fastener classification as nut."""
        model = ConstraintDetectionModel()

        # Moderate diameter, short height
        dims = [6.0, 6.0, 8.0]
        fastener_type = model._classify_fastener(None, dims)

        assert fastener_type == FastenerType.NUT


class TestIntegration:
    """Integration tests with synthetic and real data."""

    def test_constraint_detection_with_synthetic_parts(self):
        """Test constraint detection with synthetic part data."""
        model = ConstraintDetectionModel()

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
                    "centroid": [i * 10, 0, 0],
                    "bounding_box": {
                        "min_x": i * 10 - 5,
                        "max_x": i * 10 + 5,
                        "min_y": -5,
                        "max_y": 5,
                        "min_z": -5,
                        "max_z": 5,
                    }
                }
            }
            item._occ_shape = None  # No actual OCC shape
            doc.items.append(item)

        # Run model (won't find mates without real OCC shapes)
        model(doc, doc.items)

        # Should have constraint_detection in properties
        if model.has_pythonocc:
            assert "constraint_detection" in doc.properties
            result = doc.properties["constraint_detection"]
            assert "mates" in result
            assert "fasteners" in result

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.parent.parent / "data" / "test_data" / "step",
        reason="Test data directory not found"
    )
    def test_constraint_detection_with_real_step_file(self):
        """Test constraint detection with real STEP assembly file."""
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

        # Run constraint detection
        constraint_model = ConstraintDetectionModel()
        constraint_model(doc, doc.items)

        # Verify results
        if "constraint_detection" in doc.properties:
            result = doc.properties["constraint_detection"]
            assert "mates" in result
            assert "fasteners" in result
            assert "num_mates" in result
            assert isinstance(result["num_mates"], int)
            assert result["num_mates"] >= 0
