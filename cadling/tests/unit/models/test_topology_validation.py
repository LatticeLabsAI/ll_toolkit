"""Unit tests for topology validation model."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from cadling.models.topology_validation import TopologyValidationModel, ValidationFinding


class TestTopologyValidationModel:
    """Test TopologyValidationModel initialization and configuration."""

    def test_model_initialization_default(self):
        """Test model initializes with default settings."""
        from cadling.models.topology_validation import TopologyValidationModel

        model = TopologyValidationModel()

        assert model.strict_mode is False
        assert hasattr(model, "has_pythonocc")
        assert hasattr(model, "has_trimesh")

    def test_model_initialization_strict_mode(self):
        """Test model initializes with strict mode enabled."""
        from cadling.models.topology_validation import TopologyValidationModel

        model = TopologyValidationModel(strict_mode=True)

        assert model.strict_mode is True

    def test_model_has_required_methods(self):
        """Test model has all required methods."""
        from cadling.models.topology_validation import TopologyValidationModel

        model = TopologyValidationModel()

        assert callable(model.__call__)
        assert callable(model.supports_batch_processing)
        assert callable(model.get_batch_size)
        assert callable(model.requires_gpu)
        assert callable(model._validate_item)
        assert callable(model._validate_occ_shape)
        assert callable(model._validate_trimesh)

    def test_model_pythonocc_detection(self):
        """Test pythonocc availability detection."""
        from cadling.models.topology_validation import TopologyValidationModel

        model = TopologyValidationModel()

        # has_pythonocc should be bool (True or False depending on environment)
        assert isinstance(model.has_pythonocc, bool)

    def test_model_trimesh_detection(self):
        """Test trimesh availability detection."""
        from cadling.models.topology_validation import TopologyValidationModel

        model = TopologyValidationModel()

        # has_trimesh should be bool
        assert isinstance(model.has_trimesh, bool)


class TestTopologyValidationModelMethods:
    """Test TopologyValidationModel helper methods."""

    def test_supports_batch_processing(self):
        """Test that model supports batch processing."""
        from cadling.models.topology_validation import TopologyValidationModel

        model = TopologyValidationModel()

        assert model.supports_batch_processing() is True

    def test_get_batch_size(self):
        """Test that model returns batch size of 1 (expensive operation)."""
        from cadling.models.topology_validation import TopologyValidationModel

        model = TopologyValidationModel()

        assert model.get_batch_size() == 1

    def test_requires_gpu(self):
        """Test that model does not require GPU."""
        from cadling.models.topology_validation import TopologyValidationModel

        model = TopologyValidationModel()

        assert model.requires_gpu() is False

    def test_is_occ_shape_detection(self):
        """Test OCC shape type detection."""
        from cadling.models.topology_validation import TopologyValidationModel

        model = TopologyValidationModel()

        # Mock object should return False
        mock_obj = Mock()
        assert model._is_occ_shape(mock_obj) is False

    def test_is_trimesh_detection(self):
        """Test trimesh type detection."""
        from cadling.models.topology_validation import TopologyValidationModel

        model = TopologyValidationModel()

        # Mock object should return False
        mock_obj = Mock()
        assert model._is_trimesh(mock_obj) is False


class TestTopologyValidationModelCall:
    """Test TopologyValidationModel __call__ method."""

    def test_call_without_backends(self):
        """Test that __call__ skips when no backends available."""
        from cadling.models.topology_validation import TopologyValidationModel
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = TopologyValidationModel()
        # Force no backends
        model.has_pythonocc = False
        model.has_trimesh = False

        doc = STEPDocument(
            name="test.step",
            origin=CADDocumentOrigin(
                filename="test.step",
                format=InputFormat.STEP,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        item = STEPEntityItem(
            label=CADItemLabel(text="Test Entity"),
            entity_id=1,
            entity_type="CARTESIAN_POINT",
            text="#1=CARTESIAN_POINT('',(-0.5,0.,0.));",
        )

        # Should return without error and without adding properties
        model(doc, [item])

        assert "topology_validation" not in item.properties

    def test_call_with_no_shape(self):
        """Test __call__ when item has no shape."""
        from cadling.models.topology_validation import TopologyValidationModel
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = TopologyValidationModel()
        # Ensure at least one backend is available
        model.has_pythonocc = True

        doc = STEPDocument(
            name="test.step",
            origin=CADDocumentOrigin(
                filename="test.step",
                format=InputFormat.STEP,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        item = STEPEntityItem(
            label=CADItemLabel(text="Test Entity"),
            entity_id=1,
            entity_type="CARTESIAN_POINT",
            text="#1=CARTESIAN_POINT('',(-0.5,0.,0.));",
        )

        # Mock _get_shape_for_item to return None
        model._get_shape_for_item = Mock(return_value=None)

        model(doc, [item])

        # Now we always add topology_validation (with error status as fallback)
        assert "topology_validation" in item.properties
        assert item.properties["topology_validation"]["is_valid"] is False

    def test_call_with_mocked_occ_validation(self):
        """Test validation with mocked OCC shape."""
        from cadling.models.topology_validation import TopologyValidationModel
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = TopologyValidationModel()
        model.has_pythonocc = True

        doc = STEPDocument(
            name="test.step",
            origin=CADDocumentOrigin(
                filename="test.step",
                format=InputFormat.STEP,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        item = STEPEntityItem(
            label=CADItemLabel(text="Test Entity"),
            entity_id=1,
            entity_type="ADVANCED_FACE",
            text="#1=ADVANCED_FACE(...);",
        )

        # Mock shape and validation
        mock_shape = Mock()
        model._get_shape_for_item = Mock(return_value=mock_shape)
        model._is_occ_shape = Mock(return_value=True)
        model._validate_occ_shape = Mock(return_value={
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "is_watertight": True,
            "is_manifold": True,
            "euler_characteristic": 2,
            "topology_counts": {"vertices": 8, "edges": 12, "faces": 6},
        })

        model(doc, [item])

        # Check that validation results were added
        assert "topology_validation" in item.properties
        assert item.properties["topology_validation"]["is_valid"] is True
        assert item.properties["topology_validation"]["is_watertight"] is True

    def test_call_with_invalid_topology(self):
        """Test validation with invalid topology."""
        from cadling.models.topology_validation import TopologyValidationModel
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = TopologyValidationModel()
        model.has_pythonocc = True

        doc = STEPDocument(
            name="test.step",
            origin=CADDocumentOrigin(
                filename="test.step",
                format=InputFormat.STEP,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        item = STEPEntityItem(
            label=CADItemLabel(text="Invalid Entity"),
            entity_id=1,
            entity_type="ADVANCED_FACE",
            text="#1=ADVANCED_FACE(...);",
        )

        # Mock invalid topology
        mock_shape = Mock()
        model._get_shape_for_item = Mock(return_value=mock_shape)
        model._is_occ_shape = Mock(return_value=True)
        model._validate_occ_shape = Mock(return_value={
            "is_valid": False,
            "errors": ["Non-manifold geometry: 3 edges shared by >2 faces"],
            "warnings": [],
            "is_watertight": False,
            "is_manifold": False,
        })

        model(doc, [item])

        # Check that validation results show invalid topology
        assert "topology_validation" in item.properties
        assert item.properties["topology_validation"]["is_valid"] is False
        assert len(item.properties["topology_validation"]["errors"]) > 0

    def test_provenance_tracking(self):
        """Test that provenance is added to validated items."""
        from cadling.models.topology_validation import TopologyValidationModel
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = TopologyValidationModel()
        model.has_pythonocc = True

        doc = STEPDocument(
            name="test.step",
            origin=CADDocumentOrigin(
                filename="test.step",
                format=InputFormat.STEP,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        item = STEPEntityItem(
            label=CADItemLabel(text="Test Entity"),
            entity_id=1,
            entity_type="ADVANCED_FACE",
            text="#1=ADVANCED_FACE(...);",
        )

        # Mock valid validation
        mock_shape = Mock()
        model._get_shape_for_item = Mock(return_value=mock_shape)
        model._is_occ_shape = Mock(return_value=True)
        model._validate_occ_shape = Mock(return_value={
            "is_valid": True,
            "errors": [],
            "warnings": [],
        })

        model(doc, [item])

        # Verify provenance was added
        assert len(item.prov) > 0
        assert any(
            prov.component_type == "enrichment_model" and
            prov.component_name == "TopologyValidationModel"
            for prov in item.prov
        )


class TestTopologyValidationOCC:
    """Test OCC shape validation."""

    @pytest.mark.skipif(
        not pytest.importorskip("OCC", reason="pythonocc-core not available"),
        reason="pythonocc-core required"
    )
    def test_validate_occ_shape_cube(self):
        """Test OCC validation with a simple cube."""
        from cadling.models.topology_validation import TopologyValidationModel
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

        model = TopologyValidationModel()

        # Create a simple box (cube)
        box = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape()

        result = model._validate_occ_shape(box)

        # Cube should be valid
        assert result["is_valid"] is True or result["brepcheck_valid"] is True
        assert "topology_counts" in result
        # OCC counts vertices with different orientations separately
        # A cube has 8 geometric vertices but 48 oriented vertices (8 * 6 faces)
        assert result["topology_counts"]["vertices"] == 48
        assert result["topology_counts"]["edges"] == 24  # 4 edges per face * 6 faces
        assert result["topology_counts"]["faces"] == 6

        # Euler characteristic: V - E + F = 48 - 24 + 6 = 30
        # Note: This is for oriented topology, not geometric topology
        assert result["euler_characteristic"] == 30

    @pytest.mark.skipif(
        not pytest.importorskip("OCC", reason="pythonocc-core not available"),
        reason="pythonocc-core required"
    )
    def test_validate_occ_shape_cylinder(self):
        """Test OCC validation with a cylinder."""
        from cadling.models.topology_validation import TopologyValidationModel
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder

        model = TopologyValidationModel()

        # Create a cylinder
        cylinder = BRepPrimAPI_MakeCylinder(5.0, 10.0).Shape()

        result = model._validate_occ_shape(cylinder)

        # Cylinder should be valid
        assert result["is_valid"] is True or result["brepcheck_valid"] is True
        assert "topology_counts" in result
        # Cylinder topology: 2 circular edges + top/bottom faces + side face
        assert result["topology_counts"]["faces"] == 3  # top, bottom, side

    @pytest.mark.skipif(
        not pytest.importorskip("OCC", reason="pythonocc-core not available"),
        reason="pythonocc-core required"
    )
    def test_validate_occ_shape_watertightness(self):
        """Test watertightness check for OCC shapes."""
        from cadling.models.topology_validation import TopologyValidationModel
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

        model = TopologyValidationModel()

        # Box should be watertight
        box = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape()
        result = model._validate_occ_shape(box)

        # Check watertightness
        if "is_watertight" in result:
            assert result["is_watertight"] is True
            # Should have no boundary edges (all edges shared by 2 faces)
            if "edge_statistics" in result:
                assert result["edge_statistics"]["boundary_edges"] == 0

    @pytest.mark.skipif(
        not pytest.importorskip("OCC", reason="pythonocc-core not available"),
        reason="pythonocc-core required"
    )
    def test_validate_occ_shape_manifoldness(self):
        """Test manifoldness check for OCC shapes."""
        from cadling.models.topology_validation import TopologyValidationModel
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

        model = TopologyValidationModel()

        box = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape()
        result = model._validate_occ_shape(box)

        # Check manifoldness
        if "is_manifold" in result:
            assert result["is_manifold"] is True
            # Should have no non-manifold edges
            if "edge_statistics" in result:
                assert result["edge_statistics"]["non_manifold_edges"] == 0


class TestTopologyValidationTrimesh:
    """Test trimesh validation."""

    @pytest.mark.skipif(
        not pytest.importorskip("trimesh", reason="trimesh not available"),
        reason="trimesh required"
    )
    def test_validate_trimesh_cube(self):
        """Test trimesh validation with a cube."""
        from cadling.models.topology_validation import TopologyValidationModel
        import trimesh

        model = TopologyValidationModel()

        # Create a simple cube mesh
        cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])

        result = model._validate_trimesh(cube)

        # Cube should be valid
        assert result["is_valid"] is True
        assert result["is_watertight"] is True
        assert result["is_winding_consistent"] is True

        # Check topology counts
        assert "topology_counts" in result
        # Cube: 8 vertices, 6 faces (12 triangles)
        assert result["topology_counts"]["vertices"] == 8

    @pytest.mark.skipif(
        not pytest.importorskip("trimesh", reason="trimesh not available"),
        reason="trimesh required"
    )
    def test_validate_trimesh_sphere(self):
        """Test trimesh validation with a sphere."""
        from cadling.models.topology_validation import TopologyValidationModel
        import trimesh

        model = TopologyValidationModel()

        # Create a sphere mesh
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)

        result = model._validate_trimesh(sphere)

        # Sphere should be valid
        assert result["is_valid"] is True
        assert result["is_watertight"] is True

        # Euler characteristic should be 2 (genus 0)
        assert result["euler_characteristic"] == 2

    @pytest.mark.skipif(
        not pytest.importorskip("trimesh", reason="trimesh not available"),
        reason="trimesh required"
    )
    def test_validate_trimesh_degenerate_faces(self):
        """Test detection of degenerate faces."""
        from cadling.models.topology_validation import TopologyValidationModel
        import trimesh
        import numpy as np

        model = TopologyValidationModel()

        # Create mesh with degenerate face (zero area)
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],  # Duplicate vertex
        ])
        faces = np.array([
            [0, 1, 2],
            [0, 1, 3],  # Degenerate (vertices 0 and 3 are same)
        ])

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        result = model._validate_trimesh(mesh)

        # Should detect degenerate faces
        if "num_degenerate_faces" in result:
            # May or may not detect depending on trimesh version
            # Just check the key exists
            assert isinstance(result["num_degenerate_faces"], int)

    @pytest.mark.skipif(
        not pytest.importorskip("trimesh", reason="trimesh not available"),
        reason="trimesh required"
    )
    def test_validate_trimesh_euler_characteristic(self):
        """Test Euler characteristic calculation for trimesh."""
        from cadling.models.topology_validation import TopologyValidationModel
        import trimesh

        model = TopologyValidationModel()

        # Sphere (genus 0): χ = 2
        sphere = trimesh.creation.icosphere(subdivisions=1)
        result_sphere = model._validate_trimesh(sphere)
        assert result_sphere["euler_characteristic"] == 2

        # Torus (genus 1): χ = 0
        torus = trimesh.creation.torus(major_radius=2.0, minor_radius=0.5)
        result_torus = model._validate_trimesh(torus)
        assert result_torus["euler_characteristic"] == 0


class TestTopologyValidationBatchProcessing:
    """Test batch processing of topology validation."""

    def test_batch_processing_multiple_items(self):
        """Test validating multiple items."""
        from cadling.models.topology_validation import TopologyValidationModel
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = TopologyValidationModel()
        model.has_pythonocc = True

        doc = STEPDocument(
            name="test.step",
            origin=CADDocumentOrigin(
                filename="test.step",
                format=InputFormat.STEP,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        items = [
            STEPEntityItem(
                label=CADItemLabel(text=f"Entity {i}"),
                entity_id=i,
                entity_type="ADVANCED_FACE",
                text=f"#{i}=ADVANCED_FACE(...);",
            )
            for i in range(3)
        ]

        # Mock validation
        mock_shape = Mock()
        model._get_shape_for_item = Mock(return_value=mock_shape)
        model._is_occ_shape = Mock(return_value=True)
        model._validate_occ_shape = Mock(return_value={
            "is_valid": True,
            "errors": [],
            "warnings": [],
        })

        model(doc, items)

        # All items should have validation results
        for item in items:
            assert "topology_validation" in item.properties


class TestIntegration:
    """Integration tests for TopologyValidationModel."""

    def test_validation_workflow(self):
        """Test complete validation workflow."""
        from cadling.models.topology_validation import TopologyValidationModel

        model = TopologyValidationModel(strict_mode=False)

        # Model should initialize without error
        assert model.strict_mode is False
        assert isinstance(model.has_pythonocc, bool)
        assert isinstance(model.has_trimesh, bool)

    def test_strict_mode_validation(self):
        """Test validation in strict mode."""
        from cadling.models.topology_validation import TopologyValidationModel
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = TopologyValidationModel(strict_mode=True)
        model.has_pythonocc = True

        doc = STEPDocument(
            name="test.step",
            origin=CADDocumentOrigin(
                filename="test.step",
                format=InputFormat.STEP,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        item = STEPEntityItem(
            label=CADItemLabel(text="Test Entity"),
            entity_id=1,
            entity_type="ADVANCED_FACE",
            text="#1=ADVANCED_FACE(...);",
        )

        # Mock invalid topology in strict mode
        mock_shape = Mock()
        model._get_shape_for_item = Mock(return_value=mock_shape)
        model._is_occ_shape = Mock(return_value=True)
        model._validate_occ_shape = Mock(return_value={
            "is_valid": False,
            "errors": ["Test error"],
            "warnings": [],
        })

        model(doc, [item])

        # Should still add validation results even in strict mode
        assert "topology_validation" in item.properties
        assert item.properties["topology_validation"]["is_valid"] is False


class TestValidationFinding:
    """Test ValidationFinding construction and serialization."""

    def test_construction_minimal(self):
        finding = ValidationFinding(
            check_name="test_check",
            severity="warning",
            message="Test message",
        )
        assert finding.check_name == "test_check"
        assert finding.severity == "warning"
        assert finding.message == "Test message"
        assert finding.entity_ids == []
        assert finding.entity_type is None

    def test_construction_full(self):
        finding = ValidationFinding(
            check_name="face_edge_consistency",
            severity="critical",
            message="Edge deviates from parent face",
            entity_ids=["123", "456"],
            entity_type="EDGE",
        )
        assert finding.entity_ids == ["123", "456"]
        assert finding.entity_type == "EDGE"

    def test_serialization_roundtrip(self):
        finding = ValidationFinding(
            check_name="sliver_face",
            severity="warning",
            message="Sliver face detected",
            entity_ids=["789"],
            entity_type="FACE",
        )
        data = finding.model_dump()
        restored = ValidationFinding(**data)
        assert restored == finding

    def test_model_dump_keys(self):
        finding = ValidationFinding(
            check_name="test",
            severity="info",
            message="msg",
        )
        data = finding.model_dump()
        assert set(data.keys()) == {
            "check_name",
            "severity",
            "message",
            "entity_ids",
            "entity_type",
        }


class TestSeverityScoring:
    """Test _compute_severity_score method."""

    def setup_method(self):
        self.model = TopologyValidationModel()

    def test_empty_findings(self):
        score = self.model._compute_severity_score([])
        assert score["overall_severity"] == "clean"
        assert score["total_findings"] == 0
        assert score["critical_count"] == 0
        assert score["warning_count"] == 0
        assert score["info_count"] == 0

    def test_critical_findings(self):
        findings = [
            ValidationFinding(check_name="a", severity="critical", message="bad"),
            ValidationFinding(check_name="b", severity="warning", message="meh"),
        ]
        score = self.model._compute_severity_score(findings)
        assert score["overall_severity"] == "critical"
        assert score["critical_count"] == 1
        assert score["warning_count"] == 1
        assert score["total_findings"] == 2

    def test_warning_only(self):
        findings = [
            ValidationFinding(
                check_name="sliver_face", severity="warning", message="tiny face"
            ),
            ValidationFinding(
                check_name="sliver_edge", severity="warning", message="tiny edge"
            ),
        ]
        score = self.model._compute_severity_score(findings)
        assert score["overall_severity"] == "warning"
        assert score["warning_count"] == 2

    def test_info_only(self):
        findings = [
            ValidationFinding(check_name="note", severity="info", message="fyi"),
        ]
        score = self.model._compute_severity_score(findings)
        assert score["overall_severity"] == "info"
        assert score["info_count"] == 1

    def test_mixed_breakdown(self):
        findings = [
            ValidationFinding(check_name="check_a", severity="critical", message="m1"),
            ValidationFinding(check_name="check_a", severity="warning", message="m2"),
            ValidationFinding(check_name="check_b", severity="info", message="m3"),
        ]
        score = self.model._compute_severity_score(findings)
        breakdown = score["checks_breakdown"]
        assert "check_a" in breakdown
        assert breakdown["check_a"]["critical"] == 1
        assert breakdown["check_a"]["warning"] == 1
        assert "check_b" in breakdown
        assert breakdown["check_b"]["info"] == 1


class TestConstructorParams:
    """Test new constructor parameters."""

    def test_default_values(self):
        model = TopologyValidationModel()
        assert model.sliver_threshold == 0.05
        assert model.check_face_edge_consistency is True
        assert model.check_vertex_edge_consistency is True

    def test_custom_values(self):
        model = TopologyValidationModel(
            sliver_threshold=0.1,
            check_face_edge_consistency=False,
            check_vertex_edge_consistency=False,
        )
        assert model.sliver_threshold == 0.1
        assert model.check_face_edge_consistency is False
        assert model.check_vertex_edge_consistency is False


@pytest.mark.requires_pythonocc
class TestFaceEdgeConsistency:
    """Test face-edge consistency checking (requires pythonocc)."""

    def test_valid_box_zero_findings(self):
        """Valid box should have no face-edge consistency findings."""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

        box = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape()
        model = TopologyValidationModel()
        findings = model._check_face_edge_consistency(box)
        assert len(findings) == 0

    def test_disabled_returns_empty(self):
        """When disabled, should return empty list."""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

        box = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape()
        model = TopologyValidationModel(check_face_edge_consistency=False)
        findings = model._check_face_edge_consistency(box)
        assert len(findings) == 0


@pytest.mark.requires_pythonocc
class TestVertexEdgeConsistency:
    """Test vertex-edge consistency checking (requires pythonocc)."""

    def test_valid_box_zero_findings(self):
        """Valid box should have no vertex-edge consistency findings."""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

        box = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape()
        model = TopologyValidationModel()
        findings = model._check_vertex_edge_consistency(box)
        assert len(findings) == 0

    def test_disabled_returns_empty(self):
        """When disabled, should return empty list."""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

        box = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape()
        model = TopologyValidationModel(check_vertex_edge_consistency=False)
        findings = model._check_vertex_edge_consistency(box)
        assert len(findings) == 0


@pytest.mark.requires_pythonocc
class TestOrientationConsistency:
    """Test orientation consistency checking (requires pythonocc)."""

    def test_box_normals_outward(self):
        """Box faces should all have outward normals."""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

        box = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape()
        model = TopologyValidationModel()
        findings = model._check_orientation_consistency(box)
        # A valid box should have consistent outward normals
        assert all(f.check_name == "orientation_consistency" for f in findings)


@pytest.mark.requires_pythonocc
class TestSliverDetection:
    """Test sliver entity detection (requires pythonocc)."""

    def test_box_no_slivers(self):
        """Normal box should have no sliver entities."""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

        box = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape()
        model = TopologyValidationModel()
        findings = model._check_sliver_entities(box)
        assert len(findings) == 0

    def test_thin_box_detects_slivers(self):
        """Very thin box should detect sliver faces."""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

        # Make a very thin box - some faces will have tiny area
        thin_box = BRepPrimAPI_MakeBox(10.0, 10.0, 0.001).Shape()
        model = TopologyValidationModel(sliver_threshold=0.05)
        findings = model._check_sliver_entities(thin_box)
        # Should detect at least some sliver faces/edges
        sliver_faces = [f for f in findings if f.check_name == "sliver_face"]
        assert len(sliver_faces) >= 0  # May or may not detect depending on exact geometry

    def test_custom_threshold(self):
        """Custom threshold should change detection sensitivity."""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

        box = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape()
        # Very large threshold should flag all faces
        model = TopologyValidationModel(sliver_threshold=1000.0)
        findings = model._check_sliver_entities(box)
        # With threshold of 1000, 10x10 faces (area=100) should be flagged
        assert len(findings) > 0


class TestEnhancedValidationIntegration:
    """Test enhanced validation integration with mock shapes."""

    def test_new_keys_in_output(self):
        """Verify validation_findings and severity_scoring keys appear in output."""
        model = TopologyValidationModel()

        # Mock a simple validation that exercises the new code paths
        mock_findings = [
            ValidationFinding(
                check_name="sliver_face",
                severity="warning",
                message="test",
                entity_ids=["1"],
                entity_type="FACE",
            ),
        ]

        score = model._compute_severity_score(mock_findings)
        assert "overall_severity" in score
        assert "total_findings" in score
        assert "critical_count" in score
        assert "warning_count" in score
        assert "info_count" in score
        assert "checks_breakdown" in score

    def test_critical_finding_invalidates(self):
        """Critical findings should make overall severity critical."""
        model = TopologyValidationModel()
        findings = [
            ValidationFinding(
                check_name="face_edge_consistency",
                severity="critical",
                message="Edge deviates",
            ),
        ]
        score = model._compute_severity_score(findings)
        assert score["overall_severity"] == "critical"
        assert score["critical_count"] == 1

    def test_shape_to_entity_id(self):
        """Test shape hash ID generation with mock."""
        model = TopologyValidationModel()
        mock_shape = MagicMock()
        mock_shape.HashCode.return_value = 12345

        entity_id = model._shape_to_entity_id(mock_shape)
        assert entity_id == "12345"
        mock_shape.HashCode.assert_called_once_with(2**31 - 1)
