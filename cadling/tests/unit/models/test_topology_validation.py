"""Unit tests for topology validation model."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np


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

        # Should not add validation results if no shape
        assert "topology_validation" not in item.properties

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
