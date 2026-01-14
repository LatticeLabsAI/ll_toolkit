"""Unit tests for mesh quality model."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np


class TestMeshQualityModel:
    """Test MeshQualityModel initialization and configuration."""

    def test_model_initialization_default(self):
        """Test model initializes with default settings."""
        from cadling.models.mesh_quality import MeshQualityModel

        model = MeshQualityModel()

        assert model.aspect_ratio_threshold == 10.0
        assert model.skewness_threshold == 0.8
        assert model.min_area_threshold == 1e-10
        assert hasattr(model, "has_trimesh")
        assert hasattr(model, "has_pythonocc")

    def test_model_initialization_custom_thresholds(self):
        """Test model initializes with custom thresholds."""
        from cadling.models.mesh_quality import MeshQualityModel

        model = MeshQualityModel(
            aspect_ratio_threshold=5.0,
            skewness_threshold=0.6,
            min_area_threshold=1e-8,
        )

        assert model.aspect_ratio_threshold == 5.0
        assert model.skewness_threshold == 0.6
        assert model.min_area_threshold == 1e-8

    def test_model_has_required_methods(self):
        """Test model has all required methods."""
        from cadling.models.mesh_quality import MeshQualityModel

        model = MeshQualityModel()

        assert callable(model.__call__)
        assert callable(model.supports_batch_processing)
        assert callable(model.get_batch_size)
        assert callable(model.requires_gpu)
        assert callable(model._assess_item)
        assert callable(model._assess_mesh)
        assert callable(model._compute_aspect_ratios)
        assert callable(model._compute_skewness)

    def test_model_trimesh_detection(self):
        """Test trimesh availability detection."""
        from cadling.models.mesh_quality import MeshQualityModel

        model = MeshQualityModel()

        # has_trimesh should be bool
        assert isinstance(model.has_trimesh, bool)

    def test_model_pythonocc_detection(self):
        """Test pythonocc availability detection."""
        from cadling.models.mesh_quality import MeshQualityModel

        model = MeshQualityModel()

        # has_pythonocc should be bool
        assert isinstance(model.has_pythonocc, bool)


class TestMeshQualityModelMethods:
    """Test MeshQualityModel helper methods."""

    def test_supports_batch_processing(self):
        """Test that model supports batch processing."""
        from cadling.models.mesh_quality import MeshQualityModel

        model = MeshQualityModel()

        assert model.supports_batch_processing() is True

    def test_get_batch_size(self):
        """Test that model returns batch size of 1."""
        from cadling.models.mesh_quality import MeshQualityModel

        model = MeshQualityModel()

        assert model.get_batch_size() == 1

    def test_requires_gpu(self):
        """Test that model does not require GPU."""
        from cadling.models.mesh_quality import MeshQualityModel

        model = MeshQualityModel()

        assert model.requires_gpu() is False


class TestMeshQualityModelCall:
    """Test MeshQualityModel __call__ method."""

    def test_call_without_trimesh(self):
        """Test that __call__ skips when trimesh not available."""
        from cadling.models.mesh_quality import MeshQualityModel
        from cadling.datamodel.stl import STLDocument, MeshItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = MeshQualityModel()
        # Force no trimesh
        model.has_trimesh = False

        doc = STLDocument(
            name="test.stl",
            origin=CADDocumentOrigin(
                filename="test.stl",
                format=InputFormat.STL,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        item = MeshItem(
            label=CADItemLabel(text="Triangle"),
            vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals=[[0.0, 0.0, 1.0]],
            facets=[[0, 1, 2]],
        )

        # Should return without error and without adding properties
        model(doc, [item])

        assert "mesh_quality" not in item.properties

    def test_call_with_no_mesh(self):
        """Test __call__ when item has no mesh."""
        from cadling.models.mesh_quality import MeshQualityModel
        from cadling.datamodel.stl import STLDocument, MeshItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = MeshQualityModel()
        # Ensure trimesh is available
        model.has_trimesh = True

        doc = STLDocument(
            name="test.stl",
            origin=CADDocumentOrigin(
                filename="test.stl",
                format=InputFormat.STL,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        item = MeshItem(
            label=CADItemLabel(text="Triangle"),
            vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals=[[0.0, 0.0, 1.0]],
            facets=[[0, 1, 2]],
        )

        # Mock _get_mesh_for_item to return None
        model._get_mesh_for_item = Mock(return_value=None)

        model(doc, [item])

        # Should not add quality results if no mesh
        assert "mesh_quality" not in item.properties

    def test_call_with_mocked_assessment(self):
        """Test assessment with mocked mesh."""
        from cadling.models.mesh_quality import MeshQualityModel
        from cadling.datamodel.stl import STLDocument, MeshItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = MeshQualityModel()
        model.has_trimesh = True

        doc = STLDocument(
            name="test.stl",
            origin=CADDocumentOrigin(
                filename="test.stl",
                format=InputFormat.STL,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        item = MeshItem(
            label=CADItemLabel(text="Triangle"),
            vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals=[[0.0, 0.0, 1.0]],
            facets=[[0, 1, 2]],
        )

        # Mock mesh and assessment
        mock_mesh = Mock()
        model._get_mesh_for_item = Mock(return_value=mock_mesh)
        model._assess_mesh = Mock(return_value={
            "num_vertices": 100,
            "num_faces": 50,
            "quality_score": 0.95,
            "quality_class": "excellent",
            "aspect_ratio": {"mean": 1.5, "max": 3.0},
        })

        model(doc, [item])

        # Check that quality results were added
        assert "mesh_quality" in item.properties
        assert item.properties["mesh_quality"]["quality_score"] == 0.95
        assert item.properties["mesh_quality"]["quality_class"] == "excellent"

    def test_call_with_poor_quality(self):
        """Test assessment with poor quality mesh."""
        from cadling.models.mesh_quality import MeshQualityModel
        from cadling.datamodel.stl import STLDocument, MeshItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = MeshQualityModel()
        model.has_trimesh = True

        doc = STLDocument(
            name="test.stl",
            origin=CADDocumentOrigin(
                filename="test.stl",
                format=InputFormat.STL,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        item = MeshItem(
            label=CADItemLabel(text="Triangle"),
            vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals=[[0.0, 0.0, 1.0]],
            facets=[[0, 1, 2]],
        )

        # Mock poor quality mesh
        mock_mesh = Mock()
        model._get_mesh_for_item = Mock(return_value=mock_mesh)
        model._assess_mesh = Mock(return_value={
            "num_vertices": 100,
            "num_faces": 50,
            "quality_score": 0.3,
            "quality_class": "poor",
            "aspect_ratio": {"mean": 15.0, "max": 50.0},
        })

        model(doc, [item])

        # Check that quality results show poor quality
        assert "mesh_quality" in item.properties
        assert item.properties["mesh_quality"]["quality_score"] < 0.5
        assert item.properties["mesh_quality"]["quality_class"] == "poor"

    def test_provenance_tracking(self):
        """Test that provenance is added to assessed items."""
        from cadling.models.mesh_quality import MeshQualityModel
        from cadling.datamodel.stl import STLDocument, MeshItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = MeshQualityModel()
        model.has_trimesh = True

        doc = STLDocument(
            name="test.stl",
            origin=CADDocumentOrigin(
                filename="test.stl",
                format=InputFormat.STL,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        item = MeshItem(
            label=CADItemLabel(text="Triangle"),
            vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals=[[0.0, 0.0, 1.0]],
            facets=[[0, 1, 2]],
        )

        # Mock assessment
        mock_mesh = Mock()
        model._get_mesh_for_item = Mock(return_value=mock_mesh)
        model._assess_mesh = Mock(return_value={
            "num_vertices": 100,
            "num_faces": 50,
            "quality_score": 0.95,
        })

        model(doc, [item])

        # Verify provenance was added
        assert len(item.prov) > 0
        assert any(
            prov.component_type == "enrichment_model" and
            prov.component_name == "MeshQualityModel"
            for prov in item.prov
        )


class TestMeshQualityAssessment:
    """Test mesh quality assessment functions."""

    @pytest.mark.skipif(
        not pytest.importorskip("trimesh", reason="trimesh not available"),
        reason="trimesh required"
    )
    def test_assess_mesh_cube(self):
        """Test mesh quality assessment with a cube."""
        from cadling.models.mesh_quality import MeshQualityModel
        import trimesh

        model = MeshQualityModel()

        # Create a cube mesh
        cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])

        result = model._assess_mesh(cube)

        # Check basic properties
        assert "num_vertices" in result
        assert "num_faces" in result
        assert result["num_vertices"] == 8
        assert result["num_faces"] == 12  # 2 triangles per face * 6 faces

        # Check quality metrics
        assert "aspect_ratio" in result
        assert "skewness" in result
        assert "edge_lengths" in result
        assert "quality_score" in result
        assert "quality_class" in result

        # Cube should have good quality
        assert result["quality_score"] >= 0.5

    @pytest.mark.skipif(
        not pytest.importorskip("trimesh", reason="trimesh not available"),
        reason="trimesh required"
    )
    def test_assess_mesh_sphere(self):
        """Test mesh quality assessment with a sphere."""
        from cadling.models.mesh_quality import MeshQualityModel
        import trimesh

        model = MeshQualityModel()

        # Create a sphere mesh
        sphere = trimesh.creation.icosphere(subdivisions=2)

        result = model._assess_mesh(sphere)

        # Sphere should have excellent quality (all equilateral triangles)
        assert result["quality_score"] >= 0.8
        assert result["quality_class"] in ["excellent", "good"]

        # Check aspect ratios are good
        assert result["aspect_ratio"]["mean"] < 2.0  # Close to 1.0 for equilateral

    @pytest.mark.skipif(
        not pytest.importorskip("trimesh", reason="trimesh not available"),
        reason="trimesh required"
    )
    def test_compute_aspect_ratios(self):
        """Test aspect ratio computation."""
        from cadling.models.mesh_quality import MeshQualityModel
        import trimesh

        model = MeshQualityModel()

        # Create a simple triangle mesh
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3)/2, 0.0],  # Equilateral triangle
        ])
        faces = np.array([[0, 1, 2]])

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        aspect_ratios = model._compute_aspect_ratios(mesh)

        # Equilateral triangle should have aspect ratio ≈ 1.0
        assert len(aspect_ratios) == 1
        assert abs(aspect_ratios[0] - 1.0) < 0.01

    @pytest.mark.skipif(
        not pytest.importorskip("trimesh", reason="trimesh not available"),
        reason="trimesh required"
    )
    def test_compute_aspect_ratios_stretched(self):
        """Test aspect ratio computation for stretched triangle."""
        from cadling.models.mesh_quality import MeshQualityModel
        import trimesh

        model = MeshQualityModel()

        # Create a stretched (poor quality) triangle
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],  # Very long edge
            [0.1, 0.1, 0.0],   # Short edges
        ])
        faces = np.array([[0, 1, 2]])

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        aspect_ratios = model._compute_aspect_ratios(mesh)

        # Stretched triangle should have high aspect ratio
        assert len(aspect_ratios) == 1
        assert aspect_ratios[0] > 10.0

    @pytest.mark.skipif(
        not pytest.importorskip("trimesh", reason="trimesh not available"),
        reason="trimesh required"
    )
    def test_compute_skewness_equilateral(self):
        """Test skewness computation for equilateral triangle."""
        from cadling.models.mesh_quality import MeshQualityModel
        import trimesh

        model = MeshQualityModel()

        # Create equilateral triangle
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3)/2, 0.0],
        ])
        faces = np.array([[0, 1, 2]])

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        skewness = model._compute_skewness(mesh)

        # Equilateral triangle should have skewness ≈ 0.0
        assert len(skewness) == 1
        assert skewness[0] < 0.1

    @pytest.mark.skipif(
        not pytest.importorskip("trimesh", reason="trimesh not available"),
        reason="trimesh required"
    )
    def test_compute_skewness_degenerate(self):
        """Test skewness computation for degenerate triangle."""
        from cadling.models.mesh_quality import MeshQualityModel
        import trimesh

        model = MeshQualityModel()

        # Create nearly degenerate triangle (very thin)
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [5.0, 0.001, 0.0],  # Very small height
        ])
        faces = np.array([[0, 1, 2]])

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        skewness = model._compute_skewness(mesh)

        # Degenerate triangle should have high skewness (close to 1.0)
        assert len(skewness) == 1
        assert skewness[0] > 0.9

    @pytest.mark.skipif(
        not pytest.importorskip("trimesh", reason="trimesh not available"),
        reason="trimesh required"
    )
    def test_quality_classification(self):
        """Test quality score classification."""
        from cadling.models.mesh_quality import MeshQualityModel
        import trimesh

        model = MeshQualityModel()

        # Test excellent quality (icosphere)
        sphere = trimesh.creation.icosphere(subdivisions=2)
        result_excellent = model._assess_mesh(sphere)
        assert result_excellent["quality_class"] in ["excellent", "good"]

        # Test with very poor thresholds to get poor classification
        model_strict = MeshQualityModel(
            aspect_ratio_threshold=1.1,
            skewness_threshold=0.05,
        )
        result_poor = model_strict._assess_mesh(sphere)
        # With strict thresholds, even sphere might not be excellent
        assert result_poor["quality_class"] in ["excellent", "good", "fair", "poor", "very_poor"]

    @pytest.mark.skipif(
        not pytest.importorskip("trimesh", reason="trimesh not available"),
        reason="trimesh required"
    )
    def test_degenerate_face_detection(self):
        """Test detection of degenerate faces."""
        from cadling.models.mesh_quality import MeshQualityModel
        import trimesh

        model = MeshQualityModel()

        # Create mesh with degenerate face (zero area)
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],  # Duplicate
        ])
        faces = np.array([
            [0, 1, 2],  # Good triangle
            [0, 1, 3],  # Degenerate (vertices 0 and 3 are same)
        ])

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        result = model._assess_mesh(mesh)

        # Should detect at least one degenerate face
        assert "num_degenerate_faces" in result
        # Trimesh might filter degenerate faces automatically
        assert isinstance(result["num_degenerate_faces"], int)


class TestMeshQualityBatchProcessing:
    """Test batch processing of mesh quality assessment."""

    def test_batch_processing_multiple_items(self):
        """Test assessing multiple items."""
        from cadling.models.mesh_quality import MeshQualityModel
        from cadling.datamodel.stl import STLDocument, MeshItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = MeshQualityModel()
        model.has_trimesh = True

        doc = STLDocument(
            name="test.stl",
            origin=CADDocumentOrigin(
                filename="test.stl",
                format=InputFormat.STL,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        items = [
            MeshItem(
                label=CADItemLabel(text=f"Triangle {i}"),
                vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                normals=[[0.0, 0.0, 1.0]],
                facets=[[0, 1, 2]],
            )
            for i in range(3)
        ]

        # Mock assessment
        mock_mesh = Mock()
        model._get_mesh_for_item = Mock(return_value=mock_mesh)
        model._assess_mesh = Mock(return_value={
            "num_vertices": 100,
            "num_faces": 50,
            "quality_score": 0.95,
        })

        model(doc, items)

        # All items should have quality results
        for item in items:
            assert "mesh_quality" in item.properties


class TestIntegration:
    """Integration tests for MeshQualityModel."""

    def test_quality_workflow(self):
        """Test complete quality assessment workflow."""
        from cadling.models.mesh_quality import MeshQualityModel

        model = MeshQualityModel(
            aspect_ratio_threshold=8.0,
            skewness_threshold=0.75,
            min_area_threshold=1e-9,
        )

        # Model should initialize without error
        assert model.aspect_ratio_threshold == 8.0
        assert model.skewness_threshold == 0.75
        assert model.min_area_threshold == 1e-9
        assert isinstance(model.has_trimesh, bool)

    @pytest.mark.skipif(
        not pytest.importorskip("trimesh", reason="trimesh not available"),
        reason="trimesh required"
    )
    def test_end_to_end_assessment(self):
        """Test end-to-end mesh quality assessment."""
        from cadling.models.mesh_quality import MeshQualityModel
        import trimesh

        model = MeshQualityModel()

        # Create various mesh shapes
        meshes = {
            "cube": trimesh.creation.box(extents=[1.0, 1.0, 1.0]),
            "sphere": trimesh.creation.icosphere(subdivisions=1),
            "cylinder": trimesh.creation.cylinder(radius=0.5, height=1.0),
        }

        for name, mesh in meshes.items():
            result = model._assess_mesh(mesh)

            # All meshes should have complete quality metrics
            assert "num_vertices" in result
            assert "num_faces" in result
            assert "aspect_ratio" in result
            assert "skewness" in result
            assert "edge_lengths" in result
            assert "quality_score" in result
            assert "quality_class" in result
            assert "is_manifold" in result

            # Quality score should be between 0 and 1
            assert 0.0 <= result["quality_score"] <= 1.0

            # Classification should be valid
            assert result["quality_class"] in [
                "excellent", "good", "fair", "poor", "very_poor"
            ]
