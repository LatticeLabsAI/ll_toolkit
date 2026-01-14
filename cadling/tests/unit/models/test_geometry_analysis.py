"""Unit tests for geometry analysis model."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np


class TestGeometryAnalysisModel:
    """Test GeometryAnalysisModel initialization and configuration."""

    def test_model_initialization_default(self):
        """Test model initializes with default settings."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel

        model = GeometryAnalysisModel()

        assert model.density == 1.0
        assert hasattr(model, "has_pythonocc")
        assert hasattr(model, "has_trimesh")

    def test_model_initialization_custom_density(self):
        """Test model initializes with custom density."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel

        model = GeometryAnalysisModel(density=7.85)  # Steel density

        assert model.density == 7.85

    def test_model_has_required_methods(self):
        """Test model has all required methods."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel

        model = GeometryAnalysisModel()

        assert callable(model.__call__)
        assert callable(model.supports_batch_processing)
        assert callable(model.get_batch_size)
        assert callable(model.requires_gpu)
        assert callable(model._analyze_item)
        assert callable(model._analyze_occ_shape)
        assert callable(model._analyze_trimesh)

    def test_model_pythonocc_detection(self):
        """Test pythonocc availability detection."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel

        model = GeometryAnalysisModel()

        # has_pythonocc should be bool
        assert isinstance(model.has_pythonocc, bool)

    def test_model_trimesh_detection(self):
        """Test trimesh availability detection."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel

        model = GeometryAnalysisModel()

        # has_trimesh should be bool
        assert isinstance(model.has_trimesh, bool)


class TestGeometryAnalysisModelMethods:
    """Test GeometryAnalysisModel helper methods."""

    def test_supports_batch_processing(self):
        """Test that model supports batch processing."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel

        model = GeometryAnalysisModel()

        assert model.supports_batch_processing() is True

    def test_get_batch_size(self):
        """Test that model returns batch size of 1."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel

        model = GeometryAnalysisModel()

        assert model.get_batch_size() == 1

    def test_requires_gpu(self):
        """Test that model does not require GPU."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel

        model = GeometryAnalysisModel()

        assert model.requires_gpu() is False

    def test_is_occ_shape_detection(self):
        """Test OCC shape type detection."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel

        model = GeometryAnalysisModel()

        # Mock object should return False
        mock_obj = Mock()
        assert model._is_occ_shape(mock_obj) is False

    def test_is_trimesh_detection(self):
        """Test trimesh type detection."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel

        model = GeometryAnalysisModel()

        # Mock object should return False
        mock_obj = Mock()
        assert model._is_trimesh(mock_obj) is False


class TestGeometryAnalysisModelCall:
    """Test GeometryAnalysisModel __call__ method."""

    def test_call_without_backends(self):
        """Test that __call__ skips when no backends available."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = GeometryAnalysisModel()
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
            entity_type="ADVANCED_FACE",
            text="#1=ADVANCED_FACE(...);",
        )

        # Should return without error and without adding properties
        model(doc, [item])

        assert "geometry_analysis" not in item.properties

    def test_call_with_no_shape(self):
        """Test __call__ when item has no shape."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = GeometryAnalysisModel()
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
            entity_type="ADVANCED_FACE",
            text="#1=ADVANCED_FACE(...);",
        )

        # Mock _get_shape_for_item to return None
        model._get_shape_for_item = Mock(return_value=None)

        model(doc, [item])

        # Should not add analysis results if no shape
        assert "geometry_analysis" not in item.properties

    def test_call_with_mocked_occ_analysis(self):
        """Test analysis with mocked OCC shape."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = GeometryAnalysisModel()
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

        # Mock shape and analysis
        mock_shape = Mock()
        model._get_shape_for_item = Mock(return_value=mock_shape)
        model._is_occ_shape = Mock(return_value=True)
        model._analyze_occ_shape = Mock(return_value={
            "volume": 1000.0,
            "surface_area": 600.0,
            "mass": 7850.0,
            "center_of_mass": {"x": 5.0, "y": 5.0, "z": 5.0},
            "bounding_box": {
                "xmin": 0.0, "ymin": 0.0, "zmin": 0.0,
                "xmax": 10.0, "ymax": 10.0, "zmax": 10.0,
                "dx": 10.0, "dy": 10.0, "dz": 10.0,
            },
        })

        model(doc, [item])

        # Check that analysis results were added
        assert "geometry_analysis" in item.properties
        assert item.properties["geometry_analysis"]["volume"] == 1000.0
        assert item.properties["geometry_analysis"]["surface_area"] == 600.0

    def test_provenance_tracking(self):
        """Test that provenance is added to analyzed items."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = GeometryAnalysisModel()
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

        # Mock analysis
        mock_shape = Mock()
        model._get_shape_for_item = Mock(return_value=mock_shape)
        model._is_occ_shape = Mock(return_value=True)
        model._analyze_occ_shape = Mock(return_value={
            "volume": 1000.0,
            "surface_area": 600.0,
        })

        model(doc, [item])

        # Verify provenance was added
        assert len(item.prov) > 0
        assert any(
            prov.component_type == "enrichment_model" and
            prov.component_name == "GeometryAnalysisModel"
            for prov in item.prov
        )


class TestGeometryAnalysisOCC:
    """Test OCC shape analysis."""

    @pytest.mark.skipif(
        not pytest.importorskip("OCC", reason="pythonocc-core not available"),
        reason="pythonocc-core required"
    )
    def test_analyze_occ_shape_box(self):
        """Test OCC analysis with a box."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

        model = GeometryAnalysisModel(density=1.0)

        # Create a 10x10x10 box
        box = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape()

        result = model._analyze_occ_shape(box)

        # Check basic properties
        assert "volume" in result
        assert "surface_area" in result
        assert "mass" in result
        assert "center_of_mass" in result
        assert "bounding_box" in result
        assert "inertia_tensor" in result

        # Verify volume (should be 1000 cubic units)
        assert abs(result["volume"] - 1000.0) < 1.0

        # Verify surface area (6 faces * 100 = 600)
        assert abs(result["surface_area"] - 600.0) < 1.0

        # Verify mass (volume * density)
        assert abs(result["mass"] - 1000.0) < 1.0

        # Verify center of mass (should be at 5, 5, 5)
        assert abs(result["center_of_mass"]["x"] - 5.0) < 0.1
        assert abs(result["center_of_mass"]["y"] - 5.0) < 0.1
        assert abs(result["center_of_mass"]["z"] - 5.0) < 0.1

        # Verify bounding box
        assert abs(result["bounding_box"]["dx"] - 10.0) < 0.1
        assert abs(result["bounding_box"]["dy"] - 10.0) < 0.1
        assert abs(result["bounding_box"]["dz"] - 10.0) < 0.1

    @pytest.mark.skipif(
        not pytest.importorskip("OCC", reason="pythonocc-core not available"),
        reason="pythonocc-core required"
    )
    def test_analyze_occ_shape_cylinder(self):
        """Test OCC analysis with a cylinder."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder

        model = GeometryAnalysisModel(density=2.7)  # Aluminum

        # Create a cylinder: radius=5, height=10
        cylinder = BRepPrimAPI_MakeCylinder(5.0, 10.0).Shape()

        result = model._analyze_occ_shape(cylinder)

        # Volume = π * r² * h = π * 25 * 10 ≈ 785.4
        expected_volume = np.pi * 25.0 * 10.0
        assert abs(result["volume"] - expected_volume) < 10.0

        # Check mass calculation
        expected_mass = expected_volume * 2.7
        assert abs(result["mass"] - expected_mass) < 30.0

    @pytest.mark.skipif(
        not pytest.importorskip("OCC", reason="pythonocc-core not available"),
        reason="pythonocc-core required"
    )
    def test_analyze_occ_compactness(self):
        """Test compactness calculation for OCC shapes."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere

        model = GeometryAnalysisModel()

        # Box has compactness = 1.0 (fills its bounding box)
        box = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape()
        result_box = model._analyze_occ_shape(box)
        assert abs(result_box["compactness"] - 1.0) < 0.01

        # Sphere has compactness < 1.0 (doesn't fill bounding box)
        sphere = BRepPrimAPI_MakeSphere(5.0).Shape()
        result_sphere = model._analyze_occ_shape(sphere)
        # Sphere compactness ≈ π/6 ≈ 0.524
        assert 0.4 < result_sphere["compactness"] < 0.6

    @pytest.mark.skipif(
        not pytest.importorskip("OCC", reason="pythonocc-core not available"),
        reason="pythonocc-core required"
    )
    def test_analyze_occ_surface_to_volume(self):
        """Test surface-to-volume ratio calculation."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

        model = GeometryAnalysisModel()

        # 10x10x10 box: SA=600, V=1000, ratio=0.6
        box = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape()
        result = model._analyze_occ_shape(box)

        expected_ratio = 600.0 / 1000.0
        assert abs(result["surface_to_volume_ratio"] - expected_ratio) < 0.01

    @pytest.mark.skipif(
        not pytest.importorskip("OCC", reason="pythonocc-core not available"),
        reason="pythonocc-core required"
    )
    def test_analyze_occ_inertia_tensor(self):
        """Test inertia tensor computation."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

        model = GeometryAnalysisModel()

        box = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape()
        result = model._analyze_occ_shape(box)

        # Check that inertia tensor has all components
        assert "inertia_tensor" in result
        assert "Ixx" in result["inertia_tensor"]
        assert "Iyy" in result["inertia_tensor"]
        assert "Izz" in result["inertia_tensor"]
        assert "Ixy" in result["inertia_tensor"]
        assert "Ixz" in result["inertia_tensor"]
        assert "Iyz" in result["inertia_tensor"]

        # For a cube, Ixx = Iyy = Izz (symmetric)
        Ixx = result["inertia_tensor"]["Ixx"]
        Iyy = result["inertia_tensor"]["Iyy"]
        Izz = result["inertia_tensor"]["Izz"]

        # Check symmetry (within tolerance)
        assert abs(Ixx - Iyy) / max(Ixx, Iyy) < 0.01
        assert abs(Iyy - Izz) / max(Iyy, Izz) < 0.01


class TestGeometryAnalysisTrimesh:
    """Test trimesh analysis."""

    @pytest.mark.skipif(
        not pytest.importorskip("trimesh", reason="trimesh not available"),
        reason="trimesh required"
    )
    def test_analyze_trimesh_cube(self):
        """Test trimesh analysis with a cube."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel
        import trimesh

        model = GeometryAnalysisModel(density=1.0)

        # Create a cube mesh (1x1x1)
        cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])

        result = model._analyze_trimesh(cube)

        # Check basic properties
        assert "volume" in result
        assert "surface_area" in result
        assert "mass" in result
        assert "center_of_mass" in result
        assert "bounding_box" in result
        assert "num_vertices" in result
        assert "num_faces" in result
        assert "is_watertight" in result

        # Verify volume (should be 1.0)
        assert abs(result["volume"] - 1.0) < 0.01

        # Verify mass
        assert abs(result["mass"] - 1.0) < 0.01

        # Cube should be watertight
        assert result["is_watertight"] is True

    @pytest.mark.skipif(
        not pytest.importorskip("trimesh", reason="trimesh not available"),
        reason="trimesh required"
    )
    def test_analyze_trimesh_sphere(self):
        """Test trimesh analysis with a sphere."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel
        import trimesh

        model = GeometryAnalysisModel(density=2.0)

        # Create a sphere mesh (radius=1)
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)

        result = model._analyze_trimesh(sphere)

        # Volume ≈ (4/3)πr³ ≈ 4.19
        expected_volume = (4.0 / 3.0) * np.pi * (1.0 ** 3)
        assert abs(result["volume"] - expected_volume) < 0.2

        # Check mass
        expected_mass = expected_volume * 2.0
        assert abs(result["mass"] - expected_mass) < 0.5

        # Sphere should be watertight
        assert result["is_watertight"] is True

    @pytest.mark.skipif(
        not pytest.importorskip("trimesh", reason="trimesh not available"),
        reason="trimesh required"
    )
    def test_analyze_trimesh_compactness(self):
        """Test compactness calculation for trimesh."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel
        import trimesh

        model = GeometryAnalysisModel()

        # Sphere has compactness ≈ π/6 ≈ 0.524
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        result = model._analyze_trimesh(sphere)

        assert 0.4 < result["compactness"] < 0.7

    @pytest.mark.skipif(
        not pytest.importorskip("trimesh", reason="trimesh not available"),
        reason="trimesh required"
    )
    def test_analyze_trimesh_inertia_watertight(self):
        """Test inertia tensor for watertight mesh."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel
        import trimesh

        model = GeometryAnalysisModel()

        # Watertight cube should have inertia tensor
        cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        result = model._analyze_trimesh(cube)

        assert result["inertia_tensor"] is not None
        assert "Ixx" in result["inertia_tensor"]
        assert "Iyy" in result["inertia_tensor"]
        assert "Izz" in result["inertia_tensor"]

    @pytest.mark.skipif(
        not pytest.importorskip("trimesh", reason="trimesh not available"),
        reason="trimesh required"
    )
    def test_analyze_trimesh_inertia_non_watertight(self):
        """Test inertia tensor for non-watertight mesh."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel
        import trimesh

        model = GeometryAnalysisModel()

        # Create a non-watertight mesh (single triangle)
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
        ])
        faces = np.array([[0, 1, 2]])

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        result = model._analyze_trimesh(mesh)

        # Non-watertight mesh should have None inertia tensor
        assert result["inertia_tensor"] is None

    @pytest.mark.skipif(
        not pytest.importorskip("trimesh", reason="trimesh not available"),
        reason="trimesh required"
    )
    def test_analyze_trimesh_center_of_mass(self):
        """Test center of mass computation."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel
        import trimesh

        model = GeometryAnalysisModel()

        # Cube centered at origin
        cube = trimesh.creation.box(extents=[2.0, 2.0, 2.0])

        result = model._analyze_trimesh(cube)

        # Center should be close to origin
        assert abs(result["center_of_mass"]["x"]) < 0.1
        assert abs(result["center_of_mass"]["y"]) < 0.1
        assert abs(result["center_of_mass"]["z"]) < 0.1


class TestGeometryAnalysisBatchProcessing:
    """Test batch processing of geometry analysis."""

    def test_batch_processing_multiple_items(self):
        """Test analyzing multiple items."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = GeometryAnalysisModel()
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

        # Mock analysis
        mock_shape = Mock()
        model._get_shape_for_item = Mock(return_value=mock_shape)
        model._is_occ_shape = Mock(return_value=True)
        model._analyze_occ_shape = Mock(return_value={
            "volume": 1000.0,
            "surface_area": 600.0,
        })

        model(doc, items)

        # All items should have analysis results
        for item in items:
            assert "geometry_analysis" in item.properties


class TestIntegration:
    """Integration tests for GeometryAnalysisModel."""

    def test_analysis_workflow(self):
        """Test complete analysis workflow."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel

        model = GeometryAnalysisModel(density=7.85)

        # Model should initialize without error
        assert model.density == 7.85
        assert isinstance(model.has_pythonocc, bool)
        assert isinstance(model.has_trimesh, bool)

    @pytest.mark.skipif(
        not pytest.importorskip("trimesh", reason="trimesh not available"),
        reason="trimesh required"
    )
    def test_end_to_end_analysis(self):
        """Test end-to-end geometry analysis."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel
        import trimesh

        model = GeometryAnalysisModel(density=1.0)

        # Test various shapes
        shapes = {
            "cube": trimesh.creation.box(extents=[1.0, 1.0, 1.0]),
            "sphere": trimesh.creation.icosphere(subdivisions=1, radius=1.0),
            "cylinder": trimesh.creation.cylinder(radius=0.5, height=2.0),
        }

        for name, mesh in shapes.items():
            result = model._analyze_trimesh(mesh)

            # All meshes should have complete analysis
            assert "volume" in result
            assert "surface_area" in result
            assert "mass" in result
            assert "center_of_mass" in result
            assert "bounding_box" in result
            assert "compactness" in result
            assert "surface_to_volume_ratio" in result
            assert "num_vertices" in result
            assert "num_faces" in result
            assert "is_watertight" in result

            # Volume should be positive
            assert result["volume"] > 0

            # Surface area should be positive
            assert result["surface_area"] > 0

            # Compactness should be between 0 and 1
            assert 0.0 <= result["compactness"] <= 1.0

    @pytest.mark.skipif(
        not pytest.importorskip("OCC", reason="pythonocc-core not available"),
        reason="pythonocc-core required"
    )
    def test_occ_shape_variations(self):
        """Test OCC analysis with different shape sizes."""
        from cadling.models.geometry_analysis import GeometryAnalysisModel
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

        model = GeometryAnalysisModel(density=1.0)

        # Test different sized boxes
        sizes = [(1, 1, 1), (10, 10, 10), (2, 3, 5)]

        for x, y, z in sizes:
            box = BRepPrimAPI_MakeBox(float(x), float(y), float(z)).Shape()
            result = model._analyze_occ_shape(box)

            expected_volume = x * y * z
            expected_surface_area = 2 * (x * y + y * z + z * x)

            # Check volume
            assert abs(result["volume"] - expected_volume) < 0.1

            # Check surface area
            assert abs(result["surface_area"] - expected_surface_area) < 0.5

            # Check bounding box dimensions
            assert abs(result["bounding_box"]["dx"] - x) < 0.1
            assert abs(result["bounding_box"]["dy"] - y) < 0.1
            assert abs(result["bounding_box"]["dz"] - z) < 0.1
