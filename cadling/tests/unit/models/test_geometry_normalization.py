"""Unit tests for GeometryNormalizationModel."""

import pytest
import numpy as np
from pathlib import Path

from cadling.models.geometry_normalization import GeometryNormalizationModel
from cadling.datamodel.base_models import CADItem, CADItemLabel, CADlingDocument, InputFormat


# Skip all tests if pythonocc is not available
pythonocc = pytest.importorskip("OCC.Core.BRepPrimAPI")


class TestGeometryNormalizationModel:
    """Test GeometryNormalizationModel class."""

    @pytest.fixture
    def normalization_model(self):
        """Create GeometryNormalizationModel instance with default settings."""
        return GeometryNormalizationModel(
            center=True,
            scale_to_unit=True,
            align_principal_axes=False
        )

    @pytest.fixture
    def normalization_model_with_pca(self):
        """Create GeometryNormalizationModel instance with PCA alignment."""
        return GeometryNormalizationModel(
            center=True,
            scale_to_unit=True,
            align_principal_axes=True
        )

    @pytest.fixture
    def mock_document(self):
        """Create mock CADlingDocument."""
        return CADlingDocument(
            name="test.step",
            format=InputFormat.STEP,
        )

    def _create_box_shape(self, width=10.0, depth=20.0, height=30.0, offset_x=5.0, offset_y=10.0, offset_z=15.0):
        """Create a box shape at a specific offset for testing normalization.

        Args:
            width: Box width (X dimension)
            depth: Box depth (Y dimension)
            height: Box height (Z dimension)
            offset_x: X offset from origin
            offset_y: Y offset from origin
            offset_z: Z offset from origin

        Returns:
            TopoDS_Shape representing a box
        """
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCC.Core.gp import gp_Pnt

        # Create box at offset position
        box = BRepPrimAPI_MakeBox(
            gp_Pnt(offset_x, offset_y, offset_z),
            width, depth, height
        ).Shape()

        return box

    def _create_oriented_box_shape(self):
        """Create a box shape oriented along a diagonal for PCA testing.

        Returns:
            TopoDS_Shape representing an oriented box
        """
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCC.Core.gp import gp_Trsf, gp_Ax1, gp_Pnt, gp_Dir
        import math

        # Create box at origin
        box = BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape()

        # Rotate 45 degrees around Z axis
        trsf = gp_Trsf()
        axis = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
        trsf.SetRotation(axis, math.pi / 4)

        # Apply transformation
        transform = BRepBuilderAPI_Transform(box, trsf)
        oriented_box = transform.Shape()

        return oriented_box

    def _create_mock_item_with_shape(self, shape, item_id=1):
        """Create mock CADItem with OCC shape attached.

        Args:
            shape: TopoDS_Shape object
            item_id: Item ID

        Returns:
            CADItem with _shape attribute set
        """
        item = CADItem(
            item_type="solid",
            label=CADItemLabel(text=f"Solid_{item_id}"),
        )

        # Attach OCC shape to item
        item._shape = shape

        return item

    def test_model_initialization(self, normalization_model):
        """Test GeometryNormalizationModel initialization."""
        assert normalization_model.has_pythonocc is True
        assert normalization_model.center is True
        assert normalization_model.scale_to_unit is True
        assert normalization_model.align_principal_axes is False

    def test_model_initialization_with_pca(self, normalization_model_with_pca):
        """Test GeometryNormalizationModel initialization with PCA enabled."""
        assert normalization_model_with_pca.align_principal_axes is True

    def test_centering_normalization(self, normalization_model, mock_document):
        """Test centering transformation with known offset.

        Expected:
        - Centroid should be computed correctly
        - Offset should be negative of centroid
        """
        # Create box at offset (5, 10, 15) with dimensions (10, 20, 30)
        # Centroid should be at (5 + 10/2, 10 + 20/2, 15 + 30/2) = (10, 20, 30)
        box_shape = self._create_box_shape(
            width=10.0, depth=20.0, height=30.0,
            offset_x=5.0, offset_y=10.0, offset_z=15.0
        )
        item = self._create_mock_item_with_shape(box_shape, item_id=1)

        # Add to document
        mock_document.add_item(item)

        # Run normalization
        normalization_model(mock_document, [item])

        # Verify results
        assert "geometry_normalization" in item.properties

        norm = item.properties["geometry_normalization"]

        # Check translation
        assert "translation" in norm
        centroid = norm["translation"]["centroid"]
        offset = norm["translation"]["offset"]

        # Centroid should be approximately [10, 20, 30]
        assert abs(centroid[0] - 10.0) < 1.0
        assert abs(centroid[1] - 20.0) < 1.0
        assert abs(centroid[2] - 30.0) < 1.0

        # Offset should be negative of centroid
        assert abs(offset[0] + centroid[0]) < 1e-6
        assert abs(offset[1] + centroid[1]) < 1e-6
        assert abs(offset[2] + centroid[2]) < 1e-6

    def test_scaling_normalization(self, normalization_model, mock_document):
        """Test scaling transformation.

        Expected:
        - Max dimension should be identified correctly
        - Scale factor should normalize to unit size (1.0)
        """
        # Create box with dimensions (10, 20, 30)
        # Max dimension is 30
        box_shape = self._create_box_shape(
            width=10.0, depth=20.0, height=30.0,
            offset_x=0.0, offset_y=0.0, offset_z=0.0
        )
        item = self._create_mock_item_with_shape(box_shape, item_id=1)

        # Add to document
        mock_document.add_item(item)

        # Run normalization
        normalization_model(mock_document, [item])

        # Verify results
        norm = item.properties["geometry_normalization"]

        # Check scaling
        assert "scaling" in norm
        scale_factor = norm["scaling"]["scale_factor"]
        original_max_dimension = norm["scaling"]["original_max_dimension"]
        dimensions = norm["scaling"]["dimensions"]

        # Max dimension should be ~30
        assert abs(original_max_dimension - 30.0) < 1.0

        # Scale factor should be 1/30
        expected_scale = 1.0 / 30.0
        assert abs(scale_factor - expected_scale) < 0.01

        # Dimensions should be [10, 20, 30]
        assert len(dimensions) == 3
        assert abs(dimensions[0] - 10.0) < 1.0
        assert abs(dimensions[1] - 20.0) < 1.0
        assert abs(dimensions[2] - 30.0) < 1.0

    def test_pca_alignment(self, normalization_model_with_pca, mock_document):
        """Test PCA alignment with oriented box.

        Expected:
        - Rotation matrix should be 3x3
        - Eigenvalues should be in descending order
        - Rotation matrix should be orthogonal
        - Determinant should be +1 (right-handed)
        """
        # Create oriented box
        oriented_box = self._create_oriented_box_shape()
        item = self._create_mock_item_with_shape(oriented_box, item_id=1)

        # Add to document
        mock_document.add_item(item)

        # Run normalization with PCA
        normalization_model_with_pca(mock_document, [item])

        # Verify results
        norm = item.properties["geometry_normalization"]

        # Check alignment
        assert "alignment" in norm
        rotation_matrix = np.array(norm["alignment"]["rotation_matrix"])
        eigenvalues = np.array(norm["alignment"]["eigenvalues"])

        # Check rotation matrix shape
        assert rotation_matrix.shape == (3, 3)

        # Check eigenvalues are in descending order
        assert eigenvalues[0] >= eigenvalues[1]
        assert eigenvalues[1] >= eigenvalues[2]

        # Check rotation matrix is orthogonal (R^T @ R should be identity)
        identity_approx = rotation_matrix.T @ rotation_matrix
        assert np.allclose(identity_approx, np.eye(3), atol=1e-6)

        # Check determinant is +1 (right-handed coordinate system)
        det = np.linalg.det(rotation_matrix)
        assert abs(det - 1.0) < 1e-6

    def test_combined_transformations(self, normalization_model_with_pca, mock_document):
        """Test all three transformations together."""
        # Create offset oriented box
        box_shape = self._create_box_shape(
            width=10.0, depth=20.0, height=30.0,
            offset_x=5.0, offset_y=10.0, offset_z=15.0
        )
        item = self._create_mock_item_with_shape(box_shape, item_id=1)

        # Add to document
        mock_document.add_item(item)

        # Run normalization with all transformations
        normalization_model_with_pca(mock_document, [item])

        # Verify all three components are present
        norm = item.properties["geometry_normalization"]

        assert "translation" in norm
        assert "scaling" in norm
        assert "alignment" in norm

        # Verify translation
        assert "centroid" in norm["translation"]
        assert "offset" in norm["translation"]

        # Verify scaling
        assert "scale_factor" in norm["scaling"]
        assert "original_max_dimension" in norm["scaling"]

        # Verify alignment
        assert "rotation_matrix" in norm["alignment"]
        assert "eigenvalues" in norm["alignment"]

    def test_centering_only(self, mock_document):
        """Test with only centering enabled."""
        model = GeometryNormalizationModel(
            center=True,
            scale_to_unit=False,
            align_principal_axes=False
        )

        box_shape = self._create_box_shape()
        item = self._create_mock_item_with_shape(box_shape, item_id=1)
        mock_document.add_item(item)

        model(mock_document, [item])

        norm = item.properties["geometry_normalization"]

        # Should have translation only
        assert "translation" in norm
        assert "scaling" not in norm
        assert "alignment" not in norm

    def test_scaling_only(self, mock_document):
        """Test with only scaling enabled."""
        model = GeometryNormalizationModel(
            center=False,
            scale_to_unit=True,
            align_principal_axes=False
        )

        box_shape = self._create_box_shape()
        item = self._create_mock_item_with_shape(box_shape, item_id=1)
        mock_document.add_item(item)

        model(mock_document, [item])

        norm = item.properties["geometry_normalization"]

        # Should have scaling only
        assert "translation" not in norm
        assert "scaling" in norm
        assert "alignment" not in norm

    def test_batch_processing(self, normalization_model, mock_document):
        """Test normalizing multiple items in a batch."""
        # Create multiple boxes
        box1 = self._create_box_shape(width=10.0, depth=10.0, height=10.0)
        box2 = self._create_box_shape(width=20.0, depth=20.0, height=20.0)
        box3 = self._create_box_shape(width=5.0, depth=5.0, height=5.0)

        item1 = self._create_mock_item_with_shape(box1, item_id=1)
        item2 = self._create_mock_item_with_shape(box2, item_id=2)
        item3 = self._create_mock_item_with_shape(box3, item_id=3)

        items = [item1, item2, item3]

        # Add to document
        for item in items:
            mock_document.add_item(item)

        # Run normalization on batch
        normalization_model(mock_document, items)

        # Verify all items were normalized
        for item in items:
            assert "geometry_normalization" in item.properties
            norm = item.properties["geometry_normalization"]
            assert "translation" in norm
            assert "scaling" in norm

    def test_provenance_tracking(self, normalization_model, mock_document):
        """Test that provenance is correctly added."""
        box_shape = self._create_box_shape()
        item = self._create_mock_item_with_shape(box_shape, item_id=1)

        mock_document.add_item(item)

        # Run normalization
        normalization_model(mock_document, [item])

        # Check provenance (stored in 'prov' attribute)
        assert len(item.prov) > 0

        # Find provenance entry for GeometryNormalizationModel
        found_provenance = False
        for prov in item.prov:
            if prov.component_name == "GeometryNormalizationModel":
                found_provenance = True
                assert prov.component_type == "enrichment_model"
                break

        assert found_provenance, "GeometryNormalizationModel provenance not found"

    def test_missing_shape(self, normalization_model, mock_document):
        """Test handling of item without OCC shape."""
        # Create item without _shape attribute
        item = CADItem(
            item_type="solid",
            label=CADItemLabel(text="Solid_NoShape"),
        )

        mock_document.add_item(item)

        # Run normalization (should not crash)
        normalization_model(mock_document, [item])

        # Verify item was skipped (no geometry_normalization property)
        assert "geometry_normalization" not in item.properties

    def test_degenerate_geometry(self, normalization_model, mock_document):
        """Test handling of degenerate geometry (very small dimension).

        Note: OCC has precision limits and cannot create boxes below a certain size.
        This test verifies the normalization handles edge cases gracefully.
        """
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

        # Try to create a very small box
        # If OCC rejects it, that's expected behavior - skip the test
        try:
            tiny_box = BRepPrimAPI_MakeBox(1e-8, 1e-8, 1e-8).Shape()
        except RuntimeError:
            # OCC cannot create boxes below its precision tolerance
            # This is expected - skip this test
            pytest.skip("OCC precision limits prevent creating very small boxes")
            return

        item = self._create_mock_item_with_shape(tiny_box, item_id=1)
        mock_document.add_item(item)

        # Run normalization (should not crash)
        normalization_model(mock_document, [item])

        # If normalization computed, scale factor should be 1.0 (fallback)
        # since max_dimension is <= 1e-10
        if "geometry_normalization" in item.properties:
            norm = item.properties["geometry_normalization"]
            if "scaling" in norm:
                # Should use fallback scale factor of 1.0 for very small dimensions
                assert norm["scaling"]["scale_factor"] == 1.0

    def test_model_info(self, normalization_model):
        """Test get_model_info() method."""
        info = normalization_model.get_model_info()

        assert info["model_class"] == "GeometryNormalizationModel"
        assert info["supports_batch"] == "True"
        assert info["batch_size"] == "1"
        assert info["requires_gpu"] == "False"

    def test_trimesh_support(self, normalization_model, mock_document):
        """Test normalization with trimesh objects (if trimesh available)."""
        try:
            import trimesh

            # Create a simple cube mesh
            mesh = trimesh.creation.box(extents=[10.0, 20.0, 30.0])

            # Offset mesh
            mesh.apply_translation([5.0, 10.0, 15.0])

            # Create item with trimesh
            item = CADItem(
                item_type="mesh",
                label=CADItemLabel(text="Mesh_1"),
            )
            item._shape = mesh

            mock_document.add_item(item)

            # Run normalization
            normalization_model(mock_document, [item])

            # Verify normalization was computed
            assert "geometry_normalization" in item.properties
            norm = item.properties["geometry_normalization"]

            # Check dimensions match mesh
            assert "scaling" in norm
            dimensions = norm["scaling"]["dimensions"]
            assert abs(dimensions[0] - 10.0) < 1.0
            assert abs(dimensions[1] - 20.0) < 1.0
            assert abs(dimensions[2] - 30.0) < 1.0

        except ImportError:
            pytest.skip("trimesh not available")
