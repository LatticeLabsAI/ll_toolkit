"""
Unit tests for GeometricConstraintModel and related classes.

Tests cover:
- ConstraintType enumeration
- GeometricConstraint and ConstraintGraph models
- Model initialization
- Orientation constraint extraction
- Alignment constraint extraction
- Symmetry constraint extraction
- Dimensional constraint extraction
- Constraint graph building
"""

import pytest
from unittest.mock import Mock

from cadling.experimental.models import (
    GeometricConstraintModel,
    GeometricConstraint,
    ConstraintType,
    ConstraintGraph,
)


@pytest.fixture
def mock_doc_with_topology():
    """Create a mock CADlingDocument with topology."""
    doc = Mock()
    doc.topology = {
        "num_faces": 6,
        "faces": [
            {"id": "face_0", "normal": [0, 0, 1]},  # Top
            {"id": "face_1", "normal": [0, 0, -1]},  # Bottom (parallel to top)
            {"id": "face_2", "normal": [1, 0, 0]},  # Right (perpendicular)
            {"id": "face_3", "normal": [-1, 0, 0]},  # Left (parallel to right)
            {"id": "face_4", "normal": [0, 1, 0]},  # Front
            {"id": "face_5", "normal": [0, -1, 0]},  # Back (parallel to front)
        ],
    }
    return doc


@pytest.fixture
def mock_item_with_features():
    """Create a mock CADItem with detected features."""
    item = Mock()
    item.self_ref = "test_item"
    item.properties = {
        "bounding_box": {"x": 100, "y": 50, "z": 20},
        "machining_features": [
            {
                "feature_type": "hole",
                "parameters": {"diameter": 8.0},
                "location": [10, 10, 0],
            },
            {
                "feature_type": "hole",
                "parameters": {"diameter": 8.0},
                "location": [10, 40, 0],
            },
            {
                "feature_type": "hole",
                "parameters": {"diameter": 6.0},
                "location": [50, 25, 0],
            },
        ],
        "pmi_annotations": [
            {"type": "dimension", "value": 10.0, "unit": "mm", "text": "10mm", "confidence": 0.9},
            {"type": "dimension", "value": 50.0, "unit": "mm", "text": "50mm", "confidence": 0.85},
        ],
    }
    return item


class TestConstraintType:
    """Test ConstraintType enumeration."""

    def test_orientation_constraints(self):
        """Test orientation constraint types."""
        assert ConstraintType.PARALLEL == "parallel"
        assert ConstraintType.PERPENDICULAR == "perpendicular"
        assert ConstraintType.TANGENT == "tangent"

    def test_alignment_constraints(self):
        """Test alignment constraint types."""
        assert ConstraintType.CONCENTRIC == "concentric"
        assert ConstraintType.COAXIAL == "coaxial"
        assert ConstraintType.COINCIDENT == "coincident"

    def test_symmetry_constraints(self):
        """Test symmetry constraint types."""
        assert ConstraintType.SYMMETRIC == "symmetric"
        assert ConstraintType.SYMMETRIC_ABOUT_PLANE == "symmetric_about_plane"
        assert ConstraintType.SYMMETRIC_ABOUT_AXIS == "symmetric_about_axis"

    def test_dimensional_constraints(self):
        """Test dimensional constraint types."""
        assert ConstraintType.DISTANCE == "distance"
        assert ConstraintType.ANGLE == "angle"
        assert ConstraintType.EQUAL_LENGTH == "equal_length"
        assert ConstraintType.EQUAL_RADIUS == "equal_radius"

    def test_topological_constraints(self):
        """Test topological constraint types."""
        assert ConstraintType.CONNECTED == "connected"
        assert ConstraintType.ADJACENT == "adjacent"

    def test_constraint_type_iteration(self):
        """Test iterating over constraint types."""
        all_types = list(ConstraintType)
        assert len(all_types) >= 15  # At least 15 types defined


class TestGeometricConstraint:
    """Test GeometricConstraint pydantic model."""

    def test_initialization(self):
        """Test GeometricConstraint initialization."""
        constraint = GeometricConstraint(
            constraint_type=ConstraintType.PARALLEL,
            entities=["face_0", "face_1"],
            parameters={"dot_product": 0.999},
            confidence=0.95,
            description="Faces are parallel",
            is_explicit=False,
        )

        assert constraint.constraint_type == ConstraintType.PARALLEL
        assert len(constraint.entities) == 2
        assert constraint.parameters["dot_product"] == 0.999
        assert constraint.confidence == 0.95
        assert constraint.is_explicit is False

    def test_confidence_validation(self):
        """Test confidence validation (0-1)."""
        # Valid
        constraint = GeometricConstraint(
            constraint_type=ConstraintType.DISTANCE, confidence=0.5
        )
        assert constraint.confidence == 0.5

        # Out of bounds
        with pytest.raises(Exception):  # Pydantic ValidationError
            GeometricConstraint(
                constraint_type=ConstraintType.DISTANCE, confidence=1.5
            )

        with pytest.raises(Exception):
            GeometricConstraint(
                constraint_type=ConstraintType.DISTANCE, confidence=-0.1
            )

    def test_optional_fields(self):
        """Test constraint with minimal fields."""
        constraint = GeometricConstraint(constraint_type=ConstraintType.CONCENTRIC)

        assert constraint.entities == []
        assert constraint.parameters == {}
        assert constraint.confidence == 1.0
        assert constraint.description == ""


class TestConstraintGraph:
    """Test ConstraintGraph pydantic model."""

    def test_initialization(self):
        """Test ConstraintGraph initialization."""
        constraint1 = GeometricConstraint(
            constraint_type=ConstraintType.PARALLEL,
            entities=["face_0", "face_1"],
        )
        constraint2 = GeometricConstraint(
            constraint_type=ConstraintType.PERPENDICULAR,
            entities=["face_0", "face_2"],
        )

        graph = ConstraintGraph(
            nodes=["face_0", "face_1", "face_2"],
            edges=[constraint1, constraint2],
            clusters=[["face_0", "face_1", "face_2"]],
        )

        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2
        assert len(graph.clusters) == 1

    def test_empty_graph(self):
        """Test empty constraint graph."""
        graph = ConstraintGraph()

        assert graph.nodes == []
        assert graph.edges == []
        assert graph.clusters == []


class TestGeometricConstraintModel:
    """Test GeometricConstraintModel."""

    def test_initialization(self):
        """Test model initialization."""
        model = GeometricConstraintModel(tolerance=0.01, min_confidence=0.75)

        assert model.tolerance == 0.01
        assert model.min_confidence == 0.75

    def test_default_initialization(self):
        """Test model initialization with defaults."""
        model = GeometricConstraintModel()

        assert model.tolerance == 0.001
        assert model.min_confidence == 0.7

    def test_extract_orientation_constraints_parallel(
        self, mock_doc_with_topology, mock_item_with_features
    ):
        """Test parallel face detection."""
        model = GeometricConstraintModel()

        constraints = model._extract_orientation_constraints(
            mock_doc_with_topology, mock_item_with_features
        )

        # Should find parallel faces (top-bottom, right-left, front-back)
        parallel_constraints = [
            c for c in constraints if c.constraint_type == ConstraintType.PARALLEL
        ]
        assert len(parallel_constraints) >= 3

    def test_extract_orientation_constraints_perpendicular(
        self, mock_doc_with_topology, mock_item_with_features
    ):
        """Test perpendicular face detection."""
        model = GeometricConstraintModel()

        constraints = model._extract_orientation_constraints(
            mock_doc_with_topology, mock_item_with_features
        )

        # Should find perpendicular faces (e.g., top-right, top-front)
        perpendicular_constraints = [
            c for c in constraints if c.constraint_type == ConstraintType.PERPENDICULAR
        ]
        assert len(perpendicular_constraints) >= 4

    def test_extract_alignment_constraints_concentric(
        self, mock_doc_with_topology, mock_item_with_features
    ):
        """Test concentric feature detection."""
        model = GeometricConstraintModel(tolerance=0.001)

        # Add concentric holes (same location)
        item = Mock()
        item.self_ref = "test"
        item.properties = {
            "bounding_box": {},
            "machining_features": [
                {"feature_type": "hole", "parameters": {}, "location": [10, 10, 0]},
                {"feature_type": "hole", "parameters": {}, "location": [10, 10, 0]},
            ],
        }

        constraints = model._extract_alignment_constraints(mock_doc_with_topology, item)

        # Should detect concentricity
        concentric_constraints = [
            c for c in constraints if c.constraint_type == ConstraintType.CONCENTRIC
        ]
        assert len(concentric_constraints) > 0

    def test_extract_alignment_constraints_non_concentric(
        self, mock_doc_with_topology, mock_item_with_features
    ):
        """Test that non-concentric features are not flagged as concentric."""
        model = GeometricConstraintModel()

        constraints = model._extract_alignment_constraints(
            mock_doc_with_topology, mock_item_with_features
        )

        # Holes at [10,10], [10,40], [50,25] are not concentric
        concentric_constraints = [
            c for c in constraints if c.constraint_type == ConstraintType.CONCENTRIC
        ]
        assert len(concentric_constraints) == 0

    def test_extract_symmetry_constraints_planar(
        self, mock_doc_with_topology, mock_item_with_features
    ):
        """Test planar symmetry detection."""
        model = GeometricConstraintModel()

        # Thin part (100 x 50 x 20)
        constraints = model._extract_symmetry_constraints(
            mock_doc_with_topology, mock_item_with_features
        )

        # Should detect planar symmetry
        symmetry_constraints = [
            c
            for c in constraints
            if c.constraint_type == ConstraintType.SYMMETRIC_ABOUT_PLANE
        ]
        assert len(symmetry_constraints) > 0

    def test_extract_symmetry_constraints_feature_pattern(
        self, mock_doc_with_topology, mock_item_with_features
    ):
        """Test symmetric feature pattern detection."""
        model = GeometricConstraintModel()

        # Item has multiple holes - should detect potential pattern
        constraints = model._extract_symmetry_constraints(
            mock_doc_with_topology, mock_item_with_features
        )

        # Should detect symmetric pattern
        pattern_constraints = [
            c for c in constraints if c.constraint_type == ConstraintType.SYMMETRIC
        ]
        assert len(pattern_constraints) > 0

    def test_extract_dimensional_constraints_from_pmi(
        self, mock_doc_with_topology, mock_item_with_features
    ):
        """Test dimension extraction from PMI."""
        model = GeometricConstraintModel()

        constraints = model._extract_dimensional_constraints(
            mock_doc_with_topology, mock_item_with_features
        )

        # Should extract 2 dimension constraints from PMI
        distance_constraints = [
            c for c in constraints if c.constraint_type == ConstraintType.DISTANCE
        ]
        assert len(distance_constraints) == 2
        assert distance_constraints[0].is_explicit is True

    def test_extract_dimensional_constraints_equal_radii(
        self, mock_doc_with_topology, mock_item_with_features
    ):
        """Test equal radius detection."""
        model = GeometricConstraintModel()

        # Item has 2 holes with diameter 8.0
        constraints = model._extract_dimensional_constraints(
            mock_doc_with_topology, mock_item_with_features
        )

        # Should detect equal radii
        equal_radius_constraints = [
            c for c in constraints if c.constraint_type == ConstraintType.EQUAL_RADIUS
        ]
        assert len(equal_radius_constraints) == 1
        assert equal_radius_constraints[0].parameters["radius"] == 4.0  # diameter/2

    def test_build_constraint_graph(self, mock_doc_with_topology):
        """Test constraint graph building."""
        model = GeometricConstraintModel()

        constraints = [
            GeometricConstraint(
                constraint_type=ConstraintType.PARALLEL,
                entities=["face_0", "face_1"],
            ),
            GeometricConstraint(
                constraint_type=ConstraintType.PERPENDICULAR,
                entities=["face_0", "face_2"],
            ),
            GeometricConstraint(
                constraint_type=ConstraintType.CONCENTRIC,
                entities=["hole_0", "hole_1"],
            ),
        ]

        graph = model._build_constraint_graph(constraints)

        # Should have 5 nodes
        assert len(graph.nodes) == 5
        # Should have 3 edges (constraints)
        assert len(graph.edges) == 3
        # Should have clusters
        assert len(graph.clusters) > 0

    def test_find_clusters(self):
        """Test cluster finding with union-find."""
        model = GeometricConstraintModel()

        nodes = ["a", "b", "c", "d", "e"]
        constraints = [
            GeometricConstraint(
                constraint_type=ConstraintType.PARALLEL, entities=["a", "b"]
            ),
            GeometricConstraint(
                constraint_type=ConstraintType.PARALLEL, entities=["b", "c"]
            ),
            # d and e are separate
        ]

        clusters = model._find_clusters(nodes, constraints)

        # Should have 3 clusters: {a,b,c}, {d}, {e}
        assert len(clusters) == 3
        # Find the large cluster
        large_cluster = max(clusters, key=len)
        assert len(large_cluster) == 3

    def test_call_complete_workflow(
        self, mock_doc_with_topology, mock_item_with_features
    ):
        """Test complete constraint extraction workflow."""
        model = GeometricConstraintModel(min_confidence=0.5)

        model(mock_doc_with_topology, [mock_item_with_features])

        # Check constraints were extracted
        assert "constraints" in mock_item_with_features.properties
        constraints = mock_item_with_features.properties["constraints"]
        assert len(constraints) > 0

        # Check graph was built
        assert "constraint_graph" in mock_item_with_features.properties
        graph = mock_item_with_features.properties["constraint_graph"]
        assert graph is not None
        assert "nodes" in graph
        assert "edges" in graph

    def test_confidence_filtering(self, mock_doc_with_topology, mock_item_with_features):
        """Test that low-confidence constraints are filtered."""
        model = GeometricConstraintModel(min_confidence=0.95)

        model(mock_doc_with_topology, [mock_item_with_features])

        constraints = mock_item_with_features.properties["constraints"]

        # All constraints should meet minimum confidence
        assert all(c["confidence"] >= 0.95 for c in constraints)

    def test_no_topology_handling(self, mock_item_with_features):
        """Test handling when topology is missing."""
        model = GeometricConstraintModel()

        # Doc without topology
        doc = Mock()
        doc.topology = None

        model(doc, [mock_item_with_features])

        # Should not crash, but should have minimal constraints
        constraints = mock_item_with_features.properties["constraints"]
        # May still extract some constraints from features/PMI
        assert isinstance(constraints, list)

    def test_error_handling(self, mock_doc_with_topology):
        """Test error handling during extraction."""
        model = GeometricConstraintModel()

        # Item that will cause errors
        bad_item = Mock()
        bad_item.self_ref = "bad_item"
        bad_item.properties = None  # Will cause AttributeError

        # Should not crash
        model(mock_doc_with_topology, [bad_item])

        # Should have error recorded
        assert "constraint_extraction_error" in bad_item.properties

    def test_multiple_items(self, mock_doc_with_topology, mock_item_with_features):
        """Test processing multiple items."""
        model = GeometricConstraintModel()

        item1 = Mock()
        item1.self_ref = "item1"
        item1.properties = {"bounding_box": {}, "machining_features": []}

        item2 = Mock()
        item2.self_ref = "item2"
        item2.properties = {"bounding_box": {}, "machining_features": []}

        model(mock_doc_with_topology, [item1, item2])

        # Both items should have constraints
        assert "constraints" in item1.properties
        assert "constraints" in item2.properties

    def test_supports_batch_processing(self):
        """Test batch processing support."""
        model = GeometricConstraintModel()

        assert model.supports_batch_processing() is True
        assert model.get_batch_size() == 10

    def test_requires_gpu(self):
        """Test GPU requirements."""
        model = GeometricConstraintModel()

        assert model.requires_gpu() is False

    def test_get_model_info(self):
        """Test model info retrieval."""
        model = GeometricConstraintModel(tolerance=0.05, min_confidence=0.8)

        info = model.get_model_info()

        assert "tolerance" in info
        assert info["tolerance"] == "0.05"
        assert "min_confidence" in info
        assert info["min_confidence"] == "0.8"
        assert "constraint_types" in info

    def test_provenance_tracking(self, mock_doc_with_topology, mock_item_with_features):
        """Test that provenance is tracked."""
        model = GeometricConstraintModel()

        # Add provenance tracking method
        mock_item_with_features.add_provenance = Mock()

        model(mock_doc_with_topology, [mock_item_with_features])

        # Check provenance was added
        if hasattr(mock_item_with_features, "add_provenance"):
            mock_item_with_features.add_provenance.assert_called()

    def test_empty_features(self, mock_doc_with_topology):
        """Test handling of item with no features."""
        model = GeometricConstraintModel()

        item = Mock()
        item.self_ref = "empty_item"
        item.properties = {"bounding_box": {}, "machining_features": []}

        model(mock_doc_with_topology, [item])

        # Should still work, just fewer constraints
        assert "constraints" in item.properties
        constraints = item.properties["constraints"]
        # May have orientation constraints from topology
        assert isinstance(constraints, list)

    def test_tolerance_sensitivity(self, mock_doc_with_topology):
        """Test that tolerance affects constraint detection."""
        # Strict tolerance
        model_strict = GeometricConstraintModel(tolerance=0.0001)

        # Loose tolerance
        model_loose = GeometricConstraintModel(tolerance=0.1)

        # Item with slightly non-parallel faces
        item = Mock()
        item.self_ref = "test"
        item.properties = {"bounding_box": {}, "machining_features": []}

        doc = Mock()
        doc.topology = {
            "faces": [
                {"id": "f0", "normal": [0, 0, 1.0]},
                {"id": "f1", "normal": [0, 0, 0.99]},  # Slightly different
            ]
        }

        model_strict(doc, [item])
        constraints_strict = item.properties["constraints"]

        item.properties = {"bounding_box": {}, "machining_features": []}
        model_loose(doc, [item])
        constraints_loose = item.properties["constraints"]

        # Loose tolerance should find more constraints
        # (This is a soft assertion - depends on implementation)
        assert len(constraints_loose) >= len(constraints_strict)
