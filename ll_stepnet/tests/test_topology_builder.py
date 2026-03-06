"""Tests for STEPTopologyBuilder and topology graph construction.

Tests cover:
- Reference graph building
- Adjacency matrix construction
- Edge index (PyG format)
- Node feature extraction
- Compact features (48-dim)
- Coedge structure (BRepNet)
- Cadling conversion
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Skip entire module if torch is not installed
torch = pytest.importorskip("torch")


# ============================================================================
# SECTION 1: Reference Graph Building Tests
# ============================================================================


class TestReferenceGraphBuilding:
    """Test reference graph construction from STEP entities."""

    def test_build_reference_graph_simple(self) -> None:
        """Test building reference graph from simple STEP entities."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        # Simple mock STEP data using correct key names
        entities = [
            {"entity_id": 1, "entity_type": "ADVANCED_FACE", "references": [2, 3]},
            {"entity_id": 2, "entity_type": "EDGE_CURVE", "references": [4]},
            {"entity_id": 3, "entity_type": "EDGE_CURVE", "references": [4, 5]},
            {"entity_id": 4, "entity_type": "VERTEX_POINT", "references": []},
            {"entity_id": 5, "entity_type": "VERTEX_POINT", "references": []},
        ]

        graph = builder.build_reference_graph(entities)

        assert graph is not None
        assert "adjacency_dict" in graph
        assert "edge_list" in graph
        assert "num_nodes" in graph
        assert "node_ids" in graph
        assert "id_to_idx" in graph

    def test_build_reference_graph_with_orphans(self) -> None:
        """Test handling of orphan nodes (no references)."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        entities = [
            {"entity_id": 1, "entity_type": "ADVANCED_FACE", "references": []},  # Orphan
            {"entity_id": 2, "entity_type": "EDGE_CURVE", "references": []},  # Orphan
        ]

        graph = builder.build_reference_graph(entities)

        # Should still create nodes for orphans
        assert graph is not None
        assert graph["num_nodes"] == 2
        assert 1 in graph["node_ids"]
        assert 2 in graph["node_ids"]

    def test_build_reference_graph_with_cycles(self) -> None:
        """Test handling of cyclic references."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        # Cyclic: 1 -> 2 -> 3 -> 1
        entities = [
            {"entity_id": 1, "entity_type": "A", "references": [2]},
            {"entity_id": 2, "entity_type": "B", "references": [3]},
            {"entity_id": 3, "entity_type": "C", "references": [1]},
        ]

        graph = builder.build_reference_graph(entities)

        # Should handle cycles without infinite loop
        assert graph is not None
        assert graph["num_nodes"] == 3
        assert len(graph["edge_list"]) == 3

    def test_build_reference_graph_edge_list(self) -> None:
        """Test edge list is correctly built."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        entities = [
            {"entity_id": 1, "entity_type": "A", "references": [2, 3]},
            {"entity_id": 2, "entity_type": "B", "references": []},
            {"entity_id": 3, "entity_type": "C", "references": []},
        ]

        graph = builder.build_reference_graph(entities)

        # Should have 2 edges: (1,2) and (1,3)
        assert len(graph["edge_list"]) == 2
        assert (1, 2) in graph["edge_list"]
        assert (1, 3) in graph["edge_list"]


# ============================================================================
# SECTION 2: Adjacency Matrix Tests
# ============================================================================


class TestAdjacencyMatrix:
    """Test adjacency matrix construction."""

    def test_adjacency_matrix_shape(self) -> None:
        """Test adjacency matrix has correct shape [N, N]."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        entities = [
            {"entity_id": i, "entity_type": "NODE", "references": [(i + 1) % 10] if i < 9 else []}
            for i in range(10)
        ]

        graph = builder.build_reference_graph(entities)
        adj = builder.build_adjacency_matrix(graph)

        assert adj.shape == (graph["num_nodes"], graph["num_nodes"])

    def test_adjacency_matrix_values(self) -> None:
        """Test adjacency matrix contains correct 0/1 values."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        entities = [
            {"entity_id": 1, "entity_type": "A", "references": [2]},
            {"entity_id": 2, "entity_type": "B", "references": [3]},
            {"entity_id": 3, "entity_type": "C", "references": []},
        ]

        graph = builder.build_reference_graph(entities)
        adj = builder.build_adjacency_matrix(graph)

        # Convert sparse to dense for value checking
        adj_dense = adj.to_dense() if adj.is_sparse else adj

        # Should only contain 0s and 1s
        assert ((adj_dense == 0) | (adj_dense == 1)).all()

    def test_adjacency_matrix_directed_edges(self) -> None:
        """Test adjacency matrix represents directed edges correctly."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        entities = [
            {"entity_id": 1, "entity_type": "A", "references": [2]},
            {"entity_id": 2, "entity_type": "B", "references": []},
        ]

        graph = builder.build_reference_graph(entities)
        adj = builder.build_adjacency_matrix(graph)
        adj_dense = adj.to_dense() if adj.is_sparse else adj

        id_to_idx = graph["id_to_idx"]
        idx_1 = id_to_idx[1]
        idx_2 = id_to_idx[2]

        # Edge 1 -> 2 should exist
        assert adj_dense[idx_1, idx_2] == 1.0
        # No reverse edge
        assert adj_dense[idx_2, idx_1] == 0.0

    def test_adjacency_matrix_empty_graph(self) -> None:
        """Test adjacency matrix for graph with no edges."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        entities = [
            {"entity_id": i, "entity_type": "NODE", "references": []}
            for i in range(5)
        ]

        graph = builder.build_reference_graph(entities)
        adj = builder.build_adjacency_matrix(graph)

        assert adj.shape == (5, 5)
        assert adj.sum() == 0


# ============================================================================
# SECTION 3: Edge Index Tests (PyG Format)
# ============================================================================


class TestEdgeIndex:
    """Test edge index construction for PyTorch Geometric."""

    def test_edge_index_shape(self) -> None:
        """Test edge index has shape [2, num_edges]."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        entities = [
            {"entity_id": 1, "entity_type": "A", "references": [2]},
            {"entity_id": 2, "entity_type": "B", "references": [3]},
            {"entity_id": 3, "entity_type": "C", "references": [4]},
            {"entity_id": 4, "entity_type": "D", "references": []},
        ]

        graph = builder.build_reference_graph(entities)
        edge_index = builder.build_edge_index(graph)

        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] == 3  # 3 edges

    def test_edge_index_valid_indices(self) -> None:
        """Test all edge indices are valid node indices."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        num_nodes = 10
        entities = [
            {"entity_id": i, "entity_type": "NODE", "references": [(i + 1) % num_nodes]}
            for i in range(num_nodes)
        ]

        graph = builder.build_reference_graph(entities)
        edge_index = builder.build_edge_index(graph)

        assert (edge_index >= 0).all()
        assert (edge_index < graph["num_nodes"]).all()

    def test_edge_index_empty(self) -> None:
        """Test edge index for empty edge list."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        entities = [
            {"entity_id": i, "entity_type": "NODE", "references": []}
            for i in range(5)
        ]

        graph = builder.build_reference_graph(entities)
        edge_index = builder.build_edge_index(graph)

        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] == 0

    def test_edge_index_dtype(self) -> None:
        """Test edge index has long dtype."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        entities = [
            {"entity_id": 1, "entity_type": "A", "references": [2]},
            {"entity_id": 2, "entity_type": "B", "references": []},
        ]

        graph = builder.build_reference_graph(entities)
        edge_index = builder.build_edge_index(graph)

        assert edge_index.dtype == torch.long


# ============================================================================
# SECTION 4: Node Feature Tests (129-dim legacy format)
# ============================================================================


class TestNodeFeatures:
    """Test node feature extraction (129-dim legacy format)."""

    def test_node_features_shape(self) -> None:
        """Test node features have correct shape [num_nodes, 129]."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        entities = [
            {"entity_id": 1, "entity_type": "ADVANCED_FACE", "numeric_params": [1.0, 2.0, 3.0]},
            {"entity_id": 2, "entity_type": "EDGE_CURVE", "numeric_params": [4.0, 5.0]},
            {"entity_id": 3, "entity_type": "VERTEX_POINT", "numeric_params": [6.0]},
        ]

        graph = builder.build_reference_graph(entities)
        features = builder.build_node_features(entities, graph)

        assert features.shape[0] == graph["num_nodes"]
        assert features.shape[1] == 129  # 128 numeric + 1 type hash

    def test_node_features_numeric_params(self) -> None:
        """Test numeric parameters are properly encoded."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        entities = [
            {"entity_id": 1, "entity_type": "TEST", "numeric_params": [1.5, 2.5, 3.5]},
        ]

        graph = builder.build_reference_graph(entities)
        features = builder.build_node_features(entities, graph)

        idx = graph["id_to_idx"][1]
        # First few dims should contain numeric params
        assert features[idx, :3].tolist() == pytest.approx([1.5, 2.5, 3.5], rel=1e-4)

    def test_node_features_type_hash(self) -> None:
        """Test entity type is hashed in last dimension."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        entities = [
            {"entity_id": 1, "entity_type": "ADVANCED_FACE", "numeric_params": []},
            {"entity_id": 2, "entity_type": "EDGE_CURVE", "numeric_params": []},
        ]

        graph = builder.build_reference_graph(entities)
        features = builder.build_node_features(entities, graph)

        # Last dimension should have type hash (different for different types)
        idx_1 = graph["id_to_idx"][1]
        idx_2 = graph["id_to_idx"][2]
        assert features[idx_1, -1] != features[idx_2, -1]


# ============================================================================
# SECTION 5: Compact Node Features (48-dim) Tests
# ============================================================================


class TestCompactNodeFeatures:
    """Test compact 48-dimensional node features."""

    def test_compact_features_shape(self) -> None:
        """Test compact features have shape [num_nodes, 48]."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        entities = [
            {"entity_id": i, "entity_type": "ADVANCED_FACE", "numeric_params": []}
            for i in range(5)
        ]

        graph = builder.build_reference_graph(entities)
        features = builder.build_compact_node_features(entities, graph)

        assert features.shape == (5, 48)

    def test_compact_features_numeric_slots(self) -> None:
        """Test first 32 dims contain numeric parameters."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        params = list(range(32))
        entities = [{"entity_id": 1, "entity_type": "ADVANCED_FACE", "numeric_params": params}]

        graph = builder.build_reference_graph(entities)
        features = builder.build_compact_node_features(entities, graph)

        idx = graph["id_to_idx"][1]
        # First 32 should be numeric params
        expected = torch.tensor(params, dtype=torch.float32)
        assert torch.allclose(features[idx, :32], expected)

    def test_compact_features_type_onehot(self) -> None:
        """Test last 16 dims contain type one-hot encoding."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        # ADVANCED_FACE is in the known types list
        entities = [{"entity_id": 1, "entity_type": "ADVANCED_FACE", "numeric_params": []}]

        graph = builder.build_reference_graph(entities)
        features = builder.build_compact_node_features(entities, graph)

        # Last 16 dims should have exactly one 1.0 (one-hot)
        type_encoding = features[0, 32:]
        assert type_encoding.sum() == 1.0
        # ADVANCED_FACE is at index 0 in _BREP_TYPES
        assert type_encoding[0] == 1.0

    def test_compact_features_unknown_type(self) -> None:
        """Test unknown entity types have zero one-hot encoding."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        entities = [{"entity_id": 1, "entity_type": "UNKNOWN_TYPE", "numeric_params": []}]

        graph = builder.build_reference_graph(entities)
        features = builder.build_compact_node_features(entities, graph)

        # Last 16 dims should be all zeros for unknown type
        type_encoding = features[0, 32:]
        assert type_encoding.sum() == 0.0


# ============================================================================
# SECTION 6: Coedge Structure Tests (BRepNet)
# ============================================================================


class TestCoedgeStructure:
    """Test coedge structure for BRepNet-style processing."""

    def test_coedge_structure_from_face_edge_entities(self) -> None:
        """Test coedge structure built from face/edge entities."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        # Mock STEP data with face-edge hierarchy
        entities = [
            {"entity_id": 1, "entity_type": "ADVANCED_FACE", "references": [2]},
            {"entity_id": 2, "entity_type": "FACE_BOUND", "references": [3]},
            {"entity_id": 3, "entity_type": "EDGE_LOOP", "references": [4, 5, 6]},
            {"entity_id": 4, "entity_type": "ORIENTED_EDGE", "references": [7]},
            {"entity_id": 5, "entity_type": "ORIENTED_EDGE", "references": [8]},
            {"entity_id": 6, "entity_type": "ORIENTED_EDGE", "references": [9]},
            {"entity_id": 7, "entity_type": "EDGE_CURVE", "references": [], "numeric_params": [1.0]},
            {"entity_id": 8, "entity_type": "EDGE_CURVE", "references": [], "numeric_params": [2.0]},
            {"entity_id": 9, "entity_type": "EDGE_CURVE", "references": [], "numeric_params": [3.0]},
        ]

        structure = builder.build_coedge_structure(entities)

        assert "coedge_features" in structure
        assert "next_indices" in structure
        assert "prev_indices" in structure
        assert "mate_indices" in structure
        assert "face_indices" in structure
        assert "num_coedges" in structure
        assert "num_faces" in structure

    def test_coedge_features_shape(self) -> None:
        """Test coedge features have 12 dimensions."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        entities = [
            {"entity_id": 1, "entity_type": "ADVANCED_FACE", "references": [2]},
            {"entity_id": 2, "entity_type": "FACE_BOUND", "references": [3]},
            {"entity_id": 3, "entity_type": "EDGE_LOOP", "references": [4]},
            {"entity_id": 4, "entity_type": "ORIENTED_EDGE", "references": [5]},
            {"entity_id": 5, "entity_type": "EDGE_CURVE", "references": [], "numeric_params": [1.0]},
        ]

        structure = builder.build_coedge_structure(entities)

        if structure["num_coedges"] > 0:
            assert structure["coedge_features"].shape[-1] == 12

    def test_coedge_empty_topology(self) -> None:
        """Test coedge structure handles empty topology."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        # No faces or edges
        entities = [
            {"entity_id": 1, "entity_type": "VERTEX_POINT", "references": []},
        ]

        structure = builder.build_coedge_structure(entities)

        assert structure["num_coedges"] == 0
        assert structure["coedge_features"].shape == (0, 12)


# ============================================================================
# SECTION 7: Complete Topology Build Tests
# ============================================================================


class TestCompleteTopology:
    """Test complete topology building."""

    def test_build_complete_topology_all_fields(self) -> None:
        """Test complete topology has all required fields."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        entities = [
            {"entity_id": 1, "entity_type": "ADVANCED_FACE", "references": [2], "numeric_params": [1.0]},
            {"entity_id": 2, "entity_type": "EDGE_CURVE", "references": [3], "numeric_params": [2.0]},
            {"entity_id": 3, "entity_type": "VERTEX_POINT", "references": [], "numeric_params": [3.0]},
        ]

        topology = builder.build_complete_topology(entities)

        assert "reference_graph" in topology
        assert "adjacency_matrix" in topology
        assert "edge_index" in topology
        assert "node_degrees" in topology
        assert "topology_types" in topology
        assert "node_features" in topology
        assert "num_nodes" in topology
        assert "num_edges" in topology

    def test_build_complete_topology_compact_mode(self) -> None:
        """Test complete topology in compact mode (48-dim features)."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        entities = [
            {"entity_id": i, "entity_type": "ADVANCED_FACE", "references": [], "numeric_params": []}
            for i in range(5)
        ]

        topology = builder.build_complete_topology(entities, compact=True)

        assert topology["node_features"].shape[-1] == 48

    def test_build_complete_topology_legacy_mode(self) -> None:
        """Test complete topology in legacy mode (129-dim features)."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        entities = [
            {"entity_id": i, "entity_type": "ADVANCED_FACE", "references": [], "numeric_params": []}
            for i in range(5)
        ]

        topology = builder.build_complete_topology(entities, compact=False)

        assert topology["node_features"].shape[-1] == 129


# ============================================================================
# SECTION 8: Node Degrees Tests
# ============================================================================


class TestNodeDegrees:
    """Test node degree computation."""

    def test_compute_node_degrees(self) -> None:
        """Test in/out degree computation."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        entities = [
            {"entity_id": 1, "entity_type": "A", "references": [2, 3]},  # out: 2
            {"entity_id": 2, "entity_type": "B", "references": [3]},     # out: 1, in: 1
            {"entity_id": 3, "entity_type": "C", "references": []},      # in: 2
        ]

        graph = builder.build_reference_graph(entities)
        degrees = builder.compute_node_degrees(graph)

        assert degrees[1]["out_degree"] == 2
        assert degrees[1]["in_degree"] == 0
        assert degrees[2]["out_degree"] == 1
        assert degrees[2]["in_degree"] == 1
        assert degrees[3]["out_degree"] == 0
        assert degrees[3]["in_degree"] == 2


# ============================================================================
# SECTION 9: Topology Types Tests
# ============================================================================


class TestTopologyTypes:
    """Test entity type categorization."""

    def test_identify_topology_types(self) -> None:
        """Test entities are categorized by topological role."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        entities = [
            {"entity_id": 1, "entity_type": "ADVANCED_FACE", "references": []},
            {"entity_id": 2, "entity_type": "EDGE_CURVE", "references": []},
            {"entity_id": 3, "entity_type": "VERTEX_POINT", "references": []},
            {"entity_id": 4, "entity_type": "CLOSED_SHELL", "references": []},
            {"entity_id": 5, "entity_type": "PLANE", "references": []},
        ]

        types = builder.identify_topology_types(entities)

        assert "faces" in types
        assert "edges" in types
        assert "vertices" in types
        assert "shells" in types
        assert "geometry" in types

        assert 1 in types["faces"]
        assert 2 in types["edges"]
        assert 3 in types["vertices"]
        assert 4 in types["shells"]
        assert 5 in types["geometry"]


# ============================================================================
# SECTION 10: Cadling Conversion Tests
# ============================================================================


class TestCadlingConversion:
    """Test conversion to/from cadling TopologyGraph format."""

    def test_to_cadling_topology_graph_requires_cadling(self) -> None:
        """Test conversion to cadling TopologyGraph requires cadling installed."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        entities = [
            {"entity_id": 1, "entity_type": "ADVANCED_FACE", "references": [2], "numeric_params": []},
            {"entity_id": 2, "entity_type": "EDGE_CURVE", "references": [], "numeric_params": []},
        ]

        internal_topology = builder.build_complete_topology(entities, compact=True)

        try:
            cadling_graph = STEPTopologyBuilder.to_cadling_topology_graph(internal_topology)
            # If cadling is installed, check basic structure
            assert cadling_graph is not None
            assert hasattr(cadling_graph, 'num_nodes')
            assert hasattr(cadling_graph, 'adjacency_list')
        except ImportError:
            # Expected if cadling is not installed
            pass

    def test_topology_dict_has_required_keys(self) -> None:
        """Test topology dict has keys needed for cadling conversion."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        entities = [
            {"entity_id": 1, "entity_type": "ADVANCED_FACE", "references": [2], "numeric_params": []},
            {"entity_id": 2, "entity_type": "EDGE_CURVE", "references": [], "numeric_params": []},
        ]

        topology = builder.build_complete_topology(entities, compact=True)

        # Keys needed for to_cadling_topology_graph
        assert "adjacency_matrix" in topology
        assert "node_features" in topology
        assert isinstance(topology["adjacency_matrix"], torch.Tensor)
        assert isinstance(topology["node_features"], torch.Tensor)


# ============================================================================
# SECTION 11: Module Import Tests
# ============================================================================


class TestTopologyModuleImports:
    """Test topology module imports."""

    def test_import_topology_builder(self) -> None:
        """Test STEPTopologyBuilder can be imported."""
        from stepnet.topology import STEPTopologyBuilder
        assert STEPTopologyBuilder is not None

    def test_builder_instantiation(self) -> None:
        """Test builder can be instantiated."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()
        assert builder is not None


# ============================================================================
# SECTION 12: Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_entities(self) -> None:
        """Test handling of empty entity list."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        graph = builder.build_reference_graph([])

        assert graph["num_nodes"] == 0
        assert len(graph["edge_list"]) == 0

    def test_single_node(self) -> None:
        """Test topology with single node."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        entities = [{"entity_id": 1, "entity_type": "VERTEX_POINT", "references": [], "numeric_params": []}]

        topology = builder.build_complete_topology(entities)

        assert topology["num_nodes"] == 1
        assert topology["adjacency_matrix"].shape == (1, 1)

    def test_large_graph(self) -> None:
        """Test handling of large graphs."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        # Large graph
        num_nodes = 1000
        entities = [
            {"entity_id": i, "entity_type": "NODE", "references": [(i + 1) % num_nodes], "numeric_params": []}
            for i in range(num_nodes)
        ]

        topology = builder.build_complete_topology(entities)

        assert topology is not None
        assert topology["node_features"].shape[0] == num_nodes

    def test_missing_references_key(self) -> None:
        """Test handling of entities without references key."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        # Missing 'references' key - should use empty list
        entities = [
            {"entity_id": 1, "entity_type": "FACE"},
            {"entity_id": 2, "entity_type": "EDGE"},
        ]

        graph = builder.build_reference_graph(entities)

        # Should handle gracefully
        assert graph["num_nodes"] == 2
        assert len(graph["edge_list"]) == 0

    def test_self_referencing_entity(self) -> None:
        """Test handling of self-referencing entities."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        entities = [
            {"entity_id": 1, "entity_type": "A", "references": [1]},  # Self-reference
        ]

        graph = builder.build_reference_graph(entities)

        # Should handle self-reference
        assert graph is not None
        assert (1, 1) in graph["edge_list"]


class TestNonManifoldCoedge:
    """Test coedge prev/next pointers with duplicate edges (non-manifold geometry)."""

    def test_duplicate_edge_in_face_produces_correct_prev_next(self) -> None:
        """When the same edge appears twice in a face loop, prev/next must
        follow positional order, not collapse to the same coedge index."""
        from stepnet.topology import STEPTopologyBuilder

        builder = STEPTopologyBuilder()

        # ADVANCED_FACE #10 directly references 3 EDGE_CURVEs: [100, 200, 100].
        # Edge 100 appears at positions 0 AND 2 — non-manifold / seam edge.
        entities = [
            {"entity_id": 10, "entity_type": "ADVANCED_FACE", "references": [100, 200, 100]},
            {"entity_id": 100, "entity_type": "EDGE_CURVE", "references": [], "numeric_params": [1.0]},
            {"entity_id": 200, "entity_type": "EDGE_CURVE", "references": [], "numeric_params": [2.0]},
        ]

        structure = builder.build_coedge_structure(entities)

        n = structure["num_coedges"]
        assert n == 3, f"Expected 3 coedges, got {n}"

        next_idx = structure["next_indices"]
        prev_idx = structure["prev_indices"]

        # Each coedge must have distinct prev and next (no self-loops in a 3-edge loop)
        for i in range(n):
            assert next_idx[i].item() != i, (
                f"Coedge {i} next points to itself"
            )
            assert prev_idx[i].item() != i, (
                f"Coedge {i} prev points to itself"
            )

        # Verify cyclic consistency: following next 3 times returns to start
        visited = set()
        cur = 0
        for _ in range(n):
            visited.add(cur)
            cur = next_idx[cur].item()
        assert cur == 0, "next chain is not a proper cycle"
        assert len(visited) == n, "next chain does not visit all coedges"
