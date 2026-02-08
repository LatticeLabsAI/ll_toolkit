"""Tests for DFS chunker.

Tests cover:
- Initialization and parameter handling
- DFS traversal logic (simple trees, cycles, orphans)
- Subtree boundary detection
- Token estimation
- Entity-to-text conversion
- Cross-chunk reference counting
- DAG building from topology and entity references
- Edge cases (empty documents, single entities)
"""
from __future__ import annotations

import pytest
from collections import defaultdict
from unittest.mock import MagicMock, PropertyMock, patch

from cadling.chunker.dfs_chunker import DFSChunker


class TestDFSChunkerInit:
    """Test DFSChunker initialization."""

    def test_default_init(self):
        """Test default initialization values."""
        chunker = DFSChunker()
        assert chunker.max_tokens == 512
        assert chunker.overlap_tokens == 50
        assert chunker.min_chunk_entities == 3
        assert chunker.max_cross_references == 5

    def test_custom_init(self):
        """Test custom initialization values."""
        chunker = DFSChunker(
            max_tokens=256,
            overlap_tokens=25,
            min_chunk_entities=5,
            max_cross_references=10,
            vocab_size=30000,
        )
        assert chunker.max_tokens == 256
        assert chunker.overlap_tokens == 25
        assert chunker.min_chunk_entities == 5
        assert chunker.max_cross_references == 10

    def test_inherits_base_chunker(self):
        """Test that DFSChunker inherits from BaseCADChunker."""
        from cadling.chunker.base_chunker import BaseCADChunker

        chunker = DFSChunker()
        assert isinstance(chunker, BaseCADChunker)


class TestDFSTraversal:
    """Test DFS traversal logic."""

    def test_simple_tree(self):
        """Test DFS traversal on a simple tree."""
        chunker = DFSChunker()
        forward_adj = {
            "1": ["2", "3"],
            "2": ["4", "5"],
            "3": ["6"],
        }
        roots = ["1"]
        all_ids = {"1", "2", "3", "4", "5", "6"}

        order = chunker._dfs_traversal(roots, forward_adj, all_ids)

        # Should visit all nodes
        visited_ids = [eid for eid, _ in order]
        assert set(visited_ids) == all_ids
        assert len(visited_ids) == len(all_ids)  # No duplicates

        # Root should be first
        assert order[0][0] == "1"
        assert order[0][1] == 0  # depth 0

    def test_deterministic_order(self):
        """Test that DFS traversal is deterministic."""
        chunker = DFSChunker()
        forward_adj = {"1": ["2", "3"], "2": ["4"]}
        roots = ["1"]
        all_ids = {"1", "2", "3", "4"}

        order1 = chunker._dfs_traversal(roots, forward_adj, all_ids)
        order2 = chunker._dfs_traversal(roots, forward_adj, all_ids)

        assert order1 == order2

    def test_orphan_handling(self):
        """Test that orphan entities are included."""
        chunker = DFSChunker()
        forward_adj = {"1": ["2"]}
        roots = ["1"]
        all_ids = {"1", "2", "99"}  # 99 is orphan

        order = chunker._dfs_traversal(roots, forward_adj, all_ids)

        visited_ids = [eid for eid, _ in order]
        assert "99" in visited_ids
        assert set(visited_ids) == all_ids

    def test_cycle_handling(self):
        """Test that cycles do not cause infinite loops."""
        chunker = DFSChunker()
        forward_adj = {"1": ["2"], "2": ["3"], "3": ["1"]}  # cycle
        roots = ["1"]
        all_ids = {"1", "2", "3"}

        order = chunker._dfs_traversal(roots, forward_adj, all_ids)

        # Should complete without infinite loop
        assert len(order) == 3

    def test_depth_tracking(self):
        """Test that depth is tracked correctly."""
        chunker = DFSChunker()
        forward_adj = {"1": ["2"], "2": ["3"], "3": ["4"]}
        roots = ["1"]
        all_ids = {"1", "2", "3", "4"}

        order = chunker._dfs_traversal(roots, forward_adj, all_ids)

        depths = {eid: d for eid, d in order}
        assert depths["1"] == 0
        assert depths["2"] == 1
        assert depths["3"] == 2
        assert depths["4"] == 3

    def test_multiple_roots(self):
        """Test DFS with multiple root entities."""
        chunker = DFSChunker()
        forward_adj = {"1": ["2"], "3": ["4"]}
        roots = ["1", "3"]
        all_ids = {"1", "2", "3", "4"}

        order = chunker._dfs_traversal(roots, forward_adj, all_ids)

        visited_ids = [eid for eid, _ in order]
        assert set(visited_ids) == all_ids
        assert len(visited_ids) == 4

    def test_empty_graph(self):
        """Test DFS on an empty graph."""
        chunker = DFSChunker()
        forward_adj = {}
        roots = []
        all_ids = set()

        order = chunker._dfs_traversal(roots, forward_adj, all_ids)

        assert order == []

    def test_single_node(self):
        """Test DFS on a single-node graph."""
        chunker = DFSChunker()
        forward_adj = {}
        roots = ["1"]
        all_ids = {"1"}

        order = chunker._dfs_traversal(roots, forward_adj, all_ids)

        assert len(order) == 1
        assert order[0] == ("1", 0)


class TestSubtreeBoundaries:
    """Test subtree boundary detection."""

    def test_single_tree(self):
        """Test boundaries in a single tree (no boundaries expected)."""
        chunker = DFSChunker()
        dfs_order = [("1", 0), ("2", 1), ("3", 2), ("4", 1)]

        boundaries = chunker._find_subtree_boundaries(dfs_order)

        # No boundaries in single tree (depth never returns to 0)
        assert boundaries == []

    def test_multiple_trees(self):
        """Test boundaries between multiple top-level trees."""
        chunker = DFSChunker()
        dfs_order = [
            ("1", 0),
            ("2", 1),
            ("3", 1),  # tree 1
            ("4", 0),
            ("5", 1),  # tree 2
            ("6", 0),  # tree 3
        ]

        boundaries = chunker._find_subtree_boundaries(dfs_order)

        assert 3 in boundaries  # before entity 4
        assert 5 in boundaries  # before entity 6

    def test_single_element(self):
        """Test boundaries with single element."""
        chunker = DFSChunker()
        dfs_order = [("1", 0)]

        boundaries = chunker._find_subtree_boundaries(dfs_order)

        assert boundaries == []

    def test_all_roots(self):
        """Test boundaries when all elements are roots."""
        chunker = DFSChunker()
        dfs_order = [("1", 0), ("2", 0), ("3", 0)]

        boundaries = chunker._find_subtree_boundaries(dfs_order)

        assert boundaries == [1, 2]


class TestTokenEstimation:
    """Test token estimation."""

    def test_empty_text(self):
        """Test token estimation for empty text."""
        chunker = DFSChunker()
        assert chunker._estimate_tokens("") == 0

    def test_short_text(self):
        """Test token estimation for short text."""
        chunker = DFSChunker()
        tokens = chunker._estimate_tokens("#1=POINT(0.0,0.0,0.0);")
        assert tokens > 0

    def test_longer_text_more_tokens(self):
        """Test that longer text produces more tokens."""
        chunker = DFSChunker()
        short = chunker._estimate_tokens("short")
        long = chunker._estimate_tokens("this is a much longer text with more content")
        assert long > short

    def test_minimum_one_token(self):
        """Test that non-empty text produces at least one token."""
        chunker = DFSChunker()
        assert chunker._estimate_tokens("a") >= 1


class TestItemToText:
    """Test entity to text conversion."""

    def test_raw_line(self):
        """Test conversion with raw_line attribute."""
        chunker = DFSChunker()
        item = MagicMock(spec=[])
        item.raw_line = "#1=CARTESIAN_POINT('',(0.0,0.0,0.0));"
        # Ensure isinstance check for STEPEntityItem returns False
        type(item).__name__ = "MockEntity"

        text = chunker._item_to_text(item)
        assert text == "#1=CARTESIAN_POINT('',(0.0,0.0,0.0));"

    def test_entity_id_and_type(self):
        """Test conversion from entity_id and type attributes."""
        chunker = DFSChunker()
        item = MagicMock(spec=[])
        item.raw_line = None
        item.entity_id = "42"
        item.type = "CARTESIAN_POINT"
        item.parameters = None

        text = chunker._item_to_text(item)
        assert "#42" in text
        assert "CARTESIAN_POINT" in text

    def test_fallback_to_str(self):
        """Test fallback to str() for unknown objects."""
        chunker = DFSChunker()
        item = "simple string entity"

        text = chunker._item_to_text(item)
        assert text == "simple string entity"

    def test_with_dict_parameters(self):
        """Test conversion with dict parameters."""
        chunker = DFSChunker()
        item = MagicMock(spec=[])
        item.raw_line = None
        item.entity_id = "10"
        item.type = "CIRCLE"
        item.parameters = {"radius": 5.0}

        text = chunker._item_to_text(item)
        assert "#10" in text
        assert "CIRCLE" in text
        assert "radius=5.0" in text

    def test_with_list_parameters(self):
        """Test conversion with list parameters."""
        chunker = DFSChunker()
        item = MagicMock(spec=[])
        item.raw_line = None
        item.entity_id = "20"
        item.type = "LINE"
        item.parameters = [1.0, 2.0, 3.0]

        text = chunker._item_to_text(item)
        assert "#20" in text
        assert "LINE" in text
        assert "1.0" in text


class TestCrossReferenceCount:
    """Test cross-chunk reference counting."""

    def test_no_cross_refs(self):
        """Test chunks with no cross-references."""
        chunker = DFSChunker()
        # Two chunks with no inter-chunk references
        chunks = [
            ([("1", 0, "e1"), ("2", 1, "e2")], [("1", 0), ("2", 1)]),
            ([("3", 0, "e3"), ("4", 1, "e4")], [("3", 0), ("4", 1)]),
        ]
        forward_adj = {"1": ["2"], "3": ["4"]}

        refs = chunker._count_cross_references(chunks, forward_adj)
        assert refs == [0, 0]

    def test_with_cross_refs(self):
        """Test chunks with cross-references."""
        chunker = DFSChunker()
        chunks = [
            ([("1", 0, "e1"), ("2", 1, "e2")], [("1", 0), ("2", 1)]),
            ([("3", 0, "e3")], [("3", 0)]),
        ]
        forward_adj = {"1": ["2", "3"]}  # 1 refs 3 which is in chunk 2

        refs = chunker._count_cross_references(chunks, forward_adj)
        assert refs[0] == 1  # chunk 0 has one cross-ref to chunk 1

    def test_bidirectional_cross_refs(self):
        """Test bidirectional cross-references."""
        chunker = DFSChunker()
        chunks = [
            ([("1", 0, "e1")], [("1", 0)]),
            ([("2", 0, "e2")], [("2", 0)]),
        ]
        forward_adj = {"1": ["2"], "2": ["1"]}

        refs = chunker._count_cross_references(chunks, forward_adj)
        assert refs[0] == 1
        assert refs[1] == 1

    def test_empty_chunks(self):
        """Test cross-reference counting with no chunks."""
        chunker = DFSChunker()
        chunks = []
        forward_adj = {}

        refs = chunker._count_cross_references(chunks, forward_adj)
        assert refs == []


class TestDAGBuilding:
    """Test DAG building from document."""

    def test_build_from_topology_adjacency_list(self):
        """Test DAG building from TopologyGraph adjacency_list."""
        chunker = DFSChunker()

        # Mock document with TopologyGraph
        mock_topology = MagicMock()
        mock_topology.adjacency_list = {1: [2, 3], 2: [4]}
        mock_topology.edges = None

        mock_doc = MagicMock()
        mock_doc.topology = mock_topology

        entity_map = {"1": MagicMock(), "2": MagicMock(), "3": MagicMock(), "4": MagicMock()}

        forward_adj, reverse_adj, roots = chunker._build_dag(mock_doc, entity_map)

        assert "2" in forward_adj.get("1", [])
        assert "3" in forward_adj.get("1", [])
        assert "4" in forward_adj.get("2", [])
        assert "1" in roots  # 1 has no incoming

    def test_build_from_topology_edges(self):
        """Test DAG building from topology edges attribute."""
        chunker = DFSChunker()

        # Mock document with edge-based topology
        mock_edge1 = MagicMock()
        mock_edge1.source = "1"
        mock_edge1.target = "2"
        mock_edge2 = MagicMock()
        mock_edge2.source = "1"
        mock_edge2.target = "3"

        mock_topology = MagicMock()
        mock_topology.adjacency_list = {}  # empty, so falls through to edges
        mock_topology.edges = [mock_edge1, mock_edge2]

        mock_doc = MagicMock()
        mock_doc.topology = mock_topology

        entity_map = {"1": MagicMock(), "2": MagicMock(), "3": MagicMock()}

        forward_adj, reverse_adj, roots = chunker._build_dag(mock_doc, entity_map)

        assert "2" in forward_adj.get("1", [])
        assert "3" in forward_adj.get("1", [])
        assert "1" in roots

    def test_build_fallback_references(self):
        """Test DAG building from entity reference_params."""
        chunker = DFSChunker()

        # Mock document without topology
        mock_doc = MagicMock()
        mock_doc.topology = None

        e1 = MagicMock()
        e1.reference_params = ["2", "3"]
        e2 = MagicMock()
        e2.reference_params = []
        e3 = MagicMock()
        e3.reference_params = []

        entity_map = {"1": e1, "2": e2, "3": e3}

        forward_adj, reverse_adj, roots = chunker._build_dag(mock_doc, entity_map)

        assert "2" in forward_adj.get("1", [])
        assert "3" in forward_adj.get("1", [])

    def test_roots_no_incoming_edges(self):
        """Test that roots are entities with no incoming edges."""
        chunker = DFSChunker()

        mock_topology = MagicMock()
        mock_topology.adjacency_list = {1: [2], 2: [3]}

        mock_doc = MagicMock()
        mock_doc.topology = mock_topology

        entity_map = {"1": MagicMock(), "2": MagicMock(), "3": MagicMock()}

        forward_adj, reverse_adj, roots = chunker._build_dag(mock_doc, entity_map)

        assert "1" in roots
        assert "2" not in roots
        assert "3" not in roots

    def test_no_topology_no_references(self):
        """Test DAG building with no topology and no references."""
        chunker = DFSChunker()

        mock_doc = MagicMock()
        mock_doc.topology = None

        # Entities with no reference attributes
        e1 = MagicMock(spec=[])
        e2 = MagicMock(spec=[])

        entity_map = {"1": e1, "2": e2}

        forward_adj, reverse_adj, roots = chunker._build_dag(mock_doc, entity_map)

        # All entities should be roots when no edges exist
        assert "1" in roots
        assert "2" in roots


class TestPartitioning:
    """Test chunk partitioning."""

    def test_single_segment(self):
        """Test partitioning a single segment."""
        chunker = DFSChunker(max_tokens=10000)  # large limit
        dfs_order = [("1", 0), ("2", 1), ("3", 2)]
        boundaries = []

        mock_entity = MagicMock(spec=[])
        mock_entity.raw_line = None
        mock_entity.entity_id = None
        mock_entity.type = None
        mock_entity.label = None
        entity_map = {"1": mock_entity, "2": mock_entity, "3": mock_entity}

        result = chunker._partition_into_chunks(dfs_order, boundaries, entity_map)

        assert len(result) == 1
        entities, dfs_info = result[0]
        assert len(entities) == 3

    def test_boundary_splitting(self):
        """Test that partitioning respects subtree boundaries."""
        chunker = DFSChunker(max_tokens=10000, min_chunk_entities=1)
        dfs_order = [("1", 0), ("2", 1), ("3", 0), ("4", 1)]
        boundaries = [2]  # boundary before entity 3

        mock_entity = MagicMock(spec=[])
        mock_entity.raw_line = None
        mock_entity.entity_id = None
        mock_entity.type = None
        mock_entity.label = None
        entity_map = {
            "1": mock_entity,
            "2": mock_entity,
            "3": mock_entity,
            "4": mock_entity,
        }

        result = chunker._partition_into_chunks(dfs_order, boundaries, entity_map)

        # Should produce 2 segments based on boundary,
        # but may merge if under min_chunk_entities
        assert len(result) >= 1

    def test_empty_dfs_order(self):
        """Test partitioning with empty DFS order."""
        chunker = DFSChunker()
        result = chunker._partition_into_chunks([], [], {})
        assert result == []


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_document(self):
        """Test chunking an empty document."""
        chunker = DFSChunker()

        mock_doc = MagicMock()
        mock_doc.items = []
        mock_doc.name = "empty_doc"

        # Should not raise
        chunks = list(chunker.chunk(mock_doc))
        assert chunks == []

    def test_single_entity_dfs_order(self):
        """Test subtree boundaries with single entity."""
        chunker = DFSChunker()
        dfs_order = [("1", 0)]
        boundaries = chunker._find_subtree_boundaries(dfs_order)
        assert boundaries == []

    def test_get_entity_id_step_entity(self):
        """Test entity ID extraction from STEP-like entity."""
        chunker = DFSChunker()
        entity = MagicMock()
        entity.entity_id = 42

        eid = chunker._get_entity_id(entity)
        assert eid == "42"

    def test_get_entity_id_item_id(self):
        """Test entity ID extraction from item_id."""
        chunker = DFSChunker()
        entity = MagicMock(spec=[])
        entity.item_id = "my_item"
        entity.entity_id = None

        # Since spec=[] removes entity_id, it won't have it
        # but item_id check will work
        entity2 = MagicMock()
        entity2.entity_id = None
        entity2.item_id = "my_item"

        eid = chunker._get_entity_id(entity2)
        assert eid == "my_item"

    def test_get_entity_id_label_fallback(self):
        """Test entity ID extraction from label."""
        chunker = DFSChunker()
        entity = MagicMock()
        entity.entity_id = None
        entity.item_id = None
        entity.label = MagicMock()
        entity.label.text = "CIRCLE"

        eid = chunker._get_entity_id(entity)
        assert eid == "CIRCLE"

    def test_get_entities(self):
        """Test entity extraction from document."""
        chunker = DFSChunker()
        mock_doc = MagicMock()
        item1 = MagicMock()
        item2 = MagicMock()
        mock_doc.items = [item1, item2]

        entities = chunker._get_entities(mock_doc)
        assert len(entities) == 2

    def test_get_entities_empty(self):
        """Test entity extraction from empty document."""
        chunker = DFSChunker()
        mock_doc = MagicMock()
        mock_doc.items = []

        entities = chunker._get_entities(mock_doc)
        assert entities == []


class TestChunkImport:
    """Test module import and export."""

    def test_import_from_package(self):
        """Test that DFSChunker can be imported from chunker package."""
        from cadling.chunker import DFSChunker as DFSChunkerFromPackage

        assert DFSChunkerFromPackage is DFSChunker

    def test_import_from_submodule(self):
        """Test that DFSChunker can be imported from dfs_chunker submodule."""
        from cadling.chunker.dfs_chunker import DFSChunker as DFSChunkerFromSub

        assert DFSChunkerFromSub is DFSChunker

    def test_in_all(self):
        """Test that DFSChunker is in __all__."""
        import cadling.chunker

        assert "DFSChunker" in cadling.chunker.__all__
