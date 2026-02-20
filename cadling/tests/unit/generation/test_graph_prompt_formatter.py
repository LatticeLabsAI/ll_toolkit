"""Unit tests for graph prompt formatter module.

Tests the GraphToPromptFormatter that converts decoded graph features
into structured prompts for LLM-based CadQuery code generation.
"""

from __future__ import annotations

import pytest
import numpy as np


class TestGraphPromptFormatterImports:
    """Test module imports."""

    def test_module_imports(self):
        """Test that the module imports successfully."""
        from cadling.generation.codegen.graph_prompt_formatter import (
            GraphToPromptFormatter,
            SURFACE_TYPES,
            CURVE_TYPES,
            STEP_ENTITY_TYPES,
        )

        assert GraphToPromptFormatter is not None
        assert len(SURFACE_TYPES) == 10
        assert len(CURVE_TYPES) == 6


class TestGraphToPromptFormatterInit:
    """Test GraphToPromptFormatter initialization."""

    def test_default_init(self):
        """Test default initialization."""
        from cadling.generation.codegen.graph_prompt_formatter import (
            GraphToPromptFormatter,
        )

        formatter = GraphToPromptFormatter()

        assert formatter.include_coordinates is True
        assert formatter.include_dimensions is True
        assert formatter.max_faces_detailed == 20
        assert formatter.decimal_places == 3

    def test_custom_init(self):
        """Test custom initialization."""
        from cadling.generation.codegen.graph_prompt_formatter import (
            GraphToPromptFormatter,
        )

        formatter = GraphToPromptFormatter(
            include_coordinates=False,
            include_dimensions=False,
            max_faces_detailed=5,
            decimal_places=2,
        )

        assert formatter.include_coordinates is False
        assert formatter.include_dimensions is False
        assert formatter.max_faces_detailed == 5
        assert formatter.decimal_places == 2


class TestGraphToPromptFormatterFormatting:
    """Test prompt formatting methods."""

    def test_format_for_cadquery_basic(self):
        """Test basic prompt formatting."""
        from cadling.generation.codegen.graph_prompt_formatter import (
            GraphToPromptFormatter,
        )

        formatter = GraphToPromptFormatter()

        # Create simple node features (3 faces: 2 planes, 1 cylinder)
        node_features = np.zeros((3, 24), dtype=np.float32)
        # Face 0: PLANE
        node_features[0, 0] = 1.0  # surface_type = PLANE
        node_features[0, 10] = 10.0  # area
        # Face 1: PLANE
        node_features[1, 0] = 1.0  # surface_type = PLANE
        node_features[1, 10] = 10.0  # area
        # Face 2: CYLINDER
        node_features[2, 1] = 1.0  # surface_type = CYLINDER
        node_features[2, 10] = 15.0  # area

        prompt = formatter.format_for_cadquery(node_features)

        # Check that prompt contains expected sections
        assert "You are generating CadQuery code" in prompt
        assert "Number of faces: 3" in prompt
        assert "Entity Summary" in prompt
        assert "PLANE" in prompt
        assert "CYLINDER" in prompt
        assert "Face Details" in prompt
        assert "CadQuery Generation Instructions" in prompt

    def test_format_for_cadquery_with_metadata(self):
        """Test prompt formatting with metadata."""
        from cadling.generation.codegen.graph_prompt_formatter import (
            GraphToPromptFormatter,
        )

        formatter = GraphToPromptFormatter()

        node_features = np.zeros((2, 24), dtype=np.float32)
        node_features[0, 0] = 1.0  # PLANE
        node_features[1, 0] = 1.0  # PLANE

        metadata = {
            "volume": 1.0,
            "surface_area": 6.0,
            "bbox_dimensions": [1.0, 1.0, 1.0],
        }

        prompt = formatter.format_for_cadquery(node_features, metadata=metadata)

        assert "volume" in prompt.lower()
        assert "surface area" in prompt.lower()
        assert "bounding box" in prompt.lower()

    def test_format_for_cadquery_with_adjacency(self):
        """Test prompt formatting with adjacency."""
        from cadling.generation.codegen.graph_prompt_formatter import (
            GraphToPromptFormatter,
        )

        formatter = GraphToPromptFormatter()

        node_features = np.zeros((3, 24), dtype=np.float32)
        node_features[0, 0] = 1.0  # PLANE
        node_features[1, 0] = 1.0  # PLANE
        node_features[2, 1] = 1.0  # CYLINDER

        adjacency = {
            0: [1, 2],
            1: [0, 2],
            2: [0, 1],
        }

        prompt = formatter.format_for_cadquery(
            node_features,
            adjacency=adjacency,
        )

        assert "Topology Relationships" in prompt
        assert "adjacency" in prompt.lower() or "connect" in prompt.lower()

    def test_format_for_cadquery_with_edge_index(self):
        """Test prompt formatting with edge index."""
        from cadling.generation.codegen.graph_prompt_formatter import (
            GraphToPromptFormatter,
        )

        formatter = GraphToPromptFormatter()

        node_features = np.zeros((2, 24), dtype=np.float32)
        node_features[0, 0] = 1.0  # PLANE
        node_features[1, 0] = 1.0  # PLANE

        edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)

        prompt = formatter.format_for_cadquery(
            node_features,
            edge_index=edge_index,
        )

        assert "Topology Relationships" in prompt


class TestGraphToPromptFormatterHelpers:
    """Test helper methods."""

    def test_decode_surface_type(self):
        """Test surface type decoding from one-hot."""
        from cadling.generation.codegen.graph_prompt_formatter import (
            GraphToPromptFormatter,
        )

        formatter = GraphToPromptFormatter()

        # PLANE
        features = np.zeros(24)
        features[0] = 1.0
        assert formatter._decode_surface_type(features) == "PLANE"

        # CYLINDER
        features = np.zeros(24)
        features[1] = 1.0
        assert formatter._decode_surface_type(features) == "CYLINDER"

        # SPHERE
        features = np.zeros(24)
        features[3] = 1.0
        assert formatter._decode_surface_type(features) == "SPHERE"

        # OTHER (no clear winner)
        features = np.zeros(24)
        features[9] = 1.0
        assert formatter._decode_surface_type(features) == "OTHER"

    def test_describe_face(self):
        """Test face description generation."""
        from cadling.generation.codegen.graph_prompt_formatter import (
            GraphToPromptFormatter,
        )

        formatter = GraphToPromptFormatter()

        features = np.zeros(24)
        features[0] = 1.0  # PLANE
        features[10] = 10.5  # area
        features[11] = 0.0  # gaussian curvature
        features[12] = 0.0  # mean curvature
        features[16:19] = [1.0, 2.0, 3.0]  # centroid

        desc = formatter._describe_face(0, features)

        assert "Face 0" in desc
        assert "PLANE" in desc
        assert "area=10.500" in desc

    def test_describe_face_with_curvature(self):
        """Test face description with non-zero curvature."""
        from cadling.generation.codegen.graph_prompt_formatter import (
            GraphToPromptFormatter,
        )

        formatter = GraphToPromptFormatter()

        features = np.zeros(24)
        features[3] = 1.0  # SPHERE
        features[10] = 12.57  # area
        features[11] = 1.0  # gaussian curvature
        features[12] = 1.0  # mean curvature

        desc = formatter._describe_face(0, features)

        assert "SPHERE" in desc
        assert "K=" in desc
        assert "H=" in desc

    def test_edge_index_to_adjacency(self):
        """Test edge index to adjacency conversion."""
        from cadling.generation.codegen.graph_prompt_formatter import (
            GraphToPromptFormatter,
        )

        formatter = GraphToPromptFormatter()

        edge_index = np.array([
            [0, 0, 1, 1, 2, 2],
            [1, 2, 0, 2, 0, 1],
        ], dtype=np.int64)

        adjacency = formatter._edge_index_to_adjacency(edge_index)

        assert adjacency[0] == [1, 2]
        assert adjacency[1] == [0, 2]
        assert adjacency[2] == [0, 1]

    def test_format_entity_summary(self):
        """Test entity summary formatting."""
        from cadling.generation.codegen.graph_prompt_formatter import (
            GraphToPromptFormatter,
        )

        formatter = GraphToPromptFormatter()

        node_features = np.zeros((5, 24), dtype=np.float32)
        # 3 planes, 2 cylinders
        node_features[0, 0] = 1.0
        node_features[1, 0] = 1.0
        node_features[2, 0] = 1.0
        node_features[3, 1] = 1.0
        node_features[4, 1] = 1.0

        summary = formatter._format_entity_summary(node_features)

        assert "Entity Summary" in summary
        assert "PLANE: 3" in summary
        assert "CYLINDER: 2" in summary

    def test_format_adjacency(self):
        """Test adjacency formatting."""
        from cadling.generation.codegen.graph_prompt_formatter import (
            GraphToPromptFormatter,
        )

        formatter = GraphToPromptFormatter()

        node_features = np.zeros((3, 24), dtype=np.float32)
        node_features[0, 0] = 1.0  # PLANE
        node_features[1, 0] = 1.0  # PLANE
        node_features[2, 1] = 1.0  # CYLINDER

        adjacency = {0: [1, 2], 1: [0], 2: [0]}

        result = formatter._format_adjacency(adjacency, node_features)

        assert "Topology Relationships" in result

    def test_format_instructions(self):
        """Test CadQuery instruction formatting."""
        from cadling.generation.codegen.graph_prompt_formatter import (
            GraphToPromptFormatter,
        )

        formatter = GraphToPromptFormatter()

        instructions = formatter._format_instructions()

        assert "CadQuery" in instructions
        assert "result" in instructions
        assert "boolean" in instructions.lower() or "operations" in instructions.lower()


class TestGraphToPromptFormatterSTEP:
    """Test STEP entity formatting."""

    def test_format_from_step_features(self):
        """Test formatting from STEP entity features."""
        from cadling.generation.codegen.graph_prompt_formatter import (
            GraphToPromptFormatter,
        )

        formatter = GraphToPromptFormatter()

        entity_features = [
            {"entity_type": "ADVANCED_FACE", "properties": {"surface_type": "PLANE"}},
            {"entity_type": "ADVANCED_FACE", "properties": {"surface_type": "CYLINDER"}},
            {"entity_type": "EDGE_CURVE", "properties": {"curve_type": "LINE"}},
        ]

        prompt = formatter.format_from_step_features(entity_features)

        assert "You are generating CadQuery code" in prompt
        assert "ADVANCED_FACE" in prompt
        assert "STEP entities: 3" in prompt


class TestGraphToPromptFormatterEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_node_features(self):
        """Test handling of empty node features."""
        from cadling.generation.codegen.graph_prompt_formatter import (
            GraphToPromptFormatter,
        )

        formatter = GraphToPromptFormatter()

        node_features = np.zeros((0, 24), dtype=np.float32)

        prompt = formatter.format_for_cadquery(node_features)

        assert "Number of faces: 0" in prompt

    def test_single_face(self):
        """Test handling of single face."""
        from cadling.generation.codegen.graph_prompt_formatter import (
            GraphToPromptFormatter,
        )

        formatter = GraphToPromptFormatter()

        node_features = np.zeros((1, 24), dtype=np.float32)
        node_features[0, 0] = 1.0  # PLANE

        prompt = formatter.format_for_cadquery(node_features)

        assert "Number of faces: 1" in prompt
        assert "Face 0" in prompt

    def test_many_faces_truncated(self):
        """Test that many faces are truncated."""
        from cadling.generation.codegen.graph_prompt_formatter import (
            GraphToPromptFormatter,
        )

        formatter = GraphToPromptFormatter(max_faces_detailed=5)

        # Create 10 faces
        node_features = np.zeros((10, 24), dtype=np.float32)
        for i in range(10):
            node_features[i, 0] = 1.0  # PLANE

        prompt = formatter.format_for_cadquery(node_features)

        # Should mention truncation
        assert "more faces" in prompt.lower()
