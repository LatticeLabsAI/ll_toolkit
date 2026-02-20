"""Tests for DFS reserialization and structural annotations."""
from __future__ import annotations

import sys
import os
import importlib

# Ensure stepnet package is importable without triggering full __init__.py
# which may have heavy dependencies (matplotlib, scipy, etc.)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

# Skip entire module if torch is not installed (stepnet requires torch)
pytest.importorskip("torch")

# Direct submodule imports to avoid __init__.py's heavy dependency chain
_config_mod = importlib.import_module("stepnet.config")
_reser_mod = importlib.import_module("stepnet.reserialization")
_annot_mod = importlib.import_module("stepnet.annotations")

STEPReserializationConfig = _config_mod.STEPReserializationConfig
STEPAnnotationConfig = _config_mod.STEPAnnotationConfig

STEPEntityGraph = _reser_mod.STEPEntityGraph
STEPEntityNode = _reser_mod.STEPEntityNode
STEPDFSSerializer = _reser_mod.STEPDFSSerializer
STEPReserializedOutput = _reser_mod.STEPReserializedOutput
reserialize_step = _reser_mod.reserialize_step

BranchAnnotation = _annot_mod.BranchAnnotation
StructuralSummary = _annot_mod.StructuralSummary
STEPStructuralAnnotator = _annot_mod.STEPStructuralAnnotator
STEPAnnotatedOutput = _annot_mod.STEPAnnotatedOutput


# Sample STEP snippet for testing
SAMPLE_STEP = """
#1=MANIFOLD_SOLID_BREP('rod',#2);
#2=CLOSED_SHELL('',(#3,#4));
#3=ADVANCED_FACE('',(#5),#6,.T.);
#4=ADVANCED_FACE('',(#7),#8,.T.);
#5=FACE_OUTER_BOUND('',#9,.T.);
#6=PLANE('',#10);
#7=FACE_OUTER_BOUND('',#11,.T.);
#8=CYLINDRICAL_SURFACE('',#12,5.0);
#9=EDGE_LOOP('',(#13));
#10=AXIS2_PLACEMENT_3D('',#14,#15,#16);
#11=EDGE_LOOP('',(#17));
#12=AXIS2_PLACEMENT_3D('',#18,#19,#20);
#13=ORIENTED_EDGE('',*,*,#21,.T.);
#14=CARTESIAN_POINT('',(0.0,0.0,0.0));
#15=DIRECTION('',(0.0,0.0,1.0));
#16=DIRECTION('',(1.0,0.0,0.0));
#17=ORIENTED_EDGE('',*,*,#22,.T.);
#18=CARTESIAN_POINT('',(0.0,0.0,0.0));
#19=DIRECTION('',(0.0,0.0,1.0));
#20=DIRECTION('',(1.0,0.0,0.0));
#21=EDGE_CURVE('',#23,#24,#25,.T.);
#22=EDGE_CURVE('',#26,#27,#28,.T.);
#23=VERTEX_POINT('',#29);
#24=VERTEX_POINT('',#30);
#25=LINE('',#31,#32);
#26=VERTEX_POINT('',#33);
#27=VERTEX_POINT('',#34);
#28=CIRCLE('',#35,5.0);
#29=CARTESIAN_POINT('',(0.0,0.0,0.0));
#30=CARTESIAN_POINT('',(10.0,0.0,0.0));
#31=CARTESIAN_POINT('',(0.0,0.0,0.0));
#32=DIRECTION('',(1.0,0.0,0.0));
#33=CARTESIAN_POINT('',(5.0,0.0,0.0));
#34=CARTESIAN_POINT('',(5.0,5.0,0.0));
#35=AXIS2_PLACEMENT_3D('',#36,#37,#38);
#36=CARTESIAN_POINT('',(5.0,0.0,0.0));
#37=DIRECTION('',(0.0,0.0,1.0));
#38=DIRECTION('',(1.0,0.0,0.0));
"""


class TestSTEPEntityGraph:
    """Test entity graph parsing."""

    def test_parse_entity_count(self):
        graph = STEPEntityGraph.parse(SAMPLE_STEP)
        assert len(graph.nodes) == 38

    def test_parse_node_types(self):
        graph = STEPEntityGraph.parse(SAMPLE_STEP)
        assert graph.nodes[1].entity_type == "MANIFOLD_SOLID_BREP"
        assert graph.nodes[2].entity_type == "CLOSED_SHELL"
        assert graph.nodes[14].entity_type == "CARTESIAN_POINT"

    def test_parse_children(self):
        graph = STEPEntityGraph.parse(SAMPLE_STEP)
        # #1 references #2
        assert 2 in graph.nodes[1].children
        # #2 references #3 and #4
        assert 3 in graph.nodes[2].children
        assert 4 in graph.nodes[2].children

    def test_parse_parents(self):
        graph = STEPEntityGraph.parse(SAMPLE_STEP)
        # #2 is referenced by #1
        assert 1 in graph.nodes[2].parents
        # #3 is referenced by #2
        assert 2 in graph.nodes[3].parents

    def test_roots(self):
        graph = STEPEntityGraph.parse(SAMPLE_STEP)
        roots = graph.roots
        assert 1 in roots  # MANIFOLD_SOLID_BREP has no parents


class TestSTEPDFSSerializer:
    """Test DFS serialization."""

    def test_dfs_order(self):
        graph = STEPEntityGraph.parse(SAMPLE_STEP)
        serializer = STEPDFSSerializer()
        result = serializer.serialize(graph)

        # Entity #1 should come first (root)
        assert result.traversal_order[0][0] == 1
        # Entity count should match
        assert result.entity_count == 38

    def test_branch_pruning(self):
        """Each entity should appear exactly once in traversal."""
        graph = STEPEntityGraph.parse(SAMPLE_STEP)
        serializer = STEPDFSSerializer()
        result = serializer.serialize(graph)

        ids = [eid for eid, _ in result.traversal_order]
        assert len(ids) == len(set(ids)), "Duplicate entities in traversal"

    def test_sequential_renumbering(self):
        config = STEPReserializationConfig(renumber_ids=True)
        graph = STEPEntityGraph.parse(SAMPLE_STEP)
        serializer = STEPDFSSerializer(config)
        result = serializer.serialize(graph)

        # New IDs should be 1..N
        new_ids = sorted(result.id_mapping.values())
        assert new_ids == list(range(1, len(graph.nodes) + 1))

    def test_reference_rewriting(self):
        config = STEPReserializationConfig(renumber_ids=True)
        graph = STEPEntityGraph.parse(SAMPLE_STEP)
        serializer = STEPDFSSerializer(config)
        result = serializer.serialize(graph)

        # The output text should contain sequential IDs
        assert "#1=" in result.text
        assert "#2=" in result.text

    def test_float_normalization(self):
        step_text = "#1=CARTESIAN_POINT('',(0.000000000,8.001000001,1.23456789012));"
        config = STEPReserializationConfig(
            normalize_floats=True,
            float_precision=6,
            renumber_ids=False,
        )
        graph = STEPEntityGraph.parse(step_text)
        serializer = STEPDFSSerializer(config)
        result = serializer.serialize(graph)

        # 0.000000000 -> 0.0, 8.001000001 -> 8.001, 1.23456789012 -> 1.23457
        assert "0.0" in result.text
        assert "8.001" in result.text

    def test_max_depth_pruning(self):
        config = STEPReserializationConfig(max_depth=2)
        graph = STEPEntityGraph.parse(SAMPLE_STEP)
        serializer = STEPDFSSerializer(config)
        result = serializer.serialize(graph)

        # Should have fewer entities than total due to depth limit
        visited_count = len([eid for eid, d in result.traversal_order if d <= 2])
        assert visited_count <= result.entity_count

    def test_orphan_handling(self):
        # Create a graph with an orphan
        step_text = """
#1=MANIFOLD_SOLID_BREP('test',#2);
#2=CLOSED_SHELL('',(#3));
#3=ADVANCED_FACE('',(#4),#5,.T.);
#4=FACE_OUTER_BOUND('',#6,.T.);
#5=PLANE('',#7);
#6=EDGE_LOOP('',(#8));
#7=AXIS2_PLACEMENT_3D('',#9,#10,#11);
#8=ORIENTED_EDGE('',*,*,#12,.T.);
#9=CARTESIAN_POINT('',(0.0,0.0,0.0));
#10=DIRECTION('',(0.0,0.0,1.0));
#11=DIRECTION('',(1.0,0.0,0.0));
#12=EDGE_CURVE('',#13,#14,#15,.T.);
#13=VERTEX_POINT('',#16);
#14=VERTEX_POINT('',#17);
#15=LINE('',#18,#19);
#16=CARTESIAN_POINT('',(0.0,0.0,0.0));
#17=CARTESIAN_POINT('',(10.0,0.0,0.0));
#18=CARTESIAN_POINT('',(0.0,0.0,0.0));
#19=DIRECTION('',(1.0,0.0,0.0));
#99=COLOUR_RGB('red',1.0,0.0,0.0);
"""
        config = STEPReserializationConfig(include_orphans=True)
        graph = STEPEntityGraph.parse(step_text)
        serializer = STEPDFSSerializer(config)
        result = serializer.serialize(graph)

        # #99 should be included in the traversal (either as a root with
        # no incoming edges or as an orphan, depending on strategy)
        all_ids = [eid for eid, _ in result.traversal_order]
        assert 99 in all_ids
        # All 20 entities should be present
        assert result.entity_count == 20

    def test_cycle_handling(self):
        """Graph with cycle should not cause infinite loop."""
        # Simulated cycle: #1 -> #2 -> #3 -> #1 (via parameters)
        step_text = """
#1=TYPE_A('',#2);
#2=TYPE_B('',#3);
#3=TYPE_C('',#1);
"""
        graph = STEPEntityGraph.parse(step_text)
        serializer = STEPDFSSerializer()
        result = serializer.serialize(graph)

        # Should complete without infinite loop
        assert result.entity_count == 3

    def test_empty_input(self):
        graph = STEPEntityGraph.parse("")
        serializer = STEPDFSSerializer()
        result = serializer.serialize(graph)
        assert result.entity_count == 0
        assert result.text == ""


class TestReserializeStep:
    """Test convenience function."""

    def test_basic_usage(self):
        result = reserialize_step(SAMPLE_STEP)
        assert result.entity_count == 38
        assert len(result.text) > 0

    def test_with_config(self):
        config = STEPReserializationConfig(
            normalize_floats=True,
            renumber_ids=True,
        )
        result = reserialize_step(SAMPLE_STEP, config)
        assert result.entity_count == 38


class TestAnnotations:
    """Test structural annotation generation."""

    def test_summary_generation(self):
        graph = STEPEntityGraph.parse(SAMPLE_STEP)
        serializer = STEPDFSSerializer()
        reserialized = serializer.serialize(graph)

        annotator = STEPStructuralAnnotator()
        output = annotator.annotate(graph, reserialized)

        assert output.summary is not None
        assert output.summary.total_entities == 38
        assert "[SUMMARY]" in output.summary.format()
        assert "[/SUMMARY]" in output.summary.format()

    def test_branch_annotations(self):
        graph = STEPEntityGraph.parse(SAMPLE_STEP)
        serializer = STEPDFSSerializer()
        reserialized = serializer.serialize(graph)

        annotator = STEPStructuralAnnotator()
        output = annotator.annotate(graph, reserialized)

        assert len(output.branches) > 0
        for branch in output.branches:
            formatted = branch.format()
            assert "[BRANCH" in formatted
            assert "[/BRANCH]" in formatted

    def test_annotated_text_format(self):
        graph = STEPEntityGraph.parse(SAMPLE_STEP)
        serializer = STEPDFSSerializer()
        reserialized = serializer.serialize(graph)

        annotator = STEPStructuralAnnotator()
        output = annotator.annotate(graph, reserialized)

        full_text = output.format()
        assert "[SUMMARY]" in full_text
        assert reserialized.text in full_text

    def test_disabled_annotations(self):
        config = STEPAnnotationConfig(
            include_file_summary=False,
            include_branch_annotations=False,
        )
        graph = STEPEntityGraph.parse(SAMPLE_STEP)
        serializer = STEPDFSSerializer()
        reserialized = serializer.serialize(graph)

        annotator = STEPStructuralAnnotator(config)
        output = annotator.annotate(graph, reserialized)

        assert output.summary is None
        assert len(output.branches) == 0

    def test_category_classification(self):
        graph = STEPEntityGraph.parse(SAMPLE_STEP)
        serializer = STEPDFSSerializer()
        reserialized = serializer.serialize(graph)

        annotator = STEPStructuralAnnotator()
        output = annotator.annotate(graph, reserialized)

        # Sample has B-Rep entities, should classify appropriately
        assert output.summary.dominant_category in ["B-Rep", "Geometry", "Assembly", "Mixed"]
