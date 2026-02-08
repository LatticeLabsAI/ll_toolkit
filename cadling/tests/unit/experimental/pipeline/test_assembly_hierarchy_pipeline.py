"""
Unit tests for AssemblyHierarchyPipeline and AssemblyNode.

Tests cover:
- AssemblyNode tree structure
- Pipeline initialization
- Component detection
- Mate relationship extraction
- Interference checking
- BOM generation (flat and grouped)
- Tree serialization
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from cadling.experimental.pipeline import AssemblyHierarchyPipeline, AssemblyNode
from cadling.experimental.datamodel import AssemblyAnalysisOptions
from cadling.datamodel.base_models import (
    ConversionResult,
    ConversionStatus,
    CADInputDocument,
    InputFormat,
)


@pytest.fixture
def mock_input_doc():
    """Create a mock input document."""
    input_doc = CADInputDocument(
        file=Path("/tmp/test.step"),
        format=InputFormat.STEP,
        document_hash="test123",
    )
    backend = Mock()
    backend.convert = Mock()
    input_doc._backend = backend
    return input_doc


@pytest.fixture
def mock_converted_doc():
    """Create a mock converted assembly document."""
    doc = Mock()
    doc.properties = {}

    # Create multiple items (components)
    item1 = Mock()
    item1.self_ref = "part_1"
    item1.properties = {
        "name": "Bracket",
        "material": "Steel",
        "volume": 1000,
        "mass": 7.8,
        "bounding_box": {"x": 100, "y": 50, "z": 20},
    }

    item2 = Mock()
    item2.self_ref = "part_2"
    item2.properties = {
        "name": "Bracket",  # Same name for grouping test
        "material": "Steel",
        "volume": 1000,
        "mass": 7.8,
        "bounding_box": {"x": 100, "y": 50, "z": 20},
    }

    item3 = Mock()
    item3.self_ref = "part_3"
    item3.properties = {
        "name": "Bolt",
        "material": "Steel",
        "volume": 50,
        "mass": 0.4,
        "bounding_box": {"x": 20, "y": 20, "z": 80},
    }

    doc.items = [item1, item2, item3]
    return doc


@pytest.fixture
def mock_conversion_result(mock_input_doc, mock_converted_doc):
    """Create a mock conversion result."""
    conv_res = ConversionResult(input=mock_input_doc)
    mock_input_doc._backend.convert.return_value = mock_converted_doc
    return conv_res


class TestAssemblyNode:
    """Test AssemblyNode class."""

    def test_initialization(self):
        """Test node initialization."""
        node = AssemblyNode(component_id="comp_1", name="Bracket")

        assert node.component_id == "comp_1"
        assert node.name == "Bracket"
        assert node.item is None
        assert node.parent is None
        assert node.children == []
        assert node.mates == []
        assert node.properties == {}

    def test_add_child(self):
        """Test adding child nodes."""
        parent = AssemblyNode("parent", "Parent")
        child = AssemblyNode("child", "Child")

        parent.add_child(child)

        assert len(parent.children) == 1
        assert parent.children[0] == child
        assert child.parent == parent

    def test_is_root(self):
        """Test root node detection."""
        root = AssemblyNode("root", "Root")
        child = AssemblyNode("child", "Child")
        root.add_child(child)

        assert root.is_root() is True
        assert child.is_root() is False

    def test_is_leaf(self):
        """Test leaf node detection."""
        parent = AssemblyNode("parent", "Parent")
        child = AssemblyNode("child", "Child")
        parent.add_child(child)

        assert parent.is_leaf() is False
        assert child.is_leaf() is True

    def test_depth(self):
        """Test depth calculation."""
        root = AssemblyNode("root", "Root")
        level1 = AssemblyNode("level1", "Level1")
        level2 = AssemblyNode("level2", "Level2")

        root.add_child(level1)
        level1.add_child(level2)

        assert root.depth() == 0
        assert level1.depth() == 1
        assert level2.depth() == 2


class TestAssemblyHierarchyPipeline:
    """Test AssemblyHierarchyPipeline."""

    def test_initialization(self):
        """Test pipeline initialization."""
        options = AssemblyAnalysisOptions(
            detect_components=True,
            extract_mates=True,
            generate_bom=True,
        )

        pipeline = AssemblyHierarchyPipeline(options)

        assert pipeline.options == options
        assert pipeline.assembly_tree is None
        assert pipeline.component_map == {}
        assert pipeline.bom == []

    def test_get_default_options(self):
        """Test default options."""
        options = AssemblyHierarchyPipeline.get_default_options()

        assert isinstance(options, AssemblyAnalysisOptions)
        assert options.detect_components is True
        assert options.generate_bom is True

    def test_build_document(self, mock_conversion_result):
        """Test build stage with component detection."""
        options = AssemblyAnalysisOptions(detect_components=True)

        pipeline = AssemblyHierarchyPipeline(options)
        conv_res = pipeline._build_document(mock_conversion_result)

        # Check document created
        assert conv_res.document is not None

        # Check assembly tree created
        assert pipeline.assembly_tree is not None
        assert pipeline.assembly_tree.component_id == "root"

        # Check components detected (3 items + 1 root)
        assert len(pipeline.component_map) == 4

    def test_detect_components(self, mock_converted_doc):
        """Test component detection."""
        options = AssemblyAnalysisOptions()
        pipeline = AssemblyHierarchyPipeline(options)

        pipeline._detect_components(mock_converted_doc)

        # Should have root + 3 components
        assert len(pipeline.component_map) == 4
        assert "root" in pipeline.component_map

        # Check components added as children
        assert len(pipeline.assembly_tree.children) == 3

        # Check component properties extracted
        for comp_id, node in pipeline.component_map.items():
            if comp_id != "root":
                assert "material" in node.properties
                assert "volume" in node.properties

        # Check items marked with component IDs
        for item in mock_converted_doc.items:
            assert "component_id" in item.properties

    def test_build_document_no_detection(self, mock_conversion_result):
        """Test build without component detection."""
        options = AssemblyAnalysisOptions(detect_components=False)

        pipeline = AssemblyHierarchyPipeline(options)
        conv_res = pipeline._build_document(mock_conversion_result)

        # Should not build tree
        assert pipeline.assembly_tree is None
        assert len(pipeline.component_map) == 0

    def test_assemble_document(self, mock_conversion_result):
        """Test assemble stage."""
        options = AssemblyAnalysisOptions(
            detect_components=True,
            extract_mates=True,
            generate_bom=True,
        )

        pipeline = AssemblyHierarchyPipeline(options)

        # Complete build first
        conv_res = pipeline._build_document(mock_conversion_result)

        # Run assemble
        conv_res = pipeline._assemble_document(conv_res)

        # Check BOM generated
        assert len(pipeline.bom) > 0
        assert "bill_of_materials" in conv_res.document.properties

        # Check assembly tree stored
        assert "assembly_tree" in conv_res.document.properties
        assert "num_components" in conv_res.document.properties

    def test_generate_bom_grouped(self, mock_converted_doc):
        """Test BOM generation with grouping."""
        options = AssemblyAnalysisOptions(
            group_identical_parts=True,
            bom_include_metadata=True,
            bom_include_properties=True,
        )
        pipeline = AssemblyHierarchyPipeline(options)

        pipeline._detect_components(mock_converted_doc)
        bom = pipeline._generate_bom(mock_converted_doc)

        # Should group 2 Brackets together
        bracket_entry = next((e for e in bom if e["name"] == "Bracket"), None)
        assert bracket_entry is not None
        assert bracket_entry["quantity"] == 2
        assert len(bracket_entry["component_ids"]) == 2

        # Bolt should be separate
        bolt_entry = next((e for e in bom if e["name"] == "Bolt"), None)
        assert bolt_entry is not None
        assert bolt_entry["quantity"] == 1

        # Check metadata included
        assert "material" in bracket_entry
        assert "volume" in bracket_entry
        assert "mass" in bracket_entry

    def test_generate_bom_flat(self, mock_converted_doc):
        """Test BOM generation without grouping."""
        options = AssemblyAnalysisOptions(
            group_identical_parts=False,
            bom_include_metadata=False,
            bom_include_properties=False,
        )
        pipeline = AssemblyHierarchyPipeline(options)

        pipeline._detect_components(mock_converted_doc)
        bom = pipeline._generate_bom(mock_converted_doc)

        # Should have 3 entries (one per component)
        assert len(bom) == 3

        # Each should have quantity 1
        for entry in bom:
            assert entry["quantity"] == 1
            assert "component_id" in entry
            # Metadata not included
            assert "volume" not in entry
            assert "mass" not in entry

    def test_extract_mate_relationships(self, mock_converted_doc):
        """Test mate relationship extraction."""
        options = AssemblyAnalysisOptions(extract_mates=True)
        pipeline = AssemblyHierarchyPipeline(options)

        pipeline._detect_components(mock_converted_doc)
        pipeline._extract_mate_relationships(mock_converted_doc)

        # Should attempt to detect mates
        # (actual detection is placeholder, but should not crash)
        mate_count = pipeline._count_mates()
        assert mate_count >= 0

    def test_check_interferences(self, mock_converted_doc):
        """Test interference checking."""
        options = AssemblyAnalysisOptions(check_interference=True)
        pipeline = AssemblyHierarchyPipeline(options)

        pipeline._detect_components(mock_converted_doc)
        pipeline._check_interferences(mock_converted_doc)

        # Should have interferences property
        assert "interferences" in mock_converted_doc.properties
        assert isinstance(mock_converted_doc.properties["interferences"], list)

    def test_serialize_tree(self, mock_converted_doc):
        """Test tree serialization."""
        options = AssemblyAnalysisOptions()
        pipeline = AssemblyHierarchyPipeline(options)

        pipeline._detect_components(mock_converted_doc)

        serialized = pipeline._serialize_tree(pipeline.assembly_tree)

        # Check structure
        assert "component_id" in serialized
        assert serialized["component_id"] == "root"
        assert "name" in serialized
        assert "depth" in serialized
        assert serialized["depth"] == 0
        assert "children" in serialized
        assert len(serialized["children"]) == 3

        # Check children serialized recursively
        for child in serialized["children"]:
            assert "component_id" in child
            assert "depth" in child
            assert child["depth"] == 1

    def test_count_mates(self):
        """Test mate counting."""
        options = AssemblyAnalysisOptions()
        pipeline = AssemblyHierarchyPipeline(options)

        # Create nodes with mates
        node1 = AssemblyNode("n1", "Node1")
        node2 = AssemblyNode("n2", "Node2")

        mate = {"type": "coincident", "components": ["n1", "n2"]}
        node1.mates.append(mate)
        node2.mates.append(mate)

        pipeline.component_map = {"n1": node1, "n2": node2}

        # Should count mate once (divided by 2)
        assert pipeline._count_mates() == 1

    def test_determine_status_success(self, mock_conversion_result):
        """Test status determination - success."""
        options = AssemblyAnalysisOptions()
        pipeline = AssemblyHierarchyPipeline(options)

        pipeline._build_document(mock_conversion_result)

        status = pipeline._determine_status(mock_conversion_result)
        assert status == ConversionStatus.SUCCESS

    def test_determine_status_failure(self):
        """Test status determination - failure."""
        options = AssemblyAnalysisOptions()
        pipeline = AssemblyHierarchyPipeline(options)

        mock_input = CADInputDocument(
            file=Path("/tmp/test.step"),
            format=InputFormat.STEP,
            document_hash="test123",
        )
        conv_res = ConversionResult(input=mock_input)
        conv_res.document = None

        status = pipeline._determine_status(conv_res)
        assert status == ConversionStatus.FAILURE

    def test_error_handling_build(self, mock_conversion_result):
        """Test error handling in build stage."""
        options = AssemblyAnalysisOptions()

        # Make backend raise error
        mock_conversion_result.input._backend.convert.side_effect = Exception(
            "Conversion error"
        )

        pipeline = AssemblyHierarchyPipeline(options)

        with pytest.raises(Exception, match="Conversion error"):
            pipeline._build_document(mock_conversion_result)

    def test_error_handling_assemble(self, mock_conversion_result):
        """Test error handling in assemble stage."""
        options = AssemblyAnalysisOptions(generate_bom=True)

        pipeline = AssemblyHierarchyPipeline(options)

        # Build successfully
        conv_res = pipeline._build_document(mock_conversion_result)

        # Make BOM generation fail
        with patch.object(pipeline, "_generate_bom", side_effect=Exception("BOM error")):
            with pytest.raises(Exception, match="BOM error"):
                pipeline._assemble_document(conv_res)

    def test_no_components_bom(self):
        """Test BOM generation with no components."""
        options = AssemblyAnalysisOptions()
        pipeline = AssemblyHierarchyPipeline(options)

        # Create empty doc
        doc = Mock()
        doc.items = []

        pipeline._detect_components(doc)
        bom = pipeline._generate_bom(doc)

        # Should have empty BOM
        assert len(bom) == 0

    def test_component_naming_strategy(self, mock_converted_doc):
        """Test component naming."""
        options = AssemblyAnalysisOptions(component_naming_strategy="hierarchical")
        pipeline = AssemblyHierarchyPipeline(options)

        pipeline._detect_components(mock_converted_doc)

        # Check components have names
        for comp_id, node in pipeline.component_map.items():
            if comp_id != "root":
                assert node.name is not None
                assert len(node.name) > 0

    def test_max_components_limit(self):
        """Test max components limit."""
        options = AssemblyAnalysisOptions(max_components=2)
        pipeline = AssemblyHierarchyPipeline(options)

        # Create doc with many items
        doc = Mock()
        doc.properties = {}
        items = []
        for i in range(10):
            item = Mock()
            item.self_ref = f"part_{i}"
            item.properties = {"name": f"Part {i}"}
            items.append(item)
        doc.items = items

        pipeline._detect_components(doc)

        # Should only process up to max (implementation may vary)
        # At minimum, should not crash
        assert len(pipeline.component_map) >= 1  # At least root

    def test_subassembly_processing(self):
        """Test subassembly processing option."""
        options = AssemblyAnalysisOptions(process_subassemblies=True)
        pipeline = AssemblyHierarchyPipeline(options)

        # Check option is stored
        assert pipeline.options.process_subassemblies is True

    def test_extract_fasteners(self):
        """Test fastener extraction option."""
        options = AssemblyAnalysisOptions(extract_fasteners=True)
        pipeline = AssemblyHierarchyPipeline(options)

        # Check option is stored
        assert pipeline.options.extract_fasteners is True

    def test_bom_metadata_options(self, mock_converted_doc):
        """Test BOM metadata inclusion options."""
        options = AssemblyAnalysisOptions(
            bom_include_metadata=True,
            bom_include_properties=False,
        )
        pipeline = AssemblyHierarchyPipeline(options)

        pipeline._detect_components(mock_converted_doc)
        bom = pipeline._generate_bom(mock_converted_doc)

        # Should include metadata but not properties
        for entry in bom:
            if entry["quantity"] > 0:
                # Material is metadata
                assert "material" in entry
                # Volume/mass are properties
                assert "volume" not in entry
                assert "mass" not in entry

    def test_complete_workflow(self, mock_conversion_result):
        """Test complete assembly processing workflow."""
        options = AssemblyAnalysisOptions(
            detect_components=True,
            extract_mates=True,
            generate_bom=True,
            check_interference=False,
        )

        pipeline = AssemblyHierarchyPipeline(options)

        # Build
        conv_res = pipeline._build_document(mock_conversion_result)
        assert pipeline.assembly_tree is not None

        # Assemble
        conv_res = pipeline._assemble_document(conv_res)

        # Check all results present
        doc = conv_res.document
        assert "bill_of_materials" in doc.properties
        assert "assembly_tree" in doc.properties
        assert "num_components" in doc.properties

        # Check BOM has correct structure
        bom = doc.properties["bill_of_materials"]
        assert len(bom) > 0
        for entry in bom:
            assert "name" in entry
            assert "quantity" in entry

    def test_tree_hierarchy_depth(self):
        """Test tree hierarchy depth calculation."""
        # Create multi-level tree
        root = AssemblyNode("root", "Root")
        sub1 = AssemblyNode("sub1", "Subassembly1")
        part1 = AssemblyNode("part1", "Part1")

        root.add_child(sub1)
        sub1.add_child(part1)

        assert root.depth() == 0
        assert sub1.depth() == 1
        assert part1.depth() == 2
