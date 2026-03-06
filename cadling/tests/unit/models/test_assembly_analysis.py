"""Tests for assembly analysis model.

Tests cover:
- AssemblyAnalysisModel initialization and configuration
- Assembly graph construction
- Mating surface detection
- BOM generation
- Subassembly identification
- Integration with real STEP assembly files
"""

import pytest
from pathlib import Path
from unittest.mock import Mock

from cadling.models.assembly_analysis import (
    AssemblyAnalysisModel,
    AssemblyGraph,
    Contact,
    BillOfMaterials,
    Subassembly
)
from cadling.datamodel.base_models import CADItemLabel


class TestAssemblyAnalysisModel:
    """Test AssemblyAnalysisModel initialization and basic functionality."""

    def test_model_initialization_default(self):
        """Test model initialization with default parameters."""
        model = AssemblyAnalysisModel()

        assert model.detect_contacts is True
        assert model.compute_bom is True
        assert model.identify_subassemblies is True
        assert model.contact_tolerance == 0.01

    def test_model_initialization_custom(self):
        """Test model initialization with custom parameters."""
        model = AssemblyAnalysisModel(
            detect_contacts=False,
            compute_bom=False,
            identify_subassemblies=False,
            contact_tolerance=0.05
        )

        assert model.detect_contacts is False
        assert model.compute_bom is False
        assert model.identify_subassemblies is False
        assert model.contact_tolerance == 0.05

    def test_model_has_required_attributes(self):
        """Test model has required attributes."""
        model = AssemblyAnalysisModel()

        assert hasattr(model, 'has_pythonocc')
        assert isinstance(model.has_pythonocc, bool)

    def test_model_call_with_empty_document(self):
        """Test model call with empty document."""
        model = AssemblyAnalysisModel()

        # Create empty document
        class SimpleDoc:
            def __init__(self):
                self.properties = {}
                self.items = []

        doc = SimpleDoc()

        # Should not crash
        model(doc, [])

        # Should have results if pythonocc available
        if model.has_pythonocc:
            assert "assembly_analysis" in doc.properties


class TestAssemblyGraph:
    """Test AssemblyGraph data structure."""

    def test_graph_initialization(self):
        """Test graph initialization."""
        graph = AssemblyGraph()

        assert isinstance(graph.nodes, dict)
        assert isinstance(graph.edges, list)
        assert graph.root_id is None
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_add_node(self):
        """Test adding nodes to graph."""
        graph = AssemblyGraph()

        graph.add_node("part1", node_type="part", name="Part 1")
        graph.add_node("asm1", node_type="assembly", name="Assembly 1")

        assert len(graph.nodes) == 2
        assert "part1" in graph.nodes
        assert "asm1" in graph.nodes
        assert graph.nodes["part1"]["type"] == "part"
        assert graph.nodes["asm1"]["type"] == "assembly"

    def test_add_edge(self):
        """Test adding edges to graph."""
        graph = AssemblyGraph()

        graph.add_node("asm1", node_type="assembly")
        graph.add_node("part1", node_type="part")
        graph.add_edge("asm1", "part1", edge_type="contains")

        assert len(graph.edges) == 1
        source, target, data = graph.edges[0]
        assert source == "asm1"
        assert target == "part1"
        assert data["type"] == "contains"

    def test_get_children(self):
        """Test getting children of a node."""
        graph = AssemblyGraph()

        graph.add_node("asm1", node_type="assembly")
        graph.add_node("part1", node_type="part")
        graph.add_node("part2", node_type="part")
        graph.add_edge("asm1", "part1", edge_type="contains")
        graph.add_edge("asm1", "part2", edge_type="contains")

        children = graph.get_children("asm1")
        assert len(children) == 2
        assert "part1" in children
        assert "part2" in children

    def test_get_depth_single_level(self):
        """Test depth calculation for single level."""
        graph = AssemblyGraph()

        graph.add_node("root", node_type="assembly")
        graph.add_node("part1", node_type="part")
        graph.add_edge("root", "part1", edge_type="contains")

        depth = graph.get_depth("root")
        assert depth == 1

    def test_get_depth_multiple_levels(self):
        """Test depth calculation for multiple levels."""
        graph = AssemblyGraph()

        graph.add_node("root", node_type="assembly")
        graph.add_node("subasm1", node_type="assembly")
        graph.add_node("part1", node_type="part")
        graph.add_edge("root", "subasm1", edge_type="contains")
        graph.add_edge("subasm1", "part1", edge_type="contains")

        depth = graph.get_depth("root")
        assert depth == 2


class TestBillOfMaterials:
    """Test BillOfMaterials functionality."""

    def test_bom_initialization(self):
        """Test BOM initialization."""
        bom = BillOfMaterials(assembly_id="asm1")

        assert bom.assembly_id == "asm1"
        assert len(bom.items) == 0
        assert bom.total_parts == 0
        assert bom.unique_parts == 0
        assert bom.hierarchy_depth == 0

    def test_add_item(self):
        """Test adding items to BOM."""
        bom = BillOfMaterials(assembly_id="asm1")

        bom.add_item("part1", "Part 1", quantity=2, level=0)
        bom.add_item("part2", "Part 2", quantity=1, level=1)

        assert len(bom.items) == 2
        assert bom.total_parts == 3  # 2 + 1
        assert bom.hierarchy_depth == 2  # level 0 and 1

    def test_add_item_with_metadata(self):
        """Test adding items with metadata."""
        bom = BillOfMaterials(assembly_id="asm1")

        bom.add_item(
            "part1",
            "Part 1",
            quantity=1,
            level=0,
            material="Steel",
            mass=1.5
        )

        assert len(bom.items) == 1
        assert bom.items[0]["material"] == "Steel"
        assert bom.items[0]["mass"] == 1.5


class TestContact:
    """Test Contact data structure."""

    def test_contact_initialization(self):
        """Test Contact initialization."""
        contact = Contact(
            part1_id="part1",
            part2_id="part2",
            contact_type="planar"
        )

        assert contact.part1_id == "part1"
        assert contact.part2_id == "part2"
        assert contact.contact_type == "planar"
        assert contact.contact_area == 0.0
        assert contact.distance == 0.0
        assert contact.confidence == 0.5

    def test_contact_with_geometry(self):
        """Test Contact with geometric data."""
        contact = Contact(
            part1_id="part1",
            part2_id="part2",
            contact_type="cylindrical",
            contact_area=10.5,
            contact_center=[1.0, 2.0, 3.0],
            contact_normal=[0.0, 0.0, 1.0],
            distance=0.0,
            confidence=0.9
        )

        assert contact.contact_area == 10.5
        assert contact.contact_center == [1.0, 2.0, 3.0]
        assert contact.contact_normal == [0.0, 0.0, 1.0]
        assert contact.confidence == 0.9


class TestSubassembly:
    """Test Subassembly data structure."""

    def test_subassembly_initialization(self):
        """Test Subassembly initialization."""
        subasm = Subassembly(
            subassembly_id="subasm1",
            name="Wheel Assembly"
        )

        assert subasm.subassembly_id == "subasm1"
        assert subasm.name == "Wheel Assembly"
        assert len(subasm.part_ids) == 0
        assert subasm.parent_id is None
        assert subasm.transform is None

    def test_subassembly_with_parts(self):
        """Test Subassembly with part IDs."""
        subasm = Subassembly(
            subassembly_id="subasm1",
            name="Wheel Assembly",
            part_ids=["part1", "part2", "part3"],
            parent_id="root",
            metadata={"category": "drivetrain"}
        )

        assert len(subasm.part_ids) == 3
        assert "part1" in subasm.part_ids
        assert subasm.parent_id == "root"
        assert subasm.metadata["category"] == "drivetrain"


class TestAssemblyGraphBuilding:
    """Test assembly graph construction."""

    def test_build_graph_from_empty_document(self):
        """Test building graph from document with no items."""
        pytest.importorskip("numpy")

        model = AssemblyAnalysisModel()

        class SimpleDoc:
            def __init__(self):
                self.properties = {}
                self.items = []

        doc = SimpleDoc()
        graph = model.build_assembly_graph(doc)

        # Should create root assembly even with no items
        assert graph.root_id == "root_assembly"
        assert len(graph.nodes) == 1

    def test_build_graph_from_parts_only(self):
        """Test building graph from parts with no assembly items."""
        pytest.importorskip("numpy")

        model = AssemblyAnalysisModel()

        # Create mock parts
        part1 = Mock()
        part1.item_type = "part"
        part1.item_id = "part1"
        part1.name = "Part 1"

        part2 = Mock()
        part2.item_type = "part"
        part2.item_id = "part2"
        part2.name = "Part 2"

        class SimpleDoc:
            def __init__(self):
                self.properties = {}
                self.items = [part1, part2]

        doc = SimpleDoc()
        graph = model.build_assembly_graph(doc)

        # Should create virtual root and add parts
        assert graph.root_id == "root_assembly"
        assert len(graph.nodes) == 3  # root + 2 parts
        assert "part1" in graph.nodes
        assert "part2" in graph.nodes

    def test_build_graph_from_assembly(self):
        """Test building graph from assembly with components."""
        pytest.importorskip("numpy")

        from cadling.datamodel.stl import AssemblyItem

        model = AssemblyAnalysisModel()

        # Create parts
        part1 = Mock()
        part1.item_type = "part"
        part1.item_id = "part1"
        part1.name = "Part 1"

        part2 = Mock()
        part2.item_type = "part"
        part2.item_id = "part2"
        part2.name = "Part 2"

        # Create assembly
        from cadling.datamodel.base_models import CADItemLabel
        asm = AssemblyItem(
            label=CADItemLabel(text="Assembly 1"),
            item_id="asm1"
        )
        asm.add_component("part1", "Part 1")
        asm.add_component("part2", "Part 2")

        class SimpleDoc:
            def __init__(self):
                self.properties = {}
                self.items = [asm, part1, part2]

        doc = SimpleDoc()
        graph = model.build_assembly_graph(doc)

        assert graph.root_id == "asm1"
        assert len(graph.nodes) == 3  # assembly + 2 parts
        assert "part1" in graph.nodes
        assert "part2" in graph.nodes


class TestBOMGeneration:
    """Test BOM generation."""

    def test_compute_bom_from_flat_assembly(self):
        """Test BOM generation from flat assembly."""
        pytest.importorskip("numpy")

        model = AssemblyAnalysisModel()

        # Create simple graph
        graph = AssemblyGraph()
        graph.add_node("root", node_type="assembly", name="Root")
        graph.add_node("part1", node_type="part", name="Part 1")
        graph.add_node("part2", node_type="part", name="Part 2")
        graph.add_edge("root", "part1", edge_type="contains")
        graph.add_edge("root", "part2", edge_type="contains")
        graph.root_id = "root"

        bom = model.compute_assembly_bom(None, graph)

        assert bom.total_parts == 2
        assert bom.unique_parts == 2
        assert len(bom.items) == 2

    def test_compute_bom_with_duplicates(self):
        """Test BOM with duplicate parts."""
        pytest.importorskip("numpy")

        model = AssemblyAnalysisModel()

        # Create graph with repeated parts
        graph = AssemblyGraph()
        graph.add_node("root", node_type="assembly", name="Root")
        graph.add_node("subasm1", node_type="assembly", name="Sub 1")
        graph.add_node("subasm2", node_type="assembly", name="Sub 2")
        graph.add_node("bolt1", node_type="part", name="Bolt M8")
        graph.add_node("bolt2", node_type="part", name="Bolt M8")

        graph.add_edge("root", "subasm1", edge_type="contains")
        graph.add_edge("root", "subasm2", edge_type="contains")
        graph.add_edge("subasm1", "bolt1", edge_type="contains")
        graph.add_edge("subasm2", "bolt2", edge_type="contains")
        graph.root_id = "root"

        bom = model.compute_assembly_bom(None, graph)

        # Should count bolts separately (different IDs even though same name)
        assert bom.unique_parts == 2
        assert bom.total_parts == 2


class TestSubassemblyDetection:
    """Test subassembly identification."""

    def test_detect_explicit_subassemblies(self):
        """Test detection of explicit assembly nodes."""
        pytest.importorskip("numpy")

        model = AssemblyAnalysisModel()

        graph = AssemblyGraph()
        graph.add_node("root", node_type="assembly", name="Root")
        graph.add_node("subasm1", node_type="assembly", name="Wheel Assembly")
        graph.add_node("part1", node_type="part", name="Tire")
        graph.add_node("part2", node_type="part", name="Rim")

        graph.add_edge("root", "subasm1", edge_type="contains")
        graph.add_edge("subasm1", "part1", edge_type="contains")
        graph.add_edge("subasm1", "part2", edge_type="contains")
        graph.root_id = "root"

        subassemblies = model.detect_subassemblies(graph)

        assert len(subassemblies) >= 1
        # Find the wheel assembly
        wheel_asm = [s for s in subassemblies if s.name == "Wheel Assembly"]
        assert len(wheel_asm) == 1
        assert "part1" in wheel_asm[0].part_ids
        assert "part2" in wheel_asm[0].part_ids

    def test_detect_naming_pattern_subassemblies(self):
        """Test detection via naming patterns with numeric instance suffixes."""
        pytest.importorskip("numpy")

        model = AssemblyAnalysisModel()

        graph = AssemblyGraph()
        graph.add_node("root", node_type="assembly", name="Root")
        graph.add_node("wheel_1", node_type="part", name="Wheel_Assembly_1")
        graph.add_node("wheel_2", node_type="part", name="Wheel_Assembly_2")
        graph.add_node("wheel_3", node_type="part", name="Wheel_Assembly_3")

        graph.root_id = "root"

        subassemblies = model.detect_subassemblies(graph)

        # Should find a group for the Wheel_Assembly prefix
        assert len(subassemblies) >= 1
        wheel_groups = [s for s in subassemblies if "Wheel_Assembly" in s.name]
        assert len(wheel_groups) == 1
        assert len(wheel_groups[0].part_ids) == 3

    def test_hardware_parts_not_grouped_as_subassembly(self):
        """Regression: Bolt_M6_1, Bolt_M6_2 must NOT form a subassembly."""
        pytest.importorskip("numpy")

        model = AssemblyAnalysisModel()

        graph = AssemblyGraph()
        graph.add_node("root", node_type="assembly", name="Root")
        graph.add_node("bolt1", node_type="part", name="Bolt_M6_1")
        graph.add_node("bolt2", node_type="part", name="Bolt_M6_2")
        graph.add_node("bolt3", node_type="part", name="Bolt_M6_3")
        graph.add_node("nut1", node_type="part", name="Nut_M6_1")
        graph.add_node("nut2", node_type="part", name="Nut_M6_2")
        graph.add_node("screw1", node_type="part", name="Screw_Cap_1")
        graph.add_node("screw2", node_type="part", name="Screw_Cap_2")

        graph.root_id = "root"

        subassemblies = model.detect_subassemblies(graph)

        # No hardware should be grouped as subassemblies
        names = [s.name for s in subassemblies]
        assert not any("Bolt" in n for n in names), f"Bolts grouped: {names}"
        assert not any("Nut" in n for n in names), f"Nuts grouped: {names}"
        assert not any("Screw" in n for n in names), f"Screws grouped: {names}"

    def test_non_numeric_suffix_not_grouped(self):
        """Parts with non-numeric trailing segments should not be grouped."""
        pytest.importorskip("numpy")

        model = AssemblyAnalysisModel()

        graph = AssemblyGraph()
        graph.add_node("root", node_type="assembly", name="Root")
        graph.add_node("w1", node_type="part", name="Wheel_Front_Left")
        graph.add_node("w2", node_type="part", name="Wheel_Front_Right")

        graph.root_id = "root"

        subassemblies = model.detect_subassemblies(graph)

        # Non-numeric suffixes should not trigger grouping
        assert len(subassemblies) == 0


class TestMatingDetection:
    """Test mating surface detection."""

    def test_detect_mating_surfaces_without_occ(self):
        """Test mating detection without pythonocc."""
        model = AssemblyAnalysisModel()

        part1 = Mock()
        part1.item_id = "part1"

        part2 = Mock()
        part2.item_id = "part2"

        contacts = model.detect_mating_surfaces(part1, part2)

        # Should return empty if no OCC support
        if not model.has_pythonocc:
            assert len(contacts) == 0

    def test_detect_mating_surfaces_with_shapes(self):
        """Test mating detection with OCC shapes — overlapping boxes at origin."""
        pytest.importorskip("OCC")
        pytest.importorskip("numpy")

        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

        model = AssemblyAnalysisModel()

        # Two boxes at origin — all 6 faces overlap
        box1 = BRepPrimAPI_MakeBox(10, 10, 10).Shape()
        box2 = BRepPrimAPI_MakeBox(10, 10, 10).Shape()

        part1 = Mock()
        part1.item_id = "part1"
        part1._occ_shape = box1

        part2 = Mock()
        part2.item_id = "part2"
        part2._occ_shape = box2

        contacts = model.detect_mating_surfaces(part1, part2)

        # Overlapping boxes produce face-level contacts
        assert len(contacts) >= 1
        assert contacts[0].part1_id == "part1"
        assert contacts[0].part2_id == "part2"
        # Contacts should have real surface types, not generic "proximity"
        for c in contacts:
            assert c.contact_type != "proximity"
            assert c.confidence >= 0.7

    def test_detect_mating_surfaces_adjacent_boxes(self):
        """Test mating detection with boxes sharing a face."""
        pytest.importorskip("OCC")
        pytest.importorskip("numpy")

        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCC.Core.gp import gp_Pnt

        model = AssemblyAnalysisModel()

        # Box1 at origin, Box2 adjacent (touching at x=10)
        box1 = BRepPrimAPI_MakeBox(10, 10, 10).Shape()
        box2 = BRepPrimAPI_MakeBox(gp_Pnt(10, 0, 0), gp_Pnt(20, 10, 10)).Shape()

        part1 = Mock()
        part1.item_id = "part1"
        part1._occ_shape = box1

        part2 = Mock()
        part2.item_id = "part2"
        part2._occ_shape = box2

        contacts = model.detect_mating_surfaces(part1, part2)

        # Should detect the shared planar face
        assert len(contacts) >= 1
        planar = [c for c in contacts if c.contact_type == "planar"]
        assert len(planar) >= 1
        assert planar[0].distance < 1e-6
        assert planar[0].confidence >= 0.9

    def test_detect_mating_surfaces_distant_no_contact(self):
        """Test that distant parts produce no contacts."""
        pytest.importorskip("OCC")
        pytest.importorskip("numpy")

        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCC.Core.gp import gp_Pnt

        model = AssemblyAnalysisModel()

        # Box1 at origin, Box2 far away
        box1 = BRepPrimAPI_MakeBox(10, 10, 10).Shape()
        box2 = BRepPrimAPI_MakeBox(gp_Pnt(100, 100, 100), gp_Pnt(110, 110, 110)).Shape()

        part1 = Mock()
        part1.item_id = "part1"
        part1._occ_shape = box1

        part2 = Mock()
        part2.item_id = "part2"
        part2._occ_shape = box2

        contacts = model.detect_mating_surfaces(part1, part2)

        # No contacts — parts are far apart
        assert len(contacts) == 0


class TestIntegration:
    """Integration tests with real assembly files."""

    def test_assembly_analysis_with_synthetic_data(self):
        """Test assembly analysis with synthetic assembly data."""
        pytest.importorskip("numpy")

        from cadling.datamodel.stl import AssemblyItem

        model = AssemblyAnalysisModel()

        # Create synthetic assembly
        part1 = Mock()
        part1.item_type = "part"
        part1.item_id = "bolt_1"
        part1.name = "Bolt M8"

        part2 = Mock()
        part2.item_type = "part"
        part2.item_id = "bolt_2"
        part2.name = "Bolt M8"

        part3 = Mock()
        part3.item_type = "part"
        part3.item_id = "plate_1"
        part3.name = "Mounting Plate"

        asm = AssemblyItem(
            item_id="asm1",
            label=CADItemLabel(text="Assembly 1")
        )
        asm.add_component("bolt_1", "Bolt M8")
        asm.add_component("bolt_2", "Bolt M8")
        asm.add_component("plate_1", "Mounting Plate")

        class SimpleDoc:
            def __init__(self):
                self.properties = {}
                self.items = [asm, part1, part2, part3]

        doc = SimpleDoc()

        # Run analysis
        model(doc, doc.items)

        # Verify results
        assert "assembly_analysis" in doc.properties
        result = doc.properties["assembly_analysis"]

        assert "assembly_graph" in result
        assert "num_parts" in result
        assert result["num_parts"] == 3
        assert "bom" in result
        assert result["bom"]["total_parts"] == 3
        assert result["bom"]["unique_parts"] == 3

    def test_assembly_analysis_with_real_step_file(self):
        """Test assembly analysis with real STEP file if available."""
        pytest.importorskip("OCC")
        pytest.importorskip("numpy")

        from cadling.backend.document_converter import DocumentConverter

        # Find test STEP files
        test_data_dir = Path(__file__).parent.parent.parent.parent / "data" / "test_data" / "step"

        if not test_data_dir.exists():
            pytest.skip("Test data directory not found")

        step_files = list(test_data_dir.glob("*.stp"))[:1]
        if not step_files:
            step_files = list(test_data_dir.glob("*.step"))[:1]

        if not step_files:
            pytest.skip("No STEP files found in test data")

        step_file = step_files[0]

        # Load STEP file
        converter = DocumentConverter()
        result = converter.convert(str(step_file))

        if not result or not result.document:
            pytest.skip(f"Could not load STEP file: {step_file.name}")

        doc = result.document

        # Run assembly analysis
        model = AssemblyAnalysisModel()
        model(doc, doc.items)

        # Verify structure (permissive - just check it ran)
        if hasattr(doc, 'properties') and "assembly_analysis" in doc.properties:
            result = doc.properties["assembly_analysis"]
            assert "assembly_graph" in result
            assert "num_parts" in result
            # Success - analysis ran and produced results
        else:
            # No assembly data, but test passed (model ran successfully)
            pass
