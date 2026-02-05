"""
Unit tests for AssemblyAnalysisOptions and MateType.

Tests cover:
- MateType enumeration
- AssemblyAnalysisOptions initialization with defaults
- Component detection settings
- Mate extraction configuration
- BOM generation options
"""

import pytest
from pydantic import ValidationError

from cadling.experimental.datamodel import AssemblyAnalysisOptions, MateType


class TestMateType:
    """Test MateType enumeration."""

    def test_mate_types(self):
        """Test all mate type values."""
        assert MateType.COINCIDENT == "coincident"
        assert MateType.CONCENTRIC == "concentric"
        assert MateType.PARALLEL == "parallel"
        assert MateType.PERPENDICULAR == "perpendicular"
        assert MateType.TANGENT == "tangent"
        assert MateType.DISTANCE == "distance"
        assert MateType.ANGLE == "angle"
        assert MateType.FASTENER == "fastener"
        assert MateType.WELD == "weld"
        assert MateType.UNKNOWN == "unknown"

    def test_mate_type_from_string(self):
        """Test creating MateType from string."""
        mate = MateType("coincident")
        assert mate == MateType.COINCIDENT

        mate = MateType("fastener")
        assert mate == MateType.FASTENER

    def test_mate_type_invalid(self):
        """Test invalid mate type raises error."""
        with pytest.raises(ValueError):
            MateType("invalid_mate")

    def test_mate_type_iteration(self):
        """Test iterating over mate types."""
        all_mates = list(MateType)

        assert len(all_mates) == 10
        assert MateType.COINCIDENT in all_mates
        assert MateType.UNKNOWN in all_mates


class TestAssemblyAnalysisOptions:
    """Test AssemblyAnalysisOptions pydantic model."""

    def test_default_initialization(self):
        """Test initialization with default values."""
        options = AssemblyAnalysisOptions()

        assert options.detect_components is True
        assert options.extract_mates is True
        assert options.generate_bom is True
        assert options.check_interference is False
        assert options.max_components == 1000
        assert options.bom_include_metadata is True
        assert options.bom_include_properties is True
        assert options.interference_tolerance == 0.01
        assert options.process_subassemblies is True
        assert options.component_naming_strategy == "hierarchical"
        assert options.extract_fasteners is True
        assert options.group_identical_parts is True

    def test_default_mate_types(self):
        """Test default mate types to extract."""
        options = AssemblyAnalysisOptions()

        assert len(options.mate_types_to_extract) == 5
        assert MateType.COINCIDENT in options.mate_types_to_extract
        assert MateType.CONCENTRIC in options.mate_types_to_extract
        assert MateType.PARALLEL in options.mate_types_to_extract
        assert MateType.PERPENDICULAR in options.mate_types_to_extract
        assert MateType.FASTENER in options.mate_types_to_extract

    def test_custom_mate_types(self):
        """Test custom mate types configuration."""
        options = AssemblyAnalysisOptions(
            mate_types_to_extract=[MateType.COINCIDENT, MateType.DISTANCE]
        )

        assert len(options.mate_types_to_extract) == 2
        assert MateType.COINCIDENT in options.mate_types_to_extract
        assert MateType.DISTANCE in options.mate_types_to_extract

    def test_disable_component_detection(self):
        """Test disabling component detection."""
        options = AssemblyAnalysisOptions(detect_components=False)

        assert options.detect_components is False

    def test_disable_mate_extraction(self):
        """Test disabling mate extraction."""
        options = AssemblyAnalysisOptions(extract_mates=False)

        assert options.extract_mates is False

    def test_disable_bom_generation(self):
        """Test disabling BOM generation."""
        options = AssemblyAnalysisOptions(generate_bom=False)

        assert options.generate_bom is False

    def test_enable_interference_checking(self):
        """Test enabling interference checking."""
        options = AssemblyAnalysisOptions(check_interference=True)

        assert options.check_interference is True

    def test_max_components_validation(self):
        """Test max components validation."""
        # Valid values
        options = AssemblyAnalysisOptions(max_components=1)
        assert options.max_components == 1

        options = AssemblyAnalysisOptions(max_components=10000)
        assert options.max_components == 10000

        options = AssemblyAnalysisOptions(max_components=500)
        assert options.max_components == 500

        # Out of bounds
        with pytest.raises(ValidationError):
            AssemblyAnalysisOptions(max_components=0)

        with pytest.raises(ValidationError):
            AssemblyAnalysisOptions(max_components=10001)

    def test_interference_tolerance_validation(self):
        """Test interference tolerance validation."""
        # Valid values
        options = AssemblyAnalysisOptions(interference_tolerance=0.0)
        assert options.interference_tolerance == 0.0

        options = AssemblyAnalysisOptions(interference_tolerance=0.1)
        assert options.interference_tolerance == 0.1

        # Negative should fail
        with pytest.raises(ValidationError):
            AssemblyAnalysisOptions(interference_tolerance=-0.01)

    def test_bom_metadata_options(self):
        """Test BOM metadata inclusion options."""
        options = AssemblyAnalysisOptions(
            bom_include_metadata=False, bom_include_properties=False
        )

        assert options.bom_include_metadata is False
        assert options.bom_include_properties is False

    def test_naming_strategies(self):
        """Test different component naming strategies."""
        # Hierarchical
        options = AssemblyAnalysisOptions(component_naming_strategy="hierarchical")
        assert options.component_naming_strategy == "hierarchical"

        # Flat
        options = AssemblyAnalysisOptions(component_naming_strategy="flat")
        assert options.component_naming_strategy == "flat"

        # Preserve original
        options = AssemblyAnalysisOptions(
            component_naming_strategy="preserve_original"
        )
        assert options.component_naming_strategy == "preserve_original"

    def test_disable_subassembly_processing(self):
        """Test disabling subassembly processing."""
        options = AssemblyAnalysisOptions(process_subassemblies=False)

        assert options.process_subassemblies is False

    def test_disable_fastener_extraction(self):
        """Test disabling fastener extraction."""
        options = AssemblyAnalysisOptions(extract_fasteners=False)

        assert options.extract_fasteners is False

    def test_disable_identical_parts_grouping(self):
        """Test disabling identical parts grouping."""
        options = AssemblyAnalysisOptions(group_identical_parts=False)

        assert options.group_identical_parts is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        options = AssemblyAnalysisOptions(
            detect_components=True,
            generate_bom=True,
            max_components=500,
        )

        data = options.model_dump()

        assert data["detect_components"] is True
        assert data["generate_bom"] is True
        assert data["max_components"] == 500
        assert "mate_types_to_extract" in data

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "detect_components": False,
            "extract_mates": True,
            "generate_bom": True,
            "max_components": 200,
            "mate_types_to_extract": ["coincident", "parallel"],
        }

        options = AssemblyAnalysisOptions(**data)

        assert options.detect_components is False
        assert options.max_components == 200
        assert len(options.mate_types_to_extract) == 2

    def test_kind_field(self):
        """Test that kind field is set correctly."""
        options = AssemblyAnalysisOptions()

        assert options.kind == "cadling_experimental_assembly"

    def test_minimal_configuration(self):
        """Test minimal configuration (everything disabled)."""
        options = AssemblyAnalysisOptions(
            detect_components=False,
            extract_mates=False,
            generate_bom=False,
            check_interference=False,
            process_subassemblies=False,
            extract_fasteners=False,
            group_identical_parts=False,
        )

        assert options.detect_components is False
        assert options.extract_mates is False
        assert options.generate_bom is False
        assert options.check_interference is False
        assert options.process_subassemblies is False
        assert options.extract_fasteners is False
        assert options.group_identical_parts is False

    def test_maximal_configuration(self):
        """Test maximal configuration (everything enabled)."""
        options = AssemblyAnalysisOptions(
            detect_components=True,
            extract_mates=True,
            generate_bom=True,
            check_interference=True,
            process_subassemblies=True,
            extract_fasteners=True,
            group_identical_parts=True,
            max_components=10000,
            mate_types_to_extract=list(MateType),
        )

        assert options.detect_components is True
        assert options.extract_mates is True
        assert options.generate_bom is True
        assert options.check_interference is True
        assert options.max_components == 10000
        assert len(options.mate_types_to_extract) == len(MateType)

    def test_empty_mate_types(self):
        """Test with empty mate types list."""
        options = AssemblyAnalysisOptions(mate_types_to_extract=[])

        assert options.mate_types_to_extract == []
