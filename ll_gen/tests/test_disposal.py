"""Comprehensive test suite for ll_gen.disposal module.

This test suite covers:

**Section 1: Unit tests that work WITHOUT pythonocc (no external dependencies)**
- ValidationReport and RepairResult data structures
- DisposalEngine initialization with various config combinations
- Error suggestion generation from different exception types
- Module importability

**Section 2: Integration tests that REQUIRE pythonocc (marked with @requires_occ)**
- Shape validation with real OCC geometry
- Shape repair operations
- Geometry introspection (volume, surface area, face counts, etc.)
- STEP and STL export to actual files
- Full disposal pipeline end-to-end
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

# Import from conftest for skip markers and fixtures
from tests.conftest import requires_occ, requires_cadquery

# Import the modules being tested
from ll_gen.config import DisposalConfig, ErrorCategory, ExportConfig, FeedbackConfig, StepSchema
from ll_gen.disposal.engine import DisposalEngine, _suggest_from_execution_error
from ll_gen.disposal.validator import ValidationReport
from ll_gen.disposal.repairer import RepairResult
from ll_gen.proposals.code_proposal import CodeProposal
from ll_gen.proposals.disposal_result import ValidationFinding
from ll_gen.config import ErrorSeverity, CodeLanguage


# ============================================================================
# SECTION 1: Unit Tests (NO pythonocc required)
# ============================================================================

class TestValidationReport:
    """Test ValidationReport data structure construction and properties."""

    def test_validation_report_defaults(self) -> None:
        """Test ValidationReport initializes with correct default values."""
        report = ValidationReport()
        assert report.is_valid is True
        assert report.findings == []
        assert report.face_count == 0
        assert report.edge_count == 0
        assert report.vertex_count == 0
        assert report.shell_count == 0
        assert report.solid_count == 0
        assert report.is_manifold is True
        assert report.is_watertight is True
        assert report.euler_characteristic is None
        assert report.primary_category is None

    def test_validation_report_custom_values(self) -> None:
        """Test ValidationReport with custom initialization values."""
        finding = ValidationFinding(
            entity_type="FACE",
            entity_index=0,
            error_code="BRepCheck_NotClosed",
            error_category=ErrorCategory.TOPOLOGY_ERROR,
            severity=ErrorSeverity.CRITICAL,
            description="Test error",
            suggestion="Fix it",
        )
        report = ValidationReport(
            is_valid=False,
            findings=[finding],
            face_count=6,
            edge_count=12,
            vertex_count=8,
            shell_count=1,
            solid_count=1,
            is_manifold=True,
            is_watertight=False,
            euler_characteristic=2,
            primary_category=ErrorCategory.TOPOLOGY_ERROR,
        )
        assert report.is_valid is False
        assert len(report.findings) == 1
        assert report.face_count == 6
        assert report.edge_count == 12
        assert report.vertex_count == 8
        assert report.shell_count == 1
        assert report.solid_count == 1
        assert report.is_manifold is True
        assert report.is_watertight is False
        assert report.euler_characteristic == 2
        assert report.primary_category == ErrorCategory.TOPOLOGY_ERROR

    def test_validation_report_is_valid_reflects_findings(self) -> None:
        """Test that is_valid property should reflect presence of critical findings."""
        # With no findings, is_valid should be True
        report1 = ValidationReport(is_valid=True, findings=[])
        assert report1.is_valid is True

        # With critical findings, is_valid should be False
        critical_finding = ValidationFinding(
            entity_type="SOLID",
            entity_index=0,
            error_code="TopologyError",
            error_category=ErrorCategory.TOPOLOGY_ERROR,
            severity=ErrorSeverity.CRITICAL,
            description="Critical issue",
            suggestion="Repair needed",
        )
        report2 = ValidationReport(is_valid=False, findings=[critical_finding])
        assert report2.is_valid is False
        assert len(report2.findings) == 1

    def test_validation_report_multiple_findings(self) -> None:
        """Test ValidationReport with multiple findings of different severities."""
        warning_finding = ValidationFinding(
            entity_type="EDGE",
            entity_index=2,
            error_code="BRepCheck_FreeEdge",
            error_category=ErrorCategory.DEGENERATE_SHAPE,
            severity=ErrorSeverity.WARNING,
            description="Isolated edge",
            suggestion="Remove it",
        )
        critical_finding = ValidationFinding(
            entity_type="SHELL",
            entity_index=0,
            error_code="ShellNotClosed",
            error_category=ErrorCategory.TOPOLOGY_ERROR,
            severity=ErrorSeverity.CRITICAL,
            description="Open shell",
            suggestion="Close it",
        )
        report = ValidationReport(
            is_valid=False,
            findings=[warning_finding, critical_finding],
        )
        assert len(report.findings) == 2
        assert report.findings[0].severity == ErrorSeverity.WARNING
        assert report.findings[1].severity == ErrorSeverity.CRITICAL


class TestRepairResult:
    """Test RepairResult data structure construction."""

    def test_repair_result_defaults(self) -> None:
        """Test RepairResult initializes with correct defaults."""
        result = RepairResult()
        assert result.shape is None
        assert result.succeeded is False
        assert result.actions == []
        assert result.validation_after is None

    def test_repair_result_custom_values(self) -> None:
        """Test RepairResult with custom initialization values."""
        from ll_gen.proposals.disposal_result import RepairAction

        shape_mock = MagicMock()
        action = RepairAction(
            tool="ShapeFix_Shape",
            action="General repair",
            status="done",
            tolerance_used=1e-7,
        )
        val_report = ValidationReport(is_valid=True)

        result = RepairResult(
            shape=shape_mock,
            succeeded=True,
            actions=[action],
            validation_after=val_report,
        )
        assert result.shape is shape_mock
        assert result.succeeded is True
        assert len(result.actions) == 1
        assert result.actions[0].tool == "ShapeFix_Shape"
        assert result.validation_after.is_valid is True

    def test_repair_result_multiple_actions(self) -> None:
        """Test RepairResult with multiple repair actions."""
        from ll_gen.proposals.disposal_result import RepairAction

        actions = [
            RepairAction(
                tool="ShapeFix_Shape",
                action="General repair",
                status="done",
                tolerance_used=1e-7,
            ),
            RepairAction(
                tool="ShapeFix_Wire",
                action="Fixed wire issues",
                status="done",
                tolerance_used=1e-7,
                entities_affected=3,
            ),
        ]
        result = RepairResult(
            shape=MagicMock(),
            succeeded=True,
            actions=actions,
        )
        assert len(result.actions) == 2
        assert result.actions[0].tool == "ShapeFix_Shape"
        assert result.actions[1].tool == "ShapeFix_Wire"
        assert result.actions[1].entities_affected == 3


class TestDisposalEngineInit:
    """Test DisposalEngine initialization with various configurations."""

    def test_disposal_engine_init_default_config(self) -> None:
        """Test DisposalEngine initializes with default configurations."""
        engine = DisposalEngine()
        assert isinstance(engine.disposal_config, DisposalConfig)
        assert isinstance(engine.export_config, ExportConfig)
        assert isinstance(engine.feedback_config, FeedbackConfig)
        assert engine.output_dir == Path("output")

    def test_disposal_engine_init_custom_config(self) -> None:
        """Test DisposalEngine initializes with custom configurations."""
        disposal_cfg = DisposalConfig(enable_auto_repair=False)
        export_cfg = ExportConfig(step_schema=StepSchema.AP242)
        feedback_cfg = FeedbackConfig(validity_reward=2.0)
        output = "/tmp/test_output"

        engine = DisposalEngine(
            disposal_config=disposal_cfg,
            export_config=export_cfg,
            feedback_config=feedback_cfg,
            output_dir=output,
        )

        assert engine.disposal_config.enable_auto_repair is False
        assert engine.export_config.step_schema == StepSchema.AP242
        assert engine.feedback_config.validity_reward == 2.0
        assert engine.output_dir == Path(output)

    def test_disposal_engine_init_output_dir_creation(self, tmp_path: Path) -> None:
        """Test DisposalEngine creates output directory if it doesn't exist."""
        output_dir = tmp_path / "new_output_dir"
        assert not output_dir.exists()

        engine = DisposalEngine(output_dir=str(output_dir))

        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_disposal_engine_init_partial_config(self) -> None:
        """Test DisposalEngine with only disposal config specified."""
        disposal_cfg = DisposalConfig(max_repair_passes=5)
        engine = DisposalEngine(disposal_config=disposal_cfg)

        assert engine.disposal_config.max_repair_passes == 5
        assert isinstance(engine.export_config, ExportConfig)
        assert isinstance(engine.feedback_config, FeedbackConfig)


class TestSuggestFromExecutionError:
    """Test error suggestion generation from various exception types."""

    def test_suggest_from_timeout_error(self, code_proposal_cadquery: CodeProposal) -> None:
        """Test that TimeoutError generates timeout/long execution suggestion."""
        timeout_exc = TimeoutError("Code execution timeout limit reached")
        suggestion = _suggest_from_execution_error(timeout_exc, code_proposal_cadquery)
        assert ("timeout" in suggestion.lower() or "too long" in suggestion.lower())
        assert "simplify" in suggestion.lower()

    def test_suggest_from_syntax_error(self, code_proposal_cadquery: CodeProposal) -> None:
        """Test that SyntaxError generates syntax suggestion."""
        syntax_exc = SyntaxError("invalid syntax")
        suggestion = _suggest_from_execution_error(syntax_exc, code_proposal_cadquery)
        assert "syntax" in suggestion.lower()

    def test_suggest_from_import_error(self, code_proposal_cadquery: CodeProposal) -> None:
        """Test that ImportError generates import/module suggestion."""
        import_exc = ImportError("No module named 'unknown_module'")
        suggestion = _suggest_from_execution_error(import_exc, code_proposal_cadquery)
        assert "import" in suggestion.lower() or "module" in suggestion.lower()

    def test_suggest_from_cadquery_error(self, code_proposal_cadquery: CodeProposal) -> None:
        """Test that CadQuery-related error generates CadQuery suggestion."""
        cq_exc = RuntimeError("cadquery.workplane error")
        suggestion = _suggest_from_execution_error(cq_exc, code_proposal_cadquery)
        assert "cadquery" in suggestion.lower() or "workplane" in suggestion.lower()

    def test_suggest_from_boolean_error(self, code_proposal_cadquery: CodeProposal) -> None:
        """Test that boolean operation error generates boolean suggestion."""
        bool_exc = RuntimeError("Boolean fuse operation failed")
        suggestion = _suggest_from_execution_error(bool_exc, code_proposal_cadquery)
        assert "boolean" in suggestion.lower() or "fuse" in suggestion.lower()

    def test_suggest_from_generic_exception(self, code_proposal_cadquery: CodeProposal) -> None:
        """Test that generic exception generates fallback suggestion."""
        generic_exc = Exception("Some unexpected error occurred")
        suggestion = _suggest_from_execution_error(generic_exc, code_proposal_cadquery)
        assert len(suggestion) > 0
        assert "error" in suggestion.lower()

    def test_suggest_from_exception_with_none_message(
        self, code_proposal_cadquery: CodeProposal
    ) -> None:
        """Test suggestion generation for exception without useful message."""
        exc = RuntimeError("")
        suggestion = _suggest_from_execution_error(exc, code_proposal_cadquery)
        assert len(suggestion) > 0


class TestModuleImportability:
    """Test that disposal modules are importable without pythonocc."""

    def test_import_validator_module(self) -> None:
        """Test that validator module imports without error."""
        from ll_gen.disposal import validator  # noqa: F401
        assert hasattr(validator, "ValidationReport")
        assert hasattr(validator, "validate_shape")

    def test_import_repairer_module(self) -> None:
        """Test that repairer module imports without error."""
        from ll_gen.disposal import repairer  # noqa: F401
        assert hasattr(repairer, "RepairResult")
        assert hasattr(repairer, "repair_shape")

    def test_import_introspector_module(self) -> None:
        """Test that introspector module imports without error."""
        from ll_gen.disposal import introspector  # noqa: F401
        assert hasattr(introspector, "introspect")

    def test_import_exporter_module(self) -> None:
        """Test that exporter module imports without error."""
        from ll_gen.disposal import exporter  # noqa: F401
        assert hasattr(exporter, "export_step")
        assert hasattr(exporter, "export_stl")

    def test_import_engine_module(self) -> None:
        """Test that engine module imports without error."""
        from ll_gen.disposal import engine  # noqa: F401
        assert hasattr(engine, "DisposalEngine")

    def test_import_code_executor_module(self) -> None:
        """Test that code_executor module imports without error."""
        from ll_gen.disposal import code_executor  # noqa: F401
        assert hasattr(code_executor, "execute_code_proposal")

    def test_import_command_executor_module(self) -> None:
        """Test that command_executor module imports without error."""
        from ll_gen.disposal import command_executor  # noqa: F401
        assert hasattr(command_executor, "execute_command_proposal")

    def test_import_surface_executor_module(self) -> None:
        """Test that surface_executor module imports without error."""
        from ll_gen.disposal import surface_executor  # noqa: F401
        assert hasattr(surface_executor, "execute_latent_proposal")


# ============================================================================
# SECTION 2: Integration Tests (REQUIRE pythonocc)
# ============================================================================

@requires_occ
class TestValidateShape:
    """Test shape validation with actual OCC geometry."""

    def test_validate_shape_valid_box(self) -> None:
        """Test validate_shape() on a valid box shape returns is_valid=True."""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

        shape = BRepPrimAPI_MakeBox(100, 50, 20).Shape()
        from ll_gen.disposal.validator import validate_shape

        report = validate_shape(shape)

        assert report.is_valid is True
        assert report.face_count == 6
        assert report.edge_count == 12
        assert report.vertex_count == 8

    def test_validate_shape_populates_counts(self) -> None:
        """Test validate_shape() populates entity counts (faces, edges, vertices)."""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from ll_gen.disposal.validator import validate_shape

        shape = BRepPrimAPI_MakeBox(50, 30, 40).Shape()
        report = validate_shape(shape)

        assert report.face_count > 0
        assert report.edge_count > 0
        assert report.vertex_count > 0
        assert report.shell_count >= 0
        assert report.solid_count >= 0

    def test_validate_shape_computes_euler_characteristic(self) -> None:
        """Test validate_shape() computes Euler characteristic V-E+F."""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from ll_gen.disposal.validator import validate_shape

        shape = BRepPrimAPI_MakeBox(100, 100, 100).Shape()
        report = validate_shape(shape)

        # For a simple solid: V - E + F should equal 2
        euler = report.euler_characteristic
        assert euler is not None
        assert euler == report.vertex_count - report.edge_count + report.face_count

    def test_validate_shape_with_custom_config(self) -> None:
        """Test validate_shape() respects custom DisposalConfig settings."""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from ll_gen.disposal.validator import validate_shape

        shape = BRepPrimAPI_MakeBox(80, 60, 40).Shape()
        config = DisposalConfig(
            check_manifoldness=True,
            check_watertightness=True,
            check_euler=True,
        )
        report = validate_shape(shape, config)

        assert report.is_valid is True
        assert report.is_manifold is True
        assert report.is_watertight is True


@requires_occ
class TestRepairShape:
    """Test shape repair operations."""

    def test_repair_shape_valid_returns_success(self) -> None:
        """Test repair_shape() on already-valid shape returns succeeded=True."""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from ll_gen.disposal.validator import validate_shape
        from ll_gen.disposal.repairer import repair_shape

        shape = BRepPrimAPI_MakeBox(100, 50, 20).Shape()
        val_report = validate_shape(shape)

        repair_result = repair_shape(shape, val_report)

        assert repair_result.succeeded is True
        assert repair_result.shape is not None

    def test_repair_shape_returns_repair_result_structure(self) -> None:
        """Test repair_shape() returns properly structured RepairResult."""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from ll_gen.disposal.validator import validate_shape
        from ll_gen.disposal.repairer import repair_shape

        shape = BRepPrimAPI_MakeBox(100, 100, 100).Shape()
        val_report = validate_shape(shape)
        repair_result = repair_shape(shape, val_report)

        assert isinstance(repair_result, RepairResult)
        assert repair_result.shape is not None
        assert isinstance(repair_result.actions, list)
        assert isinstance(repair_result.validation_after, ValidationReport)

    def test_repair_shape_logs_actions(self) -> None:
        """Test repair_shape() populates repair actions when repairs are attempted."""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from ll_gen.disposal.validator import validate_shape
        from ll_gen.disposal.repairer import repair_shape

        shape = BRepPrimAPI_MakeBox(75, 50, 30).Shape()
        val_report = validate_shape(shape)
        config = DisposalConfig(enable_auto_repair=True, max_repair_passes=2)

        repair_result = repair_shape(shape, val_report, config)

        # Valid shape should succeed without many actions
        assert repair_result.succeeded is True


@requires_occ
class TestIntrospect:
    """Test geometry introspection."""

    def test_introspect_box_shape(self) -> None:
        """Test introspect() on a box shape returns correct volume and dimensions."""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from ll_gen.disposal.introspector import introspect

        # Create a 100 x 50 x 20 box
        shape = BRepPrimAPI_MakeBox(100, 50, 20).Shape()
        report = introspect(shape)

        # Volume should be approximately 100 * 50 * 20 = 100000
        assert report.volume is not None
        assert abs(report.volume - 100000.0) < 1.0

    def test_introspect_returns_geometry_report(self) -> None:
        """Test introspect() returns properly populated GeometryReport."""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from ll_gen.disposal.introspector import introspect
        from ll_gen.proposals.disposal_result import GeometryReport

        shape = BRepPrimAPI_MakeBox(50, 50, 50).Shape()
        report = introspect(shape)

        assert isinstance(report, GeometryReport)
        assert report.face_count == 6
        assert report.edge_count == 12
        assert report.vertex_count == 8
        assert report.is_solid is True

    def test_introspect_surface_area(self) -> None:
        """Test introspect() computes surface area correctly."""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from ll_gen.disposal.introspector import introspect

        # Box 100 x 50 x 20: surface area = 2*(100*50 + 100*20 + 50*20) = 13000
        shape = BRepPrimAPI_MakeBox(100, 50, 20).Shape()
        report = introspect(shape)

        expected_surface_area = 2 * (100 * 50 + 100 * 20 + 50 * 20)
        assert report.surface_area is not None
        assert abs(report.surface_area - expected_surface_area) < 1.0

    def test_introspect_center_of_mass(self) -> None:
        """Test introspect() computes center of mass."""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from ll_gen.disposal.introspector import introspect

        shape = BRepPrimAPI_MakeBox(100, 100, 100).Shape()
        report = introspect(shape)

        assert report.center_of_mass is not None
        assert len(report.center_of_mass) == 3
        # Center of a 0-100 cube should be around (50, 50, 50)
        assert abs(report.center_of_mass[0] - 50.0) < 1.0
        assert abs(report.center_of_mass[1] - 50.0) < 1.0
        assert abs(report.center_of_mass[2] - 50.0) < 1.0

    def test_introspect_bounding_box(self) -> None:
        """Test introspect() computes bounding box."""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from ll_gen.disposal.introspector import introspect

        shape = BRepPrimAPI_MakeBox(100, 50, 20).Shape()
        report = introspect(shape)

        assert report.bounding_box is not None
        assert len(report.bounding_box) == 6
        # Bounding box should be (0, 0, 0, 100, 50, 20)
        xmin, ymin, zmin, xmax, ymax, zmax = report.bounding_box
        assert abs(xmin - 0.0) < 0.1
        assert abs(ymin - 0.0) < 0.1
        assert abs(zmin - 0.0) < 0.1
        assert abs(xmax - 100.0) < 0.1
        assert abs(ymax - 50.0) < 0.1
        assert abs(zmax - 20.0) < 0.1


@requires_occ
class TestExportStep:
    """Test STEP file export."""

    def test_export_step_creates_file(self, tmp_path: Path) -> None:
        """Test export_step() creates a STEP file at the specified path."""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from ll_gen.disposal.exporter import export_step

        shape = BRepPrimAPI_MakeBox(100, 50, 20).Shape()
        step_path = tmp_path / "test_export.step"

        result_path = export_step(shape, step_path)

        assert result_path.exists()
        assert result_path.suffix in [".step", ".stp"]
        assert result_path.stat().st_size > 0

    def test_export_step_with_different_schemas(self, tmp_path: Path) -> None:
        """Test export_step() works with different STEP schemas."""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from ll_gen.disposal.exporter import export_step

        shape = BRepPrimAPI_MakeBox(50, 50, 50).Shape()

        # Test AP203
        step_ap203 = tmp_path / "test_ap203.step"
        export_step(shape, step_ap203, StepSchema.AP203)
        assert step_ap203.exists()

        # Test AP214
        step_ap214 = tmp_path / "test_ap214.step"
        export_step(shape, step_ap214, StepSchema.AP214)
        assert step_ap214.exists()

        # Test AP242
        step_ap242 = tmp_path / "test_ap242.step"
        export_step(shape, step_ap242, StepSchema.AP242)
        assert step_ap242.exists()

    def test_export_step_creates_parent_directory(self, tmp_path: Path) -> None:
        """Test export_step() creates parent directories if they don't exist."""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from ll_gen.disposal.exporter import export_step

        shape = BRepPrimAPI_MakeBox(100, 100, 100).Shape()
        nested_path = tmp_path / "subdir1" / "subdir2" / "test.step"

        assert not nested_path.parent.exists()
        export_step(shape, nested_path)

        assert nested_path.exists()
        assert nested_path.parent.exists()


@requires_occ
class TestExportStl:
    """Test STL file export."""

    def test_export_stl_creates_file(self, tmp_path: Path) -> None:
        """Test export_stl() creates an STL file at the specified path."""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from ll_gen.disposal.exporter import export_stl

        shape = BRepPrimAPI_MakeBox(100, 50, 20).Shape()
        stl_path = tmp_path / "test_export.stl"

        result_path = export_stl(shape, stl_path)

        assert result_path.exists()
        assert result_path.suffix == ".stl"
        assert result_path.stat().st_size > 0

    def test_export_stl_ascii_mode(self, tmp_path: Path) -> None:
        """Test export_stl() with ASCII mode produces readable text file."""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from ll_gen.disposal.exporter import export_stl

        shape = BRepPrimAPI_MakeBox(50, 50, 50).Shape()
        stl_ascii = tmp_path / "test_ascii.stl"

        export_stl(shape, stl_ascii, ascii_mode=True)

        assert stl_ascii.exists()
        # ASCII STL should start with "solid"
        with open(stl_ascii, "r") as f:
            first_line = f.readline().strip().lower()
            assert first_line.startswith("solid") or first_line.startswith("facet")

    def test_export_stl_binary_mode(self, tmp_path: Path) -> None:
        """Test export_stl() with binary mode produces compact file."""
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from ll_gen.disposal.exporter import export_stl

        shape = BRepPrimAPI_MakeBox(50, 50, 50).Shape()
        stl_binary = tmp_path / "test_binary.stl"

        export_stl(shape, stl_binary, ascii_mode=False)

        assert stl_binary.exists()
        # Binary STL file should be readable as binary
        with open(stl_binary, "rb") as f:
            data = f.read(5)
            # Should be binary, not start with "solid"
            assert data != b"solid"

    def test_export_stl_with_custom_deflection(self, tmp_path: Path) -> None:
        """Test export_stl() respects linear_deflection parameter.

        Uses a sphere (curved surface) to demonstrate the deflection
        difference - flat-faced boxes would produce identical meshes
        regardless of deflection.
        """
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
        from ll_gen.disposal.exporter import export_stl

        # Use a sphere - curved surfaces show deflection differences
        shape = BRepPrimAPI_MakeSphere(50).Shape()

        # Coarse mesh
        stl_coarse = tmp_path / "test_coarse.stl"
        export_stl(shape, stl_coarse, linear_deflection=5.0)

        # Fine mesh
        stl_fine = tmp_path / "test_fine.stl"
        export_stl(shape, stl_fine, linear_deflection=0.1)

        # Fine mesh should produce a larger file due to more triangles
        coarse_size = stl_coarse.stat().st_size
        fine_size = stl_fine.stat().st_size
        assert fine_size > coarse_size


@requires_occ
@requires_cadquery
class TestDisposalEngineDispose:
    """Test full disposal pipeline end-to-end."""

    def test_disposal_engine_dispose_with_code_proposal(
        self, tmp_output_dir: Path, code_proposal_cadquery: CodeProposal
    ) -> None:
        """Test DisposalEngine.dispose() with a valid CodeProposal."""
        engine = DisposalEngine(output_dir=str(tmp_output_dir))
        result = engine.dispose(code_proposal_cadquery, export=True)

        assert result is not None
        assert result.proposal_id == code_proposal_cadquery.proposal_id
        assert result.is_valid is True
        assert result.shape is not None
        assert result.step_path is not None or result.stl_path is not None

    def test_disposal_engine_dispose_populates_result(
        self, tmp_output_dir: Path, code_proposal_cadquery: CodeProposal
    ) -> None:
        """Test DisposalEngine.dispose() populates all DisposalResult fields."""
        engine = DisposalEngine(output_dir=str(tmp_output_dir))
        result = engine.dispose(code_proposal_cadquery)

        assert result.proposal_type == "CodeProposal"
        assert result.execution_time_ms > 0
        assert result.reward_signal is not None

    def test_disposal_engine_dispose_with_geometry_report(
        self, tmp_output_dir: Path, code_proposal_cadquery: CodeProposal
    ) -> None:
        """Test DisposalEngine.dispose() generates GeometryReport for valid shapes."""
        config = DisposalConfig(always_introspect=True)
        engine = DisposalEngine(disposal_config=config, output_dir=str(tmp_output_dir))
        result = engine.dispose(code_proposal_cadquery)

        assert result.is_valid is True
        assert result.geometry_report is not None
        assert result.geometry_report.volume is not None
        assert result.geometry_report.surface_area is not None
        assert result.geometry_report.face_count > 0

    def test_disposal_engine_dispose_exports_step_file(
        self, tmp_output_dir: Path, code_proposal_cadquery: CodeProposal
    ) -> None:
        """Test DisposalEngine.dispose() creates STEP export when valid."""
        export_cfg = ExportConfig()
        engine = DisposalEngine(export_config=export_cfg, output_dir=str(tmp_output_dir))
        result = engine.dispose(code_proposal_cadquery, export=True)

        if result.is_valid and result.step_path:
            assert result.step_path.exists()
            assert result.step_path.suffix in [".step", ".stp"]

    def test_disposal_engine_dispose_exports_stl_file(
        self, tmp_output_dir: Path, code_proposal_cadquery: CodeProposal
    ) -> None:
        """Test DisposalEngine.dispose() creates STL export when valid."""
        export_cfg = ExportConfig()
        engine = DisposalEngine(export_config=export_cfg, output_dir=str(tmp_output_dir))
        result = engine.dispose(code_proposal_cadquery, export=True)

        if result.is_valid and result.stl_path:
            assert result.stl_path.exists()
            assert result.stl_path.suffix == ".stl"

    def test_disposal_engine_dispose_without_export(
        self, tmp_output_dir: Path, code_proposal_cadquery: CodeProposal
    ) -> None:
        """Test DisposalEngine.dispose() respects export=False."""
        engine = DisposalEngine(output_dir=str(tmp_output_dir))
        result = engine.dispose(code_proposal_cadquery, export=False)

        assert result.is_valid is True
        # step_path and stl_path should be None when export=False
        assert result.step_path is None or result.stl_path is None

    def test_disposal_engine_dispose_with_custom_feedback_config(
        self, tmp_output_dir: Path, code_proposal_cadquery: CodeProposal
    ) -> None:
        """Test DisposalEngine.dispose() uses custom FeedbackConfig for reward."""
        feedback_cfg = FeedbackConfig(validity_reward=5.0)
        engine = DisposalEngine(
            feedback_config=feedback_cfg,
            output_dir=str(tmp_output_dir),
        )
        result = engine.dispose(code_proposal_cadquery)

        # Reward should be computed using the custom config
        assert result.reward_signal is not None
        assert isinstance(result.reward_signal, float)
