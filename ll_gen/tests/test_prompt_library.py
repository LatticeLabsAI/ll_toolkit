"""Comprehensive test suite for ll_gen.codegen.prompt_library module.

Tests prompt template functionality:
- API reference content
- Few-shot example formatting
- System prompt assembly
- Error recovery templates
- Repair prompt construction
"""
from __future__ import annotations

from typing import Dict, Optional

import pytest

from ll_gen.config import ErrorCategory
from ll_gen.codegen.prompt_library import (
    CADQUERY_API_REFERENCE,
    CADQUERY_EXAMPLES,
    ERROR_RECOVERY_TEMPLATES,
    get_system_prompt,
    get_repair_prompt,
)


# ============================================================================
# SECTION 1: API Reference Tests
# ============================================================================


class TestCadQueryAPIReference:
    """Test CadQuery API reference content."""

    def test_api_reference_exists(self) -> None:
        """Test CADQUERY_API_REFERENCE is defined."""
        assert CADQUERY_API_REFERENCE is not None
        assert len(CADQUERY_API_REFERENCE) > 0

    def test_api_reference_contains_initialization(self) -> None:
        """Test API reference contains initialization section."""
        assert "INITIALIZATION" in CADQUERY_API_REFERENCE
        assert 'cq("XY")' in CADQUERY_API_REFERENCE

    def test_api_reference_contains_sketching(self) -> None:
        """Test API reference contains sketching section."""
        assert "SKETCHING" in CADQUERY_API_REFERENCE
        assert ".circle" in CADQUERY_API_REFERENCE
        assert ".rect" in CADQUERY_API_REFERENCE

    def test_api_reference_contains_selection(self) -> None:
        """Test API reference contains selection section."""
        assert "SELECTION" in CADQUERY_API_REFERENCE or ".faces" in CADQUERY_API_REFERENCE
        assert '.faces(">Z")' in CADQUERY_API_REFERENCE

    def test_api_reference_contains_features(self) -> None:
        """Test API reference contains feature operations."""
        assert ".hole" in CADQUERY_API_REFERENCE
        assert ".fillet" in CADQUERY_API_REFERENCE
        assert ".chamfer" in CADQUERY_API_REFERENCE

    def test_api_reference_contains_3d_operations(self) -> None:
        """Test API reference contains 3D operations."""
        assert ".extrude" in CADQUERY_API_REFERENCE
        assert ".cut" in CADQUERY_API_REFERENCE or ".union" in CADQUERY_API_REFERENCE


# ============================================================================
# SECTION 2: CadQuery Examples Tests
# ============================================================================


class TestCadQueryExamples:
    """Test CadQuery few-shot examples."""

    def test_examples_dict_exists(self) -> None:
        """Test CADQUERY_EXAMPLES is defined."""
        assert CADQUERY_EXAMPLES is not None
        assert isinstance(CADQUERY_EXAMPLES, dict)

    def test_examples_contain_bracket(self) -> None:
        """Test examples include bracket example."""
        assert "bracket" in CADQUERY_EXAMPLES
        bracket_code = CADQUERY_EXAMPLES["bracket"]
        assert "from cadquery import Workplane as cq" in bracket_code
        assert "result.val()" in bracket_code

    def test_examples_contain_gear(self) -> None:
        """Test examples include gear example."""
        assert "gear" in CADQUERY_EXAMPLES
        gear_code = CADQUERY_EXAMPLES["gear"]
        assert "num_teeth" in gear_code

    def test_examples_contain_enclosure(self) -> None:
        """Test examples include enclosure example."""
        assert "enclosure" in CADQUERY_EXAMPLES
        enclosure_code = CADQUERY_EXAMPLES["enclosure"]
        assert "wall_thickness" in enclosure_code

    def test_examples_contain_hinge(self) -> None:
        """Test examples include hinge example."""
        assert "hinge" in CADQUERY_EXAMPLES

    def test_examples_contain_spacer(self) -> None:
        """Test examples include spacer example."""
        assert "spacer" in CADQUERY_EXAMPLES

    def test_all_examples_have_proper_imports(self) -> None:
        """Test all examples have proper CadQuery imports."""
        for name, code in CADQUERY_EXAMPLES.items():
            assert "from cadquery import Workplane as cq" in code, f"{name} missing import"

    def test_all_examples_return_result(self) -> None:
        """Test all examples call result.val()."""
        for name, code in CADQUERY_EXAMPLES.items():
            assert "result.val()" in code, f"{name} missing result.val()"


# ============================================================================
# SECTION 3: Error Recovery Templates Tests
# ============================================================================


class TestErrorRecoveryTemplates:
    """Test error recovery templates."""

    def test_templates_dict_exists(self) -> None:
        """Test ERROR_RECOVERY_TEMPLATES is defined."""
        assert ERROR_RECOVERY_TEMPLATES is not None
        assert isinstance(ERROR_RECOVERY_TEMPLATES, dict)

    def test_templates_cover_all_error_categories(self) -> None:
        """Test templates cover all ErrorCategory values."""
        expected_categories = [
            ErrorCategory.INVALID_PARAMS,
            ErrorCategory.TOPOLOGY_ERROR,
            ErrorCategory.BOOLEAN_FAILURE,
            ErrorCategory.SELF_INTERSECTION,
            ErrorCategory.DEGENERATE_SHAPE,
            ErrorCategory.TOLERANCE_VIOLATION,
        ]
        for category in expected_categories:
            assert category in ERROR_RECOVERY_TEMPLATES, f"Missing template for {category}"

    def test_invalid_params_template(self) -> None:
        """Test INVALID_PARAMS template content."""
        template = ERROR_RECOVERY_TEMPLATES[ErrorCategory.INVALID_PARAMS]
        assert "invalid parameters" in template.lower()
        assert "numeric values" in template.lower() or "dimensions" in template.lower()

    def test_topology_error_template(self) -> None:
        """Test TOPOLOGY_ERROR template content."""
        template = ERROR_RECOVERY_TEMPLATES[ErrorCategory.TOPOLOGY_ERROR]
        assert "topolog" in template.lower()
        assert "sketch" in template.lower() or "close" in template.lower()

    def test_boolean_failure_template(self) -> None:
        """Test BOOLEAN_FAILURE template content."""
        template = ERROR_RECOVERY_TEMPLATES[ErrorCategory.BOOLEAN_FAILURE]
        assert "boolean" in template.lower()
        assert "overlap" in template.lower() or "intersect" in template.lower()

    def test_self_intersection_template(self) -> None:
        """Test SELF_INTERSECTION template content."""
        template = ERROR_RECOVERY_TEMPLATES[ErrorCategory.SELF_INTERSECTION]
        assert "self-intersect" in template.lower()

    def test_degenerate_shape_template(self) -> None:
        """Test DEGENERATE_SHAPE template content."""
        template = ERROR_RECOVERY_TEMPLATES[ErrorCategory.DEGENERATE_SHAPE]
        assert "degenerate" in template.lower() or "zero" in template.lower()

    def test_tolerance_violation_template(self) -> None:
        """Test TOLERANCE_VIOLATION template content."""
        template = ERROR_RECOVERY_TEMPLATES[ErrorCategory.TOLERANCE_VIOLATION]
        assert "tolerance" in template.lower()


# ============================================================================
# SECTION 4: get_system_prompt Tests
# ============================================================================


class TestGetSystemPrompt:
    """Test get_system_prompt function."""

    def test_cadquery_default_prompt(self) -> None:
        """Test default CadQuery system prompt."""
        prompt = get_system_prompt()
        assert "CadQuery" in prompt
        assert "cadquery" in prompt.lower()
        assert "from cadquery import Workplane as cq" in prompt

    def test_cadquery_backend_explicit(self) -> None:
        """Test explicit CadQuery backend."""
        prompt = get_system_prompt(backend="cadquery")
        assert "CadQuery" in prompt

    def test_openscad_backend(self) -> None:
        """Test OpenSCAD backend."""
        prompt = get_system_prompt(backend="openscad")
        assert "OpenSCAD" in prompt
        assert "cube" in prompt or "sphere" in prompt

    def test_unknown_backend_raises_error(self) -> None:
        """Test unknown backend raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_system_prompt(backend="unknown")
        assert "Unknown backend" in str(exc_info.value)

    def test_include_examples_true(self) -> None:
        """Test system prompt includes examples when requested."""
        prompt = get_system_prompt(backend="cadquery", include_examples=True)
        assert "EXAMPLES" in prompt.upper() or "bracket" in prompt.lower()

    def test_include_examples_false(self) -> None:
        """Test system prompt excludes examples when not requested."""
        prompt_with = get_system_prompt(backend="cadquery", include_examples=True)
        prompt_without = get_system_prompt(backend="cadquery", include_examples=False)
        # Prompt without examples should be shorter
        assert len(prompt_without) < len(prompt_with)

    def test_error_context_adds_recovery_guidance(self) -> None:
        """Test error context adds recovery guidance."""
        error_context = {
            "category": ErrorCategory.TOPOLOGY_ERROR,
            "message": "Sketch not closed",
        }
        prompt = get_system_prompt(backend="cadquery", error_context=error_context)
        assert "ERROR RECOVERY" in prompt.upper() or "topolog" in prompt.lower()


# ============================================================================
# SECTION 5: get_repair_prompt Tests
# ============================================================================


class TestGetRepairPrompt:
    """Test get_repair_prompt function."""

    def test_repair_prompt_contains_code(self) -> None:
        """Test repair prompt contains the failed code."""
        code = "result = cq('XY').box(100, 50, 20)"
        error_message = "Syntax error on line 1"
        prompt = get_repair_prompt(code, error_message)
        assert code in prompt

    def test_repair_prompt_contains_error(self) -> None:
        """Test repair prompt contains the error message."""
        code = "result = cq('XY').box(100, 50, 20)"
        error_message = "BRepCheck_NotClosed: Shell is not watertight"
        prompt = get_repair_prompt(code, error_message)
        assert "BRepCheck_NotClosed" in prompt

    def test_repair_prompt_contains_suggestion(self) -> None:
        """Test repair prompt contains suggestion when provided."""
        code = "result = cq('XY').box(100, 50, 20)"
        error_message = "Failed"
        suggestion = "Close the sketch loop before extruding"
        prompt = get_repair_prompt(code, error_message, suggestion)
        assert "Close the sketch" in prompt

    def test_repair_prompt_without_suggestion(self) -> None:
        """Test repair prompt works without suggestion."""
        code = "result = cq('XY').box(100, 50, 20)"
        error_message = "Failed"
        prompt = get_repair_prompt(code, error_message)
        assert "FAILED CODE" in prompt.upper()
        assert "ERROR" in prompt.upper()

    def test_repair_prompt_has_code_block(self) -> None:
        """Test repair prompt wraps code in code block."""
        code = "result = cq('XY').box(100, 50, 20)"
        error_message = "Failed"
        prompt = get_repair_prompt(code, error_message)
        assert "```" in prompt

    def test_repair_prompt_asks_for_regeneration(self) -> None:
        """Test repair prompt asks for code regeneration."""
        code = "result = cq('XY').box(100, 50, 20)"
        error_message = "Failed"
        prompt = get_repair_prompt(code, error_message)
        assert "regenerate" in prompt.lower() or "fix" in prompt.lower()


# ============================================================================
# SECTION 6: OpenSCAD System Prompt Tests
# ============================================================================


class TestOpenSCADSystemPrompt:
    """Test OpenSCAD-specific system prompt."""

    def test_openscad_contains_instructions(self) -> None:
        """Test OpenSCAD prompt contains instructions."""
        prompt = get_system_prompt(backend="openscad")
        assert "INSTRUCTIONS" in prompt.upper()

    def test_openscad_contains_primitives(self) -> None:
        """Test OpenSCAD prompt contains primitives."""
        prompt = get_system_prompt(backend="openscad")
        assert "cube" in prompt
        assert "sphere" in prompt
        assert "cylinder" in prompt

    def test_openscad_contains_transforms(self) -> None:
        """Test OpenSCAD prompt contains transforms."""
        prompt = get_system_prompt(backend="openscad")
        assert "translate" in prompt
        assert "rotate" in prompt

    def test_openscad_contains_booleans(self) -> None:
        """Test OpenSCAD prompt contains boolean operations."""
        prompt = get_system_prompt(backend="openscad")
        assert "union" in prompt
        assert "difference" in prompt
        assert "intersection" in prompt

    def test_openscad_examples_when_requested(self) -> None:
        """Test OpenSCAD prompt includes examples when requested."""
        prompt = get_system_prompt(backend="openscad", include_examples=True)
        # Should contain working examples
        assert "BRACKET" in prompt.upper() or "SPACER" in prompt.upper()


# ============================================================================
# SECTION 7: Module Import Tests
# ============================================================================


class TestModuleImport:
    """Test module import."""

    def test_module_importable(self) -> None:
        """Test that prompt_library module is importable."""
        from ll_gen.codegen import prompt_library
        assert hasattr(prompt_library, "get_system_prompt")
        assert hasattr(prompt_library, "get_repair_prompt")
        assert hasattr(prompt_library, "CADQUERY_API_REFERENCE")
        assert hasattr(prompt_library, "CADQUERY_EXAMPLES")
        assert hasattr(prompt_library, "ERROR_RECOVERY_TEMPLATES")

    def test_functions_are_callable(self) -> None:
        """Test that exported functions are callable."""
        assert callable(get_system_prompt)
        assert callable(get_repair_prompt)


# ============================================================================
# SECTION 8: Integration Tests
# ============================================================================


class TestPromptLibraryIntegration:
    """Test prompt library integration scenarios."""

    def test_full_workflow_cadquery(self) -> None:
        """Test full workflow: initial prompt, then repair prompt."""
        # Initial generation
        system_prompt = get_system_prompt(backend="cadquery", include_examples=True)
        assert "CadQuery" in system_prompt
        assert len(system_prompt) > 1000  # Should be substantial

        # Simulated failure
        failed_code = "result = cq('XY').box(100, 50, 20)\nresult.val()"
        error_message = "BRepCheck_NotClosed"
        suggestion = "Close the sketch"

        # Repair prompt
        repair_prompt = get_repair_prompt(failed_code, error_message, suggestion)
        assert failed_code in repair_prompt
        assert error_message in repair_prompt

        # Second attempt with error context
        error_context = {
            "category": ErrorCategory.TOPOLOGY_ERROR,
            "message": error_message,
        }
        system_prompt_2 = get_system_prompt(
            backend="cadquery",
            error_context=error_context,
        )
        assert "TOPOLOGY" in system_prompt_2.upper() or "topology" in system_prompt_2.lower()
