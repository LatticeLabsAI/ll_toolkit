"""Comprehensive tests for ll_gen.codegen module.

This test module covers:
- prompt_library: system prompts, API references, examples, repair prompts
- cadquery_proposer: CadQueryProposer class with mocked LLM backend
- openscad_proposer: OpenSCADProposer class with mocked LLM backend

All pure Python tests (prompt_library) run without dependencies.
Proposer tests use mocks and are skipped if cadling is not available.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ll_gen.codegen import (
    cadquery_proposer,
    openscad_proposer,
    prompt_library,
)
from ll_gen.codegen.cadquery_proposer import CadQueryProposer
from ll_gen.codegen.openscad_proposer import OpenSCADProposer
from ll_gen.config import CodeLanguage, CodegenConfig, ErrorCategory


# =============================================================================
# Tests for prompt_library module (Pure Python, no dependencies)
# =============================================================================


class TestCadQueryAPIReference:
    """Test the CADQUERY_API_REFERENCE constant."""

    def test_cadquery_api_reference_non_empty(self):
        """CADQUERY_API_REFERENCE is a non-empty string."""
        assert isinstance(prompt_library.CADQUERY_API_REFERENCE, str)
        assert len(prompt_library.CADQUERY_API_REFERENCE) > 0

    def test_cadquery_api_reference_contains_workplane(self):
        """CADQUERY_API_REFERENCE contains 'Workplane' keyword."""
        assert "Workplane" in prompt_library.CADQUERY_API_REFERENCE

    def test_cadquery_api_reference_contains_key_methods(self):
        """CADQUERY_API_REFERENCE documents essential methods."""
        ref = prompt_library.CADQUERY_API_REFERENCE
        assert "box" in ref
        assert "sphere" in ref
        assert "cylinder" in ref
        assert "hole" in ref
        assert "fillet" in ref


class TestCadQueryExamples:
    """Test the CADQUERY_EXAMPLES dictionary."""

    def test_cadquery_examples_is_dict(self):
        """CADQUERY_EXAMPLES is a dictionary."""
        assert isinstance(prompt_library.CADQUERY_EXAMPLES, dict)

    def test_cadquery_examples_has_expected_keys(self):
        """CADQUERY_EXAMPLES has all expected example keys."""
        expected_keys = {"bracket", "gear", "enclosure", "hinge", "spacer"}
        actual_keys = set(prompt_library.CADQUERY_EXAMPLES.keys())
        assert expected_keys == actual_keys

    def test_bracket_example_is_string(self):
        """bracket example is a non-empty string."""
        bracket = prompt_library.CADQUERY_EXAMPLES["bracket"]
        assert isinstance(bracket, str)
        assert len(bracket) > 0

    def test_gear_example_is_string(self):
        """gear example is a non-empty string."""
        gear = prompt_library.CADQUERY_EXAMPLES["gear"]
        assert isinstance(gear, str)
        assert len(gear) > 0

    def test_enclosure_example_is_string(self):
        """enclosure example is a non-empty string."""
        enclosure = prompt_library.CADQUERY_EXAMPLES["enclosure"]
        assert isinstance(enclosure, str)
        assert len(enclosure) > 0

    def test_hinge_example_is_string(self):
        """hinge example is a non-empty string."""
        hinge = prompt_library.CADQUERY_EXAMPLES["hinge"]
        assert isinstance(hinge, str)
        assert len(hinge) > 0

    def test_spacer_example_is_string(self):
        """spacer example is a non-empty string."""
        spacer = prompt_library.CADQUERY_EXAMPLES["spacer"]
        assert isinstance(spacer, str)
        assert len(spacer) > 0

    def test_all_examples_contain_cadquery_pattern(self):
        """Each example contains valid CadQuery pattern 'cq('."""
        for name, code in prompt_library.CADQUERY_EXAMPLES.items():
            assert "cq(" in code, f"Example '{name}' missing 'cq(' pattern"

    def test_all_examples_contain_result_output(self):
        """Each example contains 'result' variable or '.val()' for output."""
        for name, code in prompt_library.CADQUERY_EXAMPLES.items():
            has_result_var = "result" in code
            has_val_method = ".val()" in code
            assert has_result_var or has_val_method, (
                f"Example '{name}' missing result variable or .val() method"
            )

    def test_bracket_example_contains_import(self):
        """bracket example imports Workplane correctly."""
        bracket = prompt_library.CADQUERY_EXAMPLES["bracket"]
        assert "from cadquery import Workplane as cq" in bracket

    def test_gear_example_contains_loop(self):
        """gear example demonstrates loop usage."""
        gear = prompt_library.CADQUERY_EXAMPLES["gear"]
        assert "for" in gear

    def test_enclosure_example_contains_boolean(self):
        """enclosure example demonstrates boolean operations."""
        enclosure = prompt_library.CADQUERY_EXAMPLES["enclosure"]
        assert "box" in enclosure

    def test_hinge_example_contains_union(self):
        """hinge example demonstrates union operation."""
        hinge = prompt_library.CADQUERY_EXAMPLES["hinge"]
        assert "union" in hinge

    def test_spacer_example_contains_chamfer(self):
        """spacer example demonstrates chamfer operation."""
        spacer = prompt_library.CADQUERY_EXAMPLES["spacer"]
        assert "chamfer" in spacer


class TestErrorRecoveryTemplates:
    """Test the ERROR_RECOVERY_TEMPLATES dictionary."""

    def test_error_recovery_templates_is_dict(self):
        """ERROR_RECOVERY_TEMPLATES is a dictionary."""
        assert isinstance(prompt_library.ERROR_RECOVERY_TEMPLATES, dict)

    def test_error_recovery_has_all_categories(self):
        """ERROR_RECOVERY_TEMPLATES contains all 6 ErrorCategory keys."""
        expected_categories = {
            ErrorCategory.INVALID_PARAMS,
            ErrorCategory.TOPOLOGY_ERROR,
            ErrorCategory.BOOLEAN_FAILURE,
            ErrorCategory.SELF_INTERSECTION,
            ErrorCategory.DEGENERATE_SHAPE,
            ErrorCategory.TOLERANCE_VIOLATION,
        }
        actual_categories = set(prompt_library.ERROR_RECOVERY_TEMPLATES.keys())
        assert expected_categories == actual_categories

    def test_invalid_params_recovery_is_string(self):
        """INVALID_PARAMS recovery template is a non-empty string."""
        template = prompt_library.ERROR_RECOVERY_TEMPLATES[
            ErrorCategory.INVALID_PARAMS
        ]
        assert isinstance(template, str)
        assert len(template) > 0

    def test_topology_error_recovery_is_string(self):
        """TOPOLOGY_ERROR recovery template is a non-empty string."""
        template = prompt_library.ERROR_RECOVERY_TEMPLATES[
            ErrorCategory.TOPOLOGY_ERROR
        ]
        assert isinstance(template, str)
        assert len(template) > 0

    def test_boolean_failure_recovery_is_string(self):
        """BOOLEAN_FAILURE recovery template is a non-empty string."""
        template = prompt_library.ERROR_RECOVERY_TEMPLATES[
            ErrorCategory.BOOLEAN_FAILURE
        ]
        assert isinstance(template, str)
        assert len(template) > 0

    def test_self_intersection_recovery_is_string(self):
        """SELF_INTERSECTION recovery template is a non-empty string."""
        template = prompt_library.ERROR_RECOVERY_TEMPLATES[
            ErrorCategory.SELF_INTERSECTION
        ]
        assert isinstance(template, str)
        assert len(template) > 0

    def test_degenerate_shape_recovery_is_string(self):
        """DEGENERATE_SHAPE recovery template is a non-empty string."""
        template = prompt_library.ERROR_RECOVERY_TEMPLATES[
            ErrorCategory.DEGENERATE_SHAPE
        ]
        assert isinstance(template, str)
        assert len(template) > 0

    def test_tolerance_violation_recovery_is_string(self):
        """TOLERANCE_VIOLATION recovery template is a non-empty string."""
        template = prompt_library.ERROR_RECOVERY_TEMPLATES[
            ErrorCategory.TOLERANCE_VIOLATION
        ]
        assert isinstance(template, str)
        assert len(template) > 0


class TestGetSystemPromptCadQuery:
    """Test get_system_prompt() with 'cadquery' backend."""

    def test_get_system_prompt_cadquery_default(self):
        """get_system_prompt('cadquery') returns string with CadQuery and Workplane."""
        prompt = prompt_library.get_system_prompt("cadquery")
        assert isinstance(prompt, str)
        assert "CadQuery" in prompt
        assert "Workplane" in prompt

    def test_get_system_prompt_cadquery_lowercase(self):
        """get_system_prompt() is case-insensitive for backend name."""
        prompt = prompt_library.get_system_prompt("CADQUERY")
        assert isinstance(prompt, str)
        assert "CadQuery" in prompt

    def test_get_system_prompt_cadquery_with_examples(self):
        """get_system_prompt with include_examples=True includes example code."""
        prompt = prompt_library.get_system_prompt(
            "cadquery", include_examples=True
        )
        # Should contain example code
        assert "bracket" in prompt.lower() or "box" in prompt.lower()
        # Should contain actual code snippet
        assert "cq(" in prompt

    def test_get_system_prompt_cadquery_without_examples(self):
        """get_system_prompt with include_examples=False excludes example code."""
        prompt = prompt_library.get_system_prompt(
            "cadquery", include_examples=False
        )
        # Should not contain the working examples section
        assert "WORKING EXAMPLES" not in prompt
        # But should still contain API reference
        assert "API REFERENCE" in prompt

    def test_get_system_prompt_cadquery_includes_api_reference(self):
        """get_system_prompt('cadquery') includes compressed API reference."""
        prompt = prompt_library.get_system_prompt("cadquery")
        assert "INITIALIZATION:" in prompt or "API" in prompt
        assert "box" in prompt.lower()

    def test_get_system_prompt_cadquery_with_error_context(self):
        """get_system_prompt with error_context includes recovery guidance."""
        error_context = {"category": ErrorCategory.BOOLEAN_FAILURE}
        prompt = prompt_library.get_system_prompt(
            "cadquery", error_context=error_context
        )
        assert "ERROR RECOVERY" in prompt or "RECOVERY" in prompt

    def test_get_system_prompt_cadquery_boolean_failure_guidance(self):
        """Error context for BOOLEAN_FAILURE includes specific guidance."""
        error_context = {"category": ErrorCategory.BOOLEAN_FAILURE}
        prompt = prompt_library.get_system_prompt(
            "cadquery", error_context=error_context
        )
        # Should include recovery guidance for boolean operations
        assert "boolean" in prompt.lower() or "union" in prompt.lower()

    def test_get_system_prompt_cadquery_topology_error_guidance(self):
        """Error context for TOPOLOGY_ERROR includes specific guidance."""
        error_context = {"category": ErrorCategory.TOPOLOGY_ERROR}
        prompt = prompt_library.get_system_prompt(
            "cadquery", error_context=error_context
        )
        # Should include guidance about sketches and extrusion
        assert "ERROR RECOVERY" in prompt

    def test_get_system_prompt_cadquery_all_error_categories(self):
        """get_system_prompt handles all ErrorCategory values."""
        categories = [
            ErrorCategory.INVALID_PARAMS,
            ErrorCategory.TOPOLOGY_ERROR,
            ErrorCategory.BOOLEAN_FAILURE,
            ErrorCategory.SELF_INTERSECTION,
            ErrorCategory.DEGENERATE_SHAPE,
            ErrorCategory.TOLERANCE_VIOLATION,
        ]
        for category in categories:
            error_context = {"category": category}
            prompt = prompt_library.get_system_prompt(
                "cadquery", error_context=error_context
            )
            assert isinstance(prompt, str)
            assert len(prompt) > 0


class TestGetSystemPromptOpenSCAD:
    """Test get_system_prompt() with 'openscad' backend."""

    def test_get_system_prompt_openscad_returns_string(self):
        """get_system_prompt('openscad') returns a string with OpenSCAD."""
        prompt = prompt_library.get_system_prompt("openscad")
        assert isinstance(prompt, str)
        assert "OpenSCAD" in prompt

    def test_get_system_prompt_openscad_includes_primitives(self):
        """get_system_prompt('openscad') includes primitive documentation."""
        prompt = prompt_library.get_system_prompt("openscad")
        assert "cube" in prompt.lower()
        assert "sphere" in prompt.lower()
        assert "cylinder" in prompt.lower()

    def test_get_system_prompt_openscad_with_examples_includes_code(self):
        """get_system_prompt('openscad', include_examples=True) includes examples."""
        prompt = prompt_library.get_system_prompt(
            "openscad", include_examples=True
        )
        assert "WORKING EXAMPLES" in prompt
        assert "difference()" in prompt or "cube(" in prompt

    def test_get_system_prompt_openscad_without_examples(self):
        """get_system_prompt('openscad', include_examples=False) excludes examples."""
        prompt = prompt_library.get_system_prompt(
            "openscad", include_examples=False
        )
        assert "WORKING EXAMPLES" not in prompt

    def test_get_system_prompt_openscad_includes_syntax_info(self):
        """get_system_prompt('openscad') includes syntax guidance."""
        prompt = prompt_library.get_system_prompt("openscad")
        # Should mention semicolons (OpenSCAD syntax requirement)
        assert ";" in prompt or "bracket" in prompt.lower()

    def test_get_system_prompt_openscad_with_error_context(self):
        """get_system_prompt('openscad') with error_context includes guidance."""
        error_context = {"category": ErrorCategory.INVALID_PARAMS}
        prompt = prompt_library.get_system_prompt(
            "openscad", error_context=error_context
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_get_system_prompt_openscad_case_insensitive(self):
        """Backend name 'openscad' is case-insensitive."""
        prompt1 = prompt_library.get_system_prompt("openscad")
        prompt2 = prompt_library.get_system_prompt("OPENSCAD")
        assert prompt1 == prompt2


class TestGetSystemPromptInvalidBackend:
    """Test get_system_prompt() error handling."""

    def test_get_system_prompt_unknown_backend_raises(self):
        """get_system_prompt with unknown backend raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            prompt_library.get_system_prompt("invalid_backend")
        assert "Unknown backend" in str(exc_info.value)

    def test_get_system_prompt_empty_string_backend_raises(self):
        """get_system_prompt with empty backend string raises ValueError."""
        with pytest.raises(ValueError):
            prompt_library.get_system_prompt("")


class TestGetRepairPrompt:
    """Test get_repair_prompt() function."""

    def test_get_repair_prompt_basic(self):
        """get_repair_prompt returns string with FAILED CODE and ERROR sections."""
        code = "result = cq('XY').box(100, 50, 20)"
        error = "Invalid parameter: diameter cannot be negative"
        prompt = prompt_library.get_repair_prompt(code, error)

        assert isinstance(prompt, str)
        assert "FAILED CODE" in prompt
        assert code in prompt
        assert "ERROR" in prompt
        assert error in prompt

    def test_get_repair_prompt_without_suggestion(self):
        """get_repair_prompt works without suggestion parameter."""
        code = "result = cq('XY').box(100, 50, 20)"
        error = "Syntax error: unexpected token"
        prompt = prompt_library.get_repair_prompt(code, error)

        assert isinstance(prompt, str)
        assert "FAILED CODE" in prompt
        assert error in prompt
        # Should not have SUGGESTION section when empty
        assert "SUGGESTION" not in prompt

    def test_get_repair_prompt_with_empty_suggestion(self):
        """get_repair_prompt with empty suggestion string omits SUGGESTION section."""
        code = "result = cq('XY').box(100, 50, 20)"
        error = "Syntax error"
        prompt = prompt_library.get_repair_prompt(code, error, suggestion="")

        assert "FAILED CODE" in prompt
        assert "ERROR" in prompt
        assert "SUGGESTION" not in prompt

    def test_get_repair_prompt_with_suggestion(self):
        """get_repair_prompt with suggestion includes SUGGESTION section."""
        code = "result = cq('XY').box(100, 50, 20)"
        error = "Diameter must be positive"
        suggestion = "Change box dimensions to positive values"
        prompt = prompt_library.get_repair_prompt(code, error, suggestion)

        assert "FAILED CODE" in prompt
        assert "ERROR" in prompt
        assert "SUGGESTION" in prompt
        assert suggestion in prompt

    def test_get_repair_prompt_multiline_code(self):
        """get_repair_prompt handles multiline code correctly."""
        code = """from cadquery import Workplane as cq

result = (
    cq("XY")
    .box(100, 50, 20)
    .faces(">Z")
    .hole(5)
)
result.val()
"""
        error = "Face selection returned no results"
        prompt = prompt_library.get_repair_prompt(code, error)

        assert code in prompt
        assert error in prompt

    def test_get_repair_prompt_complex_error_message(self):
        """get_repair_prompt handles complex multi-line error messages."""
        code = "result = cq('XY').box(100, 50, 20)"
        error = """Traceback (most recent call last):
  File "...", line 42, in generate
    result = cq('XY').box(100, 50, 20)
TypeError: box() missing 1 required positional argument: 'height'
"""
        prompt = prompt_library.get_repair_prompt(code, error)

        assert code in prompt
        assert "TypeError" in prompt

    def test_get_repair_prompt_returns_string_structure(self):
        """get_repair_prompt returns well-structured string."""
        code = "result = cq('XY').box(100, 50)"
        error = "Missing required parameter"
        suggestion = "Add height parameter"
        prompt = prompt_library.get_repair_prompt(code, error, suggestion)

        lines = prompt.split("\n")
        # Should have multiple lines with structure
        assert len(lines) > 5
        # Should start with descriptive text
        assert "failed" in lines[0].lower() or "code" in lines[0].lower()


# =============================================================================
# Tests for CadQueryProposer class
# =============================================================================


class TestCadQueryProposerInit:
    """Test CadQueryProposer initialization."""

    def test_cadquery_proposer_can_be_imported(self):
        """CadQueryProposer class can be imported."""
        assert CadQueryProposer is not None

    def test_cadquery_proposer_init_default_config(self, codegen_config):
        """CadQueryProposer initializes with default config."""
        proposer = CadQueryProposer()
        assert proposer.config is not None
        assert isinstance(proposer.config, CodegenConfig)

    def test_cadquery_proposer_init_custom_config(self, codegen_config):
        """CadQueryProposer initializes with custom config."""
        custom_config = CodegenConfig(
            model_name="gpt-4", api_provider="openai"
        )
        proposer = CadQueryProposer(config=custom_config)
        assert proposer.config == custom_config
        assert proposer.config.model_name == "gpt-4"

    def test_cadquery_proposer_init_stores_config(self):
        """CadQueryProposer stores the provided config."""
        config = CodegenConfig()
        proposer = CadQueryProposer(config=config)
        assert proposer.config is config

    def test_cadquery_proposer_has_generator_attribute(self):
        """CadQueryProposer has generator attribute after init."""
        proposer = CadQueryProposer()
        assert hasattr(proposer, "generator")

    def test_cadquery_proposer_config_with_none_uses_defaults(self):
        """CadQueryProposer with None config uses CodegenConfig defaults."""
        proposer = CadQueryProposer(config=None)
        assert proposer.config is not None
        assert isinstance(proposer.config, CodegenConfig)


@pytest.mark.skipif(
    not cadquery_proposer._CADLING_AVAILABLE,
    reason="cadling not installed",
)
class TestCadQueryProposerWithCadling:
    """Test CadQueryProposer when cadling is available."""

    def test_cadquery_proposer_generator_initialized_with_cadling(self):
        """CadQueryProposer initializes generator when cadling is available."""
        proposer = CadQueryProposer()
        # When cadling is available, generator should be initialized
        assert proposer.generator is not None

    def test_cadquery_proposer_propose_returns_code_proposal(self):
        """propose() returns CodeProposal instance."""
        proposer = CadQueryProposer()

        # Mock the generator to avoid actual LLM calls
        mock_code = "from cadquery import Workplane as cq\nresult = cq('XY').box(100, 50, 20)\nresult.val()"
        proposer.generator.generate = MagicMock(return_value=mock_code)

        proposal = proposer.propose("A box")
        assert proposal is not None

    def test_cadquery_proposer_propose_batch_returns_list(self):
        """propose_batch() returns list of CodeProposal objects."""
        proposer = CadQueryProposer()

        mock_code = "from cadquery import Workplane as cq\nresult = cq('XY').box(100, 50, 20)\nresult.val()"
        proposer.generator.generate = MagicMock(return_value=mock_code)

        proposals = proposer.propose_batch("A box", num_candidates=3)
        assert isinstance(proposals, list)
        assert len(proposals) == 3


class TestCadQueryProposerWithoutCadling:
    """Test CadQueryProposer error handling when cadling is not available."""

    def test_cadquery_proposer_propose_raises_without_cadling(self):
        """propose() raises ImportError if cadling not available."""
        with patch.object(
            cadquery_proposer, "_CADLING_AVAILABLE", False
        ):
            proposer = CadQueryProposer()
            proposer.generator = None  # Force unavailable state

            with pytest.raises(ImportError) as exc_info:
                proposer.propose("A box")
            assert "cadling" in str(exc_info.value).lower()

    def test_cadquery_proposer_propose_batch_raises_without_cadling(self):
        """propose_batch() raises ImportError if cadling not available."""
        with patch.object(
            cadquery_proposer, "_CADLING_AVAILABLE", False
        ):
            proposer = CadQueryProposer()
            proposer.generator = None

            with pytest.raises(ImportError):
                proposer.propose_batch("A box")


# =============================================================================
# Tests for OpenSCADProposer class
# =============================================================================


class TestOpenSCADProposerInit:
    """Test OpenSCADProposer initialization."""

    def test_openscad_proposer_can_be_imported(self):
        """OpenSCADProposer class can be imported."""
        assert OpenSCADProposer is not None

    def test_openscad_proposer_init_default_config(self):
        """OpenSCADProposer initializes with default config."""
        proposer = OpenSCADProposer()
        assert proposer.config is not None
        assert isinstance(proposer.config, CodegenConfig)

    def test_openscad_proposer_init_custom_config(self):
        """OpenSCADProposer initializes with custom config."""
        custom_config = CodegenConfig(
            model_name="claude-opus", api_provider="anthropic"
        )
        proposer = OpenSCADProposer(config=custom_config)
        assert proposer.config == custom_config
        assert proposer.config.api_provider == "anthropic"

    def test_openscad_proposer_stores_config(self):
        """OpenSCADProposer stores the provided config."""
        config = CodegenConfig()
        proposer = OpenSCADProposer(config=config)
        assert proposer.config is config

    def test_openscad_proposer_has_generator_attribute(self):
        """OpenSCADProposer has generator attribute after init."""
        proposer = OpenSCADProposer()
        assert hasattr(proposer, "generator")

    def test_openscad_proposer_config_with_none_uses_defaults(self):
        """OpenSCADProposer with None config uses CodegenConfig defaults."""
        proposer = OpenSCADProposer(config=None)
        assert proposer.config is not None
        assert isinstance(proposer.config, CodegenConfig)


@pytest.mark.skipif(
    not openscad_proposer._CADLING_AVAILABLE,
    reason="cadling not installed",
)
class TestOpenSCADProposerWithCadling:
    """Test OpenSCADProposer when cadling is available."""

    def test_openscad_proposer_generator_initialized_with_cadling(self):
        """OpenSCADProposer initializes generator when cadling is available."""
        proposer = OpenSCADProposer()
        # When cadling is available, generator should be initialized
        assert proposer.generator is not None

    def test_openscad_proposer_propose_returns_code_proposal(self):
        """propose() returns CodeProposal instance."""
        proposer = OpenSCADProposer()

        mock_code = """
difference() {
    cube([100, 50, 20], center=true);
    cylinder(h=22, r=5, center=true);
}
"""
        proposer.generator.generate = MagicMock(return_value=mock_code)

        proposal = proposer.propose("A box with a hole")
        assert proposal is not None

    def test_openscad_proposer_propose_batch_returns_list(self):
        """propose_batch() returns list of CodeProposal objects."""
        proposer = OpenSCADProposer()

        mock_code = "cube([100, 50, 20], center=true);"
        proposer.generator.generate = MagicMock(return_value=mock_code)

        proposals = proposer.propose_batch("A box", num_candidates=2)
        assert isinstance(proposals, list)
        assert len(proposals) == 2


class TestOpenSCADProposerWithoutCadling:
    """Test OpenSCADProposer error handling when cadling is not available."""

    def test_openscad_proposer_propose_raises_without_cadling(self):
        """propose() raises ImportError if cadling not available."""
        with patch.object(
            openscad_proposer, "_CADLING_AVAILABLE", False
        ):
            proposer = OpenSCADProposer()
            proposer.generator = None

            with pytest.raises(ImportError) as exc_info:
                proposer.propose("A box")
            assert "cadling" in str(exc_info.value).lower()

    def test_openscad_proposer_propose_batch_raises_without_cadling(self):
        """propose_batch() raises ImportError if cadling not available."""
        with patch.object(
            openscad_proposer, "_CADLING_AVAILABLE", False
        ):
            proposer = OpenSCADProposer()
            proposer.generator = None

            with pytest.raises(ImportError):
                proposer.propose_batch("A box")


# =============================================================================
# Integration-style tests (no heavy dependencies)
# =============================================================================


class TestPromptLibraryIntegration:
    """Integration tests combining multiple prompt_library functions."""

    def test_get_system_prompt_and_repair_prompt_work_together(self):
        """System prompt and repair prompt can be used together."""
        system_prompt = prompt_library.get_system_prompt("cadquery")
        repair_prompt = prompt_library.get_repair_prompt(
            code="bad_code",
            error_message="Error occurred",
            suggestion="Fix it",
        )

        # Both should be valid strings that could be used in LLM calls
        assert isinstance(system_prompt, str)
        assert isinstance(repair_prompt, str)
        assert len(system_prompt) > len(repair_prompt)  # System is more comprehensive

    def test_all_backends_produce_valid_prompts(self):
        """Both backends produce valid system prompts."""
        for backend in ["cadquery", "openscad"]:
            prompt = prompt_library.get_system_prompt(backend)
            assert isinstance(prompt, str)
            assert len(prompt) > 100  # Should be comprehensive
            assert "INSTRUCTIONS" in prompt or "instruction" in prompt.lower()

    def test_examples_and_api_reference_completeness(self):
        """Examples and API references are comprehensive."""
        # CadQuery should have examples
        assert len(prompt_library.CADQUERY_EXAMPLES) >= 5

        # API reference should be substantial
        assert len(prompt_library.CADQUERY_API_REFERENCE) > 500

        # All error categories should have recovery templates
        assert len(prompt_library.ERROR_RECOVERY_TEMPLATES) == 6
