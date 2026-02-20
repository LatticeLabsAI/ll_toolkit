"""Comprehensive test suite for ll_gen proposals module.

Tests all proposal types and related classes:
- BaseProposal: retry logic, attempt tracking, error context
- CodeProposal: syntax validation, import extraction, retry behavior
- CommandSequenceProposal: command properties, quantization, dequantization
- LatentProposal: geometry validation, bounding box computation, face areas
- DisposalResult: validation results, repair tracking, serialization
- GeometryReport: dimension matching, bounding box introspection
- ValidationFinding and RepairAction: construction and serialization
"""
from __future__ import annotations

import copy
import hashlib
import uuid
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from ll_gen.config import CodeLanguage, ErrorCategory, ErrorSeverity
from ll_gen.proposals.base import BaseProposal
from ll_gen.proposals.code_proposal import CodeProposal
from ll_gen.proposals.command_proposal import CommandSequenceProposal
from ll_gen.proposals.disposal_result import (
    DisposalResult,
    GeometryReport,
    RepairAction,
    ValidationFinding,
)
from ll_gen.proposals.latent_proposal import LatentProposal


# =============================================================================
# BaseProposal Tests
# =============================================================================

class TestBaseProposalConstruction:
    """Test BaseProposal initialization and defaults."""

    def test_default_construction(self):
        """BaseProposal with minimal args has sensible defaults."""
        prop = BaseProposal()
        assert prop.confidence == 0.0
        assert prop.attempt == 1
        assert prop.max_attempts == 3
        assert prop.source_prompt == ""
        assert prop.conditioning_source is None
        assert prop.error_context is None
        assert len(prop.alternatives) == 0
        assert isinstance(prop.proposal_id, str)
        assert len(prop.proposal_id) > 0

    def test_custom_construction(self, base_proposal):
        """BaseProposal accepts custom values for all fields."""
        assert base_proposal.proposal_id == "test_base_0001"
        assert base_proposal.confidence == 0.75
        assert base_proposal.attempt == 1
        assert base_proposal.max_attempts == 3
        assert base_proposal.source_prompt == "A simple test shape"
        assert base_proposal.conditioning_source == "text"

    def test_unique_proposal_ids(self):
        """Default-constructed proposals have unique IDs."""
        p1 = BaseProposal()
        p2 = BaseProposal()
        assert p1.proposal_id != p2.proposal_id

    def test_timestamp_is_set(self, base_proposal):
        """Timestamp is populated at construction."""
        assert base_proposal.timestamp is not None
        assert "T" in base_proposal.timestamp  # ISO format check
        assert "Z" in base_proposal.timestamp or "+" in base_proposal.timestamp


class TestBaseProposalRetryLogic:
    """Test retry-related methods and properties."""

    def test_should_retry_first_attempt(self, base_proposal):
        """should_retry() returns True when attempt < max_attempts."""
        # base_proposal is attempt 1 of 3
        assert base_proposal.should_retry() is True

    def test_should_retry_last_attempt(self):
        """should_retry() returns False when attempt == max_attempts."""
        prop = BaseProposal(attempt=3, max_attempts=3)
        assert prop.should_retry() is False

    def test_should_retry_exact_boundary(self):
        """should_retry() boundary: attempt=2, max=3."""
        prop = BaseProposal(attempt=2, max_attempts=3)
        assert prop.should_retry() is True

    def test_next_attempt_increments(self):
        """next_attempt() returns attempt + 1."""
        prop = BaseProposal(attempt=1, max_attempts=3)
        assert prop.next_attempt() == 2

    def test_next_attempt_clamps_at_max(self):
        """next_attempt() doesn't exceed max_attempts."""
        prop = BaseProposal(attempt=3, max_attempts=3)
        assert prop.next_attempt() == 3

    def test_is_first_attempt_true(self):
        """is_first_attempt property returns True for attempt=1."""
        prop = BaseProposal(attempt=1)
        assert prop.is_first_attempt is True

    def test_is_first_attempt_false(self):
        """is_first_attempt property returns False for attempt > 1."""
        prop = BaseProposal(attempt=2)
        assert prop.is_first_attempt is False

    def test_is_retry_false_first_attempt(self):
        """is_retry property returns False for attempt=1."""
        prop = BaseProposal(attempt=1)
        assert prop.is_retry is False

    def test_is_retry_true_second_attempt(self):
        """is_retry property returns True for attempt=2."""
        prop = BaseProposal(attempt=2)
        assert prop.is_retry is True

    def test_is_retry_true_third_attempt(self):
        """is_retry property returns True for attempt=3."""
        prop = BaseProposal(attempt=3)
        assert prop.is_retry is True


class TestBaseProposalErrorContext:
    """Test error context and retry proposal creation."""

    def test_with_error_context_creates_copy(self, base_proposal):
        """with_error_context() returns a new instance."""
        error = {"error_code": "TEST_ERROR", "message": "Test"}
        new_prop = base_proposal.with_error_context(error)
        assert new_prop is not base_proposal

    def test_with_error_context_increments_attempt(self, base_proposal):
        """with_error_context() increments attempt to next_attempt()."""
        error = {"error_code": "TEST_ERROR"}
        new_prop = base_proposal.with_error_context(error)
        assert new_prop.attempt == 2

    def test_with_error_context_new_proposal_id(self, base_proposal):
        """with_error_context() generates a new proposal_id."""
        error = {"error_code": "TEST_ERROR"}
        new_prop = base_proposal.with_error_context(error)
        assert new_prop.proposal_id != base_proposal.proposal_id

    def test_with_error_context_stores_error(self, base_proposal):
        """with_error_context() stores the error dict."""
        error = {"error_code": "TEST_ERROR", "details": "Some problem"}
        new_prop = base_proposal.with_error_context(error)
        assert new_prop.error_context == error

    def test_with_error_context_preserves_prompt(self, base_proposal):
        """with_error_context() preserves source_prompt."""
        error = {"error_code": "TEST_ERROR"}
        new_prop = base_proposal.with_error_context(error)
        assert new_prop.source_prompt == base_proposal.source_prompt

    def test_with_error_context_updates_timestamp(self, base_proposal):
        """with_error_context() creates new timestamp."""
        import time
        error = {"error_code": "TEST_ERROR"}
        time.sleep(0.01)  # Ensure time difference
        new_prop = base_proposal.with_error_context(error)
        assert new_prop.timestamp != base_proposal.timestamp

    def test_with_error_context_does_not_mutate_original(self, base_proposal):
        """with_error_context() doesn't modify the original proposal."""
        original_id = base_proposal.proposal_id
        original_attempt = base_proposal.attempt
        error = {"error_code": "TEST_ERROR"}
        base_proposal.with_error_context(error)
        assert base_proposal.proposal_id == original_id
        assert base_proposal.attempt == original_attempt
        assert base_proposal.error_context is None


class TestBaseProposalSummary:
    """Test summary() method."""

    def test_summary_contains_type(self, base_proposal):
        """summary() includes type field."""
        summary = base_proposal.summary()
        assert "type" in summary
        assert summary["type"] == "BaseProposal"

    def test_summary_contains_proposal_id(self, base_proposal):
        """summary() includes proposal_id."""
        summary = base_proposal.summary()
        assert summary["proposal_id"] == base_proposal.proposal_id

    def test_summary_contains_confidence(self, base_proposal):
        """summary() includes confidence."""
        summary = base_proposal.summary()
        assert summary["confidence"] == 0.75

    def test_summary_attempt_format(self, base_proposal):
        """summary() formats attempt as 'current/max'."""
        summary = base_proposal.summary()
        assert summary["attempt"] == "1/3"

    def test_summary_source_prompt_len(self, base_proposal):
        """summary() includes source_prompt_len."""
        summary = base_proposal.summary()
        assert summary["source_prompt_len"] == len("A simple test shape")

    def test_summary_has_error_context_false(self, base_proposal):
        """summary() indicates no error_context initially."""
        summary = base_proposal.summary()
        assert summary["has_error_context"] is False

    def test_summary_has_error_context_true(self, base_proposal):
        """summary() indicates error_context when present."""
        error = {"error_code": "TEST"}
        new_prop = base_proposal.with_error_context(error)
        summary = new_prop.summary()
        assert summary["has_error_context"] is True

    def test_summary_num_alternatives(self):
        """summary() counts alternatives."""
        prop = BaseProposal(alternatives=[1, 2, 3])
        summary = prop.summary()
        assert summary["num_alternatives"] == 3


# =============================================================================
# CodeProposal Tests
# =============================================================================

class TestCodeProposalPostInit:
    """Test __post_init__ behavior for hash and imports."""

    def test_code_hash_computed_on_init(self, code_proposal_cadquery):
        """__post_init__ computes code_hash from code."""
        expected_hash = hashlib.sha256(
            code_proposal_cadquery.code.encode()
        ).hexdigest()
        assert code_proposal_cadquery.code_hash == expected_hash

    def test_imports_extracted_on_init(self):
        """__post_init__ extracts imports from code with import statements."""
        # Use code with actual import statement for this test
        code = """from cadquery import Workplane as cq
result = cq("XY").box(100, 50, 20)
result.val()
"""
        prop = CodeProposal(code=code, language=CodeLanguage.CADQUERY)
        assert "cadquery" in prop.imports_required

    def test_code_hash_not_recomputed_if_set(self):
        """__post_init__ skips hash if already set."""
        code = "from cadquery import Workplane"
        preset_hash = "preset123"
        prop = CodeProposal(code=code, code_hash=preset_hash)
        assert prop.code_hash == preset_hash

    def test_empty_code_no_hash_error(self):
        """__post_init__ handles empty code gracefully."""
        prop = CodeProposal(code="")
        assert prop.code_hash is None
        assert prop.imports_required == []


class TestCodeProposalValidateSyntax:
    """Test syntax validation for Python and OpenSCAD."""

    def test_valid_cadquery_syntax(self, code_proposal_cadquery):
        """validate_syntax() returns True for valid CadQuery code."""
        result = code_proposal_cadquery.validate_syntax()
        assert result is True
        assert code_proposal_cadquery.syntax_valid is True

    def test_valid_cadquery_bracket_syntax(self, code_proposal_bracket):
        """validate_syntax() handles complex CadQuery code."""
        result = code_proposal_bracket.validate_syntax()
        assert result is True

    def test_invalid_cadquery_syntax(self, code_proposal_syntax_error):
        """validate_syntax() returns False for Python syntax errors."""
        result = code_proposal_syntax_error.validate_syntax()
        assert result is False
        assert code_proposal_syntax_error.syntax_valid is False

    def test_valid_openscad_syntax(self, code_proposal_openscad):
        """validate_syntax() returns True for valid OpenSCAD code."""
        result = code_proposal_openscad.validate_syntax()
        assert result is True
        assert code_proposal_openscad.syntax_valid is True

    def test_invalid_openscad_unbalanced_braces(self, code_proposal_openscad_bad):
        """validate_syntax() detects unbalanced braces in OpenSCAD."""
        result = code_proposal_openscad_bad.validate_syntax()
        assert result is False

    def test_openscad_with_primitive_call_passes(self):
        """validate_syntax() passes for OpenSCAD with a primitive call.

        The heuristic checks brace/bracket balance + primitive presence.
        A single primitive call with balanced brackets passes.
        """
        code = "cube([10, 10, 10])"
        prop = CodeProposal(
            code=code,
            language=CodeLanguage.OPENSCAD,
        )
        result = prop.validate_syntax()
        assert result is True

    def test_openscad_requires_primitive(self):
        """validate_syntax() requires at least one OpenSCAD primitive."""
        code = "// Just a comment\n// No code here"
        prop = CodeProposal(
            code=code,
            language=CodeLanguage.OPENSCAD,
        )
        result = prop.validate_syntax()
        assert result is False


class TestCodeProposalImportExtraction:
    """Test _extract_imports() behavior."""

    def test_extract_cadquery_import(self):
        """_extract_imports() finds cadquery import from code with imports."""
        # Use code with actual import statement for this test
        code = """from cadquery import Workplane as cq
result = cq("XY").box(100, 50, 20)
result.val()
"""
        prop = CodeProposal(code=code, language=CodeLanguage.CADQUERY)
        imports = prop.imports_required
        assert "cadquery" in imports

    def test_extract_multiple_imports(self):
        """_extract_imports() handles multiple imports."""
        code = """
import math
import numpy as np
from cadquery import Workplane
"""
        prop = CodeProposal(code=code, language=CodeLanguage.CADQUERY)
        imports = prop.imports_required
        assert "math" in imports
        assert "numpy" in imports
        assert "cadquery" in imports

    def test_extract_openscad_includes(self):
        """_extract_imports() extracts OpenSCAD include/use directives."""
        code = """
use <lib/shapes.scad>
include <constants.scad>
cube([10, 10, 10]);
"""
        prop = CodeProposal(code=code, language=CodeLanguage.OPENSCAD)
        imports = prop.imports_required
        assert "lib/shapes.scad" in imports
        assert "constants.scad" in imports

    def test_imports_sorted_and_unique(self):
        """_extract_imports() returns sorted unique list."""
        code = """
import os
import sys
import os
"""
        prop = CodeProposal(code=code, language=CodeLanguage.CADQUERY)
        imports = prop.imports_required
        assert imports == sorted(set(imports))
        assert imports.count("os") == 1


class TestCodeProposalLineAndTokenEstimates:
    """Test code introspection properties."""

    def test_line_count_simple(self, code_proposal_cadquery):
        """line_count property counts lines correctly."""
        count = code_proposal_cadquery.line_count
        assert count > 0
        assert count == len(code_proposal_cadquery.code.strip().splitlines())

    def test_line_count_empty_code(self):
        """line_count is 0 for empty code."""
        prop = CodeProposal(code="")
        assert prop.line_count == 0

    def test_line_count_single_line(self):
        """line_count handles single-line code."""
        prop = CodeProposal(code="x = 1")
        assert prop.line_count == 1

    def test_token_estimate(self, code_proposal_cadquery):
        """token_estimate property estimates tokens as len/4."""
        estimate = code_proposal_cadquery.token_estimate
        expected = len(code_proposal_cadquery.code) // 4
        assert estimate == expected

    def test_token_estimate_zero_for_empty(self):
        """token_estimate is 0 for empty code."""
        prop = CodeProposal(code="")
        assert prop.token_estimate == 0


class TestCodeProposalWithErrorContext:
    """Test CodeProposal retry behavior."""

    def test_with_error_context_clears_code(self, code_proposal_cadquery):
        """with_error_context() clears the code."""
        error = {"error_code": "SYNTAX_ERROR"}
        new_prop = code_proposal_cadquery.with_error_context(error)
        assert new_prop.code == ""

    def test_with_error_context_resets_hash(self, code_proposal_cadquery):
        """with_error_context() clears code_hash."""
        error = {"error_code": "SYNTAX_ERROR"}
        new_prop = code_proposal_cadquery.with_error_context(error)
        assert new_prop.code_hash is None

    def test_with_error_context_resets_syntax_valid(self, code_proposal_cadquery):
        """with_error_context() clears syntax_valid."""
        error = {"error_code": "SYNTAX_ERROR"}
        new_prop = code_proposal_cadquery.with_error_context(error)
        assert new_prop.syntax_valid is None

    def test_with_error_context_clears_imports(self, code_proposal_cadquery):
        """with_error_context() clears imports_required."""
        error = {"error_code": "SYNTAX_ERROR"}
        new_prop = code_proposal_cadquery.with_error_context(error)
        assert new_prop.imports_required == []

    def test_with_error_context_resets_confidence(self, code_proposal_cadquery):
        """with_error_context() resets confidence to 0."""
        error = {"error_code": "SYNTAX_ERROR"}
        new_prop = code_proposal_cadquery.with_error_context(error)
        assert new_prop.confidence == 0.0

    def test_with_error_context_preserves_language(self, code_proposal_cadquery):
        """with_error_context() preserves language."""
        error = {"error_code": "SYNTAX_ERROR"}
        new_prop = code_proposal_cadquery.with_error_context(error)
        assert new_prop.language == CodeLanguage.CADQUERY


class TestCodeProposalSummary:
    """Test CodeProposal.summary() extension."""

    def test_summary_includes_language(self, code_proposal_cadquery):
        """summary() includes language field."""
        summary = code_proposal_cadquery.summary()
        assert summary["language"] == CodeLanguage.CADQUERY.value

    def test_summary_includes_syntax_valid(self, code_proposal_cadquery):
        """summary() includes syntax_valid field."""
        code_proposal_cadquery.validate_syntax()
        summary = code_proposal_cadquery.summary()
        assert "syntax_valid" in summary

    def test_summary_includes_line_count(self, code_proposal_cadquery):
        """summary() includes line_count."""
        summary = code_proposal_cadquery.summary()
        assert summary["line_count"] == code_proposal_cadquery.line_count

    def test_summary_includes_token_estimate(self, code_proposal_cadquery):
        """summary() includes token_estimate."""
        summary = code_proposal_cadquery.summary()
        assert summary["token_estimate"] == code_proposal_cadquery.token_estimate

    def test_summary_includes_imports(self, code_proposal_cadquery):
        """summary() includes imports list."""
        summary = code_proposal_cadquery.summary()
        assert "imports" in summary
        assert isinstance(summary["imports"], list)

    def test_summary_inherits_base_fields(self, code_proposal_cadquery):
        """summary() includes inherited BaseProposal fields."""
        summary = code_proposal_cadquery.summary()
        assert "proposal_id" in summary
        assert "confidence" in summary
        assert "attempt" in summary
        assert "type" in summary


# =============================================================================
# CommandSequenceProposal Tests
# =============================================================================

class TestCommandSequenceProposalConstruction:
    """Test CommandSequenceProposal initialization."""

    def test_construction_with_command_dicts(self, command_proposal):
        """CommandSequenceProposal accepts command_dicts."""
        assert len(command_proposal.command_dicts) > 0
        assert command_proposal.quantization_bits == 8
        assert command_proposal.normalization_range == 2.0

    def test_construction_with_token_ids(self, command_proposal_token_ids):
        """CommandSequenceProposal accepts token_ids."""
        assert len(command_proposal_token_ids.token_ids) > 0
        assert command_proposal_token_ids.quantization_bits == 8

    def test_defaults_quantization_bits(self):
        """Default quantization_bits is 8."""
        prop = CommandSequenceProposal(proposal_id="test")
        assert prop.quantization_bits == 8

    def test_defaults_normalization_range(self):
        """Default normalization_range is 2.0."""
        prop = CommandSequenceProposal(proposal_id="test")
        assert prop.normalization_range == 2.0

    def test_defaults_precision_tier(self):
        """Default precision_tier is 'STANDARD'."""
        prop = CommandSequenceProposal(proposal_id="test")
        assert prop.precision_tier == "STANDARD"


class TestCommandSequenceProposalSequenceLength:
    """Test sequence_length property."""

    def test_sequence_length_from_command_dicts(self, command_proposal):
        """sequence_length counts command_dicts."""
        length = command_proposal.sequence_length
        assert length == len(command_proposal.command_dicts)

    def test_sequence_length_from_token_ids(self, command_proposal_token_ids):
        """sequence_length counts token_ids when no dicts."""
        length = command_proposal_token_ids.sequence_length
        assert length == len(command_proposal_token_ids.token_ids)

    def test_sequence_length_empty(self):
        """sequence_length is 0 for empty proposal."""
        prop = CommandSequenceProposal(proposal_id="test")
        assert prop.sequence_length == 0


class TestCommandSequenceProposalCommandCounts:
    """Test num_sketch_loops and num_extrusions."""

    def test_num_sketch_loops_from_dicts(self, command_proposal):
        """num_sketch_loops counts SOL commands in dicts."""
        count = command_proposal.num_sketch_loops
        # Rectangle extrusion fixture has 1 SOL
        assert count == 1

    def test_num_sketch_loops_from_token_ids(self, command_proposal_token_ids):
        """num_sketch_loops counts token_id 6 (SOL)."""
        count = command_proposal_token_ids.num_sketch_loops
        assert count == 1  # Fixture has 1 SOL

    def test_num_extrusions_from_dicts(self, command_proposal):
        """num_extrusions counts EXTRUDE commands in dicts."""
        count = command_proposal.num_extrusions
        assert count == 1

    def test_num_extrusions_from_token_ids(self, command_proposal_token_ids):
        """num_extrusions counts token_id 10 (EXTRUDE)."""
        count = command_proposal_token_ids.num_extrusions
        assert count == 1


class TestCommandSequenceProposalDequantize:
    """Test dequantize() method for parameter conversion."""

    def test_dequantize_basic(self, command_proposal):
        """dequantize() converts quantized params to floats."""
        dequant = command_proposal.dequantize()
        assert isinstance(dequant, list)
        assert len(dequant) > 0
        for cmd in dequant:
            assert "command_type" in cmd
            assert "parameters" in cmd
            assert isinstance(cmd["parameters"], list)
            for p in cmd["parameters"]:
                assert isinstance(p, float)

    def test_dequantize_range_mapping(self, command_proposal):
        """dequantize() maps quantized values to [-range/2, range/2]."""
        # For 8-bit quantization: levels = 256
        # 0 maps to: (0/255) * 2.0 - 1.0 = -1.0
        # 255 maps to: (255/255) * 2.0 - 1.0 = 1.0
        # 128 (middle) maps to: (128/255) * 2.0 - 1.0 ≈ 0.003
        dequant = command_proposal.dequantize()
        # Check that at least one parameter is close to expected range
        for cmd in dequant:
            for param in cmd["parameters"]:
                assert -1.1 < param < 1.1  # Slightly wider for floating point

    def test_dequantize_128_middle_value(self):
        """dequantize() 128 with 8-bit quantization ≈ 0."""
        command_dicts = [
            {
                "command_type": "LINE",
                "parameters": [128, 128, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "parameter_mask": [True, True, True] + [False] * 13,
            }
        ]
        prop = CommandSequenceProposal(
            proposal_id="test",
            command_dicts=command_dicts,
            quantization_bits=8,
            normalization_range=2.0,
        )
        dequant = prop.dequantize()
        # 128/255 * 2.0 - 1.0 ≈ 0.00392
        assert abs(dequant[0]["parameters"][0] - 0.004) < 0.01

    def test_dequantize_255_max_value(self):
        """dequantize() 255 with 8-bit quantization ≈ 1.0."""
        command_dicts = [
            {
                "command_type": "LINE",
                "parameters": [255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "parameter_mask": [True] + [False] * 15,
            }
        ]
        prop = CommandSequenceProposal(
            proposal_id="test",
            command_dicts=command_dicts,
            quantization_bits=8,
            normalization_range=2.0,
        )
        dequant = prop.dequantize()
        # 255/255 * 2.0 - 1.0 = 1.0
        assert abs(dequant[0]["parameters"][0] - 1.0) < 0.01

    def test_dequantize_respects_mask(self):
        """dequantize() only converts masked parameters."""
        command_dicts = [
            {
                "command_type": "LINE",
                "parameters": [128, 200, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "parameter_mask": [True, False, True] + [False] * 13,
            }
        ]
        prop = CommandSequenceProposal(
            proposal_id="test",
            command_dicts=command_dicts,
            quantization_bits=8,
            normalization_range=2.0,
        )
        dequant = prop.dequantize()
        # param[0] masked=True, so converted
        # param[1] masked=False, so should be 0.0
        # param[2] masked=True, so converted
        assert dequant[0]["parameters"][1] == 0.0


class TestCommandSequenceProposalWithErrorContext:
    """Test CommandSequenceProposal retry behavior."""

    def test_with_error_context_clears_tokens(self, command_proposal):
        """with_error_context() clears token_ids and command_dicts."""
        error = {"error_code": "INVALID_SHAPE"}
        new_prop = command_proposal.with_error_context(error)
        assert new_prop.token_ids == []
        assert new_prop.command_dicts == []

    def test_with_error_context_preserves_latent(self, command_proposal):
        """with_error_context() keeps latent_vector."""
        error = {"error_code": "INVALID_SHAPE"}
        new_prop = command_proposal.with_error_context(error)
        if command_proposal.latent_vector is not None:
            assert new_prop.latent_vector is not None
            # Should be a copy, not the same object
            assert new_prop.latent_vector is not command_proposal.latent_vector

    def test_with_error_context_resets_confidence(self, command_proposal):
        """with_error_context() resets confidence to 0."""
        error = {"error_code": "INVALID_SHAPE"}
        new_prop = command_proposal.with_error_context(error)
        assert new_prop.confidence == 0.0

    def test_with_error_context_increments_attempt(self, command_proposal):
        """with_error_context() increments attempt."""
        error = {"error_code": "INVALID_SHAPE"}
        new_prop = command_proposal.with_error_context(error)
        assert new_prop.attempt == command_proposal.attempt + 1


class TestCommandSequenceProposalSummary:
    """Test CommandSequenceProposal.summary() extension."""

    def test_summary_includes_sequence_length(self, command_proposal):
        """summary() includes sequence_length."""
        summary = command_proposal.summary()
        assert summary["sequence_length"] == command_proposal.sequence_length

    def test_summary_includes_sketch_loops(self, command_proposal):
        """summary() includes num_sketch_loops."""
        summary = command_proposal.summary()
        assert summary["num_sketch_loops"] == command_proposal.num_sketch_loops

    def test_summary_includes_extrusions(self, command_proposal):
        """summary() includes num_extrusions."""
        summary = command_proposal.summary()
        assert summary["num_extrusions"] == command_proposal.num_extrusions

    def test_summary_includes_quantization_bits(self, command_proposal):
        """summary() includes quantization_bits."""
        summary = command_proposal.summary()
        assert summary["quantization_bits"] == 8

    def test_summary_includes_precision_tier(self, command_proposal):
        """summary() includes precision_tier."""
        summary = command_proposal.summary()
        assert summary["precision_tier"] == "STANDARD"

    def test_summary_has_latent_flag(self, command_proposal):
        """summary() includes has_latent flag."""
        summary = command_proposal.summary()
        assert "has_latent" in summary
        assert summary["has_latent"] is True


# =============================================================================
# LatentProposal Tests
# =============================================================================

class TestLatentProposalProperties:
    """Test latent proposal properties."""

    def test_num_faces(self, latent_proposal):
        """num_faces counts face_grids."""
        assert latent_proposal.num_faces == 6

    def test_num_edges(self, latent_proposal):
        """num_edges counts edge_points."""
        assert latent_proposal.num_edges == 12

    def test_num_vertices(self, latent_proposal):
        """num_vertices counts vertex_positions."""
        assert latent_proposal.num_vertices == 8

    def test_num_vertices_none(self, latent_proposal_minimal):
        """num_vertices is 0 when vertex_positions is None."""
        assert latent_proposal_minimal.num_vertices == 0

    def test_face_grid_resolution(self, latent_proposal):
        """face_grid_resolution returns (U, V) tuple."""
        res = latent_proposal.face_grid_resolution
        assert res == (32, 32)

    def test_face_grid_resolution_empty(self):
        """face_grid_resolution is None for empty grids."""
        prop = LatentProposal(proposal_id="test")
        assert prop.face_grid_resolution is None

    def test_total_points(self, latent_proposal):
        """total_points sums face and edge points."""
        face_pts = sum(g.shape[0] * g.shape[1] for g in latent_proposal.face_grids)
        edge_pts = sum(e.shape[0] for e in latent_proposal.edge_points)
        assert latent_proposal.total_points == face_pts + edge_pts

    def test_total_points_minimal(self, latent_proposal_minimal):
        """total_points for minimal proposal."""
        # 1 face of 4x4, no edges
        assert latent_proposal_minimal.total_points == 16


class TestLatentProposalValidateShapes:
    """Test validate_shapes() method."""

    def test_validate_shapes_valid(self, latent_proposal):
        """validate_shapes() returns empty list for valid shapes."""
        errors = latent_proposal.validate_shapes()
        assert errors == []

    def test_validate_shapes_wrong_face_ndim(self):
        """validate_shapes() detects wrong ndim in face_grids."""
        grid = np.zeros((32, 32))  # 2D instead of 3D
        prop = LatentProposal(proposal_id="test", face_grids=[grid])
        errors = prop.validate_shapes()
        assert any("ndim" in e for e in errors)

    def test_validate_shapes_wrong_face_last_dim(self):
        """validate_shapes() detects wrong last dimension in face_grids."""
        grid = np.zeros((32, 32, 4))  # 4 instead of 3
        prop = LatentProposal(proposal_id="test", face_grids=[grid])
        errors = prop.validate_shapes()
        assert any("last dim" in e for e in errors)

    def test_validate_shapes_wrong_edge_ndim(self):
        """validate_shapes() detects wrong ndim in edge_points."""
        points = np.zeros((64,))  # 1D instead of 2D
        prop = LatentProposal(proposal_id="test", edge_points=[points])
        errors = prop.validate_shapes()
        assert any("ndim" in e for e in errors)

    def test_validate_shapes_wrong_edge_last_dim(self):
        """validate_shapes() detects wrong last dimension in edge_points."""
        points = np.zeros((64, 4))  # 4 instead of 3
        prop = LatentProposal(proposal_id="test", edge_points=[points])
        errors = prop.validate_shapes()
        assert any("last dim" in e for e in errors)

    def test_validate_shapes_face_bboxes_mismatch(self, latent_proposal):
        """validate_shapes() detects face_bboxes shape mismatch."""
        # Set wrong number of face bboxes
        wrong_bboxes = np.zeros((5, 6))  # Should be (6, 6)
        latent_proposal.face_bboxes = wrong_bboxes
        errors = latent_proposal.validate_shapes()
        assert any("face_bboxes" in e for e in errors)

    def test_validate_shapes_edge_bboxes_mismatch(self, latent_proposal):
        """validate_shapes() detects edge_bboxes shape mismatch."""
        # Set wrong number of edge bboxes
        wrong_bboxes = np.zeros((11, 6))  # Should be (12, 6)
        latent_proposal.edge_bboxes = wrong_bboxes
        errors = latent_proposal.validate_shapes()
        assert any("edge_bboxes" in e for e in errors)

    def test_validate_shapes_vertex_positions_wrong_shape(self):
        """validate_shapes() detects vertex_positions shape error."""
        grid = np.zeros((4, 4, 3))
        vertices = np.zeros((8, 4))  # Should be (V, 3)
        prop = LatentProposal(
            proposal_id="test",
            face_grids=[grid],
            vertex_positions=vertices,
        )
        errors = prop.validate_shapes()
        assert any("vertex_positions" in e for e in errors)


class TestLatentProposalComputeBoundingBox:
    """Test compute_bounding_box() method."""

    def test_compute_bounding_box_valid(self, latent_proposal):
        """compute_bounding_box() returns (6,) array."""
        bbox = latent_proposal.compute_bounding_box()
        assert isinstance(bbox, np.ndarray)
        assert bbox.shape == (6,)

    def test_compute_bounding_box_correct_range(self):
        """compute_bounding_box() computes correct min/max."""
        # Create face grid with known bounds
        grid = np.zeros((4, 4, 3), dtype=np.float32)
        for u in range(4):
            for v in range(4):
                grid[u, v] = [u * 10.0, v * 10.0, 0.0]

        prop = LatentProposal(proposal_id="test", face_grids=[grid])
        bbox = prop.compute_bounding_box()

        # Grid ranges: x: 0-30, y: 0-30, z: 0
        np.testing.assert_array_almost_equal(
            bbox,
            [0.0, 0.0, 0.0, 30.0, 30.0, 0.0],
        )

    def test_compute_bounding_box_empty(self):
        """compute_bounding_box() returns None for empty proposal."""
        prop = LatentProposal(proposal_id="test")
        bbox = prop.compute_bounding_box()
        assert bbox is None

    def test_compute_bounding_box_multiple_grids(self):
        """compute_bounding_box() combines all geometry."""
        grid1 = np.array([[[0, 0, 0], [10, 10, 10]]], dtype=np.float32).reshape(1, 2, 3)
        grid2 = np.array([[[20, 20, 20], [30, 30, 30]]], dtype=np.float32).reshape(1, 2, 3)
        prop = LatentProposal(proposal_id="test", face_grids=[grid1, grid2])
        bbox = prop.compute_bounding_box()

        # Combined range: x: 0-30, y: 0-30, z: 0-30
        assert bbox[0] == 0.0
        assert bbox[3] == 30.0


class TestLatentProposalComputeFaceAreas:
    """Test compute_face_areas_approximate() method."""

    def test_compute_face_areas_returns_list(self, latent_proposal):
        """compute_face_areas_approximate() returns list of floats."""
        areas = latent_proposal.compute_face_areas_approximate()
        assert isinstance(areas, list)
        assert len(areas) == latent_proposal.num_faces

    def test_compute_face_areas_all_positive(self, latent_proposal):
        """compute_face_areas_approximate() returns positive areas."""
        areas = latent_proposal.compute_face_areas_approximate()
        for area in areas:
            assert isinstance(area, float)
            assert area >= 0.0

    def test_compute_face_areas_flat_square(self, latent_proposal_minimal):
        """compute_face_areas_approximate() for flat square.

        The grid is 4×4 with 10.0 spacing → 3×3 = 9 cells.
        Each cell has two triangles, each with area 0.5 * 10 * 10 = 50.
        Total area = 9 * 2 * 50 = 900.
        """
        areas = latent_proposal_minimal.compute_face_areas_approximate()
        assert len(areas) == 1
        # 4×4 grid with 10.0 spacing: 3×3 cells × 2 triangles × 50 = 900
        assert 850 < areas[0] < 950

    def test_compute_face_areas_empty(self):
        """compute_face_areas_approximate() returns empty list for no faces."""
        prop = LatentProposal(proposal_id="test")
        areas = prop.compute_face_areas_approximate()
        assert areas == []


class TestLatentProposalWithErrorContext:
    """Test LatentProposal retry behavior."""

    def test_with_error_context_clears_geometry(self, latent_proposal):
        """with_error_context() clears geometry arrays."""
        error = {"error_code": "INVALID_SHAPE"}
        new_prop = latent_proposal.with_error_context(error)
        assert new_prop.face_grids == []
        assert new_prop.edge_points == []
        assert new_prop.face_bboxes is None
        assert new_prop.edge_bboxes is None
        assert new_prop.vertex_positions is None
        assert new_prop.face_edge_adjacency is None

    def test_with_error_context_preserves_stage_latents(self, latent_proposal):
        """with_error_context() preserves stage_latents."""
        latent_proposal.stage_latents = {"face_positions": np.zeros(10)}
        error = {"error_code": "INVALID_SHAPE"}
        new_prop = latent_proposal.with_error_context(error)
        assert new_prop.stage_latents == latent_proposal.stage_latents

    def test_with_error_context_increments_attempt(self, latent_proposal):
        """with_error_context() increments attempt."""
        error = {"error_code": "INVALID_SHAPE"}
        new_prop = latent_proposal.with_error_context(error)
        assert new_prop.attempt == 2


class TestLatentProposalSummary:
    """Test LatentProposal.summary() extension."""

    def test_summary_includes_num_faces(self, latent_proposal):
        """summary() includes num_faces."""
        summary = latent_proposal.summary()
        assert summary["num_faces"] == 6

    def test_summary_includes_num_edges(self, latent_proposal):
        """summary() includes num_edges."""
        summary = latent_proposal.summary()
        assert summary["num_edges"] == 12

    def test_summary_includes_num_vertices(self, latent_proposal):
        """summary() includes num_vertices."""
        summary = latent_proposal.summary()
        assert summary["num_vertices"] == 8

    def test_summary_includes_face_grid_resolution(self, latent_proposal):
        """summary() includes face_grid_resolution."""
        summary = latent_proposal.summary()
        assert summary["face_grid_resolution"] == (32, 32)

    def test_summary_includes_total_points(self, latent_proposal):
        """summary() includes total_points."""
        summary = latent_proposal.summary()
        assert summary["total_points"] == latent_proposal.total_points

    def test_summary_has_stage_latents_flag(self, latent_proposal):
        """summary() includes has_stage_latents."""
        summary = latent_proposal.summary()
        assert summary["has_stage_latents"] is False


# =============================================================================
# GeometryReport Tests
# =============================================================================

class TestGeometryReportBboxDimensions:
    """Test bbox_dimensions property."""

    def test_bbox_dimensions_box(self, geometry_report_box):
        """bbox_dimensions extracts (width, height, depth) correctly."""
        dims = geometry_report_box.bbox_dimensions
        assert dims == (100.0, 50.0, 20.0)

    def test_bbox_dimensions_none_when_no_bbox(self, geometry_report_no_bbox):
        """bbox_dimensions returns None when bounding_box is None."""
        dims = geometry_report_no_bbox.bbox_dimensions
        assert dims is None

    def test_bbox_dimensions_short_tuple(self):
        """bbox_dimensions handles malformed bbox tuple."""
        report = GeometryReport(bounding_box=(0, 0, 0, 10))  # Too short
        dims = report.bbox_dimensions
        assert dims is None


class TestGeometryReportBboxDiagonal:
    """Test bbox_diagonal property."""

    def test_bbox_diagonal_box(self, geometry_report_box):
        """bbox_diagonal computes correct diagonal."""
        diag = geometry_report_box.bbox_diagonal
        # diagonal = sqrt(100^2 + 50^2 + 20^2) = sqrt(12900) ≈ 113.58
        expected = (100.0 ** 2 + 50.0 ** 2 + 20.0 ** 2) ** 0.5
        assert abs(diag - expected) < 0.01

    def test_bbox_diagonal_none_when_no_bbox(self, geometry_report_no_bbox):
        """bbox_diagonal returns None when no bounding_box."""
        diag = geometry_report_no_bbox.bbox_diagonal
        assert diag is None


class TestGeometryReportMatchesDimensions:
    """Test matches_dimensions() method."""

    def test_matches_dimensions_exact(self, geometry_report_box):
        """matches_dimensions() returns True for exact match."""
        result = geometry_report_box.matches_dimensions((100, 50, 20))
        assert result is True

    def test_matches_dimensions_within_tolerance(self, geometry_report_box):
        """matches_dimensions() accepts 5% deviation with 10% tolerance."""
        # 100 * 0.95 = 95 (5% off)
        result = geometry_report_box.matches_dimensions((95, 47.5, 19))
        assert result is True

    def test_matches_dimensions_exceeds_tolerance(self, geometry_report_box):
        """matches_dimensions() rejects 15% deviation with 10% tolerance."""
        # 100 * 0.85 = 85 (15% off)
        result = geometry_report_box.matches_dimensions((85, 42.5, 17))
        assert result is False

    def test_matches_dimensions_axis_order_independent(self, geometry_report_box):
        """matches_dimensions() ignores axis order."""
        # Should match in any order
        assert geometry_report_box.matches_dimensions((100, 50, 20)) is True
        assert geometry_report_box.matches_dimensions((20, 100, 50)) is True
        assert geometry_report_box.matches_dimensions((50, 20, 100)) is True

    def test_matches_dimensions_no_bbox(self, geometry_report_no_bbox):
        """matches_dimensions() returns False when no bbox."""
        result = geometry_report_no_bbox.matches_dimensions((100, 50, 20))
        assert result is False

    def test_matches_dimensions_zero_target(self, geometry_report_box):
        """matches_dimensions() handles zero target dimensions."""
        result = geometry_report_box.matches_dimensions((100, 50, 0))
        assert result is False

    def test_matches_dimensions_custom_tolerance(self, geometry_report_box):
        """matches_dimensions() respects tolerance_pct parameter."""
        # 15% off with 20% tolerance should pass
        result = geometry_report_box.matches_dimensions(
            (85, 42.5, 17),
            tolerance_pct=0.20,
        )
        assert result is True


# =============================================================================
# ValidationFinding Tests
# =============================================================================

class TestValidationFinding:
    """Test ValidationFinding construction and defaults."""

    def test_construction_defaults(self):
        """ValidationFinding has sensible defaults."""
        finding = ValidationFinding()
        assert finding.entity_type == ""
        assert finding.entity_index == 0
        assert finding.error_code == ""
        assert finding.error_category == ErrorCategory.TOPOLOGY_ERROR
        assert finding.severity == ErrorSeverity.CRITICAL
        assert finding.description == ""
        assert finding.suggestion == ""

    def test_construction_custom_values(self):
        """ValidationFinding accepts custom values."""
        finding = ValidationFinding(
            entity_type="FACE",
            entity_index=3,
            error_code="BRepCheck_NotClosed",
            error_category=ErrorCategory.TOPOLOGY_ERROR,
            severity=ErrorSeverity.CRITICAL,
            description="Face is not closed",
            suggestion="Check sketch closure",
        )
        assert finding.entity_type == "FACE"
        assert finding.entity_index == 3
        assert finding.error_code == "BRepCheck_NotClosed"
        assert finding.severity == ErrorSeverity.CRITICAL


# =============================================================================
# RepairAction Tests
# =============================================================================

class TestRepairAction:
    """Test RepairAction construction and defaults."""

    def test_construction_defaults(self):
        """RepairAction has sensible defaults."""
        action = RepairAction()
        assert action.tool == ""
        assert action.action == ""
        assert action.status == "done"
        assert action.tolerance_used is None
        assert action.entities_affected == 0

    def test_construction_custom_values(self):
        """RepairAction accepts custom values."""
        action = RepairAction(
            tool="ShapeFix_Wire",
            action="Fixed wire issues",
            status="done",
            tolerance_used=1e-7,
            entities_affected=2,
        )
        assert action.tool == "ShapeFix_Wire"
        assert action.action == "Fixed wire issues"
        assert action.tolerance_used == 1e-7
        assert action.entities_affected == 2


# =============================================================================
# DisposalResult Tests
# =============================================================================

class TestDisposalResultHasShape:
    """Test has_shape property."""

    def test_has_shape_true(self, disposal_result_valid):
        """has_shape returns True when shape is set."""
        assert disposal_result_valid.has_shape is True

    def test_has_shape_false(self, disposal_result_no_shape):
        """has_shape returns False when shape is None."""
        assert disposal_result_no_shape.has_shape is False


class TestDisposalResultHasGeometryReport:
    """Test has_geometry_report property."""

    def test_has_geometry_report_true(self, disposal_result_valid):
        """has_geometry_report returns True when report is set."""
        assert disposal_result_valid.has_geometry_report is True

    def test_has_geometry_report_false(self, disposal_result_no_shape):
        """has_geometry_report returns False when report is None."""
        assert disposal_result_no_shape.has_geometry_report is False


class TestDisposalResultErrorCounts:
    """Test error counting properties."""

    def test_num_errors_valid(self, disposal_result_valid):
        """num_errors counts error_details."""
        assert disposal_result_valid.num_errors == 0

    def test_num_errors_invalid(self, disposal_result_invalid):
        """num_errors counts error_details correctly."""
        assert disposal_result_invalid.num_errors == 2

    def test_critical_errors_filters(self, disposal_result_invalid):
        """critical_errors filters by CRITICAL severity."""
        critical = disposal_result_invalid.critical_errors
        assert len(critical) > 0
        for finding in critical:
            assert finding.severity == ErrorSeverity.CRITICAL

    def test_num_critical_errors_count(self, disposal_result_invalid):
        """num_critical_errors counts CRITICAL findings."""
        num = disposal_result_invalid.num_critical_errors
        assert num == len(disposal_result_invalid.critical_errors)
        assert num > 0


class TestDisposalResultErrorsByCategory:
    """Test errors_by_category property."""

    def test_errors_by_category_groups(self, disposal_result_invalid):
        """errors_by_category groups findings by category."""
        grouped = disposal_result_invalid.errors_by_category
        assert isinstance(grouped, dict)
        # Should have at least topology errors
        assert ErrorCategory.TOPOLOGY_ERROR in grouped

    def test_errors_by_category_empty_for_valid(self, disposal_result_valid):
        """errors_by_category is empty for valid results."""
        grouped = disposal_result_valid.errors_by_category
        assert grouped == {}

    def test_errors_by_category_self_intersection(self, disposal_result_self_intersection):
        """errors_by_category groups self-intersection errors."""
        grouped = disposal_result_self_intersection.errors_by_category
        assert ErrorCategory.SELF_INTERSECTION in grouped
        assert len(grouped[ErrorCategory.SELF_INTERSECTION]) > 0


class TestDisposalResultWasRepaired:
    """Test was_repaired property."""

    def test_was_repaired_true(self, disposal_result_repaired):
        """was_repaired returns True when repair_attempted and repair_succeeded."""
        assert disposal_result_repaired.was_repaired is True

    def test_was_repaired_false_not_attempted(self, disposal_result_valid):
        """was_repaired returns False when repair not attempted."""
        assert disposal_result_valid.was_repaired is False

    def test_was_repaired_false_attempted_not_succeeded(self, disposal_result_invalid):
        """was_repaired returns False when repair attempted but failed."""
        assert disposal_result_invalid.was_repaired is False


class TestDisposalResultSerialization:
    """Test to_dict() serialization."""

    def test_to_dict_excludes_shape(self, disposal_result_valid):
        """to_dict() excludes the shape object."""
        d = disposal_result_valid.to_dict()
        assert "shape" not in d

    def test_to_dict_includes_validity(self, disposal_result_valid):
        """to_dict() includes is_valid."""
        d = disposal_result_valid.to_dict()
        assert d["is_valid"] is True

    def test_to_dict_serializes_error_category(self, disposal_result_invalid):
        """to_dict() serializes error_category as string."""
        d = disposal_result_invalid.to_dict()
        assert isinstance(d["error_category"], str)

    def test_to_dict_serializes_error_details(self, disposal_result_invalid):
        """to_dict() serializes error_details as dicts."""
        d = disposal_result_invalid.to_dict()
        assert isinstance(d["error_details"], list)
        for detail in d["error_details"]:
            assert isinstance(detail, dict)
            assert "entity_type" in detail
            assert "error_code" in detail

    def test_to_dict_serializes_geometry_report(self, disposal_result_valid):
        """to_dict() includes geometry_report as dict."""
        d = disposal_result_valid.to_dict()
        assert isinstance(d["geometry_report"], dict)
        assert "volume" in d["geometry_report"]

    def test_to_dict_geometry_report_none(self, disposal_result_no_shape):
        """to_dict() sets geometry_report to None when missing."""
        d = disposal_result_no_shape.to_dict()
        assert d["geometry_report"] is None

    def test_to_dict_serializes_paths_as_strings(self, disposal_result_valid):
        """to_dict() converts Path objects to strings."""
        d = disposal_result_valid.to_dict()
        if disposal_result_valid.step_path:
            assert isinstance(d["step_path"], str)

    def test_to_dict_handles_missing_paths(self, disposal_result_no_shape):
        """to_dict() sets missing paths to None."""
        d = disposal_result_no_shape.to_dict()
        assert d["step_path"] is None
        assert d["stl_path"] is None


class TestDisposalResultSummary:
    """Test summary() method."""

    def test_summary_compact_format(self, disposal_result_valid):
        """summary() returns compact dict."""
        summary = disposal_result_valid.summary()
        assert isinstance(summary, dict)
        assert len(summary) < 15  # Compact

    def test_summary_includes_proposal_info(self, disposal_result_valid):
        """summary() includes proposal tracking fields."""
        summary = disposal_result_valid.summary()
        assert "proposal_id" in summary
        assert "proposal_type" in summary

    def test_summary_includes_validity(self, disposal_result_valid):
        """summary() includes is_valid."""
        summary = disposal_result_valid.summary()
        assert summary["is_valid"] is True

    def test_summary_includes_shape_flag(self, disposal_result_valid):
        """summary() includes has_shape."""
        summary = disposal_result_valid.summary()
        assert summary["has_shape"] is True

    def test_summary_includes_error_counts(self, disposal_result_invalid):
        """summary() includes error counts."""
        summary = disposal_result_invalid.summary()
        assert "num_errors" in summary
        assert "num_critical" in summary

    def test_summary_includes_repair_status(self, disposal_result_repaired):
        """summary() includes repair flags."""
        summary = disposal_result_repaired.summary()
        assert summary["repair_attempted"] is True
        assert summary["repair_succeeded"] is True

    def test_summary_includes_reward_and_time(self, disposal_result_valid):
        """summary() includes reward_signal and execution_time_ms."""
        summary = disposal_result_valid.summary()
        assert "reward_signal" in summary
        assert "execution_time_ms" in summary


# =============================================================================
# Integration Tests
# =============================================================================

class TestProposalTypeInheritance:
    """Test that subclasses properly inherit BaseProposal behavior."""

    def test_code_proposal_is_base_proposal(self, code_proposal_cadquery):
        """CodeProposal inherits from BaseProposal."""
        assert isinstance(code_proposal_cadquery, BaseProposal)

    def test_code_proposal_has_base_methods(self, code_proposal_cadquery):
        """CodeProposal has BaseProposal methods."""
        assert hasattr(code_proposal_cadquery, "should_retry")
        assert hasattr(code_proposal_cadquery, "next_attempt")
        assert hasattr(code_proposal_cadquery, "is_first_attempt")

    def test_command_proposal_is_base_proposal(self, command_proposal):
        """CommandSequenceProposal inherits from BaseProposal."""
        assert isinstance(command_proposal, BaseProposal)

    def test_latent_proposal_is_base_proposal(self, latent_proposal):
        """LatentProposal inherits from BaseProposal."""
        assert isinstance(latent_proposal, BaseProposal)


class TestErrorContextChaining:
    """Test that error context chains properly through retries."""

    def test_code_proposal_error_chain(self, code_proposal_cadquery):
        """CodeProposal error context creates retry chain."""
        error1 = {"error": "first_attempt"}
        retry1 = code_proposal_cadquery.with_error_context(error1)
        assert retry1.attempt == 2
        assert retry1.error_context == error1

        error2 = {"error": "second_attempt"}
        retry2 = retry1.with_error_context(error2)
        assert retry2.attempt == 3
        assert retry2.error_context == error2

    def test_command_proposal_error_chain(self, command_proposal):
        """CommandSequenceProposal error context creates retry chain."""
        error = {"error": "shape_invalid"}
        retry = command_proposal.with_error_context(error)
        assert retry.attempt == 2
        assert retry.error_context == error

    def test_latent_proposal_error_chain(self, latent_proposal):
        """LatentProposal error context creates retry chain."""
        error = {"error": "geometry_issue"}
        retry = latent_proposal.with_error_context(error)
        assert retry.attempt == 2
        assert retry.error_context == error


class TestDisposalResultRealisticScenarios:
    """Test DisposalResult with realistic workflow scenarios."""

    def test_valid_workflow_full_chain(self, disposal_result_valid):
        """Complete successful workflow produces valid result."""
        assert disposal_result_valid.is_valid is True
        assert disposal_result_valid.has_shape is True
        assert disposal_result_valid.has_geometry_report is True
        assert disposal_result_valid.num_errors == 0
        assert disposal_result_valid.reward_signal == 1.0
        assert not disposal_result_valid.was_repaired

    def test_invalid_with_repair_workflow(self, disposal_result_repaired):
        """Repair workflow produces valid shape."""
        assert disposal_result_repaired.is_valid is True
        assert disposal_result_repaired.repair_attempted is True
        assert disposal_result_repaired.repair_succeeded is True
        assert disposal_result_repaired.was_repaired is True

    def test_failed_execution_workflow(self, disposal_result_no_shape):
        """Execution failure produces no shape."""
        assert disposal_result_no_shape.has_shape is False
        assert disposal_result_no_shape.has_geometry_report is False
        assert disposal_result_no_shape.reward_signal == 0.0

    def test_serialization_roundtrip(self, disposal_result_invalid):
        """DisposalResult can be serialized and has all key fields."""
        d = disposal_result_invalid.to_dict()
        # Verify all important fields are in the dict
        assert d["is_valid"] is False
        assert d["error_category"] is not None
        assert len(d["error_details"]) > 0
        assert d["repair_attempted"] is True
        assert d["repair_succeeded"] is False
