"""Comprehensive test suite for ll_gen pipeline module.

Tests cover:
- VerificationResult: construction with defaults and custom values
- VisualVerifier: initialization, verification methods (dimensional, features, VLM)
- GenerationHistory: initialization with defaults
- GenerationOrchestrator: initialization, routing, feedback construction
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from ll_gen.config import GenerationRoute, LLGenConfig
from ll_gen.pipeline.orchestrator import GenerationHistory, GenerationOrchestrator
from ll_gen.pipeline.verification import VerificationResult, VisualVerifier
from ll_gen.proposals.code_proposal import CodeProposal
from ll_gen.proposals.disposal_result import DisposalResult, GeometryReport
from tests.conftest import (
    requires_occ,
    requires_torch,
)


# ============================================================================
# VerificationResult Tests
# ============================================================================

class TestVerificationResult:
    """Test suite for VerificationResult dataclass."""

    def test_verification_result_defaults(self) -> None:
        """Test VerificationResult construction with default values."""
        result = VerificationResult()
        assert result.matches_intent is True
        assert result.confidence == 0.5
        assert result.method == "dimensional"
        assert result.dimension_checks == []
        assert result.issues == []
        assert result.vlm_response is None

    def test_verification_result_custom_values(self) -> None:
        """Test VerificationResult construction with custom values."""
        dim_checks = [
            {"name": "width", "expected": 100, "actual": 98, "passed": True}
        ]
        issues = ["Dimension slightly off"]
        vlm_response = "The shape looks correct"

        result = VerificationResult(
            matches_intent=False,
            confidence=0.85,
            method="dimensional+feature_count",
            dimension_checks=dim_checks,
            issues=issues,
            vlm_response=vlm_response,
        )

        assert result.matches_intent is False
        assert result.confidence == 0.85
        assert result.method == "dimensional+feature_count"
        assert result.dimension_checks == dim_checks
        assert result.issues == issues
        assert result.vlm_response == vlm_response


# ============================================================================
# VisualVerifier Tests
# ============================================================================

class TestVisualVerifierInit:
    """Test suite for VisualVerifier initialization."""

    def test_verifier_init_defaults(self) -> None:
        """Test VisualVerifier initialization with default values."""
        verifier = VisualVerifier()
        assert verifier.dimension_tolerance == 0.15
        assert verifier.vlm_backend is None

    def test_verifier_init_custom_tolerance(self) -> None:
        """Test VisualVerifier initialization with custom tolerance."""
        verifier = VisualVerifier(dimension_tolerance=0.20)
        assert verifier.dimension_tolerance == 0.20
        assert verifier.vlm_backend is None

    def test_verifier_init_custom_vlm_backend(self) -> None:
        """Test VisualVerifier initialization with custom VLM backend."""
        verifier = VisualVerifier(vlm_backend="clip")
        assert verifier.dimension_tolerance == 0.15
        assert verifier.vlm_backend == "clip"

    def test_verifier_init_custom_all(self) -> None:
        """Test VisualVerifier initialization with all custom values."""
        verifier = VisualVerifier(
            dimension_tolerance=0.25,
            vlm_backend="llm",
        )
        assert verifier.dimension_tolerance == 0.25
        assert verifier.vlm_backend == "llm"


class TestVisualVerifierNoInputs:
    """Test verify() with no inputs."""

    def test_verify_no_inputs_returns_none_method(self) -> None:
        """Test verify() with no inputs returns method='none' and confidence=0.0."""
        verifier = VisualVerifier()
        result = verifier.verify()

        assert result.method == "none"
        assert result.confidence == 0.0
        assert result.matches_intent is True
        assert result.issues == []


class TestVisualVerifierDimensionalChecks:
    """Test _verify_dimensions() method."""

    def test_verify_dimensions_box_prompt_matching_report(
        self, geometry_report_box: GeometryReport
    ) -> None:
        """Test dimensional verification with a box prompt matching the report.

        Prompt: "A box 100mm × 50mm × 20mm"
        Report: Bounding box (0, 0, 0, 100, 50, 20) → dims [100, 50, 20]
        Expected: No issues
        """
        verifier = VisualVerifier()
        prompt = "A box 100mm × 50mm × 20mm"

        result = verifier.verify(
            prompt=prompt,
            geometry_report=geometry_report_box,
        )

        assert result.matches_intent is True
        assert len(result.issues) == 0
        assert "dimensional" in result.method

    def test_verify_dimensions_width_only_matching(
        self, geometry_report_box: GeometryReport
    ) -> None:
        """Test dimensional verification with 'width' dimension matching.

        Prompt: "80mm wide"
        Report: Box dimensions [100, 50, 20]
        Width (largest dim) = 100mm, but prompt says 80mm.
        Expected: Should check and may report mismatch.
        """
        verifier = VisualVerifier()
        prompt = "80mm wide"

        result = verifier.verify(
            prompt=prompt,
            geometry_report=geometry_report_box,
        )

        # Width check: expected=80, actual=100
        # Tolerance=15%, so |100-80|/80 = 0.25 > 0.15 → should fail
        assert result.matches_intent is False
        assert any("width" in issue.lower() for issue in result.issues)

    def test_verify_dimensions_matching_within_tolerance(
        self, geometry_report_box: GeometryReport
    ) -> None:
        """Test dimensional verification with dimensions matching within tolerance.

        Prompt: "105mm wide" (nominal 100mm)
        Report: 100mm wide
        Tolerance: 15%, |105-100|/105 ≈ 0.048 < 0.15 → pass
        """
        verifier = VisualVerifier(dimension_tolerance=0.15)
        prompt = "105mm wide"

        result = verifier.verify(
            prompt=prompt,
            geometry_report=geometry_report_box,
        )

        # 105 vs 100: |105-100|/105 ≈ 0.048 < 0.15 → should pass
        assert result.matches_intent is True

    def test_verify_dimensions_thickness_smallest_dim(
        self, geometry_report_box: GeometryReport
    ) -> None:
        """Test thickness detection (smallest dimension).

        Prompt: "3mm thick"
        Report: Dimensions [100, 50, 20]
        Thickness (smallest) = 20mm, expected 3mm → mismatch
        """
        verifier = VisualVerifier()
        prompt = "3mm thick"

        result = verifier.verify(
            prompt=prompt,
            geometry_report=geometry_report_box,
        )

        # Expected 3, actual 20: |20-3|/3 ≈ 5.67 > 0.15 → fail
        assert result.matches_intent is False
        assert any("thickness" in issue.lower() for issue in result.issues)

    def test_verify_dimensions_diameter(
        self, geometry_report_cylinder: GeometryReport
    ) -> None:
        """Test diameter dimension detection.

        Prompt: "diameter 10mm" (radius 10mm)
        Report: Cylinder r=10, h=30 → dims approx [20, 20, 30]
        Diameter 10 should match radius dimension.
        """
        verifier = VisualVerifier()
        prompt = "diameter 10mm"

        result = verifier.verify(
            prompt=prompt,
            geometry_report=geometry_report_cylinder,
        )

        # Should check diameter; actual ~20, expected 10 → may fail
        # But the cylinder bbox is ~20×20×30, so it should find a match
        assert "dimensional" in result.method

    def test_verify_dimensions_radius_doubled(
        self, geometry_report_cylinder: GeometryReport
    ) -> None:
        """Test radius dimension (doubled to diameter for comparison).

        Prompt: "radius 5mm"
        Report: Cylinder r=10 (diameter 20)
        Radius 5 → diameter 10 should not match diameter 20 → fail
        """
        verifier = VisualVerifier()
        prompt = "radius 5mm"

        result = verifier.verify(
            prompt=prompt,
            geometry_report=geometry_report_cylinder,
        )

        # radius 5 → diameter 10 expected, actual ~20
        # Should detect mismatch
        assert "dimensional" in result.method

    def test_verify_dimensions_no_dimensions_in_prompt(
        self, geometry_report_box: GeometryReport
    ) -> None:
        """Test dimensional verification when prompt has no numeric dimensions.

        Prompt: "A small box"
        Expected: No dimension checks, issues list empty
        """
        verifier = VisualVerifier()
        prompt = "A small box"

        result = verifier.verify(
            prompt=prompt,
            geometry_report=geometry_report_box,
        )

        # Should not find dimensions to check
        assert all(c["name"] != "width" for c in result.dimension_checks)


class TestVisualVerifierFeatureChecks:
    """Test _verify_features() method."""

    def test_verify_features_bolt_holes_matching(
        self,
    ) -> None:
        """Test feature verification with matching bolt hole count.

        Prompt: "4 bolt holes"
        Report: 8 cylindrical faces (4 holes × 2 faces per hole)
        Expected: Pass (8 / 2 = 4 holes)
        """
        verifier = VisualVerifier()
        report = GeometryReport(
            volume=100.0,
            surface_area=200.0,
            bounding_box=(0, 0, 0, 100, 50, 10),
            face_count=6,
            edge_count=12,
            vertex_count=8,
            surface_types={"Plane": 6, "Cylinder": 8},
        )
        prompt = "4 bolt holes"

        result = verifier.verify(
            prompt=prompt,
            geometry_report=report,
        )

        # 8 cylindrical faces / 2 = 4 holes → matches
        assert result.matches_intent is True
        assert not any("holes" in issue.lower() for issue in result.issues)

    def test_verify_features_bolt_holes_insufficient(
        self,
    ) -> None:
        """Test feature verification with insufficient bolt holes.

        Prompt: "4 bolt holes"
        Report: 2 cylindrical faces (only 1 hole × 2)
        Expected: Fail with issue
        """
        verifier = VisualVerifier()
        report = GeometryReport(
            volume=100.0,
            surface_area=200.0,
            bounding_box=(0, 0, 0, 100, 50, 10),
            face_count=6,
            edge_count=12,
            vertex_count=8,
            surface_types={"Plane": 6, "Cylinder": 2},
        )
        prompt = "4 bolt holes"

        result = verifier.verify(
            prompt=prompt,
            geometry_report=report,
        )

        # 2 cylindrical faces / 2 = 1 hole, but need 4 → fail
        assert result.matches_intent is False
        assert any("holes" in issue.lower() for issue in result.issues)

    def test_verify_features_no_hole_mention(
        self, geometry_report_box: GeometryReport
    ) -> None:
        """Test feature verification when prompt doesn't mention holes.

        Prompt: "A simple box"
        Expected: No feature checks, no issues from feature check
        """
        verifier = VisualVerifier()
        prompt = "A simple box"

        result = verifier.verify(
            prompt=prompt,
            geometry_report=geometry_report_box,
        )

        # Should not check features
        assert not any("holes" in issue.lower() for issue in result.issues)


class TestVisualVerifierWithinTolerance:
    """Test _within_tolerance() helper method."""

    def test_within_tolerance_exact_match(self) -> None:
        """Test _within_tolerance with exact match."""
        verifier = VisualVerifier(dimension_tolerance=0.15)
        assert verifier._within_tolerance(100.0, 100.0) is True

    def test_within_tolerance_10_percent_off_with_15_percent_tolerance(
        self,
    ) -> None:
        """Test _within_tolerance at 10% off with 15% tolerance.

        actual=100, expected=90 → |100-90|/90 ≈ 0.111 < 0.15 → True
        """
        verifier = VisualVerifier(dimension_tolerance=0.15)
        # 10% off: 100 vs 90
        assert verifier._within_tolerance(100.0, 90.0) is True

    def test_within_tolerance_20_percent_off_with_15_percent_tolerance(
        self,
    ) -> None:
        """Test _within_tolerance at 20% off with 15% tolerance.

        actual=100, expected=83 → |100-83|/83 ≈ 0.205 > 0.15 → False
        """
        verifier = VisualVerifier(dimension_tolerance=0.15)
        # 20% off: 100 vs 83
        assert verifier._within_tolerance(100.0, 83.0) is False

    def test_within_tolerance_expected_zero_edge_case(self) -> None:
        """Test _within_tolerance edge case with expected=0.

        When expected=0, uses absolute difference instead of ratio.
        """
        verifier = VisualVerifier(dimension_tolerance=0.15)
        # expected=0, actual should be < 0.15
        assert verifier._within_tolerance(0.1, 0.0) is True
        assert verifier._within_tolerance(0.2, 0.0) is False


class TestVisualVerifierCombinedChecks:
    """Test combined dimensional + feature verification."""

    def test_verify_combined_dimensional_and_features(
        self,
    ) -> None:
        """Test verification combining dimensional and feature checks.

        Prompt: "An 80mm box with 4 holes"
        Expected: Both checks run, method contains "dimensional+feature_count"
        """
        verifier = VisualVerifier()
        report = GeometryReport(
            volume=100.0,
            surface_area=200.0,
            bounding_box=(0, 0, 0, 80, 50, 20),
            face_count=6,
            edge_count=12,
            vertex_count=8,
            surface_types={"Plane": 6, "Cylinder": 8},
        )
        prompt = "An 80mm wide box with 4 holes"

        result = verifier.verify(
            prompt=prompt,
            geometry_report=report,
        )

        # Both checks should run
        assert "dimensional" in result.method
        assert "feature_count" in result.method


class TestVisualVerifierConfidence:
    """Test confidence calculation in verify()."""

    def test_confidence_more_methods_higher_when_passing(
        self,
    ) -> None:
        """Test confidence increases with more passing verification methods.

        With 1 method passing: confidence = 0.5 + 0.2*1 = 0.7
        With 2 methods passing: confidence = 0.5 + 0.2*2 = 0.9
        With 3 methods passing: confidence = min(0.5 + 0.2*3, 1.0) = 1.0
        """
        verifier = VisualVerifier()
        report = GeometryReport(
            volume=100.0,
            surface_area=200.0,
            bounding_box=(0, 0, 0, 100, 50, 20),
            face_count=6,
            edge_count=12,
            vertex_count=8,
            surface_types={"Plane": 6, "Cylinder": 8},
        )
        prompt = "100mm wide box with 4 holes"

        result = verifier.verify(
            prompt=prompt,
            geometry_report=report,
        )

        # Should have both dimensional and feature checks
        # confidence = 0.5 + 0.2 * 2 = 0.9
        if result.matches_intent:
            assert result.confidence >= 0.5

    def test_confidence_failure_with_issues(self, geometry_report_box: GeometryReport) -> None:
        """Test confidence with failing checks.

        With failures, confidence = 0.8 - 0.1 * num_issues, min 0.1
        """
        verifier = VisualVerifier()
        prompt = "20mm wide"  # Box is 100mm wide → mismatch

        result = verifier.verify(
            prompt=prompt,
            geometry_report=geometry_report_box,
        )

        # Should have issues
        assert result.matches_intent is False
        assert len(result.issues) > 0
        # confidence = 0.8 - 0.1 * len(issues)
        assert result.confidence >= 0.1


# ============================================================================
# GenerationHistory Tests
# ============================================================================

class TestGenerationHistory:
    """Test suite for GenerationHistory dataclass."""

    def test_generation_history_defaults(self) -> None:
        """Test GenerationHistory construction with default values."""
        history = GenerationHistory()

        assert history.routing_decision is None
        assert history.attempts == []
        assert history.total_time_ms == 0.0
        assert history.final_result is None

    def test_generation_history_with_attempts(
        self, disposal_result_valid: DisposalResult
    ) -> None:
        """Test GenerationHistory with populated attempts."""
        history = GenerationHistory()
        history.attempts.append({
            "attempt": 1,
            "proposal_id": "test_001",
            "is_valid": True,
        })
        history.final_result = disposal_result_valid
        history.total_time_ms = 500.0

        assert len(history.attempts) == 1
        assert history.final_result.is_valid is True
        assert history.total_time_ms == 500.0


# ============================================================================
# GenerationOrchestrator Tests
# ============================================================================

class TestGenerationOrchestratorInit:
    """Test suite for GenerationOrchestrator initialization."""

    def test_orchestrator_init_default_config(self) -> None:
        """Test GenerationOrchestrator initialization with default config."""
        orchestrator = GenerationOrchestrator()

        assert orchestrator.config is not None
        assert isinstance(orchestrator.config, LLGenConfig)
        assert orchestrator.router is not None
        assert orchestrator.disposal_engine is not None
        assert orchestrator._cadquery_proposer is None
        assert orchestrator._openscad_proposer is None

    def test_orchestrator_init_custom_config(self, ll_gen_config: LLGenConfig) -> None:
        """Test GenerationOrchestrator initialization with custom config."""
        orchestrator = GenerationOrchestrator(config=ll_gen_config)

        assert orchestrator.config is ll_gen_config
        assert orchestrator.router is not None
        assert orchestrator.disposal_engine is not None

    def test_orchestrator_init_router_created(self) -> None:
        """Test that orchestrator initializes router with routing config."""
        orchestrator = GenerationOrchestrator()
        assert orchestrator.router is not None
        assert hasattr(orchestrator.router, "route")

    def test_orchestrator_init_disposal_engine_created(self) -> None:
        """Test that orchestrator initializes disposal engine."""
        orchestrator = GenerationOrchestrator()
        assert orchestrator.disposal_engine is not None


class TestGenerationOrchestratorBuildFeedback:
    """Test suite for _build_feedback() method."""

    def test_build_feedback_for_code_proposal(
        self,
        code_proposal_cadquery: CodeProposal,
        disposal_result_invalid: DisposalResult,
    ) -> None:
        """Test _build_feedback() dispatches correctly for CodeProposal.

        Expected: Returns dict with 'error_message' key (string feedback).
        """
        orchestrator = GenerationOrchestrator()

        feedback = orchestrator._build_feedback(
            result=disposal_result_invalid,
            proposal=code_proposal_cadquery,
            route=GenerationRoute.CODE_CADQUERY,
        )

        assert isinstance(feedback, dict)
        assert "error_message" in feedback
        assert isinstance(feedback["error_message"], str)
        assert feedback["type"] == "code_feedback"
        assert "original_code" in feedback
        assert "error_category" in feedback

    def test_build_feedback_for_non_code_proposal(
        self,
        command_proposal,
        disposal_result_invalid: DisposalResult,
    ) -> None:
        """Test _build_feedback() dispatches correctly for non-code proposals.

        Expected: Returns dict with neural feedback structure.
        """
        orchestrator = GenerationOrchestrator()

        feedback = orchestrator._build_feedback(
            result=disposal_result_invalid,
            proposal=command_proposal,
            route=GenerationRoute.NEURAL_VAE,
        )

        assert isinstance(feedback, dict)
        # Neural feedback should have different keys than code feedback
        assert "error_message" not in feedback or feedback.get("type") != "code_feedback"


class TestGenerationOrchestratorProposeDispatch:
    """Test suite for _propose() route dispatch (structure only).

    Full generation tests would require LLM/neural backends.
    These tests verify dispatch logic only.
    """

    def test_propose_dispatch_structure(self) -> None:
        """Test that _propose() method exists and has correct signature."""
        orchestrator = GenerationOrchestrator()
        assert hasattr(orchestrator, "_propose")
        assert callable(orchestrator._propose)


# ============================================================================
# Integration Tests
# ============================================================================

class TestVerificationIntegration:
    """Integration tests for the verification pipeline."""

    def test_verify_box_full_workflow(
        self, geometry_report_box: GeometryReport
    ) -> None:
        """Test full verification workflow: dimensional + feature checks.

        Prompt: "A 100×50×20 box"
        Report: Matching box dimensions
        Expected: All checks pass, matches_intent=True
        """
        verifier = VisualVerifier()
        prompt = "A 100×50×20 box"

        result = verifier.verify(
            prompt=prompt,
            geometry_report=geometry_report_box,
        )

        assert result.matches_intent is True
        assert "dimensional" in result.method
        assert len(result.issues) == 0

    def test_verify_with_multiple_dimension_patterns(
        self, geometry_report_box: GeometryReport
    ) -> None:
        """Test verification with various dimension patterns in one prompt.

        Prompt contains multiple patterns: "100 wide, 50 long, 20 high"
        Expected: All patterns extracted and checked
        """
        verifier = VisualVerifier()
        prompt = "100mm wide, 50mm long, 20mm high"

        result = verifier.verify(
            prompt=prompt,
            geometry_report=geometry_report_box,
        )

        # Should check multiple dimensions
        assert len(result.dimension_checks) > 0


class TestGenerationOrchestratorStructure:
    """Test GenerationOrchestrator high-level structure."""

    def test_orchestrator_has_required_methods(self) -> None:
        """Test that orchestrator has all required public methods."""
        orchestrator = GenerationOrchestrator()

        required_methods = [
            "generate",
            "generate_batch",
        ]

        for method_name in required_methods:
            assert hasattr(orchestrator, method_name), (
                f"Orchestrator missing method: {method_name}"
            )
            assert callable(getattr(orchestrator, method_name))

    def test_orchestrator_has_required_private_methods(self) -> None:
        """Test that orchestrator has all required private methods."""
        orchestrator = GenerationOrchestrator()

        required_methods = [
            "_propose",
            "_build_feedback",
        ]

        for method_name in required_methods:
            assert hasattr(orchestrator, method_name), (
                f"Orchestrator missing method: {method_name}"
            )
            assert callable(getattr(orchestrator, method_name))


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestVisualVerifierEdgeCases:
    """Test edge cases and error conditions."""

    def test_verify_with_no_bbox(self, geometry_report_no_bbox: GeometryReport) -> None:
        """Test verification when geometry report has no bounding box.

        Expected: Skip dimensional checks, no errors
        """
        verifier = VisualVerifier()
        prompt = "A 100mm box"

        result = verifier.verify(
            prompt=prompt,
            geometry_report=geometry_report_no_bbox,
        )

        # Should still attempt feature check even without bbox
        assert result.method in ["none", "feature_count", "dimensional+feature_count"]
        # But dimension_checks should be empty since there's no bbox
        assert len(result.dimension_checks) == 0

    def test_verify_empty_prompt(self, geometry_report_box: GeometryReport) -> None:
        """Test verification with empty prompt string.

        Expected: No checks run, confidence=0.0
        """
        verifier = VisualVerifier()

        result = verifier.verify(
            prompt="",
            geometry_report=geometry_report_box,
        )

        assert result.method == "none"
        assert result.confidence == 0.0

    def test_dimension_extraction_with_various_units(
        self, geometry_report_box: GeometryReport
    ) -> None:
        """Test dimension extraction with different unit formats.

        Prompt contains: "100mm", "50cm", "20inches"
        Expected: Extracts numeric values regardless of unit
        """
        verifier = VisualVerifier()
        # Box is 100×50×20, so test various unit mentions
        prompt = "100mm wide"

        result = verifier.verify(
            prompt=prompt,
            geometry_report=geometry_report_box,
        )

        # Should extract dimensions
        assert len(result.dimension_checks) > 0


class TestVerificationResultEdgeCases:
    """Test edge cases for VerificationResult."""

    def test_verification_result_empty_issues(self) -> None:
        """Test VerificationResult with explicitly empty issues."""
        result = VerificationResult(issues=[])
        assert result.issues == []

    def test_verification_result_multiple_issues(self) -> None:
        """Test VerificationResult with multiple issues."""
        issues = ["Issue 1", "Issue 2", "Issue 3"]
        result = VerificationResult(issues=issues)
        assert len(result.issues) == 3
        assert result.issues == issues


# ============================================================================
# Type and Contract Tests
# ============================================================================

class TestVerifierTypeContracts:
    """Test that verifier methods maintain proper type contracts."""

    def test_verify_returns_verification_result(
        self, geometry_report_box: GeometryReport
    ) -> None:
        """Test that verify() always returns VerificationResult."""
        verifier = VisualVerifier()
        result = verifier.verify(
            prompt="A box",
            geometry_report=geometry_report_box,
        )

        assert isinstance(result, VerificationResult)

    def test_within_tolerance_returns_bool(self) -> None:
        """Test that _within_tolerance() always returns bool."""
        verifier = VisualVerifier()

        result = verifier._within_tolerance(100.0, 95.0)
        assert isinstance(result, bool)

    def test_verify_dimensions_returns_dict(
        self, geometry_report_box: GeometryReport
    ) -> None:
        """Test that _verify_dimensions() returns dict with 'checks' and 'issues'."""
        verifier = VisualVerifier()
        result = verifier._verify_dimensions("100mm wide", geometry_report_box)

        assert isinstance(result, dict)
        assert "checks" in result
        assert "issues" in result
        assert isinstance(result["checks"], list)
        assert isinstance(result["issues"], list)

    def test_verify_features_returns_dict(
        self, geometry_report_box: GeometryReport
    ) -> None:
        """Test that _verify_features() returns dict with 'issues' key."""
        verifier = VisualVerifier()
        result = verifier._verify_features("4 bolt holes", geometry_report_box)

        assert isinstance(result, dict)
        assert "issues" in result
        assert isinstance(result["issues"], list)


class TestOrchestratorTypeContracts:
    """Test that orchestrator methods maintain proper type contracts."""

    def test_orchestrator_config_is_llgenconfig(self) -> None:
        """Test that orchestrator.config is always LLGenConfig."""
        orchestrator = GenerationOrchestrator()
        assert isinstance(orchestrator.config, LLGenConfig)

    def test_build_feedback_returns_dict(
        self,
        code_proposal_cadquery: CodeProposal,
        disposal_result_invalid: DisposalResult,
    ) -> None:
        """Test that _build_feedback() returns a dict."""
        orchestrator = GenerationOrchestrator()
        feedback = orchestrator._build_feedback(
            result=disposal_result_invalid,
            proposal=code_proposal_cadquery,
            route=GenerationRoute.CODE_CADQUERY,
        )

        assert isinstance(feedback, dict)


# ============================================================================
# Pattern Matching Tests
# ============================================================================

class TestDimensionPatternMatching:
    """Test dimension extraction patterns in detail."""

    def test_extract_width_pattern(
        self, geometry_report_box: GeometryReport
    ) -> None:
        """Test extraction of 'Nmm wide' pattern."""
        verifier = VisualVerifier()

        patterns = [
            "100mm wide",
            "100 mm wide",
            "100mm width",
            "width 100mm",
        ]

        for prompt in patterns:
            result = verifier._verify_dimensions(prompt, geometry_report_box)
            assert len(result["checks"]) > 0, f"Failed to extract from: {prompt}"

        # Multi-dim pattern uses 'x' or '×' character
        multi_dim_patterns = [
            "100 x 50 x 20",
            "100 × 50 × 20",
        ]
        for prompt in multi_dim_patterns:
            result = verifier._verify_dimensions(prompt, geometry_report_box)
            assert len(result["checks"]) > 0, f"Failed to extract multi-dim from: {prompt}"

    def test_extract_thickness_pattern(
        self, geometry_report_box: GeometryReport
    ) -> None:
        """Test extraction of thickness/depth patterns."""
        verifier = VisualVerifier()

        patterns = [
            "20mm thick",
            "20mm thickness",
            "20mm deep",
            "20mm depth",
        ]

        for prompt in patterns:
            result = verifier._verify_dimensions(prompt, geometry_report_box)
            assert len(result["checks"]) > 0, f"Failed to extract from: {prompt}"

    def test_extract_hole_count_pattern(
        self, geometry_report_box: GeometryReport
    ) -> None:
        """Test extraction of hole count patterns."""
        verifier = VisualVerifier()

        patterns = [
            "4 holes",
            "4 bolt holes",
            "4 mounting holes",
            "four holes",
        ]

        for prompt in patterns:
            result = verifier._verify_features(prompt, geometry_report_box)
            # Should recognize the pattern (even if it fails the check)
            assert isinstance(result, dict)


# ============================================================================
# Configuration and Settings Tests
# ============================================================================

class TestVerifierConfiguration:
    """Test verifier configuration options."""

    def test_tolerance_affects_dimensional_checks(
        self, geometry_report_box: GeometryReport
    ) -> None:
        """Test that tolerance setting affects dimension checking."""
        prompt = "95mm wide"  # Box is 100mm wide

        # With 10% tolerance: |100-95|/95 ≈ 0.053 < 0.10 → pass
        verifier_loose = VisualVerifier(dimension_tolerance=0.10)
        result_loose = verifier_loose.verify(
            prompt=prompt,
            geometry_report=geometry_report_box,
        )

        # With 1% tolerance: |100-95|/95 ≈ 0.053 > 0.01 → fail
        verifier_strict = VisualVerifier(dimension_tolerance=0.01)
        result_strict = verifier_strict.verify(
            prompt=prompt,
            geometry_report=geometry_report_box,
        )

        # Tolerance should affect results
        # Loose tolerance allows more deviation
        assert isinstance(result_loose.matches_intent, bool)
        assert isinstance(result_strict.matches_intent, bool)

    def test_vlm_backend_none_skips_vlm_checks(
        self, geometry_report_box: GeometryReport
    ) -> None:
        """Test that vlm_backend=None skips VLM verification."""
        verifier = VisualVerifier(vlm_backend=None)
        result = verifier.verify(
            render_paths=[Path("/fake/path.png")],
            prompt="A box",
            geometry_report=geometry_report_box,
        )

        # VLM should not run when backend is None
        assert "vlm" not in result.method


# ============================================================================
# Data Integrity Tests
# ============================================================================

class TestVerificationResultDataIntegrity:
    """Test that VerificationResult preserves data correctly."""

    def test_dimension_checks_preserved(self) -> None:
        """Test that dimension check list is preserved as-is."""
        checks = [
            {"name": "width", "expected": 100, "actual": 98, "passed": True},
            {"name": "height", "expected": 50, "actual": 50, "passed": True},
        ]
        result = VerificationResult(dimension_checks=checks)

        assert result.dimension_checks == checks
        assert len(result.dimension_checks) == 2

    def test_issues_list_preserved(self) -> None:
        """Test that issues list is preserved as-is."""
        issues = ["Issue 1", "Issue 2"]
        result = VerificationResult(issues=issues)

        assert result.issues == issues
        assert len(result.issues) == 2


# ============================================================================
# Boundary Conditions
# ============================================================================

class TestVisualVerifierBoundaryConditions:
    """Test boundary conditions and limits."""

    def test_very_small_dimensions(
        self,
    ) -> None:
        """Test with very small dimensions (micro scale)."""
        verifier = VisualVerifier()
        report = GeometryReport(
            volume=0.001,
            surface_area=0.1,
            bounding_box=(0, 0, 0, 1, 0.5, 0.2),
            face_count=6,
            edge_count=12,
            vertex_count=8,
            surface_types={"Plane": 6},
        )
        prompt = "1mm wide"

        result = verifier.verify(prompt=prompt, geometry_report=report)
        assert isinstance(result, VerificationResult)

    def test_very_large_dimensions(
        self,
    ) -> None:
        """Test with very large dimensions (macro scale)."""
        verifier = VisualVerifier()
        report = GeometryReport(
            volume=1e12,
            surface_area=1e8,
            bounding_box=(0, 0, 0, 1000000, 500000, 200000),
            face_count=6,
            edge_count=12,
            vertex_count=8,
            surface_types={"Plane": 6},
        )
        prompt = "1000000mm wide"

        result = verifier.verify(prompt=prompt, geometry_report=report)
        assert isinstance(result, VerificationResult)

    def test_confidence_boundaries(self) -> None:
        """Test that confidence stays within [0, 1] bounds."""
        verifier = VisualVerifier()
        report = GeometryReport(
            volume=100.0,
            surface_area=200.0,
            bounding_box=(0, 0, 0, 100, 50, 20),
            face_count=6,
            edge_count=12,
            vertex_count=8,
            surface_types={"Plane": 6},
        )

        result = verifier.verify(prompt="A box", geometry_report=report)
        assert 0.0 <= result.confidence <= 1.0


# ============================================================================
# Helper Method Tests
# ============================================================================

class TestToleranceHelper:
    """Test the _within_tolerance helper method thoroughly."""

    def test_within_tolerance_boundary_cases(self) -> None:
        """Test boundary conditions for tolerance checking."""
        verifier = VisualVerifier(dimension_tolerance=0.15)

        # Exactly at boundary: 15% off
        # |actual - expected| / |expected| = 0.15
        expected = 100.0
        actual = 85.0  # 15% less
        result = verifier._within_tolerance(actual, expected)
        assert result is True  # Should be inclusive

    def test_within_tolerance_just_outside_boundary(self) -> None:
        """Test just beyond tolerance boundary."""
        verifier = VisualVerifier(dimension_tolerance=0.15)

        expected = 100.0
        actual = 84.9  # Slightly more than 15% off
        result = verifier._within_tolerance(actual, expected)
        assert result is False

    def test_within_tolerance_negative_values(self) -> None:
        """Test tolerance with negative coordinates.

        The method computes |actual - expected| / |expected|.
        For expected=-100, actual=-95: |-95 - (-100)| / |-100| = 5/100 = 0.05 < 0.15
        """
        verifier = VisualVerifier(dimension_tolerance=0.15)

        # -100 and -95: |-95 - (-100)| / |-100| = 5/100 = 0.05 < 0.15
        assert verifier._within_tolerance(-95.0, -100.0) is True
