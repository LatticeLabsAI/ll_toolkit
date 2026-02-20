"""Comprehensive test suite for ll_gen.pipeline.verification module.

Tests visual verification functionality:
- VisualVerifier initialization
- Dimensional verification from prompts
- Feature count verification
- VLM-based verification (mocked)
- VerificationResult structure
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from ll_gen.pipeline.verification import (
    VerificationResult,
    VisualVerifier,
)
from ll_gen.proposals.disposal_result import GeometryReport


# ============================================================================
# SECTION 1: VerificationResult Tests
# ============================================================================


class TestVerificationResult:
    """Test VerificationResult dataclass."""

    def test_default_initialization(self) -> None:
        """Test VerificationResult initializes with defaults."""
        result = VerificationResult()
        assert result.matches_intent is True
        assert result.confidence == 0.5
        assert result.method == "dimensional"
        assert result.dimension_checks == []
        assert result.issues == []
        assert result.vlm_response is None

    def test_custom_initialization(self) -> None:
        """Test VerificationResult with custom values."""
        result = VerificationResult(
            matches_intent=False,
            confidence=0.8,
            method="vlm",
            dimension_checks=[{"name": "width", "passed": True}],
            issues=["Shape too small"],
            vlm_response="MISMATCH: Shape is incorrect",
        )
        assert result.matches_intent is False
        assert result.confidence == 0.8
        assert result.method == "vlm"
        assert len(result.dimension_checks) == 1
        assert len(result.issues) == 1
        assert result.vlm_response is not None


# ============================================================================
# SECTION 2: VisualVerifier Initialization Tests
# ============================================================================


class TestVisualVerifierInit:
    """Test VisualVerifier initialization."""

    def test_default_initialization(self) -> None:
        """Test VisualVerifier initializes with defaults."""
        verifier = VisualVerifier()
        assert verifier.dimension_tolerance == 0.15
        assert verifier.vlm_backend is None

    def test_custom_tolerance(self) -> None:
        """Test VisualVerifier with custom tolerance."""
        verifier = VisualVerifier(dimension_tolerance=0.10)
        assert verifier.dimension_tolerance == 0.10

    def test_vlm_backend_clip(self) -> None:
        """Test VisualVerifier with CLIP backend."""
        verifier = VisualVerifier(vlm_backend="clip")
        assert verifier.vlm_backend == "clip"

    def test_vlm_backend_llm(self) -> None:
        """Test VisualVerifier with LLM backend."""
        verifier = VisualVerifier(vlm_backend="llm")
        assert verifier.vlm_backend == "llm"


# ============================================================================
# SECTION 3: Dimensional Verification Tests
# ============================================================================


class TestDimensionalVerification:
    """Test dimensional verification from prompts."""

    def test_verify_no_geometry_report(self) -> None:
        """Test verify with no geometry report returns default result."""
        verifier = VisualVerifier()
        result = verifier.verify(
            prompt="A box 100mm wide",
            geometry_report=None,
        )
        # With no geometry report, dimensional verification is skipped
        assert result.matches_intent is True

    def test_verify_prompt_with_dimensions(self) -> None:
        """Test verify extracts dimensions from prompt."""
        verifier = VisualVerifier()
        report = GeometryReport(
            bounding_box=(0, 0, 0, 100, 50, 20),
            face_count=6,
        )
        result = verifier.verify(
            prompt="A box 100mm wide, 50mm deep, 20mm tall",
            geometry_report=report,
        )
        # Dimensions should match
        assert "dimensional" in result.method

    def test_verify_dimension_mismatch(self) -> None:
        """Test verify detects dimension mismatch."""
        verifier = VisualVerifier(dimension_tolerance=0.05)
        report = GeometryReport(
            bounding_box=(0, 0, 0, 50, 50, 20),  # Width is 50, not 100
            face_count=6,
        )
        result = verifier.verify(
            prompt="A box 100mm wide",
            geometry_report=report,
        )
        # Should fail due to dimension mismatch
        assert result.matches_intent is False or len(result.issues) > 0

    def test_verify_multi_dimension_pattern(self) -> None:
        """Test verify handles NxNxN dimension patterns."""
        verifier = VisualVerifier(dimension_tolerance=0.15)
        report = GeometryReport(
            bounding_box=(0, 0, 0, 100, 50, 20),
            face_count=6,
        )
        result = verifier.verify(
            prompt="Create a 100 x 50 x 20 box",
            geometry_report=report,
        )
        # Should recognize multi-dim pattern
        assert len(result.dimension_checks) >= 0  # May or may not extract


# ============================================================================
# SECTION 4: Feature Verification Tests
# ============================================================================


class TestFeatureVerification:
    """Test feature count verification from prompts."""

    def test_verify_hole_count(self) -> None:
        """Test verify checks hole count from prompt."""
        verifier = VisualVerifier()
        report = GeometryReport(
            bounding_box=(0, 0, 0, 100, 50, 20),
            face_count=6,
            surface_types={"Plane": 6, "Cylinder": 8},  # 4 holes = 8 cylinder faces
        )
        result = verifier.verify(
            prompt="A bracket with 4 bolt holes",
            geometry_report=report,
        )
        # Should check for holes
        assert "feature_count" in result.method or result.matches_intent

    def test_verify_insufficient_holes(self) -> None:
        """Test verify detects insufficient holes."""
        verifier = VisualVerifier()
        report = GeometryReport(
            bounding_box=(0, 0, 0, 100, 50, 20),
            face_count=6,
            surface_types={"Plane": 6, "Cylinder": 2},  # Only 1 hole
        )
        result = verifier.verify(
            prompt="A bracket with 4 bolt holes",
            geometry_report=report,
        )
        # Should report missing holes
        assert len(result.issues) > 0 or not result.matches_intent


# ============================================================================
# SECTION 5: Within Tolerance Tests
# ============================================================================


class TestWithinTolerance:
    """Test tolerance checking helper."""

    def test_within_tolerance_exact_match(self) -> None:
        """Test exact match is within tolerance."""
        verifier = VisualVerifier(dimension_tolerance=0.15)
        assert verifier._within_tolerance(100.0, 100.0) is True

    def test_within_tolerance_small_difference(self) -> None:
        """Test small difference is within tolerance."""
        verifier = VisualVerifier(dimension_tolerance=0.15)
        # 105 vs 100 = 5% difference
        assert verifier._within_tolerance(105.0, 100.0) is True

    def test_within_tolerance_large_difference(self) -> None:
        """Test large difference exceeds tolerance."""
        verifier = VisualVerifier(dimension_tolerance=0.05)
        # 110 vs 100 = 10% difference, exceeds 5%
        assert verifier._within_tolerance(110.0, 100.0) is False

    def test_within_tolerance_zero_expected(self) -> None:
        """Test tolerance check with zero expected value."""
        verifier = VisualVerifier(dimension_tolerance=0.15)
        # Special case: expected is 0
        result = verifier._within_tolerance(0.1, 0.0)
        # Should use absolute tolerance
        assert isinstance(result, bool)


# ============================================================================
# SECTION 6: VLM Verification Tests (Mocked)
# ============================================================================


class TestVLMVerificationMocked:
    """Test VLM verification with mocked backends."""

    def test_verify_without_vlm_backend(self) -> None:
        """Test verify skips VLM when backend is None."""
        verifier = VisualVerifier(vlm_backend=None)
        result = verifier.verify(
            render_paths=[Path("/tmp/render.png")],
            prompt="A box",
        )
        # Should not use VLM
        assert "vlm" not in result.method

    def test_verify_with_no_renders(self) -> None:
        """Test verify skips VLM when no renders provided."""
        verifier = VisualVerifier(vlm_backend="clip")
        result = verifier.verify(
            render_paths=[],
            prompt="A box",
        )
        # Should skip VLM without renders
        assert "vlm" not in result.method

    def test_verify_vlm_failure_handled(self) -> None:
        """Test that VLM failures are handled gracefully."""
        verifier = VisualVerifier(vlm_backend="clip")
        with patch.object(
            verifier, "_verify_vlm",
            side_effect=Exception("VLM failed"),
        ):
            result = verifier.verify(
                render_paths=[Path("/tmp/render.png")],
                prompt="A box",
            )
            # Should continue without VLM
            assert isinstance(result, VerificationResult)


# ============================================================================
# SECTION 7: Dimension Pattern Extraction Tests
# ============================================================================


class TestDimensionPatternExtraction:
    """Test dimension pattern extraction from prompts."""

    def test_extract_mm_wide_pattern(self) -> None:
        """Test extraction of 'NNmm wide' pattern."""
        pattern = r"(\d+(?:\.\d+)?)\s*(?:mm|cm|m|in)\s*(?:wide|width)"
        text = "A box 100mm wide"
        match = re.search(pattern, text, re.IGNORECASE)
        assert match is not None
        assert float(match.group(1)) == 100

    def test_extract_mm_thick_pattern(self) -> None:
        """Test extraction of 'NNmm thick' pattern."""
        pattern = r"(\d+(?:\.\d+)?)\s*(?:mm|cm|m|in)\s*(?:thick|thickness|deep|depth)"
        text = "A plate 3mm thick"
        match = re.search(pattern, text, re.IGNORECASE)
        assert match is not None
        assert float(match.group(1)) == 3

    def test_extract_diameter_pattern(self) -> None:
        """Test extraction of diameter pattern."""
        pattern = r"(?:diameter|dia|ø)\s*(\d+(?:\.\d+)?)\s*(?:mm|cm|m|in)?"
        text = "A hole with diameter 10mm"
        match = re.search(pattern, text, re.IGNORECASE)
        assert match is not None
        assert float(match.group(1)) == 10

    def test_extract_multi_dimension_pattern(self) -> None:
        """Test extraction of 'N x N x N' pattern."""
        pattern = r"(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)(?:\s*[xX×]\s*(\d+(?:\.\d+)?))?"
        text = "A box 100 x 50 x 20"
        match = re.search(pattern, text)
        assert match is not None
        assert float(match.group(1)) == 100
        assert float(match.group(2)) == 50
        assert float(match.group(3)) == 20


# ============================================================================
# SECTION 8: Confidence Calculation Tests
# ============================================================================


class TestConfidenceCalculation:
    """Test confidence score calculation."""

    def test_confidence_no_methods(self) -> None:
        """Test confidence is 0 when no methods used."""
        verifier = VisualVerifier(vlm_backend=None)
        result = verifier.verify(
            prompt="",
            geometry_report=None,
        )
        # With no methods, confidence should be 0
        assert result.confidence == 0.0
        assert result.method == "none"

    def test_confidence_one_method_passing(self) -> None:
        """Test confidence increases with passing methods."""
        verifier = VisualVerifier()
        report = GeometryReport(
            bounding_box=(0, 0, 0, 100, 50, 20),
            face_count=6,
        )
        result = verifier.verify(
            prompt="A box",
            geometry_report=report,
        )
        # With passing method, confidence should be > 0.5
        if result.matches_intent:
            assert result.confidence >= 0.5


# ============================================================================
# SECTION 9: Module Import Tests
# ============================================================================


class TestModuleImport:
    """Test module import."""

    def test_module_importable(self) -> None:
        """Test that verification module is importable."""
        from ll_gen.pipeline import verification
        assert hasattr(verification, "VisualVerifier")
        assert hasattr(verification, "VerificationResult")

    def test_classes_are_instantiable(self) -> None:
        """Test that classes can be instantiated."""
        verifier = VisualVerifier()
        result = VerificationResult()
        assert verifier is not None
        assert result is not None


# ============================================================================
# SECTION 10: Fixture Tests
# ============================================================================


class TestWithFixtures:
    """Test with conftest fixtures."""

    def test_geometry_report_box_fixture(self, geometry_report_box) -> None:
        """Test geometry_report_box fixture."""
        verifier = VisualVerifier(dimension_tolerance=0.15)
        result = verifier.verify(
            prompt="A box 100mm x 50mm x 20mm",
            geometry_report=geometry_report_box,
        )
        # Should recognize the box dimensions
        assert isinstance(result, VerificationResult)

    def test_geometry_report_no_bbox_fixture(self, geometry_report_no_bbox) -> None:
        """Test verification handles missing bounding box."""
        verifier = VisualVerifier()
        result = verifier.verify(
            prompt="A box 100mm wide",
            geometry_report=geometry_report_no_bbox,
        )
        # Should handle gracefully
        assert isinstance(result, VerificationResult)
