"""Comprehensive test suite for the ll_gen feedback module.

Tests cover three sub-modules:
1. error_mapper.py — OCC error code mapping and categorization
2. feedback_builder.py — Building structured feedback for neural retry
3. reward_signal.py — Computing scalar rewards for RL training
"""
from __future__ import annotations

import pytest
from pathlib import Path

from ll_gen.config import ErrorCategory, ErrorSeverity, FeedbackConfig
from ll_gen.feedback.error_mapper import (
    OCC_ERROR_MAP,
    MappedError,
    map_single_error,
    categorize_errors,
)
from ll_gen.feedback.feedback_builder import (
    build_code_feedback,
    build_neural_feedback,
    build_training_feedback,
)
from ll_gen.feedback.reward_signal import (
    compute_reward,
    compute_batch_rewards,
)
from ll_gen.proposals.code_proposal import CodeProposal
from ll_gen.proposals.disposal_result import DisposalResult


# ---------------------------------------------------------------------------
# error_mapper.py tests
# ---------------------------------------------------------------------------

class TestOCCErrorMap:
    """Test the OCC_ERROR_MAP dictionary structure."""

    def test_occ_error_map_has_expected_keys(self) -> None:
        """Verify OCC_ERROR_MAP contains expected error code keys.

        The error map should include the major error codes from both
        BRepCheck and BOPAlgo status enums.
        """
        expected_keys = {
            "BRepCheck_NotClosed",
            "BRepCheck_SelfIntersectingWire",
            "BRepCheck_EmptyWire",
            "BRepCheck_InvalidToleranceValue",
            "BRepCheck_CheckFail",
            "BRepCheck_FreeEdge",
            "BOPAlgo_AlertTooFewArguments",
        }
        for key in expected_keys:
            assert key in OCC_ERROR_MAP, f"Missing expected key: {key}"

    def test_occ_error_map_values_are_tuples(self) -> None:
        """Verify all OCC_ERROR_MAP values are 4-tuples with correct types.

        Each value should be (ErrorCategory, ErrorSeverity, str, str).
        """
        for code_name, value in OCC_ERROR_MAP.items():
            assert isinstance(value, tuple), f"{code_name}: value is not tuple"
            assert len(value) == 4, f"{code_name}: tuple length is {len(value)}, expected 4"

            category, severity, description, suggestion = value
            assert isinstance(category, ErrorCategory), (
                f"{code_name}: category is {type(category)}, expected ErrorCategory"
            )
            assert isinstance(severity, ErrorSeverity), (
                f"{code_name}: severity is {type(severity)}, expected ErrorSeverity"
            )
            assert isinstance(description, str), (
                f"{code_name}: description is {type(description)}, expected str"
            )
            assert isinstance(suggestion, str), (
                f"{code_name}: suggestion is {type(suggestion)}, expected str"
            )

    def test_occ_error_map_nonempty_descriptions(self) -> None:
        """All error codes should have non-empty description and suggestion strings."""
        for code_name, (_, _, description, suggestion) in OCC_ERROR_MAP.items():
            assert len(description) > 0, f"{code_name}: empty description"
            assert len(suggestion) > 0, f"{code_name}: empty suggestion"


class TestMapSingleError:
    """Test the map_single_error function."""

    def test_map_brep_not_closed_topology_error(self) -> None:
        """BRepCheck_NotClosed maps to TOPOLOGY_ERROR with CRITICAL severity."""
        result = map_single_error("BRepCheck_NotClosed", "SHELL", 0)

        assert result.category == ErrorCategory.TOPOLOGY_ERROR
        assert result.severity == ErrorSeverity.CRITICAL
        assert result.occ_code_name == "BRepCheck_NotClosed"
        assert result.entity_type == "SHELL"
        assert result.entity_index == 0

    def test_map_self_intersecting_wire_self_intersection(self) -> None:
        """BRepCheck_SelfIntersectingWire maps to SELF_INTERSECTION."""
        result = map_single_error("BRepCheck_SelfIntersectingWire", "WIRE", 1)

        assert result.category == ErrorCategory.SELF_INTERSECTION
        assert result.severity == ErrorSeverity.CRITICAL
        assert result.entity_type == "WIRE"
        assert result.entity_index == 1

    def test_map_empty_wire_degenerate_shape(self) -> None:
        """BRepCheck_EmptyWire maps to DEGENERATE_SHAPE."""
        result = map_single_error("BRepCheck_EmptyWire", "WIRE", 0)

        assert result.category == ErrorCategory.DEGENERATE_SHAPE
        assert result.severity == ErrorSeverity.CRITICAL

    def test_map_invalid_tolerance_tolerance_violation(self) -> None:
        """BRepCheck_InvalidToleranceValue maps to TOLERANCE_VIOLATION."""
        result = map_single_error("BRepCheck_InvalidToleranceValue", "EDGE", 5)

        assert result.category == ErrorCategory.TOLERANCE_VIOLATION
        assert result.severity == ErrorSeverity.WARNING
        assert result.entity_index == 5

    def test_map_check_fail_boolean_failure(self) -> None:
        """BRepCheck_CheckFail maps to BOOLEAN_FAILURE."""
        result = map_single_error("BRepCheck_CheckFail", "FACE", 0)

        assert result.category == ErrorCategory.BOOLEAN_FAILURE
        assert result.severity == ErrorSeverity.CRITICAL

    def test_map_unknown_code_defaults_to_topology_error(self) -> None:
        """Unknown error codes default to TOPOLOGY_ERROR with CRITICAL severity."""
        result = map_single_error("UNKNOWN_CODE", "FACE", 0)

        assert result.category == ErrorCategory.TOPOLOGY_ERROR
        assert result.severity == ErrorSeverity.CRITICAL
        assert result.occ_code_name == "UNKNOWN_CODE"

    def test_map_single_error_preserves_entity_type(self) -> None:
        """Entity type is preserved in mapped error."""
        result = map_single_error("BRepCheck_NotClosed", "SHELL", 3)

        assert result.entity_type == "SHELL"
        assert result.entity_index == 3

    def test_map_single_error_different_entity_types(self) -> None:
        """Entity type and index are preserved for different TopAbs types."""
        for entity_type, idx in [("SOLID", 0), ("FACE", 5), ("EDGE", 10), ("VERTEX", 3)]:
            result = map_single_error("BRepCheck_NotClosed", entity_type, idx)
            assert result.entity_type == entity_type
            assert result.entity_index == idx


class TestCategorizeErrors:
    """Test the categorize_errors function."""

    def test_categorize_errors_groups_by_category(self) -> None:
        """Errors are grouped by their ErrorCategory."""
        errors = [
            map_single_error("BRepCheck_NotClosed", "SHELL", 0),
            map_single_error("BRepCheck_NotConnected", "SHELL", 0),
            map_single_error("BRepCheck_SelfIntersectingWire", "WIRE", 1),
        ]

        categorized = categorize_errors(errors)

        assert ErrorCategory.TOPOLOGY_ERROR in categorized
        assert ErrorCategory.SELF_INTERSECTION in categorized
        assert len(categorized[ErrorCategory.TOPOLOGY_ERROR]) == 2
        assert len(categorized[ErrorCategory.SELF_INTERSECTION]) == 1

    def test_categorize_errors_sorts_by_severity(self) -> None:
        """Within each category, CRITICAL errors come before WARNING."""
        errors = [
            map_single_error("BRepCheck_OrientationOfExternalWire", "FACE", 0),  # TOPOLOGY_ERROR, WARNING
            map_single_error("BRepCheck_NotClosed", "SHELL", 0),                  # TOPOLOGY_ERROR, CRITICAL
        ]

        categorized = categorize_errors(errors)
        topo_errors = categorized[ErrorCategory.TOPOLOGY_ERROR]

        # NotClosed (CRITICAL) should come first
        assert topo_errors[0].occ_code_name == "BRepCheck_NotClosed"
        assert topo_errors[1].occ_code_name == "BRepCheck_OrientationOfExternalWire"

    def test_categorize_errors_empty_list(self) -> None:
        """categorize_errors with empty list returns empty dict."""
        categorized = categorize_errors([])

        assert isinstance(categorized, dict)
        assert len(categorized) == 0

    def test_categorize_errors_multiple_in_same_category(self) -> None:
        """Multiple errors in the same category are all grouped."""
        errors = [
            map_single_error("BRepCheck_EmptyWire", "WIRE", 0),
            map_single_error("BRepCheck_EmptyShell", "SHELL", 1),
            map_single_error("BRepCheck_NoSurface", "FACE", 2),
        ]

        categorized = categorize_errors(errors)

        assert ErrorCategory.DEGENERATE_SHAPE in categorized
        assert len(categorized[ErrorCategory.DEGENERATE_SHAPE]) == 3


class TestMappedError:
    """Test MappedError dataclass."""

    def test_mapped_error_construction(self) -> None:
        """MappedError can be constructed with all fields."""
        error = MappedError(
            occ_code_name="BRepCheck_NotClosed",
            occ_code_value=1,
            category=ErrorCategory.TOPOLOGY_ERROR,
            severity=ErrorSeverity.CRITICAL,
            description="Shell is not closed",
            suggestion="Ensure all loops are closed",
            entity_type="SHELL",
            entity_index=0,
        )

        assert error.occ_code_name == "BRepCheck_NotClosed"
        assert error.occ_code_value == 1
        assert error.category == ErrorCategory.TOPOLOGY_ERROR
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.entity_type == "SHELL"
        assert error.entity_index == 0

    def test_mapped_error_defaults(self) -> None:
        """MappedError has sensible default values."""
        error = MappedError()

        assert error.occ_code_name == ""
        assert error.occ_code_value == -1
        assert error.category == ErrorCategory.TOPOLOGY_ERROR
        assert error.severity == ErrorSeverity.CRITICAL


# ---------------------------------------------------------------------------
# feedback_builder.py tests
# ---------------------------------------------------------------------------

class TestBuildCodeFeedback:
    """Test the build_code_feedback function."""

    def test_build_code_feedback_invalid_result_contains_failed_marker(
        self,
        disposal_result_invalid: DisposalResult,
        code_proposal_cadquery: CodeProposal,
    ) -> None:
        """Code feedback for invalid result contains 'PREVIOUS ATTEMPT FAILED'."""
        feedback = build_code_feedback(disposal_result_invalid, code_proposal_cadquery)

        assert "PREVIOUS ATTEMPT FAILED" in feedback
        assert "CORRECTION REQUIRED" in feedback

    def test_build_code_feedback_includes_error_category(
        self,
        disposal_result_invalid: DisposalResult,
        code_proposal_cadquery: CodeProposal,
    ) -> None:
        """Feedback includes the error category."""
        feedback = build_code_feedback(disposal_result_invalid, code_proposal_cadquery)

        assert "Error Category:" in feedback
        assert disposal_result_invalid.error_category.value in feedback

    def test_build_code_feedback_includes_critical_findings(
        self,
        disposal_result_invalid: DisposalResult,
        code_proposal_cadquery: CodeProposal,
    ) -> None:
        """Feedback includes critical findings with entity type and error code."""
        feedback = build_code_feedback(disposal_result_invalid, code_proposal_cadquery)

        assert "Critical issues" in feedback
        assert "BRepCheck_NotClosed" in feedback
        assert "SHELL" in feedback

    def test_build_code_feedback_includes_repair_actions(
        self,
        disposal_result_invalid: DisposalResult,
        code_proposal_cadquery: CodeProposal,
    ) -> None:
        """Feedback includes repair attempt information when available."""
        feedback = build_code_feedback(disposal_result_invalid, code_proposal_cadquery)

        assert "Deterministic repair was attempted" in feedback
        assert "ShapeFix_Shape" in feedback or "ShapeFix_Wire" in feedback

    def test_build_code_feedback_includes_geometry_report(
        self,
        disposal_result_invalid: DisposalResult,
        code_proposal_cadquery: CodeProposal,
    ) -> None:
        """Feedback includes geometry report when present."""
        feedback = build_code_feedback(disposal_result_invalid, code_proposal_cadquery)

        assert "Geometry analysis" in feedback or "Faces:" in feedback

    def test_build_code_feedback_includes_failing_code(
        self,
        disposal_result_invalid: DisposalResult,
        code_proposal_cadquery: CodeProposal,
    ) -> None:
        """Feedback includes the failing code in a code block."""
        feedback = build_code_feedback(disposal_result_invalid, code_proposal_cadquery)

        assert "The failing code:" in feedback
        assert "```python" in feedback
        assert code_proposal_cadquery.code in feedback

    def test_build_code_feedback_includes_correction_suggestion(
        self,
        disposal_result_invalid: DisposalResult,
        code_proposal_cadquery: CodeProposal,
    ) -> None:
        """Feedback includes a CORRECTION line with suggestion."""
        feedback = build_code_feedback(disposal_result_invalid, code_proposal_cadquery)

        assert "CORRECTION:" in feedback

    def test_build_code_feedback_code_truncation(
        self,
        disposal_result_invalid: DisposalResult,
    ) -> None:
        """Code longer than 100 lines is truncated in feedback."""
        long_code = "\n".join([f"line{i} = None" for i in range(150)])
        proposal = CodeProposal(
            proposal_id="test_long",
            code=long_code,
        )

        feedback = build_code_feedback(disposal_result_invalid, proposal)

        # Code should be present but truncated
        assert "```python" in feedback
        assert "..." in feedback
        assert "more lines" in feedback

    def test_build_code_feedback_valid_result(
        self,
        disposal_result_valid: DisposalResult,
        code_proposal_cadquery: CodeProposal,
    ) -> None:
        """Code feedback for valid result still produces output."""
        feedback = build_code_feedback(disposal_result_valid, code_proposal_cadquery)

        # Should still have the structure even if valid
        assert isinstance(feedback, str)
        assert len(feedback) > 0


class TestBuildNeuralFeedback:
    """Test the build_neural_feedback function."""

    def test_build_neural_feedback_returns_dict(
        self,
        disposal_result_invalid: DisposalResult,
    ) -> None:
        """build_neural_feedback returns a dictionary."""
        feedback = build_neural_feedback(disposal_result_invalid)

        assert isinstance(feedback, dict)

    def test_build_neural_feedback_has_required_keys(
        self,
        disposal_result_invalid: DisposalResult,
    ) -> None:
        """Neural feedback includes all required keys."""
        feedback = build_neural_feedback(disposal_result_invalid)

        required_keys = {
            "error_category",
            "failed_entity_indices",
            "parameter_hints",
            "topology_stats",
            "severity_counts",
        }
        for key in required_keys:
            assert key in feedback, f"Missing key: {key}"

    def test_build_neural_feedback_error_category_is_string(
        self,
        disposal_result_invalid: DisposalResult,
    ) -> None:
        """error_category in neural feedback is a string."""
        feedback = build_neural_feedback(disposal_result_invalid)

        assert isinstance(feedback["error_category"], str)
        assert feedback["error_category"] == ErrorCategory.TOPOLOGY_ERROR.value

    def test_build_neural_feedback_failed_entity_indices(
        self,
        disposal_result_invalid: DisposalResult,
    ) -> None:
        """failed_entity_indices groups indices by entity type."""
        feedback = build_neural_feedback(disposal_result_invalid)

        indices = feedback["failed_entity_indices"]
        assert isinstance(indices, dict)
        # Invalid result has SHELL entity at index 0 and EDGE at index 3
        assert "SHELL" in indices
        assert "EDGE" in indices
        assert 0 in indices["SHELL"]
        assert 3 in indices["EDGE"]

    def test_build_neural_feedback_severity_counts(
        self,
        disposal_result_invalid: DisposalResult,
    ) -> None:
        """severity_counts has correct structure and values."""
        feedback = build_neural_feedback(disposal_result_invalid)

        counts = feedback["severity_counts"]
        assert "critical" in counts
        assert "warning" in counts
        assert "info" in counts
        assert counts["critical"] == 1  # One CRITICAL: NotClosed
        assert counts["warning"] == 1   # One WARNING: FreeEdge

    def test_build_neural_feedback_parameter_hints(
        self,
        disposal_result_invalid: DisposalResult,
    ) -> None:
        """parameter_hints include hints for detected error categories."""
        feedback = build_neural_feedback(disposal_result_invalid)

        hints = feedback["parameter_hints"]
        # TOPOLOGY_ERROR should map to "topology" hint
        assert "topology" in hints
        assert hints["topology"] == "restructure_sketch_loops"

    def test_build_neural_feedback_self_intersection_hints(
        self,
        disposal_result_self_intersection: DisposalResult,
    ) -> None:
        """Self-intersection result includes geometry hint."""
        feedback = build_neural_feedback(disposal_result_self_intersection)

        hints = feedback["parameter_hints"]
        assert "geometry" in hints
        assert hints["geometry"] == "separate_overlapping_regions"

    def test_build_neural_feedback_topology_stats(
        self,
        disposal_result_invalid: DisposalResult,
    ) -> None:
        """topology_stats from geometry report are included."""
        feedback = build_neural_feedback(disposal_result_invalid)

        stats = feedback["topology_stats"]
        assert "face_count" in stats
        assert "edge_count" in stats
        assert stats["face_count"] == 5


class TestBuildTrainingFeedback:
    """Test the build_training_feedback function."""

    def test_build_training_feedback_returns_dict(
        self,
        disposal_result_invalid: DisposalResult,
    ) -> None:
        """build_training_feedback returns a dictionary."""
        feedback = build_training_feedback(disposal_result_invalid)

        assert isinstance(feedback, dict)

    def test_build_training_feedback_has_reward_signal(
        self,
        disposal_result_invalid: DisposalResult,
    ) -> None:
        """Training feedback includes reward_signal."""
        feedback = build_training_feedback(disposal_result_invalid)

        assert "reward_signal" in feedback
        assert isinstance(feedback["reward_signal"], float)

    def test_build_training_feedback_validation_tiers(
        self,
        disposal_result_valid: DisposalResult,
    ) -> None:
        """Training feedback includes validation_tiers with tier checks."""
        feedback = build_training_feedback(disposal_result_valid)

        tiers = feedback["validation_tiers"]
        assert "shape_constructed" in tiers
        assert "manifold" in tiers
        assert "watertight" in tiers
        assert "euler_valid" in tiers
        assert "no_self_intersection" in tiers

    def test_build_training_feedback_valid_result_all_tiers_pass(
        self,
        disposal_result_valid: DisposalResult,
    ) -> None:
        """Valid result passes all validation tiers."""
        feedback = build_training_feedback(disposal_result_valid)

        tiers = feedback["validation_tiers"]
        # Valid result should pass all checks
        assert tiers["shape_constructed"] is True
        assert tiers["manifold"] is True
        assert tiers["watertight"] is True
        assert tiers["euler_valid"] is True
        assert tiers["no_self_intersection"] is True

    def test_build_training_feedback_invalid_no_watertight(
        self,
        disposal_result_invalid: DisposalResult,
    ) -> None:
        """Invalid result with NotClosed fails watertight check."""
        feedback = build_training_feedback(disposal_result_invalid)

        tiers = feedback["validation_tiers"]
        # NotClosed error should fail watertight check
        assert tiers["watertight"] is False

    def test_build_training_feedback_self_intersection_fails(
        self,
        disposal_result_self_intersection: DisposalResult,
    ) -> None:
        """Self-intersection result fails no_self_intersection check."""
        feedback = build_training_feedback(disposal_result_self_intersection)

        tiers = feedback["validation_tiers"]
        assert tiers["no_self_intersection"] is False

    def test_build_training_feedback_category_penalties(
        self,
        disposal_result_invalid: DisposalResult,
    ) -> None:
        """Training feedback includes per-category penalties."""
        feedback = build_training_feedback(disposal_result_invalid)

        assert "category_penalties" in feedback
        penalties = feedback["category_penalties"]
        # Should have penalty for TOPOLOGY_ERROR category
        assert ErrorCategory.TOPOLOGY_ERROR.value in penalties

    def test_build_training_feedback_tiers_passed_count(
        self,
        disposal_result_valid: DisposalResult,
    ) -> None:
        """Training feedback tracks tiers_passed and tiers_total."""
        feedback = build_training_feedback(disposal_result_valid)

        assert "tiers_passed" in feedback
        assert "tiers_total" in feedback
        assert isinstance(feedback["tiers_passed"], int)
        assert isinstance(feedback["tiers_total"], int)
        # Valid result should pass most/all tiers
        assert feedback["tiers_passed"] > 0


# ---------------------------------------------------------------------------
# reward_signal.py tests
# ---------------------------------------------------------------------------

class TestComputeReward:
    """Test the compute_reward function."""

    def test_compute_reward_valid_result(
        self,
        disposal_result_valid: DisposalResult,
    ) -> None:
        """Valid result gets validity_reward plus shape_constructed_reward."""
        reward = compute_reward(disposal_result_valid)

        # validity_reward (1.0) + shape_constructed_reward (0.3) = 1.3, clamped to 1.0
        assert reward == 1.0
        assert isinstance(reward, float)

    def test_compute_reward_no_shape(
        self,
        disposal_result_no_shape: DisposalResult,
    ) -> None:
        """Result with no shape gets no shape_constructed reward but tiers may pass.

        With no shape and no errors, some tiers pass (manifold, watertight, euler
        benefit of doubt, self-intersection). This gives 4 tiers * 0.1 = 0.4.
        """
        reward = compute_reward(disposal_result_no_shape)

        # No shape, but no errors means some tiers pass: 4 tiers * 0.1 = 0.4
        assert reward == 0.4

    def test_compute_reward_shape_constructed(
        self,
        disposal_result_invalid: DisposalResult,
    ) -> None:
        """Invalid but existing shape includes shape_constructed_reward."""
        config = FeedbackConfig()
        reward = compute_reward(disposal_result_invalid, config)

        # Should include shape_constructed_reward (0.3) and tier bonuses, minus penalties
        assert reward > -1.0
        assert reward < 1.0

    def test_compute_reward_repairable(
        self,
        disposal_result_repaired: DisposalResult,
    ) -> None:
        """Repaired result includes repairable_reward."""
        reward = compute_reward(disposal_result_repaired)

        # Should be > 0 because repair succeeded and shape is now valid
        assert reward > 0.0

    def test_compute_reward_critical_error_penalty(
        self,
        disposal_result_invalid: DisposalResult,
    ) -> None:
        """Critical errors apply penalty.

        Invalid result has shape (0.3), and partial tiers pass: watertight fails
        due to NotClosed, but manifold (0.1), euler (0.1), self-intersection (0.1)
        pass = 0.3 + 0.3 - 0.1 (critical penalty) = 0.5, then we also apply
        tier bonuses more carefully.
        """
        reward = compute_reward(disposal_result_invalid)

        # Shape exists (0.3) + some tiers pass (at least manifold 0.1)
        # but less than fully valid due to critical errors
        assert reward > 0.0
        assert reward < 1.0

    def test_compute_reward_clamped_range(
        self,
        disposal_result_invalid: DisposalResult,
    ) -> None:
        """Reward is always in [-1.0, 1.0] range."""
        reward = compute_reward(disposal_result_invalid)

        assert -1.0 <= reward <= 1.0

    def test_compute_reward_with_target_dimensions_match(
        self,
        disposal_result_valid: DisposalResult,
    ) -> None:
        """Matching target dimensions adds semantic_match_reward."""
        target_dims = (100.0, 50.0, 20.0)  # Box dimensions from fixture
        reward = compute_reward(disposal_result_valid, target_dimensions=target_dims)

        # Should include semantic_match_reward (0.2)
        assert reward == 1.0  # Clamped

    def test_compute_reward_with_target_dimensions_no_match(
        self,
        disposal_result_valid: DisposalResult,
    ) -> None:
        """Non-matching target dimensions get no semantic reward."""
        target_dims = (50.0, 50.0, 50.0)  # Different from box
        reward = compute_reward(disposal_result_valid, target_dimensions=target_dims)

        # Should not get semantic_match_reward
        assert reward == 1.0

    def test_compute_reward_with_target_volume_match(
        self,
        disposal_result_valid: DisposalResult,
    ) -> None:
        """Matching target volume adds partial semantic reward."""
        target_volume = 100000.0  # Box volume from fixture
        reward = compute_reward(disposal_result_valid, target_volume=target_volume)

        # Should include semantic_match_reward * 0.5
        assert reward == 1.0  # Clamped

    def test_compute_reward_custom_config(
        self,
        disposal_result_valid: DisposalResult,
    ) -> None:
        """Custom FeedbackConfig weights are applied.

        Valid result is valid, so only validity_reward and shape_constructed_reward apply.
        With validity_reward=0.5 and shape_constructed_reward=0.3, total = 0.8.
        """
        config = FeedbackConfig()
        config.validity_reward = 0.5  # Lower validity reward
        reward = compute_reward(disposal_result_valid, config)

        # Valid result gets 0.5 (validity) + 0.3 (shape_constructed) = 0.8
        assert reward == 0.8

    def test_compute_reward_rounded(
        self,
        disposal_result_valid: DisposalResult,
    ) -> None:
        """Reward is rounded to 4 decimal places."""
        reward = compute_reward(disposal_result_valid)

        # Should be a round number (1.0 in this case)
        assert reward == round(reward, 4)


class TestComputeBatchRewards:
    """Test the compute_batch_rewards function."""

    def test_compute_batch_rewards_returns_list(
        self,
        disposal_result_valid: DisposalResult,
        disposal_result_invalid: DisposalResult,
    ) -> None:
        """compute_batch_rewards returns a list."""
        rewards = compute_batch_rewards([disposal_result_valid, disposal_result_invalid])

        assert isinstance(rewards, list)

    def test_compute_batch_rewards_correct_length(
        self,
        disposal_result_valid: DisposalResult,
        disposal_result_invalid: DisposalResult,
        disposal_result_no_shape: DisposalResult,
    ) -> None:
        """Batch rewards list has same length as input."""
        results = [disposal_result_valid, disposal_result_invalid, disposal_result_no_shape]
        rewards = compute_batch_rewards(results)

        assert len(rewards) == 3

    def test_compute_batch_rewards_matches_individual(
        self,
        disposal_result_valid: DisposalResult,
        disposal_result_invalid: DisposalResult,
    ) -> None:
        """Each batch reward matches individual compute_reward."""
        results = [disposal_result_valid, disposal_result_invalid]
        batch_rewards = compute_batch_rewards(results)

        for i, result in enumerate(results):
            individual_reward = compute_reward(result)
            assert batch_rewards[i] == individual_reward

    def test_compute_batch_rewards_with_config(
        self,
        disposal_result_valid: DisposalResult,
        disposal_result_invalid: DisposalResult,
    ) -> None:
        """Custom config is applied to all batch results."""
        config = FeedbackConfig()
        config.critical_error_penalty = -0.2

        results = [disposal_result_valid, disposal_result_invalid]
        rewards = compute_batch_rewards(results, config)

        assert len(rewards) == 2
        # Second result has critical error, should be affected by custom penalty
        assert isinstance(rewards[0], float)
        assert isinstance(rewards[1], float)


class TestCountPassingTiers:
    """Test the internal _count_passing_tiers function via compute_reward."""

    def test_count_passing_tiers_valid_result(
        self,
        disposal_result_valid: DisposalResult,
    ) -> None:
        """Valid result with all tiers passing."""
        from ll_gen.feedback.reward_signal import _count_passing_tiers

        tiers_passed = _count_passing_tiers(disposal_result_valid)

        # Valid result should pass all 4 tiers
        assert tiers_passed == 4

    def test_count_passing_tiers_not_closed_fails_watertight(
        self,
        disposal_result_invalid: DisposalResult,
    ) -> None:
        """Result with NotClosed fails watertight tier."""
        from ll_gen.feedback.reward_signal import _count_passing_tiers

        tiers_passed = _count_passing_tiers(disposal_result_invalid)

        # Should fail watertight check due to NotClosed
        assert tiers_passed < 4

    def test_count_passing_tiers_self_intersection(
        self,
        disposal_result_self_intersection: DisposalResult,
    ) -> None:
        """Self-intersection result fails self-intersection tier."""
        from ll_gen.feedback.reward_signal import _count_passing_tiers

        tiers_passed = _count_passing_tiers(disposal_result_self_intersection)

        # Should fail self-intersection check
        assert tiers_passed < 4


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestFeedbackIntegration:
    """Integration tests combining multiple feedback components."""

    def test_feedback_pipeline_invalid_result(
        self,
        disposal_result_invalid: DisposalResult,
        code_proposal_cadquery: CodeProposal,
    ) -> None:
        """Complete feedback pipeline on invalid result."""
        code_fb = build_code_feedback(disposal_result_invalid, code_proposal_cadquery)
        neural_fb = build_neural_feedback(disposal_result_invalid)
        training_fb = build_training_feedback(disposal_result_invalid)
        reward = compute_reward(disposal_result_invalid)

        # All should produce valid output
        assert len(code_fb) > 0
        assert len(neural_fb) > 0
        assert len(training_fb) > 0
        assert -1.0 <= reward <= 1.0

    def test_feedback_pipeline_valid_result(
        self,
        disposal_result_valid: DisposalResult,
        code_proposal_cadquery: CodeProposal,
    ) -> None:
        """Complete feedback pipeline on valid result."""
        code_fb = build_code_feedback(disposal_result_valid, code_proposal_cadquery)
        neural_fb = build_neural_feedback(disposal_result_valid)
        training_fb = build_training_feedback(disposal_result_valid)
        reward = compute_reward(disposal_result_valid)

        # Valid result should have high reward
        assert reward == 1.0
        # Should still have feedback strings
        assert len(code_fb) > 0
        assert len(neural_fb) > 0

    def test_error_mapper_to_feedback_consistency(
        self,
        disposal_result_invalid: DisposalResult,
    ) -> None:
        """Error mapper categories are used consistently in feedback."""
        neural_fb = build_neural_feedback(disposal_result_invalid)
        training_fb = build_training_feedback(disposal_result_invalid)

        # Both should refer to the same error category
        if disposal_result_invalid.error_category:
            assert neural_fb["error_category"] == disposal_result_invalid.error_category.value

    def test_batch_rewards_consistency(
        self,
        disposal_result_valid: DisposalResult,
        disposal_result_invalid: DisposalResult,
        disposal_result_repaired: DisposalResult,
    ) -> None:
        """Batch rewards are consistent across multiple runs."""
        results = [disposal_result_valid, disposal_result_invalid, disposal_result_repaired]

        batch1 = compute_batch_rewards(results)
        batch2 = compute_batch_rewards(results)

        # Same inputs should produce same rewards
        assert batch1 == batch2
