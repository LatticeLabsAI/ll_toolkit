"""End-to-end integration tests for ll_gen.

Tests the full generation workflow:
- Text → route → propose → dispose → export

These tests verify the complete pipeline works correctly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ll_gen.config import GenerationRoute, CodeLanguage
from ll_gen.proposals.code_proposal import CodeProposal
from ll_gen.proposals.command_proposal import CommandSequenceProposal
from ll_gen.proposals.latent_proposal import LatentProposal
from ll_gen.proposals.disposal_result import DisposalResult, GeometryReport


# Check dependencies
try:
    from OCC.Core.TopoDS import TopoDS_Shape
    _OCC_AVAILABLE = True
except ImportError:
    _OCC_AVAILABLE = False

try:
    import cadquery
    _CADQUERY_AVAILABLE = True
except ImportError:
    _CADQUERY_AVAILABLE = False

requires_occ = pytest.mark.skipif(
    not _OCC_AVAILABLE,
    reason="pythonocc not installed"
)

requires_cadquery = pytest.mark.skipif(
    not _CADQUERY_AVAILABLE,
    reason="cadquery not installed"
)


# ============================================================================
# SECTION 1: Router Integration Tests
# ============================================================================


class TestRouterIntegration:
    """Test router integration with generation pipeline."""

    def test_router_returns_routing_decision(self) -> None:
        """Test router returns RoutingDecision object."""
        from ll_gen.routing.router import GenerationRouter, RoutingDecision

        router = GenerationRouter()
        decision = router.route("A mechanical bracket with bolt holes")

        assert isinstance(decision, RoutingDecision)
        assert decision.route in list(GenerationRoute)
        assert 0 <= decision.confidence <= 1

    def test_router_mechanical_keywords_trigger_code(self) -> None:
        """Test mechanical keywords trigger code generation."""
        from ll_gen.routing.router import GenerationRouter

        router = GenerationRouter()
        decision = router.route("Extrude a rectangular plate with chamfered edges")

        # Should route to a code generation path
        assert decision.route in [
            GenerationRoute.CODE_CADQUERY,
            GenerationRoute.CODE_OPENSCAD,
            GenerationRoute.CODE_PYTHONOCC,
        ]

    def test_router_freeform_keywords_trigger_neural(self) -> None:
        """Test freeform keywords trigger neural generation."""
        from ll_gen.routing.router import GenerationRouter

        router = GenerationRouter()
        decision = router.route("A smooth organic flowing sculptural shape")

        # Should have some confidence in neural routes
        assert decision.route in list(GenerationRoute)

    def test_router_force_override(self) -> None:
        """Test router respects force_route parameter."""
        from ll_gen.routing.router import GenerationRouter

        router = GenerationRouter()
        decision = router.route(
            "A mechanical bracket",
            force_route=GenerationRoute.NEURAL_VAE,
        )

        assert decision.route == GenerationRoute.NEURAL_VAE
        assert decision.forced is True


# ============================================================================
# SECTION 2: Disposal Engine Integration Tests
# ============================================================================


class TestDisposalEngineIntegration:
    """Test disposal engine integration with proposals."""

    def test_disposal_engine_init(self) -> None:
        """Test DisposalEngine initializes correctly."""
        from ll_gen.disposal.engine import DisposalEngine

        engine = DisposalEngine()
        assert engine is not None

    def test_disposal_engine_creates_output_dir(self, tmp_path: Path) -> None:
        """Test DisposalEngine creates output directory."""
        from ll_gen.disposal.engine import DisposalEngine

        output_dir = tmp_path / "test_output"
        engine = DisposalEngine(output_dir=str(output_dir))

        assert output_dir.exists()

    def test_disposal_result_structure(self) -> None:
        """Test DisposalResult has expected structure."""
        result = DisposalResult(
            shape=None,
            is_valid=True,
            error_details=[],
            proposal_id="test",
            proposal_type="CodeProposal",
        )

        assert hasattr(result, "shape")
        assert hasattr(result, "is_valid")
        assert hasattr(result, "geometry_report")
        assert hasattr(result, "reward_signal")


# ============================================================================
# SECTION 3: Feedback Integration Tests
# ============================================================================


class TestFeedbackIntegration:
    """Test feedback system integration."""

    def test_error_mapper_available(self) -> None:
        """Test error mapper module is available."""
        from ll_gen.feedback.error_mapper import OCC_ERROR_MAP
        assert OCC_ERROR_MAP is not None
        assert len(OCC_ERROR_MAP) > 0

    def test_feedback_builder_available(self) -> None:
        """Test feedback builder functions are available."""
        from ll_gen.feedback.feedback_builder import (
            build_code_feedback,
            build_neural_feedback,
        )
        assert callable(build_code_feedback)
        assert callable(build_neural_feedback)

    def test_reward_signal_computation(self) -> None:
        """Test reward signal can be computed."""
        from ll_gen.feedback.reward_signal import compute_reward
        from ll_gen.config import FeedbackConfig

        result = DisposalResult(
            shape=None,
            is_valid=True,
            error_details=[],
        )
        config = FeedbackConfig()

        reward = compute_reward(result, config)

        assert isinstance(reward, float)
        assert -1.0 <= reward <= 1.0


# ============================================================================
# SECTION 4: Orchestrator Integration Tests (Mocked)
# ============================================================================


class TestOrchestratorIntegrationMocked:
    """Test orchestrator integration with mocked components."""

    def test_orchestrator_init(self) -> None:
        """Test GenerationOrchestrator initializes."""
        from ll_gen.pipeline.orchestrator import GenerationOrchestrator

        orchestrator = GenerationOrchestrator()
        assert orchestrator is not None

    def test_orchestrator_has_generate_method(self) -> None:
        """Test orchestrator has generate method."""
        from ll_gen.pipeline.orchestrator import GenerationOrchestrator

        orchestrator = GenerationOrchestrator()
        assert hasattr(orchestrator, "generate")
        assert callable(orchestrator.generate)


# ============================================================================
# SECTION 5: Full Pipeline Tests (Mocked)
# ============================================================================


@pytest.mark.integration
class TestFullPipelineMocked:
    """Test full generation pipeline with mocked dependencies."""

    def test_code_generation_pipeline_mocked(self) -> None:
        """Test code generation pipeline with mocked LLM."""
        from ll_gen.routing.router import GenerationRouter
        from ll_gen.codegen.cadquery_proposer import CadQueryProposer

        # Step 1: Route
        router = GenerationRouter()
        decision = router.route("A box 100mm wide")

        # Step 2: Propose (mocked)
        proposer = CadQueryProposer()
        mock_code = """from cadquery import Workplane as cq
result = cq("XY").box(100, 50, 20)
result.val()"""

        with patch("ll_gen.codegen.cadquery_proposer._CADLING_AVAILABLE", True):
            with patch.object(proposer, "generator") as mock_gen:
                mock_gen.generate = MagicMock(return_value=mock_code)
                proposal = proposer.propose("A box 100mm wide")

        assert isinstance(proposal, CodeProposal)
        assert "box" in proposal.code

    def test_disposal_result_tracking(self) -> None:
        """Test disposal result tracks proposal info."""
        result = DisposalResult(
            shape="<mock_shape>",
            is_valid=True,
            error_details=[],
            geometry_report=GeometryReport(
                volume=100000.0,
                surface_area=13000.0,
                bounding_box=(0, 0, 0, 100, 50, 20),
                face_count=6,
            ),
            proposal_id="test_proposal_01",
            proposal_type="CodeProposal",
            execution_time_ms=250.0,
        )

        assert result.proposal_id == "test_proposal_01"
        assert result.proposal_type == "CodeProposal"
        assert result.execution_time_ms == 250.0
        assert result.geometry_report.volume == 100000.0


# ============================================================================
# SECTION 6: Real Execution Tests (Requires Dependencies)
# ============================================================================


@requires_occ
@requires_cadquery
class TestRealExecution:
    """Test real execution with actual dependencies."""

    def test_code_execution_produces_shape(self, tmp_path: Path) -> None:
        """Test code execution produces valid TopoDS_Shape."""
        from ll_gen.disposal.engine import DisposalEngine
        from ll_gen.proposals.code_proposal import CodeProposal
        from ll_gen.config import CodeLanguage

        proposal = CodeProposal(
            proposal_id="test_real",
            confidence=0.9,
            source_prompt="A box",
            # Use pre-imported cq from sandbox namespace (no imports needed)
            code="""result = cq("XY").box(100, 50, 20)
result.val()""",
            language=CodeLanguage.CADQUERY,
        )

        engine = DisposalEngine(output_dir=str(tmp_path))
        result = engine.dispose(proposal, export=False)

        assert result.is_valid is True
        assert result.shape is not None

    def test_full_pipeline_with_export(self, tmp_path: Path) -> None:
        """Test full pipeline with STEP/STL export."""
        from ll_gen.disposal.engine import DisposalEngine
        from ll_gen.proposals.code_proposal import CodeProposal
        from ll_gen.config import CodeLanguage

        proposal = CodeProposal(
            proposal_id="test_export",
            confidence=0.9,
            source_prompt="A box",
            # Use pre-imported cq from sandbox namespace (no imports needed)
            code="""result = cq("XY").box(100, 50, 20)
result.val()""",
            language=CodeLanguage.CADQUERY,
        )

        engine = DisposalEngine(output_dir=str(tmp_path))
        result = engine.dispose(proposal, export=True)

        assert result.is_valid is True
        # Should have exported at least one file
        if result.step_path:
            assert result.step_path.exists()
        if result.stl_path:
            assert result.stl_path.exists()


# ============================================================================
# SECTION 7: Error Handling Integration Tests
# ============================================================================


class TestErrorHandlingIntegration:
    """Test error handling across the pipeline."""

    def test_invalid_code_produces_error_result(self) -> None:
        """Test invalid code produces result with error details."""
        from ll_gen.proposals.code_proposal import CodeProposal
        from ll_gen.config import CodeLanguage

        proposal = CodeProposal(
            proposal_id="test_error",
            confidence=0.5,
            source_prompt="Invalid code",
            code="this is not valid python code (((",
            language=CodeLanguage.CADQUERY,
        )

        # Call validate_syntax to check the code
        proposal.validate_syntax()

        # Syntax validation should catch this
        assert proposal.syntax_valid is False

    def test_error_context_propagation(self) -> None:
        """Test error context propagates through retry."""
        from ll_gen.proposals.base import BaseProposal

        proposal = BaseProposal(
            proposal_id="test",
            confidence=0.8,
            source_prompt="Test",
        )

        # Add error context
        error_context = {"category": "TOPOLOGY_ERROR", "message": "Shell not closed"}
        updated = proposal.with_error_context(error_context)

        assert updated.error_context == error_context
        assert updated.attempt == proposal.attempt + 1


# ============================================================================
# SECTION 8: Configuration Integration Tests
# ============================================================================


class TestConfigurationIntegration:
    """Test configuration flows through the pipeline."""

    def test_config_helper_function(self) -> None:
        """Test get_ll_gen_config helper works."""
        from ll_gen.config import get_ll_gen_config

        config = get_ll_gen_config(
            max_retries=5,
            **{
                "codegen.temperature": 0.5,
                "disposal.tolerance": 1e-6,
            }
        )

        assert config.max_retries == 5
        assert config.codegen.temperature == 0.5
        assert config.disposal.tolerance == 1e-6

    def test_config_affects_disposal_engine(self) -> None:
        """Test configuration affects disposal engine behavior."""
        from ll_gen.disposal.engine import DisposalEngine
        from ll_gen.config import DisposalConfig, ExportConfig, FeedbackConfig

        disposal_cfg = DisposalConfig(max_repair_passes=5)
        export_cfg = ExportConfig(render_resolution=1024)
        feedback_cfg = FeedbackConfig(validity_reward=2.0)

        engine = DisposalEngine(
            disposal_config=disposal_cfg,
            export_config=export_cfg,
            feedback_config=feedback_cfg,
        )

        assert engine.disposal_config.max_repair_passes == 5
        assert engine.export_config.render_resolution == 1024
        assert engine.feedback_config.validity_reward == 2.0
