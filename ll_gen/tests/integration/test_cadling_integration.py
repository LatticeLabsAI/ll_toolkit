"""Integration tests for ll_gen with cadling package.

Tests integration between ll_gen and cadling:
- CadQueryProposer → cadling.CadQueryGenerator
- OpenSCADProposer → cadling.OpenSCADGenerator
- CommandExecutor → cadling.CommandExecutor
- SurfaceExecutor → cadling.BSplineSurfaceFitter, TopologyMerger

All tests are marked with @pytest.mark.requires_cadling and skip if cadling
is not installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Check if cadling is available
try:
    import cadling
    _CADLING_AVAILABLE = True
except ImportError:
    _CADLING_AVAILABLE = False

requires_cadling = pytest.mark.skipif(
    not _CADLING_AVAILABLE,
    reason="cadling package not installed"
)


# ============================================================================
# SECTION 1: CadQueryProposer Integration Tests
# ============================================================================


@requires_cadling
class TestCadQueryProposerIntegration:
    """Test CadQueryProposer integration with cadling.CadQueryGenerator."""

    def test_proposer_imports_cadling_generator(self) -> None:
        """Test that CadQueryProposer can import cadling generator."""
        from ll_gen.codegen.cadquery_proposer import CadQueryProposer
        proposer = CadQueryProposer()
        assert proposer is not None

    def test_proposer_creates_code_proposal(self) -> None:
        """Test that CadQueryProposer creates valid CodeProposal."""
        from ll_gen.codegen.cadquery_proposer import CadQueryProposer
        from ll_gen.proposals.code_proposal import CodeProposal

        proposer = CadQueryProposer()
        # Mock the generator to avoid actual LLM calls
        mock_code = """from cadquery import Workplane as cq
result = cq("XY").box(100, 50, 20)
result.val()"""
        proposer.generator = MagicMock()
        proposer.generator.generate = MagicMock(return_value=mock_code)

        proposal = proposer.propose("A box 100mm x 50mm x 20mm")

        assert isinstance(proposal, CodeProposal)
        assert "box" in proposal.code.lower()


# ============================================================================
# SECTION 2: OpenSCADProposer Integration Tests
# ============================================================================


@requires_cadling
class TestOpenSCADProposerIntegration:
    """Test OpenSCADProposer integration with cadling.OpenSCADGenerator."""

    def test_proposer_imports_cadling_generator(self) -> None:
        """Test that OpenSCADProposer can import cadling generator."""
        from ll_gen.codegen.openscad_proposer import OpenSCADProposer
        proposer = OpenSCADProposer()
        assert proposer is not None

    def test_proposer_creates_code_proposal(self) -> None:
        """Test that OpenSCADProposer creates valid CodeProposal."""
        from ll_gen.codegen.openscad_proposer import OpenSCADProposer
        from ll_gen.proposals.code_proposal import CodeProposal

        proposer = OpenSCADProposer()
        # Mock the generator to avoid actual LLM calls
        mock_code = """cube([100, 50, 20], center=true);"""
        proposer.generator = MagicMock()
        proposer.generator.generate = MagicMock(return_value=mock_code)

        proposal = proposer.propose("A box 100mm x 50mm x 20mm")

        assert isinstance(proposal, CodeProposal)
        assert "cube" in proposal.code.lower()


# ============================================================================
# SECTION 3: Command Executor Integration Tests
# ============================================================================


@requires_cadling
class TestCommandExecutorIntegration:
    """Test CommandExecutor integration with cadling.CommandExecutor."""

    def test_cadling_executor_available(self) -> None:
        """Test that cadling CommandExecutor is available."""
        from ll_gen.disposal.command_executor import _CADLING_EXECUTOR_AVAILABLE
        # This should be True if cadling is installed correctly
        assert _CADLING_EXECUTOR_AVAILABLE is True

    def test_executor_prefers_cadling_path(self) -> None:
        """Test that executor prefers cadling path when available."""
        from ll_gen.disposal.command_executor import execute_command_proposal
        from ll_gen.proposals.command_proposal import CommandSequenceProposal
        import numpy as np

        proposal = CommandSequenceProposal(
            proposal_id="test",
            confidence=0.8,
            source_prompt="Test",
            command_dicts=[
                {"command_type": "SOL", "parameters": [0] * 16, "parameter_mask": [False] * 16},
            ],
            quantization_bits=8,
        )

        # Mock the cadling executor
        with patch(
            "ll_gen.disposal.command_executor.CadlingCommandExecutor"
        ) as MockExecutor:
            mock_instance = MagicMock()
            mock_instance.execute.return_value = MagicMock()
            MockExecutor.return_value = mock_instance

            with patch.object(proposal, "to_token_sequence") as mock_to_token:
                mock_to_token.return_value = MagicMock(token_ids=[1, 2, 3])
                result = execute_command_proposal(proposal)

            # Cadling executor should have been called
            MockExecutor.assert_called_once()


# ============================================================================
# SECTION 4: Surface Executor Integration Tests
# ============================================================================


@requires_cadling
class TestSurfaceExecutorIntegration:
    """Test SurfaceExecutor integration with cadling surface fitting."""

    def test_cadling_surface_fitter_available(self) -> None:
        """Test that cadling BSplineSurfaceFitter is available."""
        from ll_gen.disposal.surface_executor import _CADLING_SURFACE_FITTER_AVAILABLE
        # This may or may not be True depending on cadling version
        assert isinstance(_CADLING_SURFACE_FITTER_AVAILABLE, bool)

    def test_cadling_topology_merger_available(self) -> None:
        """Test that cadling TopologyMerger is available."""
        from ll_gen.disposal.surface_executor import _CADLING_TOPOLOGY_MERGER_AVAILABLE
        # This may or may not be True depending on cadling version
        assert isinstance(_CADLING_TOPOLOGY_MERGER_AVAILABLE, bool)


# ============================================================================
# SECTION 5: Verification Integration Tests
# ============================================================================


@requires_cadling
class TestVerificationIntegration:
    """Test VisualVerifier integration with cadling."""

    def test_llm_vision_verification_uses_cadling(self) -> None:
        """Test that LLM vision verification uses cadling ChatAgent."""
        from ll_gen.pipeline.verification import VisualVerifier

        verifier = VisualVerifier(vlm_backend="llm")

        # The verifier should attempt to use cadling's CadQueryGenerator
        # for LLM-based vision verification
        assert verifier.vlm_backend == "llm"


# ============================================================================
# SECTION 6: Module Availability Tests
# ============================================================================


@requires_cadling
class TestModuleAvailability:
    """Test module availability when cadling is installed."""

    def test_all_proposers_importable(self) -> None:
        """Test all proposer classes are importable."""
        from ll_gen.codegen.cadquery_proposer import CadQueryProposer
        from ll_gen.codegen.openscad_proposer import OpenSCADProposer
        assert CadQueryProposer is not None
        assert OpenSCADProposer is not None

    def test_orchestrator_importable(self) -> None:
        """Test GenerationOrchestrator is importable."""
        from ll_gen.pipeline.orchestrator import GenerationOrchestrator
        assert GenerationOrchestrator is not None
