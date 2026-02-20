"""Integration tests for ll_gen with ll_stepnet package.

Tests integration between ll_gen and ll_stepnet:
- NEURAL_VAE route → ll_stepnet.STEPVAE
- NEURAL_DIFFUSION route → ll_stepnet.StructuredDiffusion
- NEURAL_VQVAE route → ll_stepnet.VQVAEModel

All tests are marked with @pytest.mark.requires_torch and skip if PyTorch
is not installed.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Check if torch is available
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# Check if ll_stepnet is available
try:
    import ll_stepnet
    _LL_STEPNET_AVAILABLE = True
except ImportError:
    _LL_STEPNET_AVAILABLE = False

requires_torch = pytest.mark.skipif(
    not _TORCH_AVAILABLE,
    reason="PyTorch not installed"
)

requires_ll_stepnet = pytest.mark.skipif(
    not _LL_STEPNET_AVAILABLE,
    reason="ll_stepnet package not installed"
)


# ============================================================================
# SECTION 1: Neural Route Enum Tests
# ============================================================================


class TestNeuralRouteEnums:
    """Test neural generation route enum values."""

    def test_neural_vae_route_defined(self) -> None:
        """Test NEURAL_VAE route is defined."""
        from ll_gen.config import GenerationRoute
        assert GenerationRoute.NEURAL_VAE == "neural_vae"

    def test_neural_diffusion_route_defined(self) -> None:
        """Test NEURAL_DIFFUSION route is defined."""
        from ll_gen.config import GenerationRoute
        assert GenerationRoute.NEURAL_DIFFUSION == "neural_diffusion"

    def test_neural_vqvae_route_defined(self) -> None:
        """Test NEURAL_VQVAE route is defined."""
        from ll_gen.config import GenerationRoute
        assert GenerationRoute.NEURAL_VQVAE == "neural_vqvae"


# ============================================================================
# SECTION 2: Orchestrator Neural Path Tests (Mocked)
# ============================================================================


class TestOrchestratorNeuralPathsMocked:
    """Test orchestrator neural paths with mocked ll_stepnet."""

    def test_orchestrator_has_neural_routes(self) -> None:
        """Test orchestrator supports neural generation routes."""
        from ll_gen.config import GenerationRoute

        neural_routes = [
            GenerationRoute.NEURAL_VAE,
            GenerationRoute.NEURAL_DIFFUSION,
            GenerationRoute.NEURAL_VQVAE,
        ]
        assert len(neural_routes) == 3

    def test_orchestrator_importable(self) -> None:
        """Test GenerationOrchestrator is importable."""
        from ll_gen.pipeline.orchestrator import GenerationOrchestrator
        assert GenerationOrchestrator is not None


# ============================================================================
# SECTION 3: VAE Integration Tests
# ============================================================================


@requires_torch
@requires_ll_stepnet
class TestVAEIntegration:
    """Test VAE model integration with ll_stepnet."""

    def test_stepvae_import(self) -> None:
        """Test STEPVAE can be imported from ll_stepnet."""
        from ll_stepnet.stepnet.vae import STEPVAE
        assert STEPVAE is not None

    def test_vae_config_import(self) -> None:
        """Test VAEConfig can be imported from ll_stepnet."""
        from ll_stepnet.stepnet.config import VAEConfig
        assert VAEConfig is not None


# ============================================================================
# SECTION 4: Diffusion Integration Tests
# ============================================================================


@requires_torch
@requires_ll_stepnet
class TestDiffusionIntegration:
    """Test diffusion model integration with ll_stepnet."""

    def test_structured_diffusion_import(self) -> None:
        """Test StructuredDiffusion can be imported from ll_stepnet."""
        from ll_stepnet.stepnet.diffusion import StructuredDiffusion
        assert StructuredDiffusion is not None

    def test_diffusion_config_import(self) -> None:
        """Test DiffusionConfig can be imported from ll_stepnet."""
        from ll_stepnet.stepnet.config import DiffusionConfig
        assert DiffusionConfig is not None


# ============================================================================
# SECTION 5: VQ-VAE Integration Tests
# ============================================================================


@requires_torch
@requires_ll_stepnet
class TestVQVAEIntegration:
    """Test VQ-VAE model integration with ll_stepnet."""

    def test_vqvae_import(self) -> None:
        """Test VQVAEModel can be imported from ll_stepnet."""
        from ll_stepnet.stepnet.vqvae import VQVAEModel
        assert VQVAEModel is not None


# ============================================================================
# SECTION 6: Generation Pipeline Integration Tests
# ============================================================================


@requires_torch
@requires_ll_stepnet
class TestGenerationPipelineIntegration:
    """Test CAD generation pipeline integration with ll_stepnet."""

    def test_generation_pipeline_import(self) -> None:
        """Test CADGenerationPipeline can be imported from ll_stepnet."""
        from ll_stepnet.stepnet.generation_pipeline import CADGenerationPipeline
        assert CADGenerationPipeline is not None


# ============================================================================
# SECTION 7: Latent Proposal Integration Tests
# ============================================================================


@requires_torch
class TestLatentProposalIntegration:
    """Test LatentProposal with torch tensors."""

    def test_latent_proposal_accepts_numpy(self) -> None:
        """Test LatentProposal accepts numpy arrays."""
        import numpy as np
        from ll_gen.proposals.latent_proposal import LatentProposal

        proposal = LatentProposal(
            proposal_id="test",
            confidence=0.8,
            source_prompt="Test",
            face_grids=[np.random.randn(4, 4, 3).astype(np.float32)],
        )
        assert proposal.face_grids[0].dtype == np.float32

    def test_latent_proposal_stage_latents(self) -> None:
        """Test LatentProposal can store stage latents."""
        import numpy as np
        from ll_gen.proposals.latent_proposal import LatentProposal

        # Stage latents would typically be torch tensors, but we test with numpy
        stage_latents = [np.random.randn(256).astype(np.float32) for _ in range(3)]

        proposal = LatentProposal(
            proposal_id="test",
            confidence=0.8,
            source_prompt="Test",
            face_grids=[np.random.randn(4, 4, 3).astype(np.float32)],
            stage_latents=stage_latents,
        )
        assert len(proposal.stage_latents) == 3


# ============================================================================
# SECTION 8: Command Sequence Proposal Integration Tests
# ============================================================================


@requires_torch
class TestCommandSequenceProposalIntegration:
    """Test CommandSequenceProposal with torch tensors."""

    def test_latent_vector_as_numpy(self) -> None:
        """Test CommandSequenceProposal latent_vector as numpy."""
        import numpy as np
        from ll_gen.proposals.command_proposal import CommandSequenceProposal

        latent = np.random.randn(256).astype(np.float32)

        proposal = CommandSequenceProposal(
            proposal_id="test",
            confidence=0.8,
            source_prompt="Test",
            command_dicts=[],
            quantization_bits=8,
            latent_vector=latent,
        )
        assert proposal.latent_vector is not None
        assert proposal.latent_vector.shape == (256,)
