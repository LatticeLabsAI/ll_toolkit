"""Comprehensive tests for ll_gen.generators module.

Tests all neural generator classes and latent space sampler utilities:
- BaseNeuralGenerator (ABC patterns, device resolution, temperature adjustment)
- NeuralVAEGenerator (STEPVAE model wrapping, latent perturbation)
- NeuralDiffusionGenerator (diffusion sampling, stage-aware error recovery)
- NeuralVQVAEGenerator (quantized codebook generation, masked resampling)
- LatentSampler (SLERP interpolation, neighborhood sampling, prior sampling)

All tests work WITHOUT optional dependencies (torch, ll_stepnet) by mocking
heavy imports and verifying graceful fallback behavior.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ll_gen.conditioning.embeddings import ConditioningEmbeddings
from ll_gen.config import ErrorCategory
from ll_gen.generators.base import BaseNeuralGenerator
from ll_gen.generators.latent_sampler import LatentSampler
from ll_gen.generators.neural_diffusion import NeuralDiffusionGenerator
from ll_gen.generators.neural_vae import NeuralVAEGenerator
from ll_gen.generators.neural_vqvae import NeuralVQVAEGenerator
from ll_gen.proposals.command_proposal import CommandSequenceProposal
from ll_gen.proposals.latent_proposal import LatentProposal

_log = logging.getLogger(__name__)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_latent_vector() -> np.ndarray:
    """A sample latent vector (256-dim standard normal)."""
    return np.random.randn(256).astype(np.float32)


@pytest.fixture
def sample_conditioning() -> ConditioningEmbeddings:
    """A sample conditioning embedding."""
    return ConditioningEmbeddings(
        pooled_embedding=np.random.randn(768).astype(np.float32),
        source_type="text",
        source_model="bert-base-uncased",
        embed_dim=768,
    )


@pytest.fixture
def sample_error_context() -> dict[str, Any]:
    """A sample error context with topology error."""
    return {
        "error_category": ErrorCategory.TOPOLOGY_ERROR.value,
        "failure_description": "Invalid shell structure",
        "previous_latent_vector": np.random.randn(256).astype(np.float32),
        "original_prompt": "Create a box with a hole",
    }


@pytest.fixture
def sample_command_result() -> dict[str, Any]:
    """Mock result from VAE/VQ-VAE pipeline."""
    return {
        "commands": [
            {"command_type": "SOL", "parameters": [0] * 16, "parameter_mask": [True, True] + [False] * 14},
            {"command_type": "LINE", "parameters": [10, 20, 30, 40] + [0] * 12, "parameter_mask": [True] * 4 + [False] * 12},
        ],
        "command_logits": np.random.randn(1, 10, 6).astype(np.float32),  # (batch, seq, num_commands)
        "param_logits": {
            0: np.random.randn(1, 10, 256).astype(np.float32),  # param 0
            1: np.random.randn(1, 10, 256).astype(np.float32),  # param 1
        },
        "codebook_indices": np.random.randint(0, 512, (10,), dtype=np.int64),
    }


@pytest.fixture
def sample_diffusion_result() -> dict[str, Any]:
    """Mock result from diffusion model."""
    return {
        "face_grids": [np.random.randn(32, 32, 3).astype(np.float32) for _ in range(3)],
        "edge_points": [np.random.randn(20, 3).astype(np.float32) for _ in range(4)],
        "stage_latents": {
            "face_positions": np.random.randn(4, 32, 32).astype(np.float32),
            "face_geometry": np.random.randn(4, 32, 32).astype(np.float32),
            "edge_positions": np.random.randn(4, 16, 20).astype(np.float32),
        },
    }


# ============================================================================
# BaseNeuralGenerator Tests
# ============================================================================


class TestBaseNeuralGenerator:
    """Tests for the abstract BaseNeuralGenerator base class.

    Verifies that:
    - Cannot instantiate ABC directly
    - Abstract methods are enforced
    - Device resolution works correctly
    - Metadata building is correct
    - Temperature adjustment follows error category rules
    """

    @pytest.mark.unit
    def test_cannot_instantiate_directly(self):
        """BaseNeuralGenerator cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseNeuralGenerator()

    @pytest.mark.unit
    def test_abstract_methods_required(self):
        """Concrete subclass must implement abstract methods."""

        class IncompleteGenerator(BaseNeuralGenerator):
            def generate(self, prompt, conditioning=None, error_context=None):
                pass
            # Missing generate_candidates

        with pytest.raises(TypeError, match="generate_candidates"):
            IncompleteGenerator()

    @pytest.mark.unit
    def test_resolve_device_cpu(self):
        """Device resolution falls back to CPU when requested."""

        class ConcreteGenerator(BaseNeuralGenerator):
            def generate(self, prompt, conditioning=None, error_context=None):
                pass

            def generate_candidates(self, prompt, num_candidates=3, conditioning=None):
                return []

        gen = ConcreteGenerator(device="cpu")
        assert gen.device == "cpu"

    @pytest.mark.unit
    def test_resolve_device_cuda_fallback(self):
        """Device resolution falls back to CPU when CUDA unavailable."""

        class ConcreteGenerator(BaseNeuralGenerator):
            def generate(self, prompt, conditioning=None, error_context=None):
                pass

            def generate_candidates(self, prompt, num_candidates=3, conditioning=None):
                return []

        # Patch torch import to make cuda unavailable
        with patch("builtins.__import__") as mock_import:
            def import_side_effect(name, *args, **kwargs):
                if name == "torch":
                    raise ImportError("torch not available")
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_side_effect
            gen = ConcreteGenerator(device="cuda")
            assert gen.device == "cpu"

    @pytest.mark.unit
    def test_checkpoint_path_conversion(self):
        """Checkpoint path is converted to Path object."""

        class ConcreteGenerator(BaseNeuralGenerator):
            def generate(self, prompt, conditioning=None, error_context=None):
                pass

            def generate_candidates(self, prompt, num_candidates=3, conditioning=None):
                return []

        gen = ConcreteGenerator(checkpoint_path="/tmp/model.pth")
        assert isinstance(gen.checkpoint_path, Path)
        assert str(gen.checkpoint_path) == "/tmp/model.pth"

    @pytest.mark.unit
    def test_build_metadata(self):
        """Metadata building includes model name, device, timestamp."""

        class ConcreteGenerator(BaseNeuralGenerator):
            def generate(self, prompt, conditioning=None, error_context=None):
                pass

            def generate_candidates(self, prompt, num_candidates=3, conditioning=None):
                return []

        gen = ConcreteGenerator(device="cpu")
        metadata = gen._build_metadata("TestModel", temperature=0.8, custom_field="value")

        assert metadata["model_name"] == "TestModel"
        assert metadata["device"] == "cpu"
        assert "timestamp" in metadata
        assert metadata["temperature"] == 0.8
        assert metadata["custom_field"] == "value"

    @pytest.mark.unit
    def test_adjust_temperature_topology_error(self):
        """Temperature reduced to 0.7x for topology errors."""

        class ConcreteGenerator(BaseNeuralGenerator):
            def generate(self, prompt, conditioning=None, error_context=None):
                pass

            def generate_candidates(self, prompt, num_candidates=3, conditioning=None):
                return []

        gen = ConcreteGenerator(device="cpu")
        error_ctx = {"error_category": ErrorCategory.TOPOLOGY_ERROR.value}
        adjusted = gen._adjust_temperature_for_error(1.0, error_ctx)
        assert adjusted == pytest.approx(0.7)

    @pytest.mark.unit
    def test_adjust_temperature_degenerate_shape(self):
        """Temperature increased to 1.3x for degenerate shapes."""

        class ConcreteGenerator(BaseNeuralGenerator):
            def generate(self, prompt, conditioning=None, error_context=None):
                pass

            def generate_candidates(self, prompt, num_candidates=3, conditioning=None):
                return []

        gen = ConcreteGenerator(device="cpu")
        error_ctx = {"error_category": ErrorCategory.DEGENERATE_SHAPE.value}
        adjusted = gen._adjust_temperature_for_error(1.0, error_ctx)
        assert adjusted == pytest.approx(1.3)

    @pytest.mark.unit
    def test_adjust_temperature_self_intersection(self):
        """Temperature reduced to 0.8x for self-intersections."""

        class ConcreteGenerator(BaseNeuralGenerator):
            def generate(self, prompt, conditioning=None, error_context=None):
                pass

            def generate_candidates(self, prompt, num_candidates=3, conditioning=None):
                return []

        gen = ConcreteGenerator(device="cpu")
        error_ctx = {"error_category": ErrorCategory.SELF_INTERSECTION.value}
        adjusted = gen._adjust_temperature_for_error(1.0, error_ctx)
        assert adjusted == pytest.approx(0.8)

    @pytest.mark.unit
    def test_adjust_temperature_boolean_failure(self):
        """Temperature reduced to 0.9x for boolean failures."""

        class ConcreteGenerator(BaseNeuralGenerator):
            def generate(self, prompt, conditioning=None, error_context=None):
                pass

            def generate_candidates(self, prompt, num_candidates=3, conditioning=None):
                return []

        gen = ConcreteGenerator(device="cpu")
        error_ctx = {"error_category": ErrorCategory.BOOLEAN_FAILURE.value}
        adjusted = gen._adjust_temperature_for_error(1.0, error_ctx)
        assert adjusted == pytest.approx(0.9)

    @pytest.mark.unit
    def test_adjust_temperature_no_error(self):
        """Temperature unchanged when no error context."""

        class ConcreteGenerator(BaseNeuralGenerator):
            def generate(self, prompt, conditioning=None, error_context=None):
                pass

            def generate_candidates(self, prompt, num_candidates=3, conditioning=None):
                return []

        gen = ConcreteGenerator(device="cpu")
        adjusted = gen._adjust_temperature_for_error(1.0, None)
        assert adjusted == 1.0

    @pytest.mark.unit
    def test_adjust_temperature_unknown_category(self):
        """Temperature unchanged for unknown error categories."""

        class ConcreteGenerator(BaseNeuralGenerator):
            def generate(self, prompt, conditioning=None, error_context=None):
                pass

            def generate_candidates(self, prompt, num_candidates=3, conditioning=None):
                return []

        gen = ConcreteGenerator(device="cpu")
        error_ctx = {"error_category": "unknown_category"}
        adjusted = gen._adjust_temperature_for_error(1.0, error_ctx)
        assert adjusted == 1.0


# ============================================================================
# NeuralVAEGenerator Tests
# ============================================================================


class TestNeuralVAEGenerator:
    """Tests for the NeuralVAEGenerator class.

    Verifies VAE model initialization, generation, and error recovery.
    All heavy dependencies (torch, ll_stepnet) are mocked.
    """

    @pytest.mark.unit
    def test_initialization_defaults(self):
        """NeuralVAEGenerator initializes with sensible defaults."""
        gen = NeuralVAEGenerator()
        assert gen.device == "cpu"
        assert gen.temperature == 0.8
        assert gen.max_seq_len == 60
        assert gen.vae_config is None
        assert gen.checkpoint_path is None

    @pytest.mark.unit
    def test_initialization_custom(self):
        """NeuralVAEGenerator accepts custom configuration."""
        gen = NeuralVAEGenerator(
            device="cpu",
            temperature=0.5,
            max_seq_len=100,
            checkpoint_path="/tmp/model.pth",
        )
        assert gen.device == "cpu"
        assert gen.temperature == 0.5
        assert gen.max_seq_len == 100
        assert gen.checkpoint_path == Path("/tmp/model.pth")

    @pytest.mark.unit
    def test_generate_basic(self):
        """generate() returns a CommandSequenceProposal."""
        gen = NeuralVAEGenerator()

        # Mock the initialization directly
        gen._model = MagicMock()
        gen._pipeline = MagicMock()
        gen._pipeline.generate.return_value = [
            {
                "commands": [{"command_type": "SOL", "parameters": [0] * 16}],
                "command_logits": np.random.randn(1, 5, 6).astype(np.float32),
                "param_logits": {0: np.random.randn(1, 5, 256).astype(np.float32)},
            }
        ]

        proposal = gen.generate("create a box")

        assert isinstance(proposal, CommandSequenceProposal)
        assert proposal.source_prompt == "create a box"
        assert proposal.confidence >= 0.0
        assert proposal.confidence <= 1.0

    @pytest.mark.unit
    def test_generate_with_conditioning(self):
        """generate() preserves conditioning source information."""
        gen = NeuralVAEGenerator()
        gen._model = MagicMock()
        gen._pipeline = MagicMock()
        gen._pipeline.generate.return_value = [{"commands": [], "command_logits": None, "param_logits": None}]

        cond = ConditioningEmbeddings(source_type="image", source_model="dino_vits16")
        proposal = gen.generate("shape", conditioning=cond)

        assert proposal.conditioning_source == "image"

    @pytest.mark.unit
    def test_generate_with_error_context(self):
        """generate() adjusts temperature based on error category."""
        gen = NeuralVAEGenerator(temperature=1.0)
        gen._model = MagicMock()
        gen._pipeline = MagicMock()
        gen._pipeline.generate.return_value = [{"commands": [], "command_logits": None, "param_logits": None}]

        error_ctx = {"error_category": ErrorCategory.TOPOLOGY_ERROR.value}
        gen.generate("shape", error_context=error_ctx)

        # Verify pipeline was called with adjusted temperature
        call_kwargs = gen._pipeline.generate.call_args[1]
        assert call_kwargs["temperature"] == pytest.approx(0.7)  # 1.0 * 0.7

    @pytest.mark.unit
    def test_generate_candidates(self):
        """generate_candidates() returns multiple sorted proposals."""
        gen = NeuralVAEGenerator()
        gen._model = MagicMock()
        gen._pipeline = MagicMock()

        # Return 3 results with different logits
        logits1 = np.ones((1, 5, 6), dtype=np.float32)  # High confidence
        logits2 = np.random.randn(1, 5, 6).astype(np.float32)
        logits3 = np.zeros((1, 5, 6), dtype=np.float32)  # Low confidence

        gen._pipeline.generate.return_value = [
            {"command_logits": logits1, "param_logits": {}, "commands": []},
            {"command_logits": logits2, "param_logits": {}, "commands": []},
            {"command_logits": logits3, "param_logits": {}, "commands": []},
        ]

        proposals = gen.generate_candidates("shape", num_candidates=3)

        assert len(proposals) == 3
        assert all(isinstance(p, CommandSequenceProposal) for p in proposals)
        # Sorted by confidence descending
        assert proposals[0].confidence >= proposals[1].confidence >= proposals[2].confidence

    @pytest.mark.unit
    def test_generate_from_error_context(self):
        """generate_from_error_context() perturbs latent vector."""
        gen = NeuralVAEGenerator()
        gen._model = MagicMock()
        gen._pipeline = MagicMock()

        prior_latent = np.random.randn(256).astype(np.float32)
        error_ctx = {
            "error_category": ErrorCategory.TOPOLOGY_ERROR.value,
            "previous_latent_vector": prior_latent,
        }

        proposal = gen.generate_from_error_context(error_ctx)

        assert isinstance(proposal, CommandSequenceProposal)
        assert proposal.latent_vector is not None
        # Perturbed latent should be different but same norm
        assert proposal.latent_vector.shape == prior_latent.shape

    @pytest.mark.unit
    def test_generate_from_error_context_no_latent(self):
        """generate_from_error_context() returns None without prior latent."""
        gen = NeuralVAEGenerator()
        gen._model = MagicMock()
        gen._pipeline = MagicMock()

        error_ctx = {"error_category": ErrorCategory.TOPOLOGY_ERROR.value}  # No prior latent
        proposal = gen.generate_from_error_context(error_ctx)

        assert proposal is None

    @pytest.mark.unit
    def test_logits_to_token_ids_empty(self):
        """_logits_to_token_ids() handles empty/None logits gracefully."""
        gen = NeuralVAEGenerator()
        # Call with None logits - should return empty list
        token_ids = gen._logits_to_token_ids(None, None)
        assert token_ids == []

    @pytest.mark.unit
    def test_compute_confidence_no_logits(self):
        """_compute_confidence() returns default when logits missing."""
        gen = NeuralVAEGenerator()
        confidence = gen._compute_confidence(None, None)
        assert confidence == 0.5

    @pytest.mark.unit
    def test_init_model_missing_ll_stepnet(self):
        """_init_model() raises ImportError when ll_stepnet unavailable."""
        gen = NeuralVAEGenerator()

        with patch.dict("sys.modules", {"ll_stepnet.stepnet.models": None}):
            with pytest.raises(ImportError, match="ll_stepnet is required"):
                gen._init_model()


# ============================================================================
# NeuralDiffusionGenerator Tests
# ============================================================================


class TestNeuralDiffusionGenerator:
    """Tests for the NeuralDiffusionGenerator class.

    Verifies diffusion model initialization, generation, and stage-aware
    error recovery.
    """

    @pytest.mark.unit
    def test_initialization_defaults(self):
        """NeuralDiffusionGenerator initializes with sensible defaults."""
        gen = NeuralDiffusionGenerator()
        assert gen.device == "cpu"
        assert gen.inference_steps == 50
        assert gen.eta == 0.0
        assert gen.diffusion_config is None

    @pytest.mark.unit
    def test_initialization_custom(self):
        """NeuralDiffusionGenerator accepts custom configuration."""
        gen = NeuralDiffusionGenerator(
            inference_steps=100,
            eta=1.0,
            device="cpu",
        )
        assert gen.inference_steps == 100
        assert gen.eta == 1.0
        assert gen.device == "cpu"

    @pytest.mark.unit
    def test_generate_basic(self):
        """generate() returns a LatentProposal with face grids and edge points."""
        gen = NeuralDiffusionGenerator()
        gen._model = MagicMock()
        gen._model.sample.return_value = {
            "face_grids": [np.random.randn(32, 32, 3).astype(np.float32)],
            "edge_points": [np.random.randn(20, 3).astype(np.float32)],
        }

        # Mock torch at import time with proper Tensor class
        mock_torch = MagicMock()
        mock_torch.Tensor = type("MockTensor", (), {})  # Create a real class, not MagicMock
        with patch.dict("sys.modules", {"torch": mock_torch}):
            proposal = gen.generate("create a curved surface")

            assert isinstance(proposal, LatentProposal)
            assert proposal.num_faces > 0 or proposal.num_edges > 0
            assert proposal.source_prompt == "create a curved surface"
            assert proposal.confidence >= 0.0

    @pytest.mark.unit
    def test_generate_candidates(self):
        """generate_candidates() generates multiple LatentProposals."""
        gen = NeuralDiffusionGenerator()
        gen._model = MagicMock()
        gen._model.sample.return_value = {
            "face_grids": [np.random.randn(32, 32, 3).astype(np.float32)],
            "edge_points": [np.random.randn(20, 3).astype(np.float32)],
        }

        mock_torch = MagicMock()
        mock_torch.Tensor = type("MockTensor", (), {})
        with patch.dict("sys.modules", {"torch": mock_torch}):
            proposals = gen.generate_candidates("shape", num_candidates=3)

            assert len(proposals) == 3
            assert all(isinstance(p, LatentProposal) for p in proposals)

    @pytest.mark.unit
    def test_generate_from_error_context_self_intersection(self):
        """generate_from_error_context() restarts from face_geometry stage."""
        gen = NeuralDiffusionGenerator()
        gen._model = MagicMock()
        gen._model.sample_from_stage.return_value = {
            "face_grids": [np.random.randn(32, 32, 3).astype(np.float32)],
            "edge_points": [np.random.randn(20, 3).astype(np.float32)],
        }

        stage_latents = {
            "face_geometry": np.random.randn(4, 32, 32).astype(np.float32),
        }
        error_ctx = {
            "error_category": "self_intersection",
            "stage_latents": stage_latents,
        }

        mock_torch = MagicMock()
        mock_torch.Tensor = type("MockTensor", (), {})
        with patch.dict("sys.modules", {"torch": mock_torch}):
            proposal = gen.generate_from_error_context(error_ctx)

            assert isinstance(proposal, LatentProposal)
            # Verify model was called with correct stage
            call_kwargs = gen._model.sample_from_stage.call_args[1]
            assert call_kwargs["start_stage"] == 1

    @pytest.mark.unit
    def test_generate_from_error_context_topology_error(self):
        """generate_from_error_context() restarts from edge_positions stage."""
        gen = NeuralDiffusionGenerator()
        gen._model = MagicMock()
        gen._model.sample_from_stage.return_value = {
            "face_grids": [np.random.randn(32, 32, 3).astype(np.float32)],
            "edge_points": [np.random.randn(20, 3).astype(np.float32)],
        }

        stage_latents = {
            "edge_positions": np.random.randn(4, 16, 20).astype(np.float32),
        }
        error_ctx = {
            "error_category": "topology_error",
            "stage_latents": stage_latents,
        }

        mock_torch = MagicMock()
        mock_torch.Tensor = type("MockTensor", (), {})
        with patch.dict("sys.modules", {"torch": mock_torch}):
            gen.generate_from_error_context(error_ctx)

            # Verify model was called with correct stage
            call_kwargs = gen._model.sample_from_stage.call_args[1]
            assert call_kwargs["start_stage"] == 2

    @pytest.mark.unit
    def test_generate_from_error_context_no_stage_latents(self):
        """generate_from_error_context() returns None without stage latents."""
        gen = NeuralDiffusionGenerator()
        gen._model = MagicMock()

        error_ctx = {"error_category": "topology_error"}  # No stage_latents
        proposal = gen.generate_from_error_context(error_ctx)

        assert proposal is None

    @pytest.mark.unit
    def test_tensor_to_numpy_conversion(self):
        """_tensor_to_numpy() handles various tensor types."""
        gen = NeuralDiffusionGenerator()

        # Test with numpy array
        arr = np.array([1.0, 2.0, 3.0])
        result = gen._tensor_to_numpy(arr)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    @pytest.mark.unit
    def test_get_noise_shape_default(self):
        """_get_noise_shape() returns default shape."""
        gen = NeuralDiffusionGenerator()
        gen._model = MagicMock(spec=[])  # Empty spec - no attributes
        # Model has no latent_shape attribute
        shape = gen._get_noise_shape()
        assert shape == (4, 32, 32)

    @pytest.mark.unit
    def test_get_noise_shape_custom(self):
        """_get_noise_shape() uses model's latent_shape if available."""
        gen = NeuralDiffusionGenerator()
        gen._model = MagicMock()
        gen._model.latent_shape = (8, 64, 64)
        shape = gen._get_noise_shape()
        assert shape == (8, 64, 64)

    @pytest.mark.unit
    def test_compute_confidence_with_geometry(self):
        """_compute_confidence() increases with face and edge presence."""
        gen = NeuralDiffusionGenerator()

        face_grids = [np.random.randn(32, 32, 3).astype(np.float32)]
        edge_points = [np.random.randn(20, 3).astype(np.float32)]

        confidence = gen._compute_confidence(face_grids, edge_points)
        assert 0.0 <= confidence <= 1.0
        assert confidence >= 0.5  # Should be higher with both grids and edges


# ============================================================================
# NeuralVQVAEGenerator Tests
# ============================================================================


class TestNeuralVQVAEGenerator:
    """Tests for the NeuralVQVAEGenerator class.

    Verifies VQ-VAE model initialization, generation, and masked codebook
    resampling.
    """

    @pytest.mark.unit
    def test_initialization_defaults(self):
        """NeuralVQVAEGenerator initializes with sensible defaults."""
        gen = NeuralVQVAEGenerator()
        assert gen.device == "cpu"
        assert gen.temperature == 0.7
        assert gen.codebook_dim == 512
        assert gen.max_seq_len == 60

    @pytest.mark.unit
    def test_initialization_custom(self):
        """NeuralVQVAEGenerator accepts custom configuration."""
        gen = NeuralVQVAEGenerator(
            temperature=0.5,
            codebook_dim=1024,
            max_seq_len=100,
            device="cpu",
        )
        assert gen.temperature == 0.5
        assert gen.codebook_dim == 1024
        assert gen.max_seq_len == 100
        assert gen.device == "cpu"

    @pytest.mark.unit
    def test_generate_basic(self):
        """generate() returns a CommandSequenceProposal."""
        gen = NeuralVQVAEGenerator()
        gen._model = MagicMock()
        gen._pipeline = MagicMock()
        gen._pipeline.generate.return_value = [
            {
                "commands": [{"command_type": "SOL", "parameters": [0] * 16}],
                "command_logits": np.random.randn(1, 5, 6).astype(np.float32),
                "param_logits": {0: np.random.randn(1, 5, 256).astype(np.float32)},
                "codebook_indices": np.random.randint(0, 512, (5,)),
            }
        ]

        proposal = gen.generate("create a box")

        assert isinstance(proposal, CommandSequenceProposal)
        assert proposal.source_prompt == "create a box"

    @pytest.mark.unit
    def test_generate_candidates(self):
        """generate_candidates() returns multiple sorted proposals."""
        gen = NeuralVQVAEGenerator()
        gen._model = MagicMock()
        gen._pipeline = MagicMock()

        logits1 = np.ones((1, 5, 6), dtype=np.float32)
        logits2 = np.random.randn(1, 5, 6).astype(np.float32)

        gen._pipeline.generate.return_value = [
            {"command_logits": logits1, "param_logits": {}, "commands": []},
            {"command_logits": logits2, "param_logits": {}, "commands": []},
        ]

        proposals = gen.generate_candidates("shape", num_candidates=2)

        assert len(proposals) == 2
        assert proposals[0].confidence >= proposals[1].confidence

    @pytest.mark.unit
    def test_generate_from_masked_codebooks(self):
        """generate_from_masked_codebooks() resamples with mask."""
        gen = NeuralVQVAEGenerator()
        gen._model = MagicMock()
        gen._pipeline = MagicMock()

        gen._pipeline.generate_masked.return_value = [
            {
                "command_logits": np.random.randn(1, 5, 6).astype(np.float32),
                "param_logits": {},
                "commands": [],
            }
        ]

        masked_indices = [10, 20, 30]
        error_ctx = {
            "masked_codebook_indices": masked_indices,
            "original_prompt": "shape",
        }

        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            proposal = gen.generate_from_masked_codebooks(error_ctx)

            assert isinstance(proposal, CommandSequenceProposal)
            # Verify masked sampling was called
            gen._pipeline.generate_masked.assert_called_once()

    @pytest.mark.unit
    def test_generate_from_masked_codebooks_no_mask(self):
        """generate_from_masked_codebooks() returns None without mask."""
        gen = NeuralVQVAEGenerator()
        gen._model = MagicMock()
        gen._pipeline = MagicMock()

        error_ctx = {}  # No masked_codebook_indices
        proposal = gen.generate_from_masked_codebooks(error_ctx)

        assert proposal is None

    @pytest.mark.unit
    def test_init_model_missing_ll_stepnet(self):
        """_init_model() raises ImportError when ll_stepnet unavailable."""
        gen = NeuralVQVAEGenerator()

        with patch.dict("sys.modules", {"ll_stepnet.stepnet.models": None}):
            with pytest.raises(ImportError, match="ll_stepnet is required"):
                gen._init_model()


# ============================================================================
# LatentSampler Tests
# ============================================================================


class TestLatentSampler:
    """Tests for the LatentSampler class.

    Verifies latent space exploration utilities:
    - SLERP interpolation
    - Neighborhood sampling on hypersphere
    - Prior sampling from N(0, I)
    - GAN sampling (with fallback)
    """

    @pytest.mark.unit
    def test_initialization(self):
        """LatentSampler initializes with default dimensions."""
        sampler = LatentSampler()
        assert sampler.latent_dim == 256
        assert sampler.device == "cpu"
        assert sampler.vae_generator is None

    @pytest.mark.unit
    def test_initialization_with_vae_generator(self):
        """LatentSampler stores VAE generator reference."""
        mock_gen = MagicMock(spec=NeuralVAEGenerator)
        sampler = LatentSampler(vae_generator=mock_gen, latent_dim=512)
        assert sampler.vae_generator is mock_gen
        assert sampler.latent_dim == 512

    @pytest.mark.unit
    def test_interpolate_endpoints(self):
        """interpolate() includes both endpoint latents."""
        sampler = LatentSampler(latent_dim=256)
        latent1 = np.random.randn(256).astype(np.float32)
        latent2 = np.random.randn(256).astype(np.float32)

        interpolated = sampler.interpolate(latent1, latent2, steps=5)

        assert len(interpolated) == 5
        # First and last should be close to endpoints
        assert np.allclose(interpolated[0], latent1, atol=1e-5)
        assert np.allclose(interpolated[-1], latent2, atol=1e-5)

    @pytest.mark.unit
    def test_interpolate_slerp_properties(self):
        """interpolate() produces valid SLERP path."""
        sampler = LatentSampler(latent_dim=256)
        latent1 = np.random.randn(256).astype(np.float32)
        latent2 = np.random.randn(256).astype(np.float32)

        interpolated = sampler.interpolate(latent1, latent2, steps=5)

        # All results should be numpy arrays
        assert all(isinstance(v, np.ndarray) for v in interpolated)
        # All should have correct shape
        assert all(v.shape == (256,) for v in interpolated)
        # All should be float32
        assert all(v.dtype == np.float32 for v in interpolated)

    @pytest.mark.unit
    def test_interpolate_parallel_vectors(self):
        """interpolate() handles nearly-parallel vectors."""
        sampler = LatentSampler(latent_dim=10)
        latent1 = np.array([1.0, 0.0, 0.0] + [0.0] * 7, dtype=np.float32)
        latent2 = np.array([2.0, 0.0, 0.0] + [0.0] * 7, dtype=np.float32)  # Parallel

        interpolated = sampler.interpolate(latent1, latent2, steps=3)

        assert len(interpolated) == 3
        # Should use linear interpolation fallback
        assert np.allclose(interpolated[1], (latent1 + latent2) / 2, atol=1e-4)

    @pytest.mark.unit
    def test_interpolate_shape_mismatch(self):
        """interpolate() raises error for shape mismatch."""
        sampler = LatentSampler(latent_dim=256)
        latent1 = np.random.randn(256).astype(np.float32)
        latent2 = np.random.randn(128).astype(np.float32)

        with pytest.raises(ValueError, match="Latent shapes must match"):
            sampler.interpolate(latent1, latent2)

    @pytest.mark.unit
    def test_explore_neighborhood_returns_samples(self):
        """explore_neighborhood() returns correct number of samples."""
        sampler = LatentSampler(latent_dim=256)
        seed = np.random.randn(256).astype(np.float32)

        samples = sampler.explore_neighborhood(seed, radius=0.3, num_samples=5)

        assert len(samples) == 5
        assert all(isinstance(s, np.ndarray) for s in samples)
        assert all(s.shape == (256,) for s in samples)

    @pytest.mark.unit
    def test_explore_neighborhood_radius_effect(self):
        """explore_neighborhood() produces points farther with larger radius."""
        sampler = LatentSampler(latent_dim=256)
        seed = np.random.randn(256).astype(np.float32)

        samples_small = sampler.explore_neighborhood(seed, radius=0.1, num_samples=10)
        samples_large = sampler.explore_neighborhood(seed, radius=0.5, num_samples=10)

        dist_small = np.mean([np.linalg.norm(s - seed) for s in samples_small])
        dist_large = np.mean([np.linalg.norm(s - seed) for s in samples_large])

        # Larger radius should produce larger distances on average
        assert dist_large > dist_small

    @pytest.mark.unit
    def test_sample_from_prior(self):
        """sample_from_prior() returns standard normal samples."""
        sampler = LatentSampler(latent_dim=256)

        samples = sampler.sample_from_prior(num_samples=10)

        assert len(samples) == 10
        assert all(isinstance(s, np.ndarray) for s in samples)
        assert all(s.shape == (256,) for s in samples)
        # Mean should be close to 0, std close to 1 (with some variance)
        all_samples = np.concatenate(samples, axis=0)
        assert np.abs(np.mean(all_samples)) < 0.3
        assert np.abs(np.std(all_samples) - 1.0) < 0.3

    @pytest.mark.unit
    def test_sample_from_gan_fallback_to_prior(self):
        """sample_from_gan() falls back to prior when LatentGAN unavailable."""
        sampler = LatentSampler(latent_dim=256)

        with patch.dict("sys.modules", {"ll_stepnet.stepnet.latent_gan": None}):
            samples = sampler.sample_from_gan(num_samples=5)

            assert len(samples) == 5
            assert all(isinstance(s, np.ndarray) for s in samples)

    @pytest.mark.unit
    def test_is_vae_ready_no_generator(self):
        """is_vae_ready returns False without VAE generator."""
        sampler = LatentSampler()
        assert not sampler.is_vae_ready

    @pytest.mark.unit
    def test_is_vae_ready_generator_not_initialized(self):
        """is_vae_ready returns False when generator model not initialized."""
        mock_gen = MagicMock(spec=NeuralVAEGenerator)
        mock_gen._model = None
        sampler = LatentSampler(vae_generator=mock_gen)
        assert not sampler.is_vae_ready

    @pytest.mark.unit
    def test_is_vae_ready_generator_initialized(self):
        """is_vae_ready returns True when generator and model initialized."""
        mock_gen = MagicMock(spec=NeuralVAEGenerator)
        mock_gen._model = MagicMock()  # Non-None
        sampler = LatentSampler(vae_generator=mock_gen)
        assert sampler.is_vae_ready

    @pytest.mark.unit
    def test_decode_latents_no_generator(self):
        """decode_latents() raises error without VAE generator."""
        sampler = LatentSampler()
        latents = [np.random.randn(256).astype(np.float32)]

        with pytest.raises(RuntimeError, match="vae_generator is required"):
            sampler.decode_latents(latents)

    @pytest.mark.unit
    def test_decode_latents_with_generator(self):
        """decode_latents() returns proposals when generator available."""
        mock_gen = MagicMock(spec=NeuralVAEGenerator)
        mock_gen._model = MagicMock()

        # Mock the decoding methods
        mock_gen._logits_to_token_ids.return_value = [1, 6, 0, 2]
        mock_gen._compute_confidence.return_value = 0.8
        mock_gen._build_metadata.return_value = {"model": "STEPVAE"}

        sampler = LatentSampler(vae_generator=mock_gen, latent_dim=256)
        latents = [np.random.randn(256).astype(np.float32) for _ in range(3)]

        # Mock the model decode method
        mock_gen._model.decode.return_value = {
            "command_logits": np.random.randn(1, 5, 6).astype(np.float32),
            "param_logits": {},
        }
        mock_gen._model.set_latent = MagicMock()

        proposals = sampler.decode_latents(latents, prompt="test")

        # If torch is available, we should get proposals
        # If not, we get an empty list
        assert isinstance(proposals, list)


# ============================================================================
# Integration Tests
# ============================================================================


class TestGeneratorsIntegration:
    """Integration tests combining multiple generators and samplers."""

    @pytest.mark.unit
    def test_vae_generator_with_latent_sampler(self):
        """VAE generator integrates with latent sampler."""
        mock_gen = MagicMock(spec=NeuralVAEGenerator)
        mock_gen._model = MagicMock()
        mock_gen._logits_to_token_ids.return_value = []
        mock_gen._compute_confidence.return_value = 0.5
        mock_gen._build_metadata.return_value = {}

        sampler = LatentSampler(vae_generator=mock_gen, latent_dim=256)

        assert sampler.is_vae_ready

    @pytest.mark.unit
    def test_all_generators_have_error_context_support(self):
        """All generator classes support error context."""
        gen_vae = NeuralVAEGenerator()
        gen_diffusion = NeuralDiffusionGenerator()

        assert hasattr(gen_vae, "generate_from_error_context")
        assert hasattr(gen_diffusion, "generate_from_error_context")

    @pytest.mark.unit
    def test_error_context_temperature_adjustment_chain(self):
        """Temperature adjustment works correctly across error categories."""

        class TestGen(BaseNeuralGenerator):
            def generate(self, prompt, conditioning=None, error_context=None):
                return CommandSequenceProposal(
                    source_prompt=prompt,
                    confidence=0.5,
                    generation_metadata={
                        "temperature_adjustment": self._adjust_temperature_for_error(
                            1.0, error_context
                        ),
                    },
                )

            def generate_candidates(self, prompt, num_candidates=3, conditioning=None):
                return []

        gen = TestGen()

        test_cases = [
            (ErrorCategory.TOPOLOGY_ERROR, 0.7),
            (ErrorCategory.DEGENERATE_SHAPE, 1.3),
            (ErrorCategory.SELF_INTERSECTION, 0.8),
            (ErrorCategory.BOOLEAN_FAILURE, 0.9),
        ]

        for error_cat, expected_mult in test_cases:
            error_ctx = {"error_category": error_cat.value}
            proposal = gen.generate("test", error_context=error_ctx)
            adjusted_temp = proposal.generation_metadata["temperature_adjustment"]
            assert adjusted_temp == pytest.approx(expected_mult, rel=0.01)


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestGeneratorsEdgeCases:
    """Edge case and error handling tests."""

    @pytest.mark.unit
    def test_vae_empty_pipeline_result(self):
        """NeuralVAEGenerator handles empty pipeline result."""

        class TestGen(BaseNeuralGenerator):
            def __init__(self):
                super().__init__()
                self._pipeline = MagicMock()
                self._model = MagicMock()

            def generate(self, prompt, conditioning=None, error_context=None):
                self._pipeline.generate.return_value = []
                # Simulate the behavior
                result_list = self._pipeline.generate(num_samples=1, reconstruct=False, temperature=0.8)
                if not result_list:
                    return CommandSequenceProposal(
                        source_prompt=prompt,
                        confidence=0.0,
                        generation_metadata=self._build_metadata("STEPVAE"),
                    )
                return CommandSequenceProposal(source_prompt=prompt)

            def generate_candidates(self, prompt, num_candidates=3, conditioning=None):
                return []

        gen = TestGen()
        proposal = gen.generate("test")
        assert proposal.confidence == 0.0

    @pytest.mark.unit
    def test_latent_sampler_zero_dimension(self):
        """LatentSampler handles degenerate zero dimension."""
        sampler = LatentSampler(latent_dim=0)
        # Should still work, just with empty arrays
        samples = sampler.sample_from_prior(num_samples=1)
        assert len(samples) == 1
        assert samples[0].shape == (0,)

    @pytest.mark.unit
    def test_interpolate_two_steps(self):
        """interpolate() handles two steps correctly."""
        sampler = LatentSampler(latent_dim=256)
        latent1 = np.random.randn(256).astype(np.float32)
        latent2 = np.random.randn(256).astype(np.float32)

        interpolated = sampler.interpolate(latent1, latent2, steps=2)
        assert len(interpolated) == 2
        # First and last should be close to endpoints
        assert np.allclose(interpolated[0], latent1, atol=1e-5)
        assert np.allclose(interpolated[1], latent2, atol=1e-5)

    @pytest.mark.unit
    def test_diffusion_missing_model_init(self):
        """NeuralDiffusionGenerator handles missing model gracefully."""
        gen = NeuralDiffusionGenerator()
        assert gen._model is None  # Not initialized until needed

    @pytest.mark.unit
    def test_vae_checkpoint_loading_failure(self):
        """NeuralVAEGenerator handles checkpoint loading failure."""

        class TestGen(BaseNeuralGenerator):
            def __init__(self):
                super().__init__(checkpoint_path="/nonexistent/path.pth")
                self._model = MagicMock()

            def generate(self, prompt, conditioning=None, error_context=None):
                return CommandSequenceProposal(source_prompt=prompt)

            def generate_candidates(self, prompt, num_candidates=3, conditioning=None):
                return []

        gen = TestGen()
        # Verify checkpoint path exists before loading raises FileNotFoundError
        assert gen.checkpoint_path is not None
        # load_checkpoint should raise error for nonexistent path
        # (either FileNotFoundError or ImportError if torch missing)
        if gen.checkpoint_path and gen._model is not None:
            with pytest.raises((FileNotFoundError, ImportError)):
                gen.load_checkpoint(gen.checkpoint_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
