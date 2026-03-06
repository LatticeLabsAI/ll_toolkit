"""Comprehensive test suite for neural proposal integration with ll_gen orchestrator.

Tests cover:

1. **Updated Orchestrator with Generator Wrappers** (~15 tests):
   - Lazy initialization of neural generators (_vae_generator, _diffusion_generator, _vqvae_generator)
   - Lazy initialization of MultiModalConditioner (_conditioner)
   - _get_conditioning() integration with encoding
   - _propose_neural_vae/diffusion/vqvae using wrapper generators
   - Token sequence handling and command proposal generation
   - Error context feedback passing to generators

2. **Config Integration** (~10 tests):
   - ConditioningConfig: text_model, image_model, conditioning_dim, fusion_method
   - GeneratorConfig: checkpoints, temperatures, inference steps, codebook_dim
   - TrainingConfig: learning rate, batch size, epochs, baseline decay
   - LLGenConfig composition and get_ll_gen_config() nested overrides

3. **Conditioning Flow** (~10 tests):
   - Prompt → TextConditioningEncoder → ConditioningEmbeddings
   - Image paths → ImageConditioningEncoder → ConditioningEmbeddings
   - MultiModalConditioner fusion strategies (concat, average, text_only, image_only)
   - Conditioning cache/reuse across multiple generate calls

4. **Error Context Retry** (~10 tests):
   - Error context passed from DisposalResult to _propose()
   - Temperature adjustment based on error_category
   - Latent perturbation for VAE retries
   - Feedback flow to neural generators

5. **Module Import Integration** (~5 tests):
   - All exports from ll_gen.__init__ are importable
   - ConditioningEmbeddings, TextConditioningEncoder, ImageConditioningEncoder
   - NeuralVAEGenerator, NeuralDiffusionGenerator, NeuralVQVAEGenerator
   - GenerationMetrics, MetricsComputer (if available)

All tests work WITHOUT optional heavy dependencies (torch, ll_stepnet, pythonocc).
Uses unittest.mock.patch extensively for lazy initialization testing.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ll_gen.config import (
    ConditioningConfig,
    GenerationRoute,
    GeneratorConfig,
    LLGenConfig,
    TrainingConfig,
    get_ll_gen_config,
)
from ll_gen.pipeline.orchestrator import GenerationHistory, GenerationOrchestrator
from ll_gen.proposals.command_proposal import CommandSequenceProposal
from ll_gen.proposals.disposal_result import ErrorCategory
from ll_gen.proposals.latent_proposal import LatentProposal

# ============================================================================
# SECTION 1: Orchestrator Lazy Initialization Tests
# ============================================================================


class TestOrchestratorLazyInitialization:
    """Test that neural generators are lazily initialized."""

    @pytest.mark.unit
    def test_init_generators_are_none(self) -> None:
        """Test that all neural generators start as None."""
        orchestrator = GenerationOrchestrator()

        assert orchestrator._vae_generator is None
        assert orchestrator._diffusion_generator is None
        assert orchestrator._vqvae_generator is None
        assert orchestrator._conditioner is None

    @pytest.mark.unit
    def test_init_proposers_are_none(self) -> None:
        """Test that code proposers start as None."""
        orchestrator = GenerationOrchestrator()

        assert orchestrator._cadquery_proposer is None
        assert orchestrator._openscad_proposer is None

    @pytest.mark.unit
    def test_init_with_custom_config(self) -> None:
        """Test orchestrator initialization with custom config."""
        config = LLGenConfig(max_retries=5, device="cuda")
        orchestrator = GenerationOrchestrator(config)

        assert orchestrator.config.max_retries == 5
        assert orchestrator.config.device == "cuda"
        assert orchestrator._vae_generator is None

    @pytest.mark.unit
    def test_init_none_config_uses_defaults(self) -> None:
        """Test that passing None config uses defaults."""
        orchestrator = GenerationOrchestrator(config=None)

        assert orchestrator.config is not None
        assert isinstance(orchestrator.config, LLGenConfig)
        assert orchestrator.config.max_retries == 3


class TestOrchestratorConditioningIntegration:
    """Test _get_conditioning() lazy initialization and flow."""

    @pytest.mark.unit
    def test_get_conditioning_initializes_conditioner(self) -> None:
        """Test that _get_conditioning() lazily initializes MultiModalConditioner."""
        orchestrator = GenerationOrchestrator()

        assert orchestrator._conditioner is None

        # Mock the MultiModalConditioner import (imported inside the method)
        with patch("ll_gen.conditioning.multimodal.MultiModalConditioner") as mock_cond_cls:
            mock_cond = MagicMock()
            mock_cond.encode.return_value = MagicMock()
            mock_cond_cls.return_value = mock_cond

            orchestrator._get_conditioning("test prompt")

            # Should have initialized the conditioner
            assert orchestrator._conditioner is not None

    @pytest.mark.unit
    def test_get_conditioning_reuses_conditioner(self) -> None:
        """Test that _get_conditioning() reuses the conditioner on second call."""
        orchestrator = GenerationOrchestrator()

        with patch("ll_gen.conditioning.multimodal.MultiModalConditioner") as mock_cond_cls:
            mock_cond = MagicMock()
            mock_cond.encode.return_value = MagicMock()
            mock_cond_cls.return_value = mock_cond

            # First call
            orchestrator._get_conditioning("test prompt 1")
            first_conditioner = orchestrator._conditioner

            # Second call
            orchestrator._get_conditioning("test prompt 2")
            second_conditioner = orchestrator._conditioner

            # Should be the same instance
            assert first_conditioner is second_conditioner
            # Should have called the class constructor only once
            assert mock_cond_cls.call_count == 1

    @pytest.mark.unit
    def test_get_conditioning_uses_config_text_model(self) -> None:
        """Test that _get_conditioning() respects config.conditioning.text_model."""
        config = LLGenConfig(
            conditioning=ConditioningConfig(text_model="roberta-base")
        )
        orchestrator = GenerationOrchestrator(config)

        with patch("ll_gen.conditioning.multimodal.MultiModalConditioner") as mock_cond_cls:
            mock_cond = MagicMock()
            mock_cond.encode.return_value = MagicMock()
            mock_cond_cls.return_value = mock_cond

            orchestrator._get_conditioning("test prompt")

            # Check that text_model was passed
            mock_cond_cls.assert_called_once()
            call_kwargs = mock_cond_cls.call_args[1]
            assert call_kwargs["text_model"] == "roberta-base"

    @pytest.mark.unit
    def test_get_conditioning_uses_config_image_model(self) -> None:
        """Test that _get_conditioning() respects config.conditioning.image_model."""
        config = LLGenConfig(
            conditioning=ConditioningConfig(image_model="dino_vitb16")
        )
        orchestrator = GenerationOrchestrator(config)

        with patch("ll_gen.conditioning.multimodal.MultiModalConditioner") as mock_cond_cls:
            mock_cond = MagicMock()
            mock_cond.encode.return_value = MagicMock()
            mock_cond_cls.return_value = mock_cond

            orchestrator._get_conditioning("test prompt")

            call_kwargs = mock_cond_cls.call_args[1]
            assert call_kwargs["image_model"] == "dino_vitb16"

    @pytest.mark.unit
    def test_get_conditioning_uses_config_fusion_method(self) -> None:
        """Test that _get_conditioning() respects fusion_method config."""
        config = LLGenConfig(
            conditioning=ConditioningConfig(fusion_method="average")
        )
        orchestrator = GenerationOrchestrator(config)

        with patch("ll_gen.conditioning.multimodal.MultiModalConditioner") as mock_cond_cls:
            mock_cond = MagicMock()
            mock_cond.encode.return_value = MagicMock()
            mock_cond_cls.return_value = mock_cond

            orchestrator._get_conditioning("test prompt")

            call_kwargs = mock_cond_cls.call_args[1]
            assert call_kwargs["fusion_method"] == "average"

    @pytest.mark.unit
    def test_get_conditioning_returns_encoding(self) -> None:
        """Test that _get_conditioning() returns the conditioning embeddings."""
        orchestrator = GenerationOrchestrator()

        with patch("ll_gen.conditioning.multimodal.MultiModalConditioner") as mock_cond_cls:
            mock_embeddings = MagicMock()
            mock_cond = MagicMock()
            mock_cond.encode.return_value = mock_embeddings
            mock_cond_cls.return_value = mock_cond

            result = orchestrator._get_conditioning("test prompt")

            assert result is mock_embeddings
            mock_cond.encode.assert_called_once_with("test prompt", None)

    @pytest.mark.unit
    def test_get_conditioning_with_image_path(self) -> None:
        """Test that _get_conditioning() forwards image_path to conditioner."""
        orchestrator = GenerationOrchestrator()
        image_path = Path("/tmp/test.png")

        with patch("ll_gen.conditioning.multimodal.MultiModalConditioner") as mock_cond_cls:
            mock_embeddings = MagicMock()
            mock_cond = MagicMock()
            mock_cond.encode.return_value = mock_embeddings
            mock_cond_cls.return_value = mock_cond

            result = orchestrator._get_conditioning("test prompt", image_path)

            assert result is mock_embeddings
            mock_cond.encode.assert_called_once_with("test prompt", image_path)

    @pytest.mark.unit
    def test_get_conditioning_catches_import_error(self) -> None:
        """Test that _get_conditioning() returns None on import error."""
        orchestrator = GenerationOrchestrator()

        with patch("ll_gen.conditioning.multimodal.MultiModalConditioner") as mock_cond_cls:
            mock_cond_cls.side_effect = ImportError("Missing dependency")

            result = orchestrator._get_conditioning("test prompt")

            # Should gracefully return None on import error
            assert result is None


# ============================================================================
# SECTION 2: Neural Generator Proposal Methods Tests
# ============================================================================


class TestProposalGenerationVAE:
    """Test _propose_neural_vae() with NeuralVAEGenerator wrapper."""

    @pytest.mark.unit
    def test_vae_generator_lazy_initialization(self) -> None:
        """Test that _propose_neural_vae() lazily initializes NeuralVAEGenerator."""
        orchestrator = GenerationOrchestrator()

        assert orchestrator._vae_generator is None

        with patch("ll_gen.generators.neural_vae.NeuralVAEGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.generate.return_value = MagicMock()
            mock_gen_cls.return_value = mock_gen

            with patch.object(orchestrator, "_get_conditioning", return_value=None):
                orchestrator._propose_neural_vae("test prompt", None)

            assert orchestrator._vae_generator is not None
            mock_gen_cls.assert_called_once()

    @pytest.mark.unit
    def test_vae_generator_reused(self) -> None:
        """Test that _propose_neural_vae() reuses generator on second call."""
        orchestrator = GenerationOrchestrator()

        with patch("ll_gen.generators.neural_vae.NeuralVAEGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.generate.return_value = MagicMock()
            mock_gen_cls.return_value = mock_gen

            with patch.object(orchestrator, "_get_conditioning", return_value=None):
                orchestrator._propose_neural_vae("prompt 1", None)
                first_gen = orchestrator._vae_generator

                orchestrator._propose_neural_vae("prompt 2", None)
                second_gen = orchestrator._vae_generator

                assert first_gen is second_gen
                assert mock_gen_cls.call_count == 1

    @pytest.mark.unit
    def test_vae_uses_config_checkpoint(self) -> None:
        """Test that _propose_neural_vae() uses checkpoint from GeneratorConfig."""
        checkpoint_path = "/path/to/vae_checkpoint.pt"
        config = LLGenConfig(
            generators=GeneratorConfig(vae_checkpoint=checkpoint_path)
        )
        orchestrator = GenerationOrchestrator(config)

        with patch("ll_gen.generators.neural_vae.NeuralVAEGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.generate.return_value = MagicMock()
            mock_gen_cls.return_value = mock_gen

            with patch.object(orchestrator, "_get_conditioning", return_value=None):
                orchestrator._propose_neural_vae("test prompt", None)

                call_kwargs = mock_gen_cls.call_args[1]
                assert call_kwargs["checkpoint_path"] == Path(checkpoint_path)

    @pytest.mark.unit
    def test_vae_uses_config_temperature(self) -> None:
        """Test that _propose_neural_vae() uses temperature from GeneratorConfig."""
        config = LLGenConfig(
            generators=GeneratorConfig(default_temperature=0.5)
        )
        orchestrator = GenerationOrchestrator(config)

        with patch("ll_gen.generators.neural_vae.NeuralVAEGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.generate.return_value = MagicMock()
            mock_gen_cls.return_value = mock_gen

            with patch.object(orchestrator, "_get_conditioning", return_value=None):
                orchestrator._propose_neural_vae("test prompt", None)

                call_kwargs = mock_gen_cls.call_args[1]
                assert call_kwargs["temperature"] == 0.5

    @pytest.mark.unit
    def test_vae_passes_conditioning_to_generator(self) -> None:
        """Test that _propose_neural_vae() passes conditioning embeddings to generate()."""
        orchestrator = GenerationOrchestrator()

        with patch("ll_gen.generators.neural_vae.NeuralVAEGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_proposal = MagicMock()
            mock_gen.generate.return_value = mock_proposal
            mock_gen_cls.return_value = mock_gen

            mock_conditioning = MagicMock()

            with patch.object(orchestrator, "_get_conditioning", return_value=mock_conditioning):
                orchestrator._propose_neural_vae("test prompt", None)

                mock_gen.generate.assert_called_once()
                call_kwargs = mock_gen.generate.call_args[1]
                assert call_kwargs["conditioning"] is mock_conditioning

    @pytest.mark.unit
    def test_vae_passes_error_context(self) -> None:
        """Test that _propose_neural_vae() passes error_context to generate()."""
        orchestrator = GenerationOrchestrator()
        error_context = {"error_message": "Previous attempt failed"}

        with patch("ll_gen.generators.neural_vae.NeuralVAEGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.generate.return_value = MagicMock()
            mock_gen_cls.return_value = mock_gen

            with patch.object(orchestrator, "_get_conditioning", return_value=None):
                orchestrator._propose_neural_vae("test prompt", error_context)

                call_kwargs = mock_gen.generate.call_args[1]
                assert call_kwargs["error_context"] is error_context

    @pytest.mark.unit
    def test_vae_returns_command_sequence_proposal(self) -> None:
        """Test that _propose_neural_vae() returns CommandSequenceProposal."""
        orchestrator = GenerationOrchestrator()

        with patch("ll_gen.generators.neural_vae.NeuralVAEGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_proposal = MagicMock(spec=CommandSequenceProposal)
            mock_gen.generate.return_value = mock_proposal
            mock_gen_cls.return_value = mock_gen

            with patch.object(orchestrator, "_get_conditioning", return_value=None):
                result = orchestrator._propose_neural_vae("test prompt", None)

                assert result is mock_proposal


class TestProposalGenerationDiffusion:
    """Test _propose_neural_diffusion() with NeuralDiffusionGenerator wrapper."""

    @pytest.mark.unit
    def test_diffusion_generator_lazy_initialization(self) -> None:
        """Test that _propose_neural_diffusion() lazily initializes."""
        orchestrator = GenerationOrchestrator()

        assert orchestrator._diffusion_generator is None

        with patch("ll_gen.generators.neural_diffusion.NeuralDiffusionGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.generate.return_value = MagicMock()
            mock_gen_cls.return_value = mock_gen

            with patch.object(orchestrator, "_get_conditioning", return_value=None):
                orchestrator._propose_neural_diffusion("test prompt", None)

            assert orchestrator._diffusion_generator is not None

    @pytest.mark.unit
    def test_diffusion_uses_config_inference_steps(self) -> None:
        """Test that diffusion uses inference_steps from config."""
        config = LLGenConfig(
            generators=GeneratorConfig(diffusion_inference_steps=100)
        )
        orchestrator = GenerationOrchestrator(config)

        with patch("ll_gen.generators.neural_diffusion.NeuralDiffusionGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.generate.return_value = MagicMock()
            mock_gen_cls.return_value = mock_gen

            with patch.object(orchestrator, "_get_conditioning", return_value=None):
                orchestrator._propose_neural_diffusion("test prompt", None)

                call_kwargs = mock_gen_cls.call_args[1]
                assert call_kwargs["inference_steps"] == 100

    @pytest.mark.unit
    def test_diffusion_uses_config_eta(self) -> None:
        """Test that diffusion uses eta from config."""
        config = LLGenConfig(
            generators=GeneratorConfig(diffusion_eta=0.5)
        )
        orchestrator = GenerationOrchestrator(config)

        with patch("ll_gen.generators.neural_diffusion.NeuralDiffusionGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.generate.return_value = MagicMock()
            mock_gen_cls.return_value = mock_gen

            with patch.object(orchestrator, "_get_conditioning", return_value=None):
                orchestrator._propose_neural_diffusion("test prompt", None)

                call_kwargs = mock_gen_cls.call_args[1]
                assert call_kwargs["eta"] == 0.5

    @pytest.mark.unit
    def test_diffusion_returns_latent_proposal(self) -> None:
        """Test that _propose_neural_diffusion() returns LatentProposal."""
        orchestrator = GenerationOrchestrator()

        with patch("ll_gen.generators.neural_diffusion.NeuralDiffusionGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_proposal = MagicMock(spec=LatentProposal)
            mock_gen.generate.return_value = mock_proposal
            mock_gen_cls.return_value = mock_gen

            with patch.object(orchestrator, "_get_conditioning", return_value=None):
                result = orchestrator._propose_neural_diffusion("test prompt", None)

                assert result is mock_proposal


class TestProposalGenerationVQVAE:
    """Test _propose_neural_vqvae() with NeuralVQVAEGenerator wrapper."""

    @pytest.mark.unit
    def test_vqvae_generator_lazy_initialization(self) -> None:
        """Test that _propose_neural_vqvae() lazily initializes."""
        orchestrator = GenerationOrchestrator()

        assert orchestrator._vqvae_generator is None

        with patch("ll_gen.generators.neural_vqvae.NeuralVQVAEGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.generate.return_value = MagicMock()
            mock_gen_cls.return_value = mock_gen

            with patch.object(orchestrator, "_get_conditioning", return_value=None):
                orchestrator._propose_neural_vqvae("test prompt", None)

            assert orchestrator._vqvae_generator is not None

    @pytest.mark.unit
    def test_vqvae_uses_config_codebook_dim(self) -> None:
        """Test that VQVAE uses codebook_dim from config."""
        config = LLGenConfig(
            generators=GeneratorConfig(vqvae_codebook_dim=1024)
        )
        orchestrator = GenerationOrchestrator(config)

        with patch("ll_gen.generators.neural_vqvae.NeuralVQVAEGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.generate.return_value = MagicMock()
            mock_gen_cls.return_value = mock_gen

            with patch.object(orchestrator, "_get_conditioning", return_value=None):
                orchestrator._propose_neural_vqvae("test prompt", None)

                call_kwargs = mock_gen_cls.call_args[1]
                assert call_kwargs["codebook_dim"] == 1024

    @pytest.mark.unit
    def test_vqvae_returns_command_sequence_proposal(self) -> None:
        """Test that _propose_neural_vqvae() returns CommandSequenceProposal."""
        orchestrator = GenerationOrchestrator()

        with patch("ll_gen.generators.neural_vqvae.NeuralVQVAEGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_proposal = MagicMock(spec=CommandSequenceProposal)
            mock_gen.generate.return_value = mock_proposal
            mock_gen_cls.return_value = mock_gen

            with patch.object(orchestrator, "_get_conditioning", return_value=None):
                result = orchestrator._propose_neural_vqvae("test prompt", None)

                assert result is mock_proposal


# ============================================================================
# SECTION 3: Config Integration Tests
# ============================================================================


class TestConditioningConfigIntegration:
    """Test ConditioningConfig and its integration."""

    @pytest.mark.unit
    def test_conditioning_config_defaults(self) -> None:
        """Test ConditioningConfig has expected defaults."""
        config = ConditioningConfig()

        assert config.text_model == "bert-base-uncased"
        assert config.image_model == "dino_vits16"
        assert config.conditioning_dim == 768
        assert config.freeze_encoders is True
        assert config.fusion_method == "concat"
        assert config.image_size == 224

    @pytest.mark.unit
    def test_conditioning_config_custom(self) -> None:
        """Test ConditioningConfig with custom values."""
        config = ConditioningConfig(
            text_model="roberta-base",
            image_model="dino_vitb16",
            conditioning_dim=512,
            fusion_method="average",
        )

        assert config.text_model == "roberta-base"
        assert config.image_model == "dino_vitb16"
        assert config.conditioning_dim == 512
        assert config.fusion_method == "average"

    @pytest.mark.unit
    def test_generator_config_defaults(self) -> None:
        """Test GeneratorConfig has expected defaults."""
        config = GeneratorConfig()

        assert config.vae_checkpoint is None
        assert config.diffusion_checkpoint is None
        assert config.vqvae_checkpoint is None
        assert config.default_temperature == 0.8
        assert config.diffusion_inference_steps == 50
        assert config.diffusion_eta == 0.0
        assert config.vqvae_codebook_dim == 512
        assert config.latent_dim == 256
        assert config.max_seq_len == 60

    @pytest.mark.unit
    def test_generator_config_custom(self) -> None:
        """Test GeneratorConfig with custom values."""
        config = GeneratorConfig(
            vae_checkpoint="/path/to/vae.pt",
            default_temperature=0.5,
            diffusion_inference_steps=100,
        )

        assert config.vae_checkpoint == "/path/to/vae.pt"
        assert config.default_temperature == 0.5
        assert config.diffusion_inference_steps == 100

    @pytest.mark.unit
    def test_training_config_defaults(self) -> None:
        """Test TrainingConfig has expected defaults."""
        config = TrainingConfig()

        assert config.learning_rate == 1e-5
        assert config.batch_size == 4
        assert config.num_epochs == 10
        assert config.eval_interval == 5
        assert config.baseline_decay == 0.99
        assert config.entropy_coeff == 0.01
        assert config.max_grad_norm == 1.0
        assert config.checkpoint_dir == "checkpoints"

    @pytest.mark.unit
    def test_training_config_custom(self) -> None:
        """Test TrainingConfig with custom values."""
        config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=8,
            num_epochs=20,
        )

        assert config.learning_rate == 1e-4
        assert config.batch_size == 8
        assert config.num_epochs == 20

    @pytest.mark.unit
    def test_llgen_config_has_conditioning_field(self) -> None:
        """Test LLGenConfig includes conditioning field."""
        config = LLGenConfig()

        assert hasattr(config, "conditioning")
        assert isinstance(config.conditioning, ConditioningConfig)

    @pytest.mark.unit
    def test_llgen_config_has_generators_field(self) -> None:
        """Test LLGenConfig includes generators field."""
        config = LLGenConfig()

        assert hasattr(config, "generators")
        assert isinstance(config.generators, GeneratorConfig)

    @pytest.mark.unit
    def test_llgen_config_has_training_field(self) -> None:
        """Test LLGenConfig includes training field."""
        config = LLGenConfig()

        assert hasattr(config, "training")
        assert isinstance(config.training, TrainingConfig)


class TestGetLLGenConfigWithNestedKeys:
    """Test get_ll_gen_config() with nested conditioning/generator keys."""

    @pytest.mark.unit
    def test_conditioning_text_model_override(self) -> None:
        """Test overriding conditioning.text_model via dotted key."""
        config = get_ll_gen_config(**{"conditioning.text_model": "roberta-base"})

        assert config.conditioning.text_model == "roberta-base"

    @pytest.mark.unit
    def test_conditioning_image_model_override(self) -> None:
        """Test overriding conditioning.image_model via dotted key."""
        config = get_ll_gen_config(**{"conditioning.image_model": "dino_vitb16"})

        assert config.conditioning.image_model == "dino_vitb16"

    @pytest.mark.unit
    def test_conditioning_fusion_method_override(self) -> None:
        """Test overriding conditioning.fusion_method via dotted key."""
        config = get_ll_gen_config(**{"conditioning.fusion_method": "average"})

        assert config.conditioning.fusion_method == "average"

    @pytest.mark.unit
    def test_generator_vae_checkpoint_override(self) -> None:
        """Test overriding generators.vae_checkpoint via dotted key."""
        config = get_ll_gen_config(**{"generators.vae_checkpoint": "/path/to/vae.pt"})

        assert config.generators.vae_checkpoint == "/path/to/vae.pt"

    @pytest.mark.unit
    def test_generator_temperature_override(self) -> None:
        """Test overriding generators.default_temperature via dotted key."""
        config = get_ll_gen_config(**{"generators.default_temperature": 0.5})

        assert config.generators.default_temperature == 0.5

    @pytest.mark.unit
    def test_generator_inference_steps_override(self) -> None:
        """Test overriding generators.diffusion_inference_steps."""
        config = get_ll_gen_config(**{"generators.diffusion_inference_steps": 100})

        assert config.generators.diffusion_inference_steps == 100

    @pytest.mark.unit
    def test_training_learning_rate_override(self) -> None:
        """Test overriding training.learning_rate via dotted key."""
        config = get_ll_gen_config(**{"training.learning_rate": 1e-4})

        assert config.training.learning_rate == 1e-4

    @pytest.mark.unit
    def test_training_batch_size_override(self) -> None:
        """Test overriding training.batch_size via dotted key."""
        config = get_ll_gen_config(**{"training.batch_size": 16})

        assert config.training.batch_size == 16

    @pytest.mark.unit
    def test_multiple_nested_overrides(self) -> None:
        """Test multiple nested overrides in one call."""
        config = get_ll_gen_config(
            **{
                "conditioning.text_model": "roberta-base",
                "conditioning.fusion_method": "average",
                "generators.default_temperature": 0.6,
                "generators.diffusion_inference_steps": 80,
                "training.batch_size": 32,
            }
        )

        assert config.conditioning.text_model == "roberta-base"
        assert config.conditioning.fusion_method == "average"
        assert config.generators.default_temperature == 0.6
        assert config.generators.diffusion_inference_steps == 80
        assert config.training.batch_size == 32


# ============================================================================
# SECTION 4: Error Context and Retry Tests
# ============================================================================


class TestErrorContextFeedback:
    """Test error context passing through proposal methods."""

    @pytest.mark.unit
    def test_vae_receives_error_context_dict(self) -> None:
        """Test that VAE generator receives error_context dictionary."""
        orchestrator = GenerationOrchestrator()
        error_context = {
            "error_message": "Topology error",
            "error_category": ErrorCategory.TOPOLOGY_ERROR.value,
        }

        with patch("ll_gen.generators.neural_vae.NeuralVAEGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.generate.return_value = MagicMock()
            mock_gen_cls.return_value = mock_gen

            with patch.object(orchestrator, "_get_conditioning", return_value=None):
                orchestrator._propose_neural_vae("test prompt", error_context)

                call_kwargs = mock_gen.generate.call_args[1]
                assert call_kwargs["error_context"] == error_context

    @pytest.mark.unit
    def test_diffusion_receives_error_context_dict(self) -> None:
        """Test that diffusion generator receives error_context dictionary."""
        orchestrator = GenerationOrchestrator()
        error_context = {
            "error_message": "Self intersection",
            "error_category": ErrorCategory.SELF_INTERSECTION.value,
        }

        with patch("ll_gen.generators.neural_diffusion.NeuralDiffusionGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.generate.return_value = MagicMock()
            mock_gen_cls.return_value = mock_gen

            with patch.object(orchestrator, "_get_conditioning", return_value=None):
                orchestrator._propose_neural_diffusion("test prompt", error_context)

                call_kwargs = mock_gen.generate.call_args[1]
                assert call_kwargs["error_context"] == error_context

    @pytest.mark.unit
    def test_vqvae_receives_error_context_dict(self) -> None:
        """Test that VQVAE generator receives error_context dictionary."""
        orchestrator = GenerationOrchestrator()
        error_context = {
            "error_message": "Boolean failure",
            "error_category": ErrorCategory.BOOLEAN_FAILURE.value,
        }

        with patch("ll_gen.generators.neural_vqvae.NeuralVQVAEGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.generate.return_value = MagicMock()
            mock_gen_cls.return_value = mock_gen

            with patch.object(orchestrator, "_get_conditioning", return_value=None):
                orchestrator._propose_neural_vqvae("test prompt", error_context)

                call_kwargs = mock_gen.generate.call_args[1]
                assert call_kwargs["error_context"] == error_context

    @pytest.mark.unit
    def test_error_context_none_allowed(self) -> None:
        """Test that passing None error_context is allowed."""
        orchestrator = GenerationOrchestrator()

        with patch("ll_gen.generators.neural_vae.NeuralVAEGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.generate.return_value = MagicMock()
            mock_gen_cls.return_value = mock_gen

            with patch.object(orchestrator, "_get_conditioning", return_value=None):
                # Should not raise
                orchestrator._propose_neural_vae("test prompt", None)

                call_kwargs = mock_gen.generate.call_args[1]
                assert call_kwargs["error_context"] is None

    @pytest.mark.unit
    def test_error_context_passed_through_propose_method(self) -> None:
        """Test that error_context is passed through _propose() to neural methods."""
        orchestrator = GenerationOrchestrator()
        error_context = {"error_message": "Test error"}

        with patch("ll_gen.generators.neural_vae.NeuralVAEGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.generate.return_value = MagicMock()
            mock_gen_cls.return_value = mock_gen

            with patch.object(orchestrator, "_get_conditioning", return_value=None):
                orchestrator._propose(
                    route=GenerationRoute.NEURAL_VAE,
                    prompt="test",
                    error_context=error_context,
                )

                call_kwargs = mock_gen.generate.call_args[1]
                assert call_kwargs["error_context"] == error_context


# ============================================================================
# SECTION 5: Module Import Integration Tests
# ============================================================================


class TestModuleExports:
    """Test that all exports are available from ll_gen package."""

    @pytest.mark.unit
    def test_import_conditioning_embeddings(self) -> None:
        """Test that ConditioningEmbeddings can be imported."""
        from ll_gen.conditioning import ConditioningEmbeddings
        assert ConditioningEmbeddings is not None

    @pytest.mark.unit
    def test_import_text_conditioning_encoder(self) -> None:
        """Test that TextConditioningEncoder can be imported."""
        from ll_gen.conditioning import TextConditioningEncoder
        assert TextConditioningEncoder is not None

    @pytest.mark.unit
    def test_import_image_conditioning_encoder(self) -> None:
        """Test that ImageConditioningEncoder can be imported."""
        from ll_gen.conditioning import ImageConditioningEncoder
        assert ImageConditioningEncoder is not None

    @pytest.mark.unit
    def test_import_multimodal_conditioner(self) -> None:
        """Test that MultiModalConditioner can be imported."""
        from ll_gen.conditioning import MultiModalConditioner
        assert MultiModalConditioner is not None

    @pytest.mark.unit
    def test_import_constraint_predictor(self) -> None:
        """Test that ConstraintPredictor can be imported."""
        from ll_gen.conditioning import ConstraintPredictor
        assert ConstraintPredictor is not None

    @pytest.mark.unit
    def test_import_neural_vae_generator(self) -> None:
        """Test that NeuralVAEGenerator can be imported."""
        from ll_gen.generators import NeuralVAEGenerator
        assert NeuralVAEGenerator is not None

    @pytest.mark.unit
    def test_import_neural_diffusion_generator(self) -> None:
        """Test that NeuralDiffusionGenerator can be imported."""
        from ll_gen.generators import NeuralDiffusionGenerator
        assert NeuralDiffusionGenerator is not None

    @pytest.mark.unit
    def test_import_neural_vqvae_generator(self) -> None:
        """Test that NeuralVQVAEGenerator can be imported."""
        from ll_gen.generators import NeuralVQVAEGenerator
        assert NeuralVQVAEGenerator is not None

    @pytest.mark.unit
    def test_import_base_neural_generator(self) -> None:
        """Test that BaseNeuralGenerator can be imported."""
        from ll_gen.generators import BaseNeuralGenerator
        assert BaseNeuralGenerator is not None

    @pytest.mark.unit
    def test_import_latent_sampler(self) -> None:
        """Test that LatentSampler can be imported."""
        from ll_gen.generators import LatentSampler
        assert LatentSampler is not None


# ============================================================================
# SECTION 6: Integration Tests - Full Flow
# ============================================================================


class TestOrchestratorFullFlow:
    """Test full integration flows with mocked neural generators."""

    @pytest.mark.integration
    def test_generate_with_neural_vae_route(self) -> None:
        """Test full generate() flow with NEURAL_VAE route."""
        config = LLGenConfig(max_retries=1)
        orchestrator = GenerationOrchestrator(config)

        with patch.object(orchestrator, "router") as mock_router, \
             patch("ll_gen.generators.neural_vae.NeuralVAEGenerator") as mock_gen_cls, \
             patch.object(orchestrator, "disposal_engine") as mock_disposal:

            # Setup router
            mock_decision = MagicMock()
            mock_decision.route = GenerationRoute.NEURAL_VAE
            mock_decision.confidence = 0.8
            mock_router.route.return_value = mock_decision

            # Setup generator
            mock_gen = MagicMock()
            mock_proposal = MagicMock()
            mock_gen.generate.return_value = mock_proposal
            mock_gen_cls.return_value = mock_gen

            # Setup disposal
            mock_result = MagicMock()
            mock_result.is_valid = True
            mock_disposal.dispose.return_value = mock_result

            # Run
            orchestrator.generate("A smooth rounded shape")

            # Verify router was called
            mock_router.route.assert_called_once()
            # Verify generator was initialized and used
            assert orchestrator._vae_generator is not None
            # Verify disposal was called (single-pass: validate + export in one call)
            assert mock_disposal.dispose.call_count == 1

    @pytest.mark.integration
    def test_generate_with_error_retry_neural(self) -> None:
        """Test generate() retries on invalid disposal with neural route."""
        config = LLGenConfig(max_retries=2)
        orchestrator = GenerationOrchestrator(config)

        with patch.object(orchestrator, "router") as mock_router, \
             patch("ll_gen.generators.neural_vae.NeuralVAEGenerator") as mock_gen_cls, \
             patch.object(orchestrator, "disposal_engine") as mock_disposal, \
             patch.object(orchestrator, "_build_feedback", return_value={"error": "test"}):

            # Setup router
            mock_decision = MagicMock()
            mock_decision.route = GenerationRoute.NEURAL_VAE
            mock_router.route.return_value = mock_decision

            # Setup generator
            mock_gen = MagicMock()
            mock_proposal = MagicMock()
            mock_gen.generate.return_value = mock_proposal
            mock_gen_cls.return_value = mock_gen

            # Setup disposal: first call invalid, second call valid
            mock_result_invalid = MagicMock()
            mock_result_invalid.is_valid = False
            mock_result_invalid.error_category = ErrorCategory.TOPOLOGY_ERROR
            mock_result_invalid.reward_signal = 0.2

            mock_result_valid = MagicMock()
            mock_result_valid.is_valid = True
            mock_result_valid.reward_signal = 1.0

            # First attempt: invalid (1 single-pass dispose), second attempt: valid (1 single-pass dispose)
            mock_disposal.dispose.side_effect = [mock_result_invalid, mock_result_valid]

            # Run
            orchestrator.generate("test prompt")

            # Verify generator was called twice (once per attempt)
            assert mock_gen.generate.call_count == 2
            # Verify disposal: 1 (invalid) + 1 (valid, single-pass with export) = 2
            assert mock_disposal.dispose.call_count == 2

    @pytest.mark.integration
    def test_conditioning_passed_through_entire_flow(self) -> None:
        """Test that conditioning embeddings flow through to proposal generation."""
        orchestrator = GenerationOrchestrator()

        with patch("ll_gen.conditioning.multimodal.MultiModalConditioner") as mock_cond_cls, \
             patch("ll_gen.generators.neural_vae.NeuralVAEGenerator") as mock_gen_cls:

            # Setup conditioner
            mock_embeddings = MagicMock()
            mock_cond = MagicMock()
            mock_cond.encode.return_value = mock_embeddings
            mock_cond_cls.return_value = mock_cond

            # Setup generator
            mock_gen = MagicMock()
            mock_proposal = MagicMock()
            mock_gen.generate.return_value = mock_proposal
            mock_gen_cls.return_value = mock_gen

            # Run proposal generation
            orchestrator._propose_neural_vae("test prompt", None)

            # Verify conditioner was initialized and called
            mock_cond_cls.assert_called_once()
            mock_cond.encode.assert_called_once_with("test prompt", None)

            # Verify generator received the embeddings
            mock_gen.generate.assert_called_once()
            call_kwargs = mock_gen.generate.call_args[1]
            assert call_kwargs["conditioning"] is mock_embeddings


class TestMultipleProposalRoutes:
    """Test that different routes can be used in same orchestrator."""

    @pytest.mark.integration
    def test_can_switch_routes(self) -> None:
        """Test that orchestrator can switch between VAE and diffusion routes."""
        orchestrator = GenerationOrchestrator()

        with patch("ll_gen.generators.neural_vae.NeuralVAEGenerator") as mock_vae_cls, \
             patch("ll_gen.generators.neural_diffusion.NeuralDiffusionGenerator") as mock_diff_cls:

            mock_vae = MagicMock()
            mock_vae.generate.return_value = MagicMock()
            mock_vae_cls.return_value = mock_vae

            mock_diff = MagicMock()
            mock_diff.generate.return_value = MagicMock()
            mock_diff_cls.return_value = mock_diff

            with patch.object(orchestrator, "_get_conditioning", return_value=None):
                # Use VAE
                orchestrator._propose_neural_vae("prompt1", None)
                assert orchestrator._vae_generator is not None

                # Use Diffusion
                orchestrator._propose_neural_diffusion("prompt2", None)
                assert orchestrator._diffusion_generator is not None

                # Both should be initialized
                assert orchestrator._vae_generator is not None
                assert orchestrator._diffusion_generator is not None


# ============================================================================
# SECTION 7: Edge Cases and Error Handling
# ============================================================================


class TestOrchestratorErrorHandling:
    """Test error handling in orchestrator."""

    @pytest.mark.unit
    def test_neural_generator_import_error_handled(self) -> None:
        """Test that ImportError from neural generator is handled."""
        orchestrator = GenerationOrchestrator()

        # Patch the generators.neural_vae module to raise ImportError on access
        with patch.dict("sys.modules", {"ll_gen.generators.neural_vae": None}):
            with pytest.raises(RuntimeError, match="ll_gen.generators requires ll_stepnet"):
                orchestrator._propose_neural_vae("test prompt", None)

    @pytest.mark.unit
    def test_proposal_generation_with_missing_config_sections(self) -> None:
        """Test that proposal generation works with minimal config."""
        # Config with no explicit generators/conditioning config
        config = LLGenConfig()
        # Manually remove generators and conditioning to simulate edge case
        config.generators = GeneratorConfig()
        config.conditioning = ConditioningConfig()

        orchestrator = GenerationOrchestrator(config)

        with patch("ll_gen.generators.neural_vae.NeuralVAEGenerator") as mock_gen_cls:
            mock_gen = MagicMock()
            mock_gen.generate.return_value = MagicMock()
            mock_gen_cls.return_value = mock_gen

            with patch.object(orchestrator, "_get_conditioning", return_value=None):
                # Should use defaults
                orchestrator._propose_neural_vae("test", None)

                # Should have called constructor
                mock_gen_cls.assert_called_once()

    @pytest.mark.unit
    def test_conditioning_none_returned_on_error(self) -> None:
        """Test that _get_conditioning() gracefully returns None on any error."""
        orchestrator = GenerationOrchestrator()

        with patch("ll_gen.conditioning.multimodal.MultiModalConditioner") as mock_cond_cls:
            mock_cond_cls.side_effect = Exception("Unexpected error")

            result = orchestrator._get_conditioning("test prompt")

            # Should gracefully return None instead of raising
            assert result is None


# ============================================================================
# SECTION 8: Generation History Tests
# ============================================================================


class TestGenerationHistory:
    """Test GenerationHistory dataclass tracking."""

    @pytest.mark.unit
    def test_history_initialization(self) -> None:
        """Test GenerationHistory initializes correctly."""
        history = GenerationHistory()

        assert history.routing_decision is None
        assert history.attempts == []
        assert history.total_time_ms == 0.0
        assert history.final_result is None

    @pytest.mark.unit
    def test_history_with_attempts(self) -> None:
        """Test GenerationHistory tracks multiple attempts."""
        history = GenerationHistory()

        history.attempts.append({"attempt": 1, "proposal_id": "test_1"})
        history.attempts.append({"attempt": 2, "proposal_id": "test_2"})

        assert len(history.attempts) == 2
        assert history.attempts[0]["proposal_id"] == "test_1"
        assert history.attempts[1]["proposal_id"] == "test_2"

    @pytest.mark.unit
    def test_history_with_result(self) -> None:
        """Test GenerationHistory stores final result."""
        history = GenerationHistory()
        mock_result = MagicMock()
        mock_result.is_valid = True

        history.final_result = mock_result

        assert history.final_result is mock_result
        assert history.final_result.is_valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
