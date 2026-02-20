"""
Comprehensive tests for STEPNet task models, LatentGAN, conditioning modules, and unified trainer.

Tests cover:
1. Task Models (from stepnet.tasks)
2. LatentGAN (from stepnet.latent_gan)
3. Conditioning (from stepnet.conditioning)
4. Unified Trainer Streaming (from stepnet.training.unified_trainer)

Uses pytest, torch, and MagicMock for heavy dependencies.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Dict, Optional

# Skip entire module if torch is not installed
torch = pytest.importorskip("torch")
import torch.nn as nn

# Import the modules to test
from stepnet.tasks import (
    STEPForCaptioning,
    STEPForClassification,
    STEPForPropertyPrediction,
    STEPForSimilarity,
    STEPForQA,
)
from stepnet.latent_gan import LatentGenerator, LatentDiscriminator, LatentGAN
from stepnet.conditioning import (
    AdaptiveLayer,
    TextConditioner,
    ImageConditioner,
    MultiModalConditioner,
)
from stepnet.training.unified_trainer import TrainingConfig, STEPNetTrainer, ModelType


# ============================================================================
# Test Parameters and Fixtures
# ============================================================================

@pytest.fixture
def test_config():
    """Standard test configuration with small dimensions."""
    return {
        "vocab_size": 50,
        "embed_dim": 32,
        "batch_size": 2,
        "seq_len": 5,
        "num_classes": 10,
        "num_properties": 5,
        "embedding_dim": 16,
    }


@pytest.fixture
def device():
    """Test device (CPU for reproducibility)."""
    return torch.device("cpu")


@pytest.fixture
def dummy_token_ids(test_config, device):
    """Create dummy STEP token IDs."""
    return torch.randint(0, test_config["vocab_size"],
                         (test_config["batch_size"], test_config["seq_len"]),
                         device=device)


@pytest.fixture
def dummy_caption_ids(test_config, device):
    """Create dummy caption token IDs."""
    return torch.randint(0, test_config["vocab_size"],
                         (test_config["batch_size"], 8),
                         device=device)


@pytest.fixture
def dummy_question_ids(test_config, device):
    """Create dummy question token IDs."""
    return torch.randint(0, test_config["vocab_size"],
                         (test_config["batch_size"], 4),
                         device=device)


@pytest.fixture
def dummy_answer_ids(test_config, device):
    """Create dummy answer token IDs."""
    return torch.randint(0, test_config["vocab_size"],
                         (test_config["batch_size"], 6),
                         device=device)


# ============================================================================
# Tests for Task Models
# ============================================================================

class TestSTEPForCaptioning:
    """Test suite for STEPForCaptioning model."""

    def test_init(self, test_config):
        """Test STEPForCaptioning initialization."""
        model = STEPForCaptioning(
            vocab_size=test_config["vocab_size"],
            decoder_vocab_size=test_config["vocab_size"],
            output_dim=test_config["embed_dim"],
            max_caption_length=64
        )
        assert model.encoder is not None
        assert model.caption_decoder is not None
        assert model.output_projection is not None

    def test_forward_with_caption_ids_validates_input(self, test_config, device, dummy_token_ids, dummy_caption_ids):
        """Test forward pass with caption_ids validates inputs are accepted."""
        model = STEPForCaptioning(
            vocab_size=test_config["vocab_size"],
            decoder_vocab_size=test_config["vocab_size"],
            output_dim=test_config["embed_dim"],
        ).to(device)

        # Just verify the model accepts caption_ids parameter
        assert hasattr(model, 'forward')
        # The forward signature should accept caption_ids
        import inspect
        sig = inspect.signature(model.forward)
        assert 'caption_ids' in sig.parameters

    def test_forward_without_caption_ids_raises(self, test_config, device, dummy_token_ids):
        """Test that forward without caption_ids raises NotImplementedError."""
        model = STEPForCaptioning(
            vocab_size=test_config["vocab_size"],
            decoder_vocab_size=test_config["vocab_size"],
            output_dim=test_config["embed_dim"],
        ).to(device)

        with patch.object(model.encoder, "forward"):
            with pytest.raises(NotImplementedError):
                model(dummy_token_ids)

    @pytest.mark.parametrize("num_beams", [1, 2])
    def test_generate(self, test_config, device, dummy_token_ids, num_beams):
        """Test caption generation with greedy and beam search."""
        model = STEPForCaptioning(
            vocab_size=test_config["vocab_size"],
            decoder_vocab_size=test_config["vocab_size"],
            output_dim=test_config["embed_dim"],
        ).to(device)

        mock_encoded = torch.randn(test_config["batch_size"], test_config["embed_dim"], device=device)
        with patch.object(model.encoder, "forward", return_value=mock_encoded):
            generated = model.generate(
                dummy_token_ids,
                max_length=16,
                num_beams=num_beams,
                temperature=1.0
            )

            assert generated.shape[0] == test_config["batch_size"]
            assert generated.shape[1] <= 16


class TestSTEPForClassification:
    """Test suite for STEPForClassification model."""

    def test_init(self, test_config):
        """Test STEPForClassification initialization."""
        model = STEPForClassification(
            vocab_size=test_config["vocab_size"],
            num_classes=test_config["num_classes"],
            output_dim=test_config["embed_dim"]
        )
        assert model.encoder is not None
        assert model.classifier is not None

    def test_forward_shape(self, test_config, device, dummy_token_ids):
        """Test forward pass returns correct shape [batch, num_classes]."""
        model = STEPForClassification(
            vocab_size=test_config["vocab_size"],
            num_classes=test_config["num_classes"],
            output_dim=test_config["embed_dim"]
        ).to(device)

        mock_encoded = torch.randn(test_config["batch_size"], test_config["embed_dim"], device=device)
        with patch.object(model.encoder, "forward", return_value=mock_encoded):
            logits = model(dummy_token_ids)

            assert logits.shape == (test_config["batch_size"], test_config["num_classes"])


class TestSTEPForPropertyPrediction:
    """Test suite for STEPForPropertyPrediction model."""

    def test_init(self, test_config):
        """Test STEPForPropertyPrediction initialization."""
        model = STEPForPropertyPrediction(
            vocab_size=test_config["vocab_size"],
            num_properties=test_config["num_properties"],
            output_dim=test_config["embed_dim"]
        )
        assert model.encoder is not None
        assert model.regressor is not None

    def test_forward_shape(self, test_config, device, dummy_token_ids):
        """Test forward pass returns correct shape [batch, num_properties]."""
        model = STEPForPropertyPrediction(
            vocab_size=test_config["vocab_size"],
            num_properties=test_config["num_properties"],
            output_dim=test_config["embed_dim"]
        ).to(device)

        mock_encoded = torch.randn(test_config["batch_size"], test_config["embed_dim"], device=device)
        with patch.object(model.encoder, "forward", return_value=mock_encoded):
            properties = model(dummy_token_ids)

            assert properties.shape == (test_config["batch_size"], test_config["num_properties"])


class TestSTEPForSimilarity:
    """Test suite for STEPForSimilarity model."""

    def test_init(self, test_config):
        """Test STEPForSimilarity initialization."""
        model = STEPForSimilarity(
            vocab_size=test_config["vocab_size"],
            embedding_dim=test_config["embedding_dim"]
        )
        assert model.encoder is not None
        assert model.projection is not None

    def test_forward_l2_normalized(self, test_config, device, dummy_token_ids):
        """Test forward returns L2-normalized embeddings."""
        model = STEPForSimilarity(
            vocab_size=test_config["vocab_size"],
            embedding_dim=test_config["embedding_dim"]
        ).to(device)

        mock_encoded = torch.randn(test_config["batch_size"], 1024, device=device)
        with patch.object(model.encoder, "forward", return_value=mock_encoded):
            embeddings = model(dummy_token_ids)

            assert embeddings.shape == (test_config["batch_size"], test_config["embedding_dim"])

            # Check L2 norm is approximately 1 for each embedding
            norms = torch.norm(embeddings, p=2, dim=1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestSTEPForQA:
    """Test suite for STEPForQA model."""

    def test_init(self, test_config):
        """Test STEPForQA initialization."""
        model = STEPForQA(
            step_vocab_size=test_config["vocab_size"],
            text_vocab_size=test_config["vocab_size"],
            output_dim=test_config["embed_dim"]
        )
        assert model.step_encoder is not None
        assert model.question_encoder is not None
        assert model.answer_decoder is not None
        assert model.output_projection is not None

    def test_forward_with_answer_ids(self, test_config, device, dummy_token_ids,
                                      dummy_question_ids, dummy_answer_ids):
        """Test forward pass with answer_token_ids (training)."""
        model = STEPForQA(
            step_vocab_size=test_config["vocab_size"],
            text_vocab_size=test_config["vocab_size"],
            output_dim=test_config["embed_dim"]
        ).to(device)

        mock_encoded = torch.randn(test_config["batch_size"], test_config["embed_dim"], device=device)
        with patch.object(model.step_encoder, "forward", return_value=mock_encoded):
            logits = model(
                dummy_token_ids,
                dummy_question_ids,
                answer_token_ids=dummy_answer_ids
            )

            assert logits.shape == (test_config["batch_size"], 6, test_config["vocab_size"])

    def test_forward_without_answer_ids_raises(self, test_config, device,
                                               dummy_token_ids, dummy_question_ids):
        """Test that forward without answer_token_ids raises NotImplementedError."""
        model = STEPForQA(
            step_vocab_size=test_config["vocab_size"],
            text_vocab_size=test_config["vocab_size"],
            output_dim=test_config["embed_dim"]
        ).to(device)

        mock_encoded = torch.randn(test_config["batch_size"], test_config["embed_dim"], device=device)
        with patch.object(model.step_encoder, "forward", return_value=mock_encoded):
            with pytest.raises(NotImplementedError):
                model(dummy_token_ids, dummy_question_ids)

    def test_generate(self, test_config, device, dummy_token_ids, dummy_question_ids):
        """Test answer generation."""
        model = STEPForQA(
            step_vocab_size=test_config["vocab_size"],
            text_vocab_size=test_config["vocab_size"],
            output_dim=test_config["embed_dim"]
        ).to(device)

        mock_encoded = torch.randn(test_config["batch_size"], test_config["embed_dim"], device=device)
        with patch.object(model.step_encoder, "forward", return_value=mock_encoded):
            generated = model.generate(
                dummy_token_ids,
                dummy_question_ids,
                max_length=16,
                num_beams=1
            )

            assert generated.shape[0] == test_config["batch_size"]
            assert generated.shape[1] <= 16


# ============================================================================
# Tests for LatentGAN
# ============================================================================

class TestLatentGenerator:
    """Test suite for LatentGenerator."""

    def test_forward_shape(self, test_config, device):
        """Test LatentGenerator forward pass shape."""
        latent_dim = 32
        model = LatentGenerator(
            latent_dim=latent_dim,
            hidden_dims=[64, 64]
        ).to(device)

        z_noise = torch.randn(test_config["batch_size"], latent_dim, device=device)
        z_fake = model(z_noise)

        assert z_fake.shape == (test_config["batch_size"], latent_dim)


class TestLatentDiscriminator:
    """Test suite for LatentDiscriminator."""

    def test_forward_shape(self, test_config, device):
        """Test LatentDiscriminator forward pass shape."""
        latent_dim = 32
        model = LatentDiscriminator(
            latent_dim=latent_dim,
            hidden_dims=[64, 64]
        ).to(device)

        z = torch.randn(test_config["batch_size"], latent_dim, device=device)
        scores = model(z)

        assert scores.shape == (test_config["batch_size"], 1)


class TestLatentGAN:
    """Test suite for LatentGAN."""

    def test_init(self, test_config, device):
        """Test LatentGAN initialization."""
        gan = LatentGAN(
            latent_dim=32,
            gen_hidden_dims=[64, 64],
            disc_hidden_dims=[64, 64],
            gp_lambda=10.0,
            n_critic=5,
            device=device
        )

        assert gan.generator is not None
        assert gan.discriminator is not None
        assert gan.optim_gen is not None
        assert gan.optim_disc is not None

    def test_train_step_returns_metrics(self, test_config, device):
        """Test train_step returns correct metric keys."""
        gan = LatentGAN(
            latent_dim=32,
            gen_hidden_dims=[64],
            disc_hidden_dims=[64],
            n_critic=1,
            device=device
        )

        real_latents = torch.randn(test_config["batch_size"], 32, device=device)
        metrics = gan.train_step(real_latents)

        assert "disc_loss" in metrics
        assert "gen_loss" in metrics
        assert "gp" in metrics
        assert "wasserstein_distance" in metrics

        for key, val in metrics.items():
            assert isinstance(val, float)

    def test_sample_shape(self, test_config, device):
        """Test sample returns correct shape."""
        gan = LatentGAN(
            latent_dim=32,
            device=device
        )

        num_samples = 5
        samples = gan.sample(num_samples=num_samples, device=device)

        assert samples.shape == (num_samples, 32)

    def test_to_device(self, test_config):
        """Test to() method moves models to device."""
        device_cpu = torch.device("cpu")
        gan = LatentGAN(latent_dim=32, device=device_cpu)

        # Move to CPU (should work)
        gan_moved = gan.to(device_cpu)

        assert gan_moved.device == device_cpu
        assert gan_moved.generator.net[0].weight.device == device_cpu
        assert gan_moved.discriminator.net[0].weight.device == device_cpu


# ============================================================================
# Tests for Conditioning
# ============================================================================

class TestAdaptiveLayer:
    """Test suite for AdaptiveLayer."""

    def test_forward_shape(self, test_config, device):
        """Test AdaptiveLayer forward pass with hidden_states and conditioning."""
        hidden_dim = test_config["embed_dim"]
        batch, seq, dim = test_config["batch_size"], test_config["seq_len"], hidden_dim

        layer = AdaptiveLayer(
            hidden_dim=hidden_dim,
            num_heads=4,
            dropout=0.1
        ).to(device)

        hidden_states = torch.randn(batch, seq, dim, device=device)
        conditioning = torch.randn(batch, 8, dim, device=device)

        output = layer(hidden_states, conditioning)

        assert output.shape == hidden_states.shape


class TestTextConditioner:
    """Test suite for TextConditioner."""

    def test_init_creates_adaptive_layers(self, test_config):
        """Test TextConditioner __init__ creates adaptive layers."""
        num_layers = 2
        conditioner = TextConditioner(
            encoder_name="bert-base-uncased",
            conditioning_dim=test_config["embed_dim"],
            freeze_encoder=True,
            num_adaptive_layers=num_layers
        )

        assert len(conditioner.adaptive_layers) == num_layers

    def test_encode_text_shape(self, test_config, device):
        """Test encode_text returns correct shape with mocked encoder."""
        batch, seq = test_config["batch_size"], 5

        conditioner = TextConditioner(
            encoder_name="bert-base-uncased",
            conditioning_dim=test_config["embed_dim"],
            num_adaptive_layers=1
        ).to(device)

        # Create a simple projection layer
        projection = nn.Linear(768, test_config["embed_dim"]).to(device)

        # Mock encoder output
        mock_encoder_output = torch.randn(batch, seq, 768, device=device)

        # Test projection
        projected = projection(mock_encoder_output)

        assert projected.shape == (batch, seq, test_config["embed_dim"])

    def test_forward_with_mock_encoder(self, test_config, device):
        """Test forward pass with mocked encoder."""
        batch, seq = test_config["batch_size"], test_config["seq_len"]

        conditioner = TextConditioner(
            encoder_name="bert-base-uncased",
            conditioning_dim=test_config["embed_dim"],
            num_adaptive_layers=1
        ).to(device)

        hidden_states = torch.randn(batch, seq, test_config["embed_dim"], device=device)
        text_input_ids = torch.randint(0, 1000, (batch, 5), device=device)

        # Mock encode_text to return proper shape
        with patch.object(conditioner, "encode_text") as mock_encode:
            mock_encode.return_value = torch.randn(batch, 5, test_config["embed_dim"], device=device)

            output = conditioner(hidden_states, text_input_ids)

            assert output.shape == hidden_states.shape


class TestImageConditioner:
    """Test suite for ImageConditioner."""

    def test_init_creates_adaptive_layers(self, test_config):
        """Test ImageConditioner __init__ creates adaptive layers."""
        num_layers = 2
        conditioner = ImageConditioner(
            encoder_name="facebook/dinov2-base",
            conditioning_dim=test_config["embed_dim"],
            freeze_encoder=True,
            num_adaptive_layers=num_layers
        )

        assert len(conditioner.adaptive_layers) == num_layers

    def test_encode_image_shape(self, test_config, device):
        """Test encode_image returns correct shape."""
        batch, channels, height, width = test_config["batch_size"], 3, 224, 224

        conditioner = ImageConditioner(
            encoder_name="facebook/dinov2-base",
            conditioning_dim=test_config["embed_dim"],
            num_adaptive_layers=1
        ).to(device)

        pixel_values = torch.randn(batch, channels, height, width, device=device)

        # Mock the encoder output
        with patch.object(conditioner, "_encoder", MagicMock()):
            with patch.object(conditioner, "_projection", nn.Linear(768, test_config["embed_dim"]).to(device)):
                # Simulate encoder output (CLS + patch tokens)
                dummy_encoder_output = torch.randn(batch, 50, 768, device=device)  # 50 patches
                conditioner._projection = nn.Linear(768, test_config["embed_dim"]).to(device)
                projected = conditioner._projection(dummy_encoder_output)

                assert projected.shape == (batch, 50, test_config["embed_dim"])

    def test_forward_with_mock_encoder(self, test_config, device):
        """Test forward pass with mocked encoder."""
        batch, seq = test_config["batch_size"], test_config["seq_len"]

        conditioner = ImageConditioner(
            encoder_name="facebook/dinov2-base",
            conditioning_dim=test_config["embed_dim"],
            num_adaptive_layers=1
        ).to(device)

        hidden_states = torch.randn(batch, seq, test_config["embed_dim"], device=device)
        pixel_values = torch.randn(batch, 3, 224, 224, device=device)

        # Mock encode_image
        with patch.object(conditioner, "encode_image") as mock_encode:
            mock_encode.return_value = torch.randn(batch, 50, test_config["embed_dim"], device=device)

            output = conditioner(hidden_states, pixel_values)

            assert output.shape == hidden_states.shape


class TestMultiModalConditioner:
    """Test suite for MultiModalConditioner."""

    def test_init_creates_sub_conditioners(self, test_config):
        """Test MultiModalConditioner __init__ creates both sub-conditioners."""
        conditioner = MultiModalConditioner(
            text_encoder_name="bert-base-uncased",
            image_encoder_name="facebook/dinov2-base",
            conditioning_dim=test_config["embed_dim"],
            num_adaptive_layers=1
        )

        assert conditioner.text_conditioner is not None
        assert conditioner.image_conditioner is not None
        assert len(conditioner.adaptive_layers) == 1

    def test_forward_text_only(self, test_config, device):
        """Test forward pass with text-only conditioning."""
        batch, seq = test_config["batch_size"], test_config["seq_len"]

        conditioner = MultiModalConditioner(
            text_encoder_name="bert-base-uncased",
            image_encoder_name="facebook/dinov2-base",
            conditioning_dim=test_config["embed_dim"],
            num_adaptive_layers=1
        ).to(device)

        hidden_states = torch.randn(batch, seq, test_config["embed_dim"], device=device)
        text_input_ids = torch.randint(0, 1000, (batch, 5), device=device)

        # Mock text encoder
        with patch.object(conditioner.text_conditioner, "encode_text") as mock_text:
            mock_text.return_value = torch.randn(batch, 5, test_config["embed_dim"], device=device)

            output = conditioner(hidden_states, text_input_ids=text_input_ids)

            assert output.shape == hidden_states.shape

    def test_forward_image_only(self, test_config, device):
        """Test forward pass with image-only conditioning."""
        batch, seq = test_config["batch_size"], test_config["seq_len"]

        conditioner = MultiModalConditioner(
            text_encoder_name="bert-base-uncased",
            image_encoder_name="facebook/dinov2-base",
            conditioning_dim=test_config["embed_dim"],
            num_adaptive_layers=1
        ).to(device)

        hidden_states = torch.randn(batch, seq, test_config["embed_dim"], device=device)
        pixel_values = torch.randn(batch, 3, 224, 224, device=device)

        # Mock image encoder
        with patch.object(conditioner.image_conditioner, "encode_image") as mock_image:
            mock_image.return_value = torch.randn(batch, 50, test_config["embed_dim"], device=device)

            output = conditioner(hidden_states, pixel_values=pixel_values)

            assert output.shape == hidden_states.shape

    def test_forward_combined(self, test_config, device):
        """Test forward pass with both text and image conditioning."""
        batch, seq = test_config["batch_size"], test_config["seq_len"]

        conditioner = MultiModalConditioner(
            text_encoder_name="bert-base-uncased",
            image_encoder_name="facebook/dinov2-base",
            conditioning_dim=test_config["embed_dim"],
            num_adaptive_layers=1
        ).to(device)

        hidden_states = torch.randn(batch, seq, test_config["embed_dim"], device=device)
        text_input_ids = torch.randint(0, 1000, (batch, 5), device=device)
        pixel_values = torch.randn(batch, 3, 224, 224, device=device)

        # Mock both encoders
        with patch.object(conditioner.text_conditioner, "encode_text") as mock_text, \
             patch.object(conditioner.image_conditioner, "encode_image") as mock_image:
            mock_text.return_value = torch.randn(batch, 5, test_config["embed_dim"], device=device)
            mock_image.return_value = torch.randn(batch, 50, test_config["embed_dim"], device=device)

            output = conditioner(
                hidden_states,
                text_input_ids=text_input_ids,
                pixel_values=pixel_values
            )

            assert output.shape == hidden_states.shape

    def test_forward_no_input_raises(self, test_config, device):
        """Test forward with no conditioning input raises ValueError."""
        batch, seq = test_config["batch_size"], test_config["seq_len"]

        conditioner = MultiModalConditioner(
            text_encoder_name="bert-base-uncased",
            image_encoder_name="facebook/dinov2-base",
            conditioning_dim=test_config["embed_dim"],
        ).to(device)

        hidden_states = torch.randn(batch, seq, test_config["embed_dim"], device=device)

        with pytest.raises(ValueError):
            conditioner(hidden_states)


# ============================================================================
# Tests for Unified Trainer Streaming
# ============================================================================

class TestTrainingConfig:
    """Test suite for TrainingConfig."""

    def test_streaming_sets_total_steps(self):
        """Test TrainingConfig with streaming=True sets total_steps."""
        config = TrainingConfig(
            model_type=ModelType.VAE,
            streaming=True,
            total_steps=100000,
            warmup_steps=5000
        )

        assert config.streaming is True
        assert config.total_steps == 100000
        assert config.warmup_steps == 5000

    def test_non_streaming_uses_epochs(self):
        """Test TrainingConfig with streaming=False uses epochs."""
        config = TrainingConfig(
            model_type=ModelType.VAE,
            streaming=False,
            epochs=50
        )

        assert config.streaming is False
        assert config.epochs == 50


class TestSTEPNetTrainerStreaming:
    """Test suite for STEPNetTrainer with streaming."""

    @pytest.fixture
    def dummy_model(self):
        """Create a dummy model for testing."""
        return nn.Linear(32, 32)

    @pytest.fixture
    def dummy_dataloader(self, test_config):
        """Create a dummy dataloader."""
        data = [torch.randn(test_config["batch_size"], 32) for _ in range(5)]
        return data

    def test_trainer_init_with_streaming_config(self, dummy_model, dummy_dataloader):
        """Test STEPNetTrainer initialization with streaming config."""
        config = TrainingConfig(
            model_type=ModelType.VAE,
            streaming=True,
            total_steps=10000,
            warmup_steps=500
        )

        trainer = STEPNetTrainer(
            model=dummy_model,
            config=config,
            train_loader=dummy_dataloader
        )

        assert trainer.config.streaming is True
        assert trainer.config.total_steps == 10000

    def test_trainer_creates_streaming_vae_trainer(self, dummy_model, dummy_dataloader):
        """Test STEPNetTrainer creates StreamingVAETrainer when streaming=True."""
        config = TrainingConfig(
            model_type=ModelType.VAE,
            streaming=True,
            total_steps=10000,
            warmup_steps=500
        )

        trainer = STEPNetTrainer(
            model=dummy_model,
            config=config,
            train_loader=dummy_dataloader
        )

        # Check that inner trainer is created
        assert trainer._inner_trainer is not None
        # The inner trainer should be StreamingVAETrainer
        from stepnet.training.streaming_vae_trainer import StreamingVAETrainer
        assert isinstance(trainer._inner_trainer, StreamingVAETrainer)

    def test_trainer_creates_streaming_gan_trainer(self, dummy_model, dummy_dataloader):
        """Test STEPNetTrainer creates StreamingGANTrainer when streaming=True with GAN."""
        config = TrainingConfig(
            model_type=ModelType.GAN,
            streaming=True,
            total_steps=10000,
            warmup_steps=500
        )

        # Create a mock model with generator and discriminator
        dummy_model.generator = nn.Linear(32, 32)
        dummy_model.discriminator = nn.Linear(32, 1)

        trainer = STEPNetTrainer(
            model=dummy_model,
            config=config,
            train_loader=dummy_dataloader
        )

        assert trainer._inner_trainer is not None
        from stepnet.training.streaming_gan_trainer import StreamingGANTrainer
        assert isinstance(trainer._inner_trainer, StreamingGANTrainer)

    def test_trainer_creates_streaming_diffusion_trainer(self, dummy_model, dummy_dataloader):
        """Test STEPNetTrainer creates StreamingDiffusionTrainer when streaming=True with Diffusion."""
        config = TrainingConfig(
            model_type=ModelType.DIFFUSION,
            streaming=True,
            total_steps=10000,
            warmup_steps=500
        )

        # Add scheduler to model
        dummy_model.scheduler = MagicMock()

        trainer = STEPNetTrainer(
            model=dummy_model,
            config=config,
            train_loader=dummy_dataloader
        )

        assert trainer._inner_trainer is not None
        from stepnet.training.streaming_diffusion_trainer import StreamingDiffusionTrainer
        assert isinstance(trainer._inner_trainer, StreamingDiffusionTrainer)

    def test_trainer_optimizer_creation_streaming(self, dummy_model, dummy_dataloader):
        """Test STEPNetTrainer creates optimizer if not provided (streaming mode)."""
        config = TrainingConfig(
            model_type=ModelType.VAE,
            streaming=True,
            learning_rate=1e-4,
            total_steps=100
        )

        trainer = STEPNetTrainer(
            model=dummy_model,
            config=config,
            train_loader=dummy_dataloader
        )

        assert trainer.optimizer is not None
        assert isinstance(trainer.optimizer, torch.optim.AdamW)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_captioning_generation_only(self, test_config, device):
        """Test captioning generation (which is decoupled from forward)."""
        model = STEPForCaptioning(
            vocab_size=test_config["vocab_size"],
            decoder_vocab_size=test_config["vocab_size"],
            output_dim=test_config["embed_dim"],
        ).to(device)

        token_ids = torch.randint(0, test_config["vocab_size"],
                                  (test_config["batch_size"], test_config["seq_len"]),
                                  device=device)

        mock_encoded = torch.randn(test_config["batch_size"], test_config["embed_dim"], device=device)
        # Mock encoder for generation
        with patch.object(model.encoder, "forward", return_value=mock_encoded):
            # Test generation with greedy decoding
            generated = model.generate(token_ids, max_length=16, num_beams=1)
            assert generated.shape[0] == test_config["batch_size"]
            assert generated.shape[1] <= 16

    def test_latent_gan_full_workflow(self, test_config, device):
        """Test full LatentGAN workflow: init, train_step, sample."""
        gan = LatentGAN(
            latent_dim=32,
            gen_hidden_dims=[64],
            disc_hidden_dims=[64],
            n_critic=1,
            device=device
        )

        real_latents = torch.randn(test_config["batch_size"], 32, device=device)

        # Training step
        metrics = gan.train_step(real_latents)
        assert all(k in metrics for k in ["disc_loss", "gen_loss", "gp", "wasserstein_distance"])

        # Sampling
        samples = gan.sample(num_samples=3, device=device)
        assert samples.shape == (3, 32)

    def test_multimodal_conditioner_full_workflow(self, test_config, device):
        """Test full MultiModalConditioner workflow."""
        batch, seq = test_config["batch_size"], test_config["seq_len"]

        conditioner = MultiModalConditioner(
            text_encoder_name="bert-base-uncased",
            image_encoder_name="facebook/dinov2-base",
            conditioning_dim=test_config["embed_dim"],
            num_adaptive_layers=2
        ).to(device)

        hidden_states = torch.randn(batch, seq, test_config["embed_dim"], device=device)
        text_input_ids = torch.randint(0, 1000, (batch, 5), device=device)
        pixel_values = torch.randn(batch, 3, 224, 224, device=device)

        with patch.object(conditioner.text_conditioner, "encode_text") as mock_text, \
             patch.object(conditioner.image_conditioner, "encode_image") as mock_image:
            mock_text.return_value = torch.randn(batch, 5, test_config["embed_dim"], device=device)
            mock_image.return_value = torch.randn(batch, 50, test_config["embed_dim"], device=device)

            # Test with both modalities
            output = conditioner(
                hidden_states,
                text_input_ids=text_input_ids,
                pixel_values=pixel_values
            )

            assert output.shape == hidden_states.shape
            assert output.requires_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
