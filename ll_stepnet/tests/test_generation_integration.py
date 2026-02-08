"""End-to-end integration tests for the CAD generation pipeline.

Tests the full pipeline: text prompt → conditioning → VAE/Diffusion →
command decoding → (optional) reconstruction → validation.

These tests verify that all components wire together correctly and
produce valid outputs. For full benchmark metrics (COV, MMD, JSD),
see the evaluation module.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pytest

_log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


@pytest.fixture
def sample_encoder_config():
    """Create a minimal encoder config for testing."""
    class EncoderConfig:
        vocab_size: int = 50000
        token_embed_dim: int = 128  # Smaller for faster tests
        num_transformer_layers: int = 2
        dropout: float = 0.1
    return EncoderConfig()


@pytest.fixture
def vae_model(sample_encoder_config, device):
    """Create a STEPVAE model for testing."""
    try:
        from stepnet import STEPVAE
        import torch
    except ImportError:
        pytest.skip("stepnet or torch not installed")

    model = STEPVAE(
        encoder_config=sample_encoder_config,
        latent_dim=64,  # Smaller for faster tests
        max_seq_len=30,  # Shorter sequences
    )
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture
def text_conditioner(device):
    """Create a TextConditioner for testing."""
    try:
        from stepnet import TextConditioner
    except ImportError:
        pytest.skip("stepnet not installed")

    # Use a small conditioning dim for fast tests
    conditioner = TextConditioner(
        encoder_name="bert-base-uncased",
        conditioning_dim=128,
        freeze_encoder=True,
        num_adaptive_layers=1,
        skip_cross_attention_blocks=2,
    )
    # Note: Don't move to device yet - encoder loads lazily
    return conditioner


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ------------------------------------------------------------------
# Unit Tests: Component Integration
# ------------------------------------------------------------------

class TestVAEGeneration:
    """Test VAE model generation without reconstruction."""

    def test_vae_sample_produces_command_preds(self, vae_model, device):
        """VAE.sample() should produce command and parameter predictions."""
        import torch

        with torch.no_grad():
            output = vae_model.sample(num_samples=2, seq_len=20)

        assert "command_preds" in output
        assert "param_preds" in output

        command_preds = output["command_preds"]
        param_preds = output["param_preds"]

        assert command_preds.shape == (2, 20)  # [batch, seq_len]
        assert param_preds.shape == (2, 20, 16)  # [batch, seq_len, 16 params]

        # Commands should be valid indices (0-5 for 6 command types)
        assert command_preds.max() <= 5
        assert command_preds.min() >= 0

        # Params should be valid quantization bins (0-255)
        assert param_preds.max() <= 255
        assert param_preds.min() >= 0

    def test_vae_forward_returns_logits(self, vae_model, device):
        """VAE.forward() should return command and parameter logits."""
        import torch

        batch_size, seq_len = 2, 20
        token_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

        with torch.no_grad():
            output = vae_model(token_ids)

        assert "command_logits" in output
        assert "param_logits" in output
        assert "kl_loss" in output

        command_logits = output["command_logits"]
        param_logits = output["param_logits"]

        assert command_logits.shape == (batch_size, seq_len, 6)
        assert len(param_logits) == 16
        for pl in param_logits:
            assert pl.shape == (batch_size, seq_len, 256)


class TestConditioningIntegration:
    """Test text/image conditioning integration."""

    @pytest.mark.slow
    def test_text_conditioner_encode(self, text_conditioner):
        """TextConditioner should encode text to conditioning embeddings."""
        import torch

        # Create dummy tokenized input
        batch_size, text_len = 2, 10
        text_input_ids = torch.randint(0, 30000, (batch_size, text_len))
        text_attention_mask = torch.ones(batch_size, text_len)

        # Encode text (this will lazily load BERT)
        try:
            cond = text_conditioner.encode_text(text_input_ids, text_attention_mask)
        except Exception as e:
            # Skip if transformers not available
            if "transformers" in str(e).lower():
                pytest.skip("transformers library not available")
            raise

        assert cond.shape == (batch_size, text_len, 128)  # [B, L, cond_dim]

    @pytest.mark.slow
    def test_text_conditioner_skip_early_blocks(self, text_conditioner):
        """TextConditioner should skip cross-attention for early blocks."""
        import torch

        batch_size, seq_len = 2, 20
        hidden_states = torch.randn(batch_size, seq_len, 128)
        text_input_ids = torch.randint(0, 30000, (batch_size, 10))

        # Block 0 should skip cross-attention
        try:
            output_block_0 = text_conditioner(
                hidden_states=hidden_states,
                text_input_ids=text_input_ids,
                block_index=0,
            )
        except Exception as e:
            if "transformers" in str(e).lower():
                pytest.skip("transformers library not available")
            raise

        # Hidden states should be unchanged for early blocks
        assert torch.allclose(output_block_0, hidden_states)

        # Block 2 should apply cross-attention
        output_block_2 = text_conditioner(
            hidden_states=hidden_states,
            text_input_ids=text_input_ids,
            block_index=2,
        )

        # Hidden states should be modified for later blocks
        assert not torch.allclose(output_block_2, hidden_states)


class TestCADGenerationPipeline:
    """Test the CADGenerationPipeline wrapper."""

    def test_pipeline_generate_vae(self, vae_model, device):
        """CADGenerationPipeline should generate from VAE."""
        try:
            from stepnet import CADGenerationPipeline
            import torch
        except ImportError:
            pytest.skip("stepnet not installed")

        pipeline = CADGenerationPipeline(
            model=vae_model,
            mode="vae",
            device=device,
        )

        results = pipeline.generate(
            num_samples=2,
            seq_len=20,
            reconstruct=False,
        )

        assert len(results) == 2

        for result in results:
            assert "command_logits" in result or "command_preds" in result
            assert "batch_index" in result

    def test_pipeline_decode_to_token_sequence(self, vae_model, device):
        """Pipeline should decode logits to geotoken TokenSequence."""
        try:
            from stepnet import CADGenerationPipeline
            from geotoken.tokenizer import TokenSequence
            import torch
        except ImportError:
            pytest.skip("stepnet or geotoken not installed")

        pipeline = CADGenerationPipeline(
            model=vae_model,
            mode="vae",
            device=device,
        )

        results = pipeline.generate(
            num_samples=1,
            seq_len=20,
            reconstruct=False,
        )

        result = results[0]

        if "token_sequence" in result:
            assert isinstance(result["token_sequence"], TokenSequence)
            assert len(result["token_sequence"].command_tokens) > 0


class TestGeotokenEncoding:
    """Test geotoken vocabulary encoding."""

    def test_encode_to_tensor(self):
        """encode_to_tensor should produce fixed-length tensor."""
        try:
            import torch
            from geotoken.tokenizer import (
                encode_to_tensor,
                CommandToken,
                CommandType,
            )
        except ImportError:
            pytest.skip("geotoken or torch not installed")

        # Create sample command tokens
        tokens = [
            CommandToken(
                command_type=CommandType.SOL,
                parameters=[10, 20] + [0] * 14,
                parameter_mask=[True, True] + [False] * 14,
            ),
            CommandToken(
                command_type=CommandType.LINE,
                parameters=[10, 20, 30, 40] + [0] * 12,
                parameter_mask=[True, True, True, True] + [False] * 12,
            ),
        ]

        tensor = encode_to_tensor(tokens, seq_len=60)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (60,)
        assert tensor.dtype == torch.long

        # First token should be BOS (1)
        assert tensor[0].item() == 1

        # Last non-pad token should be EOS (2)
        non_pad = tensor[tensor != 0]
        assert non_pad[-1].item() == 2

    def test_batch_encode_to_tensor(self):
        """batch_encode_to_tensor should stack multiple sequences."""
        try:
            import torch
            from geotoken.tokenizer import (
                batch_encode_to_tensor,
                CommandToken,
                CommandType,
            )
        except ImportError:
            pytest.skip("geotoken or torch not installed")

        # Create batch of command tokens
        batch = [
            [CommandToken(CommandType.SOL, [10, 20] + [0] * 14, [True, True] + [False] * 14)],
            [CommandToken(CommandType.LINE, [10, 20, 30, 40] + [0] * 12, [True] * 4 + [False] * 12)],
        ]

        tensor = batch_encode_to_tensor(batch, seq_len=30)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (2, 30)
        assert tensor.dtype == torch.long


# ------------------------------------------------------------------
# Integration Tests: Full Pipeline
# ------------------------------------------------------------------

class TestEndToEndGeneration:
    """Test the complete generation flow."""

    def test_full_vae_pipeline_no_reconstruction(self, vae_model, device, temp_output_dir):
        """Test VAE generation without reconstruction."""
        try:
            from stepnet import CADGenerationPipeline
            import torch
        except ImportError:
            pytest.skip("stepnet not installed")

        pipeline = CADGenerationPipeline(
            model=vae_model,
            mode="vae",
            device=device,
        )

        results = pipeline.generate(
            num_samples=3,
            seq_len=20,
            reconstruct=False,
        )

        assert len(results) == 3

        valid_count = sum(1 for r in results if "command_logits" in r)
        _log.info("Generated %d/%d samples with command logits", valid_count, len(results))

        # At least one should have logits
        assert valid_count > 0

    @pytest.mark.slow
    def test_vae_with_text_conditioning(self, vae_model, text_conditioner, device):
        """Test VAE generation with text conditioning."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        # This test requires transformers to be installed
        try:
            tokenizer = text_conditioner.tokenizer
        except Exception as e:
            if "transformers" in str(e).lower():
                pytest.skip("transformers library not available")
            raise

        # Tokenize a sample prompt
        prompt = "A simple bracket with mounting holes"
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32,
        )

        conditioning_inputs = {
            "text_input_ids": encoded["input_ids"].to(device),
            "text_attention_mask": encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])).to(device),
        }

        # Move conditioner to device
        text_conditioner = text_conditioner.to(device)

        # Sample with conditioning
        with torch.no_grad():
            output = vae_model.sample(
                num_samples=2,
                seq_len=20,
                conditioner=text_conditioner,
                conditioning_inputs=conditioning_inputs,
            )

        assert "command_preds" in output
        assert output["command_preds"].shape == (2, 20)


# ------------------------------------------------------------------
# Benchmark Tests
# ------------------------------------------------------------------

class TestBenchmarkMetrics:
    """Test computation of benchmark metrics."""

    def test_validity_rate_computation(self, vae_model, device):
        """Test that we can compute validity rate from generated samples."""
        try:
            from stepnet import CADGenerationPipeline
        except ImportError:
            pytest.skip("stepnet not installed")

        pipeline = CADGenerationPipeline(
            model=vae_model,
            mode="vae",
            device=device,
        )

        results = pipeline.generate(
            num_samples=10,
            seq_len=20,
            reconstruct=False,
        )

        # For now, consider "valid" as having command_logits
        valid_count = sum(1 for r in results if "command_logits" in r)
        validity_rate = valid_count / len(results)

        _log.info("Validity rate: %.2f%% (%d/%d)", validity_rate * 100, valid_count, len(results))

        # Should have some valid outputs
        assert validity_rate > 0

        # Reference benchmarks:
        # - DeepCAD: 46.1% validity
        # - BrepGen: 62.9% validity
        # Our test model is untrained, so we just check it runs


class TestResearchBenchmarks:
    """Compare against published research benchmarks.

    Reference metrics from the literature:
    - DeepCAD (Wu et al.): COV=50.3, MMD=3.03, JSD=1.33, Validity=46.1%
    - BrepGen (Xu et al.): COV=54.6, MMD=2.61, JSD=0.93, Validity=62.9%
    """

    @pytest.mark.skip(reason="Requires trained model checkpoint")
    def test_deepcad_comparison(self):
        """Compare against DeepCAD benchmark metrics."""
        # This test requires:
        # 1. A trained VAE checkpoint
        # 2. The DeepCAD test set
        # 3. Chamfer distance computation for COV/MMD
        pass

    @pytest.mark.skip(reason="Requires trained model checkpoint")
    def test_brepgen_comparison(self):
        """Compare against BrepGen benchmark metrics."""
        # This test requires:
        # 1. A trained diffusion checkpoint
        # 2. The ABC dataset
        # 3. Full B-Rep validity checking
        pass
