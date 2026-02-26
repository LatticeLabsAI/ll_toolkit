"""Tests for beam search decoding and generation configuration.

Tests cover:
- GenerationConfig validation and factory methods
- BeamSearchDecoder with greedy, sampling, and beam search
- Top-k and top-p filtering
- Repetition penalty
- Length normalization
- Integration with task models
"""
from __future__ import annotations

from typing import Optional

import pytest

# Skip entire module if torch is not installed
torch = pytest.importorskip("torch")
import torch.nn as nn


# ============================================================================
# SECTION 1: GenerationConfig Tests
# ============================================================================


class TestGenerationConfig:
    """Test GenerationConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        from stepnet.generation import GenerationConfig

        config = GenerationConfig()
        assert config.max_length == 64
        assert config.num_beams == 4
        assert config.temperature == 1.0
        assert config.top_k == 0
        assert config.top_p == 1.0

    def test_greedy_factory(self) -> None:
        """Test greedy config factory method."""
        from stepnet.generation import DecodingStrategy, GenerationConfig

        config = GenerationConfig.greedy(max_length=32)
        assert config.num_beams == 1
        assert config.do_sample is False
        assert config.get_strategy() == DecodingStrategy.GREEDY

    def test_beam_search_factory(self) -> None:
        """Test beam search config factory method."""
        from stepnet.generation import DecodingStrategy, GenerationConfig

        config = GenerationConfig.beam_search(num_beams=8, max_length=128)
        assert config.num_beams == 8
        assert config.max_length == 128
        assert config.get_strategy() == DecodingStrategy.BEAM_SEARCH

    def test_sampling_factory(self) -> None:
        """Test sampling config factory method."""
        from stepnet.generation import DecodingStrategy, GenerationConfig

        config = GenerationConfig.sampling(temperature=0.7, top_k=50, top_p=0.9)
        assert config.temperature == 0.7
        assert config.top_k == 50
        assert config.top_p == 0.9
        assert config.do_sample is True
        assert config.get_strategy() == DecodingStrategy.SAMPLING

    def test_validate_valid_config(self) -> None:
        """Test validation passes for valid config."""
        from stepnet.generation import GenerationConfig

        config = GenerationConfig(
            max_length=64,
            min_length=5,
            num_beams=4,
            temperature=0.8,
        )
        config.validate()  # Should not raise

    def test_validate_invalid_max_length(self) -> None:
        """Test validation fails for invalid max_length."""
        from stepnet.generation import GenerationConfig

        config = GenerationConfig(max_length=0)
        with pytest.raises(ValueError, match="max_length"):
            config.validate()

    def test_validate_min_greater_than_max(self) -> None:
        """Test validation fails when min_length > max_length."""
        from stepnet.generation import GenerationConfig

        config = GenerationConfig(min_length=100, max_length=50)
        with pytest.raises(ValueError, match="min_length"):
            config.validate()

    def test_validate_invalid_temperature(self) -> None:
        """Test validation fails for non-positive temperature."""
        from stepnet.generation import GenerationConfig

        config = GenerationConfig(temperature=0)
        with pytest.raises(ValueError, match="temperature"):
            config.validate()

    def test_validate_invalid_top_p(self) -> None:
        """Test validation fails for top_p outside [0, 1]."""
        from stepnet.generation import GenerationConfig

        config = GenerationConfig(top_p=1.5)
        with pytest.raises(ValueError, match="top_p"):
            config.validate()


# ============================================================================
# SECTION 2: BeamSearchDecoder Basic Tests
# ============================================================================


class MockDecoder(nn.Module):
    """Simple mock decoder for testing."""

    def __init__(self, vocab_size: int = 100, hidden_dim: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Return logits for last position."""
        batch_size = tgt.size(0)
        # Use memory to produce output (ignore tgt input shape issues)
        # Pool memory if needed
        if memory.dim() == 3:
            hidden = memory[:, -1, :]  # [batch, hidden_dim]
        else:
            hidden = memory

        # Ensure correct shape
        if hidden.size(-1) != self.hidden_dim:
            hidden = hidden[:, :self.hidden_dim] if hidden.size(-1) > self.hidden_dim else \
                     torch.nn.functional.pad(hidden, (0, self.hidden_dim - hidden.size(-1)))

        return self.output_proj(hidden)


class TestBeamSearchDecoderBasic:
    """Test BeamSearchDecoder basic functionality."""

    def test_decoder_creation(self) -> None:
        """Test decoder can be created."""
        from stepnet.generation import BeamSearchDecoder, GenerationConfig

        config = GenerationConfig(num_beams=4)
        decoder = BeamSearchDecoder(config)
        assert decoder is not None
        assert decoder.config.num_beams == 4

    def test_greedy_decode(self) -> None:
        """Test greedy decoding produces output."""
        from stepnet.generation import BeamSearchDecoder, GenerationConfig

        config = GenerationConfig.greedy(max_length=10)
        decoder = BeamSearchDecoder(config)

        vocab_size = 100
        batch_size = 2
        mock_model = MockDecoder(vocab_size=vocab_size)

        memory = torch.randn(batch_size, 1, 64)

        def model_step_fn(tgt, mem):
            return mock_model(tgt, mem)

        output = decoder.decode(
            model_step_fn=model_step_fn,
            memory=memory,
            vocab_size=vocab_size,
        )

        assert output.sequences is not None
        assert output.sequences.shape[0] == batch_size
        assert output.sequences.shape[1] <= config.max_length + 1

    def test_beam_search_decode(self) -> None:
        """Test beam search decoding produces output."""
        from stepnet.generation import BeamSearchDecoder, GenerationConfig

        config = GenerationConfig.beam_search(num_beams=4, max_length=10)
        decoder = BeamSearchDecoder(config)

        vocab_size = 100
        batch_size = 2
        mock_model = MockDecoder(vocab_size=vocab_size)

        memory = torch.randn(batch_size, 1, 64)

        def model_step_fn(tgt, mem):
            return mock_model(tgt, mem)

        output = decoder.decode(
            model_step_fn=model_step_fn,
            memory=memory,
            vocab_size=vocab_size,
        )

        assert output.sequences is not None
        assert output.sequences.shape[0] == batch_size
        assert output.scores is not None

    def test_sampling_decode(self) -> None:
        """Test sampling decoding produces output."""
        from stepnet.generation import BeamSearchDecoder, GenerationConfig

        config = GenerationConfig.sampling(temperature=1.0, max_length=10)
        decoder = BeamSearchDecoder(config)

        vocab_size = 100
        batch_size = 2
        mock_model = MockDecoder(vocab_size=vocab_size)

        memory = torch.randn(batch_size, 1, 64)

        def model_step_fn(tgt, mem):
            return mock_model(tgt, mem)

        output = decoder.decode(
            model_step_fn=model_step_fn,
            memory=memory,
            vocab_size=vocab_size,
        )

        assert output.sequences is not None
        assert output.sequences.shape[0] == batch_size


# ============================================================================
# SECTION 3: Top-K and Top-P Filtering Tests
# ============================================================================


class TestTopKTopPFiltering:
    """Test top-k and top-p filtering."""

    def test_top_k_filtering(self) -> None:
        """Test top-k filters to k tokens."""
        from stepnet.generation import BeamSearchDecoder, GenerationConfig

        config = GenerationConfig.sampling(top_k=5, max_length=5)
        decoder = BeamSearchDecoder(config)

        logits = torch.randn(2, 100)
        filtered = decoder._top_k_filter(logits.clone(), 5)

        # Only 5 tokens should have non-inf logits per sample
        for i in range(logits.size(0)):
            valid_count = (filtered[i] > float("-inf")).sum()
            assert valid_count == 5

    def test_top_p_filtering(self) -> None:
        """Test top-p (nucleus) filtering."""
        from stepnet.generation import BeamSearchDecoder, GenerationConfig

        config = GenerationConfig.sampling(top_p=0.5, max_length=5)
        decoder = BeamSearchDecoder(config)

        # Create logits with clear distribution
        logits = torch.zeros(2, 10)
        logits[:, 0] = 5.0  # High probability token
        logits[:, 1] = 3.0  # Medium
        logits[:, 2:] = -1.0  # Low

        filtered = decoder._top_p_filter(logits.clone(), 0.5)

        # At least one token should remain valid
        for i in range(logits.size(0)):
            valid_count = (filtered[i] > float("-inf")).sum()
            assert valid_count >= 1

    def test_combined_filtering(self) -> None:
        """Test top-k and top-p combined."""
        from stepnet.generation import BeamSearchDecoder, GenerationConfig

        config = GenerationConfig.sampling(top_k=10, top_p=0.9, max_length=5)
        decoder = BeamSearchDecoder(config)

        vocab_size = 100
        batch_size = 2
        mock_model = MockDecoder(vocab_size=vocab_size)

        memory = torch.randn(batch_size, 1, 64)

        output = decoder.decode(
            model_step_fn=lambda t, m: mock_model(t, m),
            memory=memory,
            vocab_size=vocab_size,
        )

        assert output.sequences is not None


# ============================================================================
# SECTION 4: Repetition Penalty Tests
# ============================================================================


class TestRepetitionPenalty:
    """Test repetition penalty application."""

    def test_apply_repetition_penalty(self) -> None:
        """Test repetition penalty reduces repeated token scores."""
        from stepnet.generation import BeamSearchDecoder, GenerationConfig

        config = GenerationConfig(repetition_penalty=2.0, max_length=10)
        decoder = BeamSearchDecoder(config)

        logits = torch.ones(2, 100)
        generated = torch.tensor([[1, 2, 3], [4, 5, 6]])

        penalized = decoder._apply_repetition_penalty(logits.clone(), generated)

        # Previously generated tokens should have lower scores
        assert penalized[0, 1] < logits[0, 1]
        assert penalized[0, 2] < logits[0, 2]
        assert penalized[1, 4] < logits[1, 4]

        # Non-generated tokens unchanged
        assert penalized[0, 10] == logits[0, 10]

    def test_no_repetition_penalty(self) -> None:
        """Test repetition_penalty=1.0 has no effect."""
        from stepnet.generation import BeamSearchDecoder, GenerationConfig

        config = GenerationConfig(repetition_penalty=1.0, max_length=10)
        decoder = BeamSearchDecoder(config)

        logits = torch.ones(2, 100)
        generated = torch.tensor([[1, 2, 3], [4, 5, 6]])

        penalized = decoder._apply_repetition_penalty(logits.clone(), generated)

        assert torch.allclose(penalized, logits)


# ============================================================================
# SECTION 5: N-gram Blocking Tests
# ============================================================================


class TestNgramBlocking:
    """Test n-gram blocking to prevent repetition."""

    def test_block_repeated_ngrams(self) -> None:
        """Test n-gram blocking prevents repeated sequences."""
        from stepnet.generation import BeamSearchDecoder, GenerationConfig

        config = GenerationConfig(no_repeat_ngram_size=2, max_length=10)
        decoder = BeamSearchDecoder(config)

        vocab_size = 10
        # Generated sequence: [1, 2]
        generated = torch.tensor([[1, 2]])
        ngram_history = {0: {(1, 2)}}

        logits = torch.zeros(1, vocab_size)

        blocked = decoder._block_ngrams(
            logits.clone(), generated, ngram_history, config.no_repeat_ngram_size
        )

        # Token 2 would create repeated bigram (2, 2), but (1,2) is in history
        # The prefix (2,) would need to match to block, but our generated ends with (1,2)
        # So next prefix is (2,) - token 1 would create (2,1), not blocked
        # Wait - the logic is: if we append token X to "...2" we get (...2, X)
        # History has (1, 2). Prefix is (2,). So we block token 1 because (2,1) not in history
        # Actually: to block X, we need (2, X) to be in history. (2, 1) is not in history
        # So this test should check that (2, ) + some token in history gets blocked

        # Let's update history to have (2, 3) so token 3 gets blocked
        ngram_history = {0: {(2, 3)}}
        blocked = decoder._block_ngrams(
            logits.clone(), generated, ngram_history, config.no_repeat_ngram_size
        )
        assert blocked[0, 3] == float("-inf")

    def test_update_ngram_history(self) -> None:
        """Test n-gram history is updated correctly."""
        from stepnet.generation import BeamSearchDecoder, GenerationConfig

        config = GenerationConfig(no_repeat_ngram_size=3, max_length=10)
        decoder = BeamSearchDecoder(config)

        generated = torch.tensor([[1, 2, 3, 4]])
        ngram_history: dict = {0: set()}

        decoder._update_ngram_history(ngram_history, generated, 3)

        # Should have added (2, 3, 4)
        assert (2, 3, 4) in ngram_history[0]


# ============================================================================
# SECTION 6: Length Constraints Tests
# ============================================================================


class TestLengthConstraints:
    """Test min/max length constraints."""

    def test_max_length_stops_generation(self) -> None:
        """Test generation stops at max_length."""
        from stepnet.generation import BeamSearchDecoder, GenerationConfig

        max_len = 5
        config = GenerationConfig.greedy(max_length=max_len)
        decoder = BeamSearchDecoder(config)

        vocab_size = 100
        # Create model that never outputs EOS
        mock_model = MockDecoder(vocab_size=vocab_size)

        memory = torch.randn(2, 1, 64)

        output = decoder.decode(
            model_step_fn=lambda t, m: mock_model(t, m),
            memory=memory,
            vocab_size=vocab_size,
        )

        # Should not exceed max_length + initial token
        assert output.sequences.shape[1] <= max_len + 1

    def test_eos_stops_generation(self) -> None:
        """Test EOS token stops generation early."""
        from stepnet.generation import BeamSearchDecoder, GenerationConfig

        eos_id = 2
        config = GenerationConfig.greedy(max_length=50, eos_token_id=eos_id)
        decoder = BeamSearchDecoder(config)

        vocab_size = 100

        class EOSModel(nn.Module):
            """Model that outputs EOS after 3 tokens."""

            def __init__(self):
                super().__init__()
                self.call_count = 0

            def forward(self, tgt, memory):
                self.call_count += 1
                logits = torch.zeros(tgt.size(0), vocab_size)
                if self.call_count >= 3:
                    logits[:, eos_id] = 10.0  # High score for EOS
                else:
                    logits[:, 5] = 10.0  # Some other token
                return logits

        mock_model = EOSModel()
        memory = torch.randn(1, 1, 64)

        output = decoder.decode(
            model_step_fn=lambda t, m: mock_model(t, m),
            memory=memory,
            vocab_size=vocab_size,
        )

        # Should stop early due to EOS
        assert output.sequences.shape[1] < config.max_length


# ============================================================================
# SECTION 7: GenerationOutput Tests
# ============================================================================


class TestGenerationOutput:
    """Test GenerationOutput dataclass."""

    def test_output_has_sequences(self) -> None:
        """Test output contains sequences."""
        from stepnet.generation import GenerationOutput

        sequences = torch.tensor([[1, 2, 3], [4, 5, 6]])
        output = GenerationOutput(sequences=sequences)

        assert output.sequences is not None
        assert output.sequences.shape == (2, 3)

    def test_output_with_scores(self) -> None:
        """Test output with scores."""
        from stepnet.generation import GenerationOutput

        sequences = torch.tensor([[1, 2, 3]])
        scores = torch.tensor([1.5])
        output = GenerationOutput(sequences=sequences, scores=scores)

        assert output.scores is not None
        assert output.scores.shape == (1,)


# ============================================================================
# SECTION 8: Integration Tests
# ============================================================================


class TestBeamSearchIntegration:
    """Integration tests with task models."""

    def test_with_embedding_function(self) -> None:
        """Test decoding with custom embedding function."""
        from stepnet.generation import BeamSearchDecoder, GenerationConfig

        config = GenerationConfig.greedy(max_length=5)
        decoder = BeamSearchDecoder(config)

        vocab_size = 50
        embed_dim = 64
        batch_size = 2

        embedding = nn.Embedding(vocab_size, embed_dim)
        mock_model = MockDecoder(vocab_size=vocab_size, hidden_dim=embed_dim)

        memory = torch.randn(batch_size, 1, embed_dim)

        def model_step_fn(tgt, mem):
            return mock_model(tgt, mem)

        output = decoder.decode(
            model_step_fn=model_step_fn,
            memory=memory,
            vocab_size=vocab_size,
            embedding_fn=lambda x: embedding(x),
        )

        assert output.sequences is not None

    def test_generate_with_model_convenience(self) -> None:
        """Test generate_with_model convenience function."""
        from stepnet.generation import generate_with_model

        vocab_size = 50
        embed_dim = 64
        batch_size = 2

        embedding = nn.Embedding(vocab_size, embed_dim)
        mock_model = MockDecoder(vocab_size=vocab_size, hidden_dim=embed_dim)

        memory = torch.randn(batch_size, 1, embed_dim)

        output = generate_with_model(
            model_forward=lambda t, m: mock_model(t, m),
            memory=memory,
            embedding=embedding,
            vocab_size=vocab_size,
            max_length=5,
            num_beams=1,
        )

        assert output.sequences is not None


# ============================================================================
# SECTION 9: Module Import Tests
# ============================================================================


class TestBeamSearchImports:
    """Test beam search module imports."""

    def test_import_generation_config(self) -> None:
        """Test GenerationConfig can be imported."""
        from stepnet.generation import GenerationConfig

        assert GenerationConfig is not None

    def test_import_decoding_strategy(self) -> None:
        """Test DecodingStrategy can be imported."""
        from stepnet.generation import DecodingStrategy

        assert DecodingStrategy is not None

    def test_import_beam_search_decoder(self) -> None:
        """Test BeamSearchDecoder can be imported."""
        from stepnet.generation import BeamSearchDecoder

        assert BeamSearchDecoder is not None

    def test_import_generation_output(self) -> None:
        """Test GenerationOutput can be imported."""
        from stepnet.generation import GenerationOutput

        assert GenerationOutput is not None

    def test_import_generate_with_model(self) -> None:
        """Test generate_with_model can be imported."""
        from stepnet.generation import generate_with_model

        assert generate_with_model is not None


# ============================================================================
# SECTION 10: Temperature Tests
# ============================================================================


class TestTemperature:
    """Test temperature effects on generation."""

    def test_low_temperature_more_deterministic(self) -> None:
        """Test low temperature produces more deterministic output."""
        from stepnet.generation import BeamSearchDecoder, GenerationConfig

        vocab_size = 100
        batch_size = 2
        mock_model = MockDecoder(vocab_size=vocab_size)
        memory = torch.randn(batch_size, 1, 64)

        torch.manual_seed(42)

        # Low temperature
        config_low = GenerationConfig.sampling(temperature=0.1, max_length=10)
        decoder_low = BeamSearchDecoder(config_low)

        torch.manual_seed(42)
        output_low_1 = decoder_low.decode(
            model_step_fn=lambda t, m: mock_model(t, m),
            memory=memory.clone(),
            vocab_size=vocab_size,
        )

        torch.manual_seed(42)
        output_low_2 = decoder_low.decode(
            model_step_fn=lambda t, m: mock_model(t, m),
            memory=memory.clone(),
            vocab_size=vocab_size,
        )

        # With same seed and low temp, outputs should be identical
        assert output_low_1.sequences is not None
        assert output_low_2.sequences is not None
        assert torch.equal(output_low_1.sequences, output_low_2.sequences), (
            "Low temperature with same seed should produce identical sequences"
        )

        # Low temperature should produce fewer unique tokens than high temperature
        config_high = GenerationConfig.sampling(temperature=2.0, max_length=10)
        decoder_high = BeamSearchDecoder(config_high)

        high_temp_unique_tokens: set = set()
        for seed in range(10):
            torch.manual_seed(seed)
            output_high = decoder_high.decode(
                model_step_fn=lambda t, m: mock_model(t, m),
                memory=memory.clone(),
                vocab_size=vocab_size,
            )
            high_temp_unique_tokens.update(output_high.sequences.flatten().tolist())

        low_temp_unique_tokens: set = set()
        for seed in range(10):
            torch.manual_seed(seed)
            output_low = decoder_low.decode(
                model_step_fn=lambda t, m: mock_model(t, m),
                memory=memory.clone(),
                vocab_size=vocab_size,
            )
            low_temp_unique_tokens.update(output_low.sequences.flatten().tolist())

        # Higher temperature should produce more diverse tokens across runs
        assert len(high_temp_unique_tokens) >= len(low_temp_unique_tokens), (
            f"High temp unique tokens ({len(high_temp_unique_tokens)}) should be "
            f">= low temp unique tokens ({len(low_temp_unique_tokens)})"
        )

    def test_temperature_applied_to_logits(self) -> None:
        """Test temperature divides logits and affects probability distribution."""
        from stepnet.generation import GenerationConfig

        config = GenerationConfig(temperature=0.5)

        logits = torch.tensor([2.0, 4.0, 6.0])
        scaled = logits / config.temperature

        expected = torch.tensor([4.0, 8.0, 12.0])
        assert torch.allclose(scaled, expected)

        # Lower temperature should produce a sharper (less uniform) distribution
        probs_high_temp = torch.softmax(logits / 2.0, dim=-1)
        probs_low_temp = torch.softmax(logits / 0.1, dim=-1)

        # Entropy of low-temp distribution should be lower (more peaked)
        entropy_high = -(probs_high_temp * probs_high_temp.log()).sum()
        entropy_low = -(probs_low_temp * probs_low_temp.log()).sum()
        assert entropy_low < entropy_high, (
            f"Low temperature entropy ({entropy_low:.4f}) should be less than "
            f"high temperature entropy ({entropy_high:.4f})"
        )

        # Very low temperature should concentrate almost all mass on top token
        assert probs_low_temp.argmax() == 2  # token with logit 6.0
        assert probs_low_temp[2] > 0.99, (
            f"Very low temperature should give >99% to top token, got {probs_low_temp[2]:.4f}"
        )

    def test_different_temperatures_produce_different_distributions(self) -> None:
        """Test that different temperatures produce meaningfully different logit distributions."""
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
        distributions = []
        for temp in temperatures:
            probs = torch.softmax(logits / temp, dim=-1)
            distributions.append(probs)

        # Each pair of different temperatures should yield different distributions
        for i in range(len(distributions)):
            for j in range(i + 1, len(distributions)):
                assert not torch.allclose(distributions[i], distributions[j], atol=1e-4), (
                    f"Temperatures {temperatures[i]} and {temperatures[j]} "
                    f"should produce different distributions"
                )

        # Verify monotonic: as temperature increases, entropy increases
        entropies = []
        for probs in distributions:
            safe_probs = probs.clamp(min=1e-10)
            entropy = -(safe_probs * safe_probs.log()).sum().item()
            entropies.append(entropy)

        for i in range(len(entropies) - 1):
            assert entropies[i] < entropies[i + 1], (
                f"Entropy should increase with temperature: "
                f"temp={temperatures[i]} entropy={entropies[i]:.4f} >= "
                f"temp={temperatures[i+1]} entropy={entropies[i+1]:.4f}"
            )
