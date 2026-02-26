"""Beam search and sampling-based decoding for CAD models.

This module provides shared decoding functionality for autoregressive
generation from CAD-aware models including captioning and QA tasks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DecodingStrategy, GenerationConfig

_log = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    """Result of generation.

    Attributes:
        sequences: Generated token sequences [batch_size, seq_len].
        scores: Final sequence scores [batch_size] (log-prob or beam score).
        all_scores: Per-step scores if return_all_scores was True.
        attentions: Attention weights if output_attentions was True.
    """

    sequences: torch.Tensor
    scores: Optional[torch.Tensor] = None
    all_scores: Optional[List[torch.Tensor]] = None
    attentions: Optional[List[torch.Tensor]] = None


class BeamSearchDecoder:
    """Unified beam search and sampling decoder for autoregressive generation.

    This class provides configurable decoding including:
    - Greedy decoding (num_beams=1, do_sample=False)
    - Beam search (num_beams>1, do_sample=False)
    - Sampling with temperature, top-k, and top-p
    - Repetition penalty
    - Length normalization
    - Min/max length constraints

    Example:
        >>> config = GenerationConfig(num_beams=4, max_length=64)
        >>> decoder = BeamSearchDecoder(config)
        >>> output = decoder.decode(
        ...     model=captioning_model,
        ...     memory=encoded_step,
        ...     vocab_size=50000,
        ... )
        >>> generated_ids = output.sequences
    """

    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
    ) -> None:
        """Initialize the decoder.

        Args:
            config: Generation configuration. Uses defaults if not provided.
        """
        self.config = config or GenerationConfig()
        self.config.validate()

    def decode(
        self,
        model_step_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        memory: torch.Tensor,
        vocab_size: int,
        embedding_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        initial_tokens: Optional[torch.Tensor] = None,
        return_all_scores: bool = False,
    ) -> GenerationOutput:
        """Run decoding to generate token sequences.

        Args:
            model_step_fn: Function that takes (generated_embeddings, memory)
                and returns logits [batch*beams, vocab_size].
            memory: Encoder memory [batch_size, mem_len, dim].
            vocab_size: Size of output vocabulary.
            embedding_fn: Optional function to embed token IDs to vectors.
                If None, uses one-hot representations.
            initial_tokens: Optional initial tokens [batch_size, init_len].
                If None, starts with bos_token_id.
            return_all_scores: Whether to return all step scores.

        Returns:
            GenerationOutput with generated sequences and scores.
        """
        strategy = self.config.get_strategy()

        if strategy == DecodingStrategy.GREEDY:
            return self._greedy_decode(
                model_step_fn, memory, vocab_size, embedding_fn, initial_tokens
            )
        elif strategy == DecodingStrategy.SAMPLING:
            return self._sampling_decode(
                model_step_fn, memory, vocab_size, embedding_fn, initial_tokens
            )
        elif strategy == DecodingStrategy.DIVERSE_BEAM:
            return self._diverse_beam_search_decode(
                model_step_fn,
                memory,
                vocab_size,
                embedding_fn,
                initial_tokens,
                return_all_scores,
            )
        else:  # BEAM_SEARCH
            return self._beam_search_decode(
                model_step_fn,
                memory,
                vocab_size,
                embedding_fn,
                initial_tokens,
                return_all_scores,
            )

    def _greedy_decode(
        self,
        model_step_fn: Callable,
        memory: torch.Tensor,
        vocab_size: int,
        embedding_fn: Optional[Callable],
        initial_tokens: Optional[torch.Tensor],
    ) -> GenerationOutput:
        """Greedy decoding - always pick highest probability token."""
        batch_size = memory.size(0)
        device = memory.device
        config = self.config

        # Initialize sequence
        if initial_tokens is not None:
            generated = initial_tokens
        else:
            generated = torch.full(
                (batch_size, 1),
                config.bos_token_id,
                dtype=torch.long,
                device=device,
            )

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        scores = torch.zeros(batch_size, device=device)

        # N-gram tracking for no_repeat_ngram
        ngram_history: Dict[int, set] = {i: set() for i in range(batch_size)}

        for step in range(config.max_length - generated.size(1)):
            # Get embeddings
            if embedding_fn is not None:
                tgt = embedding_fn(generated)
            else:
                # Simple one-hot fallback
                tgt = F.one_hot(generated, vocab_size).float()

            # Model step
            logits = model_step_fn(tgt, memory)  # [batch, vocab]

            # Apply temperature
            if config.temperature != 1.0:
                logits = logits / config.temperature

            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits, generated)

            # Block n-grams
            if config.no_repeat_ngram_size > 0:
                logits = self._block_ngrams(
                    logits, generated, ngram_history, config.no_repeat_ngram_size
                )

            # Block EOS if min_length not reached
            if step < config.min_length - 1:
                logits[:, config.eos_token_id] = float("-inf")

            # Force EOS at max_length
            if step == config.max_length - generated.size(1) - 1:
                if config.forced_eos_token_id is not None:
                    logits = torch.full_like(logits, float("-inf"))
                    logits[:, config.forced_eos_token_id] = 0

            # Greedy selection
            next_token_scores, next_token = logits.max(dim=-1)
            next_token = next_token.unsqueeze(-1)

            # Update scores
            scores = scores + next_token_scores * (~finished).float()

            # Append token
            generated = torch.cat([generated, next_token], dim=-1)

            # Update n-gram history
            self._update_ngram_history(
                ngram_history, generated, config.no_repeat_ngram_size
            )

            # Check for EOS
            finished = finished | (next_token.squeeze(-1) == config.eos_token_id)
            if finished.all():
                break

        return GenerationOutput(sequences=generated, scores=scores)

    def _sampling_decode(
        self,
        model_step_fn: Callable,
        memory: torch.Tensor,
        vocab_size: int,
        embedding_fn: Optional[Callable],
        initial_tokens: Optional[torch.Tensor],
    ) -> GenerationOutput:
        """Sampling-based decoding with top-k and top-p filtering."""
        batch_size = memory.size(0)
        device = memory.device
        config = self.config

        # Initialize sequence
        if initial_tokens is not None:
            generated = initial_tokens
        else:
            generated = torch.full(
                (batch_size, 1),
                config.bos_token_id,
                dtype=torch.long,
                device=device,
            )

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        scores = torch.zeros(batch_size, device=device)

        ngram_history: Dict[int, set] = {i: set() for i in range(batch_size)}

        for step in range(config.max_length - generated.size(1)):
            # Get embeddings
            if embedding_fn is not None:
                tgt = embedding_fn(generated)
            else:
                tgt = F.one_hot(generated, vocab_size).float()

            # Model step
            logits = model_step_fn(tgt, memory)

            # Apply temperature
            logits = logits / config.temperature

            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits, generated)

            # Block n-grams
            if config.no_repeat_ngram_size > 0:
                logits = self._block_ngrams(
                    logits, generated, ngram_history, config.no_repeat_ngram_size
                )

            # Block EOS if min_length not reached
            if step < config.min_length - 1:
                logits[:, config.eos_token_id] = float("-inf")

            # Apply top-k filtering
            if config.top_k > 0:
                logits = self._top_k_filter(logits, config.top_k)

            # Apply top-p (nucleus) filtering
            if config.top_p < 1.0:
                logits = self._top_p_filter(logits, config.top_p)

            # Convert to probabilities and sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Get selected token scores
            next_token_scores = logits.gather(-1, next_token).squeeze(-1)

            # Update scores
            scores = scores + next_token_scores * (~finished).float()

            # Append token
            generated = torch.cat([generated, next_token], dim=-1)

            # Update n-gram history
            self._update_ngram_history(
                ngram_history, generated, config.no_repeat_ngram_size
            )

            # Check for EOS
            finished = finished | (next_token.squeeze(-1) == config.eos_token_id)
            if finished.all():
                break

        return GenerationOutput(sequences=generated, scores=scores)

    def _beam_search_decode(
        self,
        model_step_fn: Callable,
        memory: torch.Tensor,
        vocab_size: int,
        embedding_fn: Optional[Callable],
        initial_tokens: Optional[torch.Tensor],
        return_all_scores: bool = False,
    ) -> GenerationOutput:
        """Beam search decoding with length normalization."""
        batch_size = memory.size(0)
        device = memory.device
        config = self.config
        num_beams = config.num_beams

        # Initialize sequence
        if initial_tokens is not None:
            generated = initial_tokens.repeat_interleave(num_beams, dim=0)
        else:
            generated = torch.full(
                (batch_size * num_beams, 1),
                config.bos_token_id,
                dtype=torch.long,
                device=device,
            )

        # Expand memory for beams
        memory = memory.repeat_interleave(num_beams, dim=0)

        # Initialize beam scores
        beam_scores = torch.zeros(batch_size * num_beams, device=device)
        # Only first beam is active initially
        beam_scores = beam_scores.view(batch_size, num_beams)
        beam_scores[:, 1:] = float("-inf")
        beam_scores = beam_scores.view(-1)

        finished = torch.zeros(batch_size * num_beams, dtype=torch.bool, device=device)
        all_scores = [] if return_all_scores else None

        ngram_history: Dict[int, set] = {i: set() for i in range(batch_size * num_beams)}

        for step in range(config.max_length - generated.size(1)):
            # Get embeddings
            if embedding_fn is not None:
                tgt = embedding_fn(generated)
            else:
                tgt = F.one_hot(generated, vocab_size).float()

            # Model step
            logits = model_step_fn(tgt, memory)  # [batch*beams, vocab]

            # Apply temperature
            if config.temperature != 1.0:
                logits = logits / config.temperature

            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits, generated)

            # Block n-grams
            if config.no_repeat_ngram_size > 0:
                logits = self._block_ngrams(
                    logits, generated, ngram_history, config.no_repeat_ngram_size
                )

            # Block EOS if min_length not reached
            if step < config.min_length - 1:
                logits[:, config.eos_token_id] = float("-inf")

            # Get log probabilities
            log_probs = F.log_softmax(logits, dim=-1)

            # Compute next scores
            next_scores = beam_scores.unsqueeze(-1) + log_probs  # [batch*beams, vocab]

            # Reshape for beam selection [batch, beams * vocab]
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)
            next_scores, next_tokens = next_scores.topk(num_beams, dim=-1)

            if return_all_scores:
                all_scores.append(next_scores.clone())

            # Compute beam and token indices
            beam_indices = next_tokens // vocab_size
            token_indices = next_tokens % vocab_size

            # Update sequences
            batch_indices = torch.arange(batch_size, device=device).unsqueeze(-1)
            flat_beam_indices = (batch_indices * num_beams + beam_indices).view(-1)

            generated = torch.cat(
                [generated[flat_beam_indices], token_indices.view(-1, 1)], dim=-1
            )

            # Update beam scores with length penalty
            seq_len = generated.size(1)
            if config.length_penalty != 1.0:
                length_factor = ((5 + seq_len) / 6) ** config.length_penalty
            else:
                length_factor = 1.0

            beam_scores = next_scores.view(-1) / length_factor

            # Update finished status
            finished = finished[flat_beam_indices] | (
                token_indices.view(-1) == config.eos_token_id
            )

            # Update n-gram history with reordered beams
            new_history: Dict[int, set] = {}
            for i, old_idx in enumerate(flat_beam_indices.tolist()):
                new_history[i] = ngram_history.get(old_idx, set()).copy()
            ngram_history = new_history
            self._update_ngram_history(
                ngram_history, generated, config.no_repeat_ngram_size
            )

            # Early stopping
            if config.early_stopping and finished.all():
                break

        # Select best beam for each batch
        beam_scores = beam_scores.view(batch_size, num_beams)
        best_beam_idx = beam_scores.argmax(dim=-1)

        # Gather best sequences
        generated = generated.view(batch_size, num_beams, -1)
        batch_indices = torch.arange(batch_size, device=device)
        best_sequences = generated[batch_indices, best_beam_idx]

        best_scores = beam_scores[batch_indices, best_beam_idx]

        return GenerationOutput(
            sequences=best_sequences,
            scores=best_scores,
            all_scores=all_scores,
        )

    def _diverse_beam_search_decode(
        self,
        model_step_fn: Callable,
        memory: torch.Tensor,
        vocab_size: int,
        embedding_fn: Optional[Callable],
        initial_tokens: Optional[torch.Tensor],
        return_all_scores: bool = False,
    ) -> GenerationOutput:
        """Diverse beam search with group-based diversity penalty.

        Splits beams into ``num_beam_groups`` groups. Each group runs
        its own beam search, but when scoring candidates for group *g*
        a penalty is applied for tokens already selected by groups
        ``0 .. g-1`` in the current step, encouraging diverse outputs.

        Args:
            model_step_fn: Callable taking (embeddings, memory) → logits.
            memory: Encoder memory ``[batch_size, mem_len, dim]``.
            vocab_size: Output vocabulary size.
            embedding_fn: Optional token → embedding function.
            initial_tokens: Optional initial tokens ``[batch_size, init_len]``.
            return_all_scores: Whether to return per-step scores.

        Returns:
            :class:`GenerationOutput` with generated sequences and scores.
        """
        batch_size = memory.size(0)
        device = memory.device
        config = self.config
        num_beams = config.num_beams
        num_groups = config.num_beam_groups
        diversity_penalty = config.diversity_penalty
        beams_per_group = num_beams // num_groups

        # Initialize sequences per beam
        if initial_tokens is not None:
            generated = initial_tokens.repeat_interleave(num_beams, dim=0)
        else:
            generated = torch.full(
                (batch_size * num_beams, 1),
                config.bos_token_id,
                dtype=torch.long,
                device=device,
            )

        # Expand memory for all beams
        memory_expanded = memory.repeat_interleave(num_beams, dim=0)

        # Initialize beam scores — only first beam per group is active
        beam_scores = torch.zeros(batch_size, num_beams, device=device)
        beam_scores[:, :] = float("-inf")
        for g in range(num_groups):
            beam_scores[:, g * beams_per_group] = 0.0
        beam_scores = beam_scores.view(-1)

        finished = torch.zeros(batch_size * num_beams, dtype=torch.bool, device=device)
        all_scores: Optional[List[torch.Tensor]] = [] if return_all_scores else None

        ngram_history: Dict[int, set] = {i: set() for i in range(batch_size * num_beams)}

        for step in range(config.max_length - generated.size(1)):
            # Get embeddings
            if embedding_fn is not None:
                tgt = embedding_fn(generated)
            else:
                tgt = F.one_hot(generated, vocab_size).float()

            # Model step for all beams at once
            logits = model_step_fn(tgt, memory_expanded)  # [batch*beams, vocab]

            # Apply temperature
            if config.temperature != 1.0:
                logits = logits / config.temperature

            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits, generated)

            # Block n-grams
            if config.no_repeat_ngram_size > 0:
                logits = self._block_ngrams(
                    logits, generated, ngram_history, config.no_repeat_ngram_size
                )

            # Block EOS if min_length not reached
            if step < config.min_length - 1:
                logits[:, config.eos_token_id] = float("-inf")

            log_probs = F.log_softmax(logits, dim=-1)

            # Reshape to [batch, num_beams, vocab]
            log_probs = log_probs.view(batch_size, num_beams, vocab_size)

            # Track tokens selected by previous groups for diversity penalty
            # selected_tokens_per_batch[b] accumulates token counts across groups
            selected_tokens_per_batch: List[torch.Tensor] = [
                torch.zeros(vocab_size, device=device) for _ in range(batch_size)
            ]

            # New tensors to collect results across groups
            next_beam_scores_all = torch.zeros(batch_size, num_beams, device=device)
            next_beam_indices_all = torch.zeros(
                batch_size, num_beams, dtype=torch.long, device=device
            )
            next_token_indices_all = torch.zeros(
                batch_size, num_beams, dtype=torch.long, device=device
            )

            for g in range(num_groups):
                group_start = g * beams_per_group
                group_end = group_start + beams_per_group

                # Extract group beams
                group_log_probs = log_probs[:, group_start:group_end, :]  # [B, bpg, V]
                group_beam_scores = beam_scores.view(batch_size, num_beams)[
                    :, group_start:group_end
                ]  # [B, bpg]

                # Apply diversity penalty from previous groups
                if g > 0 and diversity_penalty > 0.0:
                    for b in range(batch_size):
                        penalty = diversity_penalty * selected_tokens_per_batch[b]
                        # Subtract penalty from log probs for tokens already chosen
                        group_log_probs[b] = group_log_probs[b] - penalty.unsqueeze(0)

                # Compute next scores for this group
                group_next_scores = (
                    group_beam_scores.unsqueeze(-1) + group_log_probs
                )  # [B, bpg, V]
                group_next_scores = group_next_scores.view(
                    batch_size, beams_per_group * vocab_size
                )

                # Select top beams_per_group candidates
                top_scores, top_indices = group_next_scores.topk(
                    beams_per_group, dim=-1
                )

                # Decode beam and token indices (relative to group)
                group_beam_idx = top_indices // vocab_size  # within group
                group_token_idx = top_indices % vocab_size

                # Map back to global beam indices
                global_beam_idx = group_beam_idx + group_start

                # Store results
                next_beam_scores_all[:, group_start:group_end] = top_scores
                next_beam_indices_all[:, group_start:group_end] = global_beam_idx
                next_token_indices_all[:, group_start:group_end] = group_token_idx

                # Update diversity tracking: count selected tokens per batch
                for b in range(batch_size):
                    for t in group_token_idx[b]:
                        selected_tokens_per_batch[b][t.item()] += 1.0

            if return_all_scores and all_scores is not None:
                all_scores.append(next_beam_scores_all.clone())

            # Flatten indices for gathering
            batch_indices = torch.arange(batch_size, device=device).unsqueeze(-1)
            flat_beam_indices = (
                batch_indices * num_beams + next_beam_indices_all
            ).view(-1)

            # Update sequences
            generated = torch.cat(
                [
                    generated[flat_beam_indices],
                    next_token_indices_all.view(-1, 1),
                ],
                dim=-1,
            )

            # Update beam scores with length penalty
            seq_len = generated.size(1)
            if config.length_penalty != 1.0:
                length_factor = ((5 + seq_len) / 6) ** config.length_penalty
            else:
                length_factor = 1.0

            beam_scores = next_beam_scores_all.view(-1) / length_factor

            # Update finished status
            finished = finished[flat_beam_indices] | (
                next_token_indices_all.view(-1) == config.eos_token_id
            )

            # Update n-gram history with reordered beams
            new_history: Dict[int, set] = {}
            for i, old_idx in enumerate(flat_beam_indices.tolist()):
                new_history[i] = ngram_history.get(old_idx, set()).copy()
            ngram_history = new_history
            self._update_ngram_history(
                ngram_history, generated, config.no_repeat_ngram_size
            )

            # Early stopping
            if config.early_stopping and finished.all():
                break

        # Select best beam for each batch
        beam_scores = beam_scores.view(batch_size, num_beams)
        best_beam_idx = beam_scores.argmax(dim=-1)

        generated = generated.view(batch_size, num_beams, -1)
        batch_indices = torch.arange(batch_size, device=device)
        best_sequences = generated[batch_indices, best_beam_idx]
        best_scores = beam_scores[batch_indices, best_beam_idx]

        return GenerationOutput(
            sequences=best_sequences,
            scores=best_scores,
            all_scores=all_scores,
        )

    def _apply_repetition_penalty(
        self, logits: torch.Tensor, generated: torch.Tensor
    ) -> torch.Tensor:
        """Apply repetition penalty to previously generated tokens."""
        penalty = self.config.repetition_penalty
        for i in range(logits.size(0)):
            for token_id in generated[i].unique():
                if logits[i, token_id] > 0:
                    logits[i, token_id] /= penalty
                else:
                    logits[i, token_id] *= penalty
        return logits

    def _block_ngrams(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        ngram_history: Dict[int, set],
        ngram_size: int,
    ) -> torch.Tensor:
        """Block tokens that would create repeated n-grams."""
        if generated.size(1) < ngram_size:
            return logits

        for i in range(logits.size(0)):
            # Get the (n-1)-gram prefix
            prefix = tuple(generated[i, -(ngram_size - 1) :].tolist())
            # Check which tokens would create a repeated n-gram
            for token_id in range(logits.size(1)):
                ngram = prefix + (token_id,)
                if ngram in ngram_history.get(i, set()):
                    logits[i, token_id] = float("-inf")
        return logits

    def _update_ngram_history(
        self,
        ngram_history: Dict[int, set],
        generated: torch.Tensor,
        ngram_size: int,
    ) -> None:
        """Update n-gram history with new tokens."""
        if ngram_size <= 0 or generated.size(1) < ngram_size:
            return

        for i in range(generated.size(0)):
            # Add the latest n-gram
            ngram = tuple(generated[i, -ngram_size:].tolist())
            if i not in ngram_history:
                ngram_history[i] = set()
            ngram_history[i].add(ngram)

    def _top_k_filter(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Filter to keep only top-k tokens."""
        if top_k <= 0:
            return logits

        # Get the k-th largest value
        top_k = min(top_k, logits.size(-1))
        threshold = logits.topk(top_k, dim=-1).values[:, -1].unsqueeze(-1)

        # Mask tokens below threshold
        logits = logits.masked_fill(logits < threshold, float("-inf"))
        return logits

    def _top_p_filter(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Filter to keep tokens with cumulative probability >= top_p (nucleus sampling)."""
        if top_p >= 1.0:
            return logits

        # Sort logits descending
        sorted_logits, sorted_indices = logits.sort(descending=True, dim=-1)
        cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # Find cutoff index
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift to keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter back to original order
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
        return logits


def generate_with_model(
    model_forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    memory: torch.Tensor,
    embedding: nn.Module,
    vocab_size: int,
    config: Optional[GenerationConfig] = None,
    **kwargs,
) -> GenerationOutput:
    """Convenience function for generation with a model.

    Args:
        model_forward: Model's decoder forward function.
        memory: Encoder memory tensor.
        embedding: Embedding layer for tokens.
        vocab_size: Size of output vocabulary.
        config: Generation configuration.
        **kwargs: Additional config overrides.

    Returns:
        GenerationOutput with generated sequences.
    """
    if config is None:
        config = GenerationConfig(**kwargs)
    else:
        # Apply any overrides
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)

    decoder = BeamSearchDecoder(config)
    return decoder.decode(
        model_step_fn=model_forward,
        memory=memory,
        vocab_size=vocab_size,
        embedding_fn=lambda x: embedding(x),
    )
