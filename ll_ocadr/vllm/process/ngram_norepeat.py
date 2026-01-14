"""
N-gram no-repeat logit processor for LL-OCADR.
Prevents repetitive text generation by penalizing repeated n-grams.
Adapted from DeepSeek-OCR's implementation.
"""

from typing import List, Dict, Set, Tuple
import torch
from collections import defaultdict


class NGramNoRepeatLogitsProcessor:
    """
    Logit processor that prevents repetitive n-gram generation.

    This processor tracks generated n-grams and penalizes tokens that would
    complete repeated n-grams, helping to avoid repetitive text output.

    Particularly useful for technical descriptions where the model might
    repeat patterns like "This part has... This part has... This part has..."
    """

    def __init__(
        self,
        ngram_size: int = 3,
        penalty: float = float('inf'),
        min_sequence_length: int = 10
    ):
        """
        Initialize n-gram no-repeat processor.

        Args:
            ngram_size: Size of n-grams to track (default: 3)
            penalty: Penalty to apply to repeated n-grams (default: inf for blocking)
            min_sequence_length: Minimum sequence length before applying penalty
        """
        self.ngram_size = ngram_size
        self.penalty = penalty
        self.min_sequence_length = min_sequence_length

        # Track n-grams per sequence in batch
        self.ngram_history: Dict[int, Set[Tuple[int, ...]]] = defaultdict(set)

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Process logits to penalize repeated n-grams.

        Args:
            input_ids: [batch_size, seq_len] - Generated token IDs so far
            scores: [batch_size, vocab_size] - Logits for next token

        Returns:
            scores: [batch_size, vocab_size] - Modified logits with penalties
        """
        batch_size, vocab_size = scores.shape
        seq_len = input_ids.shape[1]

        # Don't apply penalty for very short sequences
        if seq_len < self.min_sequence_length:
            return scores

        # Process each sequence in batch
        for batch_idx in range(batch_size):
            # Get current sequence for this batch item
            sequence = input_ids[batch_idx].tolist()

            # Extract all n-grams from sequence
            if seq_len >= self.ngram_size:
                # Get the last (ngram_size - 1) tokens as context
                context = tuple(sequence[-(self.ngram_size - 1):])

                # Check which next tokens would complete a repeated n-gram
                for vocab_idx in range(vocab_size):
                    # Form potential n-gram with this next token
                    potential_ngram = context + (vocab_idx,)

                    # If this n-gram already exists, penalize it
                    if potential_ngram in self.ngram_history[batch_idx]:
                        scores[batch_idx, vocab_idx] -= self.penalty

            # Update n-gram history with all n-grams in current sequence
            if seq_len >= self.ngram_size:
                for i in range(seq_len - self.ngram_size + 1):
                    ngram = tuple(sequence[i:i + self.ngram_size])
                    self.ngram_history[batch_idx].add(ngram)

        return scores

    def reset(self):
        """Reset n-gram history."""
        self.ngram_history.clear()

    def reset_batch(self, batch_idx: int):
        """Reset n-gram history for specific batch item."""
        if batch_idx in self.ngram_history:
            del self.ngram_history[batch_idx]


class BigramNoRepeatLogitsProcessor:
    """
    Simplified processor that prevents immediate bigram repetition.

    This is a lightweight version that only prevents the last bigram from
    repeating, useful for preventing "the the" or "is is" type errors.
    """

    def __init__(self, penalty: float = float('inf')):
        """
        Initialize bigram no-repeat processor.

        Args:
            penalty: Penalty for repeated bigrams (default: inf for blocking)
        """
        self.penalty = penalty

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Process logits to prevent immediate bigram repetition.

        Args:
            input_ids: [batch_size, seq_len] - Generated token IDs so far
            scores: [batch_size, vocab_size] - Logits for next token

        Returns:
            scores: [batch_size, vocab_size] - Modified logits
        """
        batch_size = scores.shape[0]
        seq_len = input_ids.shape[1]

        # Need at least 2 tokens to form a bigram
        if seq_len < 2:
            return scores

        # For each batch, penalize tokens that would repeat the last bigram
        for batch_idx in range(batch_size):
            # Get last two tokens
            last_two = input_ids[batch_idx, -2:].tolist()
            prev_token = last_two[0]
            last_token = last_two[1]

            # If the last bigram would be repeated, penalize the token
            # that would complete it
            scores[batch_idx, last_token] -= self.penalty if prev_token == last_token else 0

        return scores


class AdaptiveNGramNoRepeatProcessor:
    """
    Adaptive n-gram processor that adjusts penalty based on context.

    This processor uses a sliding scale penalty that increases with the
    number of times an n-gram has been repeated, rather than a fixed penalty.
    """

    def __init__(
        self,
        ngram_size: int = 3,
        base_penalty: float = 1.0,
        penalty_scale: float = 2.0,
        min_sequence_length: int = 10
    ):
        """
        Initialize adaptive processor.

        Args:
            ngram_size: Size of n-grams to track
            base_penalty: Base penalty for first repetition
            penalty_scale: Scaling factor for each additional repetition
            min_sequence_length: Minimum sequence length before applying
        """
        self.ngram_size = ngram_size
        self.base_penalty = base_penalty
        self.penalty_scale = penalty_scale
        self.min_sequence_length = min_sequence_length

        # Track n-gram counts per sequence
        self.ngram_counts: Dict[int, Dict[Tuple[int, ...], int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Process logits with adaptive penalties.

        Args:
            input_ids: [batch_size, seq_len] - Generated token IDs
            scores: [batch_size, vocab_size] - Logits for next token

        Returns:
            scores: Modified logits
        """
        batch_size, vocab_size = scores.shape
        seq_len = input_ids.shape[1]

        if seq_len < self.min_sequence_length:
            return scores

        for batch_idx in range(batch_size):
            sequence = input_ids[batch_idx].tolist()

            if seq_len >= self.ngram_size:
                context = tuple(sequence[-(self.ngram_size - 1):])

                for vocab_idx in range(vocab_size):
                    potential_ngram = context + (vocab_idx,)

                    # Get repetition count
                    count = self.ngram_counts[batch_idx][potential_ngram]

                    if count > 0:
                        # Apply scaled penalty based on number of repetitions
                        penalty = self.base_penalty * (self.penalty_scale ** count)
                        scores[batch_idx, vocab_idx] -= penalty

            # Update counts
            if seq_len >= self.ngram_size:
                for i in range(seq_len - self.ngram_size + 1):
                    ngram = tuple(sequence[i:i + self.ngram_size])
                    self.ngram_counts[batch_idx][ngram] += 1

        return scores

    def reset(self):
        """Reset n-gram counts."""
        self.ngram_counts.clear()

    def reset_batch(self, batch_idx: int):
        """Reset counts for specific batch item."""
        if batch_idx in self.ngram_counts:
            del self.ngram_counts[batch_idx]


def create_norepeat_processor(
    mode: str = "standard",
    ngram_size: int = 3,
    **kwargs
) -> object:
    """
    Factory function to create no-repeat processor.

    Args:
        mode: Type of processor ("standard", "bigram", or "adaptive")
        ngram_size: Size of n-grams (for standard and adaptive modes)
        **kwargs: Additional arguments for specific processor

    Returns:
        Logit processor instance
    """
    if mode == "bigram":
        return BigramNoRepeatLogitsProcessor(**kwargs)
    elif mode == "adaptive":
        return AdaptiveNGramNoRepeatProcessor(ngram_size=ngram_size, **kwargs)
    else:  # standard
        return NGramNoRepeatLogitsProcessor(ngram_size=ngram_size, **kwargs)


# Example usage configuration for LL-OCADR
def get_recommended_processor_for_cad():
    """
    Get recommended no-repeat processor for CAD descriptions.

    Uses trigram blocking to prevent repetitive technical descriptions
    while allowing necessary technical term repetition.
    """
    return NGramNoRepeatLogitsProcessor(
        ngram_size=3,
        penalty=float('inf'),  # Hard block on repetition
        min_sequence_length=15  # Allow some initial repetition for context
    )
