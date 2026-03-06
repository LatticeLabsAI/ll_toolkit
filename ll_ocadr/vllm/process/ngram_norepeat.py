"""
N-gram no-repeat logit processor for LL-OCADR.
Prevents repetitive text generation by penalizing repeated n-grams.
Adapted from DeepSeek-OCR's implementation.
"""

from typing import List, Dict, Set, Tuple, Union
import torch
from collections import defaultdict, OrderedDict


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
        min_sequence_length: int = 10,
        max_batch_entries: int = 1024,
    ):
        """
        Initialize n-gram no-repeat processor.

        Args:
            ngram_size: Size of n-grams to track (default: 3)
            penalty: Penalty to apply to repeated n-grams (default: inf for blocking)
            min_sequence_length: Minimum sequence length before applying penalty
            max_batch_entries: Maximum batch-index entries to retain before evicting
                oldest entries. Prevents unbounded memory growth in serving scenarios.
        """
        self.ngram_size = ngram_size
        self.penalty = penalty
        self.min_sequence_length = min_sequence_length
        self.max_batch_entries = max_batch_entries

        # OrderedDict so we can evict oldest batch entries (LRU-style)
        self._banned_map: OrderedDict[int, Dict[Tuple[int, ...], Set[int]]] = OrderedDict()
        # Track last seen sequence length per batch item for incremental updates
        self._last_seq_len: Dict[int, int] = {}

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Process logits to penalize repeated n-grams.

        Complexity is O(seq_len) per batch element (scanning sequence for
        matching context prefixes), NOT O(vocab_size).

        Args:
            input_ids: [batch_size, seq_len] - Generated token IDs so far
            scores: [batch_size, vocab_size] - Logits for next token

        Returns:
            scores: [batch_size, vocab_size] - Modified logits with penalties
        """
        seq_len = input_ids.shape[1]

        # Don't apply penalty for very short sequences
        if seq_len < self.min_sequence_length:
            return scores

        batch_size = scores.shape[0]
        n = self.ngram_size

        # Process each sequence in batch
        for batch_idx in range(batch_size):
            sequence = input_ids[batch_idx].tolist()

            # Ensure entry exists; move to end (most-recently-used)
            if batch_idx not in self._banned_map:
                self._banned_map[batch_idx] = defaultdict(set)
                # Evict oldest entries when we exceed the limit
                while len(self._banned_map) > self.max_batch_entries:
                    evicted_key, _ = self._banned_map.popitem(last=False)
                    self._last_seq_len.pop(evicted_key, None)
            else:
                self._banned_map.move_to_end(batch_idx)

            if seq_len >= n:
                # Get the last (n - 1) tokens as context
                context = tuple(sequence[-(n - 1):])

                # Look up banned tokens for this context — O(1)
                banned_tokens = self._banned_map[batch_idx].get(context)
                if banned_tokens:
                    for token_id in banned_tokens:
                        scores[batch_idx, token_id] -= self.penalty

                # Incrementally add only n-grams not yet seen.
                # On first call with a prompt, this registers all n-grams.
                # On subsequent autoregressive steps, only 1 new n-gram is added.
                prev_len = self._last_seq_len.get(batch_idx, 0)
                start = max(0, prev_len - n + 1)
                for i in range(start, seq_len - n + 1):
                    ngram = tuple(sequence[i:i + n])
                    self._banned_map[batch_idx][ngram[:-1]].add(ngram[-1])
                self._last_seq_len[batch_idx] = seq_len

        return scores

    def reset(self):
        """Reset n-gram history."""
        self._banned_map.clear()
        self._last_seq_len.clear()

    def reset_batch(self, batch_idx: int):
        """Reset n-gram history for specific batch item."""
        if batch_idx in self._banned_map:
            del self._banned_map[batch_idx]
        self._last_seq_len.pop(batch_idx, None)


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

        # Need at least 1 token to form a bigram context
        if seq_len < 1:
            return scores

        # For each batch, find tokens that would complete a repeated bigram
        for batch_idx in range(batch_size):
            sequence = input_ids[batch_idx].tolist()
            last_token = sequence[-1]

            # Collect all tokens that previously followed last_token,
            # forming bigrams (last_token, next_token). Penalize those
            # next_tokens so the same bigram doesn't repeat.
            banned: Set[int] = set()
            for i in range(seq_len - 1):
                if sequence[i] == last_token:
                    banned.add(sequence[i + 1])

            for token_id in banned:
                scores[batch_idx, token_id] -= self.penalty

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
        max_penalty: float = 1e4,
        min_sequence_length: int = 10,
        max_batch_entries: int = 1024,
    ):
        """
        Initialize adaptive processor.

        Args:
            ngram_size: Size of n-grams to track
            base_penalty: Base penalty for first repetition
            penalty_scale: Scaling factor for each additional repetition
            max_penalty: Upper bound on penalty to prevent inf/nan overflow
            min_sequence_length: Minimum sequence length before applying
            max_batch_entries: Maximum batch-index entries to retain before evicting
                oldest entries. Prevents unbounded memory growth in serving scenarios.
        """
        self.ngram_size = ngram_size
        self.base_penalty = base_penalty
        self.penalty_scale = penalty_scale
        self.max_penalty = max_penalty
        self.min_sequence_length = min_sequence_length
        self.max_batch_entries = max_batch_entries

        # OrderedDict so we can evict oldest batch entries (LRU-style)
        self._count_map: OrderedDict[int, Dict[Tuple[int, ...], Dict[int, int]]] = OrderedDict()
        # Track last seen sequence length per batch item for incremental updates
        self._last_seq_len: Dict[int, int] = {}

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Process logits with adaptive penalties.

        Complexity is O(num_repeated_tokens) per batch element for penalty
        application, not O(vocab_size).

        Args:
            input_ids: [batch_size, seq_len] - Generated token IDs
            scores: [batch_size, vocab_size] - Logits for next token

        Returns:
            scores: Modified logits
        """
        seq_len = input_ids.shape[1]

        if seq_len < self.min_sequence_length:
            return scores

        batch_size = scores.shape[0]
        n = self.ngram_size

        for batch_idx in range(batch_size):
            sequence = input_ids[batch_idx].tolist()

            # Ensure entry exists; move to end (most-recently-used)
            if batch_idx not in self._count_map:
                self._count_map[batch_idx] = defaultdict(lambda: defaultdict(int))
                # Evict oldest entries when we exceed the limit
                while len(self._count_map) > self.max_batch_entries:
                    evicted_key, _ = self._count_map.popitem(last=False)
                    self._last_seq_len.pop(evicted_key, None)
            else:
                self._count_map.move_to_end(batch_idx)

            if seq_len >= n:
                context = tuple(sequence[-(n - 1):])

                # Look up token counts for this context — O(1) lookup
                token_counts = self._count_map[batch_idx].get(context)
                if token_counts:
                    for token_id, count in token_counts.items():
                        if count > 0:
                            penalty = min(
                                self.base_penalty * (self.penalty_scale ** count),
                                self.max_penalty,
                            )
                            scores[batch_idx, token_id] -= penalty

                # Incrementally add only n-grams not yet seen
                prev_len = self._last_seq_len.get(batch_idx, 0)
                start = max(0, prev_len - n + 1)
                for i in range(start, seq_len - n + 1):
                    ngram = tuple(sequence[i:i + n])
                    self._count_map[batch_idx][ngram[:-1]][ngram[-1]] += 1
                self._last_seq_len[batch_idx] = seq_len

        return scores

    def reset(self):
        """Reset n-gram counts."""
        self._count_map.clear()
        self._last_seq_len.clear()

    def reset_batch(self, batch_idx: int):
        """Reset counts for specific batch item."""
        self._count_map.pop(batch_idx, None)
        self._last_seq_len.pop(batch_idx, None)


def create_norepeat_processor(
    mode: str = "standard",
    ngram_size: int = 3,
    **kwargs
) -> Union[NGramNoRepeatLogitsProcessor, BigramNoRepeatLogitsProcessor, AdaptiveNGramNoRepeatProcessor]:
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
