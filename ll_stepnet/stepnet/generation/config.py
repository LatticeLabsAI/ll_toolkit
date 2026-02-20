"""Generation configuration for CAD model decoding.

This module provides a unified configuration for text generation from
CAD-aware models including captioning, QA, and code generation tasks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

_log = logging.getLogger(__name__)


class DecodingStrategy(Enum):
    """Available decoding strategies."""

    GREEDY = auto()
    """Select highest probability token at each step."""

    BEAM_SEARCH = auto()
    """Maintain multiple hypotheses and select best overall sequence."""

    SAMPLING = auto()
    """Sample from probability distribution (optionally with top-k/top-p)."""

    DIVERSE_BEAM = auto()
    """Beam search with diversity penalty to encourage varied outputs."""


@dataclass
class GenerationConfig:
    """Configuration for text generation.

    Controls the decoding behavior for autoregressive generation from
    CAD-aware models like STEPForCaptioning and STEPForQA.

    Attributes:
        max_length: Maximum number of tokens to generate.
        min_length: Minimum number of tokens to generate (before EOS).
        num_beams: Number of beams for beam search (1 = greedy/sampling).
        temperature: Sampling temperature (lower = more deterministic).
        top_k: If > 0, only sample from top-k tokens.
        top_p: If < 1.0, sample from smallest set with cumulative prob >= top_p.
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty).
        length_penalty: Exponential penalty on sequence length (< 1 favors shorter).
        no_repeat_ngram_size: Prevent repeating n-grams of this size.
        diversity_penalty: Penalty for diverse beam search (only with DIVERSE_BEAM).
        num_beam_groups: Number of groups for diverse beam search.
        eos_token_id: End of sequence token ID.
        pad_token_id: Padding token ID.
        bos_token_id: Beginning of sequence token ID.
        early_stopping: Stop when num_beams hypotheses reach EOS.
        do_sample: Whether to sample (True) or use greedy/beam (False).
        use_cache: Whether to use KV-cache for faster generation.
        bad_words_ids: List of token IDs to never generate.
        forced_eos_token_id: Force this token at max_length.
    """

    # Length constraints
    max_length: int = 64
    min_length: int = 1

    # Beam search
    num_beams: int = 4
    num_beam_groups: int = 1
    diversity_penalty: float = 0.0

    # Sampling
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    do_sample: bool = False

    # Penalties
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0

    # Special tokens
    eos_token_id: int = 2
    pad_token_id: int = 0
    bos_token_id: int = 1
    forced_eos_token_id: Optional[int] = None

    # Control
    early_stopping: bool = True
    use_cache: bool = True

    # Bad words
    bad_words_ids: Optional[List[List[int]]] = None

    def get_strategy(self) -> DecodingStrategy:
        """Determine decoding strategy from config."""
        if self.do_sample:
            return DecodingStrategy.SAMPLING
        if self.num_beams == 1:
            return DecodingStrategy.GREEDY
        if self.diversity_penalty > 0 and self.num_beam_groups > 1:
            return DecodingStrategy.DIVERSE_BEAM
        return DecodingStrategy.BEAM_SEARCH

    def validate(self) -> None:
        """Validate configuration consistency.

        Raises:
            ValueError: If configuration has invalid combinations.
        """
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        if self.min_length < 0:
            raise ValueError(f"min_length must be non-negative, got {self.min_length}")
        if self.min_length > self.max_length:
            raise ValueError(
                f"min_length ({self.min_length}) > max_length ({self.max_length})"
            )
        if self.num_beams < 1:
            raise ValueError(f"num_beams must be >= 1, got {self.num_beams}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if not (0.0 <= self.top_p <= 1.0):
            raise ValueError(f"top_p must be in [0, 1], got {self.top_p}")
        if self.repetition_penalty < 0:
            raise ValueError(
                f"repetition_penalty must be >= 0, got {self.repetition_penalty}"
            )
        if self.diversity_penalty > 0 and self.num_beam_groups <= 1:
            _log.warning(
                "diversity_penalty > 0 but num_beam_groups <= 1, "
                "diversity penalty will have no effect"
            )
        if self.num_beam_groups > 1 and self.num_beams % self.num_beam_groups != 0:
            raise ValueError(
                f"num_beams ({self.num_beams}) must be divisible by "
                f"num_beam_groups ({self.num_beam_groups})"
            )

    @classmethod
    def greedy(cls, max_length: int = 64, **kwargs) -> "GenerationConfig":
        """Create a greedy decoding config."""
        return cls(max_length=max_length, num_beams=1, do_sample=False, **kwargs)

    @classmethod
    def beam_search(
        cls, num_beams: int = 4, max_length: int = 64, **kwargs
    ) -> "GenerationConfig":
        """Create a beam search config."""
        return cls(
            max_length=max_length, num_beams=num_beams, do_sample=False, **kwargs
        )

    @classmethod
    def sampling(
        cls,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        max_length: int = 64,
        **kwargs,
    ) -> "GenerationConfig":
        """Create a sampling config with top-k and nucleus filtering."""
        return cls(
            max_length=max_length,
            num_beams=1,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **kwargs,
        )
