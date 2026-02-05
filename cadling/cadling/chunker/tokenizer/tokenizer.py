"""Tokenization utilities for CAD chunking.

This module provides tokenizers for accurate token counting and
text splitting, supporting various tokenization strategies.

Classes:
    CADTokenizer: Base tokenizer interface
    GPTTokenizer: OpenAI GPT tokenizer (tiktoken)
    SimpleTokenizer: Whitespace-based tokenizer (fallback)
    HuggingFaceTokenizer: HuggingFace transformers tokenizer
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import List, Optional

_log = logging.getLogger(__name__)


class CADTokenizer(ABC):
    """Abstract base class for tokenizers.

    Provides interface for token counting and text tokenization.
    """

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into tokens.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        pass

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text
        """
        pass


class SimpleTokenizer(CADTokenizer):
    """Simple whitespace-based tokenizer.

    Fast fallback tokenizer that splits on whitespace and punctuation.
    """

    def __init__(self):
        """Initialize simple tokenizer."""
        self.pattern = re.compile(r"\w+|[^\w\s]")

    def count_tokens(self, text: str) -> int:
        """Count tokens by splitting on whitespace.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        return len(self.tokenize(text))

    def tokenize(self, text: str) -> List[str]:
        """Tokenize by splitting on whitespace and punctuation.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        return self.pattern.findall(text)

    def encode(self, text: str) -> List[int]:
        """Encode text (simple hash-based encoding).

        Args:
            text: Input text

        Returns:
            List of token hashes
        """
        tokens = self.tokenize(text)
        return [hash(token) % 50000 for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """Decode not supported for simple tokenizer.

        Args:
            token_ids: Token IDs

        Returns:
            Empty string (not supported)
        """
        _log.warning("Decode not supported for SimpleTokenizer")
        return ""


class GPTTokenizer(CADTokenizer):
    """OpenAI GPT tokenizer using tiktoken.

    Provides accurate token counting for GPT models.
    """

    def __init__(self, model: str = "gpt-4"):
        """Initialize GPT tokenizer.

        Args:
            model: Model name (gpt-4, gpt-3.5-turbo, etc.)
        """
        self.model = model
        self.encoding = None

        try:
            import tiktoken

            self.encoding = tiktoken.encoding_for_model(model)
            _log.info(f"Initialized GPT tokenizer for model: {model}")
        except ImportError:
            _log.warning("tiktoken not installed, falling back to simple tokenizer")
            self.fallback = SimpleTokenizer()
        except Exception as e:
            _log.warning(f"Failed to initialize tiktoken: {e}, using fallback")
            self.fallback = SimpleTokenizer()

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        if self.encoding is not None:
            return len(self.encoding.encode(text))
        else:
            return self.fallback.count_tokens(text)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text (returns token strings if possible).

        Args:
            text: Input text

        Returns:
            List of token strings
        """
        if self.encoding is not None:
            token_ids = self.encoding.encode(text)
            # Decode individual tokens
            tokens = []
            for tid in token_ids:
                try:
                    tokens.append(self.encoding.decode([tid]))
                except Exception:
                    tokens.append(f"<{tid}>")
            return tokens
        else:
            return self.fallback.tokenize(text)

    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        if self.encoding is not None:
            return self.encoding.encode(text)
        else:
            return self.fallback.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: Token IDs

        Returns:
            Decoded text
        """
        if self.encoding is not None:
            return self.encoding.decode(token_ids)
        else:
            return self.fallback.decode(token_ids)


class HuggingFaceTokenizer(CADTokenizer):
    """HuggingFace transformers tokenizer.

    Supports various transformer models from HuggingFace.
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        """Initialize HuggingFace tokenizer.

        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self.tokenizer = None

        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            _log.info(f"Initialized HuggingFace tokenizer: {model_name}")
        except ImportError:
            _log.warning("transformers not installed, falling back to simple tokenizer")
            self.fallback = SimpleTokenizer()
        except Exception as e:
            _log.warning(f"Failed to load tokenizer {model_name}: {e}, using fallback")
            self.fallback = SimpleTokenizer()

    def count_tokens(self, text: str) -> int:
        """Count tokens using HuggingFace tokenizer.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        else:
            return self.fallback.count_tokens(text)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into tokens.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        if self.tokenizer is not None:
            return self.tokenizer.tokenize(text)
        else:
            return self.fallback.tokenize(text)

    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        if self.tokenizer is not None:
            return self.tokenizer.encode(text, add_special_tokens=False)
        else:
            return self.fallback.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: Token IDs

        Returns:
            Decoded text
        """
        if self.tokenizer is not None:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        else:
            return self.fallback.decode(token_ids)


class TokenSplitter:
    """Utility for splitting text at token boundaries.

    Splits text into chunks that respect token limits while
    preserving semantic boundaries where possible.
    """

    def __init__(
        self,
        tokenizer: Optional[CADTokenizer] = None,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
    ):
        """Initialize token splitter.

        Args:
            tokenizer: Tokenizer to use (defaults to SimpleTokenizer)
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Overlap between chunks
        """
        self.tokenizer = tokenizer or SimpleTokenizer()
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def split(self, text: str) -> List[str]:
        """Split text into token-limited chunks.

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        tokens = self.tokenizer.tokenize(text)

        if len(tokens) <= self.max_tokens:
            return [text]

        chunks = []
        start = 0

        while start < len(tokens):
            end = start + self.max_tokens
            chunk_tokens = tokens[start:end]

            # Try to decode tokens back to text
            if hasattr(self.tokenizer, "decode"):
                chunk_text = self.tokenizer.decode(
                    self.tokenizer.encode(" ".join(chunk_tokens))
                )
            else:
                chunk_text = " ".join(chunk_tokens)

            chunks.append(chunk_text)

            # Move start with overlap
            start = end - self.overlap_tokens

        return chunks

    def split_at_boundaries(
        self, text: str, boundaries: Optional[List[str]] = None
    ) -> List[str]:
        """Split text at semantic boundaries while respecting token limits.

        Args:
            text: Input text
            boundaries: Boundary markers (e.g., ['\\n\\n', '\\n', '.'])

        Returns:
            List of text chunks
        """
        if boundaries is None:
            boundaries = ["\n\n", "\n", ". ", " "]

        chunks = []
        current_chunk = []
        current_tokens = 0

        # Split text by boundaries
        segments = self._split_by_boundaries(text, boundaries)

        for segment in segments:
            segment_tokens = self.tokenizer.count_tokens(segment)

            if current_tokens + segment_tokens > self.max_tokens and current_chunk:
                # Start new chunk
                chunks.append("".join(current_chunk))
                current_chunk = []
                current_tokens = 0

            current_chunk.append(segment)
            current_tokens += segment_tokens

        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks

    def _split_by_boundaries(self, text: str, boundaries: List[str]) -> List[str]:
        """Split text by boundary markers, preserving the markers.

        Args:
            text: Input text
            boundaries: Boundary markers

        Returns:
            List of segments
        """
        if not boundaries:
            return [text]

        boundary = boundaries[0]
        parts = text.split(boundary)

        if len(parts) == 1:
            # Try next boundary
            return self._split_by_boundaries(text, boundaries[1:])

        # Reconstruct with boundary preserved
        segments = []
        for i, part in enumerate(parts[:-1]):
            segments.append(part + boundary)

        # Add last part (no boundary at end)
        if parts[-1]:
            segments.append(parts[-1])

        return segments


def get_tokenizer(
    tokenizer_type: str = "simple", model: Optional[str] = None
) -> CADTokenizer:
    """Factory function to get tokenizer instance.

    Args:
        tokenizer_type: Type of tokenizer ("simple", "gpt", "huggingface")
        model: Model name for GPT or HuggingFace tokenizers

    Returns:
        CADTokenizer instance
    """
    if tokenizer_type == "gpt":
        model = model or "gpt-4"
        return GPTTokenizer(model)
    elif tokenizer_type == "huggingface":
        model = model or "bert-base-uncased"
        return HuggingFaceTokenizer(model)
    else:  # simple
        return SimpleTokenizer()
