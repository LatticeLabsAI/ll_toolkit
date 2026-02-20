"""Utilities for CAD SDG Q&A generation.

This module provides utility functions for LLM initialization,
file I/O, and text post-processing.

Functions:
    initialize_llm: Initialize LLM client
    save_to_file: Save objects to JSONL file
    load_from_file: Load objects from JSONL file
    postprocess_question: Clean generated question
    postprocess_answer: Clean generated answer

Classes:
    ChatAgent: Simple LLM chat wrapper
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterator, TypeVar

from pydantic import BaseModel

from cadling.sdg.qa.base import (
    CADGenQAC,
    CADQaChunk,
    Critique,
    LlmOptions,
    LlmProvider,
)

_log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Check available LLM libraries
_OPENAI_AVAILABLE = False
_ANTHROPIC_AVAILABLE = False
_MLX_AVAILABLE = False

try:
    import openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _log.debug("openai not available")

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _log.debug("anthropic not available")

try:
    import mlx_lm
    _MLX_AVAILABLE = True
except ImportError:
    _log.debug("mlx-lm not available")


def initialize_llm(options: LlmOptions) -> Any:
    """Initialize LLM client based on provider.

    Args:
        options: LLM configuration options

    Returns:
        LLM client instance

    Raises:
        ImportError: If required library not installed
        ValueError: If provider not supported
    """
    if options.provider == LlmProvider.OPENAI:
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )

        api_key = options.api_key.get_secret_value() if options.api_key else None
        client = openai.OpenAI(api_key=api_key)
        _log.info(f"Initialized OpenAI client for {options.model_id}")
        return client

    elif options.provider == LlmProvider.ANTHROPIC:
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )

        api_key = options.api_key.get_secret_value() if options.api_key else None
        client = anthropic.Anthropic(api_key=api_key)
        _log.info(f"Initialized Anthropic client for {options.model_id}")
        return client

    elif options.provider == LlmProvider.VLLM:
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "openai package required for vLLM (OpenAI-compatible API). "
                "Install with: pip install openai"
            )

        # vLLM exposes OpenAI-compatible API, default port 8000
        base_url = options.url or "http://localhost:8000/v1"
        api_key = options.api_key.get_secret_value() if options.api_key else "EMPTY"
        client = openai.OpenAI(base_url=base_url, api_key=api_key)
        _log.info(f"Initialized vLLM client at {base_url} for {options.model_id}")
        return client

    elif options.provider == LlmProvider.OLLAMA:
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "openai package required for Ollama (OpenAI-compatible API). "
                "Install with: pip install openai"
            )

        # Ollama exposes OpenAI-compatible API on port 11434
        base_url = options.url or "http://localhost:11434/v1"
        client = openai.OpenAI(base_url=base_url, api_key="ollama")
        _log.info(f"Initialized Ollama client at {base_url} for {options.model_id}")
        return client

    elif options.provider == LlmProvider.OPENAI_COMPATIBLE:
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "openai package required for OpenAI-compatible API. "
                "Install with: pip install openai"
            )

        if not options.url:
            raise ValueError("url is required for OPENAI_COMPATIBLE provider")

        api_key = options.api_key.get_secret_value() if options.api_key else "EMPTY"
        client = openai.OpenAI(base_url=options.url, api_key=api_key)
        _log.info(f"Initialized OpenAI-compatible client at {options.url}")
        return client

    elif options.provider == LlmProvider.MLX:
        if not _MLX_AVAILABLE:
            raise ImportError(
                "mlx-lm package required for MLX provider. "
                "Install with: pip install mlx-lm"
            )

        import mlx_lm

        # Load model and tokenizer (may return 2 or 3 values)
        result = mlx_lm.load(options.model_id)
        model = result[0]
        tokenizer = result[1]
        _log.info(f"Initialized MLX model: {options.model_id}")
        return {"model": model, "tokenizer": tokenizer, "model_id": options.model_id}

    else:
        raise ValueError(f"Unsupported provider: {options.provider}")


class ChatAgent:
    """Simple LLM chat wrapper for consistent API access.

    Provides a unified interface for making LLM calls regardless
    of the underlying provider.

    Attributes:
        client: LLM client instance
        model_id: Model identifier
        provider: LLM provider type
        temperature: Sampling temperature
        max_tokens: Maximum response tokens
    """

    def __init__(
        self,
        client: Any,
        model_id: str,
        provider: LlmProvider,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        """Initialize chat agent.

        Args:
            client: LLM client instance
            model_id: Model identifier
            provider: LLM provider type
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
        """
        self.client = client
        self.model_id = model_id
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens

    @classmethod
    def from_options(cls, options: LlmOptions) -> "ChatAgent":
        """Create ChatAgent from options.

        Args:
            options: LLM configuration options

        Returns:
            Configured ChatAgent instance
        """
        client = initialize_llm(options)
        return cls(
            client=client,
            model_id=options.model_id,
            provider=options.provider,
            temperature=options.temperature,
            max_tokens=options.max_tokens,
        )

    def ask(self, prompt: str, max_tokens: int | None = None) -> str:
        """Send prompt to LLM and get response.

        Args:
            prompt: Prompt text
            max_tokens: Override max tokens for this call

        Returns:
            LLM response text
        """
        tokens = max_tokens or self.max_tokens

        # All OpenAI-compatible providers (OpenAI, vLLM, Ollama, generic)
        openai_compatible_providers = (
            LlmProvider.OPENAI,
            LlmProvider.VLLM,
            LlmProvider.OLLAMA,
            LlmProvider.OPENAI_COMPATIBLE,
        )

        if self.provider in openai_compatible_providers:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=tokens,
            )
            return response.choices[0].message.content or ""

        elif self.provider == LlmProvider.ANTHROPIC:
            response = self.client.messages.create(
                model=self.model_id,
                max_tokens=tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        elif self.provider == LlmProvider.MLX:
            import mlx_lm
            from mlx_lm.sample_utils import make_sampler

            model = self.client["model"]
            tokenizer = self.client["tokenizer"]

            # Apply chat template if available
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                formatted_prompt = prompt

            # Create sampler with temperature
            sampler = make_sampler(temp=self.temperature)

            # Generate response using MLX
            response = mlx_lm.generate(
                model=model,
                tokenizer=tokenizer,
                prompt=formatted_prompt,
                max_tokens=tokens,
                sampler=sampler,
            )

            # Post-process MLX response
            response = self._clean_mlx_response(response, formatted_prompt)
            return response

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _clean_mlx_response(self, response: str, formatted_prompt: str) -> str:
        """Clean MLX-generated response of common artifacts.

        Args:
            response: Raw MLX response text.
            formatted_prompt: The formatted prompt that was sent to the model.

        Returns:
            Cleaned response text.
        """
        if not response:
            return response

        cleaned = response.strip()

        # Remove common end-of-text tokens from various models
        end_tokens = [
            "<|endoftext|>",
            "<|im_end|>",
            "<|eot_id|>",
            "</s>",
            "<|end|>",
            "<|assistant|>",
            "[/INST]",
        ]
        for token in end_tokens:
            if cleaned.endswith(token):
                cleaned = cleaned[: -len(token)].strip()
            # Also handle case where token appears mid-response
            if token in cleaned:
                cleaned = cleaned.split(token)[0].strip()

        # Some models echo back the prompt - strip it if present
        if formatted_prompt and cleaned.startswith(formatted_prompt):
            cleaned = cleaned[len(formatted_prompt):].strip()

        # Remove common assistant prefixes some models add
        prefixes_to_strip = [
            "Assistant:",
            "assistant:",
            "ASSISTANT:",
            "Response:",
            "response:",
        ]
        for prefix in prefixes_to_strip:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()

        return cleaned


# =============================================================================
# File I/O Utilities
# =============================================================================

def save_to_file(
    objects: list[BaseModel],
    out_file: Path,
    append: bool = False,
) -> None:
    """Save Pydantic objects to JSONL file.

    Args:
        objects: List of Pydantic model instances
        out_file: Output file path
        append: Append to file instead of overwrite
    """
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    with open(out_file, mode) as f:
        for obj in objects:
            f.write(obj.model_dump_json() + "\n")

    _log.info(f"Saved {len(objects)} objects to {out_file}")


def save_one_to_file(obj: BaseModel, out_file: Path) -> None:
    """Append single Pydantic object to JSONL file.

    Args:
        obj: Pydantic model instance
        out_file: Output file path
    """
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "a") as f:
        f.write(obj.model_dump_json() + "\n")


def load_from_file(
    in_file: Path,
    model_class: type[T],
) -> Iterator[T]:
    """Load Pydantic objects from JSONL file.

    Args:
        in_file: Input file path
        model_class: Pydantic model class to deserialize to

    Yields:
        Deserialized model instances
    """
    in_file = Path(in_file)

    if not in_file.exists():
        _log.warning(f"File not found: {in_file}")
        return

    with open(in_file) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                yield model_class.model_validate(data)
            except Exception as e:
                _log.warning(f"Failed to parse line {line_num} in {in_file}: {e}")


def load_qa_chunks(in_file: Path) -> Iterator[CADQaChunk]:
    """Load CAD Q&A chunks from JSONL file.

    Args:
        in_file: Input file path

    Yields:
        CADQaChunk instances
    """
    return load_from_file(in_file, CADQaChunk)


def load_generated_qac(in_file: Path) -> Iterator[CADGenQAC]:
    """Load generated Q&A pairs from JSONL file.

    Args:
        in_file: Input file path

    Yields:
        CADGenQAC instances
    """
    return load_from_file(in_file, CADGenQAC)


# =============================================================================
# Text Post-processing
# =============================================================================

def postprocess_question(question: str) -> str | None:
    """Clean and validate generated question.

    Args:
        question: Raw generated question

    Returns:
        Cleaned question or None if invalid
    """
    if not question:
        return None

    # Strip whitespace
    question = question.strip()

    # Remove common prefixes
    prefixes_to_remove = [
        "Question:",
        "Q:",
        "Here is a question:",
        "Generated question:",
    ]
    for prefix in prefixes_to_remove:
        if question.lower().startswith(prefix.lower()):
            question = question[len(prefix):].strip()

    # Ensure question ends with ?
    if question and not question.endswith("?"):
        question = question + "?"

    # Validate minimum length
    if len(question) < 10:
        _log.debug(f"Question too short: {question}")
        return None

    # Check for actual question content
    if not re.search(r"\b(what|how|why|when|where|which|who|does|is|are|can)\b", question.lower()):
        _log.debug(f"Question doesn't contain question words: {question}")
        # Still return it, might be valid

    return question


def postprocess_answer(answer: str) -> str:
    """Clean generated answer.

    Args:
        answer: Raw generated answer

    Returns:
        Cleaned answer
    """
    if not answer:
        return ""

    # Strip whitespace
    answer = answer.strip()

    # Remove common prefixes
    prefixes_to_remove = [
        "Answer:",
        "A:",
        "Here is the answer:",
        "Generated answer:",
        "Based on the context,",
        "According to the CAD data,",
    ]
    for prefix in prefixes_to_remove:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()

    return answer


def parse_critique_response(response: str) -> Critique:
    """Parse LLM critique response into Critique object.

    Expected format:
        RATING: [1-5]
        EVALUATION: [text]
        SUGGESTIONS: [text or None]

    Args:
        response: Raw LLM response

    Returns:
        Critique object
    """
    rating = None
    evaluation = None
    suggestions = None

    lines = response.strip().split("\n")

    for line in lines:
        line = line.strip()

        if line.upper().startswith("RATING:"):
            try:
                rating_str = line.split(":", 1)[1].strip()
                # Extract first number found
                match = re.search(r"\d+", rating_str)
                if match:
                    rating = int(match.group())
                    rating = max(1, min(5, rating))  # Clamp to 1-5
            except (ValueError, IndexError):
                pass

        elif line.upper().startswith("EVALUATION:"):
            evaluation = line.split(":", 1)[1].strip() if ":" in line else None

        elif line.upper().startswith("SUGGESTIONS:"):
            suggestions = line.split(":", 1)[1].strip() if ":" in line else None
            if suggestions and suggestions.lower() in ("none", "n/a", "-"):
                suggestions = None

    return Critique(
        dimension="",  # Set by caller
        rating=rating,
        evaluation=evaluation,
        suggestions=suggestions,
    )


def count_tokens_simple(text: str) -> int:
    """Simple token count estimate (whitespace + punctuation split).

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    # Simple approximation: split on whitespace and punctuation
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return len(tokens)
