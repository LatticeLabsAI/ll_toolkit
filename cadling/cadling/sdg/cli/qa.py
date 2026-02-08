"""QA subcommands for CAD SDG CLI.

This module provides CLI commands for Q&A generation:
- sample: Sample passages from CAD files
- generate: Generate Q&A pairs from passages
- critique: Critique and improve Q&A pairs
- conceptual: Generate conceptual Q&A from descriptions
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import typer
from pydantic import SecretStr

from cadling.sdg.qa.base import (
    AnnotationLevel,
    CADCritiqueOptions,
    CADGenerateOptions,
    CADSampleOptions,
    ChunkerType,
    LlmProvider,
    QuestionType,
)

_log = logging.getLogger(__name__)

# QA CLI application
qa_app = typer.Typer(help="Q&A generation commands")


def _get_api_key_from_env(provider: str) -> str | None:
    """Get API key from environment variable.

    Args:
        provider: Provider name

    Returns:
        API key or None
    """
    env_vars = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    env_var = env_vars.get(provider)
    if env_var:
        return os.environ.get(env_var)
    return None


@qa_app.command("sample")
def sample(
    sources: list[Path] = typer.Argument(..., help="CAD files to sample from"),
    output: Path = typer.Option(
        Path("samples.jsonl"),
        "--output", "-o",
        help="Output file for sampled passages",
    ),
    chunker: str = typer.Option(
        "hybrid",
        "--chunker", "-c",
        help="Chunker type (hybrid, step, stl, brep, topology)",
    ),
    max_passages: int = typer.Option(
        50,
        "--max-passages", "-n",
        help="Maximum number of passages to sample",
    ),
    min_tokens: int = typer.Option(
        20,
        "--min-tokens",
        help="Minimum tokens per passage",
    ),
    max_tokens: int = typer.Option(
        512,
        "--max-tokens",
        help="Maximum tokens per passage",
    ),
    seed: int = typer.Option(
        0,
        "--seed",
        help="Random seed for reproducibility",
    ),
):
    """Sample passages from CAD files for Q&A generation."""
    from cadling.sdg.qa.sample import CADPassageSampler

    typer.echo(f"Sampling from {len(sources)} CAD files...")

    try:
        chunker_type = ChunkerType(chunker)
    except ValueError:
        typer.echo(f"Invalid chunker type: {chunker}", err=True)
        raise typer.Exit(1)

    options = CADSampleOptions(
        sample_file=output,
        chunker=chunker_type,
        max_passages=max_passages,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        seed=seed,
    )

    sampler = CADPassageSampler(options)
    result = sampler.sample(sources)

    typer.echo(f"Status: {result.status.value}")
    typer.echo(f"Passages sampled: {result.num_passages}")
    typer.echo(f"Output: {result.output}")
    typer.echo(f"Time: {result.time_taken:.2f}s")

    if result.errors:
        typer.echo(f"Errors: {len(result.errors)}", err=True)
        for error in result.errors[:5]:
            typer.echo(f"  - {error}", err=True)


@qa_app.command("generate")
def generate(
    source: Path = typer.Argument(..., help="Sampled passages file (JSONL)"),
    output: Path = typer.Option(
        Path("generated.jsonl"),
        "--output", "-o",
        help="Output file for generated Q&A pairs",
    ),
    provider: str = typer.Option(
        ...,
        "--provider", "-p",
        help="LLM provider (openai, anthropic, vllm, ollama, openai_compatible)",
    ),
    model: str = typer.Option(
        ...,
        "--model", "-m",
        help="Model ID (e.g., gpt-4o, claude-3-opus, llama3:70b)",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help="API key (or set via environment variable)",
    ),
    url: str | None = typer.Option(
        None,
        "--url",
        help="Base URL for vLLM/Ollama/OpenAI-compatible",
    ),
    max_qac: int = typer.Option(
        100,
        "--max-qac", "-n",
        help="Maximum Q&A pairs to generate",
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature", "-t",
        help="Sampling temperature",
    ),
    question_types: list[str] | None = typer.Option(
        None,
        "--question-type", "-q",
        help="Question types to generate (geometry, topology, manufacturing, etc.)",
    ),
    annotation_levels: list[str] | None = typer.Option(
        None,
        "--annotation-level", "-a",
        help="Annotation levels to generate (abstract, intermediate, detailed, expert). Can specify multiple.",
    ),
):
    """Generate Q&A pairs from sampled passages."""
    from cadling.sdg.qa.generate import CADGenerator

    typer.echo(f"Generating Q&A pairs from {source}...")

    # Resolve provider
    try:
        llm_provider = LlmProvider(provider)
    except ValueError:
        typer.echo(f"Invalid provider: {provider}", err=True)
        raise typer.Exit(1)

    # Resolve API key
    resolved_api_key = api_key or _get_api_key_from_env(provider)
    if llm_provider in (LlmProvider.OPENAI, LlmProvider.ANTHROPIC) and not resolved_api_key:
        typer.echo(f"API key required for {provider}. Set --api-key or environment variable.", err=True)
        raise typer.Exit(1)

    # Resolve question types
    q_types = []
    if question_types:
        for qt in question_types:
            try:
                q_types.append(QuestionType(qt))
            except ValueError:
                typer.echo(f"Invalid question type: {qt}", err=True)
                raise typer.Exit(1)
    else:
        q_types = [QuestionType.FACT_SINGLE, QuestionType.GEOMETRY, QuestionType.TOPOLOGY]

    # Resolve annotation levels
    a_levels: list[AnnotationLevel] = []
    if annotation_levels:
        for al in annotation_levels:
            try:
                a_levels.append(AnnotationLevel(al))
            except ValueError:
                typer.echo(f"Invalid annotation level: {al}. Valid: abstract, intermediate, detailed, expert", err=True)
                raise typer.Exit(1)

    options = CADGenerateOptions(
        provider=llm_provider,
        model_id=model,
        api_key=SecretStr(resolved_api_key) if resolved_api_key else None,
        url=url,
        generated_file=output,
        max_qac=max_qac,
        temperature=temperature,
        question_types=q_types,
        annotation_levels=a_levels,
    )

    generator = CADGenerator(options)
    result = generator.generate(source)

    typer.echo(f"Status: {result.status.value}")
    typer.echo(f"Q&A pairs generated: {result.num_qac}")
    typer.echo(f"Failed: {result.num_failed}")
    typer.echo(f"Output: {result.output}")
    typer.echo(f"Time: {result.time_taken:.2f}s")


@qa_app.command("critique")
def critique(
    source: Path = typer.Argument(..., help="Generated Q&A pairs file (JSONL)"),
    output: Path = typer.Option(
        Path("critiqued.jsonl"),
        "--output", "-o",
        help="Output file for critiqued Q&A pairs",
    ),
    provider: str = typer.Option(
        ...,
        "--provider", "-p",
        help="LLM provider (openai, anthropic, vllm, ollama, openai_compatible)",
    ),
    model: str = typer.Option(
        ...,
        "--model", "-m",
        help="Model ID (e.g., gpt-4o, claude-3-opus)",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help="API key (or set via environment variable)",
    ),
    url: str | None = typer.Option(
        None,
        "--url",
        help="Base URL for vLLM/Ollama/OpenAI-compatible",
    ),
    max_qac: int = typer.Option(
        100,
        "--max-qac", "-n",
        help="Maximum Q&A pairs to critique",
    ),
    dimensions: list[str] | None = typer.Option(
        None,
        "--dimension", "-d",
        help="Critique dimensions (technical_accuracy, cad_groundedness, etc.)",
    ),
    threshold: int = typer.Option(
        3,
        "--threshold",
        help="Minimum rating threshold (1-5)",
    ),
    rewrite: bool = typer.Option(
        True,
        "--rewrite/--no-rewrite",
        help="Rewrite low-quality Q&A pairs",
    ),
):
    """Critique and optionally improve Q&A pairs."""
    from cadling.sdg.qa.critique import CADJudge

    typer.echo(f"Critiquing Q&A pairs from {source}...")

    # Resolve provider
    try:
        llm_provider = LlmProvider(provider)
    except ValueError:
        typer.echo(f"Invalid provider: {provider}", err=True)
        raise typer.Exit(1)

    # Resolve API key
    resolved_api_key = api_key or _get_api_key_from_env(provider)
    if llm_provider in (LlmProvider.OPENAI, LlmProvider.ANTHROPIC) and not resolved_api_key:
        typer.echo(f"API key required for {provider}.", err=True)
        raise typer.Exit(1)

    # Resolve dimensions
    critique_dims = dimensions or [
        "technical_accuracy",
        "cad_groundedness",
        "geometry_specificity",
    ]

    options = CADCritiqueOptions(
        provider=llm_provider,
        model_id=model,
        api_key=SecretStr(resolved_api_key) if resolved_api_key else None,
        url=url,
        critiqued_file=output,
        max_qac=max_qac,
        critique_dimensions=critique_dims,
        min_rating_threshold=threshold,
        rewrite_low_quality=rewrite,
    )

    judge = CADJudge(options)
    result = judge.critique(source)

    typer.echo(f"Status: {result.status.value}")
    typer.echo(f"Q&A pairs critiqued: {result.num_qac}")
    typer.echo(f"Passed threshold: {result.num_passed}")
    typer.echo(f"Rewritten: {result.num_rewritten}")
    if result.average_rating:
        typer.echo(f"Average rating: {result.average_rating:.2f}")
    typer.echo(f"Output: {result.output}")
    typer.echo(f"Time: {result.time_taken:.2f}s")


@qa_app.command("conceptual")
def conceptual(
    description: str = typer.Argument(..., help="CAD content description"),
    output: Path = typer.Option(
        Path("conceptual_qa.jsonl"),
        "--output", "-o",
        help="Output file for generated Q&A pairs",
    ),
    provider: str = typer.Option(
        ...,
        "--provider", "-p",
        help="LLM provider",
    ),
    model: str = typer.Option(
        ...,
        "--model", "-m",
        help="Model ID",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help="API key",
    ),
    url: str | None = typer.Option(
        None,
        "--url",
        help="Base URL",
    ),
    num_topics: int = typer.Option(
        10,
        "--num-topics",
        help="Number of topics to generate",
    ),
    questions_per_topic: int = typer.Option(
        5,
        "--questions-per-topic",
        help="Questions per topic",
    ),
):
    """Generate conceptual Q&A from a CAD description."""
    from cadling.sdg.qa.base import CADConceptualOptions
    from cadling.sdg.qa.conceptual_generate import CADConceptualGenerator

    typer.echo(f"Generating conceptual Q&A for: {description[:50]}...")

    # Resolve provider
    try:
        llm_provider = LlmProvider(provider)
    except ValueError:
        typer.echo(f"Invalid provider: {provider}", err=True)
        raise typer.Exit(1)

    # Resolve API key
    resolved_api_key = api_key or _get_api_key_from_env(provider)

    options = CADConceptualOptions(
        provider=llm_provider,
        model_id=model,
        api_key=SecretStr(resolved_api_key) if resolved_api_key else None,
        url=url,
        questions_file=output,
        num_topics=num_topics,
        questions_per_topic=questions_per_topic,
    )

    generator = CADConceptualGenerator(options)
    result = generator.generate_from_description(description)

    typer.echo(f"Status: {result.status.value}")
    typer.echo(f"Q&A pairs generated: {result.num_qac}")
    typer.echo(f"Output: {result.output}")
    typer.echo(f"Time: {result.time_taken:.2f}s")
