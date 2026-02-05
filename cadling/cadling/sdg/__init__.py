"""Synthetic data generation for CAD documents.

This package provides tools for generating synthetic training data
from CAD documents, including Q&A pairs for language model training.

Components:
    - CADPassageSampler: Sample passages from CAD files
    - CADGenerator: Generate Q&A pairs from passages
    - CADJudge: Critique and improve Q&A quality
    - CADConceptualGenerator: Generate topic-based Q&A from descriptions

Supported LLM Providers:
    - OpenAI (api.openai.com)
    - Anthropic (api.anthropic.com)
    - vLLM (localhost:8000/v1 or custom URL)
    - Ollama (localhost:11434/v1 or custom URL)
    - OpenAI-compatible (any custom URL)

Example:
    from cadling.sdg import (
        CADPassageSampler,
        CADGenerator,
        CADJudge,
        CADSampleOptions,
        CADGenerateOptions,
        CADCritiqueOptions,
        LlmProvider,
    )
    from pathlib import Path

    # Step 1: Sample passages from CAD files
    sample_opts = CADSampleOptions(sample_file=Path("samples.jsonl"))
    sampler = CADPassageSampler(sample_opts)
    sampler.sample([Path("part.step")])

    # Step 2: Generate Q&A pairs
    gen_opts = CADGenerateOptions(
        provider=LlmProvider.OPENAI,
        model_id="gpt-4o",
        generated_file=Path("generated.jsonl"),
    )
    generator = CADGenerator(gen_opts)
    generator.generate(Path("samples.jsonl"))

    # Step 3: Critique and improve Q&A pairs
    crit_opts = CADCritiqueOptions(
        provider=LlmProvider.OPENAI,
        model_id="gpt-4o",
        critiqued_file=Path("critiqued.jsonl"),
    )
    judge = CADJudge(crit_opts)
    judge.critique(Path("generated.jsonl"))
"""

from cadling.sdg.qa import (
    CADConceptualGenerator,
    CADConceptualOptions,
    CADCritiqueOptions,
    CADGenerateOptions,
    CADGenerator,
    CADGenQAC,
    CADJudge,
    CADPassageSampler,
    CADQaChunk,
    CADSampleOptions,
    ChunkerType,
    Critique,
    CritiqueResult,
    GenerateResult,
    LlmOptions,
    LlmProvider,
    QuestionType,
    SampleResult,
    Status,
)

__all__ = [
    # Core classes
    "CADPassageSampler",
    "CADGenerator",
    "CADJudge",
    "CADConceptualGenerator",
    # Enums
    "Status",
    "LlmProvider",
    "QuestionType",
    "ChunkerType",
    # Options
    "LlmOptions",
    "CADSampleOptions",
    "CADGenerateOptions",
    "CADCritiqueOptions",
    "CADConceptualOptions",
    # Results
    "SampleResult",
    "GenerateResult",
    "CritiqueResult",
    # Data models
    "CADQaChunk",
    "CADGenQAC",
    "Critique",
]
