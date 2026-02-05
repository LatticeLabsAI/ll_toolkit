"""CAD Q&A generation module.

This module provides tools for synthetic Q&A generation from CAD documents:

Classes:
    - CADPassageSampler: Sample passages from CAD files
    - CADGenerator: Generate Q&A pairs from sampled passages
    - CADJudge: Critique and improve Q&A pairs
    - CADConceptualGenerator: Generate topic-based Q&A from descriptions

Data Models:
    - Status: Operation status enum
    - LlmProvider: Supported LLM providers
    - LlmOptions: LLM configuration options
    - CADSampleOptions: Sampling options
    - CADGenerateOptions: Generation options
    - CADCritiqueOptions: Critique options
    - CADConceptualOptions: Conceptual generation options
    - SampleResult: Sampling operation result
    - GenerateResult: Generation operation result
    - CritiqueResult: Critique operation result
    - CADQaChunk: Sampled CAD passage
    - CADGenQAC: Generated Q&A pair with context
    - Critique: Single critique evaluation
    - QuestionType: Types of CAD questions
    - ChunkerType: CAD chunker types

Example:
    from cadling.sdg.qa import (
        CADPassageSampler,
        CADGenerator,
        CADJudge,
        CADSampleOptions,
        CADGenerateOptions,
        CADCritiqueOptions,
        LlmProvider,
    )

    # Sample passages
    sample_opts = CADSampleOptions(sample_file=Path("samples.jsonl"))
    sampler = CADPassageSampler(sample_opts)
    sampler.sample([Path("part.step")])

    # Generate Q&A pairs
    gen_opts = CADGenerateOptions(
        provider=LlmProvider.OPENAI,
        model_id="gpt-4o",
        generated_file=Path("generated.jsonl"),
    )
    generator = CADGenerator(gen_opts)
    generator.generate(Path("samples.jsonl"))

    # Critique Q&A pairs
    crit_opts = CADCritiqueOptions(
        provider=LlmProvider.OPENAI,
        model_id="gpt-4o",
        critiqued_file=Path("critiqued.jsonl"),
    )
    judge = CADJudge(crit_opts)
    judge.critique(Path("generated.jsonl"))
"""

from cadling.sdg.qa.base import (
    CADConceptualOptions,
    CADCritiqueOptions,
    CADGenerateOptions,
    CADGenQAC,
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
from cadling.sdg.qa.conceptual_generate import CADConceptualGenerator
from cadling.sdg.qa.critique import CADJudge
from cadling.sdg.qa.generate import CADGenerator
from cadling.sdg.qa.sample import CADPassageSampler

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
