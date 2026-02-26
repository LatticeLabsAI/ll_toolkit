"""Core models for CAD synthetic data generation.

This module provides the foundational data models for the CAD SDG pipeline,
including status enums, configuration options, and result types.

Classes:
    Status: Enum for operation status
    LlmProvider: Enum for supported LLM providers
    LlmOptions: LLM configuration options
    CADSampleOptions: Sampling configuration
    CADGenerateOptions: Generation configuration
    CADCritiqueOptions: Critique configuration
    SampleResult: Sampling operation result
    GenerateResult: Generation operation result
    CritiqueResult: Critique operation result
    CADQaChunk: Sampled CAD passage for Q&A generation
    Critique: Single critique evaluation
    CADGenQAC: Generated question-answer-context triplet
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_serializer, field_validator


class Status(str, Enum):
    """Status of an SDG operation.

    Attributes:
        SUCCESS: Operation completed successfully
        FAILURE: Operation failed completely
        PARTIAL: Operation partially completed
    """

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


class LlmProvider(str, Enum):
    """Supported LLM providers.

    Attributes:
        OPENAI: OpenAI API (GPT models)
        ANTHROPIC: Anthropic API (Claude models)
        VLLM: vLLM server (OpenAI-compatible API)
        OLLAMA: Ollama local models (OpenAI-compatible API)
        OPENAI_COMPATIBLE: Any OpenAI-compatible API endpoint
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    VLLM = "vllm"
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"
    MLX = "mlx"


class QuestionType(str, Enum):
    """Types of questions for CAD Q&A generation.

    Attributes:
        FACT_SINGLE: Single fact extraction
        GEOMETRY: Geometric properties questions
        TOPOLOGY: Topology relationship questions
        MANUFACTURING: Manufacturing/DFM questions
        MATERIAL: Material property questions
        ASSEMBLY: Assembly structure questions
        DIMENSION: Dimensional measurement questions
        TOLERANCE: Tolerance specification questions
    """

    FACT_SINGLE = "fact_single"
    GEOMETRY = "geometry"
    TOPOLOGY = "topology"
    MANUFACTURING = "manufacturing"
    MATERIAL = "material"
    ASSEMBLY = "assembly"
    DIMENSION = "dimension"
    TOLERANCE = "tolerance"


class AnnotationLevel(str, Enum):
    """Level of detail for generated Q&A annotations.

    Controls how much technical detail appears in questions and answers:
    - ABSTRACT: High-level shape description, functional purpose
    - INTERMEDIATE: Specific features, relationships between components
    - DETAILED: Dimensions, material properties, tolerances
    - EXPERT: Parametric specifications, exact coordinates, construction sequence
    """

    ABSTRACT = "abstract"
    INTERMEDIATE = "intermediate"
    DETAILED = "detailed"
    EXPERT = "expert"


class ChunkerType(str, Enum):
    """CAD chunker types for sampling.

    Attributes:
        HYBRID: Hybrid chunker (combines multiple strategies)
        STEP: STEP-specific chunker
        STL: STL-specific chunker
        BREP: BRep-specific chunker
        TOPOLOGY: Topology-based chunker
    """

    HYBRID = "hybrid"
    STEP = "step"
    STL = "stl"
    BREP = "brep"
    TOPOLOGY = "topology"


class LlmOptions(BaseModel):
    """LLM configuration options.

    Attributes:
        provider: LLM provider to use (required)
        model_id: Model identifier (required, e.g., "gpt-4o", "claude-3-opus", "llama3")
        api_key: API key for the provider (required for OpenAI/Anthropic)
        url: Base URL for vLLM/Ollama/OpenAI-compatible endpoints
        temperature: Sampling temperature
        max_tokens: Maximum tokens for response
        timeout: Request timeout in seconds
    """

    provider: LlmProvider = Field(
        description="LLM provider (openai, anthropic, vllm, ollama, openai_compatible)"
    )
    model_id: str = Field(
        description="Model identifier (e.g., 'gpt-4o', 'claude-3-opus', 'llama3:70b')"
    )
    api_key: Optional[SecretStr] = None
    url: Optional[str] = Field(
        default=None,
        description="Base URL for vLLM/Ollama/OpenAI-compatible endpoints"
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=100000)
    timeout: int = Field(default=60, ge=1)

    model_config = ConfigDict(use_enum_values=True)


class CADSampleOptions(BaseModel):
    """Options for CAD passage sampling.

    Attributes:
        sample_file: Output file path for sampled passages
        chunker: Chunker type to use
        min_tokens: Minimum tokens per passage
        max_tokens: Maximum tokens per passage
        max_passages: Maximum number of passages to sample
        item_types: CAD item types to include
        seed: Random seed for reproducibility
        include_topology: Include topology graph in samples
    """

    sample_file: Path = Field(default=Path("samples.jsonl"))
    chunker: ChunkerType = ChunkerType.HYBRID
    min_tokens: int = Field(default=20, ge=1)
    max_tokens: int = Field(default=512, ge=1)
    max_passages: int = Field(default=50, ge=1)
    item_types: List[str] = Field(
        default_factory=lambda: ["step_entity", "mesh", "brep_face", "brep_edge"]
    )
    seed: int = Field(default=0)
    include_topology: bool = True

    model_config = ConfigDict(use_enum_values=True)


class CADGenerateOptions(LlmOptions):
    """Options for Q&A generation.

    Extends LlmOptions with generation-specific settings.

    Attributes:
        generated_file: Output file path for generated Q&A pairs
        max_qac: Maximum number of Q&A pairs to generate
        question_types: Types of questions to generate
        cad_specific_prompts: Use CAD-specific prompt templates
        include_context: Include full context in output
        batch_size: Number of generations per API call batch
    """

    generated_file: Path = Field(default=Path("generated.jsonl"))
    max_qac: int = Field(default=100, ge=1)
    question_types: List[QuestionType] = Field(
        default_factory=lambda: [
            QuestionType.FACT_SINGLE,
            QuestionType.GEOMETRY,
            QuestionType.TOPOLOGY,
            QuestionType.MANUFACTURING,
        ]
    )
    cad_specific_prompts: bool = True
    include_context: bool = True
    batch_size: int = Field(default=1, ge=1, le=100)
    annotation_levels: List[AnnotationLevel] = Field(
        default_factory=list,
        description="Annotation levels for controlling Q&A detail. Empty list means no level filtering."
    )


class CADCritiqueOptions(LlmOptions):
    """Options for Q&A critique.

    Extends LlmOptions with critique-specific settings.

    Attributes:
        critiqued_file: Output file path for critiqued Q&A pairs
        max_qac: Maximum number of Q&A pairs to critique
        critique_dimensions: Dimensions to evaluate
        min_rating_threshold: Minimum rating to pass (1-5)
        rewrite_low_quality: Rewrite Q&A pairs below threshold
    """

    critiqued_file: Path = Field(default=Path("critiqued.jsonl"))
    max_qac: int = Field(default=100, ge=1)
    critique_dimensions: List[str] = Field(
        default_factory=lambda: [
            "technical_accuracy",
            "cad_groundedness",
            "geometry_specificity",
            "manufacturing_relevance",
        ]
    )
    min_rating_threshold: int = Field(default=3, ge=1, le=5)
    rewrite_low_quality: bool = True


class CADConceptualOptions(LlmOptions):
    """Options for conceptual Q&A generation.

    For generating questions from abstract CAD content descriptions.

    Attributes:
        topics_file: Output file path for generated topics
        questions_file: Output file path for generated questions
        num_topics: Number of topics to generate
        questions_per_topic: Questions per topic
        use_retrieval: Use retrieval for answer generation
    """

    topics_file: Path = Field(default=Path("topics.jsonl"))
    questions_file: Path = Field(default=Path("conceptual_questions.jsonl"))
    num_topics: int = Field(default=10, ge=1)
    questions_per_topic: int = Field(default=5, ge=1)
    use_retrieval: bool = False


class SampleResult(BaseModel):
    """Result of a sampling operation.

    Attributes:
        status: Operation status
        time_taken: Time taken in seconds
        output: Output file path
        num_passages: Number of passages sampled
        num_sources: Number of source files processed
        errors: List of error messages
    """

    status: Status
    time_taken: float = 0.0
    output: Optional[Path] = None
    num_passages: int = 0
    num_sources: int = 0
    errors: List[str] = Field(default_factory=list)


class GenerateResult(BaseModel):
    """Result of a generation operation.

    Attributes:
        status: Operation status
        time_taken: Time taken in seconds
        output: Output file path
        num_qac: Number of Q&A pairs generated
        num_failed: Number of failed generations
        errors: List of error messages
    """

    status: Status
    time_taken: float = 0.0
    output: Optional[Path] = None
    num_qac: int = 0
    num_failed: int = 0
    errors: List[str] = Field(default_factory=list)


class CritiqueResult(BaseModel):
    """Result of a critique operation.

    Attributes:
        status: Operation status
        time_taken: Time taken in seconds
        output: Output file path
        num_qac: Number of Q&A pairs critiqued
        num_passed: Number that passed threshold
        num_rewritten: Number that were rewritten
        average_rating: Average critique rating
        errors: List of error messages
    """

    status: Status
    time_taken: float = 0.0
    output: Optional[Path] = None
    num_qac: int = 0
    num_passed: int = 0
    num_rewritten: int = 0
    average_rating: Optional[float] = None
    errors: List[str] = Field(default_factory=list)


class CADQaChunk(BaseModel):
    """Sampled CAD passage for Q&A generation.

    Represents a chunk of CAD data selected for Q&A pair generation.

    Attributes:
        text: Textual representation of the chunk
        chunk_id: Unique chunk identifier
        doc_id: Source document identifier
        doc_name: Source document name
        entity_types: List of CAD entity types in chunk
        entity_ids: List of entity identifiers
        properties: Additional chunk properties
        topology_subgraph: Optional topology graph data
        token_count: Approximate token count
    """

    text: str
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str = ""
    doc_name: str = ""
    entity_types: List[str] = Field(default_factory=list)
    entity_ids: List[str] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)
    topology_subgraph: Optional[Dict[str, Any]] = None
    token_count: int = 0

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """Validate that text is not empty."""
        if not v or not v.strip():
            raise ValueError("text cannot be empty")
        return v


class Critique(BaseModel):
    """Single critique evaluation.

    Represents the evaluation of a Q&A pair on a specific dimension.

    Attributes:
        dimension: Critique dimension (e.g., "technical_accuracy")
        evaluation: Textual evaluation/feedback
        rating: Numeric rating (1-5)
        suggestions: Improvement suggestions
    """

    dimension: str
    evaluation: Optional[str] = None
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    suggestions: Optional[str] = None

    # rating range validated by Field(ge=1, le=5) above


class CADGenQAC(BaseModel):
    """Generated question-answer-context triplet.

    Full Q&A pair with metadata, critiques, and provenance.

    Attributes:
        doc_id: Source document identifier
        doc_name: Source document name
        chunk_id: Source chunk identifier
        qac_id: Unique Q&A pair identifier
        question: Generated question
        answer: Generated answer
        context: Source context text
        question_type: Type of question
        labels: Classification labels
        critiques: Critique evaluations by dimension
        metadata: Additional metadata
        created: Creation timestamp
        model: Model used for generation
        is_improved: Whether answer was improved after critique
    """

    doc_id: str = ""
    doc_name: str = ""
    chunk_id: str = ""
    qac_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    answer: str
    context: str
    question_type: Optional[QuestionType] = None
    annotation_level: Optional[AnnotationLevel] = None
    labels: Dict[str, str] = Field(default_factory=dict)
    critiques: Dict[str, Critique] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created: datetime = Field(default_factory=datetime.now)
    model: str = ""
    is_improved: bool = False

    @field_validator("question", "answer")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        """Validate that question and answer are not empty."""
        if not v or not v.strip():
            raise ValueError("question and answer cannot be empty")
        return v

    def get_average_rating(self) -> Optional[float]:
        """Calculate average rating across all critiques.

        Returns:
            Average rating or None if no ratings
        """
        ratings = [c.rating for c in self.critiques.values() if c.rating is not None]
        if not ratings:
            return None
        return sum(ratings) / len(ratings)

    def passes_threshold(self, threshold: int = 3) -> bool:
        """Check if Q&A pair passes rating threshold.

        Args:
            threshold: Minimum average rating to pass

        Returns:
            True if average rating >= threshold
        """
        avg = self.get_average_rating()
        if avg is None:
            return True  # No critiques = passes by default
        return avg >= threshold

    model_config = ConfigDict(use_enum_values=True)

    @field_serializer("created")
    @classmethod
    def serialize_datetime(cls, v: datetime) -> str:
        """Serialize datetime to ISO format."""
        return v.isoformat()
