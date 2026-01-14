"""Synthetic data generation for CAD Q&A pairs.

This module generates question-answer pairs from CAD documents for training
language models. It uses LLMs to create questions and answers based on
CAD chunks, with optional critique and improvement.

Classes:
    CADQAGenerator: Generate Q&A pairs from CAD documents
    QAPair: Single question-answer pair with metadata
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from cadling.chunker.base_chunker import CADChunk
    from cadling.datamodel.base_models import CADlingDocument

_log = logging.getLogger(__name__)

# Try to import OpenAI
_OPENAI_AVAILABLE = False
try:
    import openai

    _OPENAI_AVAILABLE = True
except ImportError:
    _log.debug("openai not available for QA generation")

# Try to import Anthropic
_ANTHROPIC_AVAILABLE = False
try:
    import anthropic

    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _log.debug("anthropic not available for QA generation")


class QAPair(BaseModel):
    """Single question-answer pair.

    Attributes:
        question: Generated question
        answer: Generated answer
        context: CAD chunk text context
        metadata: Additional metadata from chunk
        critique: Optional critique of the Q&A pair
    """

    question: str
    answer: str
    context: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    critique: Optional[str] = None


class CADQAGenerator:
    """Generate Q&A pairs from CAD documents.

    This generator creates synthetic training data for CAD understanding models.
    It uses LLMs to generate questions and answers based on CAD chunks.

    The process follows the docling-sdg pattern:
    1. Sample: Select CAD chunks for generation
    2. Generate: Create Q&A pairs from chunks
    3. Critique: Improve quality through self-critique

    Attributes:
        llm_model: LLM model name (gpt-4, claude-3-opus, etc.)
        critique_enabled: Whether to use critique step
        client: LLM API client
        provider: LLM provider (openai, anthropic)

    Example:
        from cadling.backend.document_converter import DocumentConverter
        from cadling.sdg.qa_generator import CADQAGenerator

        # Convert CAD file
        converter = DocumentConverter()
        result = converter.convert("part.step")

        # Generate Q&A pairs
        qa_gen = CADQAGenerator(llm_model="gpt-4")
        qa_pairs = qa_gen.generate_qa_pairs(result.document, num_pairs=100)

        # Save as JSONL
        with open("qa_pairs.jsonl", "w") as f:
            for qa in qa_pairs:
                f.write(qa.model_dump_json() + "\\n")
    """

    def __init__(
        self,
        llm_model: str = "gpt-4",
        critique_enabled: bool = True,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        """Initialize QA generator.

        Args:
            llm_model: LLM model name
            critique_enabled: Whether to use critique step
            api_key: Optional API key (uses environment variable if not provided)
            provider: Optional provider override (auto-detected from model name)
        """
        self.llm_model = llm_model
        self.critique_enabled = critique_enabled
        self.api_key = api_key

        # Auto-detect provider from model name
        if provider:
            self.provider = provider
        elif "gpt" in llm_model.lower() or "o1" in llm_model.lower():
            self.provider = "openai"
        elif "claude" in llm_model.lower():
            self.provider = "anthropic"
        else:
            raise ValueError(
                f"Unknown model provider for {llm_model}. "
                "Please specify provider='openai' or provider='anthropic'"
            )

        # Initialize client
        self.client = self._initialize_llm()

        _log.info(
            f"Initialized CADQAGenerator (model={llm_model}, "
            f"provider={self.provider}, critique={critique_enabled})"
        )

    def _initialize_llm(self) -> Any:
        """Initialize LLM client.

        Returns:
            LLM client instance

        Raises:
            ImportError: If required library not installed
        """
        if self.provider == "openai":
            if not _OPENAI_AVAILABLE:
                raise ImportError("openai package required. Install with: pip install openai")

            client = openai.OpenAI(api_key=self.api_key)
            _log.info(f"Initialized OpenAI client for {self.llm_model}")
            return client

        elif self.provider == "anthropic":
            if not _ANTHROPIC_AVAILABLE:
                raise ImportError(
                    "anthropic package required. Install with: pip install anthropic"
                )

            client = anthropic.Anthropic(api_key=self.api_key)
            _log.info(f"Initialized Anthropic client for {self.llm_model}")
            return client

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate_qa_pairs(
        self,
        doc: CADlingDocument,
        num_pairs: int = 100,
        chunker: Optional[Any] = None,
    ) -> List[QAPair]:
        """Generate Q&A pairs from CAD document.

        Args:
            doc: CADlingDocument to generate from
            num_pairs: Number of Q&A pairs to generate
            chunker: Optional custom chunker (uses CADHybridChunker by default)

        Returns:
            List of QAPair objects
        """
        _log.info(f"Generating {num_pairs} Q&A pairs from document '{doc.name}'")

        # Get chunker
        if chunker is None:
            from cadling.chunker.hybrid_chunker import CADHybridChunker

            chunker = CADHybridChunker(max_tokens=512)

        # Generate chunks
        chunks = list(chunker.chunk(doc))
        _log.info(f"Generated {len(chunks)} chunks from document")

        # Sample chunks for Q&A generation
        import random

        num_chunks = min(num_pairs, len(chunks))
        sampled_chunks = random.sample(chunks, num_chunks) if len(chunks) > num_chunks else chunks

        _log.info(f"Sampled {len(sampled_chunks)} chunks for Q&A generation")

        # Generate Q&A pairs
        qa_pairs = []

        for i, chunk in enumerate(sampled_chunks):
            try:
                _log.debug(f"Generating Q&A {i+1}/{len(sampled_chunks)}")

                # Generate question
                question = self._generate_question(chunk)

                # Generate answer
                answer = self._generate_answer(chunk, question)

                # Critique (optional)
                critique_text = None
                if self.critique_enabled:
                    question, answer, critique_text = self._critique_qa(
                        chunk, question, answer
                    )

                # Create Q&A pair
                qa_pair = QAPair(
                    question=question,
                    answer=answer,
                    context=chunk.text,
                    metadata={
                        "chunk_id": chunk.chunk_id,
                        "doc_name": chunk.doc_name,
                        "entity_types": chunk.meta.entity_types,
                        "entity_ids": chunk.meta.entity_ids,
                    },
                    critique=critique_text,
                )

                qa_pairs.append(qa_pair)

                _log.debug(f"Generated Q&A pair {i+1}: Q='{question[:50]}...'")

            except Exception as e:
                _log.error(f"Failed to generate Q&A for chunk {i}: {e}")

        _log.info(f"Successfully generated {len(qa_pairs)} Q&A pairs")

        return qa_pairs

    def _generate_question(self, chunk: CADChunk) -> str:
        """Generate question from CAD chunk.

        Args:
            chunk: CAD chunk

        Returns:
            Generated question
        """
        prompt = f"""Given this CAD entity information from a STEP file:

{chunk.text}

Generate a specific, technical question about:
- Geometric properties (dimensions, coordinates, shapes)
- Entity relationships or topology
- Part characteristics or features
- Manufacturing or design intent

The question should be answerable from the given context.

Generate only the question, no additional text."""

        response = self._call_llm(prompt)

        return response.strip()

    def _generate_answer(self, chunk: CADChunk, question: str) -> str:
        """Generate answer from CAD chunk and question.

        Args:
            chunk: CAD chunk
            question: Generated question

        Returns:
            Generated answer
        """
        prompt = f"""Given this CAD entity information from a STEP file:

{chunk.text}

Question: {question}

Provide a clear, technical answer based on the information above.
Be specific and reference actual values from the context.

Generate only the answer, no additional text."""

        response = self._call_llm(prompt)

        return response.strip()

    def _critique_qa(
        self, chunk: CADChunk, question: str, answer: str
    ) -> Tuple[str, str, str]:
        """Critique and improve Q&A pair.

        Args:
            chunk: CAD chunk
            question: Generated question
            answer: Generated answer

        Returns:
            Tuple of (improved_question, improved_answer, critique)
        """
        critique_prompt = f"""You are an expert in CAD and STEP files. Review this Q&A pair for accuracy and quality.

Context:
{chunk.text}

Question: {question}

Answer: {answer}

Evaluate:
1. Is the question clear and specific?
2. Is the answer accurate based on the context?
3. Are there any errors or improvements needed?

Provide:
1. Critique: Brief assessment (2-3 sentences)
2. Improved Question: (if needed, otherwise same)
3. Improved Answer: (if needed, otherwise same)

Format as:
CRITIQUE: <critique>
QUESTION: <improved question>
ANSWER: <improved answer>"""

        response = self._call_llm(critique_prompt)

        # Parse response
        critique = ""
        improved_question = question
        improved_answer = answer

        try:
            lines = response.split("\n")
            current_section = None
            current_text = []

            for line in lines:
                if line.startswith("CRITIQUE:"):
                    current_section = "critique"
                    current_text = [line.replace("CRITIQUE:", "").strip()]
                elif line.startswith("QUESTION:"):
                    if current_section == "critique":
                        critique = " ".join(current_text)
                    current_section = "question"
                    current_text = [line.replace("QUESTION:", "").strip()]
                elif line.startswith("ANSWER:"):
                    if current_section == "question":
                        improved_question = " ".join(current_text)
                    current_section = "answer"
                    current_text = [line.replace("ANSWER:", "").strip()]
                else:
                    if current_section:
                        current_text.append(line.strip())

            # Get last section
            if current_section == "answer":
                improved_answer = " ".join(current_text)

        except Exception as e:
            _log.error(f"Failed to parse critique: {e}")
            critique = response

        return improved_question, improved_answer, critique

    def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt.

        Args:
            prompt: Prompt text

        Returns:
            LLM response text
        """
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.llm_model,
                max_tokens=2048,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def save_qa_pairs(self, qa_pairs: List[QAPair], output_path: str) -> None:
        """Save Q&A pairs to JSONL file.

        Args:
            qa_pairs: List of QAPair objects
            output_path: Output file path
        """
        import json

        with open(output_path, "w") as f:
            for qa in qa_pairs:
                f.write(qa.model_dump_json() + "\n")

        _log.info(f"Saved {len(qa_pairs)} Q&A pairs to {output_path}")
