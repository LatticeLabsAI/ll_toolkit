"""CAD Q&A pair generator.

This module provides the CADGenerator class for generating
question-answer pairs from CAD document passages.

Classes:
    CADGenerator: Generate Q&A pairs from CAD passages
"""

from __future__ import annotations

import itertools
import logging
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
from collections.abc import Iterator

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADlingDocument

from cadling.sdg.qa.base import (
    AnnotationLevel,
    CADGenQAC,
    CADGenerateOptions,
    CADQaChunk,
    GenerateResult,
    QuestionType,
    Status,
)
from cadling.sdg.qa.prompts import (
    CADQaPromptTemplate,
    get_generation_prompts,
    get_prompts_for_question_type,
)
from cadling.sdg.qa.prompts.generation_prompts import get_prompts_for_level
from cadling.sdg.qa.utils import (
    ChatAgent,
    count_tokens_simple,
    load_qa_chunks,
    postprocess_answer,
    postprocess_question,
    save_one_to_file,
)

_log = logging.getLogger(__name__)


class CADGenerator:
    """Generate Q&A pairs from CAD passages.

    Creates question-answer pairs from CAD document chunks using
    LLM-based generation with CAD-specific prompts.

    Attributes:
        options: Generation configuration options
        agent: LLM chat agent
        prompts: Active generation prompts

    Example:
        from cadling.sdg.qa.generate import CADGenerator
        from cadling.sdg.qa.base import CADGenerateOptions

        options = CADGenerateOptions(
            provider="openai",
            model_id="gpt-4",
            generated_file=Path("generated.jsonl"),
            max_qac=100,
        )

        generator = CADGenerator(options)
        result = generator.generate(Path("samples.jsonl"))
    """

    def __init__(self, options: CADGenerateOptions, seed: Optional[int] = None):
        """Initialize Q&A generator.

        Args:
            options: Generation configuration options
            seed: Random seed for reproducibility
        """
        self.options = options
        self.agent = ChatAgent.from_options(options)
        self.base_prompts = self._get_base_prompts()
        self.prompts = self._init_prompts()
        self.rng = random.Random(seed)  # Dedicated RNG for reproducibility

        _log.info(
            f"Initialized CADGenerator (model={options.model_id}, "
            f"max_qac={options.max_qac}, "
            f"annotation_levels={[l.value for l in options.annotation_levels] if options.annotation_levels else 'none'})"
        )

    def _get_base_prompts(self) -> list[CADQaPromptTemplate]:
        """Get base prompt templates before annotation level expansion.

        Returns:
            List of base prompt templates filtered by question types
        """
        if not self.options.cad_specific_prompts:
            return list(get_generation_prompts().values())

        prompts = []
        for q_type in self.options.question_types:
            prompts.extend(get_prompts_for_question_type(q_type))

        if not prompts:
            _log.warning("No prompts found for selected types, using all")
            return list(get_generation_prompts().values())

        return prompts

    def _init_prompts(self) -> list[CADQaPromptTemplate]:
        """Initialize prompt templates based on options.

        If annotation levels are specified, creates level-specific prompt
        variants for each base prompt. Otherwise returns base prompts as-is.

        Returns:
            List of active prompt templates
        """
        if not self.options.annotation_levels:
            return self.base_prompts

        # Create level-specific prompt variants for each annotation level
        level_prompts: list[CADQaPromptTemplate] = []
        for level in self.options.annotation_levels:
            level_variants = get_prompts_for_level(self.base_prompts, level.value)
            level_prompts.extend(level_variants)
            _log.info(
                f"Created {len(level_variants)} prompt variants for "
                f"annotation level '{level.value}'"
            )

        if not level_prompts:
            _log.warning("No level-specific prompts generated, falling back to base prompts")
            return self.base_prompts

        return level_prompts

    def generate(self, source: Path) -> GenerateResult:
        """Generate Q&A pairs from sampled passages file.

        Args:
            source: Path to JSONL file with sampled passages

        Returns:
            GenerateResult with statistics and output path
        """
        start_time = time.time()
        errors: list[str] = []
        num_generated = 0
        num_failed = 0

        _log.info(f"Generating Q&A pairs from {source}")

        # Stream passages up to max_qac (avoid materializing entire JSONL)
        passages = itertools.islice(load_qa_chunks(source), self.options.max_qac)

        # Generate Q&A for each passage
        for i, passage in enumerate(passages):
            try:
                _log.debug(f"Generating Q&A {i + 1}/{self.options.max_qac}")

                qac = self._generate_for_passage(passage)
                if qac:
                    save_one_to_file(qac, self.options.generated_file)
                    num_generated += 1
                else:
                    num_failed += 1

            except Exception as e:
                error_msg = f"Failed to generate Q&A for passage {i}: {e}"
                _log.error(error_msg)
                errors.append(error_msg)
                num_failed += 1

        elapsed = time.time() - start_time

        # Determine status
        if num_generated == 0:
            status = Status.FAILURE
        elif num_failed > 0:
            status = Status.PARTIAL
        else:
            status = Status.SUCCESS

        result = GenerateResult(
            status=status,
            time_taken=elapsed,
            output=self.options.generated_file,
            num_qac=num_generated,
            num_failed=num_failed,
            errors=errors,
        )

        _log.info(
            f"Generation complete: {num_generated} Q&A pairs, "
            f"{num_failed} failed, {elapsed:.2f}s"
        )

        return result

    def generate_from_chunks(
        self,
        chunks: Iterator[CADQaChunk],
    ) -> GenerateResult:
        """Generate Q&A pairs from chunk iterator.

        Args:
            chunks: Iterator of CADQaChunk objects

        Returns:
            GenerateResult with statistics
        """
        start_time = time.time()
        errors: list[str] = []
        num_generated = 0
        num_failed = 0

        for i, chunk in enumerate(chunks):
            if num_generated >= self.options.max_qac:
                break

            try:
                qac = self._generate_for_passage(chunk)
                if qac:
                    save_one_to_file(qac, self.options.generated_file)
                    num_generated += 1
                else:
                    num_failed += 1

            except Exception as e:
                error_msg = f"Failed to generate Q&A for chunk {i}: {e}"
                _log.error(error_msg)
                errors.append(error_msg)
                num_failed += 1

        elapsed = time.time() - start_time

        if num_generated == 0:
            status = Status.FAILURE
        elif num_failed > 0:
            status = Status.PARTIAL
        else:
            status = Status.SUCCESS

        return GenerateResult(
            status=status,
            time_taken=elapsed,
            output=self.options.generated_file,
            num_qac=num_generated,
            num_failed=num_failed,
            errors=errors,
        )

    def _generate_for_passage(
        self,
        passage: CADQaChunk,
    ) -> CADGenQAC | None:
        """Generate Q&A pair for a single passage.

        Args:
            passage: CAD passage chunk

        Returns:
            Generated CADGenQAC or None if failed
        """
        # Select random prompt template
        prompt_template = self.rng.choice(self.prompts)

        # Generate question
        question_prompt = prompt_template.format_question_prompt(
            context=passage.text,
        )
        raw_question = self.agent.ask(question_prompt, max_tokens=256)
        question = postprocess_question(raw_question)

        if not question:
            _log.debug(f"Invalid question generated: {raw_question[:100]}")
            return None

        # Generate answer
        answer_prompt = prompt_template.format_answer_prompt(
            context=passage.text,
            question=question,
        )
        raw_answer = self.agent.ask(answer_prompt, max_tokens=self.options.max_tokens)
        answer = postprocess_answer(raw_answer)

        if not answer:
            _log.debug("Empty answer generated")
            return None

        # Resolve annotation level from prompt template
        annotation_level = None
        if prompt_template.annotation_level:
            try:
                annotation_level = AnnotationLevel(prompt_template.annotation_level)
            except ValueError:
                _log.warning(
                    f"Invalid annotation level '{prompt_template.annotation_level}' "
                    f"on prompt template '{prompt_template.name}'"
                )

        # Create Q&A pair
        qac = CADGenQAC(
            doc_id=passage.doc_id,
            doc_name=passage.doc_name,
            chunk_id=passage.chunk_id,
            question=question,
            answer=answer,
            context=passage.text if self.options.include_context else "",
            question_type=prompt_template.question_type,
            annotation_level=annotation_level,
            labels={
                "prompt_name": prompt_template.name,
            },
            metadata={
                "entity_types": passage.entity_types,
                "entity_ids": passage.entity_ids,
                "properties": passage.properties,
            },
            model=self.options.model_id,
        )

        return qac

    def generate_question(
        self,
        passage: CADQaChunk,
        question_type: QuestionType | None = None,
    ) -> str | None:
        """Generate a question for a passage.

        Args:
            passage: CAD passage chunk
            question_type: Type of question to generate

        Returns:
            Generated question or None
        """
        # Select prompt template
        if question_type:
            prompts = get_prompts_for_question_type(question_type)
            if prompts:
                prompt_template = self.rng.choice(prompts)
            else:
                prompt_template = self.rng.choice(self.prompts)
        else:
            prompt_template = self.rng.choice(self.prompts)

        question_prompt = prompt_template.format_question_prompt(
            context=passage.text,
        )
        raw_question = self.agent.ask(question_prompt, max_tokens=256)

        return postprocess_question(raw_question)

    def generate_answer(
        self,
        passage: CADQaChunk,
        question: str,
        question_type: QuestionType | None = None,
    ) -> str:
        """Generate an answer for a question.

        Args:
            passage: CAD passage chunk
            question: Question to answer
            question_type: Type of question to match answer prompt style

        Returns:
            Generated answer
        """
        # Select prompt template matching the question type
        if question_type:
            prompts = get_prompts_for_question_type(question_type)
            if prompts:
                prompt_template = self.rng.choice(prompts)
            else:
                prompt_template = self.rng.choice(self.prompts)
        else:
            prompt_template = self.rng.choice(self.prompts)

        answer_prompt = prompt_template.format_answer_prompt(
            context=passage.text,
            question=question,
        )
        raw_answer = self.agent.ask(answer_prompt, max_tokens=self.options.max_tokens)

        return postprocess_answer(raw_answer)

    def generate_from_document(
        self,
        doc: CADlingDocument,
        num_pairs: int | None = None,
        chunker: Any | None = None,
    ) -> GenerateResult:
        """Generate Q&A pairs directly from a CADlingDocument.

        This is a convenience method that combines chunking and generation
        in a single call, similar to the legacy CADQAGenerator API.

        Args:
            doc: CADlingDocument to generate from
            num_pairs: Number of Q&A pairs to generate (defaults to max_qac)
            chunker: Optional custom chunker (uses CADHybridChunker by default)

        Returns:
            GenerateResult with statistics

        Example:
            from cadling.backend.document_converter import DocumentConverter
            from cadling.sdg.qa import CADGenerator, CADGenerateOptions, LlmProvider

            # Convert CAD file
            converter = DocumentConverter()
            result = converter.convert("part.step")

            # Generate Q&A pairs directly from document
            options = CADGenerateOptions(
                provider=LlmProvider.OPENAI,
                model_id="gpt-4o",
                generated_file=Path("qa_pairs.jsonl"),
            )
            generator = CADGenerator(options)
            result = generator.generate_from_document(result.document, num_pairs=50)
        """
        _log.info(f"Generating Q&A pairs from document '{doc.name}'")

        # Get chunker
        if chunker is None:
            from cadling.chunker.hybrid_chunker import CADHybridChunker

            chunker = CADHybridChunker(max_tokens=512)

        # Generate chunks
        chunks = list(chunker.chunk(doc))
        _log.info(f"Generated {len(chunks)} chunks from document")

        # Sample chunks for Q&A generation
        max_pairs = num_pairs or self.options.max_qac
        num_to_sample = min(max_pairs, len(chunks))

        if len(chunks) > num_to_sample:
            sampled_chunks = self.rng.sample(chunks, num_to_sample)
        else:
            sampled_chunks = chunks

        _log.info(f"Sampled {len(sampled_chunks)} chunks for Q&A generation")

        # Convert CADChunks to CADQaChunks
        qa_chunks = []
        for chunk in sampled_chunks:
            qa_chunk = CADQaChunk(
                text=chunk.text,
                chunk_id=chunk.chunk_id,
                doc_id=getattr(chunk, "doc_id", ""),
                doc_name=chunk.doc_name,
                entity_types=chunk.meta.entity_types if chunk.meta else [],
                entity_ids=[str(eid) for eid in chunk.meta.entity_ids] if chunk.meta else [],
                properties=chunk.meta.properties if chunk.meta else {},
                token_count=count_tokens_simple(chunk.text),
            )
            qa_chunks.append(qa_chunk)

        # Generate Q&A pairs
        return self.generate_from_chunks(iter(qa_chunks))
