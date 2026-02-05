"""Conceptual Q&A generation for CAD documents.

This module provides the CADConceptualGenerator class for generating
questions from abstract CAD content descriptions rather than from
specific document chunks.

Classes:
    CADConceptualGenerator: Generate topic-based Q&A pairs
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator

from cadling.sdg.qa.base import (
    CADConceptualOptions,
    CADGenQAC,
    GenerateResult,
    QuestionType,
    Status,
)
from cadling.sdg.qa.utils import (
    ChatAgent,
    postprocess_answer,
    postprocess_question,
    save_one_to_file,
    save_to_file,
)

_log = logging.getLogger(__name__)


# CAD-specific topic categories for conceptual generation
CAD_TOPIC_CATEGORIES = [
    "geometric_primitives",
    "topology_relationships",
    "machining_features",
    "tolerance_specifications",
    "material_properties",
    "assembly_structure",
    "surface_finishes",
    "dimensional_analysis",
    "manufacturing_processes",
    "design_intent",
]


class CADConceptualGenerator:
    """Generate Q&A pairs from abstract CAD content descriptions.

    Unlike CADGenerator which generates from specific document chunks,
    this generator creates questions based on topic descriptions and
    CAD domain knowledge.

    Attributes:
        options: Conceptual generation options
        agent: LLM chat agent

    Example:
        from cadling.sdg.qa.conceptual_generate import CADConceptualGenerator
        from cadling.sdg.qa.base import CADConceptualOptions

        options = CADConceptualOptions(
            provider="openai",
            model_id="gpt-4o",
            num_topics=10,
            questions_per_topic=5,
        )

        generator = CADConceptualGenerator(options)

        # Generate from a description
        result = generator.generate_from_description(
            "A mechanical bracket with mounting holes and reinforcement ribs"
        )
    """

    def __init__(self, options: CADConceptualOptions):
        """Initialize conceptual generator.

        Args:
            options: Conceptual generation options
        """
        self.options = options
        self.agent = ChatAgent.from_options(options)

        _log.info(
            f"Initialized CADConceptualGenerator (model={options.model_id}, "
            f"topics={options.num_topics}, questions_per_topic={options.questions_per_topic})"
        )

    def generate_from_description(
        self,
        description: str,
        context: str | None = None,
    ) -> GenerateResult:
        """Generate Q&A pairs from a CAD content description.

        Args:
            description: High-level description of CAD content
            context: Optional additional context

        Returns:
            GenerateResult with statistics
        """
        start_time = time.time()
        errors: list[str] = []
        num_generated = 0
        num_failed = 0

        _log.info(f"Generating conceptual Q&A from description: {description[:100]}...")

        # Step 1: Generate topics from description
        topics = self._generate_topics(description)
        _log.info(f"Generated {len(topics)} topics")

        # Save topics if configured
        if self.options.topics_file:
            self._save_topics(topics, self.options.topics_file)

        # Step 2: Generate questions for each topic
        for topic in topics:
            try:
                questions = self._generate_questions_for_topic(topic, description)

                for question in questions:
                    # Step 3: Generate answer
                    answer = self._generate_answer(question, topic, description, context)

                    if answer:
                        qac = CADGenQAC(
                            question=question,
                            answer=answer,
                            context=context or description,
                            question_type=self._infer_question_type(topic),
                            labels={"topic": topic, "generation_type": "conceptual"},
                            model=self.options.model_id,
                        )
                        save_one_to_file(qac, self.options.questions_file)
                        num_generated += 1
                    else:
                        num_failed += 1

            except Exception as e:
                error_msg = f"Failed to generate for topic '{topic}': {e}"
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

        result = GenerateResult(
            status=status,
            time_taken=elapsed,
            output=self.options.questions_file,
            num_qac=num_generated,
            num_failed=num_failed,
            errors=errors,
        )

        _log.info(
            f"Conceptual generation complete: {num_generated} Q&A pairs, "
            f"{num_failed} failed, {elapsed:.2f}s"
        )

        return result

    def _generate_topics(self, description: str) -> list[str]:
        """Generate CAD-relevant topics from description.

        Args:
            description: CAD content description

        Returns:
            List of topic strings
        """
        prompt = f"""You are a CAD/mechanical engineering expert. Given this description of a CAD model:

"{description}"

Generate {self.options.num_topics} specific, technical topics that would be relevant for understanding this CAD model.
Focus on topics related to:
- Geometric features and dimensions
- Topology and surface relationships
- Manufacturing considerations
- Material and mechanical properties
- Assembly and mating features

Output ONLY a numbered list of topics, one per line. Be specific and technical.

Example format:
1. Mounting hole pattern and spacing
2. Fillet radii on stress concentration points
3. Wall thickness for injection molding"""

        response = self.agent.ask(prompt, max_tokens=512)

        # Parse topics from response
        topics = []
        for line in response.strip().split("\n"):
            line = line.strip()
            # Remove numbering
            if line and line[0].isdigit():
                # Remove "1.", "1)", etc.
                parts = line.split(".", 1) if "." in line[:3] else line.split(")", 1)
                if len(parts) > 1:
                    topic = parts[1].strip()
                else:
                    topic = line
            else:
                topic = line

            if topic and len(topic) > 3:
                topics.append(topic)

        return topics[: self.options.num_topics]

    def _generate_questions_for_topic(
        self,
        topic: str,
        description: str,
    ) -> list[str]:
        """Generate questions for a specific topic.

        Args:
            topic: Topic to generate questions about
            description: Original CAD description

        Returns:
            List of question strings
        """
        prompt = f"""You are a CAD/mechanical engineering expert. Given this CAD model description:

"{description}"

And this specific topic:
"{topic}"

Generate {self.options.questions_per_topic} specific, technical questions about this topic.
Questions should:
- Be answerable by someone with access to the CAD model
- Focus on quantitative or factual information
- Be relevant to design, manufacturing, or analysis

Output ONLY the questions, one per line, each ending with a question mark."""

        response = self.agent.ask(prompt, max_tokens=512)

        # Parse questions
        questions = []
        for line in response.strip().split("\n"):
            line = line.strip()
            # Remove numbering
            if line and line[0].isdigit():
                parts = line.split(".", 1) if "." in line[:3] else line.split(")", 1)
                if len(parts) > 1:
                    question = parts[1].strip()
                else:
                    question = line
            else:
                question = line

            question = postprocess_question(question)
            if question:
                questions.append(question)

        return questions[: self.options.questions_per_topic]

    def _generate_answer(
        self,
        question: str,
        topic: str,
        description: str,
        context: str | None = None,
    ) -> str | None:
        """Generate answer for a conceptual question.

        Args:
            question: Question to answer
            topic: Topic context
            description: CAD description
            context: Optional additional context

        Returns:
            Answer string or None
        """
        context_text = context or description

        prompt = f"""You are a CAD/mechanical engineering expert. Given this context about a CAD model:

Context: {context_text}

Topic: {topic}

Question: {question}

Provide a technical, accurate answer. If specific values are not available,
provide reasonable technical guidance or indicate what information would be needed.
Keep the answer concise but complete.

Answer:"""

        response = self.agent.ask(prompt, max_tokens=self.options.max_tokens)
        answer = postprocess_answer(response)

        return answer if answer else None

    def _infer_question_type(self, topic: str) -> QuestionType:
        """Infer question type from topic.

        Args:
            topic: Topic string

        Returns:
            Appropriate QuestionType
        """
        topic_lower = topic.lower()

        if any(kw in topic_lower for kw in ["dimension", "length", "width", "height", "radius", "diameter"]):
            return QuestionType.DIMENSION
        elif any(kw in topic_lower for kw in ["tolerance", "gd&t", "fit", "clearance"]):
            return QuestionType.TOLERANCE
        elif any(kw in topic_lower for kw in ["geometry", "shape", "curve", "surface"]):
            return QuestionType.GEOMETRY
        elif any(kw in topic_lower for kw in ["topology", "face", "edge", "vertex", "adjacent"]):
            return QuestionType.TOPOLOGY
        elif any(kw in topic_lower for kw in ["machining", "manufacturing", "process", "tooling"]):
            return QuestionType.MANUFACTURING
        elif any(kw in topic_lower for kw in ["material", "steel", "aluminum", "density"]):
            return QuestionType.MATERIAL
        elif any(kw in topic_lower for kw in ["assembly", "component", "mate", "constraint"]):
            return QuestionType.ASSEMBLY
        else:
            return QuestionType.FACT_SINGLE

    def _save_topics(self, topics: list[str], output_file: Path) -> None:
        """Save generated topics to file.

        Args:
            topics: List of topic strings
            output_file: Output file path
        """
        from pydantic import BaseModel

        class Topic(BaseModel):
            topic: str
            index: int

        topic_objects = [Topic(topic=t, index=i) for i, t in enumerate(topics)]
        save_to_file(topic_objects, output_file)
        _log.info(f"Saved {len(topics)} topics to {output_file}")

    def generate_questions_only(
        self,
        description: str,
    ) -> Iterator[str]:
        """Generate questions without answers (for later retrieval-based answering).

        Args:
            description: CAD content description

        Yields:
            Question strings
        """
        topics = self._generate_topics(description)

        for topic in topics:
            questions = self._generate_questions_for_topic(topic, description)
            yield from questions

    def generate_from_topics(
        self,
        topics: list[str],
        description: str,
        context: str | None = None,
    ) -> GenerateResult:
        """Generate Q&A pairs from pre-defined topics.

        Args:
            topics: List of topics to generate for
            description: CAD content description
            context: Optional additional context

        Returns:
            GenerateResult with statistics
        """
        start_time = time.time()
        errors: list[str] = []
        num_generated = 0
        num_failed = 0

        for topic in topics:
            try:
                questions = self._generate_questions_for_topic(topic, description)

                for question in questions:
                    answer = self._generate_answer(question, topic, description, context)

                    if answer:
                        qac = CADGenQAC(
                            question=question,
                            answer=answer,
                            context=context or description,
                            question_type=self._infer_question_type(topic),
                            labels={"topic": topic, "generation_type": "conceptual"},
                            model=self.options.model_id,
                        )
                        save_one_to_file(qac, self.options.questions_file)
                        num_generated += 1
                    else:
                        num_failed += 1

            except Exception as e:
                error_msg = f"Failed for topic '{topic}': {e}"
                _log.error(error_msg)
                errors.append(error_msg)

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
            output=self.options.questions_file,
            num_qac=num_generated,
            num_failed=num_failed,
            errors=errors,
        )
