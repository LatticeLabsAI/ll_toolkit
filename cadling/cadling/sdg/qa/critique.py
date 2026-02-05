"""CAD Q&A pair critique and quality assessment.

This module provides the CADJudge class for critiquing and
improving generated Q&A pairs.

Classes:
    CADJudge: Critique Q&A pairs for quality
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from cadling.sdg.qa.base import (
    CADCritiqueOptions,
    CADGenQAC,
    Critique,
    CritiqueResult,
    Status,
)
from cadling.sdg.qa.prompts import (
    CADCritiquePromptTemplate,
    get_critique_for_dimension,
    get_critique_prompts,
)
from cadling.sdg.qa.utils import (
    ChatAgent,
    load_generated_qac,
    parse_critique_response,
    postprocess_answer,
    save_one_to_file,
)

_log = logging.getLogger(__name__)


class CADJudge:
    """Critique Q&A pairs for quality assessment.

    Evaluates generated Q&A pairs across multiple dimensions
    using LLM-based critique with CAD-specific prompts.

    Attributes:
        options: Critique configuration options
        agent: LLM chat agent
        prompts: Active critique prompts

    Example:
        from cadling.sdg.qa.critique import CADJudge
        from cadling.sdg.qa.base import CADCritiqueOptions

        options = CADCritiqueOptions(
            provider="openai",
            model_id="gpt-4",
            critiqued_file=Path("critiqued.jsonl"),
            critique_dimensions=["technical_accuracy", "cad_groundedness"],
        )

        judge = CADJudge(options)
        result = judge.critique(Path("generated.jsonl"))
    """

    def __init__(self, options: CADCritiqueOptions):
        """Initialize Q&A judge.

        Args:
            options: Critique configuration options
        """
        self.options = options
        self.agent = ChatAgent.from_options(options)
        self.prompts = self._init_prompts()

        _log.info(
            f"Initialized CADJudge (model={options.model_id}, "
            f"dimensions={options.critique_dimensions})"
        )

    def _init_prompts(self) -> dict[str, CADCritiquePromptTemplate]:
        """Initialize critique prompt templates.

        Returns:
            Dictionary of dimension to prompt template
        """
        all_prompts = get_critique_prompts()

        # Filter by selected dimensions
        selected = {}
        for dim in self.options.critique_dimensions:
            if dim in all_prompts:
                selected[dim] = all_prompts[dim]
            else:
                _log.warning(f"Unknown critique dimension: {dim}")

        if not selected:
            _log.warning("No valid dimensions, using all")
            return all_prompts

        return selected

    def critique(self, source: Path) -> CritiqueResult:
        """Critique Q&A pairs from file.

        Args:
            source: Path to JSONL file with generated Q&A pairs

        Returns:
            CritiqueResult with statistics
        """
        start_time = time.time()
        errors: list[str] = []
        num_critiqued = 0
        num_passed = 0
        num_rewritten = 0
        total_ratings: list[float] = []

        _log.info(f"Critiquing Q&A pairs from {source}")

        # Load Q&A pairs
        qac_list = list(load_generated_qac(source))
        _log.info(f"Loaded {len(qac_list)} Q&A pairs")

        # Limit to max_qac
        if len(qac_list) > self.options.max_qac:
            qac_list = qac_list[: self.options.max_qac]

        # Critique each Q&A pair
        for i, qac in enumerate(qac_list):
            try:
                _log.debug(f"Critiquing Q&A {i + 1}/{len(qac_list)}")

                critiqued_qac = self._critique_qac(qac)

                # Check if passes threshold
                avg_rating = critiqued_qac.get_average_rating()
                if avg_rating:
                    total_ratings.append(avg_rating)

                if critiqued_qac.passes_threshold(self.options.min_rating_threshold):
                    num_passed += 1
                elif self.options.rewrite_low_quality:
                    # Attempt to rewrite
                    improved = self._rewrite_qac(critiqued_qac)
                    if improved:
                        critiqued_qac = improved
                        critiqued_qac.is_improved = True
                        num_rewritten += 1

                save_one_to_file(critiqued_qac, self.options.critiqued_file)
                num_critiqued += 1

            except Exception as e:
                error_msg = f"Failed to critique Q&A {i}: {e}"
                _log.error(error_msg)
                errors.append(error_msg)

        elapsed = time.time() - start_time
        avg_rating = sum(total_ratings) / len(total_ratings) if total_ratings else None

        # Determine status
        if num_critiqued == 0:
            status = Status.FAILURE
        elif errors:
            status = Status.PARTIAL
        else:
            status = Status.SUCCESS

        result = CritiqueResult(
            status=status,
            time_taken=elapsed,
            output=self.options.critiqued_file,
            num_qac=num_critiqued,
            num_passed=num_passed,
            num_rewritten=num_rewritten,
            average_rating=avg_rating,
            errors=errors,
        )

        _log.info(
            f"Critique complete: {num_critiqued} Q&A pairs, "
            f"{num_passed} passed, {num_rewritten} rewritten, "
            f"avg rating: {avg_rating:.2f if avg_rating else 'N/A'}"
        )

        return result

    def _critique_qac(self, qac: CADGenQAC) -> CADGenQAC:
        """Critique a single Q&A pair.

        Args:
            qac: Q&A pair to critique

        Returns:
            Q&A pair with critiques added
        """
        for dimension, prompt_template in self.prompts.items():
            critique = self._run_critique(qac, prompt_template)
            critique.dimension = dimension
            qac.critiques[dimension] = critique

        return qac

    def _run_critique(
        self,
        qac: CADGenQAC,
        prompt_template: CADCritiquePromptTemplate,
    ) -> Critique:
        """Run a single critique on a Q&A pair.

        Args:
            qac: Q&A pair to critique
            prompt_template: Critique prompt template

        Returns:
            Critique result
        """
        prompt = prompt_template.format_prompt(
            context=qac.context,
            question=qac.question,
            answer=qac.answer,
        )

        response = self.agent.ask(prompt, max_tokens=512)
        critique = parse_critique_response(response)

        return critique

    def _rewrite_qac(self, qac: CADGenQAC) -> CADGenQAC | None:
        """Attempt to rewrite a low-quality Q&A pair.

        Args:
            qac: Q&A pair to rewrite

        Returns:
            Improved Q&A pair or None if rewrite failed
        """
        # Gather critique feedback
        feedback_parts = []
        for dim, critique in qac.critiques.items():
            if critique.suggestions:
                feedback_parts.append(f"- {dim}: {critique.suggestions}")

        if not feedback_parts:
            return None

        feedback = "\n".join(feedback_parts)

        # Rewrite prompt
        rewrite_prompt = f"""You are a CAD expert. A Q&A pair was critiqued and needs improvement.

Context:
{qac.context}

Original Question: {qac.question}

Original Answer: {qac.answer}

Critique Feedback:
{feedback}

Based on the feedback, provide an improved answer that addresses the issues.
Generate ONLY the improved answer, nothing else."""

        try:
            improved_answer = self.agent.ask(rewrite_prompt, max_tokens=self.options.max_tokens)
            improved_answer = postprocess_answer(improved_answer)

            if improved_answer and improved_answer != qac.answer:
                qac.answer = improved_answer
                qac.metadata["original_answer"] = qac.answer
                return qac

        except Exception as e:
            _log.warning(f"Failed to rewrite Q&A: {e}")

        return None

    def critique_single(
        self,
        context: str,
        question: str,
        answer: str,
        dimensions: list[str] | None = None,
    ) -> dict[str, Critique]:
        """Critique a single Q&A pair without file I/O.

        Args:
            context: CAD context text
            question: Question
            answer: Answer
            dimensions: Optional list of dimensions to critique

        Returns:
            Dictionary of dimension to Critique
        """
        critiques: dict[str, Critique] = {}

        dims = dimensions or list(self.prompts.keys())

        for dim in dims:
            prompt_template = get_critique_for_dimension(dim)
            if not prompt_template:
                continue

            prompt = prompt_template.format_prompt(
                context=context,
                question=question,
                answer=answer,
            )

            response = self.agent.ask(prompt, max_tokens=512)
            critique = parse_critique_response(response)
            critique.dimension = dim
            critiques[dim] = critique

        return critiques
