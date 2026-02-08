"""Annotation level prompt modifiers for SDG Q&A generation.

Provides level-specific constraints that wrap base prompts to control
the detail level of generated questions and answers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

_log = logging.getLogger(__name__)


@dataclass
class LevelConstraints:
    """Constraints for a specific annotation level."""
    level_name: str
    question_prefix: str      # Prepended to question generation prompt
    answer_prefix: str        # Prepended to answer generation prompt
    should_include: list[str] # Types of content that SHOULD appear
    should_exclude: list[str] # Types of content that SHOULD NOT appear


# Level constraint definitions
ABSTRACT_CONSTRAINTS = LevelConstraints(
    level_name="abstract",
    question_prefix=(
        "Generate a HIGH-LEVEL question about the overall shape, purpose, "
        "or general characteristics. Do NOT ask about specific dimensions, "
        "coordinates, or technical parameters."
    ),
    answer_prefix=(
        "Provide a HIGH-LEVEL answer describing general shape, function, "
        "and purpose. Do NOT include specific dimensions, coordinates, "
        "tolerance values, or parametric details. Use plain, descriptive language."
    ),
    should_include=["shape description", "functional purpose", "general form", "category"],
    should_exclude=["coordinates", "dimensions", "tolerances", "parametric values", "construction sequence"],
)

INTERMEDIATE_CONSTRAINTS = LevelConstraints(
    level_name="intermediate",
    question_prefix=(
        "Generate a question about specific FEATURES and RELATIONSHIPS "
        "between components. Include feature names and types but avoid "
        "exact numerical specifications."
    ),
    answer_prefix=(
        "Describe specific features, their types, and relationships "
        "between components. Reference feature names (e.g., fillet, chamfer, "
        "pocket, boss) but avoid exact coordinates. Include approximate "
        "proportions and relative positions."
    ),
    should_include=["feature names", "component relationships", "feature types", "relative positions"],
    should_exclude=["exact coordinates", "construction sequence", "parametric equations"],
)

DETAILED_CONSTRAINTS = LevelConstraints(
    level_name="detailed",
    question_prefix=(
        "Generate a DETAILED question about dimensions, material properties, "
        "or tolerances. Ask about specific measurable quantities."
    ),
    answer_prefix=(
        "Provide DETAILED answers with specific dimensions, material "
        "properties, and tolerance values where available. Include units "
        "and numerical precision appropriate to the data."
    ),
    should_include=["dimensions", "material properties", "tolerances", "units", "numerical values"],
    should_exclude=["construction sequence", "parametric history"],
)

EXPERT_CONSTRAINTS = LevelConstraints(
    level_name="expert",
    question_prefix=(
        "Generate an EXPERT-LEVEL question about parametric specifications, "
        "exact coordinates, construction sequence, or geometric constraints. "
        "Assume the reader has deep CAD expertise."
    ),
    answer_prefix=(
        "Provide an EXPERT-LEVEL answer with exact coordinates, parametric "
        "specifications, construction sequence details, and geometric "
        "constraint definitions. Include all available numerical precision "
        "and technical terminology."
    ),
    should_include=[
        "exact coordinates", "parametric specifications", "construction sequence",
        "geometric constraints", "B-Rep details", "surface definitions",
    ],
    should_exclude=[],
)

_LEVEL_MAP = {
    "abstract": ABSTRACT_CONSTRAINTS,
    "intermediate": INTERMEDIATE_CONSTRAINTS,
    "detailed": DETAILED_CONSTRAINTS,
    "expert": EXPERT_CONSTRAINTS,
}


class AnnotationLevelPromptModifier:
    """Modifies base prompts with annotation level constraints.

    Wraps existing Q&A generation prompts with level-specific instructions
    that control what detail should and should not appear.
    """

    def __init__(self, level: str):
        """Initialize with annotation level name.

        Args:
            level: One of 'abstract', 'intermediate', 'detailed', 'expert'
        """
        self.level = level
        self.constraints = _LEVEL_MAP.get(level)
        if self.constraints is None:
            raise ValueError(
                f"Unknown annotation level: {level}. "
                f"Valid levels: {list(_LEVEL_MAP.keys())}"
            )

    def modify_question_prompt(self, base_prompt: str) -> str:
        """Wrap a base question generation prompt with level constraints.

        Args:
            base_prompt: The original question generation prompt

        Returns:
            Modified prompt with level-specific instructions prepended
        """
        include_str = ", ".join(self.constraints.should_include)
        exclude_str = ", ".join(self.constraints.should_exclude) if self.constraints.should_exclude else "none"

        return (
            f"[Annotation Level: {self.level.upper()}]\n"
            f"{self.constraints.question_prefix}\n"
            f"SHOULD include: {include_str}\n"
            f"SHOULD NOT include: {exclude_str}\n\n"
            f"{base_prompt}"
        )

    def modify_answer_prompt(self, base_prompt: str) -> str:
        """Wrap a base answer generation prompt with level constraints.

        Args:
            base_prompt: The original answer generation prompt

        Returns:
            Modified prompt with level-specific instructions prepended
        """
        include_str = ", ".join(self.constraints.should_include)
        exclude_str = ", ".join(self.constraints.should_exclude) if self.constraints.should_exclude else "none"

        return (
            f"[Annotation Level: {self.level.upper()}]\n"
            f"{self.constraints.answer_prefix}\n"
            f"MUST include: {include_str}\n"
            f"MUST NOT include: {exclude_str}\n\n"
            f"{base_prompt}"
        )


class AnnotationLevelCritiqueTemplate:
    """Critique template for evaluating level consistency.

    Checks whether generated Q&A pairs match their declared annotation level.
    """

    TEMPLATE = (
        "Evaluate whether the following Q&A pair is consistent with its "
        "declared annotation level: {level}.\n\n"
        "Question: {question}\n"
        "Answer: {answer}\n\n"
        "Level expectations:\n"
        "- SHOULD contain: {should_include}\n"
        "- SHOULD NOT contain: {should_exclude}\n\n"
        "Score from 1-5 where:\n"
        "1 = Completely wrong level (e.g., coordinates in ABSTRACT)\n"
        "2 = Mostly wrong level with some correct elements\n"
        "3 = Mixed - some elements match, some don't\n"
        "4 = Mostly correct level with minor deviations\n"
        "5 = Perfect level match\n\n"
        "Provide your score and a brief justification."
    )

    @classmethod
    def format(cls, level: str, question: str, answer: str) -> str:
        """Format the critique template with specific Q&A content."""
        constraints = _LEVEL_MAP.get(level)
        if constraints is None:
            return ""

        return cls.TEMPLATE.format(
            level=level.upper(),
            question=question,
            answer=answer,
            should_include=", ".join(constraints.should_include),
            should_exclude=", ".join(constraints.should_exclude) if constraints.should_exclude else "none",
        )


def get_modifier_for_level(level: str) -> AnnotationLevelPromptModifier:
    """Get the prompt modifier for a given annotation level.

    Args:
        level: Annotation level name

    Returns:
        AnnotationLevelPromptModifier configured for the level
    """
    return AnnotationLevelPromptModifier(level)
