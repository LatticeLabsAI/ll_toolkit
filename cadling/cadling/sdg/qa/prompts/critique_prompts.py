"""CAD-specific prompt templates for Q&A critique.

This module provides prompt templates for evaluating and critiquing
Q&A pairs generated from CAD documents.

Classes:
    CADCritiquePromptTemplate: Template for Q&A critique

Constants:
    TECHNICAL_ACCURACY_CRITIQUE: Critique for technical accuracy
    CAD_GROUNDEDNESS_CRITIQUE: Critique for CAD groundedness
    GEOMETRY_SPECIFICITY_CRITIQUE: Critique for geometric precision
    MANUFACTURING_RELEVANCE_CRITIQUE: Critique for manufacturing relevance
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CADCritiquePromptTemplate(BaseModel):
    """Template for critiquing CAD Q&A pairs.

    Defines the prompts used for evaluating the quality of
    generated Q&A pairs.

    Attributes:
        name: Template name (same as critique dimension)
        dimension: Critique dimension being evaluated
        prompt: Prompt template for critique
        description: Description of what this critique evaluates
        rating_scale: Description of the 1-5 rating scale
        examples: Optional example critiques
    """

    name: str
    dimension: str
    prompt: str
    description: str = ""
    rating_scale: str = ""
    examples: list[dict[str, Any]] = Field(default_factory=list)

    def format_prompt(
        self,
        context: str,
        question: str,
        answer: str,
        **kwargs,
    ) -> str:
        """Format the critique prompt with Q&A data.

        Args:
            context: Original CAD context
            question: Generated question
            answer: Generated answer
            **kwargs: Additional format arguments

        Returns:
            Formatted prompt string
        """
        return self.prompt.format(
            context=context,
            question=question,
            answer=answer,
            **kwargs,
        )


# =============================================================================
# Rating Scale Description
# =============================================================================

STANDARD_RATING_SCALE = """
Rating Scale (1-5):
1 - Completely incorrect or irrelevant
2 - Major errors or mostly incorrect
3 - Partially correct with significant issues
4 - Mostly correct with minor issues
5 - Fully correct and well-formed
"""


# =============================================================================
# CAD-Specific Critique Prompts
# =============================================================================

TECHNICAL_ACCURACY_CRITIQUE = CADCritiquePromptTemplate(
    name="technical_accuracy",
    dimension="technical_accuracy",
    description="Evaluate if the answer is technically accurate and correct",
    rating_scale=STANDARD_RATING_SCALE,
    prompt="""You are an expert CAD engineer reviewing Q&A pairs for technical accuracy.

Context (CAD Data):
{context}

Question: {question}

Answer: {answer}

Evaluate the TECHNICAL ACCURACY of this Q&A pair:
- Is the answer factually correct based on the context?
- Are technical terms used correctly?
- Are numeric values accurate (if applicable)?
- Are units correct and consistent?

Provide your evaluation in this exact format:
RATING: [1-5]
EVALUATION: [2-3 sentences explaining your rating]
SUGGESTIONS: [Optional improvements if rating < 5]
""",
    examples=[
        {
            "rating": 5,
            "evaluation": "The answer correctly identifies the cylindrical surface radius as 12.5mm, matching the value in the STEP entity data. Technical terminology is used appropriately.",
            "suggestions": None,
        },
        {
            "rating": 2,
            "evaluation": "The answer states the diameter is 25mm, but the context shows a radius of 12.5mm, making the diameter 25mm. However, the question asked about radius, not diameter.",
            "suggestions": "The answer should directly address the radius (12.5mm) rather than converting to diameter.",
        },
    ],
)

CAD_GROUNDEDNESS_CRITIQUE = CADCritiquePromptTemplate(
    name="cad_groundedness",
    dimension="cad_groundedness",
    description="Evaluate if the answer is grounded in the provided CAD context",
    rating_scale=STANDARD_RATING_SCALE,
    prompt="""You are an expert CAD engineer reviewing Q&A pairs for groundedness.

Context (CAD Data):
{context}

Question: {question}

Answer: {answer}

Evaluate the GROUNDEDNESS of this answer:
- Is every claim in the answer supported by the context?
- Does the answer avoid making up information not in the context?
- Are all referenced entities, values, and relationships present in the context?
- Does the answer stay within the scope of the provided data?

Provide your evaluation in this exact format:
RATING: [1-5]
EVALUATION: [2-3 sentences explaining your rating]
SUGGESTIONS: [Optional improvements if rating < 5]
""",
    examples=[
        {
            "rating": 5,
            "evaluation": "All information in the answer directly references entities and values found in the context. No external knowledge or fabricated details are introduced.",
            "suggestions": None,
        },
        {
            "rating": 3,
            "evaluation": "The answer correctly references the face count from the context but adds a claim about material type that is not present in the provided CAD data.",
            "suggestions": "Remove the material specification claim as it is not supported by the context.",
        },
    ],
)

GEOMETRY_SPECIFICITY_CRITIQUE = CADCritiquePromptTemplate(
    name="geometry_specificity",
    dimension="geometry_specificity",
    description="Evaluate if geometric values are precise and specific",
    rating_scale=STANDARD_RATING_SCALE,
    prompt="""You are an expert CAD engineer reviewing Q&A pairs for geometric precision.

Context (CAD Data):
{context}

Question: {question}

Answer: {answer}

Evaluate the GEOMETRIC SPECIFICITY of this answer:
- Does the answer include specific numeric values when available?
- Are coordinate values precise (not rounded excessively)?
- Are dimensional units included and correct?
- Does the answer avoid vague terms like "approximately" when exact values are available?

Provide your evaluation in this exact format:
RATING: [1-5]
EVALUATION: [2-3 sentences explaining your rating]
SUGGESTIONS: [Optional improvements if rating < 5]
""",
    examples=[
        {
            "rating": 5,
            "evaluation": "The answer provides the exact position (10.5, 20.0, 0.0) and radius (5.25 mm) as found in the context. All values include appropriate precision and units.",
            "suggestions": None,
        },
        {
            "rating": 2,
            "evaluation": "The answer says 'the hole is about 10mm' when the context clearly shows 10.5mm diameter. The lack of precision and missing coordinate information reduces utility.",
            "suggestions": "Include the exact value of 10.5mm and provide the specific position coordinates from the context.",
        },
    ],
)

MANUFACTURING_RELEVANCE_CRITIQUE = CADCritiquePromptTemplate(
    name="manufacturing_relevance",
    dimension="manufacturing_relevance",
    description="Evaluate if the answer is relevant to manufacturing contexts",
    rating_scale=STANDARD_RATING_SCALE,
    prompt="""You are an expert manufacturing engineer reviewing Q&A pairs for manufacturing relevance.

Context (CAD Data):
{context}

Question: {question}

Answer: {answer}

Evaluate the MANUFACTURING RELEVANCE of this Q&A pair:
- Is the Q&A useful for someone manufacturing this part?
- Does the answer consider practical manufacturing implications?
- Are machining features correctly identified (holes, pockets, slots, etc.)?
- Would this help with CNC programming, tooling selection, or process planning?

Provide your evaluation in this exact format:
RATING: [1-5]
EVALUATION: [2-3 sentences explaining your rating]
SUGGESTIONS: [Optional improvements if rating < 5]
""",
    examples=[
        {
            "rating": 5,
            "evaluation": "The answer correctly identifies the blind hole dimensions and suggests appropriate machining operations. This would be directly useful for CNC programming.",
            "suggestions": None,
        },
        {
            "rating": 3,
            "evaluation": "While the answer correctly identifies the feature as a pocket, it doesn't mention the depth or any tooling considerations that would be important for manufacturing.",
            "suggestions": "Include the pocket depth and discuss tool radius requirements for corner access.",
        },
    ],
)

ANSWER_COMPLETENESS_CRITIQUE = CADCritiquePromptTemplate(
    name="answer_completeness",
    dimension="answer_completeness",
    description="Evaluate if the answer fully addresses the question",
    rating_scale=STANDARD_RATING_SCALE,
    prompt="""You are an expert CAD engineer reviewing Q&A pairs for completeness.

Context (CAD Data):
{context}

Question: {question}

Answer: {answer}

Evaluate the COMPLETENESS of this answer:
- Does the answer fully address all parts of the question?
- Are there any aspects of the question left unanswered?
- Is the answer self-contained (doesn't require additional context)?
- Is the level of detail appropriate for the question?

Provide your evaluation in this exact format:
RATING: [1-5]
EVALUATION: [2-3 sentences explaining your rating]
SUGGESTIONS: [Optional improvements if rating < 5]
""",
    examples=[
        {
            "rating": 5,
            "evaluation": "The answer addresses all aspects of the question about edge connectivity, listing all connected edges and their relationships. No information is missing.",
            "suggestions": None,
        },
        {
            "rating": 3,
            "evaluation": "The answer provides the face count but fails to mention the face types (planar, cylindrical) that were also asked about in the question.",
            "suggestions": "Include a breakdown of face types: N planar faces, M cylindrical faces, etc.",
        },
    ],
)

QUESTION_CLARITY_CRITIQUE = CADCritiquePromptTemplate(
    name="question_clarity",
    dimension="question_clarity",
    description="Evaluate if the question is clear and well-formed",
    rating_scale=STANDARD_RATING_SCALE,
    prompt="""You are an expert CAD engineer reviewing Q&A pairs for question quality.

Context (CAD Data):
{context}

Question: {question}

Answer: {answer}

Evaluate the CLARITY of the question:
- Is the question clear and unambiguous?
- Does it ask for specific, identifiable information?
- Is the question actually answerable from the context?
- Is the language technically correct?

Provide your evaluation in this exact format:
RATING: [1-5]
EVALUATION: [2-3 sentences explaining your rating]
SUGGESTIONS: [Optional improvements if rating < 5]
""",
    examples=[
        {
            "rating": 5,
            "evaluation": "The question clearly asks about a specific entity (#123) and requests a specific piece of information (radius). It is unambiguous and answerable.",
            "suggestions": None,
        },
        {
            "rating": 2,
            "evaluation": "The question 'What about the holes?' is too vague. It doesn't specify which holes or what information is being requested.",
            "suggestions": "Rephrase to something like 'What are the diameters of the mounting holes in the top face?'",
        },
    ],
)


# =============================================================================
# Critique Registry
# =============================================================================

_CRITIQUE_PROMPTS: dict[str, CADCritiquePromptTemplate] = {
    "technical_accuracy": TECHNICAL_ACCURACY_CRITIQUE,
    "cad_groundedness": CAD_GROUNDEDNESS_CRITIQUE,
    "geometry_specificity": GEOMETRY_SPECIFICITY_CRITIQUE,
    "manufacturing_relevance": MANUFACTURING_RELEVANCE_CRITIQUE,
    "answer_completeness": ANSWER_COMPLETENESS_CRITIQUE,
    "question_clarity": QUESTION_CLARITY_CRITIQUE,
}


def get_critique_prompts() -> dict[str, CADCritiquePromptTemplate]:
    """Get all registered critique prompt templates.

    Returns:
        Dictionary of dimension to template
    """
    return _CRITIQUE_PROMPTS.copy()


def get_critique_for_dimension(dimension: str) -> CADCritiquePromptTemplate | None:
    """Get critique prompt for a specific dimension.

    Args:
        dimension: Critique dimension name

    Returns:
        Critique template or None if not found
    """
    return _CRITIQUE_PROMPTS.get(dimension)


def register_critique(name: str, template: CADCritiquePromptTemplate) -> None:
    """Register a custom critique template.

    Args:
        name: Name to register under
        template: Critique template to register
    """
    _CRITIQUE_PROMPTS[name] = template
