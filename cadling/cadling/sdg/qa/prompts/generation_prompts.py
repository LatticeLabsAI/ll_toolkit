"""CAD-specific prompt templates for Q&A generation.

This module provides prompt templates tailored for generating questions
and answers about CAD documents, covering various aspects like geometry,
topology, manufacturing, and dimensions.

Classes:
    CADQaPromptTemplate: Template for generating CAD Q&A pairs

Constants:
    GEOMETRY_PROMPT: Template for geometry questions
    TOPOLOGY_PROMPT: Template for topology questions
    MANUFACTURING_PROMPT: Template for manufacturing questions
    DIMENSION_PROMPT: Template for dimension questions
    FACT_SINGLE_PROMPT: Template for single fact questions
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from cadling.sdg.qa.base import QuestionType


class CADQaPromptTemplate(BaseModel):
    """Template for generating CAD Q&A pairs.

    Defines the prompts used for generating questions and answers
    from CAD document chunks.

    Attributes:
        name: Template name
        question_type: Type of question this template generates
        question_prompt: Prompt template for question generation
        answer_prompt: Prompt template for answer generation
        description: Description of this template
        examples: Optional example Q&A pairs
    """

    name: str
    question_type: QuestionType
    question_prompt: str
    answer_prompt: str
    description: str = ""
    examples: list[dict[str, Any]] = Field(default_factory=list)

    def format_question_prompt(self, context: str, **kwargs) -> str:
        """Format the question generation prompt with context.

        Args:
            context: CAD chunk text context
            **kwargs: Additional format arguments

        Returns:
            Formatted prompt string
        """
        return self.question_prompt.format(context=context, **kwargs)

    def format_answer_prompt(
        self, context: str, question: str, **kwargs
    ) -> str:
        """Format the answer generation prompt with context and question.

        Args:
            context: CAD chunk text context
            question: Generated question
            **kwargs: Additional format arguments

        Returns:
            Formatted prompt string
        """
        return self.answer_prompt.format(
            context=context, question=question, **kwargs
        )


# =============================================================================
# CAD-Specific Question Generation Prompts
# =============================================================================

GEOMETRY_PROMPT = CADQaPromptTemplate(
    name="geometry",
    question_type=QuestionType.GEOMETRY,
    description="Generate questions about geometric properties like dimensions, volumes, and coordinates",
    question_prompt="""You are a CAD expert. Given this CAD entity information:

{context}

Generate a specific, technical question about the GEOMETRIC PROPERTIES of this CAD data.
Focus on:
- Dimensions (lengths, widths, heights, radii)
- Coordinates and positions
- Volumes and surface areas
- Geometric shapes and primitives
- Spatial relationships

The question should be answerable directly from the provided context.
Generate ONLY the question, nothing else.""",
    answer_prompt="""You are a CAD expert. Given this CAD entity information:

{context}

Question: {question}

Provide a precise, technical answer based ONLY on the information above.
Include specific numeric values and units where applicable.
Generate ONLY the answer, nothing else.""",
    examples=[
        {
            "question": "What is the radius of the cylindrical hole at position (10.5, 20.0, 0.0)?",
            "answer": "The cylindrical hole at position (10.5, 20.0, 0.0) has a radius of 5.25 mm.",
        }
    ],
)

TOPOLOGY_PROMPT = CADQaPromptTemplate(
    name="topology",
    question_type=QuestionType.TOPOLOGY,
    description="Generate questions about topology relationships like faces, edges, and vertices",
    question_prompt="""You are a CAD expert. Given this CAD entity information:

{context}

Generate a specific, technical question about the TOPOLOGY of this CAD data.
Focus on:
- Face relationships and adjacencies
- Edge connectivity
- Vertex positions and connections
- Surface types (planar, cylindrical, etc.)
- Boundary representations (BRep)

The question should be answerable directly from the provided context.
Generate ONLY the question, nothing else.""",
    answer_prompt="""You are a CAD expert. Given this CAD entity information:

{context}

Question: {question}

Provide a precise, technical answer based ONLY on the information above.
Reference specific entity IDs or counts where applicable.
Generate ONLY the answer, nothing else.""",
    examples=[
        {
            "question": "How many edges are connected to vertex #123?",
            "answer": "Vertex #123 is connected to 4 edges: Edge #45, Edge #46, Edge #89, and Edge #90.",
        }
    ],
)

MANUFACTURING_PROMPT = CADQaPromptTemplate(
    name="manufacturing",
    question_type=QuestionType.MANUFACTURING,
    description="Generate questions about manufacturing processes and DFM considerations",
    question_prompt="""You are a CAD and manufacturing expert. Given this CAD entity information:

{context}

Generate a specific question about MANUFACTURING CONSIDERATIONS for this CAD data.
Focus on:
- Machining features (holes, pockets, slots)
- Material removal operations
- Design for manufacturability (DFM)
- Surface finish requirements
- Tooling considerations

The question should be answerable or inferable from the provided context.
Generate ONLY the question, nothing else.""",
    answer_prompt="""You are a CAD and manufacturing expert. Given this CAD entity information:

{context}

Question: {question}

Provide a practical, technical answer based on the information above.
Consider manufacturing processes and DFM principles.
Generate ONLY the answer, nothing else.""",
    examples=[
        {
            "question": "What machining operation would be used to create the blind hole with diameter 8mm and depth 15mm?",
            "answer": "A drilling operation followed by boring would be used to create this blind hole. Given the 8mm diameter and 15mm depth (L/D ratio of 1.875), a standard twist drill can achieve this without specialized tooling.",
        }
    ],
)

DIMENSION_PROMPT = CADQaPromptTemplate(
    name="dimension",
    question_type=QuestionType.DIMENSION,
    description="Generate questions about specific dimensional measurements",
    question_prompt="""You are a CAD expert. Given this CAD entity information:

{context}

Generate a specific question about DIMENSIONAL MEASUREMENTS in this CAD data.
Focus on:
- Specific length, width, or height measurements
- Angular dimensions
- Distance between features
- Positional dimensions
- Overall part envelope

The question should be answerable directly from the provided context.
Generate ONLY the question, nothing else.""",
    answer_prompt="""You are a CAD expert. Given this CAD entity information:

{context}

Question: {question}

Provide the exact dimensional answer based ONLY on the information above.
Include the numeric value and appropriate units.
Generate ONLY the answer, nothing else.""",
    examples=[
        {
            "question": "What is the distance between the two mounting holes?",
            "answer": "The distance between the two mounting holes is 50.0 mm, measured center-to-center.",
        }
    ],
)

FACT_SINGLE_PROMPT = CADQaPromptTemplate(
    name="fact_single",
    question_type=QuestionType.FACT_SINGLE,
    description="Generate single-fact extraction questions from CAD data",
    question_prompt="""You are a CAD expert. Given this CAD entity information:

{context}

Generate a simple, direct question that extracts ONE SPECIFIC FACT from this CAD data.
The question should:
- Be answerable with a short, factual response
- Focus on a single piece of information
- Not require complex reasoning or inference

Generate ONLY the question, nothing else.""",
    answer_prompt="""You are a CAD expert. Given this CAD entity information:

{context}

Question: {question}

Provide a concise, factual answer based ONLY on the information above.
Keep the answer brief and direct.
Generate ONLY the answer, nothing else.""",
    examples=[
        {
            "question": "What is the entity type of item #456?",
            "answer": "Item #456 is a CYLINDRICAL_SURFACE entity.",
        }
    ],
)

TOLERANCE_PROMPT = CADQaPromptTemplate(
    name="tolerance",
    question_type=QuestionType.TOLERANCE,
    description="Generate questions about tolerance specifications and GD&T",
    question_prompt="""You are a CAD expert. Given this CAD entity information:

{context}

Generate a specific question about TOLERANCE SPECIFICATIONS in this CAD data.
Focus on:
- Dimensional tolerances
- Geometric tolerances (GD&T)
- Surface finish requirements
- Fit and clearance specifications
- Datum references

The question should be answerable or inferable from the provided context.
Generate ONLY the question, nothing else.""",
    answer_prompt="""You are a CAD expert. Given this CAD entity information:

{context}

Question: {question}

Provide a precise answer about the tolerance specification based on the information above.
Include tolerance values and any relevant datum references.
Generate ONLY the answer, nothing else.""",
    examples=[
        {
            "question": "What is the positional tolerance for the bolt hole pattern?",
            "answer": "The bolt hole pattern has a positional tolerance of 0.25 mm at MMC, referenced to datum A-B-C.",
        }
    ],
)

MATERIAL_PROMPT = CADQaPromptTemplate(
    name="material",
    question_type=QuestionType.MATERIAL,
    description="Generate questions about material properties and specifications",
    question_prompt="""You are a CAD expert. Given this CAD entity information:

{context}

Generate a specific question about MATERIAL PROPERTIES in this CAD data.
Focus on:
- Material type and grade
- Mechanical properties (strength, hardness)
- Physical properties (density, thermal conductivity)
- Material specifications and standards

The question should be answerable from the provided context.
Generate ONLY the question, nothing else.""",
    answer_prompt="""You are a CAD expert. Given this CAD entity information:

{context}

Question: {question}

Provide a technical answer about the material based on the information above.
Include specific material grades or property values if available.
Generate ONLY the answer, nothing else.""",
    examples=[
        {
            "question": "What material is specified for this component?",
            "answer": "The component is specified as AISI 316L stainless steel with a density of 8.0 g/cm³.",
        }
    ],
)

ASSEMBLY_PROMPT = CADQaPromptTemplate(
    name="assembly",
    question_type=QuestionType.ASSEMBLY,
    description="Generate questions about assembly structure and component relationships",
    question_prompt="""You are a CAD expert. Given this CAD entity information:

{context}

Generate a specific question about ASSEMBLY STRUCTURE in this CAD data.
Focus on:
- Component hierarchy
- Mating relationships
- Transformation matrices
- Part counts and instances
- Assembly constraints

The question should be answerable from the provided context.
Generate ONLY the question, nothing else.""",
    answer_prompt="""You are a CAD expert. Given this CAD entity information:

{context}

Question: {question}

Provide a clear answer about the assembly structure based on the information above.
Reference specific component names or relationships.
Generate ONLY the answer, nothing else.""",
    examples=[
        {
            "question": "How many instances of the M8 bolt are used in this assembly?",
            "answer": "There are 12 instances of the M8 bolt used in this assembly, distributed across 3 mounting locations with 4 bolts each.",
        }
    ],
)


# =============================================================================
# Prompt Registry
# =============================================================================

_GENERATION_PROMPTS: dict[str, CADQaPromptTemplate] = {
    "geometry": GEOMETRY_PROMPT,
    "topology": TOPOLOGY_PROMPT,
    "manufacturing": MANUFACTURING_PROMPT,
    "dimension": DIMENSION_PROMPT,
    "fact_single": FACT_SINGLE_PROMPT,
    "tolerance": TOLERANCE_PROMPT,
    "material": MATERIAL_PROMPT,
    "assembly": ASSEMBLY_PROMPT,
}


def get_generation_prompts() -> dict[str, CADQaPromptTemplate]:
    """Get all registered generation prompt templates.

    Returns:
        Dictionary of prompt name to template
    """
    return _GENERATION_PROMPTS.copy()


def get_prompts_for_question_type(
    question_type: QuestionType,
) -> list[CADQaPromptTemplate]:
    """Get all prompts that generate the specified question type.

    Args:
        question_type: Type of question to filter by

    Returns:
        List of matching prompt templates
    """
    return [
        p for p in _GENERATION_PROMPTS.values()
        if p.question_type == question_type
    ]


def register_prompt(name: str, template: CADQaPromptTemplate) -> None:
    """Register a custom prompt template.

    Args:
        name: Name to register under
        template: Prompt template to register
    """
    _GENERATION_PROMPTS[name] = template
