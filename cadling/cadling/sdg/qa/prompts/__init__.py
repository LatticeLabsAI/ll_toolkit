"""CAD-specific prompt templates for Q&A generation.

This module provides prompt templates for generating and critiquing
Q&A pairs from CAD documents.

Exports:
    CADQaPromptTemplate: Template for Q&A generation
    CADCritiquePromptTemplate: Template for critique
    get_generation_prompts: Get all generation prompts
    get_critique_prompts: Get all critique prompts
    AnnotationLevelPromptModifier: Modify prompts with annotation level constraints
    AnnotationLevelCritiqueTemplate: Critique template for level consistency
    get_modifier_for_level: Get prompt modifier for a given annotation level
    LEVEL_CONSISTENCY_CRITIQUE: Level consistency critique prompt template
"""

from cadling.sdg.qa.prompts.generation_prompts import (
    CADQaPromptTemplate,
    get_generation_prompts,
    get_prompts_for_level,
    get_prompts_for_question_type,
    register_prompt,
    GEOMETRY_PROMPT,
    TOPOLOGY_PROMPT,
    MANUFACTURING_PROMPT,
    DIMENSION_PROMPT,
    FACT_SINGLE_PROMPT,
    TOLERANCE_PROMPT,
    MATERIAL_PROMPT,
    ASSEMBLY_PROMPT,
)

from cadling.sdg.qa.prompts.critique_prompts import (
    CADCritiquePromptTemplate,
    get_critique_prompts,
    get_critique_for_dimension,
    register_critique,
    TECHNICAL_ACCURACY_CRITIQUE,
    CAD_GROUNDEDNESS_CRITIQUE,
    GEOMETRY_SPECIFICITY_CRITIQUE,
    MANUFACTURING_RELEVANCE_CRITIQUE,
    ANSWER_COMPLETENESS_CRITIQUE,
    QUESTION_CLARITY_CRITIQUE,
    LEVEL_CONSISTENCY_CRITIQUE,
)

from cadling.sdg.qa.prompts.annotation_prompts import (
    AnnotationLevelPromptModifier,
    AnnotationLevelCritiqueTemplate,
    get_modifier_for_level,
)

__all__ = [
    # Generation
    "CADQaPromptTemplate",
    "get_generation_prompts",
    "get_prompts_for_question_type",
    "get_prompts_for_level",
    "register_prompt",
    "GEOMETRY_PROMPT",
    "TOPOLOGY_PROMPT",
    "MANUFACTURING_PROMPT",
    "DIMENSION_PROMPT",
    "FACT_SINGLE_PROMPT",
    "TOLERANCE_PROMPT",
    "MATERIAL_PROMPT",
    "ASSEMBLY_PROMPT",
    # Critique
    "CADCritiquePromptTemplate",
    "get_critique_prompts",
    "get_critique_for_dimension",
    "register_critique",
    "TECHNICAL_ACCURACY_CRITIQUE",
    "CAD_GROUNDEDNESS_CRITIQUE",
    "GEOMETRY_SPECIFICITY_CRITIQUE",
    "MANUFACTURING_RELEVANCE_CRITIQUE",
    "ANSWER_COMPLETENESS_CRITIQUE",
    "QUESTION_CLARITY_CRITIQUE",
    "LEVEL_CONSISTENCY_CRITIQUE",
    # Annotation levels
    "AnnotationLevelPromptModifier",
    "AnnotationLevelCritiqueTemplate",
    "get_modifier_for_level",
]
