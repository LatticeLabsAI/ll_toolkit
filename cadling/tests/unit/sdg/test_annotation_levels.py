"""Tests for multi-level annotation system."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from cadling.sdg.qa.base import AnnotationLevel


class TestAnnotationLevel:
    """Test AnnotationLevel enum."""

    def test_enum_values(self):
        assert AnnotationLevel.ABSTRACT == "abstract"
        assert AnnotationLevel.INTERMEDIATE == "intermediate"
        assert AnnotationLevel.DETAILED == "detailed"
        assert AnnotationLevel.EXPERT == "expert"

    def test_all_levels(self):
        levels = list(AnnotationLevel)
        assert len(levels) == 4

    def test_string_comparison(self):
        assert AnnotationLevel.ABSTRACT == "abstract"
        assert AnnotationLevel("detailed") == AnnotationLevel.DETAILED

    def test_serialization_roundtrip(self):
        level = AnnotationLevel.EXPERT
        serialized = level.value
        restored = AnnotationLevel(serialized)
        assert restored == level


class TestAnnotationLevelPromptModifier:
    """Test prompt modification per level."""

    def test_abstract_modifier(self):
        from cadling.sdg.qa.prompts.annotation_prompts import (
            AnnotationLevelPromptModifier,
        )
        modifier = AnnotationLevelPromptModifier("abstract")
        base_prompt = "Generate a question about this CAD model."
        modified = modifier.modify_question_prompt(base_prompt)

        assert "[Annotation Level: ABSTRACT]" in modified
        assert "HIGH-LEVEL" in modified
        assert base_prompt in modified

    def test_expert_modifier(self):
        from cadling.sdg.qa.prompts.annotation_prompts import (
            AnnotationLevelPromptModifier,
        )
        modifier = AnnotationLevelPromptModifier("expert")
        base_prompt = "Generate an answer about this CAD model."
        modified = modifier.modify_answer_prompt(base_prompt)

        assert "[Annotation Level: EXPERT]" in modified
        assert "exact coordinates" in modified
        assert base_prompt in modified

    def test_invalid_level_raises(self):
        from cadling.sdg.qa.prompts.annotation_prompts import (
            AnnotationLevelPromptModifier,
        )
        with pytest.raises(ValueError, match="Unknown annotation level"):
            AnnotationLevelPromptModifier("invalid_level")

    def test_all_levels_produce_different_output(self):
        from cadling.sdg.qa.prompts.annotation_prompts import (
            AnnotationLevelPromptModifier,
        )
        base = "Generate a question."
        outputs = set()
        for level in ["abstract", "intermediate", "detailed", "expert"]:
            modifier = AnnotationLevelPromptModifier(level)
            outputs.add(modifier.modify_question_prompt(base))
        assert len(outputs) == 4

    def test_answer_modifier_includes_must(self):
        from cadling.sdg.qa.prompts.annotation_prompts import (
            AnnotationLevelPromptModifier,
        )
        modifier = AnnotationLevelPromptModifier("detailed")
        modified = modifier.modify_answer_prompt("Base answer prompt.")
        assert "MUST include" in modified


class TestAnnotationLevelCritiqueTemplate:
    """Test critique template for level consistency."""

    def test_format_abstract(self):
        from cadling.sdg.qa.prompts.annotation_prompts import (
            AnnotationLevelCritiqueTemplate,
        )
        result = AnnotationLevelCritiqueTemplate.format(
            level="abstract",
            question="What is this part?",
            answer="A cylindrical rod.",
        )
        assert "ABSTRACT" in result
        assert "What is this part?" in result
        assert "A cylindrical rod." in result
        assert "Score from 1-5" in result

    def test_format_expert(self):
        from cadling.sdg.qa.prompts.annotation_prompts import (
            AnnotationLevelCritiqueTemplate,
        )
        result = AnnotationLevelCritiqueTemplate.format(
            level="expert",
            question="What are the parametric specs?",
            answer="Radius 5.0mm at origin (0,0,0).",
        )
        assert "EXPERT" in result

    def test_format_invalid_level(self):
        from cadling.sdg.qa.prompts.annotation_prompts import (
            AnnotationLevelCritiqueTemplate,
        )
        result = AnnotationLevelCritiqueTemplate.format(
            level="nonexistent",
            question="Q",
            answer="A",
        )
        assert result == ""


class TestGetModifierForLevel:
    """Test convenience function."""

    def test_returns_modifier(self):
        from cadling.sdg.qa.prompts.annotation_prompts import get_modifier_for_level
        modifier = get_modifier_for_level("intermediate")
        assert modifier.level == "intermediate"

    def test_invalid_raises(self):
        from cadling.sdg.qa.prompts.annotation_prompts import get_modifier_for_level
        with pytest.raises(ValueError):
            get_modifier_for_level("bad_level")


class TestAnnotationLevelIntegration:
    """Integration tests for annotation level in generation options."""

    def test_default_no_levels(self):
        """Default options should have no annotation levels."""
        from cadling.sdg.qa.base import CADGenerateOptions
        options = CADGenerateOptions(
            provider="openai",
            model_id="gpt-4",
        )
        assert options.annotation_levels == []

    def test_options_with_levels(self):
        """CADGenerateOptions should accept annotation levels."""
        from cadling.sdg.qa.base import CADGenerateOptions
        options = CADGenerateOptions(
            provider="openai",
            model_id="gpt-4",
            annotation_levels=[AnnotationLevel.ABSTRACT, AnnotationLevel.EXPERT],
        )
        assert len(options.annotation_levels) == 2
        assert AnnotationLevel.ABSTRACT in options.annotation_levels
        assert AnnotationLevel.EXPERT in options.annotation_levels

    def test_qac_tagged_with_level(self):
        """CADGenQAC should accept annotation_level field."""
        from cadling.sdg.qa.base import CADGenQAC, AnnotationLevel
        qac = CADGenQAC(
            question="What is this part?",
            answer="A cylindrical rod.",
            context="STEP entity data...",
            annotation_level=AnnotationLevel.ABSTRACT,
        )
        assert qac.annotation_level == AnnotationLevel.ABSTRACT

    def test_qac_default_no_level(self):
        """CADGenQAC should default to no annotation level."""
        from cadling.sdg.qa.base import CADGenQAC
        qac = CADGenQAC(
            question="What is this?",
            answer="A part.",
            context="Context.",
        )
        assert qac.annotation_level is None


class TestPromptTemplateWithAnnotationLevel:
    """Test CADQaPromptTemplate with_annotation_level method."""

    def test_with_annotation_level_creates_new_template(self):
        from cadling.sdg.qa.prompts.generation_prompts import FACT_SINGLE_PROMPT
        leveled = FACT_SINGLE_PROMPT.with_annotation_level("abstract")
        assert leveled.annotation_level == "abstract"
        assert leveled.name == "fact_single_abstract"
        assert "[Annotation Level: ABSTRACT]" in leveled.question_prompt
        assert "[Annotation Level: ABSTRACT]" in leveled.answer_prompt

    def test_with_annotation_level_preserves_question_type(self):
        from cadling.sdg.qa.prompts.generation_prompts import GEOMETRY_PROMPT
        from cadling.sdg.qa.base import QuestionType
        leveled = GEOMETRY_PROMPT.with_annotation_level("detailed")
        assert leveled.question_type == QuestionType.GEOMETRY

    def test_original_template_unchanged(self):
        from cadling.sdg.qa.prompts.generation_prompts import TOPOLOGY_PROMPT
        original_q = TOPOLOGY_PROMPT.question_prompt
        original_a = TOPOLOGY_PROMPT.answer_prompt
        _ = TOPOLOGY_PROMPT.with_annotation_level("expert")
        assert TOPOLOGY_PROMPT.question_prompt == original_q
        assert TOPOLOGY_PROMPT.answer_prompt == original_a
        assert TOPOLOGY_PROMPT.annotation_level is None


class TestGetPromptsForLevel:
    """Test get_prompts_for_level function."""

    def test_wraps_all_prompts(self):
        from cadling.sdg.qa.prompts.generation_prompts import (
            get_prompts_for_level,
            FACT_SINGLE_PROMPT,
            GEOMETRY_PROMPT,
        )
        base = [FACT_SINGLE_PROMPT, GEOMETRY_PROMPT]
        leveled = get_prompts_for_level(base, "intermediate")
        assert len(leveled) == 2
        for p in leveled:
            assert p.annotation_level == "intermediate"
            assert "[Annotation Level: INTERMEDIATE]" in p.question_prompt

    def test_empty_base_prompts(self):
        from cadling.sdg.qa.prompts.generation_prompts import get_prompts_for_level
        leveled = get_prompts_for_level([], "abstract")
        assert leveled == []


class TestLevelConsistencyCritique:
    """Test LEVEL_CONSISTENCY_CRITIQUE prompt template."""

    def test_template_exists(self):
        from cadling.sdg.qa.prompts.critique_prompts import LEVEL_CONSISTENCY_CRITIQUE
        assert "annotation_level" in LEVEL_CONSISTENCY_CRITIQUE
        assert "question" in LEVEL_CONSISTENCY_CRITIQUE
        assert "answer" in LEVEL_CONSISTENCY_CRITIQUE

    def test_template_format(self):
        from cadling.sdg.qa.prompts.critique_prompts import LEVEL_CONSISTENCY_CRITIQUE
        formatted = LEVEL_CONSISTENCY_CRITIQUE.format(
            annotation_level="ABSTRACT",
            question="What is this part?",
            answer="A cylindrical rod.",
        )
        assert "ABSTRACT" in formatted
        assert "What is this part?" in formatted
        assert "A cylindrical rod." in formatted


class TestExportsPrompts:
    """Test that all annotation-related exports are accessible."""

    def test_prompts_init_exports(self):
        from cadling.sdg.qa.prompts import (
            AnnotationLevelPromptModifier,
            AnnotationLevelCritiqueTemplate,
            get_modifier_for_level,
            LEVEL_CONSISTENCY_CRITIQUE,
        )
        assert AnnotationLevelPromptModifier is not None
        assert AnnotationLevelCritiqueTemplate is not None
        assert get_modifier_for_level is not None
        assert LEVEL_CONSISTENCY_CRITIQUE is not None

    def test_qa_init_exports(self):
        from cadling.sdg.qa import AnnotationLevel
        assert AnnotationLevel is not None
        assert AnnotationLevel.ABSTRACT == "abstract"
