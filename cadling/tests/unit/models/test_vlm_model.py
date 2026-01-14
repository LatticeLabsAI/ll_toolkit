"""
Unit tests for VLM (Vision-Language Model) models.

Tests cover:
- VlmAnnotation, VlmResponse, VlmOptions pydantic models
- Error handling for missing dependencies
- Model initialization requirements
"""

import pytest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch
import json
import base64

from cadling.models.vlm_model import (
    VlmAnnotation,
    VlmResponse,
    VlmOptions,
    ApiVlmOptions,
    InlineVlmOptions,
)


# ============================================================================
# Test Pydantic Models
# ============================================================================


class TestVlmAnnotation:
    """Test VlmAnnotation pydantic model."""

    def test_annotation_initialization(self):
        """Test basic annotation initialization."""
        annotation = VlmAnnotation(
            annotation_type="dimension",
            text="10.5 mm",
            value=10.5,
            unit="mm",
            confidence=0.95,
            bbox=[100, 100, 200, 150],
        )

        assert annotation.annotation_type == "dimension"
        assert annotation.text == "10.5 mm"
        assert annotation.value == 10.5
        assert annotation.unit == "mm"
        assert annotation.confidence == 0.95
        assert annotation.bbox == [100, 100, 200, 150]

    def test_annotation_optional_fields(self):
        """Test annotation with optional fields."""
        annotation = VlmAnnotation(
            annotation_type="tolerance",
            text="±0.05",
        )

        assert annotation.annotation_type == "tolerance"
        assert annotation.text == "±0.05"
        assert annotation.value is None
        assert annotation.unit is None
        assert annotation.confidence == 1.0  # Default value
        assert annotation.bbox is None

    def test_annotation_to_dict(self):
        """Test annotation to dict conversion."""
        annotation = VlmAnnotation(
            annotation_type="note",
            text="Material: Steel",
            confidence=0.9,
        )

        data = annotation.model_dump()

        assert data["annotation_type"] == "note"
        assert data["text"] == "Material: Steel"
        assert data["confidence"] == 0.9

    def test_annotation_from_dict(self):
        """Test annotation from dict."""
        data = {
            "annotation_type": "dimension",
            "text": "5.0",
            "value": 5.0,
            "confidence": 0.85,
        }

        annotation = VlmAnnotation(**data)

        assert annotation.annotation_type == "dimension"
        assert annotation.value == 5.0

    def test_annotation_with_bbox(self):
        """Test annotation with bounding box."""
        annotation = VlmAnnotation(
            annotation_type="label",
            text="Part A",
            bbox=[50, 60, 150, 90],
        )

        assert annotation.bbox == [50, 60, 150, 90]
        assert len(annotation.bbox) == 4


class TestVlmResponse:
    """Test VlmResponse pydantic model."""

    def test_response_initialization(self):
        """Test basic response initialization."""
        annotations = [
            VlmAnnotation(annotation_type="dimension", text="10 mm", value=10.0),
            VlmAnnotation(annotation_type="tolerance", text="±0.1"),
        ]

        response = VlmResponse(
            annotations=annotations,
            raw_text="Dimension: 10 mm ±0.1",
            metadata={"model": "gpt-4-vision", "tokens": 150},
        )

        assert len(response.annotations) == 2
        assert response.raw_text == "Dimension: 10 mm ±0.1"
        assert response.metadata["model"] == "gpt-4-vision"

    def test_response_empty_annotations(self):
        """Test response with no annotations."""
        response = VlmResponse(
            annotations=[],
            raw_text="No annotations found",
        )

        assert len(response.annotations) == 0
        assert response.raw_text == "No annotations found"
        assert response.metadata == {}  # Default factory returns empty dict

    def test_response_defaults(self):
        """Test response with default values."""
        response = VlmResponse()

        assert response.annotations == []
        assert response.raw_text == ""
        assert response.metadata == {}

    def test_response_with_metadata(self):
        """Test response with rich metadata."""
        metadata = {
            "model": "claude-3-opus",
            "tokens": 200,
            "ocr_text": [{"text": "10 mm", "confidence": 0.95}],
        }

        response = VlmResponse(
            annotations=[],
            raw_text="OCR extracted",
            metadata=metadata,
        )

        assert response.metadata["model"] == "claude-3-opus"
        assert response.metadata["tokens"] == 200
        assert len(response.metadata["ocr_text"]) == 1


class TestVlmOptions:
    """Test VlmOptions pydantic models."""

    def test_base_options(self):
        """Test base VlmOptions."""
        options = VlmOptions(
            temperature=0.7,
            max_tokens=500,
            use_ocr=True,
        )

        assert options.temperature == 0.7
        assert options.max_tokens == 500
        assert options.use_ocr is True

    def test_base_options_defaults(self):
        """Test base VlmOptions defaults."""
        options = VlmOptions()

        assert options.temperature == 0.0
        assert options.max_tokens == 4096
        assert options.use_ocr is True

    def test_api_options(self):
        """Test ApiVlmOptions."""
        options = ApiVlmOptions(
            model_name="gpt-4-vision-preview",
            api_key="test-key",
            temperature=0.5,
            timeout=60,
        )

        assert options.model_name == "gpt-4-vision-preview"
        assert options.api_key == "test-key"
        assert options.timeout == 60
        assert options.temperature == 0.5  # Inherited from base

    def test_api_options_defaults(self):
        """Test ApiVlmOptions defaults."""
        options = ApiVlmOptions(api_key="test-key")

        assert options.model_name == "gpt-4-vision-preview"
        assert options.timeout == 60
        assert options.api_base is None

    def test_api_options_custom_base(self):
        """Test ApiVlmOptions with custom API base."""
        options = ApiVlmOptions(
            api_key="test-key",
            api_base="https://custom.api.com",
        )

        assert options.api_base == "https://custom.api.com"

    def test_inline_options(self):
        """Test InlineVlmOptions."""
        options = InlineVlmOptions(
            model_path="llava-hf/llava-1.5-7b-hf",
            device="cuda",
            precision="float16",
        )

        assert options.model_path == "llava-hf/llava-1.5-7b-hf"
        assert options.device == "cuda"
        assert options.precision == "float16"

    def test_inline_options_defaults(self):
        """Test InlineVlmOptions defaults."""
        options = InlineVlmOptions()

        assert options.model_path == "llava-hf/llava-1.5-7b-hf"
        assert options.device == "cpu"
        assert options.precision == "fp32"

    def test_inline_options_different_models(self):
        """Test InlineVlmOptions with different model paths."""
        # LLaVA
        options1 = InlineVlmOptions(model_path="llava-hf/llava-1.5-13b-hf")
        assert "llava" in options1.model_path.lower()

        # Qwen
        options2 = InlineVlmOptions(model_path="Qwen/Qwen2-VL-7B-Instruct")
        assert "qwen" in options2.model_path.lower()

        # BLIP-2
        options3 = InlineVlmOptions(model_path="Salesforce/blip2-opt-2.7b")
        assert "blip" in options3.model_path.lower()


# ============================================================================
# Test Model Initialization Requirements
# ============================================================================


class TestModelRequirements:
    """Test model initialization requirements."""

    def test_api_vlm_requires_openai_or_anthropic(self):
        """Test ApiVlmModel requires API library."""
        import cadling.models.vlm_model as vlm_mod

        # Save original values
        orig_openai = vlm_mod._OPENAI_AVAILABLE
        orig_anthropic = vlm_mod._ANTHROPIC_AVAILABLE

        try:
            # Test OpenAI requirement
            vlm_mod._OPENAI_AVAILABLE = False
            vlm_mod._ANTHROPIC_AVAILABLE = False

            from cadling.models.vlm_model import ApiVlmModel

            options = ApiVlmOptions(
                model_name="gpt-4-vision-preview",
                api_key="test-key",
            )

            with pytest.raises(ImportError, match="openai package required"):
                ApiVlmModel(options)

            # Test Claude requirement
            options2 = ApiVlmOptions(
                model_name="claude-3-opus-20240229",
                api_key="test-key",
            )

            with pytest.raises(ImportError, match="anthropic package required"):
                ApiVlmModel(options2)

        finally:
            # Restore original values
            vlm_mod._OPENAI_AVAILABLE = orig_openai
            vlm_mod._ANTHROPIC_AVAILABLE = orig_anthropic

    def test_inline_vlm_requires_transformers(self):
        """Test InlineVlmModel requires transformers library."""
        import cadling.models.vlm_model as vlm_mod

        # Save original value
        orig_transformers = vlm_mod._TRANSFORMERS_AVAILABLE

        try:
            vlm_mod._TRANSFORMERS_AVAILABLE = False

            from cadling.models.vlm_model import InlineVlmModel

            options = InlineVlmOptions(
                model_path="llava-hf/llava-1.5-7b-hf",
            )

            with pytest.raises(ImportError, match="transformers package required"):
                InlineVlmModel(options)

        finally:
            # Restore original value
            vlm_mod._TRANSFORMERS_AVAILABLE = orig_transformers


# ============================================================================
# Test Annotation Types
# ============================================================================


class TestAnnotationTypes:
    """Test different annotation types."""

    def test_dimension_annotation(self):
        """Test dimension annotation."""
        annotation = VlmAnnotation(
            annotation_type="dimension",
            text="20.5 mm",
            value=20.5,
            unit="mm",
            confidence=0.92,
        )

        assert annotation.annotation_type == "dimension"
        assert annotation.value == 20.5
        assert annotation.unit == "mm"

    def test_tolerance_annotation(self):
        """Test tolerance annotation."""
        annotation = VlmAnnotation(
            annotation_type="tolerance",
            text="±0.05",
            confidence=0.88,
        )

        assert annotation.annotation_type == "tolerance"
        assert annotation.value is None
        assert annotation.unit is None

    def test_note_annotation(self):
        """Test note annotation."""
        annotation = VlmAnnotation(
            annotation_type="note",
            text="Material: Aluminum 6061-T6",
        )

        assert annotation.annotation_type == "note"
        assert "Aluminum" in annotation.text

    def test_label_annotation(self):
        """Test label annotation."""
        annotation = VlmAnnotation(
            annotation_type="label",
            text="Part A",
            bbox=[100, 100, 150, 120],
        )

        assert annotation.annotation_type == "label"
        assert annotation.bbox is not None

    def test_multiple_annotations_in_response(self):
        """Test response with multiple annotation types."""
        annotations = [
            VlmAnnotation(annotation_type="dimension", text="10 mm", value=10.0),
            VlmAnnotation(annotation_type="tolerance", text="±0.1"),
            VlmAnnotation(annotation_type="note", text="Material: Steel"),
            VlmAnnotation(annotation_type="label", text="Section A-A"),
        ]

        response = VlmResponse(annotations=annotations)

        assert len(response.annotations) == 4
        assert response.annotations[0].annotation_type == "dimension"
        assert response.annotations[1].annotation_type == "tolerance"
        assert response.annotations[2].annotation_type == "note"
        assert response.annotations[3].annotation_type == "label"


# ============================================================================
# Test JSON Serialization
# ============================================================================


class TestJSONSerialization:
    """Test JSON serialization of VLM models."""

    def test_annotation_json_serialization(self):
        """Test annotation JSON serialization."""
        annotation = VlmAnnotation(
            annotation_type="dimension",
            text="15.5 mm",
            value=15.5,
            unit="mm",
            confidence=0.95,
            bbox=[10, 20, 50, 40],
        )

        json_str = annotation.model_dump_json()
        data = json.loads(json_str)

        assert data["annotation_type"] == "dimension"
        assert data["value"] == 15.5
        assert data["bbox"] == [10, 20, 50, 40]

    def test_response_json_serialization(self):
        """Test response JSON serialization."""
        annotations = [
            VlmAnnotation(annotation_type="dimension", text="10 mm", value=10.0),
        ]

        response = VlmResponse(
            annotations=annotations,
            raw_text="Test",
            metadata={"model": "gpt-4-vision"},
        )

        json_str = response.model_dump_json()
        data = json.loads(json_str)

        assert len(data["annotations"]) == 1
        assert data["raw_text"] == "Test"
        assert data["metadata"]["model"] == "gpt-4-vision"

    def test_options_json_serialization(self):
        """Test options JSON serialization."""
        options = ApiVlmOptions(
            model_name="gpt-4-vision-preview",
            api_key="test-key",
            temperature=0.7,
        )

        json_str = options.model_dump_json()
        data = json.loads(json_str)

        assert data["model_name"] == "gpt-4-vision-preview"
        assert data["api_key"] == "test-key"
        assert data["temperature"] == 0.7


# ============================================================================
# Test Validation
# ============================================================================


class TestValidation:
    """Test pydantic validation."""

    def test_annotation_requires_type_and_text(self):
        """Test annotation requires annotation_type and text."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            VlmAnnotation(annotation_type="dimension")  # Missing text

        with pytest.raises(Exception):  # Pydantic ValidationError
            VlmAnnotation(text="10 mm")  # Missing annotation_type

    def test_api_options_requires_api_key(self):
        """Test ApiVlmOptions requires api_key."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            ApiVlmOptions(model_name="gpt-4-vision-preview")  # Missing api_key

    def test_confidence_value_range(self):
        """Test confidence should be between 0 and 1."""
        # Valid confidence
        annotation1 = VlmAnnotation(
            annotation_type="dimension",
            text="10 mm",
            confidence=0.95,
        )
        assert 0 <= annotation1.confidence <= 1

        # Edge cases
        annotation2 = VlmAnnotation(
            annotation_type="dimension",
            text="10 mm",
            confidence=0.0,
        )
        assert annotation2.confidence == 0.0

        annotation3 = VlmAnnotation(
            annotation_type="dimension",
            text="10 mm",
            confidence=1.0,
        )
        assert annotation3.confidence == 1.0
