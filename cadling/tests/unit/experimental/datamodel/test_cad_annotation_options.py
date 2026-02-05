"""
Unit tests for CADAnnotationOptions.

Tests cover:
- Model initialization with defaults
- Model validation
- Configuration options for PMI extraction
- VLM model configuration
- View processing settings
"""

import pytest
from pydantic import ValidationError

from cadling.experimental.datamodel import CADAnnotationOptions


class TestCADAnnotationOptions:
    """Test CADAnnotationOptions pydantic model."""

    def test_default_initialization(self):
        """Test initialization with default values."""
        options = CADAnnotationOptions()

        assert options.annotation_types == [
            "dimension",
            "tolerance",
            "gdt",
            "surface_finish",
            "material",
            "welding",
            "note",
        ]
        assert options.min_confidence == 0.7
        assert options.vlm_model == "gpt-4-vision-preview"
        assert options.views_to_process == ["front", "top", "right"]
        assert options.enable_cross_view_validation is True
        assert options.extraction_resolution == 2048
        assert options.include_geometric_context is True

    def test_custom_annotation_types(self):
        """Test custom annotation type configuration."""
        options = CADAnnotationOptions(
            annotation_types=["dimension", "tolerance"]
        )

        assert len(options.annotation_types) == 2
        assert "dimension" in options.annotation_types
        assert "tolerance" in options.annotation_types

    def test_custom_vlm_model(self):
        """Test custom VLM model configuration."""
        options = CADAnnotationOptions(vlm_model="claude-3-opus-20240229")

        assert options.vlm_model == "claude-3-opus-20240229"

    def test_custom_views(self):
        """Test custom view configuration."""
        options = CADAnnotationOptions(
            views_to_process=["isometric", "front"]
        )

        assert len(options.views_to_process) == 2
        assert "isometric" in options.views_to_process

    def test_confidence_threshold(self):
        """Test confidence threshold validation."""
        # Valid confidence
        options = CADAnnotationOptions(min_confidence=0.5)
        assert options.min_confidence == 0.5

        # Edge cases
        options = CADAnnotationOptions(min_confidence=0.0)
        assert options.min_confidence == 0.0

        options = CADAnnotationOptions(min_confidence=1.0)
        assert options.min_confidence == 1.0

    def test_confidence_threshold_validation(self):
        """Test confidence threshold out of bounds."""
        with pytest.raises(ValidationError):
            CADAnnotationOptions(min_confidence=-0.1)

        with pytest.raises(ValidationError):
            CADAnnotationOptions(min_confidence=1.1)

    def test_resolution_validation(self):
        """Test resolution validation."""
        # Valid resolutions
        options = CADAnnotationOptions(extraction_resolution=1024)
        assert options.extraction_resolution == 1024

        options = CADAnnotationOptions(extraction_resolution=4096)
        assert options.extraction_resolution == 4096

        # Out of bounds
        with pytest.raises(ValidationError):
            CADAnnotationOptions(extraction_resolution=255)

        with pytest.raises(ValidationError):
            CADAnnotationOptions(extraction_resolution=8193)

    def test_cross_view_validation_toggle(self):
        """Test cross-view validation can be disabled."""
        options = CADAnnotationOptions(enable_cross_view_validation=False)

        assert options.enable_cross_view_validation is False

    def test_geometric_context_toggle(self):
        """Test geometric context can be disabled."""
        options = CADAnnotationOptions(include_geometric_context=False)

        assert options.include_geometric_context is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        options = CADAnnotationOptions(
            annotation_types=["dimension"],
            min_confidence=0.8,
        )

        data = options.model_dump()

        assert data["annotation_types"] == ["dimension"]
        assert data["min_confidence"] == 0.8
        assert "vlm_model" in data

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "annotation_types": ["tolerance", "gdt"],
            "vlm_model": "gpt-4-vision-preview",
            "min_confidence": 0.9,
        }

        options = CADAnnotationOptions(**data)

        assert options.annotation_types == ["tolerance", "gdt"]
        assert options.min_confidence == 0.9

    def test_empty_annotation_types(self):
        """Test with empty annotation types list."""
        options = CADAnnotationOptions(annotation_types=[])

        assert options.annotation_types == []

    def test_empty_views(self):
        """Test with empty views list."""
        options = CADAnnotationOptions(views_to_process=[])

        assert options.views_to_process == []

    def test_kind_field(self):
        """Test that kind field is set correctly."""
        options = CADAnnotationOptions()

        assert options.kind == "cadling_experimental_annotation"

    def test_all_annotation_types(self):
        """Test with all supported annotation types."""
        all_types = [
            "dimension",
            "tolerance",
            "gdt",
            "surface_finish",
            "material",
            "welding",
            "note",
        ]

        options = CADAnnotationOptions(annotation_types=all_types)

        assert len(options.annotation_types) == len(all_types)
        for ann_type in all_types:
            assert ann_type in options.annotation_types

    def test_duplicate_annotation_types(self):
        """Test that duplicate annotation types are allowed."""
        options = CADAnnotationOptions(
            annotation_types=["dimension", "dimension", "tolerance"]
        )

        # Should preserve duplicates (user responsibility to dedupe if needed)
        assert len(options.annotation_types) == 3

    def test_duplicate_views(self):
        """Test that duplicate views are allowed."""
        options = CADAnnotationOptions(
            views_to_process=["front", "front", "top"]
        )

        assert len(options.views_to_process) == 3
