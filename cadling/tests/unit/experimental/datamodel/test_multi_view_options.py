"""
Unit tests for MultiViewOptions and ViewConfig.

Tests cover:
- ViewConfig initialization and validation
- MultiViewOptions initialization with defaults
- View configuration
- Fusion strategy validation
- Rendering settings
"""

import pytest
from pydantic import ValidationError

from cadling.experimental.datamodel import MultiViewOptions, ViewConfig


class TestViewConfig:
    """Test ViewConfig pydantic model."""

    def test_default_initialization(self):
        """Test ViewConfig with minimal parameters."""
        view = ViewConfig(name="front", azimuth=0.0, elevation=0.0)

        assert view.name == "front"
        assert view.azimuth == 0.0
        assert view.elevation == 0.0
        assert view.distance == 1.0  # Default
        assert view.projection == "perspective"  # Default

    def test_custom_distance(self):
        """Test custom camera distance."""
        view = ViewConfig(name="far", azimuth=0.0, elevation=0.0, distance=5.0)

        assert view.distance == 5.0

    def test_custom_projection(self):
        """Test custom projection type."""
        view = ViewConfig(
            name="custom", azimuth=45.0, elevation=45.0, projection="orthographic"
        )

        assert view.projection == "orthographic"

    def test_standard_views(self):
        """Test standard engineering views."""
        front = ViewConfig(name="front", azimuth=0.0, elevation=0.0)
        top = ViewConfig(name="top", azimuth=0.0, elevation=90.0)
        right = ViewConfig(name="right", azimuth=90.0, elevation=0.0)
        iso = ViewConfig(name="isometric", azimuth=45.0, elevation=35.264)

        assert front.azimuth == 0.0
        assert top.elevation == 90.0
        assert right.azimuth == 90.0
        assert abs(iso.azimuth - 45.0) < 0.01
        assert abs(iso.elevation - 35.264) < 0.01

    def test_to_dict(self):
        """Test ViewConfig to dictionary."""
        view = ViewConfig(name="test", azimuth=30.0, elevation=60.0)
        data = view.model_dump()

        assert data["name"] == "test"
        assert data["azimuth"] == 30.0
        assert data["elevation"] == 60.0


class TestMultiViewOptions:
    """Test MultiViewOptions pydantic model."""

    def test_default_initialization(self):
        """Test initialization with default values."""
        options = MultiViewOptions()

        # Check default views are created
        assert len(options.views) == 6
        view_names = [v.name for v in options.views]
        assert "front" in view_names
        assert "top" in view_names
        assert "right" in view_names
        assert "back" in view_names
        assert "bottom" in view_names
        assert "isometric" in view_names

        # Check default settings
        assert options.resolution == 2048
        assert options.fusion_strategy == "weighted_consensus"
        assert options.conflict_threshold == 0.5
        assert options.enable_lighting is True
        assert options.render_edges is True
        assert options.parallel_rendering is True

    def test_custom_views(self):
        """Test custom view configuration."""
        custom_views = [
            ViewConfig(name="front", azimuth=0.0, elevation=0.0),
            ViewConfig(name="iso", azimuth=45.0, elevation=35.264),
        ]

        options = MultiViewOptions(views=custom_views)

        assert len(options.views) == 2
        assert options.views[0].name == "front"
        assert options.views[1].name == "iso"

    def test_fusion_strategies(self):
        """Test different fusion strategies."""
        # Weighted consensus
        options = MultiViewOptions(fusion_strategy="weighted_consensus")
        assert options.fusion_strategy == "weighted_consensus"

        # Majority vote
        options = MultiViewOptions(fusion_strategy="majority_vote")
        assert options.fusion_strategy == "majority_vote"

        # Hierarchical
        options = MultiViewOptions(fusion_strategy="hierarchical")
        assert options.fusion_strategy == "hierarchical"

    def test_invalid_fusion_strategy(self):
        """Test invalid fusion strategy is accepted (no validation on enum values)."""
        # Since fusion_strategy is a string with no enum validation,
        # invalid values are accepted as-is
        options = MultiViewOptions(fusion_strategy="invalid_strategy")
        assert options.fusion_strategy == "invalid_strategy"

    def test_resolution_validation(self):
        """Test resolution validation."""
        # Valid resolutions
        options = MultiViewOptions(resolution=1024)
        assert options.resolution == 1024

        options = MultiViewOptions(resolution=4096)
        assert options.resolution == 4096

        # Out of bounds
        with pytest.raises(ValidationError):
            MultiViewOptions(resolution=255)

        with pytest.raises(ValidationError):
            MultiViewOptions(resolution=8193)

    def test_conflict_threshold_validation(self):
        """Test conflict threshold validation."""
        # Valid thresholds
        options = MultiViewOptions(conflict_threshold=0.3)
        assert options.conflict_threshold == 0.3

        options = MultiViewOptions(conflict_threshold=0.9)
        assert options.conflict_threshold == 0.9

        # Out of bounds
        with pytest.raises(ValidationError):
            MultiViewOptions(conflict_threshold=-0.1)

        with pytest.raises(ValidationError):
            MultiViewOptions(conflict_threshold=1.1)

    def test_lighting_toggle(self):
        """Test lighting can be disabled."""
        options = MultiViewOptions(enable_lighting=False)
        assert options.enable_lighting is False

    def test_edge_rendering_toggle(self):
        """Test edge rendering can be disabled."""
        options = MultiViewOptions(render_edges=False)
        assert options.render_edges is False

    def test_parallel_rendering_toggle(self):
        """Test parallel rendering can be enabled."""
        options = MultiViewOptions(parallel_rendering=True)
        assert options.parallel_rendering is True

    def test_background_color(self):
        """Test custom background color."""
        options = MultiViewOptions(background_color=(255, 0, 0))

        assert options.background_color == (255, 0, 0)

    def test_background_color_validation(self):
        """Test background color component validation."""
        # Valid colors
        options = MultiViewOptions(background_color=(0, 0, 0))
        assert options.background_color == (0, 0, 0)

        options = MultiViewOptions(background_color=(255, 255, 255))
        assert options.background_color == (255, 255, 255)

        # Tuples are not validated in the model, they are accepted as-is
        options = MultiViewOptions(background_color=(-1, 0, 0))
        assert options.background_color == (-1, 0, 0)

        options = MultiViewOptions(background_color=(0, 256, 0))
        assert options.background_color == (0, 256, 0)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        options = MultiViewOptions(
            resolution=1024,
            fusion_strategy="majority_vote",
        )

        data = options.model_dump()

        assert data["resolution"] == 1024
        assert data["fusion_strategy"] == "majority_vote"
        assert "views" in data

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "views": [
                {"name": "front", "azimuth": 0.0, "elevation": 0.0},
                {"name": "top", "azimuth": 0.0, "elevation": 90.0},
            ],
            "resolution": 1024,
            "fusion_strategy": "hierarchical",
        }

        options = MultiViewOptions(**data)

        assert len(options.views) == 2
        assert options.resolution == 1024
        assert options.fusion_strategy == "hierarchical"

    def test_kind_field(self):
        """Test that kind field is set correctly."""
        options = MultiViewOptions()

        assert options.kind == "cadling_experimental_multi_view"

    def test_single_view(self):
        """Test with single view (edge case)."""
        single_view = [ViewConfig(name="front", azimuth=0.0, elevation=0.0)]

        options = MultiViewOptions(views=single_view)

        assert len(options.views) == 1

    def test_many_views(self):
        """Test with many views."""
        many_views = [
            ViewConfig(name=f"view_{i}", azimuth=float(i * 30), elevation=0.0)
            for i in range(12)
        ]

        options = MultiViewOptions(views=many_views)

        assert len(options.views) == 12
