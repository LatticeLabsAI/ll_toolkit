"""Multi-view rendering and fusion options.

This module provides configuration options for rendering multiple views of CAD
models and fusing the extracted information across views.

Classes:
    ViewConfig: Configuration for a single view.
    MultiViewOptions: Options for multi-view rendering and fusion.

Example:
    options = MultiViewOptions(
        views=[
            ViewConfig(name="front", azimuth=0, elevation=0),
            ViewConfig(name="top", azimuth=0, elevation=90),
        ],
        fusion_strategy="weighted_consensus"
    )
"""

from __future__ import annotations

from typing import ClassVar, List, Optional

from pydantic import BaseModel, Field

from cadling.datamodel.pipeline_options import PipelineOptions


class ViewConfig(BaseModel):
    """Configuration for a single rendered view.

    Attributes:
        name: Human-readable name for the view.
        azimuth: Horizontal rotation angle in degrees (0-360).
        elevation: Vertical angle in degrees (-90 to 90).
        distance: Camera distance multiplier (1.0 = auto).
        projection: Projection type ("perspective" or "orthographic").
    """

    name: str = Field(description="Human-readable name for the view")
    azimuth: float = Field(
        default=0.0,
        ge=-180.0,
        le=360.0,
        description="Horizontal rotation angle in degrees"
    )
    elevation: float = Field(
        default=0.0,
        ge=-90.0,
        le=90.0,
        description="Vertical angle in degrees"
    )
    distance: float = Field(
        default=1.0,
        gt=0.0,
        description="Camera distance multiplier (1.0 = auto-calculated)"
    )
    projection: str = Field(
        default="perspective",
        description="Projection type (perspective or orthographic)"
    )


def default_views() -> List[ViewConfig]:
    """Generate default standard engineering views.

    Returns:
        List of ViewConfig objects for front, top, right, and isometric views.
    """
    return [
        ViewConfig(name="front", azimuth=0, elevation=0),
        ViewConfig(name="top", azimuth=0, elevation=90),
        ViewConfig(name="right", azimuth=90, elevation=0),
        ViewConfig(name="isometric", azimuth=45, elevation=35.264),
        ViewConfig(name="back", azimuth=180, elevation=0),
        ViewConfig(name="bottom", azimuth=0, elevation=-90),
    ]


class MultiViewOptions(PipelineOptions):
    """Configuration options for multi-view rendering and fusion.

    This options class configures the rendering of multiple views of a CAD model
    and the strategy for fusing information extracted from each view.

    Attributes:
        kind: Discriminator for option type.
        views: List of view configurations to render.
        resolution: Resolution for rendered images (pixels).
        fusion_strategy: Strategy for fusing multi-view information.
        conflict_threshold: Confidence difference threshold for conflict detection.
        enable_lighting: Whether to enable realistic lighting in renders.
        background_color: Background color as RGB tuple (0-255).
        anti_aliasing: Anti-aliasing quality (0=off, 1-4=quality level).
        render_edges: Whether to render model edges.
        parallel_rendering: Whether to render views in parallel.
    """

    kind: ClassVar[str] = "cadling_experimental_multi_view"

    views: List[ViewConfig] = Field(
        default_factory=default_views,
        description="List of view configurations to render"
    )
    resolution: int = Field(
        default=2048,
        ge=512,
        le=4096,
        description="Resolution for rendered images in pixels"
    )
    fusion_strategy: str = Field(
        default="weighted_consensus",
        description="Strategy for fusing multi-view information (weighted_consensus, majority_vote, hierarchical)"
    )
    conflict_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence difference threshold for detecting conflicts between views"
    )
    enable_lighting: bool = Field(
        default=True,
        description="Whether to enable realistic lighting in renders"
    )
    background_color: Optional[tuple] = Field(
        default=(255, 255, 255),
        description="Background color as RGB tuple (0-255)"
    )
    anti_aliasing: int = Field(
        default=2,
        ge=0,
        le=4,
        description="Anti-aliasing quality (0=off, 1-4=quality level)"
    )
    render_edges: bool = Field(
        default=True,
        description="Whether to render model edges for better feature visibility"
    )
    parallel_rendering: bool = Field(
        default=True,
        description="Whether to render views in parallel for performance"
    )
