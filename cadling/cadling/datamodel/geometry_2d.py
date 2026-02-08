"""2D geometry data models for DXF and PDF drawing extraction.

This module provides Pydantic models representing 2D geometric primitives,
dimension annotations, sketch profiles, and sketch items. These models
are produced by the DXF and PDF backends and consumed by the
SketchGeometryExtractor enrichment model to produce tokenizer-compatible
command sequences.

Data flow:
    DXF file / PDF file
        → DXFBackend / PDFBackend
        → List[Primitive2D] + List[DimensionAnnotation]
        → SketchProfile (grouped by layer or page)
        → Sketch2DItem (wraps profiles into CADItem)
        → SketchGeometryExtractor (enrichment model)
        → CommandSequenceTokenizer-compatible dicts

Classes:
    Primitive2D: Base class for all 2D geometric primitives.
    Line2D: 2D line segment.
    Arc2D: 2D circular arc.
    Circle2D: 2D full circle.
    Polyline2D: 2D polyline (open or closed).
    Ellipse2D: 2D ellipse or elliptical arc.
    Spline2D: 2D spline curve.
    DimensionAnnotation: Extracted dimension callout.
    SketchProfile: Collection of connected 2D primitives forming a profile.
    Sketch2DItem: CADItem subclass for 2D sketch content.

Example:
    # Build a simple rectangular profile from DXF extraction
    lines = [
        Line2D(start=(0, 0), end=(100, 0)),
        Line2D(start=(100, 0), end=(100, 50)),
        Line2D(start=(100, 50), end=(0, 50)),
        Line2D(start=(0, 50), end=(0, 0)),
    ]
    profile = SketchProfile(
        profile_id="rect_0",
        primitives=lines,
        closed=True,
    )
    item = Sketch2DItem(
        item_type="sketch_2d",
        label=CADItemLabel(text="DXF Layer 0"),
        profiles=[profile],
    )
"""

from __future__ import annotations

import logging
import math
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from cadling.datamodel.base_models import CADItem, CADItemLabel

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PrimitiveType(str, Enum):
    """Type of 2D geometric primitive."""

    LINE = "line"
    ARC = "arc"
    CIRCLE = "circle"
    POLYLINE = "polyline"
    ELLIPSE = "ellipse"
    SPLINE = "spline"
    POINT = "point"


class DimensionType(str, Enum):
    """Type of dimension annotation."""

    LINEAR = "linear"
    ANGULAR = "angular"
    RADIAL = "radial"
    DIAMETER = "diameter"
    ORDINATE = "ordinate"
    ARC_LENGTH = "arc_length"


# ---------------------------------------------------------------------------
# 2D Primitives
# ---------------------------------------------------------------------------


class Primitive2D(BaseModel):
    """Base class for all 2D geometric primitives.

    Each primitive stores its type, optional layer/color metadata from
    the source file, and a confidence score indicating extraction quality.

    Attributes:
        primitive_type: Type discriminator (line, arc, circle, etc.).
        layer: Source layer name (from DXF) or page ID (from PDF).
        color: Entity color as (R, G, B) tuple, 0-255 range.
        line_weight: Line weight in mm (0.0 = default/hairline).
        confidence: Extraction confidence score (0.0 to 1.0).
        source_entity_id: Original entity handle/ID from source file.
    """

    primitive_type: PrimitiveType
    layer: str = "0"
    color: Optional[Tuple[int, int, int]] = None
    line_weight: float = 0.0
    confidence: float = 1.0
    source_entity_id: Optional[str] = None


class Line2D(Primitive2D):
    """2D line segment defined by start and end points.

    Attributes:
        start: Start point as (x, y) in mm.
        end: End point as (x, y) in mm.
    """

    primitive_type: PrimitiveType = PrimitiveType.LINE
    start: Tuple[float, float]
    end: Tuple[float, float]

    @property
    def length(self) -> float:
        """Length of the line segment."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return math.sqrt(dx * dx + dy * dy)

    @property
    def midpoint(self) -> Tuple[float, float]:
        """Midpoint of the line segment."""
        return (
            (self.start[0] + self.end[0]) / 2,
            (self.start[1] + self.end[1]) / 2,
        )

    @property
    def direction(self) -> Tuple[float, float]:
        """Unit direction vector from start to end."""
        length = self.length
        if length < 1e-10:
            return (0.0, 0.0)
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return (dx / length, dy / length)


class Arc2D(Primitive2D):
    """2D circular arc.

    Angles are measured counter-clockwise from the positive X-axis,
    in degrees, consistent with DXF conventions.

    Attributes:
        center: Center point as (x, y) in mm.
        radius: Arc radius in mm.
        start_angle: Start angle in degrees (CCW from +X).
        end_angle: End angle in degrees (CCW from +X).
    """

    primitive_type: PrimitiveType = PrimitiveType.ARC
    center: Tuple[float, float]
    radius: float
    start_angle: float  # degrees
    end_angle: float  # degrees

    @property
    def start_point(self) -> Tuple[float, float]:
        """Start point of the arc on its circle."""
        rad = math.radians(self.start_angle)
        return (
            self.center[0] + self.radius * math.cos(rad),
            self.center[1] + self.radius * math.sin(rad),
        )

    @property
    def end_point(self) -> Tuple[float, float]:
        """End point of the arc on its circle."""
        rad = math.radians(self.end_angle)
        return (
            self.center[0] + self.radius * math.cos(rad),
            self.center[1] + self.radius * math.sin(rad),
        )

    @property
    def sweep_angle(self) -> float:
        """Sweep angle in degrees (always positive, CCW)."""
        sweep = self.end_angle - self.start_angle
        if sweep <= 0:
            sweep += 360.0
        return sweep

    @property
    def arc_length(self) -> float:
        """Length of the arc."""
        return self.radius * math.radians(self.sweep_angle)


class Circle2D(Primitive2D):
    """2D full circle.

    Attributes:
        center: Center point as (x, y) in mm.
        radius: Circle radius in mm.
    """

    primitive_type: PrimitiveType = PrimitiveType.CIRCLE
    center: Tuple[float, float]
    radius: float

    @property
    def diameter(self) -> float:
        """Circle diameter."""
        return 2.0 * self.radius

    @property
    def circumference(self) -> float:
        """Circle circumference."""
        return 2.0 * math.pi * self.radius

    @property
    def area(self) -> float:
        """Circle area."""
        return math.pi * self.radius * self.radius


class Polyline2D(Primitive2D):
    """2D polyline (sequence of connected line segments).

    Can be open or closed. When closed, the last point connects back
    to the first point to form a polygon.

    Attributes:
        points: Ordered list of (x, y) vertices in mm.
        closed: Whether the polyline forms a closed polygon.
        bulges: Optional list of bulge values per segment. A bulge of 0
            means a straight segment; non-zero bulges indicate arc segments.
            Length should equal len(points) - 1 (or len(points) if closed).
    """

    primitive_type: PrimitiveType = PrimitiveType.POLYLINE
    points: List[Tuple[float, float]]
    closed: bool = False
    bulges: Optional[List[float]] = None

    @property
    def num_vertices(self) -> int:
        """Number of vertices."""
        return len(self.points)

    @property
    def num_segments(self) -> int:
        """Number of segments (edges)."""
        if self.closed:
            return len(self.points)
        return max(0, len(self.points) - 1)

    @property
    def perimeter(self) -> float:
        """Approximate perimeter (straight-line segments only)."""
        total = 0.0
        for i in range(len(self.points) - 1):
            dx = self.points[i + 1][0] - self.points[i][0]
            dy = self.points[i + 1][1] - self.points[i][1]
            total += math.sqrt(dx * dx + dy * dy)
        if self.closed and len(self.points) > 1:
            dx = self.points[0][0] - self.points[-1][0]
            dy = self.points[0][1] - self.points[-1][1]
            total += math.sqrt(dx * dx + dy * dy)
        return total

    def to_lines(self) -> List[Line2D]:
        """Decompose polyline into individual Line2D segments.

        Ignores bulge values — all segments become straight lines.
        For arc-aware decomposition, use the SketchGeometryExtractor.

        Returns:
            List of Line2D segments.
        """
        lines = []
        for i in range(len(self.points) - 1):
            lines.append(
                Line2D(
                    start=self.points[i],
                    end=self.points[i + 1],
                    layer=self.layer,
                    color=self.color,
                    line_weight=self.line_weight,
                    confidence=self.confidence,
                )
            )
        if self.closed and len(self.points) > 1:
            lines.append(
                Line2D(
                    start=self.points[-1],
                    end=self.points[0],
                    layer=self.layer,
                    color=self.color,
                    line_weight=self.line_weight,
                    confidence=self.confidence,
                )
            )
        return lines


class Ellipse2D(Primitive2D):
    """2D ellipse or elliptical arc.

    The ellipse is defined by its center, major axis endpoint (relative to
    center), and the ratio of minor axis to major axis length. For partial
    ellipses (arcs), start_param and end_param define the sweep.

    Attributes:
        center: Center point as (x, y) in mm.
        major_axis: Major axis endpoint relative to center, as (dx, dy).
        ratio: Ratio of minor axis length to major axis length (0 < ratio <= 1).
        start_param: Start parameter in radians (0 = full ellipse start).
        end_param: End parameter in radians (2*pi = full ellipse).
    """

    primitive_type: PrimitiveType = PrimitiveType.ELLIPSE
    center: Tuple[float, float]
    major_axis: Tuple[float, float]  # endpoint relative to center
    ratio: float  # minor/major ratio
    start_param: float = 0.0  # radians
    end_param: float = 6.283185307179586  # 2 * pi (full ellipse)

    @property
    def major_radius(self) -> float:
        """Length of the major semi-axis."""
        return math.sqrt(
            self.major_axis[0] ** 2 + self.major_axis[1] ** 2
        )

    @property
    def minor_radius(self) -> float:
        """Length of the minor semi-axis."""
        return self.major_radius * self.ratio

    @property
    def is_full(self) -> bool:
        """Whether this is a full ellipse (not an arc)."""
        return abs(self.end_param - self.start_param - 2 * math.pi) < 1e-6


class Spline2D(Primitive2D):
    """2D spline curve (B-spline or NURBS).

    Attributes:
        control_points: List of control points as (x, y) tuples.
        degree: Spline degree (typically 2 or 3).
        knots: Knot vector (list of floats).
        weights: Optional NURBS weights (one per control point).
        closed: Whether the spline is periodic (closed).
    """

    primitive_type: PrimitiveType = PrimitiveType.SPLINE
    control_points: List[Tuple[float, float]]
    degree: int = 3
    knots: List[float] = Field(default_factory=list)
    weights: Optional[List[float]] = None
    closed: bool = False

    @property
    def num_control_points(self) -> int:
        """Number of control points."""
        return len(self.control_points)

    @property
    def is_rational(self) -> bool:
        """Whether this is a rational spline (NURBS)."""
        return self.weights is not None and len(self.weights) > 0


# ---------------------------------------------------------------------------
# Dimension Annotations
# ---------------------------------------------------------------------------


class DimensionAnnotation(BaseModel):
    """Extracted dimension annotation from a technical drawing.

    Represents a dimension callout with its measured value, type, and
    attachment points on the drawing. These are extracted from DXF DIMENSION
    entities or from PDF text annotations near geometry.

    Attributes:
        dim_type: Type of dimension measurement.
        value: Measured value (in drawing units, typically mm).
        text: Raw text as displayed on the drawing (e.g., "∅12.5", "R5").
        unit: Unit of measurement.
        attachment_points: Points on the drawing where the dimension
            leaders attach, as list of (x, y) tuples.
        confidence: Extraction confidence score (0.0 to 1.0).
        layer: Source layer name.
    """

    dim_type: DimensionType
    value: float
    text: str = ""
    unit: str = "mm"
    attachment_points: List[Tuple[float, float]] = Field(default_factory=list)
    confidence: float = 1.0
    layer: str = "0"


# ---------------------------------------------------------------------------
# Sketch Profile & Item
# ---------------------------------------------------------------------------


class BoundingBox2D(BaseModel):
    """2D axis-aligned bounding box.

    Attributes:
        x_min: Minimum X coordinate.
        y_min: Minimum Y coordinate.
        x_max: Maximum X coordinate.
        y_max: Maximum Y coordinate.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def center(self) -> Tuple[float, float]:
        """Center point of the bounding box."""
        return (
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
        )

    @property
    def size(self) -> Tuple[float, float]:
        """Size as (width, height)."""
        return (self.x_max - self.x_min, self.y_max - self.y_min)

    @property
    def area(self) -> float:
        """Area of the bounding box."""
        w, h = self.size
        return w * h


class SketchProfile(BaseModel):
    """A collection of connected 2D primitives forming a sketch profile.

    A profile represents a single closed or open contour extracted from a
    DXF layer or PDF page. Multiple profiles may exist per layer/page
    (e.g., outer boundary + inner holes).

    Attributes:
        profile_id: Unique identifier for this profile.
        primitives: Ordered list of 2D primitives forming the contour.
        annotations: Dimension annotations associated with this profile.
        closed: Whether the profile forms a closed contour.
        bounds: 2D bounding box of the profile (computed if not provided).
        origin: Origin point for the profile coordinate system.
        area: Enclosed area (only meaningful for closed profiles).
        properties: Additional metadata from extraction.
    """

    profile_id: str
    primitives: List[Primitive2D] = Field(default_factory=list)
    annotations: List[DimensionAnnotation] = Field(default_factory=list)
    closed: bool = False
    bounds: Optional[BoundingBox2D] = None
    origin: Tuple[float, float] = (0.0, 0.0)
    area: Optional[float] = None
    properties: Dict[str, Any] = Field(default_factory=dict)

    def compute_bounds(self) -> BoundingBox2D:
        """Compute the bounding box from all primitives.

        Iterates all primitives and collects their defining points
        to compute the axis-aligned bounding box.

        Returns:
            Computed BoundingBox2D.
        """
        all_points: List[Tuple[float, float]] = []

        for prim in self.primitives:
            if isinstance(prim, Line2D):
                all_points.extend([prim.start, prim.end])
            elif isinstance(prim, Arc2D):
                all_points.append(prim.start_point)
                all_points.append(prim.end_point)
                # Include cardinal points within the arc sweep
                all_points.append(prim.center)
            elif isinstance(prim, Circle2D):
                r = prim.radius
                c = prim.center
                all_points.extend([
                    (c[0] - r, c[1] - r),
                    (c[0] + r, c[1] + r),
                ])
            elif isinstance(prim, Polyline2D):
                all_points.extend(prim.points)
            elif isinstance(prim, Ellipse2D):
                mr = prim.major_radius
                c = prim.center
                all_points.extend([
                    (c[0] - mr, c[1] - mr),
                    (c[0] + mr, c[1] + mr),
                ])
            elif isinstance(prim, Spline2D):
                all_points.extend(prim.control_points)

        if not all_points:
            return BoundingBox2D(x_min=0, y_min=0, x_max=0, y_max=0)

        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]

        bbox = BoundingBox2D(
            x_min=min(xs),
            y_min=min(ys),
            x_max=max(xs),
            y_max=max(ys),
        )
        self.bounds = bbox
        return bbox


class Sketch2DItem(CADItem):
    """CADItem subclass for 2D sketch/drawing content.

    Produced by DXF and PDF backends. Contains one or more SketchProfiles
    representing the 2D geometry extracted from the source file.

    Attributes:
        profiles: List of sketch profiles (one per layer or page region).
        source_page: Page number (for multi-page PDFs).
        source_layer: Layer name (for DXF files).
        total_primitives: Total count of primitives across all profiles.
        total_annotations: Total count of dimension annotations.

    Example:
        item = Sketch2DItem(
            item_type="sketch_2d",
            label=CADItemLabel(text="Layer 0 - Main Profile"),
            profiles=[outer_profile, hole_profile],
            source_layer="0",
        )
    """

    item_type: str = "sketch_2d"
    profiles: List[SketchProfile] = Field(default_factory=list)
    source_page: Optional[int] = None
    source_layer: Optional[str] = None

    @property
    def total_primitives(self) -> int:
        """Total number of primitives across all profiles."""
        return sum(len(p.primitives) for p in self.profiles)

    @property
    def total_annotations(self) -> int:
        """Total number of dimension annotations across all profiles."""
        return sum(len(p.annotations) for p in self.profiles)

    @property
    def all_primitives(self) -> List[Primitive2D]:
        """Flat list of all primitives from all profiles."""
        result = []
        for p in self.profiles:
            result.extend(p.primitives)
        return result

    @property
    def all_annotations(self) -> List[DimensionAnnotation]:
        """Flat list of all annotations from all profiles."""
        result = []
        for p in self.profiles:
            result.extend(p.annotations)
        return result

    def add_profile(self, profile: SketchProfile) -> None:
        """Add a sketch profile to this item.

        Args:
            profile: SketchProfile to add.
        """
        self.profiles.append(profile)
        _log.debug(
            "Added profile %s (%d primitives) to %s",
            profile.profile_id,
            len(profile.primitives),
            self.label.text if self.label else "unnamed",
        )

    def to_geotoken_commands(self) -> List[Dict[str, Any]]:
        """Return the command sequence in GeoToken-compatible format.

        Retrieves the command sequence written by SketchGeometryExtractor
        from this item's properties. Each command is a dict with "type"
        (e.g., "SOL", "LINE", "ARC", "CIRCLE", "EOS") and "params"
        (a 16-element float list matching GeoToken's parameter masks).

        This list can be passed directly to
        ``CommandSequenceTokenizer.tokenize()`` with no transformation.

        Returns:
            List of command dicts, or empty list if extraction hasn't run.

        Example:
            tokenizer = CommandSequenceTokenizer()
            token_seq = tokenizer.tokenize(sketch_item.to_geotoken_commands())
        """
        return self.properties.get("command_sequence", [])

    def to_geotoken_constraints(self) -> List[Dict[str, Any]]:
        """Return geometric constraints in GeoToken-compatible format.

        Retrieves the constraints detected by SketchGeometryExtractor and
        reformats them for GeoToken's ``ConstraintToken`` consumption.

        Each returned dict contains:
            - type: Constraint type string (e.g., "PARALLEL", "TANGENT")
            - source_index: Index of first entity in the command sequence
            - target_index: Index of second entity
            - value: Optional quantized value (None from detection)

        Returns:
            List of constraint dicts, or empty list if extraction hasn't run.

        Example:
            constraints = sketch_item.to_geotoken_constraints()
            constraint_tokens = tokenizer.parse_constraints(constraints)
        """
        raw = self.properties.get("geometric_constraints", [])
        return [
            {
                "type": c.get("type", ""),
                "source_index": c.get("entity_a", 0),
                "target_index": c.get("entity_b", 0),
                "value": None,
            }
            for c in raw
        ]


__all__ = [
    # Enums
    "PrimitiveType",
    "DimensionType",
    # Primitives
    "Primitive2D",
    "Line2D",
    "Arc2D",
    "Circle2D",
    "Polyline2D",
    "Ellipse2D",
    "Spline2D",
    # Annotations
    "DimensionAnnotation",
    # Profiles & Items
    "BoundingBox2D",
    "SketchProfile",
    "Sketch2DItem",
]
