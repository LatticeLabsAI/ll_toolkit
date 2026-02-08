"""Unit tests for 2D geometry data models.

Tests the Pydantic models for 2D primitives, dimension annotations,
sketch profiles, and Sketch2DItem.
"""

from __future__ import annotations

import math

import pytest

from cadling.datamodel.geometry_2d import (
    Arc2D,
    BoundingBox2D,
    Circle2D,
    DimensionAnnotation,
    DimensionType,
    Ellipse2D,
    Line2D,
    Polyline2D,
    PrimitiveType,
    Sketch2DItem,
    SketchProfile,
    Spline2D,
)
from cadling.datamodel.base_models import CADItemLabel


# ---------------------------------------------------------------------------
# Line2D
# ---------------------------------------------------------------------------


class TestLine2D:
    """Tests for Line2D model."""

    def test_create_line(self):
        """Line2D can be created with start and end points."""
        line = Line2D(start=(0.0, 0.0), end=(10.0, 0.0))
        assert line.primitive_type == PrimitiveType.LINE
        assert line.start == (0.0, 0.0)
        assert line.end == (10.0, 0.0)

    def test_line_length(self):
        """Line2D computes correct length."""
        line = Line2D(start=(0.0, 0.0), end=(3.0, 4.0))
        assert line.length == pytest.approx(5.0)

    def test_line_midpoint(self):
        """Line2D computes correct midpoint."""
        line = Line2D(start=(0.0, 0.0), end=(10.0, 20.0))
        assert line.midpoint == (5.0, 10.0)

    def test_line_direction(self):
        """Line2D computes correct unit direction."""
        line = Line2D(start=(0.0, 0.0), end=(10.0, 0.0))
        dx, dy = line.direction
        assert dx == pytest.approx(1.0)
        assert dy == pytest.approx(0.0)

    def test_line_default_layer(self):
        """Line2D defaults to layer '0'."""
        line = Line2D(start=(0.0, 0.0), end=(1.0, 1.0))
        assert line.layer == "0"

    def test_line_with_metadata(self):
        """Line2D stores layer, color, and confidence."""
        line = Line2D(
            start=(0.0, 0.0),
            end=(1.0, 1.0),
            layer="OUTLINE",
            color=(255, 0, 0),
            confidence=0.95,
            source_entity_id="A1",
        )
        assert line.layer == "OUTLINE"
        assert line.color == (255, 0, 0)
        assert line.confidence == 0.95
        assert line.source_entity_id == "A1"


# ---------------------------------------------------------------------------
# Arc2D
# ---------------------------------------------------------------------------


class TestArc2D:
    """Tests for Arc2D model."""

    def test_create_arc(self):
        """Arc2D can be created with center, radius, and angles."""
        arc = Arc2D(center=(0.0, 0.0), radius=10.0, start_angle=0.0, end_angle=90.0)
        assert arc.primitive_type == PrimitiveType.ARC
        assert arc.radius == 10.0

    def test_arc_sweep_angle(self):
        """Arc2D computes correct sweep angle."""
        arc = Arc2D(center=(0.0, 0.0), radius=5.0, start_angle=30.0, end_angle=120.0)
        assert arc.sweep_angle == pytest.approx(90.0)

    def test_arc_sweep_wraps(self):
        """Arc2D sweep wraps correctly past 360."""
        arc = Arc2D(center=(0.0, 0.0), radius=5.0, start_angle=350.0, end_angle=10.0)
        assert arc.sweep_angle == pytest.approx(20.0)

    def test_arc_start_point(self):
        """Arc2D computes correct start point."""
        arc = Arc2D(center=(0.0, 0.0), radius=10.0, start_angle=0.0, end_angle=90.0)
        sx, sy = arc.start_point
        assert sx == pytest.approx(10.0)
        assert sy == pytest.approx(0.0)

    def test_arc_end_point(self):
        """Arc2D computes correct end point."""
        arc = Arc2D(center=(0.0, 0.0), radius=10.0, start_angle=0.0, end_angle=90.0)
        ex, ey = arc.end_point
        assert ex == pytest.approx(0.0, abs=1e-9)
        assert ey == pytest.approx(10.0)

    def test_arc_length(self):
        """Arc2D computes correct arc length."""
        arc = Arc2D(center=(0.0, 0.0), radius=10.0, start_angle=0.0, end_angle=180.0)
        assert arc.arc_length == pytest.approx(10.0 * math.pi)


# ---------------------------------------------------------------------------
# Circle2D
# ---------------------------------------------------------------------------


class TestCircle2D:
    """Tests for Circle2D model."""

    def test_create_circle(self):
        """Circle2D can be created with center and radius."""
        circle = Circle2D(center=(5.0, 5.0), radius=10.0)
        assert circle.primitive_type == PrimitiveType.CIRCLE
        assert circle.center == (5.0, 5.0)
        assert circle.radius == 10.0

    def test_circle_diameter(self):
        """Circle2D computes correct diameter."""
        circle = Circle2D(center=(0.0, 0.0), radius=7.5)
        assert circle.diameter == pytest.approx(15.0)

    def test_circle_area(self):
        """Circle2D computes correct area."""
        circle = Circle2D(center=(0.0, 0.0), radius=10.0)
        assert circle.area == pytest.approx(math.pi * 100.0)


# ---------------------------------------------------------------------------
# Polyline2D
# ---------------------------------------------------------------------------


class TestPolyline2D:
    """Tests for Polyline2D model."""

    def test_create_polyline(self):
        """Polyline2D can be created with points."""
        pl = Polyline2D(points=[(0, 0), (10, 0), (10, 10)])
        assert pl.num_vertices == 3
        assert pl.num_segments == 2
        assert not pl.closed

    def test_closed_polyline(self):
        """Closed Polyline2D has correct segment count."""
        pl = Polyline2D(points=[(0, 0), (10, 0), (10, 10), (0, 10)], closed=True)
        assert pl.num_segments == 4

    def test_polyline_perimeter(self):
        """Polyline2D computes correct perimeter."""
        pl = Polyline2D(
            points=[(0, 0), (10, 0), (10, 10), (0, 10)], closed=True
        )
        assert pl.perimeter == pytest.approx(40.0)

    def test_to_lines(self):
        """Polyline2D decomposes to correct number of Line2D."""
        pl = Polyline2D(points=[(0, 0), (10, 0), (10, 10)], closed=False)
        lines = pl.to_lines()
        assert len(lines) == 2
        assert all(isinstance(l, Line2D) for l in lines)

    def test_closed_to_lines(self):
        """Closed Polyline2D includes closing segment."""
        pl = Polyline2D(points=[(0, 0), (10, 0), (10, 10)], closed=True)
        lines = pl.to_lines()
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# Ellipse2D
# ---------------------------------------------------------------------------


class TestEllipse2D:
    """Tests for Ellipse2D model."""

    def test_create_ellipse(self):
        """Ellipse2D can be created."""
        ell = Ellipse2D(center=(0, 0), major_axis=(10, 0), ratio=0.5)
        assert ell.major_radius == pytest.approx(10.0)
        assert ell.minor_radius == pytest.approx(5.0)
        assert ell.is_full

    def test_partial_ellipse(self):
        """Partial Ellipse2D is not full."""
        ell = Ellipse2D(
            center=(0, 0), major_axis=(10, 0), ratio=0.5,
            start_param=0.0, end_param=math.pi,
        )
        assert not ell.is_full


# ---------------------------------------------------------------------------
# DimensionAnnotation
# ---------------------------------------------------------------------------


class TestDimensionAnnotation:
    """Tests for DimensionAnnotation model."""

    def test_create_dimension(self):
        """DimensionAnnotation can be created."""
        dim = DimensionAnnotation(
            dim_type=DimensionType.LINEAR,
            value=25.4,
            text="25.4",
        )
        assert dim.dim_type == DimensionType.LINEAR
        assert dim.value == 25.4

    def test_dimension_with_points(self):
        """DimensionAnnotation stores attachment points."""
        dim = DimensionAnnotation(
            dim_type=DimensionType.DIAMETER,
            value=12.0,
            text="∅12",
            attachment_points=[(10, 20), (30, 40)],
        )
        assert len(dim.attachment_points) == 2


# ---------------------------------------------------------------------------
# SketchProfile
# ---------------------------------------------------------------------------


class TestSketchProfile:
    """Tests for SketchProfile model."""

    def test_create_profile(self):
        """SketchProfile can hold primitives and annotations."""
        line = Line2D(start=(0, 0), end=(10, 0))
        dim = DimensionAnnotation(dim_type=DimensionType.LINEAR, value=10.0)
        profile = SketchProfile(
            profile_id="test",
            primitives=[line],
            annotations=[dim],
        )
        assert len(profile.primitives) == 1
        assert len(profile.annotations) == 1

    def test_compute_bounds(self):
        """SketchProfile computes correct bounding box."""
        lines = [
            Line2D(start=(0, 0), end=(10, 0)),
            Line2D(start=(10, 0), end=(10, 5)),
        ]
        profile = SketchProfile(profile_id="test", primitives=lines)
        bbox = profile.compute_bounds()
        assert isinstance(bbox, BoundingBox2D)
        assert bbox.x_min == pytest.approx(0.0)
        assert bbox.x_max == pytest.approx(10.0)
        assert bbox.y_min == pytest.approx(0.0)
        assert bbox.y_max == pytest.approx(5.0)

    def test_empty_profile_bounds(self):
        """Empty profile returns zero bounding box."""
        profile = SketchProfile(profile_id="empty")
        bbox = profile.compute_bounds()
        assert bbox.area == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Sketch2DItem
# ---------------------------------------------------------------------------


class TestSketch2DItem:
    """Tests for Sketch2DItem model."""

    def test_create_sketch_item(self):
        """Sketch2DItem wraps profiles correctly."""
        line = Line2D(start=(0, 0), end=(10, 0))
        profile = SketchProfile(profile_id="p1", primitives=[line])
        item = Sketch2DItem(
            label=CADItemLabel(text="Test"),
            profiles=[profile],
        )
        assert item.item_type == "sketch_2d"
        assert item.total_primitives == 1
        assert item.total_annotations == 0

    def test_add_profile(self):
        """Sketch2DItem.add_profile appends correctly."""
        item = Sketch2DItem(label=CADItemLabel(text="Test"))
        assert len(item.profiles) == 0

        line = Line2D(start=(0, 0), end=(5, 5))
        profile = SketchProfile(profile_id="new", primitives=[line])
        item.add_profile(profile)
        assert len(item.profiles) == 1
        assert item.total_primitives == 1

    def test_all_primitives(self):
        """Sketch2DItem.all_primitives flattens across profiles."""
        p1 = SketchProfile(
            profile_id="a",
            primitives=[Line2D(start=(0, 0), end=(1, 0))],
        )
        p2 = SketchProfile(
            profile_id="b",
            primitives=[
                Line2D(start=(0, 0), end=(0, 1)),
                Circle2D(center=(5, 5), radius=2),
            ],
        )
        item = Sketch2DItem(
            label=CADItemLabel(text="Test"),
            profiles=[p1, p2],
        )
        assert item.total_primitives == 3
        assert len(item.all_primitives) == 3
