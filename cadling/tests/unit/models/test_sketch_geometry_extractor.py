"""Comprehensive unit tests for SketchGeometryExtractor enrichment model.

Tests cover:
1. __call__() - Main enrichment interface with properties and provenance
2. _profiles_to_commands() - Full profile → command sequence conversion
3. _primitive_to_commands() - Individual primitive type conversion
4. _pad_params() - Parameter padding to fixed length
5. _order_primitives() - Endpoint connectivity ordering
6. _reverse_primitive() - Primitive reversal (start/end swap)
7. _bulge_to_arc_command() - DXF bulge → arc conversion
8. _detect_constraints() - Geometric constraint detection
9. Helper methods - Point distances, angles, centers, radii

Uses class-based organization with fixtures for reusable test data.
"""

import math
import pytest
from datetime import datetime

from cadling.models.segmentation.sketch_geometry_extractor import SketchGeometryExtractor
from cadling.datamodel.base_models import CADItem, CADItemLabel, CADlingDocument
from cadling.datamodel.geometry_2d import (
    Line2D,
    Arc2D,
    Circle2D,
    Polyline2D,
    Ellipse2D,
    Spline2D,
    Sketch2DItem,
    SketchProfile,
    PrimitiveType,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def extractor():
    """Create a SketchGeometryExtractor with default tolerances."""
    return SketchGeometryExtractor()


@pytest.fixture
def simple_doc():
    """Create a simple CADlingDocument for testing."""
    return CADlingDocument(name="test_doc")


@pytest.fixture
def square_profile():
    """A square profile made of 4 connected lines (0,0) -> (10,0) -> (10,10) -> (0,10) -> (0,0)."""
    lines = [
        Line2D(start=(0, 0), end=(10, 0)),
        Line2D(start=(10, 0), end=(10, 10)),
        Line2D(start=(10, 10), end=(0, 10)),
        Line2D(start=(0, 10), end=(0, 0)),
    ]
    return SketchProfile(profile_id="square", primitives=lines, closed=True)


@pytest.fixture
def single_line_profile():
    """A profile with a single line segment."""
    line = Line2D(start=(0, 0), end=(5, 5))
    return SketchProfile(profile_id="line", primitives=[line])


@pytest.fixture
def circle_profile():
    """A profile with a single circle."""
    circle = Circle2D(center=(5, 5), radius=2.5)
    return SketchProfile(profile_id="circle", primitives=[circle])


@pytest.fixture
def arc_profile():
    """A profile with a single arc."""
    arc = Arc2D(center=(0, 0), radius=5.0, start_angle=0.0, end_angle=90.0)
    return SketchProfile(profile_id="arc", primitives=[arc])


@pytest.fixture
def polyline_no_bulges():
    """A polyline without bulge values (all straight segments)."""
    polyline = Polyline2D(
        points=[(0, 0), (5, 0), (5, 5), (0, 5), (0, 0)],
        closed=True,
        bulges=None,
    )
    return SketchProfile(profile_id="polyline_straight", primitives=[polyline])


@pytest.fixture
def polyline_with_bulges():
    """A polyline with some bulge values (arc segments)."""
    polyline = Polyline2D(
        points=[(0, 0), (5, 0), (5, 5), (0, 5)],
        closed=False,
        bulges=[0.5, 0.0, -0.3],  # First segment is arc, second is line, third is arc
    )
    return SketchProfile(profile_id="polyline_bulge", primitives=[polyline])


@pytest.fixture
def ellipse_profile():
    """A profile with an ellipse (nearly circular ratio ≈ 1.0)."""
    ellipse = Ellipse2D(
        center=(5, 5),
        major_axis=(2.0, 0.0),
        ratio=0.95,
    )
    return SketchProfile(profile_id="ellipse", primitives=[ellipse])


@pytest.fixture
def ellipse_general_profile():
    """A profile with a general ellipse (ratio << 1.0)."""
    ellipse = Ellipse2D(
        center=(5, 5),
        major_axis=(4.0, 0.0),
        ratio=0.5,  # Ratio is 0.5, not close to 1.0
    )
    return SketchProfile(profile_id="ellipse_general", primitives=[ellipse])


@pytest.fixture
def spline_profile():
    """A profile with a spline (open)."""
    spline = Spline2D(
        control_points=[(0, 0), (2, 5), (5, 3), (8, 8)],
        closed=False,
    )
    return SketchProfile(profile_id="spline_open", primitives=[spline])


@pytest.fixture
def spline_closed_profile():
    """A profile with a closed spline."""
    spline = Spline2D(
        control_points=[(0, 0), (2, 5), (5, 3), (8, 8)],
        closed=True,
    )
    return SketchProfile(profile_id="spline_closed", primitives=[spline])


@pytest.fixture
def mixed_profile():
    """A profile with mixed primitive types."""
    primitives = [
        Line2D(start=(0, 0), end=(5, 0)),
        Arc2D(center=(5, 0), radius=2.5, start_angle=180.0, end_angle=270.0),
        Line2D(start=(5, -2.5), end=(0, -2.5)),
    ]
    return SketchProfile(profile_id="mixed", primitives=primitives)


@pytest.fixture
def sketch_2d_item(square_profile):
    """A Sketch2DItem with a single profile."""
    return Sketch2DItem(
        item_type="sketch_2d",
        label=CADItemLabel(text="Test Sketch"),
        profiles=[square_profile],
    )


@pytest.fixture
def sketch_2d_multi_item(square_profile, circle_profile):
    """A Sketch2DItem with multiple profiles."""
    return Sketch2DItem(
        item_type="sketch_2d",
        label=CADItemLabel(text="Multi-Profile Sketch"),
        profiles=[square_profile, circle_profile],
    )


@pytest.fixture
def non_sketch_item():
    """A non-Sketch2DItem CADItem (should be skipped)."""
    return CADItem(
        item_type="other_type",
        label=CADItemLabel(text="Not a Sketch"),
    )


# =============================================================================
# Test __call__() - Main enrichment interface
# =============================================================================


class TestSketchGeometryExtractorCall:
    """Tests for the __call__() enrichment model interface."""

    def test_call_processes_sketch_2d_item(self, extractor, simple_doc, sketch_2d_item):
        """Test that __call__() processes Sketch2DItem and writes properties."""
        extractor(simple_doc, [sketch_2d_item])

        assert "command_sequence" in sketch_2d_item.properties
        assert "geometric_constraints" in sketch_2d_item.properties
        assert "extraction_method" in sketch_2d_item.properties
        assert "num_commands" in sketch_2d_item.properties
        assert "num_profiles" in sketch_2d_item.properties

        assert sketch_2d_item.properties["extraction_method"] == "structured_2d"
        assert sketch_2d_item.properties["num_profiles"] == 1
        assert isinstance(sketch_2d_item.properties["command_sequence"], list)
        assert isinstance(sketch_2d_item.properties["geometric_constraints"], list)

    def test_call_skips_non_sketch_items(self, extractor, simple_doc, non_sketch_item):
        """Test that non-Sketch2DItem items are skipped without error."""
        extractor(simple_doc, [non_sketch_item])

        # Non-sketch items should not have these properties
        assert "command_sequence" not in non_sketch_item.properties

    def test_call_adds_provenance(self, extractor, simple_doc, sketch_2d_item):
        """Test that provenance information is added to items."""
        extractor(simple_doc, [sketch_2d_item])

        assert len(sketch_2d_item.prov) > 0
        provenance = sketch_2d_item.prov[-1]
        assert provenance.component_name == "SketchGeometryExtractor"
        assert provenance.component_type == "enrichment_model"

    def test_call_handles_multiple_items(self, extractor, simple_doc, sketch_2d_item, non_sketch_item):
        """Test that __call__() processes batches with mixed item types."""
        items = [sketch_2d_item, non_sketch_item]
        extractor(simple_doc, items)

        # Sketch item should be processed
        assert "command_sequence" in sketch_2d_item.properties
        # Non-sketch item should not
        assert "command_sequence" not in non_sketch_item.properties

    def test_call_handles_empty_batch(self, extractor, simple_doc):
        """Test that __call__() handles empty item batch gracefully."""
        extractor(simple_doc, [])
        # Should not raise any exception

    def test_call_sets_num_commands(self, extractor, simple_doc, sketch_2d_item):
        """Test that num_commands matches the command_sequence length."""
        extractor(simple_doc, [sketch_2d_item])

        assert sketch_2d_item.properties["num_commands"] == len(
            sketch_2d_item.properties["command_sequence"]
        )

    def test_call_error_handling(self, extractor, simple_doc):
        """Test that errors during extraction are logged but don't stop batch processing."""
        # Create a sketch with a profile that has no primitives
        empty_profile = SketchProfile(profile_id="empty", primitives=[])
        sketch_with_empty = Sketch2DItem(
            item_type="sketch_2d",
            label=CADItemLabel(text="Empty Sketch"),
            profiles=[empty_profile],
        )

        # Should not raise exception
        extractor(simple_doc, [sketch_with_empty])


# =============================================================================
# Test _profiles_to_commands() - Full profile → command conversion
# =============================================================================


class TestProfileToCommands:
    """Tests for _profiles_to_commands() method."""

    def test_single_profile_creates_sol_and_eos(self, extractor, square_profile):
        """Test that command sequence starts with SOL and ends with EOS."""
        commands = extractor._profiles_to_commands([square_profile])

        assert len(commands) > 0
        assert commands[0]["type"] == "SOL"
        assert commands[-1]["type"] == "EOS"

    def test_empty_profile_list_returns_only_eos(self, extractor):
        """Test that empty profile list still returns EOS."""
        commands = extractor._profiles_to_commands([])

        assert len(commands) == 1
        assert commands[0]["type"] == "EOS"

    def test_profile_with_single_line(self, extractor, single_line_profile):
        """Test profile with a single line produces SOL + LINE + EOS."""
        commands = extractor._profiles_to_commands([single_line_profile])

        assert commands[0]["type"] == "SOL"
        assert commands[1]["type"] == "LINE"
        assert commands[2]["type"] == "EOS"
        assert len(commands) == 3

    def test_multiple_profiles_have_separate_sols(self, extractor, square_profile, circle_profile):
        """Test that each profile gets its own SOL."""
        commands = extractor._profiles_to_commands([square_profile, circle_profile])

        sol_count = sum(1 for cmd in commands if cmd["type"] == "SOL")
        assert sol_count == 2  # One SOL per profile

    def test_sol_origin_from_first_primitive(self, extractor):
        """Test that SOL origin point comes from first primitive's start."""
        line = Line2D(start=(7.5, 3.2), end=(10, 5))
        profile = SketchProfile(profile_id="test", primitives=[line])

        commands = extractor._profiles_to_commands([profile])
        sol_params = commands[0]["params"]

        assert pytest.approx(sol_params[0], abs=0.01) == 7.5
        assert pytest.approx(sol_params[1], abs=0.01) == 3.2

    def test_all_commands_have_16_params(self, extractor, square_profile):
        """Test that all commands have exactly 16 parameters."""
        commands = extractor._profiles_to_commands([square_profile])

        for cmd in commands:
            assert len(cmd["params"]) == 16

    def test_profile_with_no_primitives_skipped(self, extractor):
        """Test that profiles with no primitives are skipped in sequence."""
        empty_profile = SketchProfile(profile_id="empty", primitives=[])
        line_profile = SketchProfile(
            profile_id="line",
            primitives=[Line2D(start=(0, 0), end=(5, 5))],
        )

        commands = extractor._profiles_to_commands([empty_profile, line_profile])

        # Should have: SOL (from line_profile) + LINE + EOS
        assert len(commands) == 3
        assert commands[0]["type"] == "SOL"


# =============================================================================
# Test _primitive_to_commands() - Individual primitive conversion
# =============================================================================


class TestPrimitiveToCommands:
    """Tests for _primitive_to_commands() method."""

    def test_line_2d_to_line_command(self, extractor):
        """Test Line2D → LINE command conversion.

        GeoToken compact format: [x1, y1, x2, y2, 0, 0, ...]
        """
        line = Line2D(start=(1, 2), end=(3, 4))
        commands = extractor._primitive_to_commands(line)

        assert len(commands) == 1
        assert commands[0]["type"] == "LINE"
        params = commands[0]["params"]
        assert pytest.approx(params[0], abs=0.01) == 1   # x1
        assert pytest.approx(params[1], abs=0.01) == 2   # y1
        assert pytest.approx(params[2], abs=0.01) == 3   # x2
        assert pytest.approx(params[3], abs=0.01) == 4   # y2

    def test_arc_2d_to_arc_command(self, extractor):
        """Test Arc2D → ARC command in 3-point format.

        GeoToken 3-point format: [x_start, y_start, x_mid, y_mid, x_end, y_end, 0, ...]
        Arc: center=(5,5), r=3, 0°→90°
          start = (5+3*cos0, 5+3*sin0) = (8, 5)
          mid   = (5+3*cos45, 5+3*sin45) ≈ (7.12, 7.12)
          end   = (5+3*cos90, 5+3*sin90) = (5, 8)
        """
        arc = Arc2D(center=(5, 5), radius=3.0, start_angle=0, end_angle=90)
        commands = extractor._primitive_to_commands(arc)

        assert len(commands) == 1
        assert commands[0]["type"] == "ARC"
        params = commands[0]["params"]
        # Start point: (8, 5)
        assert pytest.approx(params[0], abs=0.01) == 8.0   # x_start
        assert pytest.approx(params[1], abs=0.01) == 5.0   # y_start
        # Mid point: (5 + 3*cos45°, 5 + 3*sin45°)
        import math
        mid_x = 5 + 3 * math.cos(math.radians(45))
        mid_y = 5 + 3 * math.sin(math.radians(45))
        assert pytest.approx(params[2], abs=0.01) == mid_x  # x_mid
        assert pytest.approx(params[3], abs=0.01) == mid_y  # y_mid
        # End point: (5, 8)
        assert pytest.approx(params[4], abs=0.01) == 5.0   # x_end
        assert pytest.approx(params[5], abs=0.01) == 8.0   # y_end

    def test_circle_2d_to_circle_command(self, extractor):
        """Test Circle2D → CIRCLE command conversion.

        GeoToken compact format: [cx, cy, r, 0, 0, ...]
        """
        circle = Circle2D(center=(2, 3), radius=1.5)
        commands = extractor._primitive_to_commands(circle)

        assert len(commands) == 1
        assert commands[0]["type"] == "CIRCLE"
        params = commands[0]["params"]
        assert pytest.approx(params[0], abs=0.01) == 2    # center x
        assert pytest.approx(params[1], abs=0.01) == 3    # center y
        assert pytest.approx(params[2], abs=0.01) == 1.5  # radius (at position 2, not 3)

    def test_polyline_no_bulges_to_lines(self, extractor):
        """Test Polyline2D without bulges → multiple LINE commands."""
        polyline = Polyline2D(
            points=[(0, 0), (5, 0), (5, 5)],
            closed=False,
            bulges=None,
        )
        commands = extractor._primitive_to_commands(polyline)

        assert len(commands) == 2  # Two segments
        assert all(cmd["type"] == "LINE" for cmd in commands)

    def test_polyline_with_bulges_to_lines_and_arcs(self, extractor):
        """Test Polyline2D with bulges → mix of LINE and ARC commands."""
        polyline = Polyline2D(
            points=[(0, 0), (5, 0), (5, 5)],
            closed=False,
            bulges=[0.5, 0.0],  # First is arc, second is line
        )
        commands = extractor._primitive_to_commands(polyline)

        assert len(commands) == 2
        assert commands[0]["type"] == "ARC"  # Bulge > 0
        assert commands[1]["type"] == "LINE"  # Bulge = 0

    def test_polyline_closed_includes_closing_segment(self, extractor):
        """Test that closed polyline includes closing segment back to start."""
        polyline = Polyline2D(
            points=[(0, 0), (5, 0), (5, 5)],
            closed=True,
            bulges=None,
        )
        commands = extractor._primitive_to_commands(polyline)

        # Should have 3 LINE commands: (0,0)→(5,0), (5,0)→(5,5), (5,5)→(0,0)
        assert len(commands) == 3
        assert all(cmd["type"] == "LINE" for cmd in commands)

    def test_ellipse_nearly_circular_to_circle(self, extractor):
        """Test Ellipse2D with ratio ≈ 1.0 → CIRCLE command."""
        ellipse = Ellipse2D(
            center=(5, 5),
            major_axis=(2.0, 0.0),
            ratio=0.96,  # Within 0.05 of 1.0 (abs(0.96-1.0) = 0.04 < 0.05)
        )
        commands = extractor._primitive_to_commands(ellipse)

        assert len(commands) == 1
        assert commands[0]["type"] == "CIRCLE"

    def test_ellipse_nearly_circular_partial_to_arc(self, extractor):
        """Test partial ellipse with ratio ≈ 1.0 → ARC command."""
        ellipse = Ellipse2D(
            center=(5, 5),
            major_axis=(2.0, 0.0),
            ratio=0.98,
            start_param=0.0,
            end_param=math.pi / 2,  # Quarter ellipse
        )
        commands = extractor._primitive_to_commands(ellipse)

        assert len(commands) == 1
        assert commands[0]["type"] == "ARC"

    def test_ellipse_general_to_lines(self, extractor):
        """Test general Ellipse2D (ratio << 1.0) → multiple LINE commands."""
        ellipse = Ellipse2D(
            center=(5, 5),
            major_axis=(4.0, 0.0),
            ratio=0.5,  # Not close to 1.0
        )
        commands = extractor._primitive_to_commands(ellipse)

        # Should decompose to 16 LINE segments
        assert all(cmd["type"] == "LINE" for cmd in commands)
        assert len(commands) == 16  # 16 segments for full ellipse

    def test_spline_to_lines_via_control_polygon(self, extractor):
        """Test Spline2D → multiple LINE commands (control polygon)."""
        spline = Spline2D(
            control_points=[(0, 0), (2, 5), (5, 3), (8, 8)],
            closed=False,
        )
        commands = extractor._primitive_to_commands(spline)

        # Should have 3 LINE commands (4 points → 3 segments)
        assert len(commands) == 3
        assert all(cmd["type"] == "LINE" for cmd in commands)

    def test_spline_closed_includes_closing_line(self, extractor):
        """Test closed spline includes closing segment from last to first control point."""
        spline = Spline2D(
            control_points=[(0, 0), (2, 5), (5, 3), (8, 8)],
            closed=True,
        )
        commands = extractor._primitive_to_commands(spline)

        # Should have 4 LINE commands (4 points → 3 segments + closing)
        assert len(commands) == 4
        assert all(cmd["type"] == "LINE" for cmd in commands)


# =============================================================================
# Test _pad_params() - Parameter padding
# =============================================================================


class TestPadParams:
    """Tests for _pad_params() static method."""

    def test_empty_list_padded_with_zeros(self, extractor):
        """Test that empty list is padded to 16 zeros."""
        result = extractor._pad_params([])
        assert len(result) == 16
        assert all(p == 0.0 for p in result)

    def test_partial_list_padded_to_16(self, extractor):
        """Test that partial list is padded to 16 with zeros."""
        result = extractor._pad_params([1.0, 2.0, 3.0])
        assert len(result) == 16
        assert result[:3] == [1.0, 2.0, 3.0]
        assert all(p == 0.0 for p in result[3:])

    def test_exact_16_unchanged(self, extractor):
        """Test that list of exactly 16 elements is unchanged."""
        params = list(range(16))
        result = extractor._pad_params([float(p) for p in params])
        assert len(result) == 16
        assert result == [float(p) for p in params]

    def test_over_16_truncated(self, extractor):
        """Test that list longer than 16 is truncated to 16."""
        params = list(range(20))
        result = extractor._pad_params([float(p) for p in params])
        assert len(result) == 16
        assert result == [float(p) for p in range(16)]

    def test_converts_to_float(self, extractor):
        """Test that all returned values are floats."""
        result = extractor._pad_params([1, 2, 3])
        assert all(isinstance(p, float) for p in result)


# =============================================================================
# Test _order_primitives() - Endpoint connectivity ordering
# =============================================================================


class TestOrderPrimitives:
    """Tests for _order_primitives() endpoint connectivity method."""

    def test_single_primitive_unchanged(self, extractor):
        """Test that single primitive is returned unchanged."""
        line = Line2D(start=(0, 0), end=(5, 5))
        result = extractor._order_primitives([line])
        assert len(result) == 1
        assert result[0] is line

    def test_empty_list_unchanged(self, extractor):
        """Test that empty list is returned unchanged."""
        result = extractor._order_primitives([])
        assert result == []

    def test_connected_chain_ordered(self, extractor):
        """Test that connected primitives are ordered by endpoint connectivity."""
        line1 = Line2D(start=(0, 0), end=(5, 0))
        line2 = Line2D(start=(5, 0), end=(5, 5))  # Starts where line1 ends
        line3 = Line2D(start=(5, 5), end=(0, 5))  # Starts where line2 ends

        # Pass in already-connected order to verify chain is maintained
        result = extractor._order_primitives([line1, line2, line3])

        # Greedy algorithm starts with first element, then connects by endpoints
        assert len(result) == 3
        assert result[0].start == (0, 0)
        assert result[0].end == (5, 0)
        assert result[1].start == (5, 0)
        assert result[2].start == (5, 5)

    def test_unconnected_primitives_appended(self, extractor):
        """Test that unconnected primitives are appended at end."""
        line1 = Line2D(start=(0, 0), end=(5, 0))
        line2 = Line2D(start=(10, 10), end=(15, 15))  # Not connected
        line3 = Line2D(start=(5, 0), end=(5, 5))

        result = extractor._order_primitives([line1, line3, line2])

        # line1 starts the chain, line3 connects (start matches line1 end),
        # line2 is unconnected and appended at the end
        assert len(result) == 3
        assert result[0].start == (0, 0)
        assert result[1].start == (5, 0)
        assert result[2].start == (10, 10)

    def test_reversal_for_endpoint_connectivity(self, extractor):
        """Test that primitives are reversed if endpoints match better in reverse."""
        line1 = Line2D(start=(0, 0), end=(5, 0))
        line2 = Line2D(start=(5, 5), end=(5, 0))  # Ends where line1 ends (reversed match)

        result = extractor._order_primitives([line1, line2])

        # line2 should be reversed to connect to line1
        # After reversal, line2 should go from (5, 0) to (5, 5)
        assert result[1].start[0] == pytest.approx(5.0, abs=0.01)
        assert result[1].start[1] == pytest.approx(0.0, abs=0.01)


# =============================================================================
# Test _reverse_primitive() - Primitive reversal
# =============================================================================


class TestReversePrimitive:
    """Tests for _reverse_primitive() method."""

    def test_line_2d_reversed(self, extractor):
        """Test that Line2D start and end are swapped."""
        line = Line2D(start=(1, 2), end=(3, 4))
        reversed_line = extractor._reverse_primitive(line)

        assert reversed_line.start == (3, 4)
        assert reversed_line.end == (1, 2)

    def test_arc_2d_angles_swapped(self, extractor):
        """Test that Arc2D start_angle and end_angle are swapped."""
        arc = Arc2D(center=(5, 5), radius=3, start_angle=0, end_angle=90)
        reversed_arc = extractor._reverse_primitive(arc)

        assert reversed_arc.start_angle == 90
        assert reversed_arc.end_angle == 0
        assert reversed_arc.center == (5, 5)
        assert reversed_arc.radius == 3

    def test_polyline_points_reversed(self, extractor):
        """Test that Polyline2D points are reversed."""
        polyline = Polyline2D(
            points=[(0, 0), (5, 0), (5, 5)],
            closed=False,
        )
        reversed_poly = extractor._reverse_primitive(polyline)

        assert reversed_poly.points == [(5, 5), (5, 0), (0, 0)]

    def test_polyline_bulges_negated_and_reversed(self, extractor):
        """Test that Polyline2D bulges are negated and list is reversed."""
        polyline = Polyline2D(
            points=[(0, 0), (5, 0), (5, 5)],
            closed=False,
            bulges=[0.5, 0.3],
        )
        reversed_poly = extractor._reverse_primitive(polyline)

        # Bulges should be negated and reversed
        assert reversed_poly.bulges == [-0.3, -0.5]

    def test_polyline_no_bulges_preserves_none(self, extractor):
        """Test that polyline without bulges stays as None after reversal."""
        polyline = Polyline2D(
            points=[(0, 0), (5, 0), (5, 5)],
            closed=False,
            bulges=None,
        )
        reversed_poly = extractor._reverse_primitive(polyline)

        assert reversed_poly.bulges is None

    def test_circle_unchanged(self, extractor):
        """Test that Circle2D is returned unchanged (no direction concept)."""
        circle = Circle2D(center=(5, 5), radius=2.5)
        result = extractor._reverse_primitive(circle)

        # Should return the same primitive (no reversal for circles)
        assert result is circle


# =============================================================================
# Test _bulge_to_arc_command() - DXF bulge conversion
# =============================================================================


class TestBulgeToArc:
    """Tests for _bulge_to_arc_command() method."""

    def test_positive_bulge_ccw_arc(self, extractor):
        """Test that positive bulge produces CCW arc."""
        p1 = (0, 0)
        p2 = (10, 0)
        bulge = 0.5

        cmd = extractor._bulge_to_arc_command(p1, p2, bulge)

        assert cmd["type"] == "ARC"
        assert "params" in cmd
        assert len(cmd["params"]) == 16

    def test_negative_bulge_cw_arc(self, extractor):
        """Test that negative bulge produces CW arc."""
        p1 = (0, 0)
        p2 = (10, 0)
        bulge = -0.5

        cmd = extractor._bulge_to_arc_command(p1, p2, bulge)

        assert cmd["type"] == "ARC"
        params = cmd["params"]
        # Center should be below the chord for negative bulge
        assert params is not None

    def test_degenerate_bulge_zero_chord_to_line(self, extractor):
        """Test that degenerate bulge (nearly zero chord) returns LINE command."""
        p1 = (0, 0)
        p2 = (1e-12, 1e-12)  # Nearly same point (chord < 1e-10 threshold)
        bulge = 0.5

        cmd = extractor._bulge_to_arc_command(p1, p2, bulge)

        assert cmd["type"] == "LINE"

    def test_bulge_parameters_correct_format(self, extractor):
        """Test that bulge conversion produces correctly formatted 3-point ARC.

        GeoToken 3-point format: [x_start, y_start, x_mid, y_mid, x_end, y_end, 0, ...]
        For bulge=1.0 from (0,0) to (10,0): this is a semicircle.
        The center is at (5,0), radius=5, arc goes from 180° through 270° to 0°.
        """
        p1 = (0, 0)
        p2 = (10, 0)
        bulge = 1.0

        cmd = extractor._bulge_to_arc_command(p1, p2, bulge)

        assert cmd["type"] == "ARC"
        params = cmd["params"]
        assert len(params) == 16
        # 3-point format: start, mid, end points as 6 floats
        # Start point should be near p1
        assert pytest.approx(params[0], abs=0.5) == 0.0   # x_start ≈ p1[0]
        assert pytest.approx(params[1], abs=0.5) == 0.0   # y_start ≈ p1[1]
        # Mid point at the angular midpoint of the arc
        assert pytest.approx(params[2], abs=0.5) == 5.0   # x_mid ≈ center of chord
        assert abs(params[3]) > 0  # y_mid is non-zero (off the chord)
        # End point should be near p2
        assert pytest.approx(params[4], abs=0.5) == 10.0  # x_end ≈ p2[0]
        assert pytest.approx(params[5], abs=0.5) == 0.0   # y_end ≈ p2[1]

    def test_zero_bulge_returns_line(self, extractor):
        """Test that zero bulge (degenerate arc) might return LINE."""
        p1 = (0, 0)
        p2 = (10, 0)
        bulge = 0.0

        # This could be handled specially or computed as a degenerate arc
        cmd = extractor._bulge_to_arc_command(p1, p2, bulge)
        assert cmd["type"] in ("LINE", "ARC")


# =============================================================================
# Test _detect_constraints() - Geometric constraint detection
# =============================================================================


class TestConstraintDetection:
    """Tests for _detect_constraints() method."""

    def test_parallel_lines_detected(self, extractor):
        """Test detection of parallel lines."""
        line1 = Line2D(start=(0, 0), end=(10, 0))
        line2 = Line2D(start=(0, 5), end=(10, 5))

        constraints = extractor._detect_constraints([line1, line2])

        parallel_constraints = [c for c in constraints if c["type"] == "PARALLEL"]
        assert len(parallel_constraints) > 0

    def test_perpendicular_lines_detected(self, extractor):
        """Test detection of perpendicular lines."""
        line1 = Line2D(start=(0, 0), end=(10, 0))
        line2 = Line2D(start=(5, 0), end=(5, 10))

        constraints = extractor._detect_constraints([line1, line2])

        perp_constraints = [c for c in constraints if c["type"] == "PERPENDICULAR"]
        assert len(perp_constraints) > 0

    def test_collinear_lines_detected(self, extractor):
        """Test detection of collinear (parallel + overlapping) lines."""
        line1 = Line2D(start=(0, 0), end=(10, 0))
        line2 = Line2D(start=(5, 0), end=(15, 0))  # Same support line, overlapping

        constraints = extractor._detect_constraints([line1, line2])

        collinear_constraints = [c for c in constraints if c["type"] == "COLLINEAR"]
        assert len(collinear_constraints) > 0

    def test_concentric_circles_detected(self, extractor):
        """Test detection of concentric circles."""
        circle1 = Circle2D(center=(5, 5), radius=2)
        circle2 = Circle2D(center=(5.01, 5.01), radius=3)  # Nearly same center

        constraints = extractor._detect_constraints([circle1, circle2])

        concentric_constraints = [c for c in constraints if c["type"] == "CONCENTRIC"]
        assert len(concentric_constraints) > 0

    def test_equal_radius_circles_detected(self, extractor):
        """Test detection of circles with equal radius."""
        circle1 = Circle2D(center=(0, 0), radius=5.0)
        circle2 = Circle2D(center=(10, 0), radius=5.01)  # Nearly same radius

        constraints = extractor._detect_constraints([circle1, circle2])

        radius_constraints = [c for c in constraints if c["type"] == "EQUAL_RADIUS"]
        assert len(radius_constraints) > 0

    def test_tangent_line_to_circle_detected(self, extractor):
        """Test detection of line tangent to circle."""
        circle = Circle2D(center=(5, 0), radius=1.0)
        # Horizontal line at distance 1.0 from center = tangent
        line = Line2D(start=(0, 1.0), end=(10, 1.0))

        constraints = extractor._detect_constraints([line, circle])

        tangent_constraints = [c for c in constraints if c["type"] == "TANGENT"]
        assert len(tangent_constraints) > 0

    def test_no_constraints_for_unrelated_primitives(self, extractor):
        """Test that unrelated primitives produce no constraints."""
        line = Line2D(start=(0, 0), end=(5, 0))
        circle = Circle2D(center=(100, 100), radius=1)

        constraints = extractor._detect_constraints([line, circle])

        # Should be very few or no constraints
        tangent_constraints = [c for c in constraints if c["type"] == "TANGENT"]
        assert len(tangent_constraints) == 0

    def test_constraint_has_confidence_score(self, extractor):
        """Test that detected constraints include confidence score."""
        line1 = Line2D(start=(0, 0), end=(10, 0))
        line2 = Line2D(start=(0, 5), end=(10, 5))

        constraints = extractor._detect_constraints([line1, line2])

        for constraint in constraints:
            assert "confidence" in constraint
            assert 0.0 <= constraint["confidence"] <= 1.0

    def test_constraint_has_entity_indices(self, extractor):
        """Test that constraints reference correct entity indices."""
        line1 = Line2D(start=(0, 0), end=(10, 0))
        line2 = Line2D(start=(0, 5), end=(10, 5))

        constraints = extractor._detect_constraints([line1, line2])

        for constraint in constraints:
            assert "entity_a" in constraint
            assert "entity_b" in constraint
            assert isinstance(constraint["entity_a"], int)
            assert isinstance(constraint["entity_b"], int)


# =============================================================================
# Test Helper Methods - Point/Line distances, angles, centers, radii
# =============================================================================


class TestHelperMethods:
    """Tests for helper methods used in constraint detection and ordering."""

    def test_point_distance(self, extractor):
        """Test Euclidean distance calculation between two points."""
        p1 = (0, 0)
        p2 = (3, 4)  # 3-4-5 triangle
        dist = extractor._point_distance(p1, p2)

        assert pytest.approx(dist, abs=0.01) == 5.0

    def test_point_distance_same_point(self, extractor):
        """Test distance from point to itself is zero."""
        p = (5, 5)
        dist = extractor._point_distance(p, p)

        assert pytest.approx(dist, abs=0.001) == 0.0

    def test_angle_between_parallel_lines(self, extractor):
        """Test angle between parallel lines is ~0 or ~180 degrees."""
        line1 = Line2D(start=(0, 0), end=(10, 0))
        line2 = Line2D(start=(0, 5), end=(10, 5))

        angle = extractor._angle_between_lines(line1, line2)

        assert pytest.approx(angle, abs=1.0) in (0.0, 180.0)

    def test_angle_between_perpendicular_lines(self, extractor):
        """Test angle between perpendicular lines is ~90 degrees."""
        line1 = Line2D(start=(0, 0), end=(10, 0))
        line2 = Line2D(start=(5, 0), end=(5, 10))

        angle = extractor._angle_between_lines(line1, line2)

        assert pytest.approx(angle, abs=1.0) == 90.0

    def test_line_to_line_distance_parallel(self, extractor):
        """Test distance between parallel lines."""
        line1 = Line2D(start=(0, 0), end=(10, 0))
        line2 = Line2D(start=(0, 3), end=(10, 3))  # 3 units apart

        dist = extractor._line_to_line_distance(line1, line2)

        assert pytest.approx(dist, abs=0.1) == 3.0

    def test_line_to_line_distance_non_parallel(self, extractor):
        """Test distance computation between non-parallel lines.

        Note: _line_to_line_distance always computes the perpendicular
        distance from b.start to line a's support line, regardless of
        whether the lines are parallel. It is only used for collinear
        checks between nearly-parallel lines.
        """
        line1 = Line2D(start=(0, 0), end=(10, 0))  # Horizontal at y=0
        line2 = Line2D(start=(5, -5), end=(5, 5))   # Vertical, b.start at y=-5

        dist = extractor._line_to_line_distance(line1, line2)

        # Perpendicular distance from (5, -5) to y=0 line is 5.0
        assert pytest.approx(dist, abs=0.1) == 5.0

    def test_point_to_line_distance(self, extractor):
        """Test distance from point to line support."""
        line = Line2D(start=(0, 0), end=(10, 0))  # Horizontal line at y=0
        point = (5, 3)  # 3 units above line

        dist = extractor._point_to_line_distance(point, line)

        assert pytest.approx(dist, abs=0.1) == 3.0

    def test_point_to_line_distance_on_line(self, extractor):
        """Test distance from point on line to line is 0."""
        line = Line2D(start=(0, 0), end=(10, 0))
        point = (5, 0)  # On the line

        dist = extractor._point_to_line_distance(point, line)

        assert pytest.approx(dist, abs=0.01) == 0.0

    def test_get_primitive_start_line(self, extractor):
        """Test getting start point from Line2D."""
        line = Line2D(start=(1, 2), end=(3, 4))
        start = extractor._get_primitive_start(line)

        assert start == (1, 2)

    def test_get_primitive_start_arc(self, extractor):
        """Test getting start point from Arc2D."""
        arc = Arc2D(center=(0, 0), radius=5, start_angle=0, end_angle=90)
        start = extractor._get_primitive_start(arc)

        # At angle 0, point is at (5, 0)
        assert pytest.approx(start[0], abs=0.01) == 5.0
        assert pytest.approx(start[1], abs=0.01) == 0.0

    def test_get_primitive_start_circle(self, extractor):
        """Test getting start point from Circle2D (rightmost point)."""
        circle = Circle2D(center=(5, 5), radius=2.0)
        start = extractor._get_primitive_start(circle)

        # Circle start is rightmost point: center + radius on x-axis
        assert pytest.approx(start[0], abs=0.01) == 7.0
        assert pytest.approx(start[1], abs=0.01) == 5.0

    def test_get_primitive_start_polyline(self, extractor):
        """Test getting start point from Polyline2D."""
        polyline = Polyline2D(points=[(1, 2), (3, 4), (5, 6)])
        start = extractor._get_primitive_start(polyline)

        assert start == (1, 2)

    def test_get_primitive_end_line(self, extractor):
        """Test getting end point from Line2D."""
        line = Line2D(start=(1, 2), end=(3, 4))
        end = extractor._get_primitive_end(line)

        assert end == (3, 4)

    def test_get_primitive_end_arc(self, extractor):
        """Test getting end point from Arc2D."""
        arc = Arc2D(center=(0, 0), radius=5, start_angle=0, end_angle=90)
        end = extractor._get_primitive_end(arc)

        # At angle 90, point is at (0, 5)
        assert pytest.approx(end[0], abs=0.01) == 0.0
        assert pytest.approx(end[1], abs=0.01) == 5.0

    def test_get_primitive_end_polyline_open(self, extractor):
        """Test getting end point from open Polyline2D."""
        polyline = Polyline2D(
            points=[(1, 2), (3, 4), (5, 6)],
            closed=False,
        )
        end = extractor._get_primitive_end(polyline)

        assert end == (5, 6)

    def test_get_primitive_end_polyline_closed(self, extractor):
        """Test getting end point from closed Polyline2D (returns first point)."""
        polyline = Polyline2D(
            points=[(1, 2), (3, 4), (5, 6)],
            closed=True,
        )
        end = extractor._get_primitive_end(polyline)

        # Closed polyline returns first point as end
        assert end == (1, 2)

    def test_get_center_circle(self, extractor):
        """Test getting center from Circle2D."""
        circle = Circle2D(center=(5, 5), radius=2)
        center = extractor._get_center(circle)

        assert center == (5, 5)

    def test_get_center_arc(self, extractor):
        """Test getting center from Arc2D."""
        arc = Arc2D(center=(3, 4), radius=5, start_angle=0, end_angle=90)
        center = extractor._get_center(arc)

        assert center == (3, 4)

    def test_get_center_line_returns_none(self, extractor):
        """Test that getting center from Line2D returns None."""
        line = Line2D(start=(0, 0), end=(5, 5))
        center = extractor._get_center(line)

        assert center is None

    def test_get_radius_circle(self, extractor):
        """Test getting radius from Circle2D."""
        circle = Circle2D(center=(5, 5), radius=2.5)
        radius = extractor._get_radius(circle)

        assert pytest.approx(radius, abs=0.01) == 2.5

    def test_get_radius_arc(self, extractor):
        """Test getting radius from Arc2D."""
        arc = Arc2D(center=(0, 0), radius=3.5, start_angle=0, end_angle=90)
        radius = extractor._get_radius(arc)

        assert pytest.approx(radius, abs=0.01) == 3.5

    def test_get_radius_line_returns_none(self, extractor):
        """Test that getting radius from Line2D returns None."""
        line = Line2D(start=(0, 0), end=(5, 5))
        radius = extractor._get_radius(line)

        assert radius is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple methods."""

    def test_full_extraction_simple_square(self, extractor, simple_doc, square_profile):
        """Test full extraction pipeline on simple square profile."""
        sketch = Sketch2DItem(
            item_type="sketch_2d",
            label=CADItemLabel(text="Square"),
            profiles=[square_profile],
        )

        extractor(simple_doc, [sketch])

        assert "command_sequence" in sketch.properties
        assert len(sketch.properties["command_sequence"]) > 0
        assert sketch.properties["command_sequence"][-1]["type"] == "EOS"

    def test_full_extraction_with_constraints(self, extractor, simple_doc):
        """Test that constraints are detected during full extraction."""
        # Create a sketch with perpendicular lines (should detect PERPENDICULAR constraint)
        line1 = Line2D(start=(0, 0), end=(10, 0))
        line2 = Line2D(start=(10, 0), end=(10, 10))
        profile = SketchProfile(profile_id="perp", primitives=[line1, line2])

        sketch = Sketch2DItem(
            item_type="sketch_2d",
            label=CADItemLabel(text="Perpendicular"),
            profiles=[profile],
        )

        extractor(simple_doc, [sketch])

        constraints = sketch.properties["geometric_constraints"]
        perp_constraints = [c for c in constraints if c["type"] == "PERPENDICULAR"]
        assert len(perp_constraints) > 0

    def test_multi_profile_extraction(self, extractor, simple_doc, square_profile, circle_profile):
        """Test extraction of sketch with multiple profiles."""
        sketch = Sketch2DItem(
            item_type="sketch_2d",
            label=CADItemLabel(text="Multi-Profile"),
            profiles=[square_profile, circle_profile],
        )

        extractor(simple_doc, [sketch])

        assert sketch.properties["num_profiles"] == 2
        assert sketch.properties["num_commands"] > 0

    def test_extraction_with_mixed_primitives(self, extractor, simple_doc, mixed_profile):
        """Test extraction of profile with mixed primitive types."""
        sketch = Sketch2DItem(
            item_type="sketch_2d",
            label=CADItemLabel(text="Mixed"),
            profiles=[mixed_profile],
        )

        extractor(simple_doc, [sketch])

        commands = sketch.properties["command_sequence"]
        command_types = {cmd["type"] for cmd in commands}
        # Should have SOL, LINE, ARC, EOS
        assert "SOL" in command_types
        assert "EOS" in command_types
