"""Structured geometry extraction for 2D sketches.

This enrichment model transforms Primitive2D lists (from DXF/PDF backends) into
tokenizer-compatible command sequences that can be fed directly into the
CommandSequenceTokenizer for ML pipeline processing.

The extraction pipeline:
    1. Receives Sketch2DItem instances from DXF/PDF backends
    2. For each SketchProfile, orders primitives into connected chains
    3. Converts each primitive to a command dict: {"type": "LINE", "params": [...]}
    4. Inserts SOL (start-of-loop) and EOS (end-of-sequence) markers
    5. Detects geometric constraints (parallel, perpendicular, concentric, etc.)
    6. Writes results to item.properties for downstream tokenization

Command format (compatible with GeoToken CommandSequenceTokenizer masks):
    {"type": "SOL",    "params": [x, y, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    {"type": "LINE",   "params": [x1, y1, x2, y2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    {"type": "ARC",    "params": [x1, y1, x2, y2, x3, y3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    {"type": "CIRCLE", "params": [cx, cy, r, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    {"type": "EOS",    "params": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

GeoToken parameter mask alignment:
    - SOL:    mask[0,1] active       → [z_offset, rotation]
    - LINE:   mask[0,1,2,3] active   → [x1, y1, x2, y2]
    - ARC:    mask[0..5] active      → [x1, y1, x2, y2, x3, y3] (start, mid, end)
    - CIRCLE: mask[0,1,2] active     → [cx, cy, r]
    - EOS:    no active params

Classes:
    SketchGeometryExtractor: EnrichmentModel for converting 2D sketches
        to tokenizer-ready command sequences.

Example:
    extractor = SketchGeometryExtractor()
    extractor(doc, [sketch_item_1, sketch_item_2])
    # After calling, each item has:
    #   item.properties["command_sequence"] = [{"type": ..., "params": [...]}, ...]
    #   item.properties["geometric_constraints"] = [...]
    #   item.properties["extraction_method"] = "structured_2d"
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from cadling.datamodel.base_models import CADItem, CADlingDocument
from cadling.datamodel.geometry_2d import (
    Arc2D,
    Circle2D,
    Ellipse2D,
    Line2D,
    Polyline2D,
    Primitive2D,
    PrimitiveType,
    Sketch2DItem,
    SketchProfile,
    Spline2D,
)
from cadling.models.base_model import EnrichmentModel

_log = logging.getLogger(__name__)

# Number of parameters per command (matches CommandSequenceTokenizer)
_NUM_PARAMS = 16

# Distance tolerance for endpoint connectivity (mm)
_CONNECT_TOLERANCE = 0.1

# Angle tolerance for constraint detection (degrees)
_ANGLE_TOLERANCE = 2.0

# Distance tolerance for concentric detection (mm)
_CONCENTRIC_TOLERANCE = 0.5


class SketchGeometryExtractor(EnrichmentModel):
    """Enrichment model that converts 2D sketch geometry to command sequences.

    This model processes Sketch2DItem instances produced by the DXF and PDF
    backends and converts their Primitive2D geometry into the command format
    expected by the CommandSequenceTokenizer:

        [{"type": "SOL", "params": [...]}, {"type": "LINE", "params": [...]}, ...]

    Additionally, it detects geometric constraints between primitives
    (parallel lines, perpendicular edges, concentric circles, etc.) and
    stores them as metadata for downstream use by code generation prompts.

    The extractor writes the following properties to each item:
        - ``command_sequence``: List of command dicts ready for tokenization.
        - ``geometric_constraints``: List of detected constraint dicts.
        - ``extraction_method``: Always ``"structured_2d"``.
        - ``num_commands``: Total number of commands in the sequence.
        - ``num_profiles``: Number of profiles processed.

    Attributes:
        connect_tolerance: Distance tolerance for connecting primitives (mm).
        angle_tolerance: Angle tolerance for parallel/perpendicular detection (degrees).
        concentric_tolerance: Distance tolerance for concentric detection (mm).
    """

    def __init__(
        self,
        connect_tolerance: float = _CONNECT_TOLERANCE,
        angle_tolerance: float = _ANGLE_TOLERANCE,
        concentric_tolerance: float = _CONCENTRIC_TOLERANCE,
    ):
        """Initialize the sketch geometry extractor.

        Args:
            connect_tolerance: Max gap (mm) to consider endpoints connected.
            angle_tolerance: Angle threshold (degrees) for parallel/perpendicular.
            concentric_tolerance: Max center offset (mm) for concentric detection.
        """
        super().__init__()
        self.connect_tolerance = connect_tolerance
        self.angle_tolerance = angle_tolerance
        self.concentric_tolerance = concentric_tolerance

    def __call__(
        self,
        doc: CADlingDocument,
        item_batch: list[CADItem],
    ) -> None:
        """Extract structured geometry from 2D sketch items.

        Processes each Sketch2DItem in the batch:
        1. Converts primitives to command sequences
        2. Detects geometric constraints
        3. Writes results to item.properties

        Args:
            doc: The parent CADlingDocument.
            item_batch: List of CADItem instances (only Sketch2DItem are processed).
        """
        processed = 0
        for item in item_batch:
            if not isinstance(item, Sketch2DItem):
                _log.debug(
                    "Skipping non-sketch item: %s (type=%s)",
                    item.label.text if item.label else "unknown",
                    item.item_type,
                )
                continue

            try:
                # Step 1: Convert all profiles to command sequences
                all_commands = self._profiles_to_commands(item.profiles)

                # Step 2: Detect geometric constraints
                all_primitives = item.all_primitives
                constraints = self._detect_constraints(all_primitives)

                # Step 3: Write results to item properties
                item.properties["command_sequence"] = all_commands
                item.properties["geometric_constraints"] = constraints
                item.properties["extraction_method"] = "structured_2d"
                item.properties["num_commands"] = len(all_commands)
                item.properties["num_profiles"] = len(item.profiles)

                # Add provenance
                item.add_provenance(
                    component_type="enrichment_model",
                    component_name=self.__class__.__name__,
                )

                processed += 1
                _log.debug(
                    "Extracted %d commands and %d constraints from %s",
                    len(all_commands),
                    len(constraints),
                    item.label.text if item.label else "unnamed",
                )

            except Exception as exc:
                _log.warning(
                    "Failed to extract geometry from item %s: %s",
                    item.label.text if item.label else "unnamed",
                    exc,
                )

        _log.info(
            "SketchGeometryExtractor processed %d/%d items",
            processed,
            len(item_batch),
        )

    # ------------------------------------------------------------------
    # Profile → Command sequence conversion
    # ------------------------------------------------------------------

    def _profiles_to_commands(
        self, profiles: List[SketchProfile]
    ) -> List[Dict[str, Any]]:
        """Convert all profiles into a single ordered command sequence.

        Each profile becomes a loop starting with SOL and the full
        sequence ends with EOS.

        Args:
            profiles: List of SketchProfile instances.

        Returns:
            List of command dicts: [{"type": "SOL", "params": [...]}, ...]
        """
        commands: List[Dict[str, Any]] = []

        for profile in profiles:
            if not profile.primitives:
                continue

            # Order primitives into connected chains
            ordered_prims = self._order_primitives(profile.primitives)

            # Determine loop origin from first primitive
            origin = self._get_primitive_start(ordered_prims[0]) if ordered_prims else (0.0, 0.0)

            # Insert SOL (start-of-loop) command
            # SOL mask: [0,1] active → [z_offset_or_x, rotation_or_y]
            sol_params = self._pad_params([origin[0], origin[1]])
            commands.append({"type": "SOL", "params": sol_params})

            # Convert each primitive to command(s)
            for prim in ordered_prims:
                prim_commands = self._primitive_to_commands(prim)
                commands.extend(prim_commands)

        # End of sequence
        eos_params = self._pad_params([])
        commands.append({"type": "EOS", "params": eos_params})

        return commands

    def _order_primitives(
        self, primitives: List[Primitive2D]
    ) -> List[Primitive2D]:
        """Order primitives into connected chains by endpoint proximity.

        Starting from the first primitive, greedily connects primitives
        by matching endpoints within the connect_tolerance. Unconnected
        primitives are appended at the end.

        Args:
            primitives: Unordered list of primitives.

        Returns:
            Ordered list of primitives forming connected chains.
        """
        if len(primitives) <= 1:
            return list(primitives)

        # Build endpoint lookup
        remaining = list(primitives)
        ordered = [remaining.pop(0)]

        while remaining:
            current_end = self._get_primitive_end(ordered[-1])
            best_idx = -1
            best_dist = float("inf")
            best_reversed = False

            for i, prim in enumerate(remaining):
                # Check distance from current_end to next start
                start = self._get_primitive_start(prim)
                if start is not None:
                    dist = self._point_distance(current_end, start)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
                        best_reversed = False

                # Also check distance to end (primitive might be traversed in reverse)
                end = self._get_primitive_end(prim)
                if end is not None:
                    dist = self._point_distance(current_end, end)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
                        best_reversed = True

            if best_idx < 0 or best_dist > self.connect_tolerance * 10:
                # No nearby primitive — just append remaining
                ordered.extend(remaining)
                break

            prim = remaining.pop(best_idx)
            if best_reversed:
                prim = self._reverse_primitive(prim)
            ordered.append(prim)

        return ordered

    def _primitive_to_commands(
        self, prim: Primitive2D
    ) -> List[Dict[str, Any]]:
        """Convert a single Primitive2D to one or more command dicts.

        Args:
            prim: A Primitive2D instance.

        Returns:
            List of command dicts.
        """
        if isinstance(prim, Line2D):
            return [self._line_to_command(prim)]
        elif isinstance(prim, Arc2D):
            return [self._arc_to_command(prim)]
        elif isinstance(prim, Circle2D):
            return [self._circle_to_command(prim)]
        elif isinstance(prim, Polyline2D):
            return self._polyline_to_commands(prim)
        elif isinstance(prim, Ellipse2D):
            return self._ellipse_to_commands(prim)
        elif isinstance(prim, Spline2D):
            return self._spline_to_commands(prim)
        else:
            _log.debug("Unknown primitive type: %s", type(prim).__name__)
            return []

    def _line_to_command(self, line: Line2D) -> Dict[str, Any]:
        """Convert Line2D to LINE command.

        GeoToken LINE mask: positions [0,1,2,3] → [x1, y1, x2, y2]
        No z-interleaving — compact 2D representation.

        Format: {"type": "LINE", "params": [x1, y1, x2, y2, 0, 0, ...]}
        """
        params = [
            line.start[0], line.start[1],
            line.end[0], line.end[1],
        ]
        return {"type": "LINE", "params": self._pad_params(params)}

    def _arc_to_command(self, arc: Arc2D) -> Dict[str, Any]:
        """Convert Arc2D to ARC command in 3-point representation.

        GeoToken ARC mask: positions [0,1,2,3,4,5] → [x1, y1, x2, y2, x3, y3]
        These represent the start point, midpoint, and endpoint of the arc.

        Converts from cadling's center+radius+angles representation to the
        3-point format that GeoToken's CommandSequenceTokenizer expects.

        Format: {"type": "ARC", "params": [x_start, y_start, x_mid, y_mid, x_end, y_end, 0, ...]}
        """
        three_point = self._arc_center_to_three_point(
            arc.center[0], arc.center[1],
            arc.radius,
            arc.start_angle, arc.end_angle,
        )
        return {"type": "ARC", "params": self._pad_params(three_point)}

    def _circle_to_command(self, circle: Circle2D) -> Dict[str, Any]:
        """Convert Circle2D to CIRCLE command.

        GeoToken CIRCLE mask: positions [0,1,2] → [cx, cy, r]
        No z-coordinate — radius goes directly at position 2.

        Format: {"type": "CIRCLE", "params": [cx, cy, r, 0, 0, ...]}
        """
        params = [
            circle.center[0], circle.center[1],
            circle.radius,
        ]
        return {"type": "CIRCLE", "params": self._pad_params(params)}

    def _polyline_to_commands(self, polyline: Polyline2D) -> List[Dict[str, Any]]:
        """Convert Polyline2D to a sequence of LINE commands.

        Each segment of the polyline becomes a separate LINE command.
        Bulge values (arc segments) are converted to ARC commands if present.

        Args:
            polyline: Polyline2D instance.

        Returns:
            List of LINE and/or ARC command dicts.
        """
        commands = []
        points = polyline.points
        n = len(points)

        for i in range(n - 1):
            bulge = 0.0
            if polyline.bulges and i < len(polyline.bulges):
                bulge = polyline.bulges[i]

            if abs(bulge) > 1e-6:
                # Arc segment — convert bulge to arc parameters
                arc_cmd = self._bulge_to_arc_command(
                    points[i], points[i + 1], bulge
                )
                commands.append(arc_cmd)
            else:
                # Straight segment — compact [x1, y1, x2, y2]
                params = [
                    points[i][0], points[i][1],
                    points[i + 1][0], points[i + 1][1],
                ]
                commands.append({"type": "LINE", "params": self._pad_params(params)})

        # Handle closing segment
        if polyline.closed and n > 1:
            last_bulge = 0.0
            if polyline.bulges and (n - 1) < len(polyline.bulges):
                last_bulge = polyline.bulges[n - 1]

            if abs(last_bulge) > 1e-6:
                arc_cmd = self._bulge_to_arc_command(
                    points[-1], points[0], last_bulge
                )
                commands.append(arc_cmd)
            else:
                # Closing segment — compact [x1, y1, x2, y2]
                params = [
                    points[-1][0], points[-1][1],
                    points[0][0], points[0][1],
                ]
                commands.append({"type": "LINE", "params": self._pad_params(params)})

        return commands

    def _ellipse_to_commands(self, ellipse: Ellipse2D) -> List[Dict[str, Any]]:
        """Convert Ellipse2D to approximated ARC or LINE commands.

        Ellipses are approximated as arcs when the ratio is close to 1.0,
        otherwise decomposed into LINE segments via point sampling.

        Args:
            ellipse: Ellipse2D instance.

        Returns:
            List of ARC or LINE command dicts.
        """
        if abs(ellipse.ratio - 1.0) < 0.05:
            # Nearly circular — treat as arc or circle
            if ellipse.is_full:
                return [self._circle_to_command(
                    Circle2D(center=ellipse.center, radius=ellipse.major_radius)
                )]
            else:
                return [self._arc_to_command(
                    Arc2D(
                        center=ellipse.center,
                        radius=ellipse.major_radius,
                        start_angle=math.degrees(ellipse.start_param),
                        end_angle=math.degrees(ellipse.end_param),
                    )
                )]

        # General ellipse — sample to polyline
        num_segments = 16
        points = []
        for i in range(num_segments + 1):
            t = ellipse.start_param + (
                ellipse.end_param - ellipse.start_param
            ) * i / num_segments
            # Parametric ellipse point
            cos_t = math.cos(t)
            sin_t = math.sin(t)
            # Rotate by major axis angle
            ma_angle = math.atan2(ellipse.major_axis[1], ellipse.major_axis[0])
            cos_a = math.cos(ma_angle)
            sin_a = math.sin(ma_angle)
            a = ellipse.major_radius
            b = ellipse.minor_radius
            px = a * cos_t * cos_a - b * sin_t * sin_a + ellipse.center[0]
            py = a * cos_t * sin_a + b * sin_t * cos_a + ellipse.center[1]
            points.append((px, py))

        commands = []
        for i in range(len(points) - 1):
            # Compact [x1, y1, x2, y2] — no z-interleaving
            params = [
                points[i][0], points[i][1],
                points[i + 1][0], points[i + 1][1],
            ]
            commands.append({"type": "LINE", "params": self._pad_params(params)})

        return commands

    def _spline_to_commands(self, spline: Spline2D) -> List[Dict[str, Any]]:
        """Convert Spline2D to approximated LINE commands via control polygon.

        For simplicity, the spline is approximated by its control polygon
        (lines connecting control points). This is a rough approximation
        suitable for the tokenizer.

        Args:
            spline: Spline2D instance.

        Returns:
            List of LINE command dicts.
        """
        commands = []
        pts = spline.control_points

        for i in range(len(pts) - 1):
            # Compact [x1, y1, x2, y2] — no z-interleaving
            params = [
                pts[i][0], pts[i][1],
                pts[i + 1][0], pts[i + 1][1],
            ]
            commands.append({"type": "LINE", "params": self._pad_params(params)})

        if spline.closed and len(pts) > 1:
            params = [
                pts[-1][0], pts[-1][1],
                pts[0][0], pts[0][1],
            ]
            commands.append({"type": "LINE", "params": self._pad_params(params)})

        return commands

    # ------------------------------------------------------------------
    # Bulge → Arc conversion
    # ------------------------------------------------------------------

    def _bulge_to_arc_command(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        bulge: float,
    ) -> Dict[str, Any]:
        """Convert a DXF bulge segment to an ARC command in 3-point format.

        The bulge value encodes the tangent of 1/4 of the included angle
        of the arc. Positive = CCW, negative = CW.

        Output uses GeoToken's 3-point ARC format:
            [x_start, y_start, x_mid, y_mid, x_end, y_end]

        Args:
            p1: Start point (x, y).
            p2: End point (x, y).
            bulge: Bulge value from DXF polyline.

        Returns:
            ARC command dict in 3-point format.
        """
        # Chord midpoint and length
        mx = (p1[0] + p2[0]) / 2
        my = (p1[1] + p2[1]) / 2
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        chord_len = math.sqrt(dx * dx + dy * dy)

        if chord_len < 1e-10:
            # Degenerate — return a zero-length line (compact format)
            return {"type": "LINE", "params": self._pad_params([
                p1[0], p1[1], p2[0], p2[1],
            ])}

        # Sagitta (height of arc above chord)
        sagitta = abs(bulge) * chord_len / 2

        if sagitta < 1e-10:
            # Degenerate arc (zero bulge) — return straight line (compact format)
            return {"type": "LINE", "params": self._pad_params([
                p1[0], p1[1], p2[0], p2[1],
            ])}

        # Radius from chord and sagitta
        radius = (chord_len * chord_len / 4 + sagitta * sagitta) / (2 * sagitta)

        # Center is offset from chord midpoint along perpendicular
        # Perpendicular direction (unit)
        perp_x = -dy / chord_len
        perp_y = dx / chord_len

        # Distance from midpoint to center
        d = radius - sagitta

        # Direction depends on bulge sign
        if bulge > 0:
            cx = mx + perp_x * d
            cy = my + perp_y * d
        else:
            cx = mx - perp_x * d
            cy = my - perp_y * d

        # Compute angles for 3-point conversion
        start_angle = math.degrees(math.atan2(p1[1] - cy, p1[0] - cx)) % 360
        end_angle = math.degrees(math.atan2(p2[1] - cy, p2[0] - cx)) % 360

        # Convert center+radius+angles to 3-point format for GeoToken
        three_point = self._arc_center_to_three_point(
            cx, cy, radius, start_angle, end_angle
        )
        return {"type": "ARC", "params": self._pad_params(three_point)}

    # ------------------------------------------------------------------
    # Arc format conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _arc_center_to_three_point(
        cx: float, cy: float,
        radius: float,
        start_deg: float, end_deg: float,
    ) -> List[float]:
        """Convert arc center+radius+angles to 3-point representation.

        Computes the start, midpoint, and endpoint of the arc on the circle,
        returning them as a flat list of 6 coordinates matching GeoToken's
        ARC mask: [x_start, y_start, x_mid, y_mid, x_end, y_end].

        The midpoint is placed at the angular midpoint between start and end,
        traversing the arc in the positive (CCW) direction. If end_deg < start_deg,
        the arc wraps through 360°.

        Args:
            cx: Center x-coordinate.
            cy: Center y-coordinate.
            radius: Arc radius.
            start_deg: Start angle in degrees (0-360).
            end_deg: End angle in degrees (0-360).

        Returns:
            [x_start, y_start, x_mid, y_mid, x_end, y_end]
        """
        start_rad = math.radians(start_deg)
        end_rad = math.radians(end_deg)

        # Compute midpoint angle (handle wrap-around)
        if end_deg >= start_deg:
            mid_deg = (start_deg + end_deg) / 2.0
        else:
            # Arc wraps through 360°
            mid_deg = (start_deg + (end_deg + 360.0)) / 2.0
            if mid_deg >= 360.0:
                mid_deg -= 360.0
        mid_rad = math.radians(mid_deg)

        # Compute the three points on the arc
        x_start = cx + radius * math.cos(start_rad)
        y_start = cy + radius * math.sin(start_rad)
        x_mid = cx + radius * math.cos(mid_rad)
        y_mid = cy + radius * math.sin(mid_rad)
        x_end = cx + radius * math.cos(end_rad)
        y_end = cy + radius * math.sin(end_rad)

        return [x_start, y_start, x_mid, y_mid, x_end, y_end]

    # ------------------------------------------------------------------
    # Geometric constraint detection
    # ------------------------------------------------------------------

    def _detect_constraints(
        self, primitives: List[Primitive2D]
    ) -> List[Dict[str, Any]]:
        """Detect geometric constraints between primitives.

        Checks for:
        - PARALLEL: Two lines with nearly equal direction angles.
        - PERPENDICULAR: Two lines at ~90° angle.
        - CONCENTRIC: Two circles/arcs with nearly equal centers.
        - TANGENT: A line tangent to a circle/arc.
        - EQUAL_RADIUS: Two circles/arcs with the same radius.
        - COLLINEAR: Two lines on the same infinite line.

        Args:
            primitives: List of all primitives to check.

        Returns:
            List of constraint dicts with keys:
                - type: Constraint type string
                - entity_a: Index of first entity
                - entity_b: Index of second entity
                - confidence: Float 0.0-1.0
        """
        constraints: List[Dict[str, Any]] = []

        # Separate primitives by type for targeted checks
        lines = [(i, p) for i, p in enumerate(primitives) if isinstance(p, Line2D)]
        circles_arcs = [
            (i, p) for i, p in enumerate(primitives)
            if isinstance(p, (Circle2D, Arc2D))
        ]

        # Check line-line constraints
        for a_idx in range(len(lines)):
            for b_idx in range(a_idx + 1, len(lines)):
                i_a, line_a = lines[a_idx]
                i_b, line_b = lines[b_idx]

                angle = self._angle_between_lines(line_a, line_b)

                # Parallel check
                if angle < self.angle_tolerance or abs(angle - 180) < self.angle_tolerance:
                    constraints.append({
                        "type": "PARALLEL",
                        "entity_a": i_a,
                        "entity_b": i_b,
                        "confidence": 1.0 - angle / 90.0 if angle < 90 else 1.0 - (180 - angle) / 90.0,
                    })

                # Perpendicular check
                if abs(angle - 90) < self.angle_tolerance:
                    constraints.append({
                        "type": "PERPENDICULAR",
                        "entity_a": i_a,
                        "entity_b": i_b,
                        "confidence": 1.0 - abs(angle - 90) / self.angle_tolerance,
                    })

                # Collinear check (parallel + overlapping support line)
                if angle < self.angle_tolerance:
                    dist = self._line_to_line_distance(line_a, line_b)
                    if dist < self.connect_tolerance:
                        constraints.append({
                            "type": "COLLINEAR",
                            "entity_a": i_a,
                            "entity_b": i_b,
                            "confidence": 1.0 - dist / self.connect_tolerance,
                        })

        # Check circle/arc-circle/arc constraints
        for a_idx in range(len(circles_arcs)):
            for b_idx in range(a_idx + 1, len(circles_arcs)):
                i_a, ca_a = circles_arcs[a_idx]
                i_b, ca_b = circles_arcs[b_idx]

                center_a = self._get_center(ca_a)
                center_b = self._get_center(ca_b)
                radius_a = self._get_radius(ca_a)
                radius_b = self._get_radius(ca_b)

                if center_a is None or center_b is None:
                    continue

                # Concentric check
                center_dist = self._point_distance(center_a, center_b)
                if center_dist < self.concentric_tolerance:
                    constraints.append({
                        "type": "CONCENTRIC",
                        "entity_a": i_a,
                        "entity_b": i_b,
                        "confidence": 1.0 - center_dist / self.concentric_tolerance,
                    })

                # Equal radius check
                if radius_a and radius_b:
                    radius_diff = abs(radius_a - radius_b)
                    avg_radius = (radius_a + radius_b) / 2
                    if avg_radius > 0 and radius_diff / avg_radius < 0.02:
                        constraints.append({
                            "type": "EQUAL_RADIUS",
                            "entity_a": i_a,
                            "entity_b": i_b,
                            "confidence": 1.0 - radius_diff / avg_radius / 0.02,
                        })

        # Check line-circle tangency
        for i_l, line in lines:
            for i_c, circ in circles_arcs:
                center = self._get_center(circ)
                radius = self._get_radius(circ)
                if center is None or radius is None:
                    continue

                dist = self._point_to_line_distance(center, line)
                tangency_error = abs(dist - radius)
                if tangency_error < self.connect_tolerance:
                    constraints.append({
                        "type": "TANGENT",
                        "entity_a": i_l,
                        "entity_b": i_c,
                        "confidence": 1.0 - tangency_error / self.connect_tolerance,
                    })

        _log.debug("Detected %d geometric constraints", len(constraints))
        return constraints

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_params(params: List[float]) -> List[float]:
        """Pad parameter list to _NUM_PARAMS length with zeros.

        Args:
            params: Input parameter list.

        Returns:
            List of exactly _NUM_PARAMS floats.
        """
        padded = [float(p) for p in params]
        while len(padded) < _NUM_PARAMS:
            padded.append(0.0)
        return padded[:_NUM_PARAMS]

    @staticmethod
    def _get_primitive_start(prim: Primitive2D) -> Optional[Tuple[float, float]]:
        """Get the start point of a primitive.

        Args:
            prim: A Primitive2D instance.

        Returns:
            (x, y) start point, or None.
        """
        if isinstance(prim, Line2D):
            return prim.start
        elif isinstance(prim, Arc2D):
            return prim.start_point
        elif isinstance(prim, Circle2D):
            return (prim.center[0] + prim.radius, prim.center[1])
        elif isinstance(prim, Polyline2D) and prim.points:
            return prim.points[0]
        elif isinstance(prim, Ellipse2D):
            return prim.center
        elif isinstance(prim, Spline2D) and prim.control_points:
            return prim.control_points[0]
        return None

    @staticmethod
    def _get_primitive_end(prim: Primitive2D) -> Optional[Tuple[float, float]]:
        """Get the end point of a primitive.

        Args:
            prim: A Primitive2D instance.

        Returns:
            (x, y) end point, or None.
        """
        if isinstance(prim, Line2D):
            return prim.end
        elif isinstance(prim, Arc2D):
            return prim.end_point
        elif isinstance(prim, Circle2D):
            return (prim.center[0] + prim.radius, prim.center[1])
        elif isinstance(prim, Polyline2D) and prim.points:
            if prim.closed:
                return prim.points[0]
            return prim.points[-1]
        elif isinstance(prim, Ellipse2D):
            return prim.center
        elif isinstance(prim, Spline2D) and prim.control_points:
            if prim.closed:
                return prim.control_points[0]
            return prim.control_points[-1]
        return None

    @staticmethod
    def _reverse_primitive(prim: Primitive2D) -> Primitive2D:
        """Create a reversed copy of a primitive (swap start/end).

        Args:
            prim: Primitive to reverse.

        Returns:
            Reversed primitive (new object).
        """
        if isinstance(prim, Line2D):
            return Line2D(
                start=prim.end,
                end=prim.start,
                layer=prim.layer,
                color=prim.color,
                line_weight=prim.line_weight,
                confidence=prim.confidence,
                source_entity_id=prim.source_entity_id,
            )
        elif isinstance(prim, Arc2D):
            return Arc2D(
                center=prim.center,
                radius=prim.radius,
                start_angle=prim.end_angle,
                end_angle=prim.start_angle,
                layer=prim.layer,
                color=prim.color,
                line_weight=prim.line_weight,
                confidence=prim.confidence,
                source_entity_id=prim.source_entity_id,
            )
        elif isinstance(prim, Polyline2D):
            return Polyline2D(
                points=list(reversed(prim.points)),
                closed=prim.closed,
                bulges=(
                    list(reversed([-b for b in prim.bulges]))
                    if prim.bulges
                    else None
                ),
                layer=prim.layer,
                color=prim.color,
                line_weight=prim.line_weight,
                confidence=prim.confidence,
                source_entity_id=prim.source_entity_id,
            )
        # For circles, ellipses, splines — direction is less meaningful
        return prim

    @staticmethod
    def _point_distance(
        p1: Tuple[float, float], p2: Tuple[float, float]
    ) -> float:
        """Euclidean distance between two 2D points."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def _angle_between_lines(a: Line2D, b: Line2D) -> float:
        """Compute the acute angle between two lines in degrees.

        Args:
            a: First line.
            b: Second line.

        Returns:
            Angle in degrees (0-180).
        """
        dir_a = a.direction
        dir_b = b.direction

        dot = dir_a[0] * dir_b[0] + dir_a[1] * dir_b[1]
        dot = max(-1.0, min(1.0, dot))  # Clamp for acos safety
        angle = math.degrees(math.acos(abs(dot)))
        return angle

    @staticmethod
    def _line_to_line_distance(a: Line2D, b: Line2D) -> float:
        """Compute minimum distance between the support lines of two lines.

        For parallel lines, this is the perpendicular distance.
        For non-parallel lines, returns 0 (they intersect).

        Args:
            a: First line.
            b: Second line.

        Returns:
            Distance in mm.
        """
        # Use point-to-line distance from line b's start to line a's support
        dir_a = a.direction
        if abs(dir_a[0]) < 1e-10 and abs(dir_a[1]) < 1e-10:
            return float("inf")

        # Normal to line a
        nx = -dir_a[1]
        ny = dir_a[0]

        # Distance from b.start to line a
        dx = b.start[0] - a.start[0]
        dy = b.start[1] - a.start[1]
        dist = abs(nx * dx + ny * dy)
        return dist

    @staticmethod
    def _point_to_line_distance(
        point: Tuple[float, float], line: Line2D
    ) -> float:
        """Compute distance from a point to a line segment's support line.

        Args:
            point: (x, y) point.
            line: Line2D segment.

        Returns:
            Perpendicular distance in mm.
        """
        dir_l = line.direction
        if abs(dir_l[0]) < 1e-10 and abs(dir_l[1]) < 1e-10:
            return SketchGeometryExtractor._point_distance(point, line.start)

        nx = -dir_l[1]
        ny = dir_l[0]
        dx = point[0] - line.start[0]
        dy = point[1] - line.start[1]
        return abs(nx * dx + ny * dy)

    @staticmethod
    def _get_center(prim: Primitive2D) -> Optional[Tuple[float, float]]:
        """Get center point of a circle or arc primitive."""
        if isinstance(prim, Circle2D):
            return prim.center
        elif isinstance(prim, Arc2D):
            return prim.center
        return None

    @staticmethod
    def _get_radius(prim: Primitive2D) -> Optional[float]:
        """Get radius of a circle or arc primitive."""
        if isinstance(prim, Circle2D):
            return prim.radius
        elif isinstance(prim, Arc2D):
            return prim.radius
        return None
