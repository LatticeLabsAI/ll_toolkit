"""Command executor for CAD generation reconstruction.

Executes predicted command token sequences in OpenCASCADE to reconstruct
CAD geometry from DeepCAD-style autoregressive command sequences. The
executor decodes quantized tokens back to continuous geometric parameters,
builds sketch profiles from 2D primitives, extrudes them into solids, and
applies Boolean operations to compose the final shape.

Classes:
    CommandToken: Decoded command token with type and parameters.
    CommandExecutor: Orchestrates command-to-CAD reconstruction.

Example:
    executor = CommandExecutor(tolerance=1e-6)
    result = executor.execute(command_tokens)
    if result['valid']:
        executor.export_step(result['shape'], 'output.step')
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)

# Lazy import of pythonocc
_has_pythonocc = False
try:
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse, BRepAlgoAPI_Common
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakeWire,
        BRepBuilderAPI_Transform,
    )
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCC.Core.GC import GC_MakeArcOfCircle, GC_MakeSegment
    from OCC.Core.Geom import Geom_Plane
    from OCC.Core.gp import (
        gp_Ax1,
        gp_Ax2,
        gp_Circ,
        gp_Dir,
        gp_Pln,
        gp_Pnt,
        gp_Pnt2d,
        gp_Trsf,
        gp_Vec,
    )
    from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Shape, TopoDS_Wire
    from OCC.Core.Interface import Interface_Static

    _has_pythonocc = True
    _log.debug("pythonocc-core available for command execution")
except ImportError:
    _log.warning(
        "pythonocc-core not available. CommandExecutor will operate in "
        "validation-only mode without geometry construction."
    )


class CommandType(str, Enum):
    """Types of CAD commands in a DeepCAD-style sequence.

    These correspond to the command vocabulary used in autoregressive
    CAD generation models.
    """

    SOL = "SOL"          # Start of loop (sketch boundary)
    EOL = "EOL"          # End of loop
    EOS = "EOS"          # End of sketch
    LINE = "LINE"        # Line segment
    ARC = "ARC"          # Circular arc
    CIRCLE = "CIRCLE"    # Full circle
    EXTRUDE = "EXTRUDE"  # Linear extrusion
    BOOLEAN = "BOOLEAN"  # Boolean operation (union/cut/intersect)


class BooleanOpType(str, Enum):
    """Boolean operation types for combining solids."""

    UNION = "union"
    CUT = "cut"
    INTERSECT = "intersect"


@dataclass
class CommandToken:
    """Decoded command token with type and continuous parameters.

    Attributes:
        command_type: Type of CAD command.
        params: Continuous parameters (dequantized from token indices).
        raw_token_id: Original quantized token ID before decoding.
    """

    command_type: CommandType
    params: List[float] = field(default_factory=list)
    raw_token_id: Optional[int] = None


class CommandExecutor:
    """Execute predicted command token sequences in OpenCASCADE.

    Takes quantized command tokens from a DeepCAD-style autoregressive model,
    decodes them into continuous geometric parameters, constructs sketch
    profiles, applies extrusions, and combines solids via Boolean operations.

    The reconstruction pipeline follows:
        1. Decode tokens -> CommandToken sequence
        2. Dequantize parameters -> continuous values
        3. Build sketches: group primitives between SOL..EOL into 2D wires,
           close into faces via BRepBuilderAPI_MakeFace
        4. Apply extrusions: BRepPrimAPI_MakePrism for linear extrusion
        5. Apply Booleans: BRepAlgoAPI_Fuse / Cut / Common
        6. Validate result via BRepCheck_Analyzer
        7. Return dict with shape, validity, and any errors

    Attributes:
        tolerance: Geometric tolerance for operations.
        has_pythonocc: Whether pythonocc-core is available.
        quantization_levels: Number of quantization levels for dequantization.
        coord_range: Coordinate range tuple (min, max) for dequantization.

    Example:
        executor = CommandExecutor(tolerance=1e-6)
        result = executor.execute(command_tokens)
        if result['valid']:
            executor.export_step(result['shape'], 'output.step')
    """

    def __init__(
        self,
        tolerance: float = 1e-6,
        quantization_levels: int = 256,
        coord_range: Tuple[float, float] = (-1.0, 1.0),
    ):
        """Initialize the command executor.

        Args:
            tolerance: Geometric tolerance for edge/wire/face operations.
            quantization_levels: Number of discrete levels used in quantization.
                Must match the tokenizer's quantization config.
            coord_range: The continuous coordinate range (min, max) that the
                quantization levels map onto.
        """
        self.tolerance = tolerance
        self.has_pythonocc = _has_pythonocc
        self.quantization_levels = quantization_levels
        self.coord_range = coord_range

        if not self.has_pythonocc:
            _log.warning(
                "CommandExecutor initialized without pythonocc. "
                "Geometry construction disabled; only token decoding available."
            )

    def execute(self, command_tokens: List[int]) -> Dict[str, Any]:
        """Execute a command token sequence and reconstruct CAD geometry.

        This is the main entry point. It decodes tokens, builds sketches,
        applies extrusions and Booleans, validates the result, and returns
        a structured result dictionary.

        Args:
            command_tokens: List of quantized token IDs from the generation
                model. The sequence follows the grammar:
                [SOL LINE* EOL]+ EOS [EXTRUDE params]+ [BOOLEAN op]*

        Returns:
            Dictionary containing:
                - 'shape': The reconstructed TopoDS_Shape (or None).
                - 'valid': Whether the shape passes topology validation.
                - 'errors': List of error messages encountered.
                - 'commands': Decoded CommandToken list.
                - 'num_sketches': Number of sketch profiles built.
                - 'num_extrusions': Number of extrusions applied.
        """
        result: Dict[str, Any] = {
            "shape": None,
            "valid": False,
            "errors": [],
            "commands": [],
            "num_sketches": 0,
            "num_extrusions": 0,
        }

        # Step 1: Decode tokens into CommandToken sequence
        try:
            commands = self._decode_tokens(command_tokens)
            result["commands"] = commands
        except Exception as e:
            error_msg = f"Token decoding failed: {e}"
            _log.error(error_msg)
            result["errors"].append(error_msg)
            return result

        if not commands:
            result["errors"].append("Empty command sequence after decoding")
            return result

        if not self.has_pythonocc:
            result["errors"].append(
                "pythonocc-core not available; cannot construct geometry"
            )
            return result

        # Step 2: Dequantize parameters to continuous values
        try:
            commands = self._dequantize_commands(commands)
        except Exception as e:
            error_msg = f"Dequantization failed: {e}"
            _log.error(error_msg)
            result["errors"].append(error_msg)
            return result

        # Step 3: Group commands and build sketches
        sketch_groups = self._extract_sketch_groups(commands)
        result["num_sketches"] = len(sketch_groups)

        faces: List[TopoDS_Face] = []
        for i, group in enumerate(sketch_groups):
            try:
                face = self._build_sketch(group)
                if face is not None:
                    faces.append(face)
                else:
                    result["errors"].append(f"Sketch group {i} produced no face")
            except Exception as e:
                error_msg = f"Sketch {i} construction failed: {e}"
                _log.error(error_msg)
                result["errors"].append(error_msg)

        if not faces:
            result["errors"].append("No valid sketch faces built")
            return result

        # Step 4: Extract extrusion commands and build individual solids
        extrusion_commands = [
            cmd for cmd in commands if cmd.command_type == CommandType.EXTRUDE
        ]
        result["num_extrusions"] = len(extrusion_commands)

        # Build each extruded solid individually (no auto-fusion)
        solids: List[TopoDS_Shape] = []
        for i, face in enumerate(faces):
            try:
                if i < len(extrusion_commands):
                    ext_params = extrusion_commands[i].params
                else:
                    # Default extrusion: 1 unit along Z
                    ext_params = [0.0, 0.0, 1.0, 1.0]
                solid = self._apply_extrusion(face, ext_params)
                solids.append(solid)
            except Exception as e:
                error_msg = f"Extrusion {i} failed: {e}"
                _log.error(error_msg)
                result["errors"].append(error_msg)

        # Step 5: Apply Boolean operations to combine solids
        boolean_commands = [
            cmd for cmd in commands if cmd.command_type == CommandType.BOOLEAN
        ]

        body: Optional[TopoDS_Shape] = None
        if solids:
            body = solids[0]

            if boolean_commands:
                # Apply explicit Boolean commands: each command combines the
                # accumulated body with the next solid in the queue
                solid_queue = list(solids[1:])
                for i, bool_cmd in enumerate(boolean_commands):
                    if len(bool_cmd.params) < 1:
                        continue
                    try:
                        op_type_idx = int(bool_cmd.params[0])
                        op_type = list(BooleanOpType)[
                            min(op_type_idx, len(BooleanOpType) - 1)
                        ]
                        if solid_queue:
                            next_solid = solid_queue.pop(0)
                            body = self._apply_boolean(body, next_solid, op_type)
                            _log.debug(
                                "Boolean %d: %s applied successfully",
                                i, op_type.value,
                            )
                        else:
                            _log.debug(
                                "Boolean %d: %s skipped (no remaining operand)",
                                i, op_type.value,
                            )
                    except Exception as e:
                        error_msg = f"Boolean operation {i} failed: {e}"
                        _log.error(error_msg)
                        result["errors"].append(error_msg)

                # Fuse any remaining solids not consumed by Boolean commands
                for remaining in solid_queue:
                    try:
                        body = self._apply_boolean(
                            body, remaining, BooleanOpType.UNION
                        )
                    except Exception as e:
                        _log.warning("Fusing remaining solid failed: %s", e)
            else:
                # No explicit Booleans: default to sequential union
                for solid in solids[1:]:
                    try:
                        body = self._apply_boolean(
                            body, solid, BooleanOpType.UNION
                        )
                    except Exception as e:
                        _log.warning("Default union failed: %s", e)

        if body is None:
            result["errors"].append("No solid body produced")
            return result

        # Step 6: Validate result
        result["shape"] = body
        try:
            analyzer = BRepCheck_Analyzer(body)
            result["valid"] = analyzer.IsValid()
            if not result["valid"]:
                result["errors"].append("BRepCheck_Analyzer reports invalid shape")
        except Exception as e:
            error_msg = f"Validation failed: {e}"
            _log.error(error_msg)
            result["errors"].append(error_msg)

        _log.info(
            "Command execution complete: valid=%s, sketches=%d, extrusions=%d, errors=%d",
            result["valid"],
            result["num_sketches"],
            result["num_extrusions"],
            len(result["errors"]),
        )

        return result

    def _decode_tokens(self, token_ids: List[int]) -> List[CommandToken]:
        """Decode raw token IDs into structured CommandToken objects.

        The token vocabulary is structured as:
            - Special tokens (0-5): SOL, EOL, EOS, EXTRUDE, BOOLEAN, PAD
            - Command tokens (6-8): LINE, ARC, CIRCLE
            - Parameter tokens (9+): quantized coordinate/dimension values

        Args:
            token_ids: Raw integer token IDs from the generation model.

        Returns:
            List of CommandToken objects with decoded types and raw parameters.
        """
        # Token vocabulary mapping
        special_tokens = {
            0: CommandType.SOL,
            1: CommandType.EOL,
            2: CommandType.EOS,
            3: CommandType.EXTRUDE,
            4: CommandType.BOOLEAN,
        }
        command_tokens_map = {
            6: CommandType.LINE,
            7: CommandType.ARC,
            8: CommandType.CIRCLE,
        }

        # Expected parameter counts per command type
        param_counts = {
            CommandType.SOL: 0,
            CommandType.EOL: 0,
            CommandType.EOS: 0,
            CommandType.LINE: 4,       # x1, y1, x2, y2
            CommandType.ARC: 6,        # x1, y1, x2, y2, cx, cy
            CommandType.CIRCLE: 3,     # cx, cy, radius
            CommandType.EXTRUDE: 4,    # dx, dy, dz, distance
            CommandType.BOOLEAN: 1,    # op_type
        }

        commands: List[CommandToken] = []
        i = 0
        while i < len(token_ids):
            token_id = token_ids[i]

            # Check special tokens
            if token_id in special_tokens:
                cmd_type = special_tokens[token_id]
                commands.append(
                    CommandToken(command_type=cmd_type, raw_token_id=token_id)
                )
                i += 1
                continue

            # Check command tokens
            if token_id in command_tokens_map:
                cmd_type = command_tokens_map[token_id]
                n_params = param_counts[cmd_type]
                params: List[float] = []

                for j in range(n_params):
                    param_idx = i + 1 + j
                    if param_idx < len(token_ids):
                        # Parameter tokens start at offset 9
                        param_value = float(token_ids[param_idx] - 9)
                        params.append(param_value)
                    else:
                        params.append(0.0)
                        _log.warning(
                            "Missing parameter %d for %s at token %d",
                            j, cmd_type.value, i,
                        )

                commands.append(
                    CommandToken(
                        command_type=cmd_type,
                        params=params,
                        raw_token_id=token_id,
                    )
                )
                i += 1 + n_params
                continue

            # Check extrude/boolean parameter sequences
            if token_id == 3:  # EXTRUDE
                cmd_type = CommandType.EXTRUDE
                n_params = param_counts[cmd_type]
                params = []
                for j in range(n_params):
                    param_idx = i + 1 + j
                    if param_idx < len(token_ids):
                        params.append(float(token_ids[param_idx] - 9))
                    else:
                        params.append(0.0)
                commands.append(
                    CommandToken(
                        command_type=cmd_type, params=params, raw_token_id=token_id
                    )
                )
                i += 1 + n_params
                continue

            # Unknown token: skip
            _log.debug("Skipping unknown token %d at position %d", token_id, i)
            i += 1

        return commands

    def _dequantize_commands(
        self, commands: List[CommandToken]
    ) -> List[CommandToken]:
        """Dequantize parameters from discrete indices to continuous values.

        Maps quantized integer values back to the continuous coordinate range
        using linear interpolation:
            value = coord_min + (index / (levels - 1)) * (coord_max - coord_min)

        Args:
            commands: List of CommandTokens with quantized parameter values.

        Returns:
            Same command list with parameters replaced by continuous values.
        """
        coord_min, coord_max = self.coord_range
        levels = self.quantization_levels

        for cmd in commands:
            if cmd.params:
                dequantized = []
                for p in cmd.params:
                    # Clamp to valid range
                    idx = max(0.0, min(p, float(levels - 1)))
                    value = coord_min + (idx / max(levels - 1, 1)) * (
                        coord_max - coord_min
                    )
                    dequantized.append(value)
                cmd.params = dequantized

        return commands

    def _extract_sketch_groups(
        self, commands: List[CommandToken]
    ) -> List[List[CommandToken]]:
        """Extract sketch primitive groups delimited by SOL/EOL tokens.

        Scans the command sequence for SOL..EOL blocks, collecting the
        primitives (LINE, ARC, CIRCLE) within each block into groups.

        Args:
            commands: Decoded command token list.

        Returns:
            List of sketch groups, each containing primitive commands.
        """
        groups: List[List[CommandToken]] = []
        current_group: Optional[List[CommandToken]] = None

        for cmd in commands:
            if cmd.command_type == CommandType.SOL:
                current_group = []
            elif cmd.command_type == CommandType.EOL:
                if current_group is not None and len(current_group) > 0:
                    groups.append(current_group)
                current_group = None
            elif cmd.command_type == CommandType.EOS:
                # End of sketch; finalize any open group
                if current_group is not None and len(current_group) > 0:
                    groups.append(current_group)
                current_group = None
            elif current_group is not None:
                if cmd.command_type in (
                    CommandType.LINE,
                    CommandType.ARC,
                    CommandType.CIRCLE,
                ):
                    current_group.append(cmd)

        return groups

    def _build_sketch(self, commands: List[CommandToken]) -> Optional[TopoDS_Face]:
        """Build a 2D sketch face from a group of primitive commands.

        Constructs edges from LINE, ARC, and CIRCLE commands, wires them
        together using BRepBuilderAPI_MakeWire, and creates a planar face
        via BRepBuilderAPI_MakeFace.

        For single-CIRCLE groups, creates a full circular wire directly.

        Args:
            commands: List of primitive commands within a single sketch loop.

        Returns:
            TopoDS_Face if construction succeeds, None otherwise.
        """
        if not self.has_pythonocc:
            return None

        if not commands:
            return None

        # Special case: single circle
        if (
            len(commands) == 1
            and commands[0].command_type == CommandType.CIRCLE
        ):
            return self._build_circle_face(commands[0])

        # Build wire from edges
        wire_builder = BRepBuilderAPI_MakeWire()

        for cmd in commands:
            try:
                edge = self._command_to_edge(cmd)
                if edge is not None:
                    wire_builder.Add(edge)
            except Exception as e:
                _log.warning("Failed to add edge for %s: %s", cmd.command_type.value, e)

        if not wire_builder.IsDone():
            _log.warning("Wire construction incomplete")
            return None

        wire = wire_builder.Wire()

        # Create face from wire on XY plane
        try:
            plane = gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
            face_builder = BRepBuilderAPI_MakeFace(plane, wire, True)
            if face_builder.IsDone():
                return face_builder.Face()
            else:
                _log.warning("Face construction failed from wire")
                return None
        except Exception as e:
            _log.error("Face construction error: %s", e)
            return None

    def _command_to_edge(self, cmd: CommandToken) -> Optional[Any]:
        """Convert a single primitive command to an OCC edge.

        Args:
            cmd: A LINE, ARC, or CIRCLE command token.

        Returns:
            TopoDS_Edge if construction succeeds, None otherwise.
        """
        if cmd.command_type == CommandType.LINE:
            if len(cmd.params) < 4:
                _log.warning("LINE command has insufficient params: %d", len(cmd.params))
                return None
            x1, y1, x2, y2 = cmd.params[:4]
            p1 = gp_Pnt(x1, y1, 0.0)
            p2 = gp_Pnt(x2, y2, 0.0)
            if p1.Distance(p2) < self.tolerance:
                _log.debug("Degenerate LINE: endpoints coincide")
                return None
            edge_builder = BRepBuilderAPI_MakeEdge(
                GC_MakeSegment(p1, p2).Value()
            )
            if edge_builder.IsDone():
                return edge_builder.Edge()
            return None

        elif cmd.command_type == CommandType.ARC:
            if len(cmd.params) < 6:
                _log.warning("ARC command has insufficient params: %d", len(cmd.params))
                return None
            x1, y1, x2, y2, cx, cy = cmd.params[:6]
            p1 = gp_Pnt(x1, y1, 0.0)
            p2 = gp_Pnt(x2, y2, 0.0)
            center = gp_Pnt(cx, cy, 0.0)

            # Compute midpoint on arc for 3-point arc construction
            radius = center.Distance(p1)
            if radius < self.tolerance:
                _log.debug("Degenerate ARC: zero radius")
                return None

            # Compute angles for start/end relative to center
            angle1 = math.atan2(y1 - cy, x1 - cx)
            angle2 = math.atan2(y2 - cy, x2 - cx)
            mid_angle = (angle1 + angle2) / 2.0

            # Ensure arc goes the shorter way (counterclockwise)
            if angle2 < angle1:
                mid_angle += math.pi

            mx = cx + radius * math.cos(mid_angle)
            my = cy + radius * math.sin(mid_angle)
            p_mid = gp_Pnt(mx, my, 0.0)

            try:
                arc = GC_MakeArcOfCircle(p1, p_mid, p2)
                edge_builder = BRepBuilderAPI_MakeEdge(arc.Value())
                if edge_builder.IsDone():
                    return edge_builder.Edge()
            except Exception as e:
                _log.warning("ARC construction failed: %s", e)
            return None

        elif cmd.command_type == CommandType.CIRCLE:
            if len(cmd.params) < 3:
                _log.warning("CIRCLE command has insufficient params: %d", len(cmd.params))
                return None
            cx, cy, radius = cmd.params[:3]
            if abs(radius) < self.tolerance:
                _log.debug("Degenerate CIRCLE: zero radius")
                return None
            circle = gp_Circ(
                gp_Ax2(gp_Pnt(cx, cy, 0.0), gp_Dir(0, 0, 1)),
                abs(radius),
            )
            edge_builder = BRepBuilderAPI_MakeEdge(circle)
            if edge_builder.IsDone():
                return edge_builder.Edge()
            return None

        else:
            _log.debug("Unsupported command type for edge: %s", cmd.command_type.value)
            return None

    def _build_circle_face(self, cmd: CommandToken) -> Optional[TopoDS_Face]:
        """Build a face from a single CIRCLE command.

        Args:
            cmd: CIRCLE command token with [cx, cy, radius] params.

        Returns:
            TopoDS_Face if construction succeeds, None otherwise.
        """
        if len(cmd.params) < 3:
            return None

        cx, cy, radius = cmd.params[:3]
        if abs(radius) < self.tolerance:
            return None

        circle = gp_Circ(
            gp_Ax2(gp_Pnt(cx, cy, 0.0), gp_Dir(0, 0, 1)),
            abs(radius),
        )
        edge = BRepBuilderAPI_MakeEdge(circle).Edge()
        wire = BRepBuilderAPI_MakeWire(edge).Wire()

        plane = gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
        face_builder = BRepBuilderAPI_MakeFace(plane, wire, True)
        if face_builder.IsDone():
            return face_builder.Face()
        return None

    def _apply_extrusion(
        self,
        face: TopoDS_Face,
        extrude_params: List[float],
        existing_body: Optional[TopoDS_Shape] = None,
    ) -> TopoDS_Shape:
        """Apply a linear extrusion to a sketch face.

        Creates a solid by sweeping the face along the extrusion direction
        vector. If an existing body is provided, fuses the new solid with it.

        Args:
            face: 2D face to extrude.
            extrude_params: [dx, dy, dz, distance] direction and magnitude.
                The direction vector (dx, dy, dz) is normalized and scaled
                by distance.
            existing_body: Optional existing solid to fuse with.

        Returns:
            The resulting TopoDS_Shape after extrusion and optional fusion.

        Raises:
            RuntimeError: If extrusion or fusion fails.
        """
        if len(extrude_params) < 4:
            raise RuntimeError(
                f"EXTRUDE requires 4 params, got {len(extrude_params)}"
            )

        dx, dy, dz, distance = extrude_params[:4]

        # Normalize direction
        magnitude = math.sqrt(dx * dx + dy * dy + dz * dz)
        if magnitude < self.tolerance:
            # Default to Z-axis extrusion
            dx, dy, dz = 0.0, 0.0, 1.0
            magnitude = 1.0

        scale = abs(distance) / magnitude
        vec = gp_Vec(dx * scale, dy * scale, dz * scale)

        prism = BRepPrimAPI_MakePrism(face, vec)
        prism.Build()
        if not prism.IsDone():
            raise RuntimeError("BRepPrimAPI_MakePrism failed")

        new_solid = prism.Shape()

        if existing_body is not None:
            return self._apply_boolean(existing_body, new_solid, BooleanOpType.UNION)

        return new_solid

    def _apply_boolean(
        self,
        body1: TopoDS_Shape,
        body2: TopoDS_Shape,
        op_type: BooleanOpType,
    ) -> TopoDS_Shape:
        """Apply a Boolean operation between two shapes.

        Args:
            body1: First operand shape.
            body2: Second operand shape.
            op_type: Boolean operation type (union, cut, intersect).

        Returns:
            The resulting TopoDS_Shape after the Boolean operation.

        Raises:
            RuntimeError: If the Boolean operation fails.
        """
        if op_type == BooleanOpType.UNION:
            op = BRepAlgoAPI_Fuse(body1, body2)
        elif op_type == BooleanOpType.CUT:
            op = BRepAlgoAPI_Cut(body1, body2)
        elif op_type == BooleanOpType.INTERSECT:
            op = BRepAlgoAPI_Common(body1, body2)
        else:
            raise RuntimeError(f"Unknown Boolean operation: {op_type}")

        op.Build()
        if not op.IsDone():
            raise RuntimeError(f"Boolean {op_type.value} operation failed")

        result = op.Shape()
        if result.IsNull():
            raise RuntimeError(f"Boolean {op_type.value} produced null shape")

        return result

    def export_step(self, shape: Any, output_path: str) -> bool:
        """Export a shape to STEP format.

        Args:
            shape: TopoDS_Shape to export.
            output_path: File path for the output STEP file.

        Returns:
            True if export succeeded, False otherwise.
        """
        if not self.has_pythonocc:
            _log.error("Cannot export STEP: pythonocc not available")
            return False

        try:
            writer = STEPControl_Writer()
            Interface_Static.SetCVal("write.step.schema", "AP214")
            writer.Transfer(shape, STEPControl_AsIs)
            status = writer.Write(output_path)

            if status == 1:  # IFSelect_RetDone
                _log.info("STEP file exported to: %s", output_path)
                return True
            else:
                _log.error("STEP export failed with status: %s", status)
                return False
        except Exception as e:
            _log.error("STEP export error: %s", e)
            return False
