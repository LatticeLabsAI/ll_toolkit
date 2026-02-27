"""Execute CommandSequenceProposal objects by converting quantized token sequences to OCC geometry.

This module provides functionality to execute command sequences and produce TopoDS_Shape
geometry through sketch->extrude->boolean operations.
"""
from __future__ import annotations

import logging
from typing import Any

from ll_gen.proposals.command_proposal import CommandSequenceProposal

# Lazy imports for OCC
_OCC_AVAILABLE = False
try:
    from OCC.Core.gp import gp_Ax2, gp_Circ, gp_Pnt, gp_Pnt2d, gp_Dir, gp_Vec
    from OCC.Core.GC import GC_MakeArcOfCircle
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeWire,
        BRepBuilderAPI_MakeFace,
    )
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
    from OCC.Core.BRepAlgoAPI import (
        BRepAlgoAPI_Fuse,
        BRepAlgoAPI_Cut,
        BRepAlgoAPI_Common,
    )
    from OCC.Core.TopoDS import TopoDS_Shape
    from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
    from OCC.Core.TColgp import TColgp_HArray1OfPnt2d
    from OCC.Core.GeomAbs import GeomAbs_C2
    from OCC.Core.Geom2dAPI import Geom2dAPI_PointsToBSpline
    from OCC.Core.Geom2d import Geom2d_TrimmedCurve
    _OCC_AVAILABLE = True
except ImportError:
    pass

# Lazy import for cadling's CommandExecutor
_CADLING_EXECUTOR_AVAILABLE = False
_cadling_executor = None
CadlingCommandExecutor = None  # sentinel for unittest.mock.patch
try:
    from cadling.generation.reconstruction.command_executor import CommandExecutor as CadlingCommandExecutor
    _CADLING_EXECUTOR_AVAILABLE = True
except ImportError:
    pass

_log = logging.getLogger(__name__)


def execute_command_proposal(proposal: CommandSequenceProposal) -> Any:
    """Execute a CommandSequenceProposal to generate OCC geometry.

    Args:
        proposal: A CommandSequenceProposal object containing the command sequence
            to be executed.

    Returns:
        A TopoDS_Shape representing the generated geometry.

    Raises:
        RuntimeError: If execution fails or required OCC libraries are unavailable.
    """
    # Try to use cadling's CommandExecutor if available
    if _CADLING_EXECUTOR_AVAILABLE:
        try:
            _log.debug("Using cadling's CommandExecutor for execution")
            token_sequence = proposal.to_token_sequence()
            token_ids = token_sequence.token_ids
            executor = CadlingCommandExecutor()
            result = executor.execute(token_ids)
            _log.info("Successfully executed proposal via cadling executor")
            return result
        except Exception as e:
            _log.warning(
                f"Cadling executor failed, falling back to standalone execution: {e}"
            )
    
    # Standalone execution path
    if not _OCC_AVAILABLE:
        raise RuntimeError(
            "OCC libraries are not available. Cannot execute command proposal. "
            "Install python-occ to enable geometry execution."
        )
    
    _log.debug("Using standalone execution path")
    
    # Dequantize to get continuous parameters
    try:
        commands = proposal.dequantize()
    except Exception as e:
        _log.error(f"Failed to dequantize proposal: {e}")
        raise RuntimeError(f"Failed to dequantize command proposal: {e}") from e
    
    # Extract sketch groups
    try:
        sketch_groups = _extract_sketch_groups(commands)
    except Exception as e:
        _log.error(f"Failed to extract sketch groups: {e}")
        raise RuntimeError(f"Failed to extract sketch groups: {e}") from e
    
    # Execute sketch groups and accumulate results
    current_shape = None
    execution_step = 0
    
    for group_idx, group in enumerate(sketch_groups):
        try:
            group_shape = _execute_sketch_group(group, execution_step)
            execution_step += len(group)
            
            if group_shape is None:
                _log.warning(f"Sketch group {group_idx} produced no geometry")
                continue
            
            if current_shape is None:
                current_shape = group_shape
                _log.debug(f"Initialized base shape from sketch group {group_idx}")
            else:
                # Apply boolean operation
                current_shape = _apply_boolean_operation(
                    current_shape, group_shape, group
                )
                _log.debug(
                    f"Applied boolean operation after sketch group {group_idx}"
                )
        except Exception as e:
            _log.error(
                f"Error executing sketch group {group_idx} at step {execution_step}: {e}"
            )
            raise RuntimeError(
                f"Error executing sketch group {group_idx} at step {execution_step}: {e}"
            ) from e
    
    if current_shape is None:
        _log.error("No valid geometry generated from command sequence")
        raise RuntimeError("No valid geometry generated from command sequence")
    
    _log.info("Successfully executed command proposal")
    return current_shape


def _extract_sketch_groups(commands: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group commands by sketch loops.

    Commands are organized between SOL (Start Of Loop) markers. Each group
    contains sketch drawing commands (LINE, ARC, CIRCLE) followed by extrusion.

    Args:
        commands: List of dequantized command dictionaries.

    Returns:
        List of sketch groups, where each group is a list of commands.
    """
    groups = []
    current_group = []
    
    for cmd in commands:
        cmd_type = cmd.get("type", "")
        
        if cmd_type == "SOL":
            # Start a new group
            if current_group:
                groups.append(current_group)
            current_group = [cmd]
        else:
            current_group.append(cmd)
            
            # EOL marks the end of a sketch group
            if cmd_type == "EOL":
                groups.append(current_group)
                current_group = []
    
    # Add any remaining commands
    if current_group:
        groups.append(current_group)
    
    _log.debug(f"Extracted {len(groups)} sketch groups from {len(commands)} commands")
    return groups


def _execute_sketch_group(
    group: list[dict[str, Any]], step_offset: int = 0
) -> TopoDS_Shape | None:
    """Execute a single sketch group to produce a 3D shape.

    A sketch group contains:
    1. SOL (Start Of Loop) command
    2. Sketch commands (LINE, ARC, CIRCLE)
    3. EOL (End Of Loop) command
    4. EXTRUDE command (optional)

    Args:
        group: List of commands in the sketch group.
        step_offset: Offset for step numbering in error messages.

    Returns:
        A TopoDS_Shape representing the extruded sketch, or None if no valid shape.

    Raises:
        RuntimeError: If sketch or extrude execution fails.
    """
    if not group:
        _log.warning("Empty sketch group")
        return None
    
    # Separate sketch commands and extrude command
    sketch_commands = []
    extrude_command = None
    
    for cmd in group:
        cmd_type = cmd.get("type", "")
        if cmd_type == "EXTRUDE":
            extrude_command = cmd
        elif cmd_type != "SOL" and cmd_type != "EOL":
            sketch_commands.append(cmd)
    
    if not sketch_commands:
        _log.warning("No sketch commands in group")
        return None
    
    # Build sketch face from commands
    try:
        sketch_face = _build_sketch_face(sketch_commands, step_offset)
    except Exception as e:
        _log.error(f"Failed to build sketch face: {e}")
        raise RuntimeError(f"Failed to build sketch face: {e}") from e
    
    if sketch_face is None:
        _log.warning("Failed to create valid sketch face")
        return None
    
    # Extrude if extrude command exists
    if extrude_command:
        try:
            extruded_shape = _extrude_sketch(sketch_face, extrude_command, step_offset)
            return extruded_shape
        except Exception as e:
            _log.error(f"Failed to extrude sketch: {e}")
            raise RuntimeError(f"Failed to extrude sketch: {e}") from e
    else:
        _log.debug("No extrude command, returning sketch face")
        return sketch_face


def _build_sketch_face(
    sketch_commands: list[dict[str, Any]], step_offset: int = 0
) -> TopoDS_Shape | None:
    """Build a 2D sketch face from sketch commands.

    Args:
        sketch_commands: List of sketch commands (LINE, ARC, CIRCLE).
        step_offset: Offset for step numbering in error messages.

    Returns:
        A TopoDS_Shape representing the sketch face, or None if unsuccessful.

    Raises:
        RuntimeError: If edge or wire creation fails.
    """
    edges = []
    
    for step_idx, cmd in enumerate(sketch_commands):
        cmd_type = cmd.get("type", "")
        params = cmd.get("params", {})
        
        try:
            if cmd_type == "LINE":
                edge = _create_line_edge(params)
            elif cmd_type == "ARC":
                edge = _create_arc_edge(params)
            elif cmd_type == "CIRCLE":
                edge = _create_circle_edge(params)
            else:
                _log.warning(f"Unknown sketch command type: {cmd_type}")
                continue
            
            if edge is not None:
                edges.append(edge)
            else:
                _log.warning(
                    f"Failed to create edge for {cmd_type} at step {step_offset + step_idx}"
                )
        except Exception as e:
            _log.error(
                f"Error creating {cmd_type} edge at step {step_offset + step_idx}: {e}"
            )
            raise RuntimeError(
                f"Error creating {cmd_type} edge at step {step_offset + step_idx}: {e}"
            ) from e
    
    if not edges:
        _log.warning("No valid edges created for sketch")
        return None
    
    # Create wire from edges
    try:
        wire_maker = BRepBuilderAPI_MakeWire()
        for edge in edges:
            wire_maker.Add(edge)
        
        if not wire_maker.IsDone():
            _log.error("Failed to create wire from edges")
            return None
        
        wire = wire_maker.Wire()
        _log.debug(f"Created wire with {len(edges)} edges")
    except Exception as e:
        _log.error(f"Error creating wire: {e}")
        raise RuntimeError(f"Error creating wire: {e}") from e
    
    # Create face from wire
    try:
        face_maker = BRepBuilderAPI_MakeFace(wire, False)
        
        if not face_maker.IsDone():
            _log.error("Failed to create face from wire")
            return None
        
        face = face_maker.Face()
        _log.debug("Created face from wire")
        return face
    except Exception as e:
        _log.error(f"Error creating face: {e}")
        raise RuntimeError(f"Error creating face: {e}") from e


def _create_line_edge(params: dict[str, Any]) -> TopoDS_Shape | None:
    """Create a 2D line edge.

    Args:
        params: Parameter dictionary containing x1, y1, x2, y2.

    Returns:
        A TopoDS_Shape representing the line edge, or None if unsuccessful.
    """
    try:
        x1 = float(params.get("x1", 0.0))
        y1 = float(params.get("y1", 0.0))
        x2 = float(params.get("x2", 0.0))
        y2 = float(params.get("y2", 0.0))
        
        p1 = gp_Pnt2d(x1, y1)
        p2 = gp_Pnt2d(x2, y2)
        
        edge_maker = BRepBuilderAPI_MakeEdge(p1, p2)
        
        if not edge_maker.IsDone():
            _log.warning(f"Failed to create line edge from ({x1}, {y1}) to ({x2}, {y2})")
            return None
        
        edge = edge_maker.Edge()
        _log.debug(f"Created line edge from ({x1}, {y1}) to ({x2}, {y2})")
        return edge
    except Exception as e:
        _log.error(f"Error creating line edge: {e}")
        raise RuntimeError(f"Error creating line edge: {e}") from e


def _create_arc_edge(params: dict[str, Any]) -> TopoDS_Shape | None:
    """Create a 2D arc edge from three points.

    The arc is defined by start point, midpoint (used to compute arc properties),
    and end point.

    Args:
        params: Parameter dictionary containing x_start, y_start, x_mid, y_mid,
            x_end, y_end.

    Returns:
        A TopoDS_Shape representing the arc edge, or None if unsuccessful.
    """
    try:
        x_start = float(params.get("x_start", 0.0))
        y_start = float(params.get("y_start", 0.0))
        x_mid = float(params.get("x_mid", 0.0))
        y_mid = float(params.get("y_mid", 0.0))
        x_end = float(params.get("x_end", 0.0))
        y_end = float(params.get("y_end", 0.0))
        
        p_start = gp_Pnt(x_start, y_start, 0.0)
        p_mid = gp_Pnt(x_mid, y_mid, 0.0)
        p_end = gp_Pnt(x_end, y_end, 0.0)

        # Create arc using GC_MakeArcOfCircle with three 3D points
        arc_maker = GC_MakeArcOfCircle(p_start, p_mid, p_end)
        if not arc_maker.IsDone():
            _log.warning(
                f"Failed to create arc edge from ({x_start}, {y_start}) "
                f"through ({x_mid}, {y_mid}) to ({x_end}, {y_end})"
            )
            return None

        arc_curve = arc_maker.Value()
        edge_maker = BRepBuilderAPI_MakeEdge(arc_curve)

        if not edge_maker.IsDone():
            _log.warning(
                f"Failed to create arc edge from ({x_start}, {y_start}) "
                f"through ({x_mid}, {y_mid}) to ({x_end}, {y_end})"
            )
            return None

        edge = edge_maker.Edge()
        _log.debug(
            f"Created arc edge from ({x_start}, {y_start}) "
            f"through ({x_mid}, {y_mid}) to ({x_end}, {y_end})"
        )
        return edge
    except Exception as e:
        _log.error(f"Error creating arc edge: {e}")
        raise RuntimeError(f"Error creating arc edge: {e}") from e


def _create_circle_edge(params: dict[str, Any]) -> TopoDS_Shape | None:
    """Create a 2D circle edge.

    Args:
        params: Parameter dictionary containing cx (center x), cy (center y),
            and r (radius).

    Returns:
        A TopoDS_Shape representing the circle edge, or None if unsuccessful.
    """
    try:
        cx = float(params.get("cx", 0.0))
        cy = float(params.get("cy", 0.0))
        r = float(params.get("r", 1.0))
        
        if r <= 0:
            _log.warning(f"Invalid circle radius: {r}")
            return None
        
        # Create 3D circle on XY plane at the given center
        axis = gp_Ax2(gp_Pnt(cx, cy, 0.0), gp_Dir(0, 0, 1))
        circle = gp_Circ(axis, r)

        # Create circle edge
        edge_maker = BRepBuilderAPI_MakeEdge(circle)
        
        if not edge_maker.IsDone():
            _log.warning(f"Failed to create circle edge at ({cx}, {cy}) with radius {r}")
            return None
        
        edge = edge_maker.Edge()
        _log.debug(f"Created circle edge at ({cx}, {cy}) with radius {r}")
        return edge
    except Exception as e:
        _log.error(f"Error creating circle edge: {e}")
        raise RuntimeError(f"Error creating circle edge: {e}") from e


def _extrude_sketch(
    sketch_face: TopoDS_Shape,
    extrude_command: dict[str, Any],
    step_offset: int = 0,
) -> TopoDS_Shape:
    """Extrude a sketch face to create a 3D solid.

    Args:
        sketch_face: The 2D sketch face to extrude.
        extrude_command: Dictionary containing extrude parameters including
            dx, dy, dz (direction vector) and extent (extrusion distance).
        step_offset: Offset for step numbering in error messages.

    Returns:
        A TopoDS_Shape representing the extruded solid.

    Raises:
        RuntimeError: If extrusion fails.
    """
    try:
        params = extrude_command.get("params", {})
        
        dx = float(params.get("dx", 0.0))
        dy = float(params.get("dy", 0.0))
        dz = float(params.get("dz", 1.0))
        extent = float(params.get("extent", 1.0))
        
        # Normalize direction vector
        magnitude = (dx * dx + dy * dy + dz * dz) ** 0.5
        if magnitude == 0:
            _log.warning("Zero-magnitude extrusion direction, using default Z")
            direction = gp_Vec(0, 0, 1)
        else:
            direction = gp_Vec(dx / magnitude, dy / magnitude, dz / magnitude)
        
        # Apply extrusion
        prism_maker = BRepPrimAPI_MakePrism(sketch_face, direction * extent)
        
        if not prism_maker.IsDone():
            _log.error("Failed to create prism")
            raise RuntimeError("Failed to create prism from sketch")
        
        extruded_shape = prism_maker.Shape()
        _log.debug(
            f"Extruded sketch with direction ({dx}, {dy}, {dz}) "
            f"and extent {extent}"
        )
        return extruded_shape
    except Exception as e:
        _log.error(f"Error extruding sketch at step {step_offset}: {e}")
        raise RuntimeError(f"Error extruding sketch at step {step_offset}: {e}") from e


def _apply_boolean_operation(
    base_shape: TopoDS_Shape,
    tool_shape: TopoDS_Shape,
    group: list[dict[str, Any]],
) -> TopoDS_Shape:
    """Apply a boolean operation between two shapes.

    The boolean operation type is determined by examining the EXTRUDE command's
    params. The param at index 4 indicates the operation:
    - 0: New (no operation, return tool_shape)
    - 1: Union
    - 2: Cut (Subtraction)
    - 3: Intersection

    Args:
        base_shape: The base shape for the boolean operation.
        tool_shape: The tool shape for the boolean operation.
        group: The sketch group containing the extrude command.

    Returns:
        A TopoDS_Shape representing the result of the boolean operation.

    Raises:
        RuntimeError: If the boolean operation fails.
    """
    try:
        # Find extrude command in group
        extrude_command = None
        for cmd in group:
            if cmd.get("type") == "EXTRUDE":
                extrude_command = cmd
                break
        
        if extrude_command is None:
            _log.warning("No extrude command found for boolean operation")
            return base_shape
        
        # Extract operation type from parameters
        parameters = extrude_command.get("parameters", [])

        if len(parameters) > 13:
            # Parameter index 13 is the boolean operation type
            # After dequantization it's a float; round to nearest int
            operation_type = int(round(parameters[13]))
            # Clamp to valid range [0, 3]
            operation_type = max(0, min(3, operation_type))
        else:
            _log.warning("Boolean operation type not found, defaulting to union")
            operation_type = 1
        
        if operation_type == 0:
            # New: return tool shape
            _log.debug("Boolean operation: New (return tool shape)")
            return tool_shape
        elif operation_type == 1:
            # Union
            _log.debug("Boolean operation: Union")
            union_maker = BRepAlgoAPI_Fuse(base_shape, tool_shape)
            if not union_maker.IsDone():
                _log.error("Failed to create union")
                raise RuntimeError("Failed to create union")
            return union_maker.Shape()
        elif operation_type == 2:
            # Cut (Subtraction)
            _log.debug("Boolean operation: Cut")
            cut_maker = BRepAlgoAPI_Cut(base_shape, tool_shape)
            if not cut_maker.IsDone():
                _log.error("Failed to create cut")
                raise RuntimeError("Failed to create cut")
            return cut_maker.Shape()
        elif operation_type == 3:
            # Intersection
            _log.debug("Boolean operation: Intersection")
            intersect_maker = BRepAlgoAPI_Common(base_shape, tool_shape)
            if not intersect_maker.IsDone():
                _log.error("Failed to create intersection")
                raise RuntimeError("Failed to create intersection")
            return intersect_maker.Shape()
        else:
            _log.warning(f"Unknown boolean operation type: {operation_type}, defaulting to union")
            union_maker = BRepAlgoAPI_Fuse(base_shape, tool_shape)
            if not union_maker.IsDone():
                raise RuntimeError("Failed to create union as fallback")
            return union_maker.Shape()
    except Exception as e:
        _log.error(f"Error applying boolean operation: {e}")
        raise RuntimeError(f"Error applying boolean operation: {e}") from e
