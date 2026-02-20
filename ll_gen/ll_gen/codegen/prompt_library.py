"""System prompts and few-shot examples for CAD code generation.

This module contains compressed API references and working code examples
for CadQuery and OpenSCAD. These are used to construct system prompts
and repair prompts for the LLM-based code generators.

All CadQuery examples are syntactically valid and follow the standard
pattern:
    from cadquery import Workplane as cq
    result = cq("XY").box(...).faces(...).hole(...)
    result.val()
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Optional

from ll_gen.config import ErrorCategory


# ---------------------------------------------------------------------------
# API References
# ---------------------------------------------------------------------------

CADQUERY_API_REFERENCE = """
CadQuery Workplane API Reference (Compressed)

INITIALIZATION:
  cq("XY"), cq("XZ"), cq("YZ")  # Create 2D workplane on a plane
  cq().box(length, width, height)  # Create rectangular solid
  cq().sphere(radius)  # Create sphere
  cq().cylinder(height, radius)  # Create cylinder
  
SKETCHING (on current workplane):
  .circle(radius)  # 2D circle
  .rect(width, height)  # 2D rectangle
  .polygon(n_sides, radius)  # Regular n-gon
  .line(x, y)  # Line segment
  .arc_to(x, y)  # Arc to point
  .close()  # Close sketch into wire
  
SELECTION & FILTERING:
  .faces()  # All faces
  .faces(">Z")  # Faces in +Z direction
  .faces("|Z")  # Faces normal to Z
  .faces("#Z")  # Largest face in Z
  .edges()  # All edges
  .edges("|Z")  # Edges on Z plane
  .vertices()  # All vertices
  .last()  # Last created feature
  .first()  # First object
  
FEATURE OPERATIONS:
  .hole(diameter)  # Drill cylindrical hole
  .cboreHole(diameter, cbore_diameter, cbore_depth)  # Counterbore
  .cskHole(diameter, csk_diameter, csk_angle)  # Countersink
  .fillet(radius)  # Round edges
  .chamfer(length)  # Bevel edges
  .shell(thickness)  # Hollow out
  
3D OPERATIONS:
  .extrude(distance)  # Extrude sketch
  .cut(solid)  # Boolean subtract
  .union(solid)  # Boolean add
  .intersect(solid)  # Boolean intersection
  
POSITIONING:
  .translate(x, y, z)  # Shift in space
  .rotate(axis, point, angle)  # Rotate around axis
  .mirror(plane)  # Mirror across plane
  
EXPORT:
  .val()  # Get TopoDS_Shape result
  .exportStep(filepath)  # Export to STEP file
"""

CADQUERY_EXAMPLES: Dict[str, str] = {
    "bracket": """
from cadquery import Workplane as cq

# L-shaped mounting bracket with bolt holes
base = (
    cq("XY")
    .box(50, 50, 10)  # Base plate
    .faces(">Z")
    .workplane()
    .transformed(offset=(0, 0, 0))
    .box(20, 10, 40)  # Vertical arm
)

# Add bolt holes (M4)
result = (
    base
    .faces(">Z")
    .workplane()
    .rarray(xSpacing=30, ySpacing=30, nRows=2, nCols=2)
    .hole(4.5)
    .faces(">Y")
    .workplane()
    .rarray(xSpacing=10, ySpacing=15, nRows=2, nCols=1)
    .hole(4.5)
)

result.val()
""",
    "gear": """
from cadquery import Workplane as cq
import math

# Simple spur gear with 16 teeth
num_teeth = 16
pitch_radius = 20
tooth_height = 5
bore_diameter = 8

# Create gear teeth as rectangular extrusions
base = cq("XY").circle(pitch_radius)
gear = (
    base
    .extrude(tooth_height)
    .faces(">Z")
    .workplane()
)

# Create alternating teeth using a for-loop pattern
for i in range(num_teeth):
    angle = (360 / num_teeth) * i
    gear = (
        gear
        .transformed(offset=(pitch_radius * 0.9, 0, 0), rotate=(0, 0, angle))
        .box(3, 4, tooth_height + 1)
        .transformed(rotate=(0, 0, -angle))
    )

# Add center bore
result = gear.faces(">Z").workplane().hole(bore_diameter)

result.val()
""",
    "enclosure": """
from cadquery import Workplane as cq

# Electronic enclosure with removable lid
wall_thickness = 3
width = 100
depth = 80
height = 60

# Main body
body = (
    cq("XY")
    .box(width, depth, height)
    .faces("<Z")
    .workplane(invert=True)
    .box(width - 2*wall_thickness, depth - 2*wall_thickness, height - wall_thickness)
)

# Add ventilation holes on sides
result = (
    body
    .faces(">X")
    .workplane()
    .rarray(xSpacing=20, ySpacing=15, nRows=3, nCols=2)
    .hole(8)
    .faces("<X")
    .workplane()
    .rarray(xSpacing=20, ySpacing=15, nRows=3, nCols=2)
    .hole(8)
)

# Add mounting bosses at corners
result = (
    result
    .faces(">Z")
    .workplane()
    .rarray(xSpacing=width-10, ySpacing=depth-10, nRows=2, nCols=2)
    .circle(6)
    .extrude(5)
)

result.val()
""",
    "hinge": """
from cadquery import Workplane as cq

# Simple hinge with pin hole
knuckle_diameter = 10
knuckle_length = 20
pin_diameter = 4
flap_width = 40
flap_thickness = 3

# Create left flap
left_flap = (
    cq("XY")
    .box(flap_width, knuckle_length, flap_thickness)
)

# Create knuckle (cylindrical barrel)
knuckle = (
    cq("XZ")
    .cylinder(knuckle_length, knuckle_diameter / 2)
    .translate(((flap_width / 2) - (knuckle_diameter / 2), 0, 0))
)

# Combine flap and knuckle
result = left_flap.union(knuckle)

# Drill pin hole through knuckle
result = (
    result
    .faces(">Z")
    .workplane()
    .center(0, 0)
    .hole(pin_diameter)
)

result.val()
""",
    "spacer": """
from cadquery import Workplane as cq

# Cylindrical spacer with center hole and chamfered edges
outer_diameter = 12
inner_diameter = 6
height = 10
chamfer_radius = 0.5

# Create the spacer body
spacer = (
    cq("XY")
    .cylinder(height, outer_diameter / 2)
    .faces("|Z")
    .chamfer(chamfer_radius)
)

# Drill center hole
result = (
    spacer
    .faces(">Z")
    .workplane()
    .hole(inner_diameter)
)

# Chamfer top and bottom of center hole
result = (
    result
    .edges("|Z")
    .chamfer(0.3)
)

result.val()
""",
}

ERROR_RECOVERY_TEMPLATES: Dict[str, str] = {
    ErrorCategory.INVALID_PARAMS: """
The code failed with invalid parameters. Review the error message and adjust:
- Numeric values (dimensions, angles, radii) may be out of valid ranges
- Ensure all diameters and heights are positive and reasonable
- Check that array spacing and counts are sensible for the part size
- Verify that selection queries like .faces(">Z") actually match existing geometry

Fix the invalid parameters and regenerate the code.
""",
    ErrorCategory.TOPOLOGY_ERROR: """
TOPOLOGY ERROR: The code produced an invalid or degenerate shape. This often means:
- A sketch was not properly closed before extrusion
- A face selection returned no results (e.g., .faces(">Z") when no upward-facing face exists)
- An extrusion distance was zero or nearly zero
- Nested operations created self-touching or overlapping faces

Ensure all sketches are closed with .close(), all extrusions have positive height,
and geometry operations produce non-degenerate solids. Avoid operations on faces
that don't exist.
""",
    ErrorCategory.BOOLEAN_FAILURE: """
A boolean operation (union, cut, intersect) failed because the solids do not
properly interact. Common causes:
- The two solids don't actually overlap or touch where expected
- Floating-point tolerance issues from multiple operations
- One solid is entirely outside the other when cut() was intended

Adjust the geometry to ensure solids actually intersect, or increase the overlap
region. Consider using .shell() or .fillet() to smooth boundaries before boolean ops.
""",
    ErrorCategory.SELF_INTERSECTION: """
The generated shape has self-intersecting faces or edges. This typically means:
- A sketch polygon or path crosses itself
- Fillets or chamfers were too large for the available space
- Competing geometric constraints created an impossible shape

Simplify the sketch to avoid crossings, reduce fillet/chamfer radii, or break
the operation into separate steps.
""",
    ErrorCategory.DEGENERATE_SHAPE: """
The geometry collapsed into a zero-volume or zero-area shape:
- All vertices colinear (created a line instead of area)
- All points in the same plane (cannot extrude 0 distance)
- An array or pattern created overlapping identical geometry

Ensure sketch points form actual 2D regions before extrusion, array patterns
have positive spacing, and operations preserve dimensionality.
""",
    ErrorCategory.TOLERANCE_VIOLATION: """
Internal geometric tolerances were violated, usually from:
- Very large geometry combined with very small features (scale mismatch)
- Fillet/chamfer radii that violate CAD kernel constraints
- Extreme aspect ratios in extrusions or arrays

Scale the part more reasonably, use more moderate fillet/chamfer values, and
avoid extreme geometric ratios. Consider breaking the part into sub-assemblies.
""",
}


# ---------------------------------------------------------------------------
# System Prompt Assembly
# ---------------------------------------------------------------------------

def get_system_prompt(
    backend: str = "cadquery",
    error_context: Optional[Dict] = None,
    include_examples: bool = True,
) -> str:
    """Assemble a complete system prompt for code generation.

    This function composes a multi-part system prompt that includes:
    1. Core instructions for the backend language
    2. Compressed API reference
    3. Few-shot examples (if requested)
    4. Error recovery guidance (if error_context provided)

    Args:
        backend: The target backend ("cadquery" or "openscad"). Defaults
            to "cadquery".
        error_context: Optional dictionary with:
            - "category" (ErrorCategory): Type of error encountered
            - "message" (str): Human-readable error description
            If provided, adds error-specific recovery instructions.
        include_examples: Whether to include few-shot examples.

    Returns:
        A formatted system prompt string ready to send to the LLM.
    """
    if backend.lower() == "cadquery":
        return _assemble_cadquery_system_prompt(
            error_context=error_context,
            include_examples=include_examples,
        )
    elif backend.lower() == "openscad":
        return _assemble_openscad_system_prompt(
            error_context=error_context,
            include_examples=include_examples,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _assemble_cadquery_system_prompt(
    error_context: Optional[Dict] = None,
    include_examples: bool = True,
) -> str:
    """Assemble CadQuery-specific system prompt."""
    parts = [
        "You are an expert CadQuery code generator for mechanical CAD design.",
        "",
        "INSTRUCTIONS:",
        "- Generate valid, executable Python code that uses CadQuery",
        "- Always import: from cadquery import Workplane as cq",
        "- Always end with: result.val() to return the final shape",
        "- Use precise floating-point dimensions matching the user request",
        "- Chain methods fluently; avoid intermediate variables when possible",
        "- Use .faces('>Z'), .faces('|Z'), .faces('#Z') for smart face selection",
        "- Always close sketches with .close() before extrusion",
        "- Use .fillet() and .chamfer() to add realism to edges",
        "- For arrays: use .rarray(xSpacing, ySpacing, nRows, nCols)",
        "- Test that selections like .faces('>Z') will actually match geometry",
        "- Avoid creating degenerate or self-intersecting geometry",
        "",
        "API REFERENCE:",
        CADQUERY_API_REFERENCE,
        "",
    ]

    if include_examples:
        parts.append("WORKING EXAMPLES:")
        parts.append("")
        for name, code in CADQUERY_EXAMPLES.items():
            parts.append(f"--- {name.upper()} ---")
            parts.append(code)
            parts.append("")

    if error_context:
        category = error_context.get("category")
        if category and category in ERROR_RECOVERY_TEMPLATES:
            parts.append("ERROR RECOVERY GUIDANCE:")
            parts.append(ERROR_RECOVERY_TEMPLATES[category])
            parts.append("")

    return "\n".join(parts)


def _assemble_openscad_system_prompt(
    error_context: Optional[Dict] = None,
    include_examples: bool = True,
) -> str:
    """Assemble OpenSCAD-specific system prompt."""
    parts = [
        "You are an expert OpenSCAD code generator for parametric CAD design.",
        "",
        "INSTRUCTIONS:",
        "- Generate valid, executable OpenSCAD code",
        "- Use proper syntax: every statement ends with ;",
        "- Use translate(), rotate(), scale() for positioning",
        "- Use cube(), sphere(), cylinder() for primitives",
        "- Use union() {}, difference() {}, intersection() {} for booleans",
        "- Use linear_extrude() to create 3D shapes from 2D",
        "- Use rotate_extrude() for rotational extrusions",
        "- Comment your code with // for clarity",
        "- Use variables for repeated dimensions: w=50; h=30;",
        "- Avoid unbounded geometry; all shapes must be fully defined",
        "- Test for balanced braces and parentheses",
        "",
        "BASIC PRIMITIVES:",
        "  cube([width, depth, height], center=true);",
        "  sphere(r=radius);",
        "  cylinder(h=height, r=radius, center=true);",
        "  circle(r=radius);",
        "  square([width, height], center=true);",
        "  polygon(points=[[x1,y1], [x2,y2], ...]);",
        "",
        "POSITIONING & TRANSFORMS:",
        "  translate([x, y, z]) { ... }",
        "  rotate(angle, [x, y, z]) { ... }",
        "  scale([sx, sy, sz]) { ... }",
        "  mirror([x, y, z]) { ... }",
        "",
        "BOOLEAN OPERATIONS:",
        "  union() { shape1(); shape2(); }",
        "  difference() { shape1(); shape2(); }",
        "  intersection() { shape1(); shape2(); }",
        "",
        "3D EXTRUSION:",
        "  linear_extrude(height=h) { 2d_shape(); }",
        "  rotate_extrude() { 2d_shape(); }",
        "",
    ]

    if include_examples:
        parts.append("WORKING EXAMPLES:")
        parts.append("")
        parts.append("--- SIMPLE BRACKET ---")
        parts.append("""
difference() {
  cube([50, 50, 10], center=true);
  translate([-15, -15, 0]) cylinder(h=12, r=2.5, center=true);
  translate([15, -15, 0]) cylinder(h=12, r=2.5, center=true);
  translate([-15, 15, 0]) cylinder(h=12, r=2.5, center=true);
  translate([15, 15, 0]) cylinder(h=12, r=2.5, center=true);
}
""")
        parts.append("")
        parts.append("--- PARAMETRIC SPACER ---")
        parts.append("""
outer_d = 12;
inner_d = 6;
height = 10;

difference() {
  cylinder(h=height, r=outer_d/2, center=true);
  cylinder(h=height+1, r=inner_d/2, center=true);
}
""")
        parts.append("")

    if error_context:
        parts.append("ERROR RECOVERY GUIDANCE:")
        parts.append(
            "Review syntax carefully: braces must be balanced, all statements must end with ;"
        )
        parts.append("")

    return "\n".join(parts)


def get_repair_prompt(
    code: str,
    error_message: str,
    suggestion: str = "",
) -> str:
    """Construct a prompt for repairing failed code.

    This is used when the code generator detects a previous attempt failed
    and needs to regenerate with the error message as context.

    Args:
        code: The previous code that failed.
        error_message: The error message or validation failure.
        suggestion: Optional hint for what to fix.

    Returns:
        A formatted repair prompt.
    """
    parts = [
        "The following code failed with an error. Please fix it and regenerate:",
        "",
        "FAILED CODE:",
        "```",
        code,
        "```",
        "",
        "ERROR:",
        error_message,
        "",
    ]

    if suggestion:
        parts.append("SUGGESTION:")
        parts.append(suggestion)
        parts.append("")

    parts.extend([
        "Please regenerate the code to fix the error. Ensure the output is",
        "syntactically valid and addresses the root cause of the failure.",
    ])

    return "\n".join(parts)
