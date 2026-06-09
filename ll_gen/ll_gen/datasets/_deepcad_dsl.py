"""Parser for the DeepCAD-DSL textual CAD representation.

The public ``palapav/DeepCAD-DSL`` dataset (a DSL rendering of the DeepCAD
sketch-and-extrude corpus) encodes each model as a string like::

    SKETCH id=1 plane=XZ origin=(0, 0, 0)
      LOOP type=OUTER
        LINE start=(-0.0254, -0.0254) end=(0.0254, -0.0254)
        CIRCLE center=(0, 0) radius=0.006
        ARC center=(0, 0) radius=0.00585 start_angle=0 end_angle=2.22
      END_LOOP
    END_SKETCH
    EXTRUDE id=1 sketch=1 distance=0.0254 operation=NEW_BODY

This module flattens that DSL into the ``[{"type", "params"}]`` command-sequence
schema consumed by ``ll_gen.datasets._tokenization.tokenize_command_sequence``
(vocabulary: SOL, LINE, ARC, CIRCLE, EXTRUDE, EOS).  Each ``LOOP`` opens with a
``SOL`` (start-of-loop) marker; ``EXTRUDE`` carries its extrusion distance; the
sequence ends with ``EOS``.

Parameter conventions (match ``ll_gen.generators.base.PARAMETER_MASKS`` and the
disposal executor):
    LINE   -> [x1, y1, x2, y2]
    CIRCLE -> [cx, cy, r]
    ARC    -> [x_start, y_start, x_end, y_end, cx, cy]   (endpoints from angles)
    EXTRUDE-> [distance]
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

_log = logging.getLogger(__name__)

# A float literal, including optional sign and scientific notation.
_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

# One regex with a named alternative per DSL clause; ``finditer`` preserves the
# left-to-right command order of the source string.
_CLAUSE_RE = re.compile(
    r"(?P<loop>\bLOOP\b)"
    rf"|(?P<line>\bLINE\s+start=\(\s*(?P<lx1>{_NUM})\s*,\s*(?P<ly1>{_NUM})\s*\)"
    rf"\s+end=\(\s*(?P<lx2>{_NUM})\s*,\s*(?P<ly2>{_NUM})\s*\))"
    rf"|(?P<arc>\bARC\s+center=\(\s*(?P<ax>{_NUM})\s*,\s*(?P<ay>{_NUM})\s*\)"
    rf"\s+radius=(?P<ar>{_NUM})\s+start_angle=(?P<asa>{_NUM})\s+end_angle=(?P<aea>{_NUM}))"
    rf"|(?P<circle>\bCIRCLE\s+center=\(\s*(?P<cx>{_NUM})\s*,\s*(?P<cy>{_NUM})\s*\)"
    rf"\s+radius=(?P<cr>{_NUM}))"
    rf"|(?P<extrude>\bEXTRUDE\s+id=\d+\s+sketch=\d+\s+distance=(?P<ed>{_NUM}))"
)


def _arc_endpoints(
    cx: float, cy: float, r: float, start_angle: float, end_angle: float
) -> tuple[float, float, float, float]:
    """Convert an (center, radius, angles) arc to its start/end points."""
    x_start = cx + r * math.cos(start_angle)
    y_start = cy + r * math.sin(start_angle)
    x_end = cx + r * math.cos(end_angle)
    y_end = cy + r * math.sin(end_angle)
    return (x_start, y_start, x_end, y_end)


def parse_deepcad_dsl(dsl: str) -> list[dict[str, Any]]:
    """Flatten a DeepCAD-DSL string into a ``[{"type", "params"}]`` sequence.

    Args:
        dsl: A DeepCAD-DSL ``target`` string.

    Returns:
        Ordered command list using the SOL/LINE/ARC/CIRCLE/EXTRUDE/EOS
        vocabulary; terminated by an ``EOS`` command.  Empty (no ``EOS``) when
        the string contains no recognizable geometry.
    """
    commands: list[dict[str, Any]] = []
    if not dsl:
        return commands

    for match in _CLAUSE_RE.finditer(dsl):
        kind = match.lastgroup
        # lastgroup is the innermost matched group; recover the clause kind by
        # checking which top-level alternative actually matched.
        if match.group("loop") is not None:
            commands.append({"type": "SOL", "params": []})
        elif match.group("line") is not None:
            commands.append(
                {
                    "type": "LINE",
                    "params": [
                        float(match.group("lx1")),
                        float(match.group("ly1")),
                        float(match.group("lx2")),
                        float(match.group("ly2")),
                    ],
                }
            )
        elif match.group("arc") is not None:
            x_start, y_start, x_end, y_end = _arc_endpoints(
                float(match.group("ax")),
                float(match.group("ay")),
                float(match.group("ar")),
                float(match.group("asa")),
                float(match.group("aea")),
            )
            commands.append(
                {
                    "type": "ARC",
                    "params": [
                        x_start,
                        y_start,
                        x_end,
                        y_end,
                        float(match.group("ax")),
                        float(match.group("ay")),
                    ],
                }
            )
        elif match.group("circle") is not None:
            commands.append(
                {
                    "type": "CIRCLE",
                    "params": [
                        float(match.group("cx")),
                        float(match.group("cy")),
                        float(match.group("cr")),
                    ],
                }
            )
        elif match.group("extrude") is not None:
            commands.append({"type": "EXTRUDE", "params": [float(match.group("ed"))]})
        else:  # pragma: no cover - defensive; every branch above is exhaustive
            _log.debug("Unhandled DSL clause group: %s", kind)

    if commands:
        commands.append({"type": "EOS", "params": []})
    return commands
