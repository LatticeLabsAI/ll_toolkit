"""Unit tests for the DeepCAD-DSL parser (SPEC-1 M3, T3.1).

Exercises ``parse_deepcad_dsl`` on real ``palapav/DeepCAD-DSL`` target strings
(no network) and confirms it flattens them into the SOL/LINE/ARC/CIRCLE/EXTRUDE/
EOS command schema the tokenizer consumes.
"""

from __future__ import annotations

import math

from ll_gen.datasets._deepcad_dsl import parse_deepcad_dsl


def _types(commands):
    return [c["type"] for c in commands]


def test_empty_string_yields_no_commands() -> None:
    assert parse_deepcad_dsl("") == []
    assert parse_deepcad_dsl("   ") == []


def test_garbage_yields_no_commands() -> None:
    assert parse_deepcad_dsl("not a real dsl string at all") == []


def test_single_line_loop_box() -> None:
    dsl = (
        "SKETCH id=1 plane=XZ origin=(0, 0, 0)   LOOP type=OUTER     "
        "LINE start=(-0.0254, -0.0254) end=(0.0254, -0.0254)     "
        "LINE start=(0.0254, -0.0254) end=(0.0254, 0.0254)     "
        "LINE start=(-0.0254, 0.0254) end=(0.0254, 0.0254)     "
        "LINE start=(-0.0254, -0.0254) end=(-0.0254, 0.0254)   END_LOOP "
        "END_SKETCH EXTRUDE id=1 sketch=1 distance=0.0254 symmetric=true "
        "operation=NEW_BODY"
    )
    commands = parse_deepcad_dsl(dsl)
    assert _types(commands) == [
        "SOL",
        "LINE",
        "LINE",
        "LINE",
        "LINE",
        "EXTRUDE",
        "EOS",
    ]
    # First LINE params are [x1, y1, x2, y2].
    assert commands[1]["params"] == [-0.0254, -0.0254, 0.0254, -0.0254]
    # EXTRUDE carries the distance as its first parameter.
    assert commands[5]["params"] == [0.0254]


def test_circle_params_are_center_and_radius() -> None:
    dsl = (
        "SKETCH id=1 plane=XY origin=(0, 0, 0)   LOOP type=OUTER     "
        "CIRCLE center=(-0.00978, 0.5) radius=0.0025   END_LOOP END_SKETCH "
        "EXTRUDE id=1 sketch=1 distance=0.054 operation=NEW_BODY"
    )
    commands = parse_deepcad_dsl(dsl)
    assert _types(commands) == ["SOL", "CIRCLE", "EXTRUDE", "EOS"]
    assert commands[1]["params"] == [-0.00978, 0.5, 0.0025]


def test_arc_endpoints_derived_from_angles() -> None:
    # Unit-ish arc on a circle of radius 2 centered at origin, 0 -> pi/2.
    dsl = (
        "LOOP type=OUTER ARC center=(0, 0) radius=2 start_angle=0 "
        "end_angle=1.5707963 END_LOOP"
    )
    commands = parse_deepcad_dsl(dsl)
    assert _types(commands) == ["SOL", "ARC", "EOS"]
    p = commands[1]["params"]
    # params = [x_start, y_start, x_end, y_end, cx, cy]
    assert math.isclose(p[0], 2.0, abs_tol=1e-6)  # start at angle 0 -> (2, 0)
    assert math.isclose(p[1], 0.0, abs_tol=1e-6)
    assert math.isclose(p[2], 0.0, abs_tol=1e-5)  # end at angle pi/2 -> (0, 2)
    assert math.isclose(p[3], 2.0, abs_tol=1e-5)
    assert p[4] == 0.0 and p[5] == 0.0  # center


def test_multiple_loops_emit_multiple_sols() -> None:
    dsl = (
        "SKETCH id=1 plane=XZ origin=(0, 0, 0)   "
        "LOOP type=OUTER     CIRCLE center=(0, 0) radius=0.0075   END_LOOP   "
        "LOOP type=OUTER     CIRCLE center=(0, 0) radius=0.004   END_LOOP "
        "END_SKETCH EXTRUDE id=1 sketch=1 distance=0.00325 operation=NEW_BODY"
    )
    commands = parse_deepcad_dsl(dsl)
    assert _types(commands) == ["SOL", "CIRCLE", "SOL", "CIRCLE", "EXTRUDE", "EOS"]


def test_multiple_extrudes_preserved_in_order() -> None:
    dsl = (
        "SKETCH id=1 plane=XY origin=(0, 0, 0) LOOP type=OUTER "
        "CIRCLE center=(0, 0) radius=0.01 END_LOOP END_SKETCH "
        "EXTRUDE id=1 sketch=1 distance=0.04 operation=NEW_BODY "
        "EXTRUDE id=2 sketch=1 distance=-0.002 operation=CUT"
    )
    commands = parse_deepcad_dsl(dsl)
    assert _types(commands) == ["SOL", "CIRCLE", "EXTRUDE", "EXTRUDE", "EOS"]
    assert commands[2]["params"] == [0.04]
    assert commands[3]["params"] == [-0.002]


def test_negative_and_scientific_floats() -> None:
    dsl = "LOOP LINE start=(-1.5e-3, 0) end=(2.5, -3.0e1)"
    commands = parse_deepcad_dsl(dsl)
    assert commands[1]["params"] == [-1.5e-3, 0.0, 2.5, -30.0]
