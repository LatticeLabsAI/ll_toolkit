"""Regression tests for the command-executor schema bridge (SPEC-1 M3).

``CommandSequenceProposal.dequantize`` emits ``{command_type, parameters
(positional list), parameter_mask}`` but the OCC geometry builders consume
``{type, params (named dict)}``.  Before the bridge, every command read as an
empty ``type`` and *no* command sequence — not even a known-valid box — could
produce geometry, pinning disposal validity at 0% regardless of the model.

The pure-logic tests below verify the adapter without OCC; the marked test
disposes a known-valid rectangle+extrude end-to-end and asserts a valid solid,
which requires pythonocc-core (conda ``cadling`` env).
"""

from __future__ import annotations

import math

import pytest

from ll_gen.disposal import command_executor as ce
from ll_gen.disposal.command_executor import (
    _arc_midpoint,
    _normalize_commands,
    _positional_to_named,
)

# ---------------------------------------------------------------------------
# Adapter logic (no OCC required)
# ---------------------------------------------------------------------------


def test_positional_to_named_line() -> None:
    assert _positional_to_named("LINE", [1.0, 2.0, 3.0, 4.0]) == {
        "x1": 1.0,
        "y1": 2.0,
        "x2": 3.0,
        "y2": 4.0,
    }


def test_positional_to_named_circle() -> None:
    assert _positional_to_named("CIRCLE", [5.0, 6.0, 2.5]) == {
        "cx": 5.0,
        "cy": 6.0,
        "r": 2.5,
    }


def test_positional_to_named_extrude_defaults_to_z_axis() -> None:
    params = _positional_to_named("EXTRUDE", [1.5])
    assert params == {"dx": 0.0, "dy": 0.0, "dz": 1.0, "extent": 1.5}


def test_positional_to_named_short_list_pads_with_zeros() -> None:
    # Missing trailing coordinates must not raise — they default to 0.0.
    assert _positional_to_named("LINE", [1.0]) == {
        "x1": 1.0,
        "y1": 0.0,
        "x2": 0.0,
        "y2": 0.0,
    }


def test_positional_to_named_unknown_type_is_empty() -> None:
    assert _positional_to_named("SOL", [0, 0]) == {}


def test_arc_midpoint_lies_on_circle() -> None:
    # Quarter arc on the unit circle from (1,0) to (0,1) about the origin.
    xm, ym = _arc_midpoint(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    assert math.isclose(math.hypot(xm, ym), 1.0, rel_tol=1e-9)
    # Half-angle of a CCW quarter sweep is 45 degrees.
    assert math.isclose(xm, math.cos(math.pi / 4), rel_tol=1e-9)
    assert math.isclose(ym, math.sin(math.pi / 4), rel_tol=1e-9)


def test_arc_midpoint_degenerate_center_falls_back_to_chord() -> None:
    xm, ym = _arc_midpoint(2.0, 0.0, 0.0, 4.0, 2.0, 0.0)
    assert (xm, ym) == (1.0, 2.0)


def test_normalize_maps_command_type_and_preserves_raw() -> None:
    commands = [
        {"command_type": "LINE", "parameters": [0, 0, 1, 0], "parameter_mask": []},
        {"command_type": "EXTRUDE", "parameters": [2.0] + [0] * 15},
    ]
    out = _normalize_commands(commands)
    assert out[0]["type"] == "LINE"
    assert out[0]["params"] == {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 0.0}
    assert out[0]["raw_parameters"] == [0, 0, 1, 0]
    assert out[1]["type"] == "EXTRUDE"
    assert out[1]["params"]["extent"] == 2.0
    assert out[1]["raw_parameters"][0] == 2.0


def test_normalize_accepts_legacy_type_key() -> None:
    # A command already in internal form (type/params) must round-trip.
    out = _normalize_commands([{"type": "circle", "parameters": [0, 0, 1]}])
    assert out[0]["type"] == "CIRCLE"
    assert out[0]["params"] == {"cx": 0.0, "cy": 0.0, "r": 1.0}


# ---------------------------------------------------------------------------
# End-to-end disposal of a known-valid solid (requires pythonocc-core)
# ---------------------------------------------------------------------------


@pytest.mark.requires_pythonocc
@pytest.mark.skipif(
    not ce._OCC_AVAILABLE, reason="requires pythonocc-core (conda cadling env)"
)
def test_known_valid_box_disposes_to_solid() -> None:
    from ll_gen.config import DisposalConfig, FeedbackConfig
    from ll_gen.disposal.engine import DisposalEngine
    from ll_gen.proposals.command_proposal import CommandSequenceProposal

    # Rectangle sketch (SOL + 4 LINEs closing the loop) + EXTRUDE + EOS.
    command_dicts = [
        {"command_type": "SOL", "parameters": [0] * 16, "parameter_mask": [False] * 16},
        {
            "command_type": "LINE",
            "parameters": [0, 0, 128, 0] + [0] * 12,
            "parameter_mask": [True] * 4 + [False] * 12,
        },
        {
            "command_type": "LINE",
            "parameters": [128, 0, 128, 128] + [0] * 12,
            "parameter_mask": [True] * 4 + [False] * 12,
        },
        {
            "command_type": "LINE",
            "parameters": [128, 128, 0, 128] + [0] * 12,
            "parameter_mask": [True] * 4 + [False] * 12,
        },
        {
            "command_type": "LINE",
            "parameters": [0, 128, 0, 0] + [0] * 12,
            "parameter_mask": [True] * 4 + [False] * 12,
        },
        {
            "command_type": "EXTRUDE",
            "parameters": [200] + [0] * 15,
            "parameter_mask": [True] + [False] * 15,
        },
        {"command_type": "EOS", "parameters": [0] * 16, "parameter_mask": [False] * 16},
    ]
    proposal = CommandSequenceProposal(
        proposal_id="regression_box",
        source_prompt="A rectangular prism",
        command_dicts=command_dicts,
        quantization_bits=8,
        normalization_range=2.0,
        precision_tier="STANDARD",
    )

    engine = DisposalEngine(
        disposal_config=DisposalConfig(),
        feedback_config=FeedbackConfig(),
        output_dir="test_executor_schema_out",
    )
    result = engine.dispose(proposal, export=False)

    assert result.has_shape, f"no shape produced: {result.error_message}"
    assert result.is_valid, f"shape invalid: {result.error_category}"
    assert result.geometry_report is not None
    # A closed box: one solid, six planar faces, Euler characteristic 2.
    assert result.geometry_report.solid_count == 1
    assert result.geometry_report.face_count == 6
    assert result.geometry_report.euler_characteristic == 2
    assert result.geometry_report.volume is not None
    assert result.geometry_report.volume > 0
