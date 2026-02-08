"""Convert between cadling and DeepCAD command sequence formats.

Cadling's SketchGeometryExtractor produces commands with 16 parameters
where z-coordinates are interleaved (e.g., LINE: [x1, y1, z1, x2, y2, z2, 0...]).
DeepCAD uses compact parameters with only active values (LINE: [x1, y1, x2, y2]).

This module provides bidirectional conversion and auto-detection so that
both formats can flow through the same tokenization pipeline.
"""
from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapping tables
# ---------------------------------------------------------------------------

# For each command type, list:
#   - ``indices``: positions in the cadling 16-param vector that hold the
#     active parameters after z-interleaved values are removed.
#   - ``compact_count``: how many active parameters the compact format has.
#   - ``z_indices``: positions that hold z-coordinates (always 0.0 in 2D
#     sketch commands).

CADLING_PARAM_MAP: dict[str, dict[str, Any]] = {
    "SOL": {
        "indices": [0, 1],
        "compact_count": 2,
        "z_indices": [],
    },
    "LINE": {
        # cadling: [x1, y1, z1, x2, y2, z2, 0..0]  →  deepcad: [x1, y1, x2, y2]
        "indices": [0, 1, 3, 4],
        "compact_count": 4,
        "z_indices": [2, 5],
    },
    "ARC": {
        # cadling: [x1, y1, z1, x2, y2, z2, x3, y3, z3, 0..0]
        # deepcad: [x1, y1, x2, y2, x3, y3]
        "indices": [0, 1, 3, 4, 6, 7],
        "compact_count": 6,
        "z_indices": [2, 5, 8],
    },
    "CIRCLE": {
        # cadling: [cx, cy, z, r, 0..0]  →  deepcad: [cx, cy, r]
        "indices": [0, 1, 3],
        "compact_count": 3,
        "z_indices": [2],
    },
    "EXTRUDE": {
        # Extrusions are 3D; no z-stripping needed.
        # cadling and deepcad both use 8 active params.
        "indices": list(range(8)),
        "compact_count": 8,
        "z_indices": [],
    },
    "EOS": {
        "indices": [],
        "compact_count": 0,
        "z_indices": [],
    },
}

# Inverse map: for padding compact → cadling we need to know where each
# compact param goes in the 16-slot vector and where z-values are inserted.
DEEPCAD_TO_CADLING_INSERT: dict[str, dict[str, Any]] = {
    "SOL": {"target_indices": [0, 1], "z_insert": {}},
    "LINE": {"target_indices": [0, 1, 3, 4], "z_insert": {2: 0.0, 5: 0.0}},
    "ARC": {
        "target_indices": [0, 1, 3, 4, 6, 7],
        "z_insert": {2: 0.0, 5: 0.0, 8: 0.0},
    },
    "CIRCLE": {"target_indices": [0, 1, 3], "z_insert": {2: 0.0}},
    "EXTRUDE": {"target_indices": list(range(8)), "z_insert": {}},
    "EOS": {"target_indices": [], "z_insert": {}},
}

_CANONICAL_TYPE_NAMES: dict[str, str] = {
    "Line": "LINE",
    "Arc": "ARC",
    "Circle": "CIRCLE",
    "Ext": "EXTRUDE",
    "Sol": "SOL",
    "Eos": "EOS",
}


def _canonical_type(raw_type: str) -> str:
    """Normalize command type name to upper-case canonical form."""
    upper = raw_type.upper()
    if upper in CADLING_PARAM_MAP:
        return upper
    return _CANONICAL_TYPE_NAMES.get(raw_type, upper)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class CommandFormatConverter:
    """Bidirectional converter between cadling and DeepCAD command formats.

    **cadling format**: 16-parameter padded vectors with z-coordinates
    interleaved for 2D sketch commands.

    **DeepCAD format**: Compact parameter vectors with only the active
    values present (LINE has 4, ARC has 6, CIRCLE has 3, etc.).

    Example::

        cadling_cmds = [
            {"type": "LINE", "params": [0.1, 0.2, 0.0, 0.8, 0.9, 0.0, 0, ...]}
        ]
        deepcad_cmds = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)
        # → [{"type": "LINE", "params": [0.1, 0.2, 0.8, 0.9]}]
    """

    # ------------------------------------------------------------------
    # cadling → DeepCAD
    # ------------------------------------------------------------------

    @staticmethod
    def cadling_to_deepcad(
        commands: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert cadling 16-param padded commands to DeepCAD compact format.

        Args:
            commands: List of command dicts with ``type`` (str) and
                ``params`` (list of 16 floats).

        Returns:
            List of command dicts with compact parameter lists.
        """
        converted: list[dict[str, Any]] = []

        for cmd in commands:
            raw_type = cmd.get("type", cmd.get("command", ""))
            canon = _canonical_type(raw_type)
            params = cmd.get("params", cmd.get("parameters", []))
            if isinstance(params, dict):
                params = list(params.values())
            params = [float(p) for p in params]

            mapping = CADLING_PARAM_MAP.get(canon)
            if mapping is None:
                _log.warning(
                    "Unknown command type '%s' — passing through unchanged",
                    raw_type,
                )
                converted.append(cmd)
                continue

            # Extract only the active parameter positions
            indices = mapping["indices"]
            compact = [
                params[i] if i < len(params) else 0.0 for i in indices
            ]

            converted.append({
                "type": canon,
                "params": compact,
            })

        _log.debug(
            "Converted %d commands cadling → deepcad", len(converted)
        )
        return converted

    # ------------------------------------------------------------------
    # DeepCAD → cadling
    # ------------------------------------------------------------------

    @staticmethod
    def deepcad_to_cadling(
        commands: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert DeepCAD compact commands to cadling 16-param padded format.

        Inserts z-coordinates (0.0) at the appropriate positions and
        pads the remaining slots with 0.0 to reach length 16.

        Args:
            commands: List of compact command dicts.

        Returns:
            List of command dicts with 16-parameter padded lists.
        """
        converted: list[dict[str, Any]] = []

        for cmd in commands:
            raw_type = cmd.get("type", cmd.get("command", ""))
            canon = _canonical_type(raw_type)
            compact_params = cmd.get("params", cmd.get("parameters", []))
            if isinstance(compact_params, dict):
                compact_params = list(compact_params.values())
            compact_params = [float(p) for p in compact_params]

            insert_info = DEEPCAD_TO_CADLING_INSERT.get(canon)
            if insert_info is None:
                _log.warning(
                    "Unknown command type '%s' — passing through unchanged",
                    raw_type,
                )
                converted.append(cmd)
                continue

            target_indices = insert_info["target_indices"]
            z_insert = insert_info["z_insert"]

            # Start with 16 zeros
            padded = [0.0] * 16

            # Place compact params at their target positions
            for slot, target_idx in enumerate(target_indices):
                if slot < len(compact_params) and target_idx < 16:
                    padded[target_idx] = compact_params[slot]

            # Insert z-coordinate values
            for z_idx, z_val in z_insert.items():
                if z_idx < 16:
                    padded[z_idx] = z_val

            converted.append({
                "type": canon,
                "params": padded,
            })

        _log.debug(
            "Converted %d commands deepcad → cadling", len(converted)
        )
        return converted

    # ------------------------------------------------------------------
    # Format detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_format(commands: list[dict[str, Any]]) -> str:
        """Auto-detect whether commands use cadling or DeepCAD format.

        Heuristics:

        1. If any command has exactly 16 parameters and exhibits
           z-interleaving (zero values at z-coordinate positions), the
           format is ``"cadling"``.
        2. If parameters are compact (length ≤ 8 or length 16 without
           z-interleaving), the format is ``"deepcad"``.
        3. If the sequence is empty, returns ``"unknown"``.

        Args:
            commands: List of command dicts to inspect.

        Returns:
            ``"cadling"``, ``"deepcad"``, or ``"unknown"``.
        """
        if not commands:
            return "unknown"

        cadling_votes = 0
        deepcad_votes = 0

        for cmd in commands:
            raw_type = cmd.get("type", cmd.get("command", ""))
            canon = _canonical_type(raw_type)
            params = cmd.get("params", cmd.get("parameters", []))

            if isinstance(params, dict):
                params = list(params.values())

            n = len(params)

            # Skip types we can't distinguish on
            if canon in ("SOL", "EOS", "EXTRUDE"):
                continue

            mapping = CADLING_PARAM_MAP.get(canon)
            if mapping is None:
                continue

            compact_count = mapping["compact_count"]
            z_indices = mapping["z_indices"]

            if n == 16 and z_indices:
                # Check if z-positions are all ~0.0
                z_are_zero = all(
                    abs(float(params[zi])) < 1e-10
                    for zi in z_indices
                    if zi < n
                )
                # Check if there are non-zero values after z-positions
                active_indices = mapping["indices"]
                has_active = any(
                    abs(float(params[ai])) > 1e-10
                    for ai in active_indices
                    if ai < n
                )
                if z_are_zero and has_active:
                    cadling_votes += 1
                else:
                    deepcad_votes += 1

            elif n <= compact_count or n == compact_count:
                deepcad_votes += 1
            else:
                # Ambiguous — could be either
                deepcad_votes += 1

        if cadling_votes > deepcad_votes:
            return "cadling"
        elif deepcad_votes > 0:
            return "deepcad"
        return "unknown"

    # ------------------------------------------------------------------
    # Round-trip validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_roundtrip(
        commands: list[dict[str, Any]],
        tolerance: float = 1e-8,
    ) -> dict[str, Any]:
        """Validate that cadling → deepcad → cadling preserves parameters.

        Useful for testing and debugging format conversion.

        Args:
            commands: Original cadling-format commands.
            tolerance: Maximum acceptable parameter deviation.

        Returns:
            Dict with ``valid`` (bool), ``max_error`` (float), and
            ``mismatches`` (list of per-command errors).
        """
        deepcad = CommandFormatConverter.cadling_to_deepcad(commands)
        roundtrip = CommandFormatConverter.deepcad_to_cadling(deepcad)

        mismatches: list[dict[str, Any]] = []
        max_error = 0.0

        for i, (orig, rt) in enumerate(zip(commands, roundtrip)):
            orig_params = orig.get("params", [])
            rt_params = rt.get("params", [])

            for j in range(min(len(orig_params), len(rt_params))):
                err = abs(float(orig_params[j]) - float(rt_params[j]))
                max_error = max(max_error, err)
                if err > tolerance:
                    mismatches.append({
                        "command_index": i,
                        "param_index": j,
                        "original": float(orig_params[j]),
                        "roundtrip": float(rt_params[j]),
                        "error": err,
                    })

        return {
            "valid": len(mismatches) == 0,
            "max_error": max_error,
            "mismatches": mismatches,
            "num_commands": len(commands),
        }
