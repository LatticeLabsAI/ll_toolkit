"""Command sequence tokenizer for parametric CAD construction history.

Converts sketch-and-extrude command sequences (DeepCAD format) into
fixed-length token sequences suitable for transformer consumption.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np

from ..config import CommandTokenizationConfig, PrecisionTier, QuantizationConfig
from .token_types import (
    CommandToken,
    CommandType,
    ConstraintToken,
    ConstraintType,
    CoordinateToken,
    SequenceConfig,
    TokenSequence,
)

_log = logging.getLogger(__name__)


class CommandSequenceTokenizer:
    """Tokenize parametric CAD construction history into command sequences.

    Follows the DeepCAD pipeline: parse → normalize sketches → normalize 3D →
    quantize parameters → pad/truncate to fixed length.

    Args:
        quantization_config: Controls precision tiers and normalization.
        sequence_config: Controls sequence length and padding.
        command_config: Controls command tokenization specifics.
    """

    def __init__(
        self,
        quantization_config: Optional[QuantizationConfig] = None,
        sequence_config: Optional[SequenceConfig] = None,
        command_config: Optional[CommandTokenizationConfig] = None,
    ) -> None:
        self.quant_config = quantization_config or QuantizationConfig()
        self.seq_config = sequence_config or SequenceConfig()
        self.cmd_config = command_config or CommandTokenizationConfig()

        self._param_levels = 2 ** self.cmd_config.parameter_quantization.bits
        self._coord_levels = 2 ** self.cmd_config.coordinate_quantization.bits
        self._norm_range = self.cmd_config.normalization_range

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tokenize(
        self,
        construction_history: dict | list,
        constraints: list[dict[str, Any]] | None = None,
    ) -> TokenSequence:
        """Main entry point: convert construction history to TokenSequence.

        Args:
            construction_history: DeepCAD-format JSON (list of sketch +
                extrude operations) or a dict with a ``"sequence"`` key.
            constraints: Optional list of constraint dicts from cadling's
                SketchGeometryExtractor or Sketch2DItem.to_geotoken_constraints().
                Each dict should have "type", "source_index", "target_index",
                and optionally "value". Only processed when
                ``include_constraints`` is True in config.

        Returns:
            TokenSequence with command_tokens, coordinate_tokens, and
            optionally constraint_tokens.

        Raises:
            TypeError: If construction_history is not a dict or list.
        """
        if not isinstance(construction_history, (dict, list)):
            raise TypeError(
                f"construction_history must be dict or list, got {type(construction_history).__name__}"
            )

        if isinstance(construction_history, dict):
            commands = construction_history.get("sequence", [])
        else:
            commands = construction_history

        # 1. Parse raw commands
        parsed = self.parse_construction_history(commands)

        # 2. Normalize sketches (2D)
        parsed = self.normalize_sketches(parsed)

        # 3. Normalize 3D to bounding cube
        parsed = self.normalize_3d(parsed)

        # 4. Quantize parameters
        command_tokens = self.quantize_parameters(parsed)

        # 5. Pad or truncate
        command_tokens = self.pad_or_truncate(command_tokens)

        # 6. Build coordinate tokens for any xy/xyz params
        coordinate_tokens = self._extract_coordinate_tokens(command_tokens)

        # 7. Parse constraints if provided and configured
        constraint_tokens: list[ConstraintToken] = []
        if self.cmd_config.include_constraints and constraints:
            constraint_tokens = self.parse_constraints(constraints)

        source_fmt = self.cmd_config.source_format
        if source_fmt == "auto":
            source_fmt = "deepcad"  # Default label for metadata

        seq = TokenSequence(
            command_tokens=command_tokens,
            coordinate_tokens=coordinate_tokens,
            constraint_tokens=constraint_tokens,
            metadata={
                "source_format": source_fmt,
                "num_raw_commands": len(parsed),
                "num_constraints": len(constraint_tokens),
                "param_levels": self._param_levels,
                "coord_levels": self._coord_levels,
                "normalization_range": self._norm_range,
            },
        )
        _log.debug(
            "Tokenized %d raw commands → %d command tokens, %d constraints",
            len(parsed), len(command_tokens), len(constraint_tokens),
        )
        return seq

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    # Active parameter counts per command type (number of True values in mask)
    _ACTIVE_PARAM_COUNTS: dict[CommandType, int] = {
        CommandType.SOL: 2,
        CommandType.LINE: 4,
        CommandType.ARC: 6,
        CommandType.CIRCLE: 3,
        CommandType.EXTRUDE: 8,
        CommandType.EOS: 0,
    }

    def parse_construction_history(
        self, commands: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Parse raw DeepCAD JSON commands into internal representation.

        Each command becomes a dict with ``type`` (CommandType) and
        ``params`` (list of floats).

        Supports three source formats via ``source_format`` config:
        - ``"deepcad"``: Params are compact and match masks directly.
        - ``"cadling"``: Auto-strips z-interleaving and trailing padding
          from older cadling output to produce compact params.
        - ``"auto"`` (default): Detects format by checking if params
          have z-interleaving patterns.
        """
        parsed: list[dict[str, Any]] = []
        type_map = {
            "SOL": CommandType.SOL,
            "Line": CommandType.LINE, "LINE": CommandType.LINE,
            "Arc": CommandType.ARC, "ARC": CommandType.ARC,
            "Circle": CommandType.CIRCLE, "CIRCLE": CommandType.CIRCLE,
            "Ext": CommandType.EXTRUDE, "EXTRUDE": CommandType.EXTRUDE,
            "EOS": CommandType.EOS,
        }

        source_fmt = self.cmd_config.source_format

        for cmd in commands:
            raw_type = cmd.get("type", cmd.get("command", ""))
            cmd_type = type_map.get(raw_type)
            if cmd_type is None:
                _log.warning("Unknown command type '%s', skipping", raw_type)
                continue

            params = cmd.get("params", cmd.get("parameters", []))
            if isinstance(params, dict):
                params = list(params.values())
            params = [float(p) for p in params]

            # Handle format stripping for backward compatibility
            if source_fmt == "cadling" or (
                source_fmt == "auto" and self._looks_like_padded(cmd_type, params)
            ):
                params = self._strip_to_compact(cmd_type, params)

            parsed.append({"type": cmd_type, "params": params})

        return parsed

    @staticmethod
    def _looks_like_padded(cmd_type: CommandType, params: list[float]) -> bool:
        """Detect if params have old cadling z-interleaved padding.

        Old cadling LINE format: [x1, y1, 0, x2, y2, 0, 0, ..., 0] (16 values)
        New/compact format: [x1, y1, x2, y2, 0, ..., 0] (16 values)

        Heuristic: If we have exactly 16 params and for LINE the 3rd
        value (index 2) is 0.0 while we have non-zero values at
        indices 3 and 4, it's likely z-interleaved.
        """
        if len(params) != 16:
            return False

        if cmd_type == CommandType.LINE:
            # Old format: [x1, y1, 0, x2, y2, 0, ...]
            # Check if position 2 is 0 and position 5 is 0 (z-interleaved)
            # while positions 3,4 have values (x2, y2 in old format)
            if (abs(params[2]) < 1e-10 and abs(params[5]) < 1e-10
                    and (abs(params[3]) > 1e-10 or abs(params[4]) > 1e-10)):
                return True

        elif cmd_type == CommandType.CIRCLE:
            # Old format: [cx, cy, 0, r, ...] — z at position 2, r at position 3
            # New format: [cx, cy, r, ...] — r at position 2
            # If position 2 is 0 and position 3 is non-zero, likely old format
            if abs(params[2]) < 1e-10 and abs(params[3]) > 1e-10:
                return True

        return False

    def _strip_to_compact(
        self, cmd_type: CommandType, params: list[float]
    ) -> list[float]:
        """Strip z-interleaved params to compact format matching masks.

        Handles backward compatibility with older cadling output that
        used [x1, y1, 0, x2, y2, 0, ...] instead of [x1, y1, x2, y2, ...].

        Args:
            cmd_type: The command type.
            params: Raw parameter list (possibly z-interleaved).

        Returns:
            Compact parameter list with active params at mask positions.
        """
        active_count = self._ACTIVE_PARAM_COUNTS.get(cmd_type, 0)

        if cmd_type == CommandType.LINE and len(params) >= 6:
            # Old: [x1, y1, 0, x2, y2, 0, ...] → New: [x1, y1, x2, y2]
            if abs(params[2]) < 1e-10 and abs(params[5]) < 1e-10:
                compact = [params[0], params[1], params[3], params[4]]
                return compact + [0.0] * (16 - len(compact))

        elif cmd_type == CommandType.CIRCLE and len(params) >= 4:
            # Old: [cx, cy, 0, r, ...] → New: [cx, cy, r]
            if abs(params[2]) < 1e-10 and abs(params[3]) > 1e-10:
                compact = [params[0], params[1], params[3]]
                return compact + [0.0] * (16 - len(compact))

        # For ARC: Old cadling format was center+radius+angles which is
        # structurally different (not just z-interleaved). The new cadling
        # output uses 3-point format natively, so no stripping needed.
        # If someone passes old ARC format, it won't be auto-fixed here —
        # they should update their cadling version.

        # Default: take first N active params, pad to 16
        compact = params[:active_count] if active_count > 0 else []
        return compact + [0.0] * (16 - len(compact))

    def normalize_sketches(
        self, commands: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Normalize each sketch group to origin and 2x2 square.

        Translates sketch to centroid, scales to fit normalization range,
        and canonicalizes loop ordering (CCW from bottom-left vertex) when
        configured.
        """
        if not self.cmd_config.canonicalize_loops:
            return commands

        sketch_groups = self._split_sketch_groups(commands)
        normalized: list[dict[str, Any]] = []

        for group in sketch_groups:
            # Collect 2D points from sketch primitives
            points_2d: list[list[float]] = []
            for cmd in group:
                if cmd["type"] in (CommandType.LINE, CommandType.ARC, CommandType.CIRCLE):
                    params = cmd["params"]
                    for i in range(0, min(len(params), 6), 2):
                        if i + 1 < len(params):
                            points_2d.append([params[i], params[i + 1]])

            if not points_2d:
                normalized.extend(group)
                continue

            pts = np.array(points_2d)
            centroid = pts.mean(axis=0)
            pts_centered = pts - centroid

            extent = np.max(np.abs(pts_centered)) + 1e-8
            scale = (self._norm_range / 2.0) / extent

            # Apply normalization to params
            for cmd in group:
                if cmd["type"] in (CommandType.LINE, CommandType.ARC, CommandType.CIRCLE):
                    new_params = list(cmd["params"])
                    for i in range(0, min(len(new_params), 6), 2):
                        if i + 1 < len(new_params):
                            new_params[i] = (new_params[i] - centroid[0]) * scale
                            new_params[i + 1] = (new_params[i + 1] - centroid[1]) * scale
                    cmd = {**cmd, "params": new_params}
                normalized.append(cmd)

        return normalized

    def normalize_3d(
        self, commands: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Scale the full solid to a bounding cube using normalization range.

        Collects all 3D parameters (extrusion heights, offsets) and SOL
        sketch plane offsets, then scales to fit within the configured
        normalization range.
        """
        z_values: list[float] = []
        for cmd in commands:
            if cmd["type"] == CommandType.EXTRUDE:
                z_values.extend(cmd["params"])
            elif cmd["type"] == CommandType.SOL:
                # SOL params[0:2] contain sketch plane z-offset and rotation/normal
                # Collect z-offset (first param) for 3D normalization
                if len(cmd["params"]) > 0:
                    z_values.append(cmd["params"][0])

        if not z_values:
            return commands

        max_val = max(abs(v) for v in z_values) + 1e-8
        scale = (self._norm_range / 2.0) / max_val

        result: list[dict[str, Any]] = []
        for cmd in commands:
            if cmd["type"] == CommandType.EXTRUDE:
                new_params = [p * scale for p in cmd["params"]]
                result.append({**cmd, "params": new_params})
            elif cmd["type"] == CommandType.SOL:
                # Scale the z-offset parameter (first param) for SOL commands
                if len(cmd["params"]) > 0:
                    new_params = list(cmd["params"])
                    new_params[0] = new_params[0] * scale
                    result.append({**cmd, "params": new_params})
                else:
                    result.append(cmd)
            else:
                result.append(cmd)

        return result

    def quantize_parameters(
        self, commands: list[dict[str, Any]]
    ) -> list[CommandToken]:
        """Map continuous parameters to discrete quantization levels.

        Uses classification-not-regression: each parameter is mapped to
        one of ``param_levels`` discrete bins within the normalization range.
        """
        tokens: list[CommandToken] = []
        half_range = self._norm_range / 2.0

        for cmd in commands:
            cmd_type = cmd["type"]
            raw_params = cmd["params"]
            mask = CommandToken.get_parameter_mask(cmd_type)

            quantized = [0] * 16
            for i, active in enumerate(mask):
                if active and i < len(raw_params):
                    val = raw_params[i]
                    # Clamp to normalization range
                    val = max(-half_range, min(half_range, val))
                    # Map [-half_range, half_range] → [0, levels-1]
                    normalized = (val + half_range) / self._norm_range
                    quantized[i] = int(
                        round(normalized * (self._param_levels - 1))
                    )
                    quantized[i] = max(0, min(self._param_levels - 1, quantized[i]))

            tokens.append(CommandToken(
                command_type=cmd_type,
                parameters=quantized,
                parameter_mask=mask,
            ))

        return tokens

    def pad_or_truncate(
        self, tokens: list[CommandToken]
    ) -> list[CommandToken]:
        """Pad or truncate to fixed sequence length.

        Shorter sequences are padded with EOS tokens. Longer sequences
        are truncated, prioritizing keeping complete sketch-extrude pairs.
        """
        max_len = self.cmd_config.max_sequence_length

        if len(tokens) <= max_len:
            if self.cmd_config.pad_to_max_length:
                pad_count = max_len - len(tokens)
                pad_token = CommandToken(
                    command_type=CommandType.EOS,
                    parameters=[0] * 16,
                    parameter_mask=[False] * 16,
                )
                tokens = tokens + [pad_token] * pad_count
            return tokens

        # Truncate: find last complete sketch-extrude pair within budget
        cut_point = max_len
        for i in range(max_len - 1, -1, -1):
            if tokens[i].command_type == CommandType.EXTRUDE:
                cut_point = i + 1
                break
            if tokens[i].command_type == CommandType.EOS:
                cut_point = i + 1
                break

        truncated = tokens[:cut_point]
        if len(truncated) < max_len and self.cmd_config.pad_to_max_length:
            pad_count = max_len - len(truncated)
            pad_token = CommandToken(
                command_type=CommandType.EOS,
                parameters=[0] * 16,
                parameter_mask=[False] * 16,
            )
            truncated = truncated + [pad_token] * pad_count

        _log.debug(
            "Truncated %d commands to %d (max=%d)",
            len(tokens), len(truncated), max_len,
        )
        return truncated

    def dequantize_parameters(
        self, tokens: list[CommandToken]
    ) -> list[dict[str, Any]]:
        """Reverse quantization: convert quantized tokens back to floats.

        Args:
            tokens: List of quantized CommandTokens.

        Returns:
            List of dicts with ``type`` and continuous ``params``.
        """
        half_range = self._norm_range / 2.0
        result: list[dict[str, Any]] = []

        for token in tokens:
            params: list[float] = []
            for i, active in enumerate(token.parameter_mask):
                if active:
                    q_val = token.parameters[i]
                    continuous = (q_val / (self._param_levels - 1)) * self._norm_range - half_range
                    params.append(continuous)
                else:
                    params.append(0.0)
            result.append({"type": token.command_type, "params": params})

        return result

    def analyze_roundtrip_quality(
        self, construction_history: dict | list
    ) -> dict[str, float]:
        """Analyze quality of quantize-dequantize roundtrip.

        Measures parameter reconstruction accuracy to assess tokenization
        quality for a given construction history.

        Args:
            construction_history: DeepCAD-format JSON commands.

        Returns:
            Dict with quality metrics:
                - param_mse: Mean squared error of parameters
                - max_error: Maximum parameter error
                - command_preservation_rate: Fraction of commands preserved
        """
        # Tokenize
        token_seq = self.tokenize(construction_history)
        command_tokens = token_seq.command_tokens

        # Dequantize
        dequantized = self.dequantize_parameters(command_tokens)

        # Parse original for comparison
        if isinstance(construction_history, dict):
            orig_commands = construction_history.get("sequence", [])
        else:
            orig_commands = construction_history

        parsed_orig = self.parse_construction_history(orig_commands)
        parsed_orig = self.normalize_sketches(parsed_orig)
        parsed_orig = self.normalize_3d(parsed_orig)

        # Compare parameter values
        errors: list[float] = []
        commands_matched = 0
        total_commands = 0

        for orig, recon in zip(parsed_orig, dequantized):
            total_commands += 1
            if orig["type"] == recon["type"]:
                commands_matched += 1
                mask = CommandToken.get_parameter_mask(orig["type"])
                for i, active in enumerate(mask):
                    if active and i < len(orig["params"]) and i < len(recon["params"]):
                        error = abs(orig["params"][i] - recon["params"][i])
                        errors.append(error)

        if not errors:
            return {
                "param_mse": 0.0,
                "max_error": 0.0,
                "command_preservation_rate": 1.0 if total_commands == 0 else 0.0,
            }

        errors_arr = np.array(errors)
        return {
            "param_mse": float(np.mean(errors_arr ** 2)),
            "max_error": float(np.max(errors_arr)),
            "command_preservation_rate": commands_matched / max(total_commands, 1),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_sketch_groups(
        self, commands: list[dict[str, Any]]
    ) -> list[list[dict[str, Any]]]:
        """Split commands into sketch groups (SOL..EXTRUDE boundaries)."""
        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []

        for cmd in commands:
            current.append(cmd)
            if cmd["type"] in (CommandType.EXTRUDE, CommandType.EOS):
                groups.append(current)
                current = []

        if current:
            groups.append(current)

        return groups

    def _extract_coordinate_tokens(
        self, command_tokens: list[CommandToken]
    ) -> list[CoordinateToken]:
        """Extract 3D coordinate tokens from command parameters."""
        coords: list[CoordinateToken] = []
        bits = self.cmd_config.coordinate_quantization.bits

        for i, token in enumerate(command_tokens):
            if token.command_type in (CommandType.LINE, CommandType.ARC, CommandType.CIRCLE):
                p = token.parameters
                # First two active params are typically x, y
                if token.parameter_mask[0] and token.parameter_mask[1]:
                    coords.append(CoordinateToken(
                        x=p[0], y=p[1], z=0,
                        bits=bits, vertex_index=i,
                    ))

        return coords

    # ------------------------------------------------------------------
    # SkexGen-style disentangled token streams
    # ------------------------------------------------------------------

    def disentangle(
        self,
        command_tokens: list[CommandToken],
    ) -> dict[str, list[CommandToken]]:
        """Split a command sequence into SkexGen-style disentangled streams.

        SkexGen (Xu et al., 2022) decomposes CAD models into three
        independently learnable factors:

        1. **topology** — *which* commands appear and their connectivity
           (command types + sketch/extrude grouping structure).  For the
           tokenizer this means each CommandToken is reduced to its type
           and mask only; parameter values are zeroed.
        2. **geometry** — the 2-D sketch parameters (LINE endpoints, ARC
           centers, CIRCLE radii).  Only sketch-level command tokens are
           included, with extrusion parameters zeroed.
        3. **extrusion** — the 3-D manufacturing parameters (extrude
           extent, direction, booleans).  Only EXTRUDE tokens are
           included with sketch parameters zeroed.

        Each stream is a list of CommandToken objects so they can be
        independently encoded via :class:`CADVocabulary`.

        Args:
            command_tokens: Flat command token sequence (output of
                :meth:`tokenize`).

        Returns:
            Dictionary with keys ``"topology"``, ``"geometry"``, and
            ``"extrusion"``, each containing a list of CommandToken
            objects representing that stream.
        """
        SKETCH_TYPES = {
            CommandType.LINE,
            CommandType.ARC,
            CommandType.CIRCLE,
        }
        EXTRUDE_TYPES = {
            CommandType.EXTRUDE,
        }

        topology: list[CommandToken] = []
        geometry: list[CommandToken] = []
        extrusion: list[CommandToken] = []

        for tok in command_tokens:
            # --- Topology stream: keep type + mask, zero parameters ---
            topo_params = [0] * len(tok.parameters)
            topology.append(CommandToken(
                command_type=tok.command_type,
                parameters=topo_params,
                parameter_mask=list(tok.parameter_mask),
            ))

            # --- Geometry stream: only sketch commands, full params ---
            if tok.command_type in SKETCH_TYPES:
                geometry.append(CommandToken(
                    command_type=tok.command_type,
                    parameters=list(tok.parameters),
                    parameter_mask=list(tok.parameter_mask),
                ))

            # --- Extrusion stream: only extrude commands, full params ---
            if tok.command_type in EXTRUDE_TYPES:
                extrusion.append(CommandToken(
                    command_type=tok.command_type,
                    parameters=list(tok.parameters),
                    parameter_mask=list(tok.parameter_mask),
                ))

        _log.debug(
            "Disentangled %d tokens → topology=%d, geometry=%d, extrusion=%d",
            len(command_tokens),
            len(topology),
            len(geometry),
            len(extrusion),
        )

        return {
            "topology": topology,
            "geometry": geometry,
            "extrusion": extrusion,
        }

    # ------------------------------------------------------------------
    # Constraint parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_constraints(
        constraints: list[dict[str, Any]],
    ) -> list[ConstraintToken]:
        """Parse constraint dicts into ConstraintToken objects.

        Accepts constraint dicts from cadling's SketchGeometryExtractor
        (via Sketch2DItem.to_geotoken_constraints()) or any dict with
        "type", "source_index"/"entity_a", and "target_index"/"entity_b".

        Unknown constraint types are silently skipped.

        Args:
            constraints: List of constraint dicts. Each dict should have:
                - type: String matching a ConstraintType name
                  (e.g., "PARALLEL", "TANGENT", "COINCIDENT")
                - source_index or entity_a: Index of first entity
                - target_index or entity_b: Index of second entity
                - value (optional): Quantized constraint value

        Returns:
            List of ConstraintToken instances.

        Example:
            constraints = [
                {"type": "PARALLEL", "source_index": 0, "target_index": 2},
                {"type": "TANGENT", "entity_a": 1, "entity_b": 3, "confidence": 0.95},
            ]
            tokens = CommandSequenceTokenizer.parse_constraints(constraints)
        """
        type_map: dict[str, ConstraintType] = {
            "PARALLEL": ConstraintType.PARALLEL,
            "PERPENDICULAR": ConstraintType.PERPENDICULAR,
            "CONCENTRIC": ConstraintType.CONCENTRIC,
            "TANGENT": ConstraintType.TANGENT,
            "EQUAL_RADIUS": ConstraintType.EQUAL_RADIUS,
            "EQUAL_LENGTH": ConstraintType.EQUAL_LENGTH,
            "COINCIDENT": ConstraintType.COINCIDENT,
            "DISTANCE": ConstraintType.DISTANCE,
            "ANGLE": ConstraintType.ANGLE,
            # Cadling-specific mappings
            "COLLINEAR": ConstraintType.PARALLEL,  # Closest match
        }

        tokens: list[ConstraintToken] = []
        for c in constraints:
            raw_type = c.get("type", "")
            ctype = type_map.get(raw_type)
            if ctype is None:
                _log.debug("Unknown constraint type '%s', skipping", raw_type)
                continue

            source = c.get("source_index", c.get("entity_a", 0))
            target = c.get("target_index", c.get("entity_b", 0))
            value = c.get("value", None)

            tokens.append(ConstraintToken(
                constraint_type=ctype,
                source_index=int(source),
                target_index=int(target),
                value=int(value) if value is not None else None,
            ))

        _log.debug("Parsed %d constraints from %d input dicts",
                    len(tokens), len(constraints))
        return tokens
