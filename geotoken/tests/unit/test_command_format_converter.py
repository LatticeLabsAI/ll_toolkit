"""Tests for the CommandFormatConverter class.

Tests bidirectional conversion between cadling (16-param padded) and
DeepCAD (compact) command formats, format auto-detection, and roundtrip
validation.
"""
import pytest
from geotoken.tokenizer.command_format_converter import (
    CommandFormatConverter,
    CADLING_PARAM_MAP,
    DEEPCAD_TO_CADLING_INSERT,
)


class TestCadlingToDeepCAD:
    """Tests for cadling_to_deepcad() conversion."""

    def test_line_conversion(self):
        """Test converting LINE from cadling to DeepCAD format."""
        cadling_cmds = [
            {
                "type": "LINE",
                "params": [0.1, 0.2, 0.0, 0.8, 0.9, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        ]
        deepcad_cmds = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)

        assert len(deepcad_cmds) == 1
        assert deepcad_cmds[0]["type"] == "LINE"
        assert deepcad_cmds[0]["params"] == [0.1, 0.2, 0.8, 0.9]

    def test_arc_conversion(self):
        """Test converting ARC from cadling to DeepCAD format."""
        cadling_cmds = [
            {
                "type": "ARC",
                "params": [0.1, 0.2, 0.0, 0.5, 0.6, 0.0, 0.9, 0.95, 0.0, 0, 0, 0, 0, 0, 0, 0]
            }
        ]
        deepcad_cmds = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)

        assert len(deepcad_cmds) == 1
        assert deepcad_cmds[0]["type"] == "ARC"
        assert deepcad_cmds[0]["params"] == [0.1, 0.2, 0.5, 0.6, 0.9, 0.95]

    def test_circle_conversion(self):
        """Test converting CIRCLE from cadling to DeepCAD format."""
        cadling_cmds = [
            {
                "type": "CIRCLE",
                "params": [0.5, 0.5, 0.0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        ]
        deepcad_cmds = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)

        assert len(deepcad_cmds) == 1
        assert deepcad_cmds[0]["type"] == "CIRCLE"
        assert deepcad_cmds[0]["params"] == [0.5, 0.5, 0.25]

    def test_sol_conversion(self):
        """Test converting SOL from cadling to DeepCAD format."""
        cadling_cmds = [
            {
                "type": "SOL",
                "params": [1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        ]
        deepcad_cmds = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)

        assert len(deepcad_cmds) == 1
        assert deepcad_cmds[0]["type"] == "SOL"
        assert deepcad_cmds[0]["params"] == [1, 2]

    def test_extrude_conversion(self):
        """Test converting EXTRUDE from cadling to DeepCAD format.

        EXTRUDE has 8 active params and no z-stripping needed.
        """
        cadling_cmds = [
            {
                "type": "EXTRUDE",
                "params": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        ]
        deepcad_cmds = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)

        assert len(deepcad_cmds) == 1
        assert deepcad_cmds[0]["type"] == "EXTRUDE"
        assert deepcad_cmds[0]["params"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    def test_eos_conversion(self):
        """Test converting EOS (end-of-sketch) marker.

        EOS has no parameters.
        """
        cadling_cmds = [
            {
                "type": "EOS",
                "params": [0] * 16
            }
        ]
        deepcad_cmds = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)

        assert len(deepcad_cmds) == 1
        assert deepcad_cmds[0]["type"] == "EOS"
        assert deepcad_cmds[0]["params"] == []

    def test_multiple_commands_conversion(self):
        """Test converting multiple commands at once."""
        cadling_cmds = [
            {
                "type": "LINE",
                "params": [0.1, 0.2, 0.0, 0.8, 0.9, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            },
            {
                "type": "CIRCLE",
                "params": [0.5, 0.5, 0.0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            },
            {
                "type": "EOS",
                "params": [0] * 16
            }
        ]
        deepcad_cmds = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)

        assert len(deepcad_cmds) == 3
        assert deepcad_cmds[0]["type"] == "LINE"
        assert deepcad_cmds[1]["type"] == "CIRCLE"
        assert deepcad_cmds[2]["type"] == "EOS"

    def test_cadling_with_shorthand_type_key(self):
        """Test conversion with 'command' key instead of 'type'."""
        cadling_cmds = [
            {
                "command": "LINE",
                "params": [0.1, 0.2, 0.0, 0.8, 0.9, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        ]
        deepcad_cmds = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)

        assert len(deepcad_cmds) == 1
        assert deepcad_cmds[0]["type"] == "LINE"
        assert deepcad_cmds[0]["params"] == [0.1, 0.2, 0.8, 0.9]

    def test_cadling_with_shorthand_params_key(self):
        """Test conversion with 'parameters' key instead of 'params'."""
        cadling_cmds = [
            {
                "type": "LINE",
                "parameters": [0.1, 0.2, 0.0, 0.8, 0.9, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        ]
        deepcad_cmds = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)

        assert len(deepcad_cmds) == 1
        assert deepcad_cmds[0]["type"] == "LINE"

    def test_unknown_command_type_passthrough(self):
        """Test that unknown command types pass through unchanged."""
        cadling_cmds = [
            {
                "type": "UNKNOWN",
                "params": [1.0, 2.0, 3.0]
            }
        ]
        deepcad_cmds = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)

        assert len(deepcad_cmds) == 1
        assert deepcad_cmds[0]["type"] == "UNKNOWN"
        assert deepcad_cmds[0]["params"] == [1.0, 2.0, 3.0]

    def test_short_cadling_params_padding(self):
        """Test that missing parameters are padded with 0.0 when extracting."""
        cadling_cmds = [
            {
                "type": "LINE",
                "params": [0.1, 0.2, 0.0, 0.8, 0.9]  # Only 5 params instead of 16
            }
        ]
        deepcad_cmds = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)

        # Indices for LINE are [0, 1, 3, 4], so we should get [0.1, 0.2, 0.8, 0.9]
        assert deepcad_cmds[0]["params"] == [0.1, 0.2, 0.8, 0.9]

    def test_type_name_normalization_lowercase(self):
        """Test that type names are normalized to uppercase."""
        cadling_cmds = [
            {
                "type": "line",
                "params": [0.1, 0.2, 0.0, 0.8, 0.9, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        ]
        deepcad_cmds = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)

        assert deepcad_cmds[0]["type"] == "LINE"

    def test_type_name_normalization_mixed_case(self):
        """Test that mixed-case type names (e.g., 'Line') are normalized."""
        cadling_cmds = [
            {
                "type": "Line",
                "params": [0.1, 0.2, 0.0, 0.8, 0.9, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        ]
        deepcad_cmds = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)

        assert deepcad_cmds[0]["type"] == "LINE"

    def test_params_as_dict(self):
        """Test that dict-format params are converted to list."""
        cadling_cmds = [
            {
                "type": "LINE",
                "params": {0: 0.1, 1: 0.2, 2: 0.0, 3: 0.8, 4: 0.9, 5: 0.0}
            }
        ]
        deepcad_cmds = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)

        assert len(deepcad_cmds[0]["params"]) == 4


class TestDeepCADToCadling:
    """Tests for deepcad_to_cadling() conversion."""

    def test_line_conversion(self):
        """Test converting LINE from DeepCAD to cadling format."""
        deepcad_cmds = [
            {
                "type": "LINE",
                "params": [0.1, 0.2, 0.8, 0.9]
            }
        ]
        cadling_cmds = CommandFormatConverter.deepcad_to_cadling(deepcad_cmds)

        assert len(cadling_cmds) == 1
        assert cadling_cmds[0]["type"] == "LINE"
        # Should have 16 params: [0.1, 0.2, 0.0, 0.8, 0.9, 0.0, 0, ...]
        assert len(cadling_cmds[0]["params"]) == 16
        assert cadling_cmds[0]["params"][0] == 0.1
        assert cadling_cmds[0]["params"][1] == 0.2
        assert cadling_cmds[0]["params"][2] == 0.0  # z-value inserted
        assert cadling_cmds[0]["params"][3] == 0.8
        assert cadling_cmds[0]["params"][4] == 0.9
        assert cadling_cmds[0]["params"][5] == 0.0  # z-value inserted

    def test_arc_conversion(self):
        """Test converting ARC from DeepCAD to cadling format."""
        deepcad_cmds = [
            {
                "type": "ARC",
                "params": [0.1, 0.2, 0.5, 0.6, 0.9, 0.95]
            }
        ]
        cadling_cmds = CommandFormatConverter.deepcad_to_cadling(deepcad_cmds)

        assert len(cadling_cmds) == 1
        assert cadling_cmds[0]["type"] == "ARC"
        assert len(cadling_cmds[0]["params"]) == 16
        # Check that all z-coordinates are inserted at positions 2, 5, 8
        assert cadling_cmds[0]["params"][2] == 0.0
        assert cadling_cmds[0]["params"][5] == 0.0
        assert cadling_cmds[0]["params"][8] == 0.0

    def test_circle_conversion(self):
        """Test converting CIRCLE from DeepCAD to cadling format."""
        deepcad_cmds = [
            {
                "type": "CIRCLE",
                "params": [0.5, 0.5, 0.25]
            }
        ]
        cadling_cmds = CommandFormatConverter.deepcad_to_cadling(deepcad_cmds)

        assert len(cadling_cmds) == 1
        assert cadling_cmds[0]["type"] == "CIRCLE"
        assert len(cadling_cmds[0]["params"]) == 16
        assert cadling_cmds[0]["params"][0] == 0.5
        assert cadling_cmds[0]["params"][1] == 0.5
        assert cadling_cmds[0]["params"][2] == 0.0  # z-value inserted
        assert cadling_cmds[0]["params"][3] == 0.25

    def test_sol_conversion(self):
        """Test converting SOL from DeepCAD to cadling format."""
        deepcad_cmds = [
            {
                "type": "SOL",
                "params": [1, 2]
            }
        ]
        cadling_cmds = CommandFormatConverter.deepcad_to_cadling(deepcad_cmds)

        assert len(cadling_cmds) == 1
        assert cadling_cmds[0]["type"] == "SOL"
        assert len(cadling_cmds[0]["params"]) == 16
        assert cadling_cmds[0]["params"][0] == 1
        assert cadling_cmds[0]["params"][1] == 2

    def test_extrude_conversion(self):
        """Test converting EXTRUDE from DeepCAD to cadling format.

        EXTRUDE uses all 8 positions and has no z-stripping.
        """
        deepcad_cmds = [
            {
                "type": "EXTRUDE",
                "params": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            }
        ]
        cadling_cmds = CommandFormatConverter.deepcad_to_cadling(deepcad_cmds)

        assert len(cadling_cmds) == 1
        assert cadling_cmds[0]["type"] == "EXTRUDE"
        assert len(cadling_cmds[0]["params"]) == 16
        assert cadling_cmds[0]["params"][:8] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    def test_eos_conversion(self):
        """Test converting EOS (end-of-sketch) marker."""
        deepcad_cmds = [
            {
                "type": "EOS",
                "params": []
            }
        ]
        cadling_cmds = CommandFormatConverter.deepcad_to_cadling(deepcad_cmds)

        assert len(cadling_cmds) == 1
        assert cadling_cmds[0]["type"] == "EOS"
        assert len(cadling_cmds[0]["params"]) == 16
        assert all(p == 0.0 for p in cadling_cmds[0]["params"])

    def test_multiple_commands_conversion(self):
        """Test converting multiple DeepCAD commands at once."""
        deepcad_cmds = [
            {
                "type": "LINE",
                "params": [0.1, 0.2, 0.8, 0.9]
            },
            {
                "type": "CIRCLE",
                "params": [0.5, 0.5, 0.25]
            }
        ]
        cadling_cmds = CommandFormatConverter.deepcad_to_cadling(deepcad_cmds)

        assert len(cadling_cmds) == 2
        assert cadling_cmds[0]["type"] == "LINE"
        assert cadling_cmds[1]["type"] == "CIRCLE"
        assert all(len(cmd["params"]) == 16 for cmd in cadling_cmds)

    def test_unknown_command_type_passthrough(self):
        """Test that unknown command types pass through unchanged."""
        deepcad_cmds = [
            {
                "type": "UNKNOWN",
                "params": [1.0, 2.0, 3.0]
            }
        ]
        cadling_cmds = CommandFormatConverter.deepcad_to_cadling(deepcad_cmds)

        assert len(cadling_cmds) == 1
        assert cadling_cmds[0]["type"] == "UNKNOWN"
        assert cadling_cmds[0]["params"] == [1.0, 2.0, 3.0]

    def test_type_name_normalization(self):
        """Test that type names are normalized."""
        deepcad_cmds = [
            {
                "type": "Circle",
                "params": [0.5, 0.5, 0.25]
            }
        ]
        cadling_cmds = CommandFormatConverter.deepcad_to_cadling(deepcad_cmds)

        assert cadling_cmds[0]["type"] == "CIRCLE"

    def test_params_as_dict(self):
        """Test that dict-format params are converted to list."""
        deepcad_cmds = [
            {
                "type": "LINE",
                "parameters": {0: 0.1, 1: 0.2, 2: 0.8, 3: 0.9}
            }
        ]
        cadling_cmds = CommandFormatConverter.deepcad_to_cadling(deepcad_cmds)

        assert len(cadling_cmds[0]["params"]) == 16


class TestRoundtrip:
    """Tests for cadling → deepcad → cadling roundtrip preservation."""

    def test_line_roundtrip(self):
        """Test that LINE parameters survive cadling→deepcad→cadling roundtrip."""
        cadling_orig = [
            {
                "type": "LINE",
                "params": [0.1, 0.2, 0.0, 0.8, 0.9, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        ]

        deepcad = CommandFormatConverter.cadling_to_deepcad(cadling_orig)
        cadling_roundtrip = CommandFormatConverter.deepcad_to_cadling(deepcad)

        assert cadling_orig[0]["params"] == cadling_roundtrip[0]["params"]

    def test_arc_roundtrip(self):
        """Test that ARC parameters survive cadling→deepcad→cadling roundtrip."""
        cadling_orig = [
            {
                "type": "ARC",
                "params": [0.1, 0.2, 0.0, 0.5, 0.6, 0.0, 0.9, 0.95, 0.0, 0, 0, 0, 0, 0, 0, 0]
            }
        ]

        deepcad = CommandFormatConverter.cadling_to_deepcad(cadling_orig)
        cadling_roundtrip = CommandFormatConverter.deepcad_to_cadling(deepcad)

        assert cadling_orig[0]["params"] == cadling_roundtrip[0]["params"]

    def test_circle_roundtrip(self):
        """Test that CIRCLE parameters survive cadling→deepcad→cadling roundtrip."""
        cadling_orig = [
            {
                "type": "CIRCLE",
                "params": [0.5, 0.5, 0.0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        ]

        deepcad = CommandFormatConverter.cadling_to_deepcad(cadling_orig)
        cadling_roundtrip = CommandFormatConverter.deepcad_to_cadling(deepcad)

        assert cadling_orig[0]["params"] == cadling_roundtrip[0]["params"]

    def test_extrude_roundtrip(self):
        """Test that EXTRUDE parameters survive cadling→deepcad→cadling roundtrip."""
        cadling_orig = [
            {
                "type": "EXTRUDE",
                "params": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        ]

        deepcad = CommandFormatConverter.cadling_to_deepcad(cadling_orig)
        cadling_roundtrip = CommandFormatConverter.deepcad_to_cadling(deepcad)

        assert cadling_orig[0]["params"] == cadling_roundtrip[0]["params"]

    def test_sol_roundtrip(self):
        """Test that SOL parameters survive cadling→deepcad→cadling roundtrip."""
        cadling_orig = [
            {
                "type": "SOL",
                "params": [1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        ]

        deepcad = CommandFormatConverter.cadling_to_deepcad(cadling_orig)
        cadling_roundtrip = CommandFormatConverter.deepcad_to_cadling(deepcad)

        assert cadling_orig[0]["params"] == cadling_roundtrip[0]["params"]

    def test_mixed_sequence_roundtrip(self):
        """Test that mixed command sequences survive roundtrip."""
        cadling_orig = [
            {
                "type": "SOL",
                "params": [1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            },
            {
                "type": "LINE",
                "params": [0.1, 0.2, 0.0, 0.8, 0.9, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            },
            {
                "type": "CIRCLE",
                "params": [0.5, 0.5, 0.0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            },
            {
                "type": "EOS",
                "params": [0] * 16
            }
        ]

        deepcad = CommandFormatConverter.cadling_to_deepcad(cadling_orig)
        cadling_roundtrip = CommandFormatConverter.deepcad_to_cadling(deepcad)

        for orig, rt in zip(cadling_orig, cadling_roundtrip):
            assert orig["params"] == rt["params"]


class TestFormatDetection:
    """Tests for detect_format() auto-detection."""

    def test_detect_cadling_from_line(self):
        """Test detecting cadling format from LINE command with 16 params and z-interleaving."""
        commands = [
            {
                "type": "LINE",
                "params": [0.1, 0.2, 0.0, 0.8, 0.9, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        ]
        fmt = CommandFormatConverter.detect_format(commands)
        assert fmt == "cadling"

    def test_detect_cadling_from_arc(self):
        """Test detecting cadling format from ARC command."""
        commands = [
            {
                "type": "ARC",
                "params": [0.1, 0.2, 0.0, 0.5, 0.6, 0.0, 0.9, 0.95, 0.0, 0, 0, 0, 0, 0, 0, 0]
            }
        ]
        fmt = CommandFormatConverter.detect_format(commands)
        assert fmt == "cadling"

    def test_detect_cadling_from_circle(self):
        """Test detecting cadling format from CIRCLE command."""
        commands = [
            {
                "type": "CIRCLE",
                "params": [0.5, 0.5, 0.0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        ]
        fmt = CommandFormatConverter.detect_format(commands)
        assert fmt == "cadling"

    def test_detect_deepcad_from_line(self):
        """Test detecting DeepCAD format from LINE command with 4 params."""
        commands = [
            {
                "type": "LINE",
                "params": [0.1, 0.2, 0.8, 0.9]
            }
        ]
        fmt = CommandFormatConverter.detect_format(commands)
        assert fmt == "deepcad"

    def test_detect_deepcad_from_arc(self):
        """Test detecting DeepCAD format from ARC command with 6 params."""
        commands = [
            {
                "type": "ARC",
                "params": [0.1, 0.2, 0.5, 0.6, 0.9, 0.95]
            }
        ]
        fmt = CommandFormatConverter.detect_format(commands)
        assert fmt == "deepcad"

    def test_detect_deepcad_from_circle(self):
        """Test detecting DeepCAD format from CIRCLE command with 3 params."""
        commands = [
            {
                "type": "CIRCLE",
                "params": [0.5, 0.5, 0.25]
            }
        ]
        fmt = CommandFormatConverter.detect_format(commands)
        assert fmt == "deepcad"

    def test_detect_mixed_with_sol_eos(self):
        """Test detection with SOL and EOS commands (which are skipped in detection)."""
        commands = [
            {
                "type": "SOL",
                "params": [1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            },
            {
                "type": "LINE",
                "params": [0.1, 0.2, 0.8, 0.9]
            },
            {
                "type": "EOS",
                "params": [0] * 16
            }
        ]
        fmt = CommandFormatConverter.detect_format(commands)
        assert fmt == "deepcad"

    def test_detect_cadling_multiple_commands(self):
        """Test detecting cadling from multiple 16-param commands with z-interleaving."""
        commands = [
            {
                "type": "LINE",
                "params": [0.1, 0.2, 0.0, 0.8, 0.9, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            },
            {
                "type": "CIRCLE",
                "params": [0.5, 0.5, 0.0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        ]
        fmt = CommandFormatConverter.detect_format(commands)
        assert fmt == "cadling"

    def test_detect_empty_sequence(self):
        """Test that empty command sequence returns 'unknown'."""
        commands = []
        fmt = CommandFormatConverter.detect_format(commands)
        assert fmt == "unknown"

    def test_detect_unknown_command_type(self):
        """Test detection with unknown command type."""
        commands = [
            {
                "type": "UNKNOWN",
                "params": [1.0, 2.0, 3.0]
            }
        ]
        fmt = CommandFormatConverter.detect_format(commands)
        assert fmt == "unknown"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_command_sequence_cadling_to_deepcad(self):
        """Test converting empty command sequence."""
        cadling_cmds = []
        deepcad_cmds = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)
        assert deepcad_cmds == []

    def test_empty_command_sequence_deepcad_to_cadling(self):
        """Test converting empty DeepCAD sequence."""
        deepcad_cmds = []
        cadling_cmds = CommandFormatConverter.deepcad_to_cadling(deepcad_cmds)
        assert cadling_cmds == []

    def test_cadling_with_missing_type_key(self):
        """Test handling command with missing type/command key."""
        cadling_cmds = [
            {
                "params": [0.1, 0.2, 0.0, 0.8, 0.9, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        ]
        deepcad_cmds = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)
        # Unknown type should pass through
        assert deepcad_cmds[0] == cadling_cmds[0]

    def test_cadling_with_missing_params_key(self):
        """Test handling command with missing params/parameters key."""
        cadling_cmds = [
            {
                "type": "LINE"
            }
        ]
        deepcad_cmds = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)
        # Should get empty params list
        assert deepcad_cmds[0]["type"] == "LINE"

    def test_params_with_string_numbers(self):
        """Test that string representations of numbers are converted."""
        cadling_cmds = [
            {
                "type": "LINE",
                "params": ["0.1", "0.2", "0.0", "0.8", "0.9", "0.0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"]
            }
        ]
        deepcad_cmds = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)
        assert deepcad_cmds[0]["params"] == [0.1, 0.2, 0.8, 0.9]

    def test_negative_values_preserved(self):
        """Test that negative coordinate values are preserved."""
        cadling_cmds = [
            {
                "type": "LINE",
                "params": [-0.1, -0.2, 0.0, -0.8, -0.9, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        ]
        deepcad_cmds = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)
        assert deepcad_cmds[0]["params"] == [-0.1, -0.2, -0.8, -0.9]

    def test_large_values_preserved(self):
        """Test that large coordinate values are preserved."""
        cadling_cmds = [
            {
                "type": "LINE",
                "params": [1e6, 2e6, 0.0, 8e6, 9e6, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        ]
        deepcad_cmds = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)
        assert deepcad_cmds[0]["params"] == [1e6, 2e6, 8e6, 9e6]

    def test_very_small_values_preserved(self):
        """Test that very small (non-zero) values are preserved."""
        cadling_cmds = [
            {
                "type": "LINE",
                "params": [1e-10, 2e-10, 0.0, 8e-10, 9e-10, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        ]
        deepcad_cmds = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)
        assert deepcad_cmds[0]["params"][0] == pytest.approx(1e-10)
        assert deepcad_cmds[0]["params"][1] == pytest.approx(2e-10)

    def test_zero_values_in_all_positions(self):
        """Test handling all-zero parameters."""
        cadling_cmds = [
            {
                "type": "LINE",
                "params": [0, 0, 0.0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        ]
        deepcad_cmds = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)
        assert deepcad_cmds[0]["params"] == [0, 0, 0, 0]

    def test_excess_cadling_params_ignored(self):
        """Test that excess parameters beyond 16 are ignored."""
        cadling_cmds = [
            {
                "type": "LINE",
                "params": [0.1, 0.2, 0.0, 0.8, 0.9, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 999, 888]
            }
        ]
        deepcad_cmds = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)
        # The excess params (999, 888) should be ignored because indices only reference 0-5
        assert deepcad_cmds[0]["params"] == [0.1, 0.2, 0.8, 0.9]

    def test_deepcad_with_excess_params_ignored(self):
        """Test that excess params in DeepCAD are padded without error."""
        deepcad_cmds = [
            {
                "type": "LINE",
                "params": [0.1, 0.2, 0.8, 0.9, 999, 888]  # Extra params beyond compact_count=4
            }
        ]
        cadling_cmds = CommandFormatConverter.deepcad_to_cadling(deepcad_cmds)
        # Should still pad to 16 correctly
        assert len(cadling_cmds[0]["params"]) == 16
        assert cadling_cmds[0]["params"][0] == 0.1
        assert cadling_cmds[0]["params"][1] == 0.2
        assert cadling_cmds[0]["params"][3] == 0.8
        assert cadling_cmds[0]["params"][4] == 0.9


class TestValidateRoundtrip:
    """Tests for validate_roundtrip() method."""

    def test_validate_roundtrip_valid_line(self):
        """Test validate_roundtrip returns valid=True for LINE."""
        cadling_cmds = [
            {
                "type": "LINE",
                "params": [0.1, 0.2, 0.0, 0.8, 0.9, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        ]
        result = CommandFormatConverter.validate_roundtrip(cadling_cmds)

        assert result["valid"] is True
        assert result["max_error"] == 0.0
        assert len(result["mismatches"]) == 0
        assert result["num_commands"] == 1

    def test_validate_roundtrip_valid_mixed(self):
        """Test validate_roundtrip with mixed command sequence."""
        cadling_cmds = [
            {
                "type": "SOL",
                "params": [1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            },
            {
                "type": "LINE",
                "params": [0.1, 0.2, 0.0, 0.8, 0.9, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            },
            {
                "type": "CIRCLE",
                "params": [0.5, 0.5, 0.0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            },
            {
                "type": "EOS",
                "params": [0] * 16
            }
        ]
        result = CommandFormatConverter.validate_roundtrip(cadling_cmds)

        assert result["valid"] is True
        assert result["max_error"] == 0.0
        assert len(result["mismatches"]) == 0
        assert result["num_commands"] == 4

    def test_validate_roundtrip_reports_max_error(self):
        """Test that validate_roundtrip reports maximum error."""
        cadling_cmds = [
            {
                "type": "LINE",
                "params": [0.1, 0.2, 0.0, 0.8, 0.9, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        ]
        result = CommandFormatConverter.validate_roundtrip(cadling_cmds, tolerance=1e-8)

        assert result["max_error"] >= 0.0
        assert isinstance(result["max_error"], float)

    def test_validate_roundtrip_with_tolerance(self):
        """Test validate_roundtrip with custom tolerance."""
        cadling_cmds = [
            {
                "type": "LINE",
                "params": [0.1, 0.2, 0.0, 0.8, 0.9, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        ]
        # Very strict tolerance
        result = CommandFormatConverter.validate_roundtrip(cadling_cmds, tolerance=1e-20)

        assert isinstance(result["valid"], bool)
        assert isinstance(result["max_error"], float)

    def test_validate_roundtrip_empty_sequence(self):
        """Test validate_roundtrip with empty command sequence."""
        cadling_cmds = []
        result = CommandFormatConverter.validate_roundtrip(cadling_cmds)

        assert result["valid"] is True
        assert result["num_commands"] == 0

    def test_validate_roundtrip_structure(self):
        """Test that validate_roundtrip returns correct structure."""
        cadling_cmds = [
            {
                "type": "LINE",
                "params": [0.1, 0.2, 0.0, 0.8, 0.9, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        ]
        result = CommandFormatConverter.validate_roundtrip(cadling_cmds)

        assert "valid" in result
        assert "max_error" in result
        assert "mismatches" in result
        assert "num_commands" in result
        assert isinstance(result["mismatches"], list)


class TestTypeNameNormalization:
    """Tests for type name normalization via _canonical_type()."""

    def test_uppercase_normalization(self):
        """Test that uppercase type names stay uppercase."""
        cadling_cmds = [{"type": "LINE", "params": [0.1, 0.2, 0.0, 0.8, 0.9, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}]
        deepcad = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)
        assert deepcad[0]["type"] == "LINE"

    def test_lowercase_normalization(self):
        """Test that lowercase type names are converted to uppercase."""
        cadling_cmds = [{"type": "line", "params": [0.1, 0.2, 0.0, 0.8, 0.9, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}]
        deepcad = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)
        assert deepcad[0]["type"] == "LINE"

    def test_mixed_case_normalization_line(self):
        """Test that 'Line' is normalized to 'LINE'."""
        cadling_cmds = [{"type": "Line", "params": [0.1, 0.2, 0.0, 0.8, 0.9, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}]
        deepcad = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)
        assert deepcad[0]["type"] == "LINE"

    def test_mixed_case_normalization_arc(self):
        """Test that 'Arc' is normalized to 'ARC'."""
        cadling_cmds = [{"type": "Arc", "params": [0.1, 0.2, 0.0, 0.5, 0.6, 0.0, 0.9, 0.95, 0.0, 0, 0, 0, 0, 0, 0, 0]}]
        deepcad = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)
        assert deepcad[0]["type"] == "ARC"

    def test_mixed_case_normalization_circle(self):
        """Test that 'Circle' is normalized to 'CIRCLE'."""
        cadling_cmds = [{"type": "Circle", "params": [0.5, 0.5, 0.0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}]
        deepcad = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)
        assert deepcad[0]["type"] == "CIRCLE"

    def test_mixed_case_normalization_extrude(self):
        """Test that 'Ext' is normalized to 'EXTRUDE'."""
        cadling_cmds = [{"type": "Ext", "params": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0, 0, 0, 0, 0, 0, 0, 0]}]
        deepcad = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)
        assert deepcad[0]["type"] == "EXTRUDE"

    def test_mixed_case_normalization_sol(self):
        """Test that 'Sol' is normalized to 'SOL'."""
        cadling_cmds = [{"type": "Sol", "params": [1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}]
        deepcad = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)
        assert deepcad[0]["type"] == "SOL"

    def test_mixed_case_normalization_eos(self):
        """Test that 'Eos' is normalized to 'EOS'."""
        cadling_cmds = [{"type": "Eos", "params": [0] * 16}]
        deepcad = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)
        assert deepcad[0]["type"] == "EOS"


class TestParameterIndexMapping:
    """Tests verifying correct parameter index mapping per CADLING_PARAM_MAP."""

    def test_line_indices_correct(self):
        """Verify LINE indices [0,1,3,4] extract correct positions."""
        # cadling: [x1, y1, z1, x2, y2, z2, ...] = [0.1, 0.2, 0.0, 0.8, 0.9, 0.0, ...]
        # deepcad: [x1, y1, x2, y2] = [0.1, 0.2, 0.8, 0.9]
        assert CADLING_PARAM_MAP["LINE"]["indices"] == [0, 1, 3, 4]
        cadling_cmds = [
            {"type": "LINE", "params": [0.1, 0.2, 0.0, 0.8, 0.9, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
        ]
        deepcad = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)
        assert deepcad[0]["params"] == [0.1, 0.2, 0.8, 0.9]

    def test_arc_indices_correct(self):
        """Verify ARC indices [0,1,3,4,6,7] extract correct positions."""
        # cadling: [x1, y1, z1, x2, y2, z2, x3, y3, z3, ...]
        # deepcad: [x1, y1, x2, y2, x3, y3]
        assert CADLING_PARAM_MAP["ARC"]["indices"] == [0, 1, 3, 4, 6, 7]

    def test_circle_indices_correct(self):
        """Verify CIRCLE indices [0,1,3] extract correct positions."""
        # cadling: [cx, cy, z, r, ...]
        # deepcad: [cx, cy, r]
        assert CADLING_PARAM_MAP["CIRCLE"]["indices"] == [0, 1, 3]
        cadling_cmds = [
            {"type": "CIRCLE", "params": [0.5, 0.5, 0.0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
        ]
        deepcad = CommandFormatConverter.cadling_to_deepcad(cadling_cmds)
        assert deepcad[0]["params"] == [0.5, 0.5, 0.25]

    def test_sol_indices_correct(self):
        """Verify SOL indices [0,1] extract correct positions."""
        assert CADLING_PARAM_MAP["SOL"]["indices"] == [0, 1]

    def test_extrude_indices_correct(self):
        """Verify EXTRUDE indices are [0..7]."""
        assert CADLING_PARAM_MAP["EXTRUDE"]["indices"] == list(range(8))

    def test_eos_indices_correct(self):
        """Verify EOS has no indices."""
        assert CADLING_PARAM_MAP["EOS"]["indices"] == []
