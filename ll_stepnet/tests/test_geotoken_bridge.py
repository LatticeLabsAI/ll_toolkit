"""Tests for ll_stepnet native format alignment with geotoken/cadling.

Verifies that ll_stepnet's CommandType, PARAMETER_MASKS, STEPGraphEncoder,
and GeoTokenDataset all use native formats that match geotoken and cadling
directly — no adapters or conversion needed.
"""
import sys
import os
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stepnet.output_heads import CommandType, PARAMETER_MASKS, CompositeHead
from stepnet.topology import STEPTopologyBuilder
from stepnet.encoder import STEPGraphEncoder, STEPEncoder
from stepnet.data import GeoTokenDataset, GeoTokenCollator


# ---------------------------------------------------------------------------
# Mock types (avoid hard dependency on geotoken/cadling in ll_stepnet tests)
# ---------------------------------------------------------------------------

class MockCommandType:
    """Minimal mock of geotoken.CommandType enum."""
    def __init__(self, value: str):
        self.value = value


class MockCommandToken:
    """Minimal mock of geotoken.CommandToken."""
    def __init__(self, command_type, parameters=None, parameter_mask=None):
        self.command_type = command_type
        self.parameters = parameters or [0] * 16
        self.parameter_mask = parameter_mask or [False] * 16


class MockTokenSequence:
    """Minimal mock of geotoken.TokenSequence."""
    def __init__(self, command_tokens=None):
        self.command_tokens = command_tokens or []


# ===========================================================================
# CommandType native format alignment tests
# ===========================================================================

class TestCommandTypeAlignment:
    """Verify ll_stepnet CommandType matches geotoken's enum order."""

    def test_sol_is_zero(self):
        """SOL must be index 0 (matches geotoken enum position)."""
        assert CommandType.SOL == 0

    def test_line_is_one(self):
        """LINE must be index 1."""
        assert CommandType.LINE == 1

    def test_arc_is_two(self):
        """ARC must be index 2."""
        assert CommandType.ARC == 2

    def test_circle_is_three(self):
        """CIRCLE must be index 3."""
        assert CommandType.CIRCLE == 3

    def test_extrude_is_four(self):
        """EXTRUDE must be index 4."""
        assert CommandType.EXTRUDE == 4

    def test_eos_is_five(self):
        """EOS must be index 5."""
        assert CommandType.EOS == 5

    def test_six_command_types(self):
        """Exactly 6 command types."""
        assert len(CommandType) == 6

    def test_no_bezier_or_ellipse(self):
        """Old BEZIER/ELLIPSE names must not exist."""
        assert not hasattr(CommandType, 'BEZIER')
        assert not hasattr(CommandType, 'ELLIPSE')


# ===========================================================================
# PARAMETER_MASKS native alignment tests
# ===========================================================================

class TestParameterMasksAlignment:
    """Verify PARAMETER_MASKS match geotoken's CommandToken.get_parameter_mask()."""

    def test_sol_mask(self):
        """SOL: 2 active params [0, 1]."""
        assert PARAMETER_MASKS[CommandType.SOL] == [0, 1]

    def test_line_mask(self):
        """LINE: 4 active params [0, 1, 2, 3]."""
        assert PARAMETER_MASKS[CommandType.LINE] == [0, 1, 2, 3]

    def test_arc_mask(self):
        """ARC: 6 active params [0..5] — 3-point format, no sweep flag."""
        assert PARAMETER_MASKS[CommandType.ARC] == [0, 1, 2, 3, 4, 5]

    def test_circle_mask(self):
        """CIRCLE: 3 active params [0, 1, 2]."""
        assert PARAMETER_MASKS[CommandType.CIRCLE] == [0, 1, 2]

    def test_extrude_mask(self):
        """EXTRUDE: 8 active params [0..7]."""
        assert PARAMETER_MASKS[CommandType.EXTRUDE] == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_eos_mask(self):
        """EOS: 0 active params."""
        assert PARAMETER_MASKS[CommandType.EOS] == []

    def test_all_types_covered(self):
        """Every CommandType has a PARAMETER_MASKS entry."""
        for ct in CommandType:
            assert ct in PARAMETER_MASKS


# ===========================================================================
# STEPGraphEncoder native 48-dim tests
# ===========================================================================

class TestGraphEncoderNativeDim:
    """Verify STEPGraphEncoder defaults to cadling's 48-dim features."""

    def test_default_input_dim_is_48(self):
        """Default input_dim should be 48 (cadling native)."""
        enc = STEPGraphEncoder()
        assert enc.input_dim == 48

    def test_accepts_48dim_features(self):
        """Encoder processes 48-dim features without error."""
        enc = STEPGraphEncoder(input_dim=48)
        nodes = torch.randn(5, 48)
        adj = torch.eye(5)
        out = enc(nodes, adj)
        assert out.shape == (5, 128)  # default node_dim=128

    def test_accepts_129dim_for_backward_compat(self):
        """Encoder still works with 129-dim when explicitly configured."""
        enc = STEPGraphEncoder(input_dim=129)
        nodes = torch.randn(5, 129)
        adj = torch.eye(5)
        out = enc(nodes, adj)
        assert out.shape == (5, 128)


class TestSTEPEncoderNativeDim:
    """Verify STEPEncoder uses 48-dim graph input by default."""

    def test_default_graph_input_dim(self):
        """STEPEncoder should default to graph_input_dim=48."""
        enc = STEPEncoder()
        assert enc.graph_encoder.input_dim == 48

    def test_topology_with_48dim_features(self):
        """STEPEncoder accepts cadling's 48-dim topology natively."""
        enc = STEPEncoder(vocab_size=100, token_embed_dim=32,
                          graph_node_dim=16, graph_input_dim=48, output_dim=64)
        token_ids = torch.randint(0, 100, (2, 10))
        topo = {
            'adjacency_matrix': torch.eye(3),
            'node_features': torch.randn(3, 48),
        }
        out = enc(token_ids, topo)
        assert out.shape == (2, 64)


# ===========================================================================
# Topology compact features tests
# ===========================================================================

class TestCompactNodeFeatures:
    """Test STEPTopologyBuilder.build_compact_node_features()."""

    def _make_features_list(self):
        return [
            {'entity_id': 1, 'entity_type': 'ADVANCED_FACE', 'references': [2],
             'numeric_params': [1.0, 2.0, 3.0]},
            {'entity_id': 2, 'entity_type': 'EDGE_CURVE', 'references': [],
             'numeric_params': [4.0, 5.0]},
            {'entity_id': 3, 'entity_type': 'VERTEX_POINT', 'references': [],
             'numeric_params': [0.0, 0.0, 0.0]},
        ]

    def test_output_shape(self):
        """Compact features are (N, 48) by default."""
        builder = STEPTopologyBuilder()
        feats = builder.build_compact_node_features(self._make_features_list())
        assert feats.shape == (3, 48)

    def test_numeric_params_populated(self):
        """First 32 dims contain numeric parameters."""
        builder = STEPTopologyBuilder()
        feats = builder.build_compact_node_features(self._make_features_list())
        # Node 0: params [1, 2, 3]
        assert feats[0, 0].item() == 1.0
        assert feats[0, 1].item() == 2.0
        assert feats[0, 2].item() == 3.0

    def test_entity_type_one_hot(self):
        """Dims [32:48] contain entity type one-hot."""
        builder = STEPTopologyBuilder()
        feats = builder.build_compact_node_features(self._make_features_list())
        # ADVANCED_FACE is index 0 in the type list
        assert feats[0, 32].item() == 1.0
        # EDGE_CURVE is index 5
        assert feats[1, 37].item() == 1.0
        # VERTEX_POINT is index 6
        assert feats[2, 38].item() == 1.0

    def test_configurable_dim(self):
        """Feature dim is configurable."""
        builder = STEPTopologyBuilder()
        feats = builder.build_compact_node_features(self._make_features_list(), feature_dim=64)
        assert feats.shape == (3, 64)


# ===========================================================================
# GeoTokenDataset native format tests
# ===========================================================================

class TestGeoTokenDatasetNative:
    """Test GeoTokenDataset with natively-aligned command types."""

    def _make_token_sequence(self):
        """Create mock TokenSequence using geotoken command names."""
        cmds = [
            MockCommandToken(
                MockCommandType("SOL"),
                parameters=[10, 20] + [0] * 14,
                parameter_mask=[True, True] + [False] * 14,
            ),
            MockCommandToken(
                MockCommandType("LINE"),
                parameters=[10, 20, 30, 40] + [0] * 12,
                parameter_mask=[True, True, True, True] + [False] * 12,
            ),
            MockCommandToken(
                MockCommandType("CIRCLE"),
                parameters=[50, 50, 25] + [0] * 13,
                parameter_mask=[True, True, True] + [False] * 13,
            ),
            MockCommandToken(
                MockCommandType("EOS"),
                parameters=[0] * 16,
                parameter_mask=[False] * 16,
            ),
        ]
        return MockTokenSequence(command_tokens=cmds)

    def test_sol_maps_to_zero(self):
        """SOL command type maps to integer 0 natively."""
        dataset = GeoTokenDataset([self._make_token_sequence()], max_commands=10)
        item = dataset[0]
        assert item['command_types'][0].item() == 0  # SOL = 0

    def test_line_maps_to_one(self):
        """LINE command type maps to integer 1 natively."""
        dataset = GeoTokenDataset([self._make_token_sequence()], max_commands=10)
        item = dataset[0]
        assert item['command_types'][1].item() == 1  # LINE = 1

    def test_circle_maps_to_three(self):
        """CIRCLE command type maps to integer 3 natively."""
        dataset = GeoTokenDataset([self._make_token_sequence()], max_commands=10)
        item = dataset[0]
        assert item['command_types'][2].item() == 3  # CIRCLE = 3

    def test_eos_maps_to_five(self):
        """EOS command type maps to integer 5 natively."""
        dataset = GeoTokenDataset([self._make_token_sequence()], max_commands=10)
        item = dataset[0]
        assert item['command_types'][3].item() == 5  # EOS = 5

    def test_no_mapping_table_needed(self):
        """The integer values match CommandType IntEnum directly."""
        dataset = GeoTokenDataset([self._make_token_sequence()], max_commands=10)
        item = dataset[0]
        assert item['command_types'][0].item() == CommandType.SOL
        assert item['command_types'][1].item() == CommandType.LINE
        assert item['command_types'][2].item() == CommandType.CIRCLE
        assert item['command_types'][3].item() == CommandType.EOS

    def test_shapes(self):
        """Output tensor shapes are correct."""
        dataset = GeoTokenDataset([self._make_token_sequence()], max_commands=10)
        item = dataset[0]
        assert item['command_types'].shape == (10,)
        assert item['parameters'].shape == (10, 16)
        assert item['parameter_mask'].shape == (10, 16)
        assert item['attention_mask'].shape == (10,)

    def test_padding(self):
        """Short sequences padded correctly."""
        dataset = GeoTokenDataset([self._make_token_sequence()], max_commands=10)
        item = dataset[0]
        assert item['attention_mask'][:4].sum().item() == 4
        assert item['attention_mask'][4:].sum().item() == 0

    def test_labels(self):
        """Labels included when provided."""
        dataset = GeoTokenDataset([self._make_token_sequence()], max_commands=10, labels=[42])
        assert dataset[0]['label'].item() == 42


class TestGeoTokenCollatorNative:
    """Test GeoTokenCollator with native format."""

    def _make_item(self, max_commands=10):
        cmd_types = torch.zeros(max_commands, dtype=torch.long)
        cmd_types[:3] = torch.tensor([0, 1, 5])  # SOL, LINE, EOS
        return {
            'command_types': cmd_types,
            'parameters': torch.zeros(max_commands, 16, dtype=torch.long),
            'parameter_mask': torch.zeros(max_commands, 16, dtype=torch.bool),
            'attention_mask': torch.cat([torch.ones(3), torch.zeros(7)]).long(),
        }

    def test_batch_stacking(self):
        collator = GeoTokenCollator()
        batch = [self._make_item() for _ in range(4)]
        result = collator(batch)
        assert result['command_types'].shape == (4, 10)

    def test_labels(self):
        collator = GeoTokenCollator()
        items = [self._make_item() for _ in range(3)]
        for i, item in enumerate(items):
            item['label'] = torch.tensor(i)
        result = collator(items)
        assert result['labels'].tolist() == [0, 1, 2]


# ===========================================================================
# CompositeHead with new command types
# ===========================================================================

class TestCompositeHeadAlignment:
    """Test CompositeHead works with the updated CommandType enum."""

    def test_creates_without_error(self):
        """CompositeHead initializes with new 6 command types."""
        head = CompositeHead(embed_dim=32, num_command_types=6,
                             num_param_slots=16, num_levels=64)
        assert head.num_command_types == 6

    def test_mask_buffer_shape(self):
        """Internal mask buffer has correct shape."""
        head = CompositeHead(embed_dim=32)
        assert head._param_mask.shape == (6, 16)

    def test_sol_mask_in_buffer(self):
        """SOL mask: first 2 slots active."""
        head = CompositeHead(embed_dim=32)
        sol_mask = head._param_mask[CommandType.SOL]
        assert sol_mask[0].item() is True
        assert sol_mask[1].item() is True
        assert sol_mask[2].item() is False

    def test_extrude_mask_in_buffer(self):
        """EXTRUDE mask: first 8 slots active."""
        head = CompositeHead(embed_dim=32)
        ext_mask = head._param_mask[CommandType.EXTRUDE]
        assert ext_mask[:8].all().item() is True
        assert ext_mask[8:].any().item() is False

    def test_forward_pass(self):
        """Forward pass produces correct output shapes."""
        head = CompositeHead(embed_dim=32, num_levels=64)
        hidden = torch.randn(2, 5, 32)
        out = head(hidden)
        assert out['command_type_logits'].shape == (2, 5, 6)
        assert len(out['parameter_logits']) == 16
        assert out['parameter_logits'][0].shape == (2, 5, 64)
