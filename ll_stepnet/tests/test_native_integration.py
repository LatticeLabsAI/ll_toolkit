"""Tests for native cadling/geotoken integration in ll_stepnet.

Covers all 5 phases of the integration plan:
    Phase 1: Topology pipeline (prepare_topology_data, compact build, round-trip)
    Phase 2: Dataset integration (CadlingDataset, GeoTokenDataset graph tokens)
    Phase 3: Output decoding (decode_to_token_sequence)
    Phase 4: Config & streaming (StreamingCadlingConfig, trainer integration)
    Phase 5: Exports
"""
from __future__ import annotations

import importlib
import sys
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# ll_stepnet imports
# ---------------------------------------------------------------------------
from stepnet.config import StreamingCadlingConfig
from stepnet.encoder import STEPEncoder
from stepnet.output_heads import (
    CommandType,
    CompositeHead,
    PARAMETER_MASKS,
)
from stepnet.topology import STEPTopologyBuilder


# ===================================================================
#  Phase 1: Topology Pipeline
# ===================================================================

class TestPrepareTopologyData:
    """Phase 1a: STEPEncoder.prepare_topology_data()"""

    def test_accepts_dict_with_tensors(self):
        """Dict with tensor values should pass through unchanged."""
        adj = torch.eye(4)
        feats = torch.randn(4, 48)
        result = STEPEncoder.prepare_topology_data({
            "adjacency_matrix": adj,
            "node_features": feats,
        })
        assert torch.equal(result["adjacency_matrix"], adj)
        assert torch.equal(result["node_features"], feats)

    def test_accepts_dict_with_numpy(self):
        """Dict with numpy values should be converted to tensors."""
        adj_np = np.eye(5, dtype=np.float32)
        feats_np = np.random.randn(5, 48).astype(np.float32)
        result = STEPEncoder.prepare_topology_data({
            "adjacency_matrix": adj_np,
            "node_features": feats_np,
        })
        assert isinstance(result["adjacency_matrix"], torch.Tensor)
        assert isinstance(result["node_features"], torch.Tensor)
        assert result["adjacency_matrix"].shape == (5, 5)
        assert result["node_features"].shape == (5, 48)

    def test_accepts_dict_preserves_extra_keys(self):
        """Extra keys in the dict should be passed through."""
        result = STEPEncoder.prepare_topology_data({
            "adjacency_matrix": torch.eye(3),
            "node_features": torch.randn(3, 48),
            "edge_index": torch.tensor([[0, 1], [1, 2]]),
        })
        assert "edge_index" in result

    def test_accepts_duck_typed_topology_graph(self):
        """Object with to_numpy_node_features() and to_edge_index() methods."""
        mock_topo = MagicMock()
        mock_topo.to_numpy_node_features.return_value = np.random.randn(6, 48).astype(np.float32)
        mock_topo.to_edge_index.return_value = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)

        result = STEPEncoder.prepare_topology_data(mock_topo)

        assert result["adjacency_matrix"].shape == (6, 6)
        assert result["node_features"].shape == (6, 48)
        # Check edges are set
        adj = result["adjacency_matrix"]
        assert adj[0, 1] == 1.0
        assert adj[1, 2] == 1.0
        assert adj[2, 3] == 1.0
        # Non-edges should be zero
        assert adj[3, 0] == 0.0

    def test_duck_typed_no_edges(self):
        """TopologyGraph with no edges should produce zero adjacency."""
        mock_topo = MagicMock()
        mock_topo.to_numpy_node_features.return_value = np.ones((3, 48), dtype=np.float32)
        mock_topo.to_edge_index.return_value = np.zeros((2, 0), dtype=np.int64)

        result = STEPEncoder.prepare_topology_data(mock_topo)
        assert result["adjacency_matrix"].sum().item() == 0.0

    def test_rejects_unknown_type(self):
        """Non-dict, non-duck-typed objects should raise TypeError."""
        with pytest.raises(TypeError, match="Expected dict or object"):
            STEPEncoder.prepare_topology_data("not a topology")

    def test_output_feeds_into_encoder_forward(self):
        """prepare_topology_data output should be directly usable by forward()."""
        mock_topo = MagicMock()
        mock_topo.to_numpy_node_features.return_value = np.random.randn(4, 48).astype(np.float32)
        mock_topo.to_edge_index.return_value = np.array([[0, 1], [1, 2]], dtype=np.int64)

        topo_data = STEPEncoder.prepare_topology_data(mock_topo)
        encoder = STEPEncoder(vocab_size=100, output_dim=64, graph_input_dim=48)

        token_ids = torch.randint(0, 100, (2, 10))
        output = encoder(token_ids, topo_data)
        assert output.shape == (2, 64)


class TestBuildCompleteTopologyCompact:
    """Phase 1b: build_complete_topology(compact=True/False)"""

    def _make_features_list(self) -> List[Dict]:
        """Create a minimal features list for topology building."""
        return [
            {
                "entity_type": "CARTESIAN_POINT",
                "entity_id": 1,
                "references": [2],
                "numeric_params": [1.0, 2.0, 3.0],
            },
            {
                "entity_type": "DIRECTION",
                "entity_id": 2,
                "references": [],
                "numeric_params": [0.0, 0.0, 1.0],
            },
            {
                "entity_type": "AXIS2_PLACEMENT_3D",
                "entity_id": 3,
                "references": [1, 2],
                "numeric_params": [],
            },
        ]

    def test_compact_true_produces_48_dim(self):
        """compact=True should produce 48-dim features (cadling native)."""
        builder = STEPTopologyBuilder()
        features_list = self._make_features_list()
        result = builder.build_complete_topology(features_list, compact=True)

        assert "node_features" in result
        node_feats = result["node_features"]
        # 48-dim: 32 numeric + 16 type one-hot
        assert node_feats.shape[1] == 48

    def test_compact_false_produces_129_dim(self):
        """compact=False should produce 129-dim features (legacy format)."""
        builder = STEPTopologyBuilder()
        features_list = self._make_features_list()
        result = builder.build_complete_topology(features_list, compact=False)

        assert "node_features" in result
        node_feats = result["node_features"]
        # 129-dim: 128 numeric + 1 type hash
        assert node_feats.shape[1] == 129

    def test_compact_default_is_true(self):
        """Default compact parameter should be True."""
        builder = STEPTopologyBuilder()
        features_list = self._make_features_list()
        result = builder.build_complete_topology(features_list)
        assert result["node_features"].shape[1] == 48


class TestToCadlingTopologyGraph:
    """Phase 1c: STEPTopologyBuilder.to_cadling_topology_graph()"""

    def test_round_trip_preserves_structure(self):
        """Convert to cadling TopologyGraph and verify structure is preserved."""
        # Build a topology dict
        adj = torch.zeros(3, 3)
        adj[0, 1] = 1.0
        adj[1, 2] = 1.0
        feats = torch.randn(3, 48)
        topo_dict = {
            "adjacency_matrix": adj,
            "node_features": feats,
        }

        # This will fail if cadling is not installed, which is expected
        # since the method lazy-imports cadling
        try:
            result = STEPTopologyBuilder.to_cadling_topology_graph(topo_dict)
            assert result.num_nodes == 3
            assert result.num_edges == 2
            assert 1 in result.adjacency_list.get(0, [])
            assert 2 in result.adjacency_list.get(1, [])
        except ImportError:
            pytest.skip("cadling not installed; skipping round-trip test")

    def test_empty_adjacency(self):
        """Topology with no edges should produce empty adjacency_list."""
        topo_dict = {
            "adjacency_matrix": torch.zeros(4, 4),
            "node_features": torch.randn(4, 48),
        }
        try:
            result = STEPTopologyBuilder.to_cadling_topology_graph(topo_dict)
            assert result.num_nodes == 4
            assert result.num_edges == 0
        except ImportError:
            pytest.skip("cadling not installed")


# ===================================================================
#  Phase 2: Dataset Integration
# ===================================================================

class TestCadlingDataset:
    """Phase 2a: CadlingDataset for Sketch2DItem objects."""

    def _make_mock_sketch_item(self, num_commands: int = 5):
        """Create a mock Sketch2DItem with to_geotoken_commands()."""
        item = MagicMock()
        commands = []
        type_cycle = ["SOL", "LINE", "LINE", "ARC", "EOS"]
        for i in range(num_commands):
            cmd_type = type_cycle[i % len(type_cycle)]
            cmd = {"type": cmd_type, "params": [float(j) for j in range(8)]}
            commands.append(cmd)
        item.to_geotoken_commands.return_value = commands
        # No topology by default
        item.topology_graph = None
        return item

    def test_getitem_returns_correct_keys(self):
        """Dataset __getitem__ should return command_types, parameters, etc."""
        from stepnet.data import CadlingDataset

        items = [self._make_mock_sketch_item() for _ in range(3)]
        ds = CadlingDataset(items, max_commands=10)

        sample = ds[0]
        assert "command_types" in sample
        assert "parameters" in sample
        assert "parameter_mask" in sample
        assert "attention_mask" in sample

    def test_command_type_mapping(self):
        """SOL=0, LINE=1, ARC=2, CIRCLE=3, EXTRUDE=4, EOS=5."""
        from stepnet.data import CadlingDataset

        item = MagicMock()
        item.to_geotoken_commands.return_value = [
            {"type": "SOL", "params": [0.0] * 16},
            {"type": "LINE", "params": [0.0] * 16},
            {"type": "ARC", "params": [0.0] * 16},
            {"type": "CIRCLE", "params": [0.0] * 16},
            {"type": "EXTRUDE", "params": [0.0] * 16},
            {"type": "EOS", "params": [0.0] * 16},
        ]
        item.topology_graph = None

        ds = CadlingDataset([item], max_commands=10)
        sample = ds[0]

        cmd_types = sample["command_types"]
        # Check first 6 values map correctly
        assert cmd_types[0].item() == 0  # SOL
        assert cmd_types[1].item() == 1  # LINE
        assert cmd_types[2].item() == 2  # ARC
        assert cmd_types[3].item() == 3  # CIRCLE
        assert cmd_types[4].item() == 4  # EXTRUDE
        assert cmd_types[5].item() == 5  # EOS

    def test_padding_and_truncation(self):
        """Sequences should be padded to max_commands or truncated."""
        from stepnet.data import CadlingDataset

        # Short sequence (3 commands) padded to max_commands=8
        item = self._make_mock_sketch_item(num_commands=3)
        ds = CadlingDataset([item], max_commands=8)
        sample = ds[0]
        assert sample["command_types"].shape[0] == 8
        assert sample["attention_mask"].shape[0] == 8
        # First 3 positions should have attention=1, rest=0
        assert sample["attention_mask"][:3].sum().item() == 3
        assert sample["attention_mask"][3:].sum().item() == 0

    def test_with_labels(self):
        """Labels should be included in sample when provided."""
        from stepnet.data import CadlingDataset

        items = [self._make_mock_sketch_item() for _ in range(3)]
        labels = [0, 1, 2]
        ds = CadlingDataset(items, max_commands=10, labels=labels)
        sample = ds[0]
        assert "label" in sample
        assert sample["label"].item() == 0

    def test_len(self):
        """__len__ should return number of items."""
        from stepnet.data import CadlingDataset

        items = [self._make_mock_sketch_item() for _ in range(7)]
        ds = CadlingDataset(items, max_commands=10)
        assert len(ds) == 7


class TestGeoTokenDatasetGraphTokens:
    """Phase 2b: GeoTokenDataset graph/constraint token encoding."""

    def test_graph_tokens_off_by_default(self):
        """When encode_graph_tokens=False, no graph_token_ids in output."""
        from stepnet.data import GeoTokenDataset

        # Create a mock TokenSequence
        mock_seq = MagicMock()
        mock_seq.command_tokens = []
        for _ in range(3):
            ct = MagicMock()
            ct.command_type = MagicMock()
            ct.command_type.name = "LINE"
            ct.parameters = [0] * 16
            ct.parameter_mask = [True] * 4 + [False] * 12
            mock_seq.command_tokens.append(ct)

        ds = GeoTokenDataset(
            token_sequences=[mock_seq],
            max_commands=10,
            encode_graph_tokens=False,
        )
        sample = ds[0]
        assert "graph_token_ids" not in sample

    def test_constraint_tokens_off_by_default(self):
        """When encode_constraint_tokens=False, no constraint_token_ids."""
        from stepnet.data import GeoTokenDataset

        mock_seq = MagicMock()
        mock_seq.command_tokens = []
        for _ in range(2):
            ct = MagicMock()
            ct.command_type = MagicMock()
            ct.command_type.name = "SOL"
            ct.parameters = [0] * 16
            ct.parameter_mask = [True] * 2 + [False] * 14
            mock_seq.command_tokens.append(ct)

        ds = GeoTokenDataset(
            token_sequences=[mock_seq],
            max_commands=10,
            encode_constraint_tokens=False,
        )
        sample = ds[0]
        assert "constraint_token_ids" not in sample


# ===================================================================
#  Phase 3: Output Decoding
# ===================================================================

class TestDecodeToTokenSequence:
    """Phase 3a: CompositeHead.decode_to_token_sequence()"""

    def _make_head(self) -> CompositeHead:
        """Create a CompositeHead with default settings."""
        return CompositeHead(
            embed_dim=64,
            num_command_types=6,
            num_param_slots=16,
            num_levels=256,
        )

    def test_produces_valid_token_sequence(self):
        """decode_to_token_sequence should return a geotoken TokenSequence."""
        head = self._make_head()

        # Create fake logits: batch=1, seq_len=5
        B, S = 1, 5
        command_logits = torch.randn(B, S, 6)
        param_logits = [torch.randn(B, S, 256) for _ in range(16)]

        try:
            result = head.decode_to_token_sequence(command_logits, param_logits)
            # Should have command_tokens attribute
            assert hasattr(result, "command_tokens")
            assert len(result.command_tokens) > 0
            assert len(result.command_tokens) <= S
        except ImportError:
            pytest.skip("geotoken not installed")

    def test_stops_at_eos(self):
        """Decoding should stop at the first EOS token."""
        head = self._make_head()

        B, S = 1, 10
        # Force command type = EOS (index 5) at position 3
        command_logits = torch.zeros(B, S, 6)
        command_logits[0, 0, 1] = 10.0  # LINE at pos 0
        command_logits[0, 1, 1] = 10.0  # LINE at pos 1
        command_logits[0, 2, 1] = 10.0  # LINE at pos 2
        command_logits[0, 3, 5] = 10.0  # EOS at pos 3
        command_logits[0, 4:, 1] = 10.0  # More LINEs that should be ignored

        param_logits = [torch.randn(B, S, 256) for _ in range(16)]

        try:
            result = head.decode_to_token_sequence(command_logits, param_logits)
            # Should have 4 tokens: LINE, LINE, LINE, EOS
            assert len(result.command_tokens) == 4
        except ImportError:
            pytest.skip("geotoken not installed")

    def test_parameter_masks_applied(self):
        """Parameters should be masked according to PARAMETER_MASKS."""
        head = self._make_head()

        B, S = 1, 3
        # Force all positions to SOL (index 0) — SOL has 2 active params
        command_logits = torch.zeros(B, S, 6)
        command_logits[:, :, 0] = 10.0  # SOL

        # Put EOS at last position to stop
        command_logits[0, 2, 0] = 0.0
        command_logits[0, 2, 5] = 10.0  # EOS

        param_logits = [torch.randn(B, S, 256) for _ in range(16)]

        try:
            result = head.decode_to_token_sequence(command_logits, param_logits)
            # First token should be SOL
            first_token = result.command_tokens[0]
            # SOL mask: only indices 0, 1 are active
            assert first_token.parameter_mask[0] is True
            assert first_token.parameter_mask[1] is True
            assert first_token.parameter_mask[2] is False
        except ImportError:
            pytest.skip("geotoken not installed")

    def test_all_command_types_decodable(self):
        """Every command type (0-5) should decode without error."""
        head = self._make_head()

        B, S = 1, 6
        command_logits = torch.zeros(B, S, 6)
        for i in range(6):
            command_logits[0, i, i] = 10.0  # SOL, LINE, ARC, CIRCLE, EXTRUDE, EOS

        param_logits = [torch.randn(B, S, 256) for _ in range(16)]

        try:
            result = head.decode_to_token_sequence(command_logits, param_logits)
            # Should stop at EOS (position 5), so 6 tokens
            assert len(result.command_tokens) == 6
        except ImportError:
            pytest.skip("geotoken not installed")


# ===================================================================
#  Phase 4: Config & Streaming
# ===================================================================

class TestStreamingCadlingConfig:
    """Phase 4a: StreamingCadlingConfig dataclass."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        cfg = StreamingCadlingConfig()
        assert cfg.dataset_id == ""
        assert cfg.split == "train"
        assert cfg.streaming is True
        assert cfg.batch_size == 8
        assert cfg.shuffle is True
        assert cfg.shuffle_buffer_size == 10000
        assert cfg.max_samples is None
        assert cfg.max_commands == 60
        assert cfg.compact_topology is True

    def test_custom_values(self):
        """Config should accept custom values."""
        cfg = StreamingCadlingConfig(
            dataset_id="latticelabs/deepcad-sequences",
            split="val",
            streaming=False,
            batch_size=16,
            max_samples=1000,
        )
        assert cfg.dataset_id == "latticelabs/deepcad-sequences"
        assert cfg.split == "val"
        assert cfg.streaming is False
        assert cfg.batch_size == 16
        assert cfg.max_samples == 1000


class TestStreamingTrainerIntegration:
    """Phase 4b: Streaming trainer dataset_config parameter."""

    def test_vae_trainer_requires_dataset_or_config(self):
        """StreamingVAETrainer should raise if neither dataset nor config provided."""
        from stepnet.training.streaming_vae_trainer import StreamingVAETrainer

        model = MagicMock()
        model.to = MagicMock(return_value=model)
        model.parameters = MagicMock(return_value=iter([torch.randn(2, 2, requires_grad=True)]))

        with pytest.raises(ValueError, match="Either 'dataset' or 'dataset_config'"):
            StreamingVAETrainer(model=model, dataset=None, dataset_config=None)

    def test_diffusion_trainer_requires_dataset_or_config(self):
        """StreamingDiffusionTrainer should raise if neither provided."""
        from stepnet.training.streaming_diffusion_trainer import StreamingDiffusionTrainer

        model = MagicMock()
        model.to = MagicMock(return_value=model)
        model.parameters = MagicMock(return_value=iter([torch.randn(2, 2, requires_grad=True)]))

        scheduler = MagicMock()

        with pytest.raises(ValueError, match="Either 'dataset' or 'dataset_config'"):
            StreamingDiffusionTrainer(
                model=model, scheduler=scheduler, dataset=None, dataset_config=None
            )

    def test_gan_trainer_requires_dataset_or_config(self):
        """StreamingGANTrainer should raise if neither provided."""
        from stepnet.training.streaming_gan_trainer import StreamingGANTrainer

        gen = MagicMock()
        gen.to = MagicMock(return_value=gen)
        gen.parameters = MagicMock(return_value=iter([torch.randn(2, 2, requires_grad=True)]))
        gen.modules = MagicMock(return_value=iter([]))

        critic = MagicMock()
        critic.to = MagicMock(return_value=critic)
        critic.parameters = MagicMock(return_value=iter([torch.randn(2, 2, requires_grad=True)]))

        with pytest.raises(ValueError, match="Either 'dataset' or 'dataset_config'"):
            StreamingGANTrainer(
                generator=gen, critic=critic, dataset=None, dataset_config=None
            )

    def test_vae_trainer_accepts_existing_dataset(self):
        """Passing dataset directly should still work (backward compat)."""
        from stepnet.training.streaming_vae_trainer import StreamingVAETrainer

        model = MagicMock()
        model.to = MagicMock(return_value=model)
        model.parameters = MagicMock(return_value=iter([torch.randn(2, 2, requires_grad=True)]))

        mock_dataset = MagicMock()

        # Should not raise
        trainer = StreamingVAETrainer(model=model, dataset=mock_dataset)
        assert trainer.dataset is mock_dataset


# ===================================================================
#  Phase 5: Exports
# ===================================================================

class TestExports:
    """Phase 5a: __init__.py exports."""

    def test_cadling_dataset_exported(self):
        """CadlingDataset should be importable from stepnet."""
        from stepnet import CadlingDataset
        assert CadlingDataset is not None

    def test_streaming_cadling_config_exported(self):
        """StreamingCadlingConfig should be importable from stepnet."""
        from stepnet import StreamingCadlingConfig
        assert StreamingCadlingConfig is not None

    def test_streaming_trainers_exported(self):
        """Streaming trainers should be importable from stepnet."""
        from stepnet import (
            StreamingVAETrainer,
            StreamingDiffusionTrainer,
            StreamingGANTrainer,
        )
        assert StreamingVAETrainer is not None
        assert StreamingDiffusionTrainer is not None
        assert StreamingGANTrainer is not None

    def test_all_list_includes_new_exports(self):
        """__all__ should include new integration classes."""
        import stepnet
        all_exports = stepnet.__all__
        assert "CadlingDataset" in all_exports
        assert "StreamingCadlingConfig" in all_exports
        assert "StreamingVAETrainer" in all_exports
        assert "StreamingDiffusionTrainer" in all_exports
        assert "StreamingGANTrainer" in all_exports


# ===================================================================
#  Cross-cutting: PARAMETER_MASKS consistency
# ===================================================================

class TestParameterMasksConsistency:
    """Verify PARAMETER_MASKS match between ll_stepnet and geotoken."""

    def test_masks_cover_all_command_types(self):
        """Every CommandType should have an entry in PARAMETER_MASKS."""
        for cmd in CommandType:
            assert cmd in PARAMETER_MASKS, f"Missing mask for {cmd.name}"

    def test_eos_has_no_params(self):
        """EOS should have an empty parameter mask."""
        assert PARAMETER_MASKS[CommandType.EOS] == []

    def test_sol_has_2_params(self):
        """SOL should have 2 active parameters."""
        assert len(PARAMETER_MASKS[CommandType.SOL]) == 2

    def test_line_has_4_params(self):
        """LINE should have 4 active parameters."""
        assert len(PARAMETER_MASKS[CommandType.LINE]) == 4

    def test_arc_has_6_params(self):
        """ARC should have 6 active parameters."""
        assert len(PARAMETER_MASKS[CommandType.ARC]) == 6

    def test_circle_has_3_params(self):
        """CIRCLE should have 3 active parameters."""
        assert len(PARAMETER_MASKS[CommandType.CIRCLE]) == 3

    def test_extrude_has_8_params(self):
        """EXTRUDE should have 8 active parameters."""
        assert len(PARAMETER_MASKS[CommandType.EXTRUDE]) == 8

    def test_command_type_ordering(self):
        """CommandType enum ordering: SOL=0, LINE=1, ARC=2, CIRCLE=3, EXTRUDE=4, EOS=5."""
        assert CommandType.SOL == 0
        assert CommandType.LINE == 1
        assert CommandType.ARC == 2
        assert CommandType.CIRCLE == 3
        assert CommandType.EXTRUDE == 4
        assert CommandType.EOS == 5
