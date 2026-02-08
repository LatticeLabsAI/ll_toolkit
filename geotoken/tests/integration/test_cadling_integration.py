"""End-to-end integration tests for the cadling → geotoken → ll_stepnet pipeline.

Verifies that all three LatticeLabs toolkit packages share native data formats
and that data flows between them with no adapters, bridges, or conversion:

1. cadling produces data in geotoken's exact expected format
2. geotoken tokenizes cadling's output directly
3. ll_stepnet consumes geotoken's TokenSequences with matching enums and masks

These tests use mock data that mirrors the structure of real cadling output
so they run without actual STEP/STL files or heavy ML dependencies.
"""
from __future__ import annotations

import sys
import os
import math
import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow imports from all three packages
# ---------------------------------------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TOOLKIT = os.path.abspath(os.path.join(_ROOT, ".."))

# Add all three package roots to sys.path
for pkg in ["geotoken", "cadling", "ll_stepnet"]:
    pkg_path = os.path.join(_TOOLKIT, pkg)
    if pkg_path not in sys.path:
        sys.path.insert(0, pkg_path)


# ===========================================================================
# Imports from all three packages
# ===========================================================================

# --- geotoken ---
from geotoken import (
    CommandSequenceTokenizer,
    CommandTokenizationConfig,
    GraphTokenizer,
    GraphTokenizationConfig,
    FeatureVectorQuantizer,
    CADVocabulary,
    TokenSequence,
    CommandToken,
    CommandType as GeoCommandType,
    ConstraintType,
    GraphNodeToken,
    GraphEdgeToken,
    GraphStructureToken,
)

# --- cadling ---
from cadling.datamodel.stl import MeshItem
from cadling.datamodel.base_models import (
    TopologyGraph,
    CADItemLabel,
)
from cadling.datamodel.geometry_2d import (
    Line2D,
    Arc2D,
    Circle2D,
    SketchProfile,
    Sketch2DItem,
)

# --- ll_stepnet ---
sys.path.insert(0, os.path.join(_TOOLKIT, "ll_stepnet"))
from stepnet.output_heads import (
    CommandType as StepNetCommandType,
    PARAMETER_MASKS,
    CompositeHead,
)
from stepnet.data import GeoTokenDataset, GeoTokenCollator
from stepnet.encoder import STEPGraphEncoder, STEPEncoder
from stepnet.topology import STEPTopologyBuilder

# Try to import torch (required for ll_stepnet)
import torch


# ===========================================================================
# Helper: build mock cadling data
# ===========================================================================

def _make_mesh_item() -> MeshItem:
    """Create a simple triangulated box mesh item."""
    vertices = [
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 10.0], [0.0, 10.0, 10.0],
    ]
    facets = [
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 6, 5], [4, 7, 6],  # top
        [0, 4, 5], [0, 5, 1],  # front
        [2, 6, 7], [2, 7, 3],  # back
        [0, 3, 7], [0, 7, 4],  # left
        [1, 5, 6], [1, 6, 2],  # right
    ]
    normals = [[0, 0, -1]] * 2 + [[0, 0, 1]] * 2 + [[0, -1, 0]] * 2 + \
              [[0, 1, 0]] * 2 + [[-1, 0, 0]] * 2 + [[1, 0, 0]] * 2
    return MeshItem(
        label=CADItemLabel(text="test_mesh"),
        vertices=vertices,
        facets=facets,
        normals=normals,
        num_vertices=8,
        num_facets=12,
    )


def _make_topology_graph(num_nodes: int = 5, feature_dim: int = 48) -> TopologyGraph:
    """Create a TopologyGraph with node/edge features in cadling's native 48-dim format.

    The adjacency list defines directed edges.  We use a simple chain:
        0→1, 1→2, 2→3, 3→4, 4→0  (5 directed edges)
    so edge_features must have exactly 5 rows.
    """
    np.random.seed(42)
    node_features = np.random.randn(num_nodes, feature_dim).tolist()

    adjacency = {
        0: [1],
        1: [2],
        2: [3],
        3: [4],
        4: [0],
    }
    num_edges = 5  # matches the 5 directed edges above
    edge_features = np.random.randn(num_edges, 16).tolist()

    return TopologyGraph(
        num_nodes=num_nodes,
        num_edges=num_edges,
        adjacency_list=adjacency,
        node_features=node_features,
        edge_features=edge_features,
    )


def _make_sketch_2d_item() -> Sketch2DItem:
    """Create a Sketch2DItem with command_sequence and constraints in properties."""
    profile = SketchProfile(
        profile_id="rect_0",
        primitives=[
            Line2D(start=(0.0, 0.0), end=(10.0, 0.0)),
            Line2D(start=(10.0, 0.0), end=(10.0, 5.0)),
            Line2D(start=(10.0, 5.0), end=(0.0, 5.0)),
            Line2D(start=(0.0, 5.0), end=(0.0, 0.0)),
        ],
        closed=True,
    )

    # Simulate SketchGeometryExtractor output in geotoken-compatible format
    command_sequence = [
        {"type": "SOL", "params": [0.0, 0.0] + [0.0] * 14},
        {"type": "LINE", "params": [0.0, 0.0, 10.0, 0.0] + [0.0] * 12},
        {"type": "LINE", "params": [10.0, 0.0, 10.0, 5.0] + [0.0] * 12},
        {"type": "LINE", "params": [10.0, 5.0, 0.0, 5.0] + [0.0] * 12},
        {"type": "LINE", "params": [0.0, 5.0, 0.0, 0.0] + [0.0] * 12},
        {"type": "EOS", "params": [0.0] * 16},
    ]

    constraints = [
        {"type": "PERPENDICULAR", "entity_a": 1, "entity_b": 2},
        {"type": "PARALLEL", "entity_a": 1, "entity_b": 3},
        {"type": "PERPENDICULAR", "entity_a": 2, "entity_b": 3},
    ]

    item = Sketch2DItem(
        label=CADItemLabel(text="test_sketch"),
        profiles=[profile],
        properties={
            "command_sequence": command_sequence,
            "geometric_constraints": constraints,
        },
    )
    return item


# ===========================================================================
# 1. ENUM AND MASK ALIGNMENT TESTS
#    Verify the contract between geotoken and ll_stepnet is consistent
# ===========================================================================

class TestEnumAlignment:
    """Verify geotoken and ll_stepnet command type enums are in sync."""

    def test_geotoken_command_type_values(self):
        """geotoken CommandType has the 6 expected members."""
        expected = ["SOL", "LINE", "ARC", "CIRCLE", "EXTRUDE", "EOS"]
        actual = [ct.value for ct in GeoCommandType]
        assert actual == expected

    def test_stepnet_command_type_values(self):
        """ll_stepnet CommandType has integer values 0-5."""
        assert StepNetCommandType.SOL == 0
        assert StepNetCommandType.LINE == 1
        assert StepNetCommandType.ARC == 2
        assert StepNetCommandType.CIRCLE == 3
        assert StepNetCommandType.EXTRUDE == 4
        assert StepNetCommandType.EOS == 5

    def test_enum_ordering_matches(self):
        """geotoken string enum ordering matches ll_stepnet integer enum ordering."""
        geo_names = [ct.value for ct in GeoCommandType]
        step_names = [ct.name for ct in StepNetCommandType]
        assert geo_names == step_names

    def test_parameter_mask_active_counts_match(self):
        """Active parameter counts are identical between geotoken and ll_stepnet."""
        for geo_ct in GeoCommandType:
            # Get geotoken's mask (list of 16 bools)
            geo_mask = CommandToken.get_parameter_mask(geo_ct)
            geo_active = sum(geo_mask)

            # Get ll_stepnet's mask (list of active indices)
            step_ct = StepNetCommandType[geo_ct.value]
            step_active = len(PARAMETER_MASKS[step_ct])

            assert geo_active == step_active, (
                f"{geo_ct.value}: geotoken has {geo_active} active params, "
                f"ll_stepnet has {step_active}"
            )

    def test_parameter_mask_indices_match(self):
        """The exact active indices match between geotoken and ll_stepnet."""
        for geo_ct in GeoCommandType:
            geo_mask = CommandToken.get_parameter_mask(geo_ct)
            geo_indices = [i for i, m in enumerate(geo_mask) if m]

            step_ct = StepNetCommandType[geo_ct.value]
            step_indices = PARAMETER_MASKS[step_ct]

            assert geo_indices == step_indices, (
                f"{geo_ct.value}: geotoken indices {geo_indices} != "
                f"ll_stepnet indices {step_indices}"
            )


# ===========================================================================
# 2. CADLING → GEOTOKEN: MeshItem → GeoTokenizer
# ===========================================================================

class TestMeshItemToGeoToken:
    """Verify cadling's MeshItem.to_numpy() produces geotoken-compatible arrays."""

    def test_to_numpy_shapes(self):
        """MeshItem.to_numpy() returns (N,3) float32 + (F,3) int64."""
        mesh = _make_mesh_item()
        vertices, faces = mesh.to_numpy()
        assert vertices.shape == (8, 3)
        assert vertices.dtype == np.float32
        assert faces.shape == (12, 3)
        assert faces.dtype == np.int64

    def test_to_numpy_vertices_only(self):
        """to_numpy_vertices() returns (N,3) float32."""
        mesh = _make_mesh_item()
        verts = mesh.to_numpy_vertices()
        assert verts.shape == (8, 3)
        assert verts.dtype == np.float32

    def test_empty_mesh(self):
        """Empty mesh returns zero-shape arrays."""
        mesh = MeshItem(label=CADItemLabel(text="empty"))
        verts, faces = mesh.to_numpy()
        assert verts.shape == (0, 3)
        assert faces.shape == (0, 3)


# ===========================================================================
# 3. CADLING → GEOTOKEN: TopologyGraph → GraphTokenizer
# ===========================================================================

class TestTopologyGraphToGraphTokenizer:
    """Full pipeline: cadling TopologyGraph → geotoken GraphTokenizer."""

    def test_numpy_export_shapes(self):
        """TopologyGraph.to_numpy_*() produces correct shapes for GraphTokenizer."""
        topo = _make_topology_graph(num_nodes=5, feature_dim=48)

        node_feats = topo.to_numpy_node_features()
        edge_feats = topo.to_numpy_edge_features()
        edge_idx = topo.to_edge_index()

        assert node_feats.shape == (5, 48)
        assert node_feats.dtype == np.float32
        assert edge_feats.shape == (5, 16)
        assert edge_feats.dtype == np.float32
        assert edge_idx.shape == (2, 5)
        assert edge_idx.dtype == np.int64

    def test_graph_tokenizer_accepts_cadling_output(self):
        """GraphTokenizer.tokenize() works directly on cadling's numpy exports."""
        topo = _make_topology_graph(num_nodes=5, feature_dim=48)

        node_feats = topo.to_numpy_node_features()
        edge_idx = topo.to_edge_index()
        edge_feats = topo.to_numpy_edge_features()

        tokenizer = GraphTokenizer(GraphTokenizationConfig(
            node_bits=8, edge_bits=8,
            max_nodes=32, max_edges=64,
        ))

        seq = tokenizer.tokenize(node_feats, edge_idx, edge_feats)

        assert isinstance(seq, TokenSequence)
        assert len(seq.graph_node_tokens) == 5
        assert len(seq.graph_edge_tokens) == 5
        assert seq.metadata["num_nodes"] == 5
        assert seq.metadata["num_edges"] == 5

    def test_graph_tokenizer_roundtrip(self):
        """tokenize → detokenize preserves structure and approximate features."""
        topo = _make_topology_graph(num_nodes=5, feature_dim=48)
        node_feats = topo.to_numpy_node_features()
        edge_idx = topo.to_edge_index()
        edge_feats = topo.to_numpy_edge_features()

        tokenizer = GraphTokenizer(GraphTokenizationConfig(node_bits=8, edge_bits=8))
        seq = tokenizer.tokenize(node_feats, edge_idx, edge_feats)
        recon = tokenizer.detokenize(seq)

        # Structure preserved
        assert recon["num_nodes"] == 5
        assert recon["num_edges"] == 5

        # Feature shapes preserved
        assert recon["node_features"].shape == (5, 48)
        assert recon["edge_features"].shape == (5, 16)
        assert recon["edge_index"].shape == (2, 5)

        # Approximate reconstruction (8-bit quantization)
        node_err = np.abs(recon["node_features"] - node_feats).max()
        # 8-bit → 256 levels, per-dimension. Error depends on range.
        # For standard normal data, range ~6 → step ~6/255 ≈ 0.024
        assert node_err < 0.1, f"Max node reconstruction error {node_err:.4f}"


# ===========================================================================
# 4. CADLING → GEOTOKEN: Sketch2DItem → CommandSequenceTokenizer
# ===========================================================================

class TestSketch2DItemToCommandTokenizer:
    """Full pipeline: cadling Sketch2DItem → geotoken CommandSequenceTokenizer."""

    def test_to_geotoken_commands_format(self):
        """Sketch2DItem.to_geotoken_commands() returns valid command dicts."""
        item = _make_sketch_2d_item()
        cmds = item.to_geotoken_commands()
        assert len(cmds) == 6  # SOL, 4 LINEs, EOS
        assert cmds[0]["type"] == "SOL"
        assert cmds[-1]["type"] == "EOS"
        for cmd in cmds:
            assert "type" in cmd
            assert "params" in cmd
            assert len(cmd["params"]) == 16

    def test_to_geotoken_constraints_format(self):
        """Sketch2DItem.to_geotoken_constraints() returns valid constraint dicts."""
        item = _make_sketch_2d_item()
        constraints = item.to_geotoken_constraints()
        assert len(constraints) == 3
        for c in constraints:
            assert "type" in c
            assert "source_index" in c
            assert "target_index" in c

    def test_command_tokenizer_accepts_cadling_commands(self):
        """CommandSequenceTokenizer.tokenize() works on cadling's command format."""
        item = _make_sketch_2d_item()
        cmds = item.to_geotoken_commands()

        tokenizer = CommandSequenceTokenizer(
            command_config=CommandTokenizationConfig(
                source_format="auto",
                max_sequence_length=60,
                canonicalize_loops=False,
            ),
        )

        seq = tokenizer.tokenize(cmds)

        assert isinstance(seq, TokenSequence)
        assert len(seq.command_tokens) == 60  # padded to max
        assert seq.command_tokens[0].command_type == GeoCommandType.SOL
        assert seq.command_tokens[1].command_type == GeoCommandType.LINE
        assert seq.command_tokens[5].command_type == GeoCommandType.EOS

    def test_constraint_tokenization(self):
        """Constraint dicts from cadling are parsed into ConstraintTokens."""
        item = _make_sketch_2d_item()
        constraints = item.to_geotoken_constraints()

        tokens = CommandSequenceTokenizer.parse_constraints(constraints)
        assert len(tokens) == 3
        assert tokens[0].constraint_type == ConstraintType.PERPENDICULAR
        assert tokens[1].constraint_type == ConstraintType.PARALLEL
        assert tokens[0].source_index == 1
        assert tokens[0].target_index == 2

    def test_command_tokenizer_with_constraints(self):
        """Full tokenization with both commands and constraints."""
        item = _make_sketch_2d_item()
        cmds = item.to_geotoken_commands()
        constraints = item.to_geotoken_constraints()

        tokenizer = CommandSequenceTokenizer(
            command_config=CommandTokenizationConfig(
                source_format="auto",
                include_constraints=True,
                canonicalize_loops=False,
            ),
        )

        seq = tokenizer.tokenize(cmds, constraints=constraints)
        assert len(seq.constraint_tokens) == 3
        assert seq.metadata["num_constraints"] == 3


# ===========================================================================
# 5. GEOTOKEN → LL_STEPNET: TokenSequence → GeoTokenDataset
# ===========================================================================

class _MockCommandType:
    """Minimal mock of geotoken.CommandType enum for GeoTokenDataset."""
    def __init__(self, value: str):
        self.value = value


class _MockCommandToken:
    """Minimal mock of geotoken CommandToken for GeoTokenDataset."""
    def __init__(self, command_type_str, parameters, parameter_mask):
        self.command_type = _MockCommandType(command_type_str)
        self.parameters = parameters
        self.parameter_mask = parameter_mask


class _MockTokenSequence:
    """Minimal mock of geotoken TokenSequence for GeoTokenDataset."""
    def __init__(self, command_tokens):
        self.command_tokens = command_tokens


class TestGeoTokenToStepNet:
    """Full pipeline: geotoken TokenSequence → ll_stepnet GeoTokenDataset."""

    def _make_mock_sequence(self):
        """Create a mock TokenSequence mimicking geotoken output."""
        tokens = [
            _MockCommandToken("SOL", [0, 0] + [0] * 14, [True, True] + [False] * 14),
            _MockCommandToken("LINE", [10, 20, 30, 40] + [0] * 12,
                              [True] * 4 + [False] * 12),
            _MockCommandToken("LINE", [30, 40, 50, 60] + [0] * 12,
                              [True] * 4 + [False] * 12),
            _MockCommandToken("CIRCLE", [50, 50, 25] + [0] * 13,
                              [True] * 3 + [False] * 13),
            _MockCommandToken("EOS", [0] * 16, [False] * 16),
        ]
        return _MockTokenSequence(tokens)

    def test_dataset_native_mapping(self):
        """GeoTokenDataset maps geotoken command types to ll_stepnet integers natively."""
        seq = self._make_mock_sequence()
        dataset = GeoTokenDataset([seq], max_commands=10)
        item = dataset[0]

        # SOL=0, LINE=1, LINE=1, CIRCLE=3, EOS=5 — no conversion needed
        assert item['command_types'][0].item() == StepNetCommandType.SOL  # 0
        assert item['command_types'][1].item() == StepNetCommandType.LINE  # 1
        assert item['command_types'][2].item() == StepNetCommandType.LINE  # 1
        assert item['command_types'][3].item() == StepNetCommandType.CIRCLE  # 3
        assert item['command_types'][4].item() == StepNetCommandType.EOS  # 5

    def test_dataset_shapes(self):
        """GeoTokenDataset output tensors have correct shapes."""
        seq = self._make_mock_sequence()
        dataset = GeoTokenDataset([seq], max_commands=12)
        item = dataset[0]

        assert item['command_types'].shape == (12,)
        assert item['parameters'].shape == (12, 16)
        assert item['parameter_mask'].shape == (12, 16)
        assert item['attention_mask'].shape == (12,)

    def test_dataset_padding(self):
        """Padding fills remaining positions correctly."""
        seq = self._make_mock_sequence()
        dataset = GeoTokenDataset([seq], max_commands=10)
        item = dataset[0]

        assert item['attention_mask'][:5].sum().item() == 5
        assert item['attention_mask'][5:].sum().item() == 0

    def test_collator_batches(self):
        """GeoTokenCollator stacks multiple items into a batch."""
        seq = self._make_mock_sequence()
        dataset = GeoTokenDataset([seq, seq, seq], max_commands=8)

        collator = GeoTokenCollator()
        batch = collator([dataset[0], dataset[1], dataset[2]])

        assert batch['command_types'].shape == (3, 8)
        assert batch['parameters'].shape == (3, 8, 16)


# ===========================================================================
# 6. CADLING → GEOTOKEN → LL_STEPNET: Full TopologyGraph Pipeline
# ===========================================================================

class TestTopologyFullPipeline:
    """cadling TopologyGraph → geotoken GraphTokenizer → ll_stepnet STEPGraphEncoder."""

    def test_topology_to_stepnet_encoder(self):
        """48-dim cadling topology feeds directly into ll_stepnet's graph encoder."""
        topo = _make_topology_graph(num_nodes=5, feature_dim=48)

        # Step 1: cadling → numpy
        node_feats_np = topo.to_numpy_node_features()  # (5, 48)
        assert node_feats_np.shape == (5, 48)

        # Step 2: ll_stepnet graph encoder accepts 48-dim natively
        encoder = STEPGraphEncoder(input_dim=48, node_dim=64)
        node_feats = torch.tensor(node_feats_np, dtype=torch.float32)
        adj = torch.eye(5)  # simple identity adjacency for test
        out = encoder(node_feats, adj)

        assert out.shape == (5, 64)

    def test_compact_features_match_cadling_layout(self):
        """STEPTopologyBuilder.build_compact_node_features() uses cadling's 48-dim layout."""
        builder = STEPTopologyBuilder()
        features_list = [
            {'entity_id': 1, 'entity_type': 'ADVANCED_FACE',
             'references': [2], 'numeric_params': [1.0, 2.0, 3.0]},
            {'entity_id': 2, 'entity_type': 'EDGE_CURVE',
             'references': [], 'numeric_params': [4.0, 5.0]},
        ]
        feats = builder.build_compact_node_features(features_list)

        # Shape matches cadling's native 48-dim
        assert feats.shape == (2, 48)
        # Numeric params in first 32 dims
        assert feats[0, 0].item() == 1.0
        assert feats[0, 1].item() == 2.0
        assert feats[0, 2].item() == 3.0
        # Entity type one-hot in dims [32:48]
        # ADVANCED_FACE is index 0 → dim 32
        assert feats[0, 32].item() == 1.0


# ===========================================================================
# 7. CADLING → GEOTOKEN → LL_STEPNET: Full Command Sequence Pipeline
# ===========================================================================

class TestCommandSequenceFullPipeline:
    """cadling Sketch2DItem → geotoken tokenizer → ll_stepnet dataset/model."""

    def test_sketch_to_dataset_pipeline(self):
        """Full pipeline: Sketch2DItem → CommandSequenceTokenizer → GeoTokenDataset."""
        # Step 1: cadling produces commands
        item = _make_sketch_2d_item()
        cmds = item.to_geotoken_commands()
        assert len(cmds) == 6

        # Step 2: geotoken tokenizes commands
        tokenizer = CommandSequenceTokenizer(
            command_config=CommandTokenizationConfig(
                source_format="auto",
                canonicalize_loops=False,
            ),
        )
        seq = tokenizer.tokenize(cmds)

        # Step 3: Build mock TokenSequence that GeoTokenDataset can consume
        # (GeoTokenDataset works with objects having .command_tokens with
        #  .command_type.value and .parameters and .parameter_mask)
        mock_tokens = []
        for ct in seq.command_tokens:
            mock_tokens.append(_MockCommandToken(
                ct.command_type.value,
                ct.parameters,
                ct.parameter_mask,
            ))
        mock_seq = _MockTokenSequence(mock_tokens)

        # Step 4: ll_stepnet consumes the sequence
        dataset = GeoTokenDataset([mock_seq], max_commands=60)
        sample = dataset[0]

        # Verify command types match through the entire pipeline
        # SOL → 0, LINE → 1, LINE → 1, LINE → 1, LINE → 1, EOS → 5
        assert sample['command_types'][0].item() == StepNetCommandType.SOL
        assert sample['command_types'][1].item() == StepNetCommandType.LINE
        assert sample['command_types'][5].item() == StepNetCommandType.EOS

        # Verify shapes
        assert sample['command_types'].shape == (60,)
        assert sample['parameters'].shape == (60, 16)

    def test_composite_head_on_pipeline_output(self):
        """CompositeHead processes GeoTokenDataset output shapes correctly."""
        head = CompositeHead(embed_dim=32, num_levels=64)

        # Simulate decoder hidden states for a batch of 2, seq_len 10
        hidden = torch.randn(2, 10, 32)
        out = head(hidden)

        # 6 command types matching geotoken
        assert out['command_type_logits'].shape == (2, 10, 6)
        # 16 parameter heads
        assert len(out['parameter_logits']) == 16
        assert out['parameter_logits'][0].shape == (2, 10, 64)


# ===========================================================================
# 8. VOCABULARY ENCODING ROUNDTRIP
# ===========================================================================

class TestVocabularyEncoding:
    """Verify geotoken's CADVocabulary can encode/decode all token types."""

    def test_command_encode_decode(self):
        """Encode → decode preserves command token structure."""
        vocab = CADVocabulary()

        # Create a simple command token list
        ct = CommandToken(
            command_type=GeoCommandType.LINE,
            parameters=[128, 64, 192, 100] + [0] * 12,
            parameter_mask=CommandToken.get_parameter_mask(GeoCommandType.LINE),
        )

        # encode() takes a list of CommandTokens
        ids = vocab.encode([ct])
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_graph_tokens_encodable(self):
        """Graph tokens from GraphTokenizer are vocabulary-encodable."""
        topo = _make_topology_graph(num_nodes=3, feature_dim=48)
        node_feats = topo.to_numpy_node_features()
        edge_idx = topo.to_edge_index()
        edge_feats = topo.to_numpy_edge_features()

        tokenizer = GraphTokenizer(GraphTokenizationConfig(
            node_bits=8, edge_bits=8,
        ))
        seq = tokenizer.tokenize(node_feats, edge_idx, edge_feats)

        vocab = CADVocabulary()

        # Graph node feature tokens should be encodable (takes a list)
        node_ids = vocab.encode_graph_node_features(seq.graph_node_tokens)
        assert len(node_ids) > 0
        assert all(isinstance(i, int) for i in node_ids)

        # Graph structure tokens should be encodable (takes a list)
        struct_ids = vocab.encode_graph_structure(seq.graph_structure_tokens)
        assert len(struct_ids) > 0
        assert all(isinstance(i, int) for i in struct_ids)

    def test_full_sequence_encoding(self):
        """encode_full_sequence() handles a TokenSequence with graph tokens."""
        topo = _make_topology_graph(num_nodes=3, feature_dim=48)
        node_feats = topo.to_numpy_node_features()
        edge_idx = topo.to_edge_index()
        edge_feats = topo.to_numpy_edge_features()

        tokenizer = GraphTokenizer(GraphTokenizationConfig(
            node_bits=8, edge_bits=8,
        ))
        seq = tokenizer.tokenize(node_feats, edge_idx, edge_feats)

        vocab = CADVocabulary()
        all_ids = vocab.encode_full_sequence(seq)
        assert len(all_ids) > 0
        # Should start with BOS and end with EOS
        assert all_ids[0] == 1  # BOS_TOKEN_ID
        assert all_ids[-1] == 2  # EOS_TOKEN_ID


# ===========================================================================
# 9. FEATURE DIMENSION CONSISTENCY
# ===========================================================================

class TestFeatureDimensionConsistency:
    """Verify 48-dim is the shared native dimension across all packages."""

    def test_cadling_topology_native_dim(self):
        """TopologyGraph features default to variable dims; 48 is standard for faces."""
        topo = _make_topology_graph(num_nodes=5, feature_dim=48)
        feats = topo.to_numpy_node_features()
        assert feats.shape[1] == 48

    def test_geotoken_graph_config_default(self):
        """GraphTokenizationConfig defaults to node_feature_dim=48."""
        config = GraphTokenizationConfig()
        assert config.node_feature_dim == 48

    def test_stepnet_graph_encoder_default(self):
        """STEPGraphEncoder defaults to input_dim=48."""
        enc = STEPGraphEncoder()
        assert enc.input_dim == 48

    def test_stepnet_encoder_default(self):
        """STEPEncoder defaults to graph_input_dim=48."""
        enc = STEPEncoder()
        assert enc.graph_encoder.input_dim == 48

    def test_stepnet_compact_features_default(self):
        """build_compact_node_features() defaults to feature_dim=48."""
        builder = STEPTopologyBuilder()
        feats = builder.build_compact_node_features([
            {'entity_id': 1, 'entity_type': 'PLANE',
             'references': [], 'numeric_params': [1.0]},
        ])
        assert feats.shape == (1, 48)


# ===========================================================================
# 10. BACKWARD COMPATIBILITY
# ===========================================================================

class TestBackwardCompatibility:
    """Verify that older (non-cadling) data paths still work."""

    def test_stepnet_encoder_accepts_129dim(self):
        """STEPGraphEncoder accepts 129-dim when explicitly configured."""
        enc = STEPGraphEncoder(input_dim=129)
        nodes = torch.randn(3, 129)
        adj = torch.eye(3)
        out = enc(nodes, adj)
        assert out.shape == (3, 128)  # default node_dim

    def test_step_encoder_auto_projects_mismatched_dims(self):
        """STEPEncoder auto-projects when topology feature dim != graph_input_dim."""
        enc = STEPEncoder(
            vocab_size=100, token_embed_dim=32,
            graph_node_dim=16, graph_input_dim=48, output_dim=64,
        )
        token_ids = torch.randint(0, 100, (1, 5))
        # Pass 129-dim features with a 48-dim configured encoder → auto-project
        topo = {
            'adjacency_matrix': torch.eye(3),
            'node_features': torch.randn(3, 129),
        }
        out = enc(token_ids, topo)
        assert out.shape == (1, 64)

    def test_old_cadling_format_auto_detection(self):
        """CommandSequenceTokenizer auto-detects old cadling z-interleaved format."""
        old_cadling_commands = [
            {"type": "SOL", "params": [0.0, 0.0] + [0.0] * 14},
            {"type": "LINE", "params": [10.0, 20.0, 0.0, 30.0, 40.0, 0.0] + [0.0] * 10},
            {"type": "EOS", "params": [0.0] * 16},
        ]
        tokenizer = CommandSequenceTokenizer(
            command_config=CommandTokenizationConfig(
                source_format="auto",
                canonicalize_loops=False,
            ),
        )
        seq = tokenizer.tokenize(old_cadling_commands)
        # Should still produce valid tokens
        assert seq.command_tokens[0].command_type == GeoCommandType.SOL
        assert seq.command_tokens[1].command_type == GeoCommandType.LINE
        assert seq.command_tokens[2].command_type == GeoCommandType.EOS
