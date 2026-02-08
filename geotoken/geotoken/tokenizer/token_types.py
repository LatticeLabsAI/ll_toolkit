"""Token types for geometric data representation."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Enums for CAD command vocabulary
# ---------------------------------------------------------------------------

class CommandType(str, Enum):
    """CAD command types following DeepCAD's 6 command vocabulary."""
    SOL = "SOL"          # Start of loop
    LINE = "LINE"        # Line primitive
    ARC = "ARC"          # Arc primitive
    CIRCLE = "CIRCLE"    # Circle primitive
    EXTRUDE = "EXTRUDE"  # Extrusion operation
    EOS = "EOS"          # End of sequence


class ConstraintType(str, Enum):
    """Geometric constraint types following SketchGraphs."""
    COINCIDENT = "COINCIDENT"
    TANGENT = "TANGENT"
    PERPENDICULAR = "PERPENDICULAR"
    PARALLEL = "PARALLEL"
    CONCENTRIC = "CONCENTRIC"
    EQUAL_LENGTH = "EQUAL_LENGTH"
    EQUAL_RADIUS = "EQUAL_RADIUS"
    DISTANCE = "DISTANCE"
    ANGLE = "ANGLE"


class BooleanOpType(str, Enum):
    """Boolean/CSG operation types."""
    UNION = "UNION"
    INTERSECTION = "INTERSECTION"
    SUBTRACTION = "SUBTRACTION"


# ---------------------------------------------------------------------------
# Existing token types (unchanged)
# ---------------------------------------------------------------------------

@dataclass
class CoordinateToken:
    """A quantized 3D coordinate token."""
    x: int
    y: int
    z: int
    bits: int = 8              # Bit width used for quantization
    vertex_index: int = -1     # Original vertex index

    def to_tuple(self) -> tuple[int, int, int]:
        return (self.x, self.y, self.z)

    @property
    def levels(self) -> int:
        return 2 ** self.bits


@dataclass
class GeometryToken:
    """A token representing geometric structure (face, edge, etc.)."""
    token_type: str            # "face", "edge", "vertex", "separator"
    indices: list[int] = field(default_factory=list)
    properties: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# New command-level token types (Phase 1.1)
# ---------------------------------------------------------------------------

@dataclass
class CommandToken:
    """A single CAD operation in a sketch-and-extrude sequence.

    Represents one command in the DeepCAD-style construction history.
    Each command has a type, up to 16 quantized integer parameters,
    and a mask indicating which parameters are active.

    Args:
        command_type: The CAD operation type (SOL, LINE, ARC, etc.).
        parameters: Up to 16 quantized integer parameter values.
        parameter_mask: Boolean mask for active parameters per command type.
    """
    command_type: CommandType
    parameters: list[int] = field(default_factory=lambda: [0] * 16)
    parameter_mask: list[bool] = field(default_factory=lambda: [False] * 16)

    def active_parameters(self) -> list[int]:
        """Return only the active parameters for this command."""
        return [p for p, m in zip(self.parameters, self.parameter_mask) if m]

    @staticmethod
    def get_parameter_mask(command_type: CommandType) -> list[bool]:
        """Get the canonical parameter mask for a command type.

        Parameter semantics per command type:
        - SOL: 2 active parameters
            - params[0]: sketch plane z-offset (height from origin)
            - params[1]: sketch plane rotation/normal orientation
        - LINE: xy endpoints (4 params: x1, y1, x2, y2)
        - ARC: start/mid/end points (6 params: x1, y1, x2, y2, x3, y3)
        - CIRCLE: center + radius (3 params: cx, cy, r)
        - EXTRUDE: extent/scale/boolean params (up to 8 params)
        - EOS: no active parameters
        """
        masks = {
            CommandType.SOL: [True, True, False, False, False, False,
                              False, False, False, False, False, False,
                              False, False, False, False],
            CommandType.LINE: [True, True, True, True, False, False,
                               False, False, False, False, False, False,
                               False, False, False, False],
            CommandType.ARC: [True, True, True, True, True, True,
                              False, False, False, False, False, False,
                              False, False, False, False],
            CommandType.CIRCLE: [True, True, True, False, False, False,
                                 False, False, False, False, False, False,
                                 False, False, False, False],
            CommandType.EXTRUDE: [True, True, True, True, True, True,
                                  True, True, False, False, False, False,
                                  False, False, False, False],
            CommandType.EOS: [False] * 16,
        }
        return masks.get(command_type, [False] * 16)


@dataclass
class ConstraintToken:
    """A geometric constraint between sketch primitives.

    Maps to SketchGraphs' constraint edges — encodes designer-imposed
    relationships like parallelism, tangency, and coincidence.

    Args:
        constraint_type: Type of geometric constraint.
        source_index: Index of first primitive in the command sequence.
        target_index: Index of second primitive.
        value: Optional quantized constraint value (for distance/angle).
    """
    constraint_type: ConstraintType
    source_index: int
    target_index: int
    value: Optional[int] = None


@dataclass
class BooleanOpToken:
    """A CSG boolean operation combining solid bodies.

    Args:
        op_type: The boolean operation (union, intersection, subtraction).
        operand_indices: Indices of bodies being combined.
    """
    op_type: BooleanOpType
    operand_indices: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Graph token types (Phase 4 — B-Rep topology graph tokenization)
# ---------------------------------------------------------------------------

@dataclass
class GraphNodeToken:
    """Quantized node (face/edge/vertex) feature token.

    Represents a single node in a B-Rep topology graph with its
    feature vector quantized to discrete token values. Each dimension
    of the feature vector becomes a separate quantized integer.

    Args:
        node_index: Index of this node in the graph.
        feature_tokens: Quantized feature values (one per dimension).
        node_type: Node entity type ("face", "edge", "vertex").
        bits: Quantization bit width per feature dimension.
    """
    node_index: int
    feature_tokens: list[int] = field(default_factory=list)
    node_type: str = "face"
    bits: int = 8

    @property
    def num_features(self) -> int:
        """Number of feature dimensions."""
        return len(self.feature_tokens)


@dataclass
class GraphEdgeToken:
    """Quantized edge feature token.

    Represents a directed edge in the B-Rep topology graph with its
    feature vector quantized to discrete token values.

    Args:
        source_index: Source node index.
        target_index: Target node index.
        feature_tokens: Quantized feature values (one per dimension).
        bits: Quantization bit width per feature dimension.
    """
    source_index: int
    target_index: int
    feature_tokens: list[int] = field(default_factory=list)
    bits: int = 8

    @property
    def num_features(self) -> int:
        """Number of feature dimensions."""
        return len(self.feature_tokens)


@dataclass
class GraphStructureToken:
    """Graph structure marker token.

    Used to delimit graph, node, and edge boundaries in the
    serialized flat token sequence.

    Args:
        token_type: Structure marker type. One of:
            "graph_start" - Start of a graph (value = num_nodes)
            "graph_end"   - End of a graph
            "node_start"  - Start of a node (value = node_index)
            "node_end"    - End of a node
            "adjacency"   - Adjacency list entry (value = neighbor_index)
            "edge"        - Edge marker (value encodes src << 16 | tgt)
        value: Associated integer value.
    """
    token_type: str
    value: int = 0


@dataclass
class SequenceConfig:
    """Metadata and configuration for a command sequence.

    Controls fixed-length padding, quantization resolution, and
    normalization range for transformer consumption.

    Args:
        max_commands: Target sequence length (DeepCAD uses 60).
        quantization_bits: Bits for parameter quantization.
        coordinate_range: Normalization bounding cube size.
        padding_token_id: Token ID used for padding.
    """
    max_commands: int = 60
    quantization_bits: int = 8
    coordinate_range: float = 2.0
    padding_token_id: int = 0


# ---------------------------------------------------------------------------
# Token sequence container (updated with command_tokens field)
# ---------------------------------------------------------------------------

@dataclass
class TokenSequence:
    """A sequence of geometric tokens.

    Contains coordinate tokens (mesh-level), geometry tokens (structural),
    command tokens (parametric CAD construction history), constraint tokens,
    and graph tokens (B-Rep topology).
    """
    coordinate_tokens: list[CoordinateToken] = field(default_factory=list)
    geometry_tokens: list[GeometryToken] = field(default_factory=list)
    command_tokens: list[CommandToken] = field(default_factory=list)
    constraint_tokens: list[ConstraintToken] = field(default_factory=list)
    boolean_op_tokens: list[BooleanOpToken] = field(default_factory=list)
    graph_node_tokens: list[GraphNodeToken] = field(default_factory=list)
    graph_edge_tokens: list[GraphEdgeToken] = field(default_factory=list)
    graph_structure_tokens: list[GraphStructureToken] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return (len(self.coordinate_tokens) + len(self.geometry_tokens)
                + len(self.command_tokens) + len(self.graph_node_tokens)
                + len(self.graph_edge_tokens) + len(self.graph_structure_tokens))

    def to_array(self) -> np.ndarray:
        """Convert coordinate tokens to (N, 3) integer array."""
        if not self.coordinate_tokens:
            return np.array([], dtype=int).reshape(0, 3)
        return np.array([t.to_tuple() for t in self.coordinate_tokens], dtype=int)

    @property
    def bits_per_vertex(self) -> np.ndarray:
        """Get bit width per vertex."""
        if not self.coordinate_tokens:
            return np.array([], dtype=int)
        return np.array([t.bits for t in self.coordinate_tokens], dtype=int)

    @property
    def num_commands(self) -> int:
        """Number of command tokens in the sequence."""
        return len(self.command_tokens)

    def command_types(self) -> list[CommandType]:
        """Get the sequence of command types."""
        return [ct.command_type for ct in self.command_tokens]
