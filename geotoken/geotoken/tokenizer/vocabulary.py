"""CAD command vocabulary for transformer token ID mapping.

Builds and manages the discrete token vocabulary that maps
(command_type, parameter_index, quantized_value) triples to unique
integer token IDs for transformer consumption.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .token_types import (
    CommandToken,
    CommandType,
    ConstraintToken,
    ConstraintType,
    GraphEdgeToken,
    GraphNodeToken,
    GraphStructureToken,
)

_log = logging.getLogger(__name__)

# Special token IDs
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
SEP_TOKEN_ID = 3
UNK_TOKEN_ID = 4
NUM_SPECIAL_TOKENS = 5


class CADVocabulary:
    """Discrete token vocabulary for CAD command sequences.

    Maps every possible (command_type, parameter_index, quantized_value)
    triple to a unique integer token ID. With 6 command types, 16 parameters,
    and 256 quantization levels this yields ~24,576 tokens plus specials.

    Args:
        num_command_types: Number of command type categories.
        num_parameters: Maximum number of parameters per command.
        num_levels: Number of quantization levels per parameter.
    """

    def __init__(
        self,
        num_command_types: int = 6,
        num_parameters: int = 16,
        num_levels: int = 256,
        num_constraint_types: int = 9,
        max_constraint_index: int = 60,
        num_graph_structure_types: int = 6,
        node_feature_dim: int = 48,
        edge_feature_dim: int = 16,
        graph_feature_levels: int = 256,
    ) -> None:
        self.num_command_types = num_command_types
        self.num_parameters = num_parameters
        self.num_levels = num_levels

        # --- Block 1: Command tokens ---
        # Command type token IDs start after special tokens
        self._cmd_type_offset = NUM_SPECIAL_TOKENS
        # Parameter token IDs start after command type tokens
        self._param_offset = self._cmd_type_offset + num_command_types

        cmd_block_size = (
            num_command_types
            + num_command_types * num_parameters * num_levels
        )

        # --- Block 2: Constraint tokens ---
        self.num_constraint_types = num_constraint_types
        self.max_constraint_index = max_constraint_index
        self._constraint_offset = NUM_SPECIAL_TOKENS + cmd_block_size

        # Constraint encoding: type * max_idx * max_idx + source * max_idx + target
        constraint_block_size = (
            num_constraint_types * max_constraint_index * max_constraint_index
        )

        # Constraint type name → index mapping
        self._constraint_type_to_idx = {
            ct.value: i for i, ct in enumerate(ConstraintType)
        }
        self._idx_to_constraint_type = {
            i: ct for i, ct in enumerate(ConstraintType)
        }

        # --- Block 3: Graph tokens ---
        self.num_graph_structure_types = num_graph_structure_types
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.graph_feature_levels = graph_feature_levels

        self._graph_structure_offset = (
            self._constraint_offset + constraint_block_size
        )
        self._graph_structure_types = {
            "graph_start": 0, "graph_end": 1,
            "node_start": 2, "node_end": 3,
            "adjacency": 4, "edge": 5,
        }

        # Node feature tokens: dim * levels
        self._graph_node_offset = (
            self._graph_structure_offset + num_graph_structure_types
        )
        node_feature_block = node_feature_dim * graph_feature_levels

        # Edge feature tokens: dim * levels
        self._graph_edge_offset = self._graph_node_offset + node_feature_block
        edge_feature_block = edge_feature_dim * graph_feature_levels

        # --- Total ---
        self._total_size = (
            NUM_SPECIAL_TOKENS
            + cmd_block_size
            + constraint_block_size
            + num_graph_structure_types
            + node_feature_block
            + edge_feature_block
        )

        # Command type name → index mapping
        self._type_to_idx = {ct.value: i for i, ct in enumerate(CommandType)}
        self._idx_to_type = {i: ct for i, ct in enumerate(CommandType)}

        _log.debug(
            "CADVocabulary: %d command types, %d params, %d levels, "
            "%d constraint types, graph features (%d node dims, %d edge dims) → %d tokens",
            num_command_types, num_parameters, num_levels,
            num_constraint_types, node_feature_dim, edge_feature_dim,
            self._total_size,
        )

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return self._total_size

    @property
    def pad_token_id(self) -> int:
        return PAD_TOKEN_ID

    @property
    def bos_token_id(self) -> int:
        return BOS_TOKEN_ID

    @property
    def eos_token_id(self) -> int:
        return EOS_TOKEN_ID

    @property
    def sep_token_id(self) -> int:
        return SEP_TOKEN_ID

    @property
    def unk_token_id(self) -> int:
        return UNK_TOKEN_ID

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, command_tokens: list[CommandToken]) -> list[int]:
        """Convert command token sequence to integer ID sequence.

        Each CommandToken produces 1 (command type) + N (active parameter)
        token IDs. Sequence is wrapped with BOS and EOS.

        Args:
            command_tokens: List of CommandToken objects.

        Returns:
            List of integer token IDs.

        Raises:
            TypeError: If command_tokens is not a list or contains non-CommandToken items.
        """
        if not isinstance(command_tokens, list):
            raise TypeError(
                f"command_tokens must be a list, got {type(command_tokens).__name__}"
            )
        for i, token in enumerate(command_tokens):
            if not isinstance(token, CommandToken):
                raise TypeError(
                    f"command_tokens[{i}] must be CommandToken, got {type(token).__name__}"
                )

        ids: list[int] = [BOS_TOKEN_ID]

        for token in command_tokens:
            if token.command_type == CommandType.EOS:
                break

            # Encode command type
            type_idx = self._type_to_idx.get(token.command_type.value, 0)
            ids.append(self._cmd_type_offset + type_idx)

            # Encode active parameters
            for param_idx, (val, active) in enumerate(
                zip(token.parameters, token.parameter_mask)
            ):
                if active:
                    param_id = self._param_token_id(type_idx, param_idx, val)
                    ids.append(param_id)

        ids.append(EOS_TOKEN_ID)
        return ids

    def decode(self, token_ids: list[int]) -> list[CommandToken]:
        """Convert integer ID sequence back to CommandToken list.

        Args:
            token_ids: List of integer token IDs.

        Returns:
            List of CommandToken objects.
        """
        commands: list[CommandToken] = []
        i = 0

        # Skip BOS
        if token_ids and token_ids[0] == BOS_TOKEN_ID:
            i = 1

        while i < len(token_ids):
            tid = token_ids[i]

            # Stop at EOS or PAD
            if tid in (EOS_TOKEN_ID, PAD_TOKEN_ID):
                break

            # Skip other specials
            if tid < NUM_SPECIAL_TOKENS:
                i += 1
                continue

            # Decode command type
            if self._cmd_type_offset <= tid < self._param_offset:
                type_idx = tid - self._cmd_type_offset
                cmd_type = self._idx_to_type.get(type_idx, CommandType.EOS)
                mask = CommandToken.get_parameter_mask(cmd_type)
                params = [0] * 16

                # Read subsequent parameter tokens
                i += 1
                for param_idx, active in enumerate(mask):
                    if not active:
                        continue
                    if i >= len(token_ids):
                        break
                    ptid = token_ids[i]
                    if ptid < self._param_offset:
                        break  # Not a parameter token
                    val = self._decode_param_token(ptid, type_idx, param_idx)
                    if val is not None:
                        params[param_idx] = val
                        i += 1
                    else:
                        # Misaligned token — skip it and log warning
                        _log.warning(
                            "Skipping misaligned token at index %d (id=%d) "
                            "while decoding param_idx=%d for type_idx=%d",
                            i, ptid, param_idx, type_idx,
                        )
                        i += 1
                        break

                commands.append(CommandToken(
                    command_type=cmd_type,
                    parameters=params,
                    parameter_mask=mask,
                ))
            else:
                # Unknown token, skip
                i += 1

        return commands

    def encode_flat(self, command_tokens: list[CommandToken]) -> list[int]:
        """Flat encoding: fixed-width per command.

        Each command emits exactly 1 (command type ID) + num_active_params
        parameter IDs, padded or truncated to a fixed width determined by
        the command type's canonical mask.

        Args:
            command_tokens: List of CommandToken objects.

        Returns:
            List of integer token IDs (BOS + fixed-width commands + EOS).
        """
        ids: list[int] = [BOS_TOKEN_ID]

        for token in command_tokens:
            if token.command_type == CommandType.EOS:
                break

            type_idx = self._type_to_idx.get(token.command_type.value, 0)
            ids.append(self._cmd_type_offset + type_idx)

            # Canonical mask determines fixed width
            canonical_mask = CommandToken.get_parameter_mask(token.command_type)
            num_active = sum(canonical_mask)

            # Collect active parameter IDs
            param_ids: list[int] = []
            for param_idx, active in enumerate(canonical_mask):
                if active:
                    val = token.parameters[param_idx] if param_idx < len(token.parameters) else 0
                    param_ids.append(self._param_token_id(type_idx, param_idx, val))

            # Pad or truncate to exactly num_active
            if len(param_ids) < num_active:
                for extra_idx in range(len(param_ids), num_active):
                    param_ids.append(self._param_token_id(type_idx, extra_idx, 0))
            param_ids = param_ids[:num_active]

            ids.extend(param_ids)

        ids.append(EOS_TOKEN_ID)
        return ids

    # ------------------------------------------------------------------
    # Constraint encoding / decoding
    # ------------------------------------------------------------------

    def encode_constraints(
        self, constraint_tokens: list[ConstraintToken]
    ) -> list[int]:
        """Encode constraint tokens to integer IDs.

        Each constraint is encoded as a single integer:
            constraint_offset + type_idx * max_idx^2 + source * max_idx + target

        Args:
            constraint_tokens: List of ConstraintToken objects.

        Returns:
            List of integer token IDs for constraints.
        """
        ids: list[int] = []
        max_idx = self.max_constraint_index

        for ct in constraint_tokens:
            type_idx = self._constraint_type_to_idx.get(
                ct.constraint_type.value, 0
            )
            src = min(ct.source_index, max_idx - 1)
            tgt = min(ct.target_index, max_idx - 1)

            token_id = (
                self._constraint_offset
                + type_idx * max_idx * max_idx
                + src * max_idx
                + tgt
            )
            ids.append(token_id)

        return ids

    def decode_constraints(
        self, token_ids: list[int]
    ) -> list[ConstraintToken]:
        """Decode integer IDs back to constraint tokens.

        Args:
            token_ids: List of constraint token IDs.

        Returns:
            List of ConstraintToken objects.
        """
        constraints: list[ConstraintToken] = []
        max_idx = self.max_constraint_index

        for tid in token_ids:
            if tid < self._constraint_offset:
                continue
            offset = tid - self._constraint_offset
            max_constraint_block = (
                self.num_constraint_types * max_idx * max_idx
            )
            if offset >= max_constraint_block:
                continue

            type_idx = offset // (max_idx * max_idx)
            remainder = offset % (max_idx * max_idx)
            src = remainder // max_idx
            tgt = remainder % max_idx

            ctype = self._idx_to_constraint_type.get(
                type_idx, ConstraintType.COINCIDENT
            )
            constraints.append(ConstraintToken(
                constraint_type=ctype,
                source_index=src,
                target_index=tgt,
            ))

        return constraints

    # ------------------------------------------------------------------
    # Graph token encoding / decoding
    # ------------------------------------------------------------------

    def encode_graph_structure(
        self, structure_tokens: list[GraphStructureToken]
    ) -> list[int]:
        """Encode graph structure tokens to integer IDs.

        Args:
            structure_tokens: List of GraphStructureToken objects.

        Returns:
            List of integer token IDs.
        """
        ids: list[int] = []
        for st in structure_tokens:
            type_idx = self._graph_structure_types.get(st.token_type, 0)
            ids.append(self._graph_structure_offset + type_idx)
        return ids

    def encode_graph_node_features(
        self, node_tokens: list[GraphNodeToken]
    ) -> list[int]:
        """Encode quantized node feature tokens to integer IDs.

        Each feature dimension of each node becomes a separate token:
            graph_node_offset + dim_idx * levels + quantized_value

        Args:
            node_tokens: List of GraphNodeToken objects.

        Returns:
            List of integer token IDs (one per feature per node).
        """
        ids: list[int] = []
        for nt in node_tokens:
            for dim_idx, val in enumerate(nt.feature_tokens):
                if dim_idx >= self.node_feature_dim:
                    break
                val = max(0, min(self.graph_feature_levels - 1, val))
                token_id = (
                    self._graph_node_offset
                    + dim_idx * self.graph_feature_levels
                    + val
                )
                ids.append(token_id)
        return ids

    def encode_graph_edge_features(
        self, edge_tokens: list[GraphEdgeToken]
    ) -> list[int]:
        """Encode quantized edge feature tokens to integer IDs.

        Each feature dimension of each edge becomes a separate token:
            graph_edge_offset + dim_idx * levels + quantized_value

        Args:
            edge_tokens: List of GraphEdgeToken objects.

        Returns:
            List of integer token IDs (one per feature per edge).
        """
        ids: list[int] = []
        for et in edge_tokens:
            for dim_idx, val in enumerate(et.feature_tokens):
                if dim_idx >= self.edge_feature_dim:
                    break
                val = max(0, min(self.graph_feature_levels - 1, val))
                token_id = (
                    self._graph_edge_offset
                    + dim_idx * self.graph_feature_levels
                    + val
                )
                ids.append(token_id)
        return ids

    def encode_full_sequence(self, token_sequence) -> list[int]:
        """Encode a complete TokenSequence to integer IDs.

        Combines command tokens, constraint tokens, and graph tokens
        into a single flat integer sequence with separators.

        Args:
            token_sequence: A TokenSequence with any combination of
                command, constraint, and graph tokens.

        Returns:
            List of integer token IDs.
        """
        ids: list[int] = [BOS_TOKEN_ID]

        # Commands
        if token_sequence.command_tokens:
            ids.extend(self.encode(token_sequence.command_tokens)[1:-1])  # Strip BOS/EOS

        # Constraints
        if token_sequence.constraint_tokens:
            ids.append(SEP_TOKEN_ID)
            ids.extend(self.encode_constraints(token_sequence.constraint_tokens))

        # Graph tokens
        if token_sequence.graph_structure_tokens:
            ids.append(SEP_TOKEN_ID)
            ids.extend(self.encode_graph_structure(token_sequence.graph_structure_tokens))
            ids.extend(self.encode_graph_node_features(token_sequence.graph_node_tokens))
            ids.extend(self.encode_graph_edge_features(token_sequence.graph_edge_tokens))

        ids.append(EOS_TOKEN_ID)
        return ids

    # ------------------------------------------------------------------
    # Token ID computation
    # ------------------------------------------------------------------

    def _param_token_id(
        self, type_idx: int, param_idx: int, value: int
    ) -> int:
        """Compute unique token ID for a parameter value."""
        value = max(0, min(self.num_levels - 1, value))
        return (
            self._param_offset
            + type_idx * self.num_parameters * self.num_levels
            + param_idx * self.num_levels
            + value
        )

    def _decode_param_token(
        self, token_id: int, expected_type: int, expected_param: int
    ) -> Optional[int]:
        """Decode a parameter token ID back to its quantized value."""
        if token_id < self._param_offset:
            return None

        offset = token_id - self._param_offset
        type_idx = offset // (self.num_parameters * self.num_levels)
        remainder = offset % (self.num_parameters * self.num_levels)
        param_idx = remainder // self.num_levels
        value = remainder % self.num_levels

        if type_idx == expected_type and param_idx == expected_param:
            return value
        _log.warning(
            "Decode misalignment: token %d decoded to type_idx=%d param_idx=%d, "
            "expected type_idx=%d param_idx=%d",
            token_id, type_idx, param_idx, expected_type, expected_param,
        )
        return None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save vocabulary configuration to JSON."""
        path = Path(path)
        data = {
            "num_command_types": self.num_command_types,
            "num_parameters": self.num_parameters,
            "num_levels": self.num_levels,
            "num_constraint_types": self.num_constraint_types,
            "max_constraint_index": self.max_constraint_index,
            "num_graph_structure_types": self.num_graph_structure_types,
            "node_feature_dim": self.node_feature_dim,
            "edge_feature_dim": self.edge_feature_dim,
            "graph_feature_levels": self.graph_feature_levels,
            "vocab_size": self.vocab_size,
            "special_tokens": {
                "PAD": PAD_TOKEN_ID,
                "BOS": BOS_TOKEN_ID,
                "EOS": EOS_TOKEN_ID,
                "SEP": SEP_TOKEN_ID,
                "UNK": UNK_TOKEN_ID,
            },
        }
        path.write_text(json.dumps(data, indent=2))
        _log.info("Saved vocabulary to %s (size=%d)", path, self.vocab_size)

    def encode_to_ids(
        self,
        command_tokens: list[CommandToken],
        seq_len: int = 60,
        pad_id: Optional[int] = None,
    ) -> list[int]:
        """Encode command tokens to a fixed-length integer ID sequence.

        This bridges the variable-length output of :meth:`encode` to the
        fixed-length sequences that transformer models consume.  The sequence
        is padded (with ``pad_id``) or truncated to exactly ``seq_len``
        integer token IDs.

        Args:
            command_tokens: List of CommandToken objects.
            seq_len: Target sequence length (default 60 per DeepCAD).
            pad_id: Padding token ID.  Defaults to :attr:`pad_token_id`.

        Returns:
            List of exactly ``seq_len`` integer token IDs.
        """
        if pad_id is None:
            pad_id = self.pad_token_id

        ids = self.encode(command_tokens)

        if len(ids) > seq_len:
            # Truncate but ensure the sequence ends with EOS
            ids = ids[: seq_len - 1] + [self.eos_token_id]
        elif len(ids) < seq_len:
            ids = ids + [pad_id] * (seq_len - len(ids))

        return ids

    def encode_to_tensor(
        self,
        command_tokens: list[CommandToken],
        seq_len: int = 60,
        pad_id: Optional[int] = None,
    ):
        """Encode command tokens to a fixed-length torch.Tensor.

        Wrapper around :meth:`encode_to_ids` that returns an actual
        ``torch.Tensor`` instead of a plain list.

        Args:
            command_tokens: List of CommandToken objects.
            seq_len: Target sequence length (default 60 per DeepCAD).
            pad_id: Padding token ID.  Defaults to :attr:`pad_token_id`.

        Returns:
            torch.Tensor of shape ``[seq_len]`` with dtype ``torch.long``.
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for encode_to_tensor. "
                "Install with: conda install pytorch -c conda-forge"
            )

        ids = self.encode_to_ids(command_tokens, seq_len=seq_len, pad_id=pad_id)
        return torch.tensor(ids, dtype=torch.long)

    @classmethod
    def load(cls, path: str | Path) -> CADVocabulary:
        """Load vocabulary configuration from JSON."""
        path = Path(path)
        data = json.loads(path.read_text())
        vocab = cls(
            num_command_types=data["num_command_types"],
            num_parameters=data["num_parameters"],
            num_levels=data["num_levels"],
            num_constraint_types=data.get("num_constraint_types", 9),
            max_constraint_index=data.get("max_constraint_index", 60),
            num_graph_structure_types=data.get("num_graph_structure_types", 6),
            node_feature_dim=data.get("node_feature_dim", 48),
            edge_feature_dim=data.get("edge_feature_dim", 16),
            graph_feature_levels=data.get("graph_feature_levels", 256),
        )
        _log.info("Loaded vocabulary from %s (size=%d)", path, vocab.vocab_size)
        return vocab


def encode_to_tensor(
    command_tokens: list[CommandToken],
    vocab: Optional[CADVocabulary] = None,
    seq_len: int = 60,
    pad_id: Optional[int] = None,
):
    """Encode command tokens to a fixed-length torch.Tensor for transformer input.

    Convenience function that bridges the variable-length output of
    CADVocabulary.encode() to a fixed-length torch.Tensor that transformer
    models consume. The sequence is padded (with pad_id) or truncated to
    exactly seq_len.

    Args:
        command_tokens: List of CommandToken objects.
        vocab: CADVocabulary instance. If None, creates a default vocabulary.
        seq_len: Target sequence length (default 60 per DeepCAD).
        pad_id: Padding token ID. Defaults to vocab.pad_token_id.

    Returns:
        torch.Tensor of shape [seq_len] with dtype torch.long.

    Example:
        >>> from geotoken.tokenizer import encode_to_tensor, CommandToken, CommandType
        >>> tokens = [CommandToken(CommandType.SOL, [10, 20] + [0]*14, [True, True] + [False]*14)]
        >>> tensor = encode_to_tensor(tokens, seq_len=60)
        >>> tensor.shape
        torch.Size([60])
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for encode_to_tensor. "
            "Install with: conda install pytorch -c conda-forge"
        )

    if vocab is None:
        vocab = CADVocabulary()

    ids = vocab.encode_to_ids(command_tokens, seq_len=seq_len, pad_id=pad_id)
    return torch.tensor(ids, dtype=torch.long)


def batch_encode_to_tensor(
    batch_command_tokens: list[list[CommandToken]],
    vocab: Optional[CADVocabulary] = None,
    seq_len: int = 60,
    pad_id: Optional[int] = None,
):
    """Encode a batch of command token sequences to a stacked torch.Tensor.

    Args:
        batch_command_tokens: List of command token sequences.
        vocab: CADVocabulary instance. If None, creates a default vocabulary.
        seq_len: Target sequence length per sample.
        pad_id: Padding token ID. Defaults to vocab.pad_token_id.

    Returns:
        torch.Tensor of shape [batch_size, seq_len] with dtype torch.long.

    Example:
        >>> from geotoken.tokenizer import batch_encode_to_tensor, CommandToken, CommandType
        >>> batch = [[CommandToken(CommandType.SOL, [10, 20] + [0]*14, [True, True] + [False]*14)]]
        >>> tensor = batch_encode_to_tensor(batch, seq_len=60)
        >>> tensor.shape
        torch.Size([1, 60])
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for batch_encode_to_tensor. "
            "Install with: conda install pytorch -c conda-forge"
        )

    if vocab is None:
        vocab = CADVocabulary()

    batch_ids = [
        vocab.encode_to_ids(tokens, seq_len=seq_len, pad_id=pad_id)
        for tokens in batch_command_tokens
    ]
    return torch.tensor(batch_ids, dtype=torch.long)
