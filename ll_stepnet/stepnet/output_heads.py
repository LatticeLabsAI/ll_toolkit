"""
Decoder output heads for CAD command generation.

Following DeepCAD's separate-head design, each CAD command position is
decoded into:
    1. A command type (SOL, LINE, ARC, CIRCLE, EXTRUDE, EOS).
    2. Up to 16 parameter slots, each quantised into num_levels bins.

A CompositeHead combines both and applies command-type masking so that
only the parameters relevant to a given command type contribute to the loss.

Command types and parameter masks match geotoken's CommandType and
CommandToken.get_parameter_mask() exactly, so ll_stepnet and geotoken
share a single canonical command representation with no conversion needed.
"""

from __future__ import annotations

import logging
from enum import IntEnum
from typing import Dict, List, Optional

import torch
import torch.nn as nn

_log = logging.getLogger(__name__)


class CommandType(IntEnum):
    """CAD command types matching geotoken's 6 command vocabulary.

    Ordering matches geotoken.CommandType enum ordering so that
    integer indices are directly interchangeable between the two modules.
    """

    SOL = 0        # Start of loop (sketch plane z-offset, rotation)
    LINE = 1       # Line primitive (x1, y1, x2, y2)
    ARC = 2        # Arc primitive, 3-point format (x1, y1, x2, y2, x3, y3)
    CIRCLE = 3     # Circle primitive (cx, cy, r)
    EXTRUDE = 4    # Extrusion operation (8 params)
    EOS = 5        # End of sequence


# Mapping from command type to the indices of the 16 parameter slots that
# are active for that command.  These masks are identical to
# geotoken.CommandToken.get_parameter_mask() so data flows directly
# between cadling → geotoken → ll_stepnet with no conversion.
PARAMETER_MASKS: Dict[CommandType, List[int]] = {
    # SOL: sketch plane z-offset, rotation
    CommandType.SOL: [0, 1],
    # LINE: x1, y1, x2, y2
    CommandType.LINE: [0, 1, 2, 3],
    # ARC: 3-point format — start(x1,y1), mid(x2,y2), end(x3,y3)
    CommandType.ARC: [0, 1, 2, 3, 4, 5],
    # CIRCLE: center_x, center_y, radius
    CommandType.CIRCLE: [0, 1, 2],
    # EXTRUDE: extent, scale, boolean params (8 active)
    CommandType.EXTRUDE: [0, 1, 2, 3, 4, 5, 6, 7],
    # EOS: no parameters
    CommandType.EOS: [],
}


class CommandTypeHead(nn.Module):
    """Predicts the CAD command type at each sequence position.

    Args:
        embed_dim: Dimension of the decoder hidden states.
        num_command_types: Number of distinct command types.
    """

    def __init__(self, embed_dim: int = 256, num_command_types: int = 6) -> None:
        super().__init__()
        self.projection = nn.Linear(embed_dim, num_command_types)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute command type logits.

        Args:
            hidden_states: [batch, seq_len, embed_dim]

        Returns:
            Logits [batch, seq_len, num_command_types].
        """
        return self.projection(hidden_states)


class ParameterHeads(nn.Module):
    """Sixteen independent linear heads, one per parameter slot.

    Each head maps the decoder hidden state to num_levels logits
    representing the quantised bin for that parameter.

    Args:
        embed_dim: Dimension of the decoder hidden states.
        num_param_slots: Number of parameter slots (default 16).
        num_levels: Number of quantisation levels per parameter.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_param_slots: int = 16,
        num_levels: int = 256,
    ) -> None:
        super().__init__()
        self.num_param_slots = num_param_slots
        self.num_levels = num_levels
        self.heads = nn.ModuleList(
            [nn.Linear(embed_dim, num_levels) for _ in range(num_param_slots)]
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> List[torch.Tensor]:
        """Compute per-slot parameter logits.

        Args:
            hidden_states: [batch, seq_len, embed_dim]

        Returns:
            List of 16 tensors, each [batch, seq_len, num_levels].
        """
        return [head(hidden_states) for head in self.heads]


class CompositeHead(nn.Module):
    """Combined command-type and parameter prediction head with masking.

    During training this head:
        1. Predicts command type logits.
        2. Predicts 16 parameter logits.
        3. Applies PARAMETER_MASKS to zero-out gradients for
           parameters that do not belong to the predicted (or target)
           command type.
        4. Optionally predicts vertex positions via
           :class:`VertexPredictionHead` (when ``include_vertex_head=True``).

    Args:
        embed_dim: Dimension of the decoder hidden states.
        num_command_types: Number of distinct command types.
        num_param_slots: Number of parameter slots.
        num_levels: Number of quantisation levels per parameter.
        include_vertex_head: Whether to include the
            :class:`VertexPredictionHead` for direct 3D vertex prediction.
        max_vertices: Maximum number of vertex slots (only used when
            ``include_vertex_head=True``).
        num_refinement_steps: Number of learned vertex refinement
            iterations (only used when ``include_vertex_head=True``).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_command_types: int = 6,
        num_param_slots: int = 16,
        num_levels: int = 256,
        include_vertex_head: bool = False,
        max_vertices: int = 512,
        num_refinement_steps: int = 3,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_command_types = num_command_types
        self.num_param_slots = num_param_slots
        self.num_levels = num_levels

        self.command_head = CommandTypeHead(embed_dim, num_command_types)
        self.parameter_heads = ParameterHeads(embed_dim, num_param_slots, num_levels)

        # Optional vertex prediction head for direct 3D positions
        self.vertex_head: Optional[nn.Module] = None
        if include_vertex_head:
            from .vertex_prediction import VertexPredictionHead
            self.vertex_head = VertexPredictionHead(
                embed_dim=embed_dim,
                max_vertices=max_vertices,
                hidden_dim=embed_dim,
                num_refinement_steps=num_refinement_steps,
            )
            _log.info(
                "CompositeHead: vertex prediction enabled "
                "(max_vertices=%d, refinement_steps=%d)",
                max_vertices, num_refinement_steps,
            )

        # Pre-compute boolean mask tensors for each command type
        # Shape: [num_command_types, num_param_slots] (True = active)
        mask = torch.zeros(num_command_types, num_param_slots, dtype=torch.bool)
        for cmd_type, active_indices in PARAMETER_MASKS.items():
            for idx in active_indices:
                if idx < num_param_slots:
                    mask[int(cmd_type), idx] = True
        self.register_buffer("_param_mask", mask)

        _log.info(
            "CompositeHead initialised: embed_dim=%d, commands=%d, "
            "param_slots=%d, levels=%d",
            embed_dim, num_command_types, num_param_slots, num_levels,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        command_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        """Predict command types and parameters with optional masking.

        Args:
            hidden_states: [batch, seq_len, embed_dim]
            command_targets: [batch, seq_len] integer command-type
                targets. When provided, masking uses the ground-truth
                command types; otherwise the argmax prediction is used.

        Returns:
            Dictionary with:
                - command_type_logits: [batch, seq_len, num_command_types]
                - parameter_logits: list of 16 [batch, seq_len, num_levels]
                - parameter_mask: [batch, seq_len, num_param_slots] bool
        """
        command_logits = self.command_head(hidden_states)
        parameter_logits = self.parameter_heads(hidden_states)

        # Determine which command type governs each position
        if command_targets is not None:
            cmd_indices = command_targets  # [B, S]
        else:
            cmd_indices = command_logits.argmax(dim=-1)  # [B, S]

        # Look up the parameter mask for each position
        # _param_mask: [num_command_types, num_param_slots]
        # cmd_indices: [B, S] -> need [B, S, num_param_slots] bool
        cmd_indices_clamped = cmd_indices.clamp(0, self.num_command_types - 1)
        active_mask = self._param_mask[cmd_indices_clamped]  # [B, S, P]

        # Apply masking: set inactive parameter logits to uniform (zero logits)
        masked_parameter_logits: List[torch.Tensor] = []
        for slot_idx, slot_logits in enumerate(parameter_logits):
            slot_active = active_mask[..., slot_idx].unsqueeze(-1)  # [B, S, 1]
            # Where inactive, replace logits with zeros (uniform distribution)
            masked_logits = torch.where(
                slot_active.expand_as(slot_logits),
                slot_logits,
                torch.zeros_like(slot_logits),
            )
            masked_parameter_logits.append(masked_logits)

        result = {
            "command_type_logits": command_logits,
            "parameter_logits": masked_parameter_logits,
            "parameter_mask": active_mask,
        }

        # Optional vertex prediction (if head is enabled)
        if self.vertex_head is not None:
            result["vertex_prediction"] = self.vertex_head(hidden_states)

        return result

    @torch.no_grad()
    def decode_to_token_sequence(
        self,
        command_logits: torch.Tensor,
        param_logits: List[torch.Tensor],
        batch_index: int = 0,
    ):
        """Convert model output logits to a geotoken TokenSequence.

        Takes the raw logits from a generative model's forward pass and
        produces a geotoken-compatible ``TokenSequence`` by argmaxing
        command types and parameters, applying ``PARAMETER_MASKS``, and
        stopping at the first EOS token.

        This is the inverse of what :class:`GeoTokenDataset` does
        (TokenSequence → tensors); here we go tensors → TokenSequence.

        Args:
            command_logits: ``[B, S, num_command_types]`` logits.
            param_logits: List of 16 ``[B, S, num_levels]`` logits.
            batch_index: Which sample in the batch to decode (default 0).

        Returns:
            A ``geotoken.TokenSequence`` with decoded ``command_tokens``.
            Only command tokens are populated; graph and constraint tokens
            are not (those come from separate decoders in full models).
        """
        # Lazy imports from geotoken
        from geotoken.tokenizer.token_types import (
            CommandToken,
            CommandType as GeoCommandType,
            TokenSequence,
        )

        # Canonical name ordering — must match geotoken CommandType enum
        _INT_TO_GEO_CMD = list(GeoCommandType)  # [SOL, LINE, ARC, ...]

        seq_len = command_logits.shape[1]
        cmd_indices = command_logits[batch_index].argmax(dim=-1)  # [S]

        command_tokens = []
        for s in range(seq_len):
            cmd_idx = int(cmd_indices[s].item())
            cmd_idx = min(cmd_idx, len(_INT_TO_GEO_CMD) - 1)
            geo_cmd = _INT_TO_GEO_CMD[cmd_idx]

            # Argmax each parameter head for this position
            params = []
            for p in range(self.num_param_slots):
                if p < len(param_logits):
                    val = int(param_logits[p][batch_index, s].argmax(dim=-1).item())
                else:
                    val = 0
                params.append(val)

            # Get the parameter mask for this command type
            step_cmd = CommandType(cmd_idx)
            active = PARAMETER_MASKS.get(step_cmd, [])
            mask = [i in active for i in range(self.num_param_slots)]

            command_tokens.append(CommandToken(
                command_type=geo_cmd,
                parameters=params,
                parameter_mask=mask,
            ))

            # Stop at first EOS
            if geo_cmd == GeoCommandType.EOS:
                break

        return TokenSequence(command_tokens=command_tokens)
