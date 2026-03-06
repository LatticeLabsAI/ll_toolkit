"""Command sequence proposal — tokenized CAD construction history.

A CommandSequenceProposal carries the output of a neural decoder
(VAE, VQ-VAE) as a sequence of quantized command tokens following
the DeepCAD 6-command vocabulary (SOL, LINE, ARC, CIRCLE, EXTRUDE,
EOS) with 16 quantized integer parameters per command.

The disposal engine's ``command_executor`` receives this proposal,
dequantizes the tokens back to continuous parameter values, builds
sketch→extrude→boolean operations through OpenCASCADE, and
produces a ``TopoDS_Shape``.
"""
from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ll_gen.proposals.base import BaseProposal


@dataclass
class CommandSequenceProposal(BaseProposal):
    """A proposal containing a quantized command token sequence.

    Attributes:
        token_ids: Flat integer token ID sequence as produced by the
            neural decoder.  These are vocabulary-level IDs that include
            special tokens (PAD=0, BOS=1, EOS=2, SEP=3, UNK=4) and
            command/parameter tokens.
        command_dicts: Structured command list.  Each dict contains::

                {
                    "command_type": str,       # "SOL", "LINE", ...
                    "parameters": list[int],   # 16 quantized ints
                    "parameter_mask": list[bool],  # 16 bools
                }

            Either ``token_ids`` or ``command_dicts`` must be non-empty.
        quantization_bits: Bit-width used for parameter quantization.
        normalization_range: Bounding cube size used for normalization
            (typically 2.0 for a [-1,1] cube).
        precision_tier: Name of the precision tier ("DRAFT", "STANDARD",
            "PRECISION") from geotoken.
        latent_vector: The latent-space vector that was decoded to
            produce this sequence (if available).  Shape ``(latent_dim,)``.
    """

    token_ids: List[int] = field(default_factory=list)
    command_dicts: List[Dict[str, Any]] = field(default_factory=list)
    quantization_bits: int = 8
    normalization_range: float = 2.0
    precision_tier: str = "STANDARD"
    latent_vector: Optional[Any] = None  # np.ndarray, lazy import

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_token_sequence(self) -> Any:
        """Convert to a geotoken ``TokenSequence`` for downstream processing.

        Imports geotoken lazily so ll_gen can be used without it
        installed (the proposal dataclass itself is pure Python).

        Returns:
            A ``geotoken.TokenSequence`` populated with ``CommandToken``
            entries derived from ``command_dicts``.

        Raises:
            ImportError: If geotoken is not installed.
            ValueError: If neither ``token_ids`` nor ``command_dicts``
                are populated.
        """
        from geotoken.tokenizer.token_types import (
            CommandToken,
            CommandType,
            TokenSequence,
        )

        if not self.command_dicts and not self.token_ids:
            raise ValueError(
                "CommandSequenceProposal has neither token_ids nor "
                "command_dicts — cannot convert to TokenSequence."
            )

        if self.command_dicts:
            tokens = []
            for cmd in self.command_dicts:
                ct = CommandType(cmd["command_type"])
                params = cmd.get("parameters", [0] * 16)
                mask = cmd.get(
                    "parameter_mask",
                    CommandToken.get_parameter_mask(ct),
                )
                tokens.append(CommandToken(
                    command_type=ct,
                    parameters=list(params),
                    parameter_mask=list(mask),
                ))
            return TokenSequence(command_tokens=tokens, metadata={
                "quantization_bits": self.quantization_bits,
                "normalization_range": self.normalization_range,
                "precision_tier": self.precision_tier,
                "proposal_id": self.proposal_id,
            })

        # Fallback: decode token_ids → command_dicts first
        return self._decode_token_ids_to_sequence()

    def _decode_token_ids_to_sequence(self) -> Any:
        """Decode flat token IDs into a geotoken TokenSequence.

        This mirrors the decoding logic in ll_stepnet's
        ``CADGenerationPipeline._decode_to_token_sequence`` but
        operates on raw integer lists without requiring torch.

        Special token IDs (vocabulary convention from geotoken):
            0 = PAD, 1 = BOS, 2 = EOS, 3 = SEP, 4 = UNK
            5 = reserved
            6 = SOL, 7 = LINE, 8 = ARC, 9 = CIRCLE, 10 = EXTRUDE, 11 = EOS_CMD
            ≥ 12 = parameter tokens (value = token_id - 12)
        """
        from geotoken.tokenizer.token_types import (
            CommandToken,
            CommandType,
            TokenSequence,
        )

        COMMAND_ID_MAP = {
            6: CommandType.SOL,
            7: CommandType.LINE,
            8: CommandType.ARC,
            9: CommandType.CIRCLE,
            10: CommandType.EXTRUDE,
            11: CommandType.EOS,
        }
        PARAM_OFFSET = 12

        commands: List[CommandToken] = []
        i = 0
        ids = self.token_ids

        while i < len(ids):
            tid = ids[i]

            # EOS sequence token (must check before tid < 6 guard)
            if tid == 2 or tid == 11:
                commands.append(CommandToken(
                    command_type=CommandType.EOS,
                    parameters=[0] * 16,
                    parameter_mask=[False] * 16,
                ))
                break

            # Skip special tokens (PAD=0, BOS=1, SEP=3, UNK=4, reserved=5)
            if tid < 6:
                i += 1
                continue

            if tid not in COMMAND_ID_MAP:
                i += 1
                continue

            cmd_type = COMMAND_ID_MAP[tid]
            mask = CommandToken.get_parameter_mask(cmd_type)
            num_params = sum(mask)
            params = [0] * 16

            # Read the next num_params tokens as parameter values
            j = 0
            k = i + 1
            while j < 16 and k < len(ids):
                if not mask[j]:
                    j += 1
                    continue
                if ids[k] >= PARAM_OFFSET:
                    params[j] = ids[k] - PARAM_OFFSET
                elif ids[k] in COMMAND_ID_MAP or ids[k] == 2:
                    # Next command starts — remaining params stay 0
                    break
                k += 1
                j += 1

            commands.append(CommandToken(
                command_type=cmd_type,
                parameters=params,
                parameter_mask=mask,
            ))
            i = k

        return TokenSequence(command_tokens=commands, metadata={
            "quantization_bits": self.quantization_bits,
            "normalization_range": self.normalization_range,
            "precision_tier": self.precision_tier,
            "proposal_id": self.proposal_id,
            "decoded_from": "token_ids",
        })

    def dequantize(self) -> List[Dict[str, Any]]:
        """Dequantize command parameters to continuous float values.

        Maps quantized integer parameters back to continuous coordinates
        using the inverse of the symmetric quantization::

            value = (param / (levels - 1)) * 2 * range - range

        where ``levels = 2 ** quantization_bits`` and
        ``range = normalization_range``.  This maps ``[0, levels-1]``
        back to ``[-range, +range]``.

        Returns:
            List of dicts with ``command_type`` (str), ``parameters``
            (list of floats), and ``parameter_mask`` (list of bools).
        """
        levels = 2 ** self.quantization_bits
        result = []

        source = self.command_dicts
        if not source:
            # Build from token sequence
            ts = self.to_token_sequence()
            source = [
                {
                    "command_type": ct.command_type.value,
                    "parameters": ct.parameters,
                    "parameter_mask": ct.parameter_mask,
                }
                for ct in ts.command_tokens
            ]

        for cmd in source:
            params_int = cmd["parameters"]
            mask = cmd["parameter_mask"]
            params_float = []
            for p, m in zip(params_int, mask):
                if m:
                    val = (p / (levels - 1)) * 2.0 * self.normalization_range - self.normalization_range
                    params_float.append(float(val))
                else:
                    params_float.append(0.0)

            result.append({
                "command_type": cmd["command_type"],
                "parameters": params_float,
                "parameter_mask": mask,
            })

        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sequence_length(self) -> int:
        """Number of commands in the sequence."""
        if self.command_dicts:
            return len(self.command_dicts)
        return len(self.token_ids)

    @property
    def num_sketch_loops(self) -> int:
        """Count the number of sketch loops (SOL commands)."""
        if self.command_dicts:
            return sum(
                1 for c in self.command_dicts
                if c.get("command_type") == "SOL"
            )
        # From token IDs: SOL = 6
        return sum(1 for t in self.token_ids if t == 6)

    @property
    def num_extrusions(self) -> int:
        """Count the number of extrusion operations."""
        if self.command_dicts:
            return sum(
                1 for c in self.command_dicts
                if c.get("command_type") == "EXTRUDE"
            )
        # From token IDs: EXTRUDE = 10
        return sum(1 for t in self.token_ids if t == 10)

    # ------------------------------------------------------------------
    # Retry support
    # ------------------------------------------------------------------

    def with_error_context(self, error: Dict[str, Any]) -> "CommandSequenceProposal":
        """Create a retry proposal with error context.

        Preserves the latent vector (so the generator can perturb it)
        but clears the decoded tokens.
        """
        # Delegate to base which uses shallow copy to preserve tensor grad_fn
        new = super().with_error_context(error)
        new.token_ids = []
        new.command_dicts = []
        new.confidence = 0.0
        # Keep latent_vector for perturbed re-decoding (shallow ref is fine,
        # numpy .copy() only needed if caller mutates in-place)
        if self.latent_vector is not None:
            new.latent_vector = self.latent_vector.copy()
        return new

    def summary(self) -> Dict[str, Any]:
        """Extended summary with command-specific fields."""
        base = super().summary()
        base.update({
            "sequence_length": self.sequence_length,
            "num_sketch_loops": self.num_sketch_loops,
            "num_extrusions": self.num_extrusions,
            "quantization_bits": self.quantization_bits,
            "precision_tier": self.precision_tier,
            "has_latent": self.latent_vector is not None,
        })
        return base
