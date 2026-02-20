"""Typed proposal protocol for the propose/dispose architecture.

Every neural generation path produces a *proposal* — a structured,
typed object carrying candidate geometry plus metadata (confidence,
generation source, retry context).  The deterministic disposal engine
accepts proposals, executes them, validates the result, and returns a
DisposalResult.

Proposal types
--------------
CodeProposal
    Executable CAD code (CadQuery, OpenSCAD, or raw pythonocc).
CommandSequenceProposal
    Token sequence from a VAE / VQ-VAE decoder.
LatentProposal
    Decoded point grids from a diffusion model.

Results
-------
DisposalResult
    Complete outcome of deterministic execution + validation.
GeometryReport
    Introspection data (volume, bbox, face counts, surface types).
ValidationFinding
    Per-entity validation error with OCC error code.
RepairAction
    Record of a single deterministic repair step.
"""
from ll_gen.proposals.base import BaseProposal
from ll_gen.proposals.code_proposal import CodeProposal
from ll_gen.proposals.command_proposal import CommandSequenceProposal
from ll_gen.proposals.disposal_result import (
    DisposalResult,
    GeometryReport,
    RepairAction,
    ValidationFinding,
)
from ll_gen.proposals.latent_proposal import LatentProposal

__all__ = [
    "BaseProposal",
    "CodeProposal",
    "CommandSequenceProposal",
    "DisposalResult",
    "GeometryReport",
    "LatentProposal",
    "RepairAction",
    "ValidationFinding",
]
