"""Deterministic disposal engine — execute, validate, repair, export.

The disposal engine is the CPU-side half of the propose/dispose
architecture.  It accepts typed proposals from the neural layer,
executes them through OpenCASCADE, validates the resulting geometry
against 37 BRepCheck codes, attempts deterministic repair via
ShapeFix, computes introspection data, and exports to STEP/STL.

Modules
-------
engine
    ``DisposalEngine`` — the main entry point that orchestrates
    execution → validation → repair → introspection → export.
code_executor
    Execute ``CodeProposal`` (CadQuery/OpenSCAD/pythonocc scripts).
command_executor
    Execute ``CommandSequenceProposal`` (token sequences → OCC).
surface_executor
    Execute ``LatentProposal`` (point grids → B-spline → sewing).
validator
    Full BRepCheck validation with 37-code coverage.
repairer
    ShapeFix auto-repair with tolerance escalation.
introspector
    GeometryReport generation (BRepGProp, BRepBndLib, TopExp).
exporter
    STEP and STL export with schema selection.
"""
from ll_gen.disposal.engine import DisposalEngine

__all__ = ["DisposalEngine"]
