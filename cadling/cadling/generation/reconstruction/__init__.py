"""Output reconstruction for generated CAD models.

This package provides components for reconstructing valid CAD geometry from
generated representations, supporting both command-based (DeepCAD) and
diffusion-based (BrepGen) generation approaches.

Classes:
    CommandExecutor: Execute predicted command token sequences in OpenCASCADE
    BSplineSurfaceFitter: Fit B-spline surfaces from point grids (BrepGen)
    TopologyMerger: Merge duplicate edges/faces via mating duplication recovery
    ConstraintSolver: Solve sketch constraints for geometric consistency
    ValidationFeedbackLoop: Iterative validation and repair loop
    GenerationResult: Result container for generation pipeline output
    GraphReconstructor: Reconstruct OCC primitives from decoded graph features

Example:
    from cadling.generation.reconstruction import (
        CommandExecutor,
        ValidationFeedbackLoop,
        GraphReconstructor,
    )

    executor = CommandExecutor(tolerance=1e-6)
    loop = ValidationFeedbackLoop(executor, max_retries=3)

    result = loop.run(command_tokens, mode='command')
    if result.valid:
        executor.export_step(result.shape, "output.step")

    # Or use graph-based reconstruction
    reconstructor = GraphReconstructor()
    primitives = reconstructor.reconstruct(node_features, edge_index)
"""

from __future__ import annotations

from cadling.generation.reconstruction.command_executor import CommandExecutor

# Alias for backward compatibility with some import patterns
CommandReconstructor = CommandExecutor
from cadling.generation.reconstruction.surface_fitter import BSplineSurfaceFitter
from cadling.generation.reconstruction.topology_merger import TopologyMerger
from cadling.generation.reconstruction.constraint_solver import (
    ConstraintType,
    SketchConstraint,
    ConstraintSolver,
)
from cadling.generation.reconstruction.validation_loop import (
    GenerationResult,
    ValidationFeedbackLoop,
)
from cadling.generation.reconstruction.graph_reconstructor import (
    GraphReconstructor,
    ReconstructedPrimitive,
    ReconstructionResult,
)

__all__ = [
    "BSplineSurfaceFitter",
    "CommandExecutor",
    "CommandReconstructor",  # Alias for CommandExecutor
    "ConstraintSolver",
    "ConstraintType",
    "GenerationResult",
    "GraphReconstructor",
    "ReconstructedPrimitive",
    "ReconstructionResult",
    "SketchConstraint",
    "TopologyMerger",
    "ValidationFeedbackLoop",
]
