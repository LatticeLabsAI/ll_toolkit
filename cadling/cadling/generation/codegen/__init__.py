"""Code generation backend for text-to-CAD via LLM-generated scripts.

This module provides generators and executors for converting natural language
descriptions of 3D parts into executable CAD scripts (CadQuery or OpenSCAD),
then running those scripts to produce STEP/STL geometry.

Classes:
    CadQueryGenerator: Generate CadQuery Python scripts from text descriptions
    CadQueryExecutor: Execute CadQuery scripts in a sandboxed environment
    CadQueryValidator: Validate generated STEP output and retry on failure
    OpenSCADGenerator: Generate OpenSCAD scripts from text descriptions
    OpenSCADExecutor: Execute OpenSCAD scripts via subprocess
"""

from cadling.generation.codegen.cadquery_generator import (
    CadQueryGenerator,
    CadQueryExecutor,
    CadQueryValidator,
)
from cadling.generation.codegen.openscad_generator import (
    OpenSCADGenerator,
    OpenSCADExecutor,
)

__all__ = [
    "CadQueryGenerator",
    "CadQueryExecutor",
    "CadQueryValidator",
    "OpenSCADGenerator",
    "OpenSCADExecutor",
]
