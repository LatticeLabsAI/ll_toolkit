"""Code generation proposers — LLM → typed CodeProposal.

Wraps cadling's existing ``CadQueryGenerator`` and ``OpenSCADGenerator``
to produce typed ``CodeProposal`` objects with confidence scores,
metadata, and retry context integration.
"""
from ll_gen.codegen.cadquery_proposer import CadQueryProposer
from ll_gen.codegen.openscad_proposer import OpenSCADProposer
from ll_gen.codegen import prompt_library

__all__ = [
    "CadQueryProposer",
    "OpenSCADProposer",
    "prompt_library",
]
