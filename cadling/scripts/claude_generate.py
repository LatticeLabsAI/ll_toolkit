#!/usr/bin/env python
"""Interactive CAD generation using Claude as the LLM backend.

Usage:
    python scripts/claude_generate.py

This script demonstrates the generation pipeline by taking a text
description and generating a CadQuery script. Since Claude is already
running in the terminal, you can paste the generated code here for
execution.

For automated generation with API keys, use:
    cadling generate --from-text "..." --provider anthropic -o output.step
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add cadling to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cadling.generation.codegen.cadquery_generator import CadQueryExecutor


def execute_script(script: str, output_path: str = "/tmp/generated.step") -> dict:
    """Execute a CadQuery script and export to STEP.

    Args:
        script: CadQuery Python script string.
        output_path: Path for output STEP file.

    Returns:
        Execution result dictionary.
    """
    executor = CadQueryExecutor(timeout=30)
    result = executor.execute(script)

    if result["success"]:
        executor.export_step(result, output_path)
        print(f"✓ Exported to: {output_path}")
    else:
        print(f"✗ Execution failed: {result.get('error')}")

    return result


# Example scripts that Claude can generate
EXAMPLE_SCRIPTS = {
    "cube_with_hole": '''
import cadquery as cq

result = (
    cq.Workplane("XY")
    .box(20, 20, 20)
    .faces(">Z")
    .workplane()
    .hole(8)
)
''',
    "bracket": '''
import cadquery as cq

result = (
    cq.Workplane("XY")
    .box(60, 40, 5)
    .faces(">Z").workplane()
    .pushPoints([(-20, 0), (20, 0)])
    .hole(6)
    .faces("<Y").workplane()
    .transformed(offset=(0, 0, 15))
    .box(60, 5, 30, centered=(True, True, False))
)
''',
    "bearing_housing": '''
import cadquery as cq

result = (
    cq.Workplane("XY")
    .circle(40).extrude(10)
    .faces(">Z").workplane()
    .polarArray(30, 0, 360, 4).hole(6)
    .faces(">Z").workplane().hole(20)
    .faces(">Z").workplane()
    .circle(25).extrude(15)
    .faces(">Z").workplane().hole(20)
)
''',
}


if __name__ == "__main__":
    print("=" * 60)
    print("CAD Generation with Claude")
    print("=" * 60)
    print()
    print("Available example scripts:")
    for name in EXAMPLE_SCRIPTS:
        print(f"  - {name}")
    print()
    print("To generate custom CAD, ask Claude to write a CadQuery script")
    print("for your part, then paste it here.")
    print()

    # Demo: execute an example
    print("Demo: Executing 'bearing_housing' example...")
    print("-" * 40)
    result = execute_script(
        EXAMPLE_SCRIPTS["bearing_housing"],
        "/tmp/demo_bearing.step"
    )
    print(f"Success: {result['success']}")
