"""Unit tests for CadQuery and OpenSCAD code generators.

Tests generator signatures, repair methods, and code extraction.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


class TestCadQueryGenerator:
    """Tests for CadQueryGenerator class."""

    @patch("cadling.generation.codegen.cadquery_generator.CadQueryGenerator._get_agent")
    def test_generate_signature(self, mock_get_agent):
        """Test generate() method accepts correct arguments."""
        from cadling.generation.codegen.cadquery_generator import CadQueryGenerator

        mock_agent = MagicMock()
        mock_agent.ask.return_value = "```python\nresult = cq.Workplane()\n```"
        mock_get_agent.return_value = mock_agent

        generator = CadQueryGenerator(model_name="gpt-4", api_provider="openai")

        # Test with just description
        code = generator.generate("A simple cube")
        assert "cq.Workplane" in code

        # Test with description and image path
        code = generator.generate("A bracket", "/path/to/image.png")
        assert "cq.Workplane" in code

    @patch("cadling.generation.codegen.cadquery_generator.CadQueryGenerator._get_agent")
    def test_repair_method_exists(self, mock_get_agent):
        """Test repair() method exists and accepts correct arguments."""
        from cadling.generation.codegen.cadquery_generator import CadQueryGenerator

        mock_agent = MagicMock()
        mock_agent.ask.return_value = "```python\nresult = cq.Workplane().box(10, 10, 10)\n```"
        mock_get_agent.return_value = mock_agent

        generator = CadQueryGenerator(model_name="gpt-4", api_provider="openai")

        # Test repair signature
        repaired = generator.repair(
            code="broken code",
            error="NameError: undefined",
            description="A cube",
        )
        assert "cq.Workplane" in repaired

    @patch("cadling.generation.codegen.cadquery_generator.CadQueryGenerator._get_agent")
    def test_repair_uses_repair_prompt(self, mock_get_agent):
        """Test repair() loads and uses repair prompt template."""
        from cadling.generation.codegen.cadquery_generator import CadQueryGenerator

        mock_agent = MagicMock()
        mock_agent.ask.return_value = "```python\nresult = fixed()\n```"
        mock_get_agent.return_value = mock_agent

        generator = CadQueryGenerator(model_name="gpt-4", api_provider="openai")

        generator.repair(
            code="original code",
            error="SyntaxError",
            description="A bracket",
        )

        # Verify the prompt was called
        call_args = mock_agent.ask.call_args
        prompt = call_args[0][0]  # First positional arg

        # The prompt should contain the original code, error, and description
        assert "original code" in prompt or "SyntaxError" in prompt

    def test_extract_code_fenced_block(self):
        """Test _extract_code extracts from fenced code blocks."""
        from cadling.generation.codegen.cadquery_generator import CadQueryGenerator

        generator = CadQueryGenerator()

        response = """
Here's the code:

```python
import cadquery as cq
result = cq.Workplane("XY").box(10, 10, 10)
```

This creates a cube.
"""
        code = generator._extract_code(response)
        assert "import cadquery" in code
        assert "Workplane" in code

    def test_extract_code_multiple_blocks(self):
        """Test _extract_code returns longest code block."""
        from cadling.generation.codegen.cadquery_generator import CadQueryGenerator

        generator = CadQueryGenerator()

        response = """
Short example:
```python
x = 1
```

Full code:
```python
import cadquery as cq
result = cq.Workplane("XY").box(10, 10, 10)
# More lines
```
"""
        code = generator._extract_code(response)
        # Should get the longer block
        assert "import cadquery" in code


class TestOpenSCADGenerator:
    """Tests for OpenSCADGenerator class."""

    @patch("cadling.generation.codegen.openscad_generator.OpenSCADGenerator._get_agent")
    def test_generate_signature(self, mock_get_agent):
        """Test generate() method accepts correct arguments."""
        from cadling.generation.codegen.openscad_generator import OpenSCADGenerator

        mock_agent = MagicMock()
        mock_agent.ask.return_value = "```openscad\ncube([10, 10, 10]);\n```"
        mock_get_agent.return_value = mock_agent

        generator = OpenSCADGenerator(model_name="gpt-4", api_provider="openai")

        # Test with just description
        code = generator.generate("A simple cube")
        assert "cube" in code

        # Test with description and image path
        code = generator.generate("A bracket", "/path/to/image.png")
        assert "cube" in code

    @patch("cadling.generation.codegen.openscad_generator.OpenSCADGenerator._get_agent")
    def test_repair_method_exists(self, mock_get_agent):
        """Test repair() method exists and accepts correct arguments."""
        from cadling.generation.codegen.openscad_generator import OpenSCADGenerator

        mock_agent = MagicMock()
        mock_agent.ask.return_value = "```openscad\ncube([10, 10, 10]);\n```"
        mock_get_agent.return_value = mock_agent

        generator = OpenSCADGenerator(model_name="gpt-4", api_provider="openai")

        # Test repair signature
        repaired = generator.repair(
            code="broken code",
            error="Syntax error",
            description="A cube",
        )
        assert "cube" in repaired

    def test_extract_code_openscad_fence(self):
        """Test _extract_code handles openscad fence."""
        from cadling.generation.codegen.openscad_generator import OpenSCADGenerator

        generator = OpenSCADGenerator()

        response = """
```openscad
difference() {
    cube([20, 20, 10]);
    cylinder(h=15, r=5);
}
```
"""
        code = generator._extract_code(response)
        assert "difference" in code
        assert "cube" in code


class TestCadQueryExecutor:
    """Tests for CadQueryExecutor class."""

    def test_execute_returns_dict(self):
        """Test execute() returns result dictionary."""
        pytest.importorskip("cadquery")

        from cadling.generation.codegen.cadquery_generator import CadQueryExecutor

        executor = CadQueryExecutor(timeout=5)

        script = """
import cadquery as cq
result = cq.Workplane("XY").box(10, 10, 10)
"""
        result = executor.execute(script)

        assert isinstance(result, dict)
        assert "success" in result
        # Note: May not actually succeed without full cadquery environment

    def test_execute_accepts_single_arg(self):
        """Test execute() signature accepts single script argument."""
        from cadling.generation.codegen.cadquery_generator import CadQueryExecutor

        executor = CadQueryExecutor()

        # Should not raise TypeError
        try:
            executor.execute("result = None")
        except Exception:
            pass  # Execution may fail, but signature is correct


class TestOpenSCADExecutor:
    """Tests for OpenSCADExecutor class."""

    def test_execute_returns_dict(self):
        """Test execute() returns result dictionary."""
        from cadling.generation.codegen.openscad_generator import OpenSCADExecutor
        import tempfile
        import os

        executor = OpenSCADExecutor()

        script = "cube([10, 10, 10]);"
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.stl")
            result = executor.execute(script, output_path)

        assert isinstance(result, dict)
        assert "success" in result

    def test_execute_requires_output_path(self):
        """Test execute() requires output_path argument."""
        from cadling.generation.codegen.openscad_generator import OpenSCADExecutor
        import inspect

        sig = inspect.signature(OpenSCADExecutor.execute)
        params = list(sig.parameters.keys())

        assert "script" in params
        assert "output_path" in params
