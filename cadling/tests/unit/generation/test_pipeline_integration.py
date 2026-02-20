"""Integration tests for GenerationPipeline.

Tests the full generation flow from request to result.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestGenerationPipeline:
    """Tests for GenerationPipeline class."""

    def test_pipeline_imports(self):
        """Test GenerationPipeline can be imported."""
        from cadling.generation.pipeline import GenerationPipeline

        assert GenerationPipeline is not None

    def test_generation_config_imports(self):
        """Test GenerationConfig can be imported."""
        from cadling.datamodel.generation import GenerationConfig, GenerationBackend

        config = GenerationConfig(
            backend=GenerationBackend.CODEGEN_CADQUERY,
            max_retries=3,
            validate_output=True,
        )

        assert config.backend == GenerationBackend.CODEGEN_CADQUERY
        assert config.max_retries == 3

    def test_generation_request_imports(self):
        """Test GenerationRequest can be imported."""
        from cadling.datamodel.generation import GenerationRequest, GenerationConfig

        config = GenerationConfig()
        request = GenerationRequest(
            text_prompt="A simple bracket",
            config=config,
            output_dir="./output",
        )

        assert request.text_prompt == "A simple bracket"

    def test_generation_result_imports(self):
        """Test GenerationResult can be imported."""
        from cadling.datamodel.generation import GenerationResult

        result = GenerationResult(
            success=True,
            output_path="/path/to/output.step",
        )

        assert result.success is True


class TestCodegenPipelineFlow:
    """Tests for codegen backend flow through pipeline."""

    @patch("cadling.generation.pipeline.GenerationPipeline._load_cadquery_backend")
    def test_codegen_calls_generator_correctly(self, mock_load_backend):
        """Test pipeline calls generator with correct signature."""
        from cadling.generation.pipeline import GenerationPipeline
        from cadling.datamodel.generation import (
            GenerationConfig,
            GenerationRequest,
            GenerationBackend,
        )

        # Mock generator and executor
        mock_generator = MagicMock()
        mock_generator.generate.return_value = "result = cq.Workplane()"
        mock_executor = MagicMock()
        mock_executor.execute.return_value = {"success": True, "shape": MagicMock()}
        mock_executor.export_step.return_value = True

        mock_load_backend.return_value = (mock_generator, mock_executor)

        pipeline = GenerationPipeline()
        config = GenerationConfig(
            backend=GenerationBackend.CODEGEN_CADQUERY,
            max_retries=0,
            validate_output=False,
        )
        request = GenerationRequest(
            text_prompt="A cube",
            config=config,
            output_dir="/tmp",
        )

        pipeline.generate(request)

        # Verify generator was called with positional args, not kwargs
        mock_generator.generate.assert_called()
        call_args = mock_generator.generate.call_args

        # Should be positional args: (description, image_path)
        assert len(call_args.args) >= 1 or len(call_args.kwargs) == 0

    @patch("cadling.generation.pipeline.GenerationPipeline._load_cadquery_backend")
    def test_codegen_calls_executor_correctly(self, mock_load_backend):
        """Test pipeline calls executor with correct signature."""
        from cadling.generation.pipeline import GenerationPipeline
        from cadling.datamodel.generation import (
            GenerationConfig,
            GenerationRequest,
            GenerationBackend,
        )

        mock_generator = MagicMock()
        mock_generator.generate.return_value = "result = cq.Workplane()"

        mock_executor = MagicMock()
        mock_executor.execute.return_value = {"success": True, "shape": MagicMock()}
        mock_executor.export_step.return_value = True

        mock_load_backend.return_value = (mock_generator, mock_executor)

        pipeline = GenerationPipeline()
        config = GenerationConfig(
            backend=GenerationBackend.CODEGEN_CADQUERY,
            max_retries=0,
            validate_output=False,
        )
        request = GenerationRequest(
            text_prompt="A cube",
            config=config,
            output_dir="/tmp",
        )

        pipeline.generate(request)

        # Verify executor.execute was called with single positional arg
        mock_executor.execute.assert_called()
        call_args = mock_executor.execute.call_args

        # Should be single positional arg (the script)
        assert len(call_args.args) == 1
        assert isinstance(call_args.args[0], str)

    @patch("cadling.generation.pipeline.GenerationPipeline._load_cadquery_backend")
    def test_codegen_calls_export_step(self, mock_load_backend):
        """Test pipeline calls export_step on success."""
        from cadling.generation.pipeline import GenerationPipeline
        from cadling.datamodel.generation import (
            GenerationConfig,
            GenerationRequest,
            GenerationBackend,
        )

        mock_generator = MagicMock()
        mock_generator.generate.return_value = "result = cq.Workplane()"

        mock_executor = MagicMock()
        mock_exec_result = {"success": True, "shape": MagicMock()}
        mock_executor.execute.return_value = mock_exec_result
        mock_executor.export_step.return_value = True

        mock_load_backend.return_value = (mock_generator, mock_executor)

        pipeline = GenerationPipeline()
        config = GenerationConfig(
            backend=GenerationBackend.CODEGEN_CADQUERY,
            max_retries=0,
            validate_output=False,
        )
        request = GenerationRequest(
            text_prompt="A cube",
            config=config,
            output_dir="/tmp",
        )

        pipeline.generate(request)

        # Verify export_step was called
        mock_executor.export_step.assert_called_once()

    @patch("cadling.generation.pipeline.GenerationPipeline._load_cadquery_backend")
    def test_codegen_retry_calls_repair(self, mock_load_backend):
        """Test pipeline calls repair() on retry."""
        from cadling.generation.pipeline import GenerationPipeline
        from cadling.datamodel.generation import (
            GenerationConfig,
            GenerationRequest,
            GenerationBackend,
        )

        mock_generator = MagicMock()
        mock_generator.generate.return_value = "result = broken()"
        mock_generator.repair.return_value = "result = cq.Workplane()"

        mock_executor = MagicMock()
        # First call fails, second succeeds
        mock_executor.execute.side_effect = [
            {"success": False, "error": "Execution failed"},
            {"success": True, "shape": MagicMock()},
        ]
        mock_executor.export_step.return_value = True

        mock_load_backend.return_value = (mock_generator, mock_executor)

        pipeline = GenerationPipeline()
        config = GenerationConfig(
            backend=GenerationBackend.CODEGEN_CADQUERY,
            max_retries=1,
            validate_output=False,
        )
        request = GenerationRequest(
            text_prompt="A cube",
            config=config,
            output_dir="/tmp",
        )

        pipeline.generate(request)

        # Verify repair was called on retry
        mock_generator.repair.assert_called_once()


class TestGenerationBackendEnum:
    """Tests for GenerationBackend enum."""

    def test_backend_values(self):
        """Test expected backend values exist."""
        from cadling.datamodel.generation import GenerationBackend

        assert hasattr(GenerationBackend, "CODEGEN_CADQUERY")
        assert hasattr(GenerationBackend, "CODEGEN_OPENSCAD")
        assert hasattr(GenerationBackend, "VAE")
        assert hasattr(GenerationBackend, "VQVAE")
        assert hasattr(GenerationBackend, "DIFFUSION")
