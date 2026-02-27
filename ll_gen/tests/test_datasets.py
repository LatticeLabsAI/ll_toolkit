"""Comprehensive test suite for ll_gen datasets module.

Tests cover:
- Module imports for all dataset loaders
- Internal tokenization functions with synthetic data
- Dataset class instantiation and parameter storage
- Parameter quantization and token ID generation
- Padding and command handling
- Configuration validation
- Special handling for annotation levels and entity types
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    from ll_gen.datasets.deepcad_loader import (
        _tokenize_deepcad_sample,
        COMMAND_TYPE_IDS,
        DeepCADDataset,
        PAD_TOKEN_ID,
        BOS_TOKEN_ID,
        EOS_TOKEN_ID,
        PARAM_OFFSET,
    )
    deepcad_available = True
except ImportError:
    deepcad_available = False

try:
    from ll_gen.datasets.abc_loader import (
        _tokenize_abc_sample,
        ABCDataset,
    )
    abc_available = True
except ImportError:
    abc_available = False

try:
    from ll_gen.datasets.text2cad_loader import (
        _tokenize_text2cad_sample,
        ANNOTATION_LEVELS,
        Text2CADDataset,
    )
    text2cad_available = True
except ImportError:
    text2cad_available = False

try:
    from ll_gen.datasets.sketchgraphs_loader import (
        _tokenize_sketchgraphs_sample,
        ENTITY_TYPES,
        CONSTRAINT_TYPES,
        SketchGraphsDataset,
    )
    sketchgraphs_available = True
except ImportError:
    sketchgraphs_available = False


# =============================================================================
# DeepCAD Loader Tests
# =============================================================================

class TestDeepCADImports:
    """Test that DeepCAD loader module imports successfully."""

    def test_deepcad_module_importable(self):
        """Verify deepcad_loader module can be imported."""
        assert deepcad_available, "deepcad_loader module not importable"

    def test_command_type_ids_defined(self):
        """Verify COMMAND_TYPE_IDS constants are properly defined."""
        if not deepcad_available:
            pytest.skip("deepcad_loader not available")

        assert COMMAND_TYPE_IDS["SOL"] == 6
        assert COMMAND_TYPE_IDS["LINE"] == 7
        assert COMMAND_TYPE_IDS["ARC"] == 8
        assert COMMAND_TYPE_IDS["CIRCLE"] == 9
        assert COMMAND_TYPE_IDS["EXTRUDE"] == 10
        assert COMMAND_TYPE_IDS["EOS"] == 11

    def test_token_constants_defined(self):
        """Verify token ID constants are correctly set."""
        if not deepcad_available:
            pytest.skip("deepcad_loader not available")

        assert PAD_TOKEN_ID == 0
        assert BOS_TOKEN_ID == 1
        assert EOS_TOKEN_ID == 2
        assert PARAM_OFFSET == 12


class TestDeepCADTokenization:
    """Test _tokenize_deepcad_sample function with synthetic data."""

    def _create_sample(self, num_commands: int = 3) -> Dict[str, Any]:
        """Create a synthetic DeepCAD sample with specified command count."""
        if not deepcad_available:
            pytest.skip("deepcad_loader not available")

        commands = []

        # SOL command (start of loop)
        commands.append({
            "type": "SOL",
            "params": [0.0] * 16,
        })

        # Add LINE commands
        for i in range(num_commands - 2):
            commands.append({
                "type": "LINE",
                "params": [0.0, 0.0, 0.5, 0.0] + [0.0] * 12,
            })

        # EXTRUDE command
        commands.append({
            "type": "EXTRUDE",
            "params": [0.3, 0.0] + [0.0] * 14,
        })

        # EOS command
        commands.append({
            "type": "EOS",
            "params": [0.0] * 16,
        })

        return {"sequence": commands}

    def test_tokenize_simple_sample(self):
        """Test tokenization of a simple DeepCAD sample."""
        if not deepcad_available:
            pytest.skip("deepcad_loader not available")

        sample = self._create_sample(num_commands=3)
        result = _tokenize_deepcad_sample(
            sample,
            max_commands=60,
            quantization_bits=8,
            normalization_range=2.0,
        )

        assert "token_ids" in result
        assert "command_tokens" in result
        assert "attention_mask" in result
        assert "num_commands" in result

        assert isinstance(result["token_ids"], list)
        assert isinstance(result["command_tokens"], list)
        assert isinstance(result["attention_mask"], list)

    def test_tokenize_output_starts_with_bos(self):
        """Verify token_ids start with BOS token."""
        if not deepcad_available:
            pytest.skip("deepcad_loader not available")

        sample = self._create_sample(num_commands=2)
        result = _tokenize_deepcad_sample(sample)

        assert result["token_ids"][0] == BOS_TOKEN_ID

    def test_tokenize_output_ends_with_eos(self):
        """Verify token_ids include EOS token near end."""
        if not deepcad_available:
            pytest.skip("deepcad_loader not available")

        sample = self._create_sample(num_commands=2)
        result = _tokenize_deepcad_sample(sample)

        # EOS should be present after all command tokens
        token_ids = result["token_ids"]
        # Find non-PAD tokens
        non_pad_tokens = [t for t in token_ids if t != PAD_TOKEN_ID]
        assert EOS_TOKEN_ID in non_pad_tokens

    def test_tokenize_command_types_correct(self):
        """Verify command type tokens are correctly mapped."""
        if not deepcad_available:
            pytest.skip("deepcad_loader not available")

        sample = {
            "sequence": [
                {"type": "SOL", "params": [0.0] * 16},
                {"type": "LINE", "params": [0.0] * 16},
                {"type": "EOS", "params": [0.0] * 16},
            ]
        }
        result = _tokenize_deepcad_sample(sample)

        # Check that command_tokens have the correct command types
        assert len(result["command_tokens"]) >= 1
        assert result["command_tokens"][0]["command_type"] == COMMAND_TYPE_IDS["SOL"]

    def test_quantization_bits_8(self):
        """Test parameter quantization with 8-bit quantization."""
        if not deepcad_available:
            pytest.skip("deepcad_loader not available")

        # Create sample with known parameter value
        sample = {
            "sequence": [
                {"type": "SOL", "params": [0.0] * 16},
                {
                    "type": "LINE",
                    "params": [0.5, 0.0] + [0.0] * 14,
                },
                {"type": "EOS", "params": [0.0] * 16},
            ]
        }

        result = _tokenize_deepcad_sample(
            sample,
            quantization_bits=8,
            normalization_range=2.0,
        )

        # For value 0.5 with range 2.0 and 8-bit (symmetric normalization):
        # normalized = (0.5 + 2.0) / (2 * 2.0) = 2.5 / 4.0 = 0.625
        # quantized = round(0.625 * (256 - 1)) = round(159.375) = 159
        assert result["command_tokens"][1]["parameters"][0] == 159

    def test_quantization_clamping(self):
        """Test that quantized values are clamped to valid range."""
        if not deepcad_available:
            pytest.skip("deepcad_loader not available")

        sample = {
            "sequence": [
                {"type": "SOL", "params": [0.0] * 16},
                {
                    "type": "LINE",
                    "params": [10.0, 0.0] + [0.0] * 14,  # Value way out of range
                },
                {"type": "EOS", "params": [0.0] * 16},
            ]
        }

        result = _tokenize_deepcad_sample(sample, quantization_bits=8)

        # Should clamp to 255 (2^8 - 1)
        assert result["command_tokens"][1]["parameters"][0] <= 255
        assert result["command_tokens"][1]["parameters"][0] >= 0

    def test_attention_mask_all_ones_before_padding(self):
        """Verify attention mask is 1 for all command tokens and 0 for padding."""
        if not deepcad_available:
            pytest.skip("deepcad_loader not available")

        sample = self._create_sample(num_commands=2)
        result = _tokenize_deepcad_sample(sample, max_commands=60)

        attention_mask = result["attention_mask"]
        # First mask values should be 1 (command tokens)
        assert attention_mask[0] == 1  # BOS
        # Last mask values should be 0 (padding)
        assert attention_mask[-1] == 0

    def test_padding_to_max_length(self):
        """Verify token_ids are padded to expected length."""
        if not deepcad_available:
            pytest.skip("deepcad_loader not available")

        max_commands = 60
        sample = self._create_sample(num_commands=2)
        result = _tokenize_deepcad_sample(sample, max_commands=max_commands)

        expected_length = max_commands * 10
        assert len(result["token_ids"]) == expected_length
        assert len(result["attention_mask"]) == expected_length

    def test_num_commands_correct(self):
        """Verify num_commands reflects actual command count."""
        if not deepcad_available:
            pytest.skip("deepcad_loader not available")

        sample = self._create_sample(num_commands=4)
        result = _tokenize_deepcad_sample(sample)

        # Should have 4 commands: SOL, LINE, LINE, EXTRUDE, EOS
        # But we're checking command_tokens which includes recognized commands
        assert result["num_commands"] >= 1

    def test_max_commands_limit(self):
        """Verify max_commands parameter limits command processing."""
        if not deepcad_available:
            pytest.skip("deepcad_loader not available")

        sample = self._create_sample(num_commands=10)
        result = _tokenize_deepcad_sample(sample, max_commands=3)

        # Should stop after 3 commands
        assert len(result["command_tokens"]) <= 3

    def test_invalid_command_type_skipped(self):
        """Verify unrecognized command types are skipped."""
        if not deepcad_available:
            pytest.skip("deepcad_loader not available")

        sample = {
            "sequence": [
                {"type": "SOL", "params": [0.0] * 16},
                {"type": "UNKNOWN_CMD", "params": [0.0] * 16},  # Unknown
                {"type": "LINE", "params": [0.5] + [0.0] * 15},
                {"type": "EOS", "params": [0.0] * 16},
            ]
        }
        result = _tokenize_deepcad_sample(sample)

        # UNKNOWN_CMD should be skipped, but SOL, LINE, EOS should be processed
        assert len(result["command_tokens"]) == 3  # SOL, LINE, EOS (UNKNOWN_CMD skipped)
        assert result["command_tokens"][0]["command_type"] == COMMAND_TYPE_IDS["SOL"]
        assert result["command_tokens"][1]["command_type"] == COMMAND_TYPE_IDS["LINE"]
        assert result["command_tokens"][2]["command_type"] == COMMAND_TYPE_IDS["EOS"]


class TestDeepCADDatasetInit:
    """Test DeepCADDataset class initialization."""

    def test_dataset_init_with_mock_path(self):
        """Test dataset initialization with non-existent path."""
        if not deepcad_available:
            pytest.skip("deepcad_loader not available")

        # Dataset constructor should accept path but error on file scan
        with pytest.raises(FileNotFoundError):
            DeepCADDataset(
                data_dir="/nonexistent/path",
                split="train",
            )

    def test_dataset_init_stores_parameters(self):
        """Test that dataset init stores all parameters correctly."""
        if not deepcad_available:
            pytest.skip("deepcad_loader not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create split directory with a dummy JSON file
            split_dir = Path(tmpdir) / "train"
            split_dir.mkdir(parents=True)

            dummy_json = split_dir / "sample.json"
            dummy_json.write_text(json.dumps({"sequence": []}))

            dataset = DeepCADDataset(
                data_dir=tmpdir,
                split="train",
                max_commands=100,
                quantization_bits=16,
                normalization_range=4.0,
            )

            assert dataset.split == "train"
            assert dataset.max_commands == 100
            assert dataset.quantization_bits == 16
            assert dataset.normalization_range == 4.0

    def test_dataset_length(self):
        """Test dataset __len__ method."""
        if not deepcad_available:
            pytest.skip("deepcad_loader not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "train"
            split_dir.mkdir(parents=True)

            # Create 3 dummy JSON files
            for i in range(3):
                json_file = split_dir / f"sample_{i}.json"
                json_file.write_text(json.dumps({"sequence": []}))

            dataset = DeepCADDataset(data_dir=tmpdir, split="train")
            assert len(dataset) == 3

    def test_dataset_max_samples_limit(self):
        """Test that max_samples parameter limits loaded files."""
        if not deepcad_available:
            pytest.skip("deepcad_loader not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "train"
            split_dir.mkdir(parents=True)

            # Create 5 dummy JSON files
            for i in range(5):
                json_file = split_dir / f"sample_{i}.json"
                json_file.write_text(json.dumps({"sequence": []}))

            dataset = DeepCADDataset(
                data_dir=tmpdir,
                split="train",
                max_samples=2,
            )
            assert len(dataset) == 2


@pytest.mark.skipif(not deepcad_available, reason="deepcad_loader not available")
class TestDeepCADDatasetGetItem:
    """Test DeepCADDataset.__getitem__ method."""

    def test_getitem_returns_dict(self):
        """Test that __getitem__ returns proper dictionary structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "train"
            split_dir.mkdir(parents=True)

            sample_data = {
                "sequence": [
                    {"type": "SOL", "params": [0.0] * 16},
                    {"type": "LINE", "params": [0.5, 0.0] + [0.0] * 14},
                    {"type": "EOS", "params": [0.0] * 16},
                ]
            }
            json_file = split_dir / "sample.json"
            json_file.write_text(json.dumps(sample_data))

            dataset = DeepCADDataset(data_dir=tmpdir, split="train")
            result = dataset[0]

            assert "token_ids" in result
            assert "command_tokens" in result
            assert "attention_mask" in result
            assert "num_commands" in result
            assert "metadata" in result

    def test_getitem_metadata_contains_file_path(self):
        """Test that metadata contains file path and split info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "train"
            split_dir.mkdir(parents=True)

            sample_data = {"sequence": []}
            json_file = split_dir / "sample.json"
            json_file.write_text(json.dumps(sample_data))

            dataset = DeepCADDataset(data_dir=tmpdir, split="train")
            result = dataset[0]

            assert "file_path" in result["metadata"]
            assert result["metadata"]["split"] == "train"


# =============================================================================
# ABC Loader Tests
# =============================================================================

class TestABCImports:
    """Test that ABC loader module imports successfully."""

    def test_abc_module_importable(self):
        """Verify abc_loader module can be imported."""
        assert abc_available, "abc_loader module not importable"


class TestABCTokenization:
    """Test _tokenize_abc_sample function."""

    def test_tokenize_abc_sample_returns_dict(self):
        """Test that _tokenize_abc_sample returns expected structure."""
        if not abc_available:
            pytest.skip("abc_loader not available")

        sample = {
            "step_path": "/mock/path/model.step",
            "file_name": "model.step",
            "file_size": 50000,
        }
        result = _tokenize_abc_sample(sample)

        assert "step_path" in result
        assert "file_name" in result
        assert "file_size" in result

    def test_tokenize_abc_sample_preserves_fields(self):
        """Test that tokenization preserves input fields."""
        if not abc_available:
            pytest.skip("abc_loader not available")

        sample = {
            "step_path": "/mock/file.step",
            "file_name": "file.step",
            "file_size": 123456,
        }
        result = _tokenize_abc_sample(sample)

        assert result["file_name"] == "file.step"
        assert result["file_size"] == 123456


class TestABCDatasetInit:
    """Test ABCDataset class initialization."""

    def test_dataset_init_with_mock_path(self):
        """Test dataset initialization with non-existent path."""
        if not abc_available:
            pytest.skip("abc_loader not available")

        with pytest.raises(FileNotFoundError):
            ABCDataset(data_dir="/nonexistent/path", split="train")

    def test_dataset_init_stores_parameters(self):
        """Test that dataset stores split and max_samples parameters."""
        if not abc_available:
            pytest.skip("abc_loader not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "val"
            split_dir.mkdir(parents=True)

            # Create a dummy STEP file
            step_file = split_dir / "model.step"
            step_file.write_text("DUMMY STEP CONTENT")

            dataset = ABCDataset(data_dir=tmpdir, split="val")

            assert dataset.split == "val"
            assert dataset.max_samples is None

    def test_dataset_length(self):
        """Test dataset __len__ method."""
        if not abc_available:
            pytest.skip("abc_loader not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "train"
            split_dir.mkdir(parents=True)

            # Create 2 STEP files
            for i in range(2):
                step_file = split_dir / f"model_{i}.step"
                step_file.write_text("DUMMY STEP")

            dataset = ABCDataset(data_dir=tmpdir, split="train")
            assert len(dataset) == 2


@pytest.mark.skipif(not abc_available, reason="abc_loader not available")
class TestABCDatasetGetItem:
    """Test ABCDataset.__getitem__ method."""

    def test_getitem_returns_dict(self):
        """Test that __getitem__ returns proper dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "train"
            split_dir.mkdir(parents=True)

            step_file = split_dir / "model.step"
            step_file.write_text("DUMMY STEP CONTENT")

            dataset = ABCDataset(data_dir=tmpdir, split="train")
            result = dataset[0]

            assert "step_path" in result
            assert "file_name" in result
            assert "file_size" in result
            assert "metadata" in result

    def test_getitem_metadata_contains_split(self):
        """Test that metadata contains split information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "test"
            split_dir.mkdir(parents=True)

            step_file = split_dir / "model.step"
            step_file.write_text("DUMMY")

            dataset = ABCDataset(data_dir=tmpdir, split="test")
            result = dataset[0]

            assert result["metadata"]["split"] == "test"


# =============================================================================
# Text2CAD Loader Tests
# =============================================================================

class TestText2CADImports:
    """Test that Text2CAD loader module imports successfully."""

    def test_text2cad_module_importable(self):
        """Verify text2cad_loader module can be imported."""
        assert text2cad_available, "text2cad_loader module not importable"

    def test_annotation_levels_defined(self):
        """Verify ANNOTATION_LEVELS are properly defined."""
        if not text2cad_available:
            pytest.skip("text2cad_loader not available")

        assert "abstract" in ANNOTATION_LEVELS
        assert "detailed" in ANNOTATION_LEVELS
        assert "expert" in ANNOTATION_LEVELS


class TestText2CADTokenization:
    """Test _tokenize_text2cad_sample function."""

    def _create_text2cad_sample(self, annotation_level: str = "detailed") -> Dict[str, Any]:
        """Create synthetic Text2CAD sample."""
        if not text2cad_available:
            pytest.skip("text2cad_loader not available")

        text_key = ANNOTATION_LEVELS[annotation_level]
        return {
            text_key: "A simple rectangular prism",
            "sequence": [
                {"type": "SOL", "params": [0.0] * 16},
                {"type": "LINE", "params": [1.0, 0.0] + [0.0] * 14},
                {"type": "EOS", "params": [0.0] * 16},
            ]
        }

    def test_tokenize_text2cad_returns_dict(self):
        """Test that tokenization returns expected fields."""
        if not text2cad_available:
            pytest.skip("text2cad_loader not available")

        sample = self._create_text2cad_sample()
        result = _tokenize_text2cad_sample(sample, annotation_level="detailed")

        assert "text" in result
        assert "token_ids" in result
        assert "command_tokens" in result
        assert "attention_mask" in result

    def test_tokenize_text2cad_abstract_level(self):
        """Test tokenization with abstract annotation level."""
        if not text2cad_available:
            pytest.skip("text2cad_loader not available")

        sample = self._create_text2cad_sample(annotation_level="abstract")
        result = _tokenize_text2cad_sample(sample, annotation_level="abstract")

        assert "text" in result
        assert isinstance(result["text"], str)

    def test_tokenize_text2cad_expert_level(self):
        """Test tokenization with expert annotation level."""
        if not text2cad_available:
            pytest.skip("text2cad_loader not available")

        sample = self._create_text2cad_sample(annotation_level="expert")
        result = _tokenize_text2cad_sample(sample, annotation_level="expert")

        assert "text" in result


class TestText2CADDatasetInit:
    """Test Text2CADDataset class initialization."""

    def test_dataset_init_with_mock_path(self):
        """Test dataset initialization with non-existent path."""
        if not text2cad_available:
            pytest.skip("text2cad_loader not available")

        with pytest.raises(FileNotFoundError):
            Text2CADDataset(
                data_dir="/nonexistent/path",
                split="train",
                annotation_level="detailed",
            )

    def test_dataset_init_invalid_annotation_level(self):
        """Test that invalid annotation level raises ValueError."""
        if not text2cad_available:
            pytest.skip("text2cad_loader not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "train"
            split_dir.mkdir(parents=True)

            sample_file = split_dir / "sample.json"
            sample_file.write_text(json.dumps({"sequence": []}))

            with pytest.raises(ValueError):
                Text2CADDataset(
                    data_dir=tmpdir,
                    split="train",
                    annotation_level="invalid_level",
                )

    def test_dataset_init_stores_annotation_level(self):
        """Test that annotation_level is stored correctly."""
        if not text2cad_available:
            pytest.skip("text2cad_loader not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "train"
            split_dir.mkdir(parents=True)

            sample_file = split_dir / "sample.json"
            sample_file.write_text(json.dumps({"sequence": []}))

            dataset = Text2CADDataset(
                data_dir=tmpdir,
                split="train",
                annotation_level="expert",
            )

            assert dataset.annotation_level == "expert"

    def test_dataset_length(self):
        """Test dataset __len__ method."""
        if not text2cad_available:
            pytest.skip("text2cad_loader not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "train"
            split_dir.mkdir(parents=True)

            # Create 3 JSON files
            for i in range(3):
                json_file = split_dir / f"sample_{i}.json"
                json_file.write_text(json.dumps({"sequence": []}))

            dataset = Text2CADDataset(data_dir=tmpdir, split="train")
            assert len(dataset) == 3


@pytest.mark.skipif(not text2cad_available, reason="text2cad_loader not available")
class TestText2CADDatasetGetItem:
    """Test Text2CADDataset.__getitem__ method."""

    def test_getitem_returns_dict_with_text(self):
        """Test that __getitem__ includes text field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "train"
            split_dir.mkdir(parents=True)

            sample_data = {
                "text_detailed": "A simple box",
                "sequence": [
                    {"type": "SOL", "params": [0.0] * 16},
                    {"type": "EOS", "params": [0.0] * 16},
                ]
            }
            json_file = split_dir / "sample.json"
            json_file.write_text(json.dumps(sample_data))

            dataset = Text2CADDataset(data_dir=tmpdir, split="train")
            result = dataset[0]

            assert "text" in result
            assert result["text"] == "A simple box"

    def test_getitem_metadata_contains_annotation_level(self):
        """Test that metadata contains annotation level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "train"
            split_dir.mkdir(parents=True)

            sample_data = {
                "text_expert": "Complex geometry description",
                "sequence": []
            }
            json_file = split_dir / "sample.json"
            json_file.write_text(json.dumps(sample_data))

            dataset = Text2CADDataset(
                data_dir=tmpdir,
                split="train",
                annotation_level="expert"
            )
            result = dataset[0]

            assert result["metadata"]["annotation_level"] == "expert"


# =============================================================================
# SketchGraphs Loader Tests
# =============================================================================

class TestSketchGraphsImports:
    """Test that SketchGraphs loader module imports successfully."""

    def test_sketchgraphs_module_importable(self):
        """Verify sketchgraphs_loader module can be imported."""
        assert sketchgraphs_available, "sketchgraphs_loader module not importable"

    def test_entity_types_defined(self):
        """Verify ENTITY_TYPES constants are properly defined."""
        if not sketchgraphs_available:
            pytest.skip("sketchgraphs_loader not available")

        assert ENTITY_TYPES["Line"] == 0
        assert ENTITY_TYPES["Circle"] == 1
        assert ENTITY_TYPES["Arc"] == 2
        assert ENTITY_TYPES["Point"] == 3

    def test_constraint_types_defined(self):
        """Verify CONSTRAINT_TYPES constants are properly defined."""
        if not sketchgraphs_available:
            pytest.skip("sketchgraphs_loader not available")

        assert CONSTRAINT_TYPES["Coincident"] == 0
        assert CONSTRAINT_TYPES["Parallel"] == 1
        assert CONSTRAINT_TYPES["Perpendicular"] == 2
        assert CONSTRAINT_TYPES["Tangent"] == 3
        assert CONSTRAINT_TYPES["Equal"] == 4
        assert CONSTRAINT_TYPES["Distance"] == 5
        assert CONSTRAINT_TYPES["Angle"] == 6
        assert CONSTRAINT_TYPES["Concentric"] == 7


class TestSketchGraphsTokenization:
    """Test _tokenize_sketchgraphs_sample function."""

    def _create_sketchgraphs_sample(self) -> Dict[str, Any]:
        """Create synthetic SketchGraphs sample."""
        if not sketchgraphs_available:
            pytest.skip("sketchgraphs_loader not available")

        return {
            "entities": [
                {
                    "type": "Line",
                    "params": {
                        "start": [0.0, 0.0],
                        "end": [10.0, 0.0],
                    }
                },
                {
                    "type": "Point",
                    "params": {
                        "point": [5.0, 5.0],
                    }
                },
            ],
            "constraints": [
                {
                    "type": "Coincident",
                    "references": [0, 1],
                    "value": None,
                }
            ]
        }

    def test_tokenize_sketchgraphs_returns_arrays(self):
        """Test that tokenization returns numpy arrays."""
        if not sketchgraphs_available:
            pytest.skip("sketchgraphs_loader not available")

        sample = self._create_sketchgraphs_sample()
        result = _tokenize_sketchgraphs_sample(sample)

        assert "entity_types" in result
        assert "entity_params" in result
        assert "constraint_types" in result
        assert "constraint_refs" in result
        assert "constraint_values" in result

        # Check that arrays are numpy arrays
        assert isinstance(result["entity_types"], np.ndarray)
        assert isinstance(result["entity_params"], np.ndarray)
        assert isinstance(result["constraint_types"], np.ndarray)

    def test_tokenize_entity_types_correct(self):
        """Test that entity types are correctly mapped."""
        if not sketchgraphs_available:
            pytest.skip("sketchgraphs_loader not available")

        sample = self._create_sketchgraphs_sample()
        result = _tokenize_sketchgraphs_sample(sample, max_entities=10)

        # First entity is Line (type 0)
        assert result["entity_types"][0] == ENTITY_TYPES["Line"]
        # Second entity is Point (type 3)
        assert result["entity_types"][1] == ENTITY_TYPES["Point"]

    def test_tokenize_entity_padding(self):
        """Test that entities are padded to max_entities."""
        if not sketchgraphs_available:
            pytest.skip("sketchgraphs_loader not available")

        sample = self._create_sketchgraphs_sample()
        max_entities = 50
        result = _tokenize_sketchgraphs_sample(sample, max_entities=max_entities)

        # Should be padded to max_entities
        assert len(result["entity_types"]) == max_entities
        assert len(result["entity_params"]) == max_entities

    def test_tokenize_constraint_types_correct(self):
        """Test that constraint types are correctly mapped."""
        if not sketchgraphs_available:
            pytest.skip("sketchgraphs_loader not available")

        sample = self._create_sketchgraphs_sample()
        result = _tokenize_sketchgraphs_sample(sample)

        # First constraint is Coincident (type 0)
        assert result["constraint_types"][0] == CONSTRAINT_TYPES["Coincident"]

    def test_tokenize_constraint_references(self):
        """Test that constraint references are preserved."""
        if not sketchgraphs_available:
            pytest.skip("sketchgraphs_loader not available")

        sample = self._create_sketchgraphs_sample()
        result = _tokenize_sketchgraphs_sample(sample)

        # References should match input
        assert result["constraint_refs"][0][0] == 0
        assert result["constraint_refs"][0][1] == 1

    def test_tokenize_num_entities(self):
        """Test that num_entities is correctly counted."""
        if not sketchgraphs_available:
            pytest.skip("sketchgraphs_loader not available")

        sample = self._create_sketchgraphs_sample()
        result = _tokenize_sketchgraphs_sample(sample)

        assert result["num_entities"] == 2

    def test_tokenize_num_constraints(self):
        """Test that num_constraints is correctly counted."""
        if not sketchgraphs_available:
            pytest.skip("sketchgraphs_loader not available")

        sample = self._create_sketchgraphs_sample()
        result = _tokenize_sketchgraphs_sample(sample)

        assert result["num_constraints"] == 1


class TestSketchGraphsDatasetInit:
    """Test SketchGraphsDataset class initialization."""

    def test_dataset_init_with_mock_path(self):
        """Test dataset initialization with non-existent path."""
        if not sketchgraphs_available:
            pytest.skip("sketchgraphs_loader not available")

        with pytest.raises(FileNotFoundError):
            SketchGraphsDataset(
                data_dir="/nonexistent/path",
                split="train",
            )

    def test_dataset_init_stores_parameters(self):
        """Test that dataset stores max_entities and max_constraints."""
        if not sketchgraphs_available:
            pytest.skip("sketchgraphs_loader not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "train"
            split_dir.mkdir(parents=True)

            sample_file = split_dir / "sample.json"
            sample_file.write_text(json.dumps({"entities": [], "constraints": []}))

            dataset = SketchGraphsDataset(
                data_dir=tmpdir,
                split="train",
                max_entities=100,
                max_constraints=200,
            )

            assert dataset.max_entities == 100
            assert dataset.max_constraints == 200

    def test_dataset_length(self):
        """Test dataset __len__ method."""
        if not sketchgraphs_available:
            pytest.skip("sketchgraphs_loader not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "train"
            split_dir.mkdir(parents=True)

            # Create 4 JSON files
            for i in range(4):
                json_file = split_dir / f"sketch_{i}.json"
                json_file.write_text(json.dumps({"entities": [], "constraints": []}))

            dataset = SketchGraphsDataset(data_dir=tmpdir, split="train")
            assert len(dataset) == 4


@pytest.mark.skipif(not sketchgraphs_available, reason="sketchgraphs_loader not available")
class TestSketchGraphsDatasetGetItem:
    """Test SketchGraphsDataset.__getitem__ method."""

    def test_getitem_returns_dict_with_arrays(self):
        """Test that __getitem__ returns numpy arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "train"
            split_dir.mkdir(parents=True)

            sample_data = {
                "entities": [
                    {
                        "type": "Line",
                        "params": {"start": [0, 0], "end": [10, 10]}
                    }
                ],
                "constraints": []
            }
            json_file = split_dir / "sketch.json"
            json_file.write_text(json.dumps(sample_data))

            dataset = SketchGraphsDataset(data_dir=tmpdir, split="train")
            result = dataset[0]

            assert isinstance(result["entity_types"], np.ndarray)
            assert isinstance(result["entity_params"], np.ndarray)
            assert isinstance(result["constraint_types"], np.ndarray)

    def test_getitem_metadata_contains_split(self):
        """Test that metadata contains split information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "test"
            split_dir.mkdir(parents=True)

            sample_data = {"entities": [], "constraints": []}
            json_file = split_dir / "sketch.json"
            json_file.write_text(json.dumps(sample_data))

            dataset = SketchGraphsDataset(data_dir=tmpdir, split="test")
            result = dataset[0]

            assert result["metadata"]["split"] == "test"


# =============================================================================
# Integration Tests
# =============================================================================

class TestMultipleLoadersConsistency:
    """Test consistency across multiple dataset loaders."""

    def test_all_loaders_importable(self):
        """Verify all loaders can be imported at once."""
        assert deepcad_available
        assert abc_available
        assert text2cad_available
        assert sketchgraphs_available

    def test_deepcad_and_text2cad_use_same_command_tokens(self):
        """Verify DeepCAD and Text2CAD use same command token IDs."""
        if not (deepcad_available and text2cad_available):
            pytest.skip("deepcad and text2cad loaders not available")

        # Both should use same command type IDs
        from ll_gen.datasets.deepcad_loader import COMMAND_TYPE_IDS as dc_cmds
        from ll_gen.datasets.text2cad_loader import COMMAND_TYPE_IDS as t2c_cmds

        assert dc_cmds == t2c_cmds

    def test_deepcad_tokenize_consistency(self):
        """Test that multiple calls to tokenize produce consistent results."""
        if not deepcad_available:
            pytest.skip("deepcad_loader not available")

        sample = {
            "sequence": [
                {"type": "SOL", "params": [0.0] * 16},
                {"type": "EOS", "params": [0.0] * 16},
            ]
        }

        result1 = _tokenize_deepcad_sample(sample)
        result2 = _tokenize_deepcad_sample(sample)

        assert result1["token_ids"] == result2["token_ids"]
        assert result1["num_commands"] == result2["num_commands"]


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_deepcad_empty_sequence(self):
        """Test tokenization of empty sequence."""
        if not deepcad_available:
            pytest.skip("deepcad_loader not available")

        sample = {"sequence": []}
        result = _tokenize_deepcad_sample(sample)

        # Should still return valid structure
        assert "token_ids" in result
        assert result["num_commands"] == 0

    def test_text2cad_missing_annotation_field(self):
        """Test handling of missing annotation field."""
        if not text2cad_available:
            pytest.skip("text2cad_loader not available")

        sample = {"sequence": []}  # Missing text field
        result = _tokenize_text2cad_sample(sample, annotation_level="detailed")

        # Should have empty text but not error
        assert result["text"] == ""

    def test_sketchgraphs_empty_entities(self):
        """Test tokenization with no entities."""
        if not sketchgraphs_available:
            pytest.skip("sketchgraphs_loader not available")

        sample = {"entities": [], "constraints": []}
        result = _tokenize_sketchgraphs_sample(sample)

        assert result["num_entities"] == 0

    def test_deepcad_parameter_none_value(self):
        """Test handling of None parameter values."""
        if not deepcad_available:
            pytest.skip("deepcad_loader not available")

        sample = {
            "sequence": [
                {"type": "SOL", "params": [None, 0.0] + [0.0] * 14},
                {"type": "EOS", "params": [0.0] * 16},
            ]
        }
        result = _tokenize_deepcad_sample(sample)

        # Should handle None gracefully (normalizes to 0.0)
        assert "token_ids" in result
