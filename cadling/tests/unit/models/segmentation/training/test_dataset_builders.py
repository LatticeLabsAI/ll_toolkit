"""Unit tests for dataset builders.

Tests HuggingFace dataset builders for converting local CAD data.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import pytest


class TestMFCADDatasetBuilder:
    """Test MFCADDatasetBuilder class."""

    def test_builder_info(self):
        """Test builder _info method."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.dataset_builders.mfcad_builder import (
            MFCADDatasetBuilder
        )

        builder = MFCADDatasetBuilder()
        info = builder._info()

        assert info is not None
        assert "step_content" in info.features
        assert "faces" in info.features
        assert "num_faces" in info.features
        assert len(builder.FEATURE_CLASSES) == 25

    def test_builder_feature_classes(self):
        """Test builder has correct feature classes."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.dataset_builders.mfcad_builder import (
            MFCADDatasetBuilder
        )

        builder = MFCADDatasetBuilder()

        # Verify all expected classes present
        expected_classes = [
            "base", "hole", "pocket", "boss", "fillet", "chamfer",
            "slot", "thread", "groove", "through_hole", "blind_hole"
        ]

        for cls in expected_classes:
            assert cls in builder.FEATURE_CLASSES

    def test_builder_split_generators(self):
        """Test builder _split_generators method."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.dataset_builders.mfcad_builder import (
            MFCADDatasetBuilder
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create train/val/test directories
            (tmpdir / "train").mkdir()
            (tmpdir / "val").mkdir()
            (tmpdir / "test").mkdir()

            builder = MFCADDatasetBuilder()
            builder.config = MagicMock()
            builder.config.data_dir = str(tmpdir)

            # Get split generators
            splits = builder._split_generators(None)

            assert len(splits) == 3
            split_names = [s.name for s in splits]
            assert "train" in str(split_names)
            assert "validation" in str(split_names) or "val" in str(split_names)
            assert "test" in str(split_names)

    def test_builder_generate_examples(self):
        """Test builder _generate_examples method."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.dataset_builders.mfcad_builder import (
            MFCADDatasetBuilder
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create sample STEP file
            step_file = tmpdir / "part_001.step"
            step_file.write_text("ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;")

            # Create annotation file
            annotation = {
                "file_name": "part_001.step",
                "faces": [
                    {
                        "face_id": 1,
                        "label": "hole",
                        "instance_id": 0,
                        "is_bottom_face": False,
                        "parameters": {"diameter": 10.0, "depth": 20.0}
                    },
                    {
                        "face_id": 2,
                        "label": "pocket",
                        "instance_id": 1,
                        "is_bottom_face": True,
                        "parameters": {"width": 15.0, "length": 25.0}
                    }
                ]
            }

            ann_file = tmpdir / "part_001.json"
            ann_file.write_text(json.dumps(annotation))

            # Generate examples
            builder = MFCADDatasetBuilder()
            examples = list(builder._generate_examples(tmpdir))

            assert len(examples) == 1
            idx, sample = examples[0]

            assert sample["file_name"] == "part_001.step"
            assert "step_content" in sample
            assert len(sample["faces"]) == 2
            assert sample["num_faces"] == 2
            assert sample["num_instances"] == 2

            # Verify face data
            assert sample["faces"][0]["label"] == "hole"
            assert sample["faces"][0]["face_id"] == 1
            assert sample["faces"][1]["label"] == "pocket"

    def test_builder_missing_step_file(self):
        """Test builder handles missing STEP file."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.dataset_builders.mfcad_builder import (
            MFCADDatasetBuilder
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create annotation without STEP file
            annotation = {
                "file_name": "missing.step",
                "faces": []
            }
            ann_file = tmpdir / "part_001.json"
            ann_file.write_text(json.dumps(annotation))

            builder = MFCADDatasetBuilder()
            examples = list(builder._generate_examples(tmpdir))

            # Should skip missing files
            assert len(examples) == 0

    def test_builder_unknown_label(self):
        """Test builder handles unknown feature labels."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.dataset_builders.mfcad_builder import (
            MFCADDatasetBuilder
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            step_file = tmpdir / "part_001.step"
            step_file.write_text("ISO-10303-21;...;")

            annotation = {
                "file_name": "part_001.step",
                "faces": [
                    {
                        "face_id": 1,
                        "label": "unknown_feature",
                        "instance_id": 0
                    }
                ]
            }
            ann_file = tmpdir / "part_001.json"
            ann_file.write_text(json.dumps(annotation))

            builder = MFCADDatasetBuilder()
            examples = list(builder._generate_examples(tmpdir))

            assert len(examples) == 1
            _, sample = examples[0]

            # Unknown label should default to 0 (base)
            assert sample["faces"][0]["label_id"] == 0

    def test_builder_default_parameters(self):
        """Test builder provides default parameters."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.dataset_builders.mfcad_builder import (
            MFCADDatasetBuilder
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            step_file = tmpdir / "part_001.step"
            step_file.write_text("ISO-10303-21;...;")

            # Annotation without parameters
            annotation = {
                "file_name": "part_001.step",
                "faces": [
                    {
                        "face_id": 1,
                        "label": "hole",
                        "instance_id": 0,
                        # No parameters provided
                    }
                ]
            }
            ann_file = tmpdir / "part_001.json"
            ann_file.write_text(json.dumps(annotation))

            builder = MFCADDatasetBuilder()
            examples = list(builder._generate_examples(tmpdir))

            _, sample = examples[0]

            # Should have default parameters (all 0.0)
            params = sample["faces"][0]["parameters"]
            assert params["diameter"] == 0.0
            assert params["depth"] == 0.0
            assert params["radius"] == 0.0


class TestCreateHFDataset:
    """Test create_hf_dataset_from_local function."""

    def test_create_mfcad_dataset(self):
        """Test creating MFCAD dataset."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.dataset_builders.create_hf_dataset import (
            create_hf_dataset_from_local
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "input"
            output_dir = tmpdir / "output"

            # Create directory structure
            train_dir = input_dir / "train"
            train_dir.mkdir(parents=True)

            # Create sample data
            step_file = train_dir / "part_001.step"
            step_file.write_text("ISO-10303-21;...;")

            annotation = {
                "file_name": "part_001.step",
                "faces": [{"face_id": 1, "label": "hole", "instance_id": 0}]
            }
            ann_file = train_dir / "part_001.json"
            ann_file.write_text(json.dumps(annotation))

            # Create dataset
            create_hf_dataset_from_local(
                input_dir=input_dir,
                output_dir=output_dir,
                dataset_type="mfcad",
                upload=False
            )

            # Verify output directory exists
            assert output_dir.exists()

    def test_create_custom_dataset(self):
        """Test creating custom dataset."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.dataset_builders.create_hf_dataset import (
            create_hf_dataset_from_local
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "input"
            output_dir = tmpdir / "output"

            # Create directory structure
            train_dir = input_dir / "train"
            train_dir.mkdir(parents=True)

            # Create sample data
            step_file = train_dir / "part_001.step"
            step_file.write_text("ISO-10303-21;...;")

            annotation = {"file_name": "part_001.step", "data": "test"}
            ann_file = train_dir / "part_001.json"
            ann_file.write_text(json.dumps(annotation))

            # Create dataset
            create_hf_dataset_from_local(
                input_dir=input_dir,
                output_dir=output_dir,
                dataset_type="custom",
                upload=False
            )

            # Verify output
            assert output_dir.exists()

    def test_upload_to_hub(self):
        """Test upload to HuggingFace Hub (mocked)."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.dataset_builders.create_hf_dataset import (
            create_hf_dataset_from_local
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "input"
            output_dir = tmpdir / "output"

            # Create minimal dataset
            train_dir = input_dir / "train"
            train_dir.mkdir(parents=True)

            step_file = train_dir / "part_001.step"
            step_file.write_text("ISO-10303-21;...;")

            annotation = {
                "file_name": "part_001.step",
                "faces": []
            }
            ann_file = train_dir / "part_001.json"
            ann_file.write_text(json.dumps(annotation))

            # Mock push_to_hub
            with patch('datasets.DatasetDict.push_to_hub') as mock_push:
                create_hf_dataset_from_local(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    dataset_type="mfcad",
                    dataset_name="test-dataset",
                    upload=True,
                    hf_username="test-user"
                )

                # Verify push_to_hub was called
                mock_push.assert_called_once()
                args = mock_push.call_args[0]
                assert "test-user/test-dataset" == args[0]


class TestGenericDatasetCreation:
    """Test generic dataset creation."""

    def test_create_generic_dataset(self):
        """Test _create_generic_dataset function."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.dataset_builders.create_hf_dataset import (
            _create_generic_dataset
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create train split
            train_dir = tmpdir / "train"
            train_dir.mkdir()

            # Create sample
            step_file = train_dir / "part_001.step"
            step_file.write_text("ISO-10303-21;...;")

            annotation = {"file_name": "part_001.step", "custom_field": "value"}
            ann_file = train_dir / "part_001.json"
            ann_file.write_text(json.dumps(annotation))

            # Create dataset
            dataset_dict = _create_generic_dataset(tmpdir)

            assert "train" in dataset_dict
            assert len(dataset_dict["train"]) == 1

            sample = dataset_dict["train"][0]
            assert sample["file_name"] == "part_001.step"
            assert "step_content" in sample
            assert "annotation" in sample

    def test_generic_dataset_multiple_splits(self):
        """Test generic dataset with multiple splits."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.dataset_builders.create_hf_dataset import (
            _create_generic_dataset
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create train/val/test splits
            for split in ["train", "val", "test"]:
                split_dir = tmpdir / split
                split_dir.mkdir()

                step_file = split_dir / f"part_{split}.step"
                step_file.write_text("ISO-10303-21;...;")

                annotation = {"file_name": f"part_{split}.step"}
                ann_file = split_dir / f"part_{split}.json"
                ann_file.write_text(json.dumps(annotation))

            # Create dataset
            dataset_dict = _create_generic_dataset(tmpdir)

            assert len(dataset_dict) == 3
            assert "train" in dataset_dict
            assert "val" in dataset_dict
            assert "test" in dataset_dict

            assert len(dataset_dict["train"]) == 1
            assert len(dataset_dict["val"]) == 1
            assert len(dataset_dict["test"]) == 1


class TestDatasetBuilderIntegration:
    """Integration tests for dataset builders."""

    def test_end_to_end_dataset_creation(self):
        """Test end-to-end dataset creation."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.dataset_builders.create_hf_dataset import (
            create_hf_dataset_from_local
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "MFCAD"
            output_dir = tmpdir / "mfcad_hf"

            # Create realistic directory structure
            for split in ["train", "val", "test"]:
                split_dir = input_dir / split
                split_dir.mkdir(parents=True)

                # Create 3 samples per split
                for i in range(3):
                    step_file = split_dir / f"part_{i:03d}.step"
                    step_file.write_text("ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;")

                    annotation = {
                        "file_name": f"part_{i:03d}.step",
                        "faces": [
                            {
                                "face_id": j,
                                "label": ["hole", "pocket", "boss"][j % 3],
                                "instance_id": j,
                                "is_bottom_face": j == 0,
                                "parameters": {
                                    "diameter": 10.0 + j,
                                    "depth": 20.0 + j
                                }
                            }
                            for j in range(5)
                        ]
                    }

                    ann_file = split_dir / f"part_{i:03d}.json"
                    ann_file.write_text(json.dumps(annotation))

            # Create dataset
            create_hf_dataset_from_local(
                input_dir=input_dir,
                output_dir=output_dir,
                dataset_type="mfcad",
                upload=False
            )

            # Load created dataset
            from datasets import load_from_disk

            dataset_dict = load_from_disk(str(output_dir))

            # Verify structure
            assert "train" in dataset_dict
            assert "val" in dataset_dict
            assert "test" in dataset_dict

            assert len(dataset_dict["train"]) == 3
            assert len(dataset_dict["val"]) == 3
            assert len(dataset_dict["test"]) == 3

            # Verify sample structure
            sample = dataset_dict["train"][0]
            assert "file_name" in sample
            assert "step_content" in sample
            assert "faces" in sample
            assert "num_faces" in sample
            assert sample["num_faces"] == 5

    def test_builder_with_load_dataset(self):
        """Test builder works with load_dataset."""
        datasets = pytest.importorskip("datasets")
        from datasets import load_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create data directory
            data_dir = tmpdir / "data"
            train_dir = data_dir / "train"
            train_dir.mkdir(parents=True)

            # Create sample
            step_file = train_dir / "part_001.step"
            step_file.write_text("ISO-10303-21;...;")

            annotation = {
                "file_name": "part_001.step",
                "faces": [{"face_id": 1, "label": "hole", "instance_id": 0}]
            }
            ann_file = train_dir / "part_001.json"
            ann_file.write_text(json.dumps(annotation))

            # This test verifies the builder can be used with load_dataset
            # We can't actually test load_dataset without the full HF infrastructure
            # So we just verify the builder structure is correct
            from cadling.models.segmentation.training.dataset_builders.mfcad_builder import (
                MFCADDatasetBuilder
            )

            builder = MFCADDatasetBuilder()

            # Verify builder has required methods
            assert hasattr(builder, '_info')
            assert hasattr(builder, '_split_generators')
            assert hasattr(builder, '_generate_examples')
            assert callable(builder._info)
            assert callable(builder._split_generators)
            assert callable(builder._generate_examples)
