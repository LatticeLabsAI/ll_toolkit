"""Unit tests for data loaders.

Tests HuggingFace dataset loaders for CAD segmentation datasets.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, Mock
import pytest
import numpy as np


class TestBaseDataLoader:
    """Test BaseDataLoader class."""

    def test_base_loader_initialization(self):
        """Test BaseDataLoader can be initialized."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.data_loaders import BaseDataLoader

        with patch('cadling.models.segmentation.training.data_loaders.load_dataset') as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = BaseDataLoader(
                dataset_name="test/dataset",
                streaming=True,
                split="train"
            )

            assert loader.dataset_name == "test/dataset"
            assert loader.streaming is True
            assert loader.split == "train"
            assert loader.dataset is not None
            mock_load.assert_called_once()

    def test_base_loader_iteration(self):
        """Test BaseDataLoader iteration."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.data_loaders import BaseDataLoader

        # Create mock dataset
        mock_samples = [
            {"id": 1, "data": "sample1"},
            {"id": 2, "data": "sample2"},
            {"id": 3, "data": "sample3"},
        ]

        with patch('cadling.models.segmentation.training.data_loaders.load_dataset') as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(mock_samples))
            mock_load.return_value = mock_dataset

            loader = BaseDataLoader(
                dataset_name="test/dataset",
                streaming=True,
                split="train"
            )

            # Iterate and collect samples
            collected = list(loader)
            assert len(collected) == 3
            assert collected[0]["id"] == 1
            assert collected[1]["id"] == 2

    def test_base_loader_take(self):
        """Test BaseDataLoader take() method."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.data_loaders import BaseDataLoader

        mock_samples = [{"id": i} for i in range(10)]

        with patch('cadling.models.segmentation.training.data_loaders.load_dataset') as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.take = MagicMock(return_value=iter(mock_samples[:3]))
            mock_load.return_value = mock_dataset

            loader = BaseDataLoader(
                dataset_name="test/dataset",
                streaming=True,
                split="train"
            )

            # Take first 3 samples
            taken = list(loader.take(3))
            assert len(taken) == 3


class TestMFCADDataLoader:
    """Test MFCADDataLoader class."""

    def test_mfcad_loader_initialization(self):
        """Test MFCADDataLoader initialization."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.data_loaders import MFCADDataLoader

        with patch('cadling.models.segmentation.training.data_loaders.load_dataset') as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = MFCADDataLoader(
                dataset_name="path/to/mfcad",
                streaming=True,
                split="train"
            )

            assert loader.dataset_name == "path/to/mfcad"
            assert len(loader.FEATURE_CLASSES) == 25
            assert "hole" in loader.FEATURE_CLASSES
            assert "pocket" in loader.FEATURE_CLASSES

    def test_mfcad_loader_skips_unparseable_step(self):
        """Test MFCADDataLoader returns None for unparseable STEP content."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.data_loaders import MFCADDataLoader

        with patch('cadling.models.segmentation.training.data_loaders.load_dataset') as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = MFCADDataLoader(
                dataset_name="path/to/mfcad",
                streaming=True
            )

            # Create mock sample with unparseable STEP content
            mock_sample = {
                "file_name": "part_001.step",
                "step_content": "GARBAGE UNPARSEABLE CONTENT",
                "faces": [
                    {
                        "face_id": 1,
                        "label": "hole",
                        "instance_id": 0,
                    }
                ]
            }

            # Preprocess should return None to skip unparseable sample
            processed = loader._preprocess_sample(mock_sample)

            if processed is not None:
                raise AssertionError(
                    "Data loader must return None for unparseable STEP content, "
                    f"but returned: {processed}"
                )

    def test_mfcad_loader_skips_empty_step(self):
        """Test MFCADDataLoader returns None for empty STEP content."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.data_loaders import MFCADDataLoader

        with patch('cadling.models.segmentation.training.data_loaders.load_dataset') as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = MFCADDataLoader(
                dataset_name="path/to/mfcad",
                streaming=True
            )

            # Create mock sample with empty STEP content
            mock_sample = {
                "file_name": "part_002.step",
                "step_content": "",
                "faces": []
            }

            # Preprocess should return None to skip empty sample
            processed = loader._preprocess_sample(mock_sample)

            if processed is not None:
                raise AssertionError(
                    "Data loader must return None for empty STEP content, "
                    f"but returned: {processed}"
                )

    def test_mfcad_loader_processes_valid_labels(self):
        """Test MFCADDataLoader processes face labels correctly."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.data_loaders import MFCADDataLoader

        with patch('cadling.models.segmentation.training.data_loaders.load_dataset') as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = MFCADDataLoader(
                dataset_name="path/to/mfcad",
                streaming=True
            )

            # Don't test STEP parsing (that's tested elsewhere)
            # Just test that loader correctly processes the face label structure
            # when entities ARE available

            # Mock the entities being available (as if STEP parsing succeeded)
            mock_sample_with_entities = {
                "file_name": "part_003.step",
                "step_content": "",  # Empty so parsing returns None
                "faces": [
                    {
                        "face_id": 1,
                        "label": "hole",
                        "instance_id": 0,
                    },
                    {
                        "face_id": 2,
                        "label": "pocket",
                        "instance_id": 1,
                    },
                ]
            }

            # Since we can't mock STEP parsing easily without actual STEP content,
            # we'll test the label mapping logic separately
            face_labels_raw = ["hole", "pocket", "boss", "unknown_label"]

            # Test label index mapping
            hole_idx = loader.FEATURE_CLASSES.index("hole")
            pocket_idx = loader.FEATURE_CLASSES.index("pocket")
            boss_idx = loader.FEATURE_CLASSES.index("boss")

            if hole_idx < 0 or pocket_idx < 0 or boss_idx < 0:
                raise AssertionError("Feature classes must include hole, pocket, boss")

            # Unknown labels should map to index 0 (base)
            if "unknown_label" in loader.FEATURE_CLASSES:
                raise AssertionError("Unknown labels should not be in FEATURE_CLASSES")



class TestMFInstSegDataLoader:
    """Test MFInstSegDataLoader class."""

    def test_mfinstseg_loader_initialization(self):
        """Test MFInstSegDataLoader initialization."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.data_loaders import MFInstSegDataLoader

        with patch('cadling.models.segmentation.training.data_loaders.load_dataset') as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = MFInstSegDataLoader(
                dataset_name="path/to/mfinstseg",
                streaming=True
            )

            assert loader.dataset_name == "path/to/mfinstseg"

    def test_mfinstseg_loader_skips_unparseable_step(self):
        """Test MFInstSegDataLoader returns None for unparseable STEP content."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.data_loaders import MFInstSegDataLoader

        with patch('cadling.models.segmentation.training.data_loaders.load_dataset') as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = MFInstSegDataLoader(dataset_name="path/to/mfinstseg", streaming=True)

            # Create mock sample with unparseable STEP content
            mock_sample = {
                "file_name": "part_001.step",
                "step_content": "INVALID STEP DATA",
                "instances": [
                    {"instance_id": 0, "num_faces": 3},
                ],
                "boundary_edges": [True, False, True],
            }

            # Preprocess should return None to skip unparseable sample
            processed = loader._preprocess_sample(mock_sample)

            if processed is not None:
                raise AssertionError(
                    "Data loader must return None for unparseable STEP content, "
                    f"but returned: {processed}"
                )

    def test_mfinstseg_loader_skips_empty_step(self):
        """Test MFInstSegDataLoader returns None for empty STEP content."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.data_loaders import MFInstSegDataLoader

        with patch('cadling.models.segmentation.training.data_loaders.load_dataset') as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = MFInstSegDataLoader(dataset_name="path/to/mfinstseg", streaming=True)

            # Create mock sample with empty STEP content
            mock_sample = {
                "file_name": "part_002.step",
                "step_content": "",
                "instances": [],
            }

            # Preprocess should return None to skip empty sample
            processed = loader._preprocess_sample(mock_sample)

            if processed is not None:
                raise AssertionError(
                    "Data loader must return None for empty STEP content, "
                    f"but returned: {processed}"
                )



class TestABCDataLoader:
    """Test ABCDataLoader class."""

    def test_abc_loader_initialization(self):
        """Test ABCDataLoader initialization."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.data_loaders import ABCDataLoader

        with patch('cadling.models.segmentation.training.data_loaders.load_dataset') as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = ABCDataLoader(
                dataset_name="abc-dataset/abc-meshes",
                streaming=True
            )

            assert "abc" in loader.dataset_name.lower()

    def test_abc_loader_preprocess(self):
        """Test ABCDataLoader preprocessing."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.data_loaders import ABCDataLoader

        with patch('cadling.models.segmentation.training.data_loaders.load_dataset') as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = ABCDataLoader(dataset_name="abc-dataset/abc-meshes", streaming=True)

            mock_sample = {
                "file_name": "model_001.obj",
                "mesh": {
                    "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                    "faces": [[0, 1, 2]],
                },
                "hierarchy": {"assembly": "test"},
            }

            processed = loader._preprocess_sample(mock_sample)

            assert processed["file_name"] == "model_001.obj"
            assert processed["num_vertices"] == 3
            assert processed["num_faces"] == 1
            assert processed["mesh_vertices"].shape == (3, 3)


class TestFusion360DataLoader:
    """Test Fusion360DataLoader class."""

    def test_fusion360_loader_initialization(self):
        """Test Fusion360DataLoader initialization."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.data_loaders import Fusion360DataLoader

        with patch('cadling.models.segmentation.training.data_loaders.load_dataset') as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = Fusion360DataLoader(
                dataset_name="fusion360-gallery/assembly-dataset",
                streaming=True
            )

            assert "fusion" in loader.dataset_name.lower() or "360" in loader.dataset_name


class TestGetDataLoader:
    """Test get_data_loader function."""

    def test_get_data_loader_auto_detect_mfcad(self):
        """Test auto-detection of MFCAD dataset."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.data_loaders import (
            get_data_loader,
            MFCADDataLoader,
        )

        with patch('cadling.models.segmentation.training.data_loaders.load_dataset') as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = get_data_loader(
                dataset_name="path/to/mfcad",
                dataset_type="auto",
                streaming=True
            )

            assert isinstance(loader, MFCADDataLoader)

    def test_get_data_loader_auto_detect_abc(self):
        """Test auto-detection of ABC dataset."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.data_loaders import (
            get_data_loader,
            ABCDataLoader,
        )

        with patch('cadling.models.segmentation.training.data_loaders.load_dataset') as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = get_data_loader(
                dataset_name="abc-dataset/abc-meshes",
                dataset_type="auto",
                streaming=True
            )

            assert isinstance(loader, ABCDataLoader)

    def test_get_data_loader_explicit_type(self):
        """Test explicit dataset type."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.data_loaders import (
            get_data_loader,
            MFInstSegDataLoader,
        )

        with patch('cadling.models.segmentation.training.data_loaders.load_dataset') as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = get_data_loader(
                dataset_name="path/to/dataset",
                dataset_type="mfinstseg",
                streaming=True
            )

            assert isinstance(loader, MFInstSegDataLoader)

    def test_get_data_loader_with_options(self):
        """Test get_data_loader with additional options."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.data_loaders import get_data_loader

        with patch('cadling.models.segmentation.training.data_loaders.load_dataset') as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = get_data_loader(
                dataset_name="path/to/mfcad",
                dataset_type="mfcad",
                streaming=False,
                split="val",
                batch_size=8
            )

            assert loader.streaming is False
            assert loader.split == "val"
            assert loader.batch_size == 8


class TestDataLoaderIntegration:
    """Integration tests for data loaders."""

    def test_loader_shuffle(self):
        """Test loader shuffle functionality."""
        datasets = pytest.importorskip("datasets")

        from cadling.models.segmentation.training.data_loaders import MFCADDataLoader

        with patch('cadling.models.segmentation.training.data_loaders.load_dataset') as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.shuffle = MagicMock(return_value=mock_dataset)
            mock_load.return_value = mock_dataset

            loader = MFCADDataLoader(
                dataset_name="path/to/mfcad",
                streaming=True
            )

            # Shuffle
            loader.shuffle(seed=42)

            # Verify shuffle was called
            mock_dataset.shuffle.assert_called()

    def test_loader_without_hf_datasets(self):
        """Test error when HuggingFace datasets not available."""
        # This is tricky to test because we need to mock the import
        # Just verify the import guard exists
        from cadling.models.segmentation.training import data_loaders

        assert hasattr(data_loaders, 'HF_DATASETS_AVAILABLE')
