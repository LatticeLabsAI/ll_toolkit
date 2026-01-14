"""Unit tests for training loop.

Tests training infrastructure for CAD segmentation models.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import pytest
import tempfile
import shutil


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        torch = pytest.importorskip("torch")

        from cadling.models.segmentation.training.train import TrainingConfig

        config = TrainingConfig()

        assert config.num_epochs == 100
        assert config.learning_rate == 1e-3
        assert config.batch_size == 16
        assert config.gradient_clip == 1.0
        assert config.lr_scheduler == "plateau"
        assert config.early_stopping_patience == 10
        assert config.device == "auto"
        assert config.mixed_precision is False

    def test_training_config_custom_values(self):
        """Test TrainingConfig with custom values."""
        torch = pytest.importorskip("torch")

        from cadling.models.segmentation.training.train import TrainingConfig

        config = TrainingConfig(
            num_epochs=50,
            learning_rate=5e-4,
            batch_size=32,
            gradient_clip=None,
            lr_scheduler="cosine",
            mixed_precision=True
        )

        assert config.num_epochs == 50
        assert config.learning_rate == 5e-4
        assert config.batch_size == 32
        assert config.gradient_clip is None
        assert config.lr_scheduler == "cosine"
        assert config.mixed_precision is True


class TestSegmentationTrainer:
    """Test SegmentationTrainer class."""

    def test_trainer_initialization(self):
        """Test SegmentationTrainer initialization."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

        from cadling.models.segmentation.training.train import (
            SegmentationTrainer,
            TrainingConfig,
        )

        # Create simple real model (not mock) with actual parameters
        model = torch.nn.Linear(10, 10)

        config = TrainingConfig(
            num_epochs=10,
            checkpoint_dir=Path("test_checkpoints")
        )

        trainer = SegmentationTrainer(model, config)

        assert trainer.model is model
        assert trainer.config is config
        assert trainer.optimizer is not None
        assert trainer.criterion is not None
        assert trainer.current_epoch == 0
        assert trainer.best_val_loss == float("inf")

    def test_trainer_device_selection(self):
        """Test trainer device selection."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

        from cadling.models.segmentation.training.train import (
            SegmentationTrainer,
            TrainingConfig,
        )

        model = torch.nn.Linear(10, 10)

        # Test auto device selection
        config = TrainingConfig(device="auto")
        trainer = SegmentationTrainer(model, config)

        assert trainer.device is not None

        # Test explicit CPU
        config = TrainingConfig(device="cpu")
        trainer = SegmentationTrainer(model, config)

        assert trainer.device == torch.device("cpu")

    def test_trainer_optimizer_setup(self):
        """Test trainer optimizer setup."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

        from cadling.models.segmentation.training.train import (
            SegmentationTrainer,
            TrainingConfig,
        )

        model = torch.nn.Linear(10, 10)

        config = TrainingConfig(learning_rate=1e-4, weight_decay=1e-5)
        trainer = SegmentationTrainer(model, config)

        # Optimizer should be created
        assert trainer.optimizer is not None
        assert isinstance(trainer.optimizer, torch.optim.Adam)

    def test_trainer_lr_scheduler_plateau(self):
        """Test trainer learning rate scheduler (plateau)."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

        from cadling.models.segmentation.training.train import (
            SegmentationTrainer,
            TrainingConfig,
        )

        model = torch.nn.Linear(10, 10)

        config = TrainingConfig(lr_scheduler="plateau")
        trainer = SegmentationTrainer(model, config)

        assert trainer.scheduler is not None
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_trainer_lr_scheduler_cosine(self):
        """Test trainer learning rate scheduler (cosine)."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

        from cadling.models.segmentation.training.train import (
            SegmentationTrainer,
            TrainingConfig,
        )

        model = torch.nn.Linear(10, 10)

        config = TrainingConfig(lr_scheduler="cosine", num_epochs=50)
        trainer = SegmentationTrainer(model, config)

        assert trainer.scheduler is not None
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_trainer_mixed_precision_setup(self):
        """Test trainer mixed precision setup."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

        from cadling.models.segmentation.training.train import (
            SegmentationTrainer,
            TrainingConfig,
        )

        model = torch.nn.Linear(10, 10)

        # With mixed precision (requires CUDA)
        config = TrainingConfig(mixed_precision=True)
        trainer = SegmentationTrainer(model, config)

        # Scaler only created if CUDA available
        if torch.cuda.is_available():
            assert trainer.scaler is not None
        else:
            assert trainer.scaler is None

    def test_trainer_custom_criterion(self):
        """Test trainer with custom criterion."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

        from cadling.models.segmentation.training.train import (
            SegmentationTrainer,
            TrainingConfig,
        )

        model = torch.nn.Linear(10, 10)

        custom_criterion = torch.nn.MSELoss()
        config = TrainingConfig()

        trainer = SegmentationTrainer(model, config, criterion=custom_criterion)

        assert trainer.criterion is custom_criterion

    def test_train_epoch(self):
        """Test train_epoch method."""
        torch = pytest.importorskip("torch")
        torch_geometric = pytest.importorskip("torch_geometric")
        from torch_geometric.data import Data, Batch

        from cadling.models.segmentation.training.train import (
            SegmentationTrainer,
            TrainingConfig,
        )

        # Create simple model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(24, 24)

            def forward(self, x, edge_index, edge_attr, batch):
                return self.linear(x)

        model = SimpleModel()
        config = TrainingConfig(num_epochs=1, log_frequency=1)
        trainer = SegmentationTrainer(model, config)

        # Create mock pipeline
        mock_batches = [
            Batch(
                x=torch.randn(10, 24),
                edge_index=torch.randint(0, 10, (2, 20)),
                edge_attr=torch.randn(20, 8),
                y=torch.randint(0, 24, (10,)),
                batch=torch.zeros(10, dtype=torch.long)
            )
            for _ in range(3)
        ]

        mock_pipeline = MagicMock()
        mock_pipeline.__iter__ = MagicMock(return_value=iter(mock_batches))

        # Train one epoch
        metrics = trainer.train_epoch(mock_pipeline)

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "time" in metrics
        assert metrics["loss"] >= 0
        assert 0 <= metrics["accuracy"] <= 1

    def test_validate(self):
        """Test validate method."""
        torch = pytest.importorskip("torch")
        torch_geometric = pytest.importorskip("torch_geometric")
        from torch_geometric.data import Data, Batch

        from cadling.models.segmentation.training.train import (
            SegmentationTrainer,
            TrainingConfig,
        )

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(24, 24)

            def forward(self, x, edge_index, edge_attr, batch):
                return self.linear(x)

        model = SimpleModel()
        config = TrainingConfig()
        trainer = SegmentationTrainer(model, config)

        # Create mock validation pipeline
        mock_batches = [
            Batch(
                x=torch.randn(10, 24),
                edge_index=torch.randint(0, 10, (2, 20)),
                edge_attr=torch.randn(20, 8),
                y=torch.randint(0, 24, (10,)),
                batch=torch.zeros(10, dtype=torch.long)
            )
            for _ in range(2)
        ]

        mock_pipeline = MagicMock()
        mock_pipeline.__iter__ = MagicMock(return_value=iter(mock_batches))

        # Validate
        metrics = trainer.validate(mock_pipeline)

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert metrics["loss"] >= 0
        assert 0 <= metrics["accuracy"] <= 1

    def test_save_checkpoint(self):
        """Test save_checkpoint method."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

        from cadling.models.segmentation.training.train import (
            SegmentationTrainer,
            TrainingConfig,
        )

        model = torch.nn.Linear(10, 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            config = TrainingConfig(checkpoint_dir=tmpdir)
            trainer = SegmentationTrainer(model, config)
            trainer.current_epoch = 5

            # Save checkpoint
            checkpoint_path = tmpdir / "test_checkpoint.pt"
            trainer.save_checkpoint(checkpoint_path)

            # Verify file exists
            assert checkpoint_path.exists()

            # Load and verify contents
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            assert "epoch" in checkpoint
            assert checkpoint["epoch"] == 5
            assert "model_state_dict" in checkpoint
            assert "optimizer_state_dict" in checkpoint

    def test_load_checkpoint(self):
        """Test load_checkpoint method."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

        from cadling.models.segmentation.training.train import (
            SegmentationTrainer,
            TrainingConfig,
        )

        model = torch.nn.Linear(10, 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            config = TrainingConfig(checkpoint_dir=tmpdir)
            trainer1 = SegmentationTrainer(model, config)
            trainer1.current_epoch = 10
            trainer1.best_val_loss = 0.5

            # Save checkpoint
            checkpoint_path = tmpdir / "test_checkpoint.pt"
            trainer1.save_checkpoint(checkpoint_path)

            # Create new trainer and load
            model2 = torch.nn.Linear(10, 10)
            trainer2 = SegmentationTrainer(model2, config)
            trainer2.load_checkpoint(checkpoint_path)

            assert trainer2.current_epoch == 10
            assert trainer2.best_val_loss == 0.5

    def test_train_with_early_stopping(self):
        """Test train method with early stopping."""
        torch = pytest.importorskip("torch")
        torch_geometric = pytest.importorskip("torch_geometric")
        from torch_geometric.data import Batch

        from cadling.models.segmentation.training.train import (
            SegmentationTrainer,
            TrainingConfig,
        )

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(24, 24)

            def forward(self, x, edge_index, edge_attr, batch):
                return self.linear(x)

        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            config = TrainingConfig(
                num_epochs=100,
                early_stopping_patience=2,
                checkpoint_dir=tmpdir,
                log_frequency=1
            )
            trainer = SegmentationTrainer(model, config)

            # Create mock pipelines (single batch)
            def create_mock_pipeline():
                mock_batch = Batch(
                    x=torch.randn(5, 24),
                    edge_index=torch.randint(0, 5, (2, 10)),
                    edge_attr=torch.randn(10, 8),
                    y=torch.randint(0, 24, (5,)),
                    batch=torch.zeros(5, dtype=torch.long)
                )
                mock_pipeline = MagicMock()
                mock_pipeline.__iter__ = MagicMock(return_value=iter([mock_batch]))
                return mock_pipeline

            train_pipeline = create_mock_pipeline()
            val_pipeline = create_mock_pipeline()

            # Train (should stop early due to no improvement)
            history = trainer.train(train_pipeline, val_pipeline)

            # Should stop before 100 epochs due to early stopping
            assert trainer.current_epoch < 100

    def test_train_without_validation(self):
        """Test train method without validation pipeline."""
        torch = pytest.importorskip("torch")
        torch_geometric = pytest.importorskip("torch_geometric")
        from torch_geometric.data import Batch

        from cadling.models.segmentation.training.train import (
            SegmentationTrainer,
            TrainingConfig,
        )

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(24, 24)

            def forward(self, x, edge_index, edge_attr, batch):
                return self.linear(x)

        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            config = TrainingConfig(
                num_epochs=2,
                checkpoint_dir=tmpdir,
                log_frequency=1
            )
            trainer = SegmentationTrainer(model, config)

            # Create mock train pipeline
            mock_batch = Batch(
                x=torch.randn(5, 24),
                edge_index=torch.randint(0, 5, (2, 10)),
                edge_attr=torch.randn(10, 8),
                y=torch.randint(0, 24, (5,)),
                batch=torch.zeros(5, dtype=torch.long)
            )
            mock_pipeline = MagicMock()
            mock_pipeline.__iter__ = MagicMock(return_value=iter([mock_batch]))

            # Train without validation
            history = trainer.train(mock_pipeline, val_pipeline=None)

            assert len(history["train_loss"]) == 2
            assert len(history["train_acc"]) == 2
            assert len(history["val_loss"]) == 0
            assert len(history["val_acc"]) == 0

    def test_gradient_clipping(self):
        """Test gradient clipping."""
        torch = pytest.importorskip("torch")
        torch_geometric = pytest.importorskip("torch_geometric")
        from torch_geometric.data import Batch

        from cadling.models.segmentation.training.train import (
            SegmentationTrainer,
            TrainingConfig,
        )

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(24, 24)

            def forward(self, x, edge_index, edge_attr, batch):
                return self.linear(x)

        model = SimpleModel()

        # With gradient clipping
        config = TrainingConfig(gradient_clip=1.0)
        trainer = SegmentationTrainer(model, config)

        assert trainer.config.gradient_clip == 1.0

        # Without gradient clipping
        config = TrainingConfig(gradient_clip=None)
        trainer = SegmentationTrainer(model, config)

        assert trainer.config.gradient_clip is None


class TestTrainingIntegration:
    """Integration tests for training."""

    def test_full_training_cycle(self):
        """Test complete training cycle."""
        torch = pytest.importorskip("torch")
        torch_geometric = pytest.importorskip("torch_geometric")
        from torch_geometric.data import Batch

        from cadling.models.segmentation.training.train import (
            SegmentationTrainer,
            TrainingConfig,
        )

        # Simple model
        class SimpleSegModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(24, 64)
                self.fc2 = torch.nn.Linear(64, 24)

            def forward(self, x, edge_index, edge_attr, batch):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        model = SimpleSegModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            config = TrainingConfig(
                num_epochs=3,
                batch_size=4,
                learning_rate=1e-3,
                checkpoint_dir=tmpdir,
                checkpoint_frequency=1,
                log_frequency=1
            )

            trainer = SegmentationTrainer(model, config)

            # Create mock data
            def create_pipeline():
                batches = [
                    Batch(
                        x=torch.randn(10, 24),
                        edge_index=torch.randint(0, 10, (2, 20)),
                        edge_attr=torch.randn(20, 8),
                        y=torch.randint(0, 24, (10,)),
                        batch=torch.zeros(10, dtype=torch.long)
                    )
                    for _ in range(2)
                ]
                mock_pipeline = MagicMock()
                mock_pipeline.__iter__ = MagicMock(return_value=iter(batches))
                return mock_pipeline

            train_pipeline = create_pipeline()
            val_pipeline = create_pipeline()

            # Train
            history = trainer.train(train_pipeline, val_pipeline)

            # Verify history
            assert len(history["train_loss"]) == 3
            assert len(history["val_loss"]) == 3

            # Verify checkpoints created
            best_model_path = tmpdir / "best_model.pt"
            assert best_model_path.exists()

    def test_checkpoint_persistence(self):
        """Test that checkpoints persist model state correctly."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")

        from cadling.models.segmentation.training.train import (
            SegmentationTrainer,
            TrainingConfig,
        )

        model1 = torch.nn.Linear(10, 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Train first model
            config = TrainingConfig(checkpoint_dir=tmpdir)
            trainer1 = SegmentationTrainer(model1, config)

            # Get initial parameters
            initial_params = {name: param.clone() for name, param in model1.named_parameters()}

            # Modify model
            for param in model1.parameters():
                param.data.fill_(1.0)

            # Save
            checkpoint_path = tmpdir / "checkpoint.pt"
            trainer1.save_checkpoint(checkpoint_path)

            # Create new model and load
            model2 = torch.nn.Linear(10, 10)
            trainer2 = SegmentationTrainer(model2, config)
            trainer2.load_checkpoint(checkpoint_path)

            # Verify parameters match
            for name, param in model2.named_parameters():
                assert torch.all(param == 1.0), f"Parameter {name} not loaded correctly"
