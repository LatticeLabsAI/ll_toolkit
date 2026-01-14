"""Example training script for CAD segmentation models.

This script demonstrates end-to-end training with:
- HuggingFace streaming datasets
- Mixed precision training
- Checkpointing and early stopping
- Multi-GPU support (if available)

Usage:
    # Train B-Rep segmentation model
    python train_example.py \
        --model brep \
        --dataset_path /path/to/mfcad \
        --dataset_type mfcad \
        --output_dir checkpoints/brep_seg \
        --num_epochs 50 \
        --batch_size 16 \
        --streaming

    # Train mesh segmentation model
    python train_example.py \
        --model mesh \
        --dataset_path abc-dataset/abc-meshes \
        --dataset_type abc \
        --output_dir checkpoints/mesh_seg \
        --num_epochs 30 \
        --batch_size 32
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_brep_segmentation(
    dataset_path: str,
    dataset_type: str,
    output_dir: Path,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    streaming: bool,
    mixed_precision: bool,
) -> None:
    """Train B-Rep segmentation model.

    Args:
        dataset_path: Path to dataset or HF dataset name
        dataset_type: Dataset type ('mfcad', 'mfinstseg', 'auto')
        output_dir: Output directory for checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        streaming: Use streaming mode
        mixed_precision: Use mixed precision training
    """
    logger.info("=" * 80)
    logger.info("Training B-Rep Segmentation Model")
    logger.info("=" * 80)

    # Import modules
    from cadling.models.segmentation.architectures import HybridGATTransformer
    from cadling.models.segmentation.training import (
        create_streaming_pipeline,
        SegmentationTrainer,
        TrainingConfig,
    )
    from cadling.models.segmentation.training.streaming_pipeline import build_brep_graph

    # Create model
    logger.info("Creating HybridGATTransformer model...")
    model = HybridGATTransformer(
        in_dim=24,  # B-Rep node features
        num_classes=24,  # 24 manufacturing features
        gat_hidden_dim=256,
        gat_num_heads=8,
        gat_num_layers=3,
        transformer_hidden_dim=512,
        transformer_num_layers=4,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    # Create streaming pipelines
    logger.info(f"Creating streaming pipelines (streaming={streaming})...")

    train_pipeline = create_streaming_pipeline(
        dataset_name=dataset_path,
        graph_builder=build_brep_graph,
        dataset_type=dataset_type,
        split="train",
        batch_size=batch_size,
        streaming=streaming,
        cache_graphs=True,
        shuffle=True,
    )

    val_pipeline = create_streaming_pipeline(
        dataset_name=dataset_path,
        graph_builder=build_brep_graph,
        dataset_type=dataset_type,
        split="val",
        batch_size=batch_size,
        streaming=streaming,
        cache_graphs=True,
        shuffle=False,
    )

    # Training config
    config = TrainingConfig(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        checkpoint_dir=output_dir,
        checkpoint_frequency=5,
        log_frequency=10,
        early_stopping_patience=10,
        lr_scheduler="plateau",
        mixed_precision=mixed_precision,
    )

    logger.info(f"Training configuration:")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Mixed precision: {config.mixed_precision}")

    # Create trainer
    trainer = SegmentationTrainer(model, config)

    # Train
    logger.info("Starting training...")
    history = trainer.train(train_pipeline, val_pipeline)

    # Summary
    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to: {output_dir}")
    logger.info(f"Best model: {output_dir / 'best_model.pt'}")


def train_mesh_segmentation(
    dataset_path: str,
    dataset_type: str,
    output_dir: Path,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    streaming: bool,
    mixed_precision: bool,
) -> None:
    """Train mesh segmentation model.

    Args:
        dataset_path: Path to dataset or HF dataset name
        dataset_type: Dataset type ('abc', 'auto')
        output_dir: Output directory for checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        streaming: Use streaming mode
        mixed_precision: Use mixed precision training
    """
    logger.info("=" * 80)
    logger.info("Training Mesh Segmentation Model")
    logger.info("=" * 80)

    # Import modules
    from cadling.models.segmentation.architectures import MeshSegmentationGNN
    from cadling.models.segmentation.training import (
        create_streaming_pipeline,
        SegmentationTrainer,
        TrainingConfig,
    )
    from cadling.models.segmentation.training.streaming_pipeline import build_mesh_graph

    # Create model
    logger.info("Creating MeshSegmentationGNN model...")
    model = MeshSegmentationGNN(
        in_dim=7,  # Mesh node features (centroid + normal + area)
        num_classes=12,  # 12 mesh segments
        hidden_dims=[64, 128, 256, 512, 512],
        use_pretrained_encoders=False,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    # Create streaming pipelines
    logger.info(f"Creating streaming pipelines (streaming={streaming})...")

    train_pipeline = create_streaming_pipeline(
        dataset_name=dataset_path,
        graph_builder=build_mesh_graph,
        dataset_type=dataset_type,
        split="train",
        batch_size=batch_size,
        streaming=streaming,
        cache_graphs=True,
        shuffle=True,
    )

    val_pipeline = create_streaming_pipeline(
        dataset_name=dataset_path,
        graph_builder=build_mesh_graph,
        dataset_type=dataset_type,
        split="val",
        batch_size=batch_size,
        streaming=streaming,
        cache_graphs=True,
        shuffle=False,
    )

    # Training config
    config = TrainingConfig(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        checkpoint_dir=output_dir,
        checkpoint_frequency=5,
        log_frequency=10,
        early_stopping_patience=10,
        lr_scheduler="cosine",
        mixed_precision=mixed_precision,
    )

    logger.info(f"Training configuration:")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Batch size: {config.batch_size}")

    # Create trainer
    trainer = SegmentationTrainer(model, config)

    # Train
    logger.info("Starting training...")
    history = trainer.train(train_pipeline, val_pipeline)

    # Summary
    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to: {output_dir}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Train CAD segmentation models"
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["brep", "mesh"],
        help="Model type to train",
    )

    # Dataset
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset or HuggingFace dataset name",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="auto",
        choices=["mfcad", "mfinstseg", "abc", "fusion360", "auto"],
        help="Dataset type",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode (avoid full download)",
    )

    # Training
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("checkpoints"),
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Use mixed precision training (AMP)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    if args.model == "brep":
        train_brep_segmentation(
            dataset_path=args.dataset_path,
            dataset_type=args.dataset_type,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            streaming=args.streaming,
            mixed_precision=args.mixed_precision,
        )
    elif args.model == "mesh":
        train_mesh_segmentation(
            dataset_path=args.dataset_path,
            dataset_type=args.dataset_type,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            streaming=args.streaming,
            mixed_precision=args.mixed_precision,
        )


if __name__ == "__main__":
    main()
