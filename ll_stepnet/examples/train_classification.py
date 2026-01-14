"""
Example: Train STEP part classification model.
Classifies STEP files into part categories (bracket, housing, shaft, gear, etc.)
"""

import sys
sys.path.insert(0, '../')

import torch
from stepnet import STEPTokenizer, STEPFeatureExtractor, STEPTopologyBuilder
from stepnet.tasks import STEPForClassification
from stepnet.data import create_dataloader, load_dataset_from_directory
from stepnet.trainer import STEPTrainer


def main():
    # Configuration
    data_dir = 'data/cad_parts'  # Directory with train/val subdirs
    num_classes = 10  # Number of part categories
    batch_size = 8
    num_epochs = 10
    max_length = 2048

    print("=" * 60)
    print("STEP Part Classification Training")
    print("=" * 60)

    # Load dataset
    print("\nLoading training data...")
    train_files, train_labels = load_dataset_from_directory(
        data_dir, split='train', label_file='labels.json'
    )
    print(f"Found {len(train_files)} training files")

    print("\nLoading validation data...")
    val_files, val_labels = load_dataset_from_directory(
        data_dir, split='val', label_file='labels.json'
    )
    print(f"Found {len(val_files)} validation files")

    # Create data loaders
    print("\nCreating data loaders...")
    train_loader = create_dataloader(
        file_paths=train_files,
        labels=train_labels,
        batch_size=batch_size,
        max_length=max_length,
        use_topology=True,
        shuffle=True
    )

    val_loader = create_dataloader(
        file_paths=val_files,
        labels=val_labels,
        batch_size=batch_size,
        max_length=max_length,
        use_topology=True,
        shuffle=False
    )

    # Initialize model
    print("\nInitializing model...")
    model = STEPForClassification(
        vocab_size=50000,
        num_classes=num_classes,
        output_dim=1024
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = STEPTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        checkpoint_dir='checkpoints/classification'
    )

    # Train
    print("\nStarting training...")
    trainer.train(num_epochs=num_epochs, save_every=2)

    print("\nTraining complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == '__main__':
    main()
