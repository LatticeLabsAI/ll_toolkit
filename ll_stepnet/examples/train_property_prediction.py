"""
Example: Train STEP property prediction model.
Predicts physical properties (volume, mass, surface area, etc.) from STEP files.
"""

import sys
sys.path.insert(0, '../')

import torch
import torch.nn as nn
from stepnet.tasks import STEPForPropertyPrediction
from stepnet.data import create_dataloader, load_dataset_from_directory
from stepnet.trainer import STEPTrainer


def main():
    # Configuration
    data_dir = 'data/cad_properties'
    num_properties = 6  # [volume, surface_area, mass, bbox_x, bbox_y, bbox_z]
    batch_size = 8
    num_epochs = 20
    max_length = 2048

    print("=" * 60)
    print("STEP Property Prediction Training")
    print("=" * 60)

    # Load dataset
    print("\nLoading training data...")
    train_files, train_properties = load_dataset_from_directory(
        data_dir, split='train', label_file='properties.json'
    )
    print(f"Found {len(train_files)} training files")

    print("\nLoading validation data...")
    val_files, val_properties = load_dataset_from_directory(
        data_dir, split='val', label_file='properties.json'
    )
    print(f"Found {len(val_files)} validation files")

    # Create data loaders
    print("\nCreating data loaders...")
    train_loader = create_dataloader(
        file_paths=train_files,
        labels=train_properties,
        batch_size=batch_size,
        max_length=max_length,
        use_topology=True,
        shuffle=True
    )

    val_loader = create_dataloader(
        file_paths=val_files,
        labels=val_properties,
        batch_size=batch_size,
        max_length=max_length,
        use_topology=True,
        shuffle=False
    )

    # Initialize model
    print("\nInitializing model...")
    model = STEPForPropertyPrediction(
        vocab_size=50000,
        num_properties=num_properties,
        output_dim=1024
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")

    # Custom loss function (MSE with normalization)
    loss_fn = nn.MSELoss()

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = STEPTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        loss_fn=loss_fn,
        checkpoint_dir='checkpoints/property_prediction'
    )

    # Train
    print("\nStarting training...")
    trainer.train(num_epochs=num_epochs, save_every=5)

    print("\nTraining complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == '__main__':
    main()
