"""
Example: Analyze Data Requirements for STEP Models

This script demonstrates how to use the data requirement determination tool
to analyze scaling laws and predict dataset size needs for STEP models.

Usage:
    python examples/analyze_data_requirements.py

The script will:
1. Generate learning curves by training on varying dataset sizes
2. Fit transformer scaling laws to the data
3. Predict required dataset sizes for target performance
4. Visualize results with scaling law plots
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from stepnet import STEPForClassification
from stepnet.data import load_dataset_from_directory, STEPDataset, create_dataloader
from stepnet.data_requirements import (
    STEPLearningCurveGenerator,
    STEPScalingLawAnalyzer,
    plot_learning_curve_with_scaling_law,
    suggest_dataset_size,
    count_model_parameters
)


def main():
    print("="*70, flush=True)
    print("STEP Model Data Requirements Analysis", flush=True)
    print("="*70, flush=True)
    print("Starting analysis...", flush=True)

    # =========================================================================
    # Step 1: Load Dataset
    # =========================================================================
    print("\n[Step 1] Loading dataset...", flush=True)

    data_dir = 'data/learning_curve'  # Using collected STEP files

    # For a real analysis, you would have a proper train/val split
    # Here we'll create a simple example
    file_paths = list(Path(data_dir).glob('*.step')) + list(Path(data_dir).glob('*.stp'))

    if len(file_paths) == 0:
        print(f"No STEP files found in {data_dir}")
        print("Please ensure you have STEP files in the data/learning_curve directory")
        return

    print(f"Found {len(file_paths)} STEP files")

    # Create synthetic labels for demonstration
    # In real use, you would load actual labels
    labels = np.random.randint(0, 3, size=len(file_paths))  # 3 classes

    # Split into train/val
    n_train = int(0.8 * len(file_paths))
    train_files = [str(f) for f in file_paths[:n_train]]
    val_files = [str(f) for f in file_paths[n_train:]]
    train_labels = labels[:n_train].tolist()
    val_labels = labels[n_train:].tolist()

    print(f"  Training samples: {len(train_files)}")
    print(f"  Validation samples: {len(val_files)}")

    # Create datasets
    from stepnet.tokenizer import STEPTokenizer

    print("  Creating tokenizer...")
    tokenizer = STEPTokenizer()

    print(f"  Creating training dataset with {len(train_files)} files...")
    train_dataset = STEPDataset(
        file_paths=train_files,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=256,  # Reduced for faster processing
        use_topology=False  # Disable topology for faster demo
    )
    print("  Training dataset created successfully!")

    print(f"  Creating validation dataset with {len(val_files)} files...")
    val_dataset = STEPDataset(
        file_paths=val_files,
        labels=val_labels,
        tokenizer=tokenizer,
        max_length=256,  # Reduced for faster processing
        use_topology=False
    )
    print("  Validation dataset created successfully!")

    # =========================================================================
    # Step 2: Create Model and Analyze Parameters
    # =========================================================================
    print("\n[Step 2] Analyzing model architecture...")

    model = STEPForClassification(
        vocab_size=50000,  # Must match tokenizer vocab_size
        num_classes=3,
        output_dim=256  # Reduced for demo
    )

    num_params = count_model_parameters(model, exclude_embeddings=True)
    print(f"Model created with {num_params:,} non-embedding parameters")

    # Get dataset size recommendations
    recommendations = suggest_dataset_size(
        model,
        task_type='classification',
        quality_level='good'
    )

    # =========================================================================
    # Step 3: Generate Learning Curves
    # =========================================================================
    print("\n[Step 3] Generating learning curves...")
    print("This will train models on varying dataset sizes...")

    # Define sample fractions to test
    # For demo, use fewer points to complete faster
    sample_fractions = [0.2, 0.5, 1.0]

    generator = STEPLearningCurveGenerator(
        model_class=STEPForClassification,
        model_kwargs={
            'vocab_size': 50000,  # Must match tokenizer vocab_size
            'num_classes': 3,
            'output_dim': 256  # Reduced from 512
        },
        train_kwargs={
            'num_epochs': 2,  # Reduced for faster demo
            'learning_rate': 1e-4,
            'batch_size': 32,  # Increased for better GPU utilization
            'patience': 2
        }
    )

    # Generate learning curve
    # Note: This will take time as it trains multiple models
    print(f"Will train 3 models: {[int(f*len(train_dataset)) for f in sample_fractions]} samples each")
    results = generator.generate_learning_curve(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        sample_fractions=sample_fractions,
        n_iterations=1,  # Reduced to 1 for faster completion
        save_dir='learning_curves'
    )

    # =========================================================================
    # Step 4: Fit Scaling Laws
    # =========================================================================
    print("\n[Step 4] Fitting scaling laws...")

    analyzer = STEPScalingLawAnalyzer()

    # Get mean validation losses
    sample_sizes = results['sample_sizes']
    val_losses = results['val_losses']
    val_accuracies = results['val_accuracies']

    mean_val_losses = np.mean(val_losses, axis=1)

    # Fit OpenAI-style scaling law
    fitted_params = analyzer.fit_power_law(
        sample_sizes,
        mean_val_losses,
        law_type='openai'
    )

    # =========================================================================
    # Step 5: Predict Data Requirements
    # =========================================================================
    print("\n[Step 5] Predicting data requirements...")

    # Predict samples needed for target loss
    target_loss = 0.3  # Example target
    required_samples = analyzer.predict_required_samples(target_loss)

    # Predict performance at larger dataset size
    target_size = int(len(train_dataset) * 2)  # 2x current size
    predicted_loss = analyzer.extrapolate_performance(target_size)

    # =========================================================================
    # Step 6: Visualize Results
    # =========================================================================
    print("\n[Step 6] Generating visualizations...")

    plot_learning_curve_with_scaling_law(
        sample_sizes=sample_sizes,
        val_losses=val_losses,
        val_accuracies=val_accuracies,
        fitted_params=fitted_params,
        fit_type='openai',
        extrapolate_to=target_size,
        save_path='learning_curves/scaling_law_plot.png'
    )

    # =========================================================================
    # Summary Report
    # =========================================================================
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)

    print(f"\nModel Configuration:")
    print(f"  Non-embedding parameters: {num_params:,}")
    print(f"  Task: Classification (3 classes)")

    print(f"\nCurrent Dataset:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Current performance: {mean_val_losses[-1]:.4f} loss")
    if val_accuracies is not None:
        print(f"  Current accuracy: {np.mean(val_accuracies[-1]):.4f}")

    print(f"\nScaling Law Parameters:")
    if fitted_params:
        for key, value in fitted_params.items():
            print(f"  {key} = {value:.4e}")

    print(f"\nPredictions:")
    if required_samples:
        print(f"  Samples needed for {target_loss:.4f} loss: {required_samples:,}")
    if predicted_loss:
        print(f"  Expected loss at {target_size:,} samples: {predicted_loss:.4f}")

    print(f"\nRecommendations (from literature):")
    print(f"  Minimum dataset: {recommendations['minimum']:,} samples")
    print(f"  Recommended dataset: {recommendations['recommended']:,} samples")
    print(f"  Excellent dataset: {recommendations['excellent']:,} samples")

    print("\n" + "="*70)
    print("Analysis complete! Check 'learning_curves/' directory for results.")
    print("="*70)


def quick_estimate_example():
    """
    Quick example: Estimate data requirements without training.

    This uses literature-based recommendations and Chinchilla scaling laws.
    """
    print("\n" + "="*70)
    print("QUICK DATA REQUIREMENTS ESTIMATE (No Training Required)")
    print("="*70)

    # Create model
    model = STEPForClassification(
        vocab_size=50000,
        num_classes=10,
        output_dim=1024
    )

    print("\nAnalyzing model...")
    num_params = count_model_parameters(model, exclude_embeddings=True)

    # Get recommendations for different tasks
    tasks = ['classification', 'property', 'captioning']

    for task in tasks:
        print(f"\n--- {task.upper()} TASK ---")
        recommendations = suggest_dataset_size(
            model,
            task_type=task,
            quality_level='good'
        )


if __name__ == '__main__':
    # Choose which example to run
    import argparse

    parser = argparse.ArgumentParser(description='Analyze data requirements for STEP models')
    parser.add_argument(
        '--mode',
        choices=['full', 'quick'],
        default='quick',
        help='Analysis mode: full (with training) or quick (estimates only)'
    )

    args = parser.parse_args()

    if args.mode == 'full':
        main()
    else:
        quick_estimate_example()
