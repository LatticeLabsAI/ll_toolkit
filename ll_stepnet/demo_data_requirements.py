#!/usr/bin/env python
"""
Quick Demo: Data Requirements Analysis for STEP Models

This script provides a quick demonstration of the data requirements tool
using the test STEP files included in the repository.

Run this to see:
1. Model parameter analysis
2. Dataset size recommendations based on literature
3. Chinchilla scaling law estimates

Usage:
    python demo_data_requirements.py
"""

import torch
import numpy as np
from pathlib import Path

# Import STEP model components
from stepnet import (
    STEPForClassification,
    STEPForPropertyPrediction,
    count_model_parameters,
    suggest_dataset_size,
    chinchilla_optimal_tokens
)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def analyze_model(model, model_name, task_type):
    """Analyze a single model and print recommendations."""
    print(f"\n--- {model_name} ---")

    # Count parameters
    num_params = count_model_parameters(model, exclude_embeddings=True)

    # Chinchilla optimal
    optimal_tokens = chinchilla_optimal_tokens(num_params)

    # Get recommendations (call once with 'good' quality level)
    recs = suggest_dataset_size(model, task_type=task_type, quality_level='good')

    print(f"\n{task_type.upper()} TASK - Data Requirements:")
    print(f"\n  MINIMUM Quality Level:")
    print(f"    Suggested dataset: {recs['minimum']:,} samples")

    print(f"\n  RECOMMENDED (GOOD) Quality Level:")
    print(f"    Suggested dataset: {recs['recommended']:,} samples")
    print(f"    Chinchilla-optimal: {recs['chinchilla_optimal']:,} tokens/samples")

    print(f"\n  EXCELLENT Quality Level:")
    print(f"    Suggested dataset: {recs['excellent']:,} samples")


def main():
    print_section("STEP Model Data Requirements Analysis - Quick Demo")

    print("\nThis demo shows how to estimate training data requirements")
    print("for different STEP model architectures and tasks.")

    # =========================================================================
    # Example 1: Small Classification Model
    # =========================================================================
    print_section("Example 1: Small Classification Model")

    small_classifier = STEPForClassification(
        vocab_size=50000,
        num_classes=5,
        output_dim=256  # Small model
    )

    analyze_model(small_classifier, "Small Classifier (5 classes)", "classification")

    # =========================================================================
    # Example 2: Large Classification Model
    # =========================================================================
    print_section("Example 2: Large Classification Model")

    large_classifier = STEPForClassification(
        vocab_size=50000,
        num_classes=20,
        output_dim=2048  # Large model
    )

    analyze_model(large_classifier, "Large Classifier (20 classes)", "classification")

    # =========================================================================
    # Example 3: Property Prediction Model
    # =========================================================================
    print_section("Example 3: Property Prediction Model")

    property_model = STEPForPropertyPrediction(
        vocab_size=50000,
        num_properties=6,
        output_dim=1024
    )

    analyze_model(property_model, "Property Predictor (6 properties)", "property")

    # =========================================================================
    # Example 4: Comparison Across Tasks
    # =========================================================================
    print_section("Example 4: Same Model, Different Tasks")

    print("\nA single model architecture can be used for different tasks,")
    print("but data requirements vary significantly:")

    base_model = STEPForClassification(
        vocab_size=50000,
        num_classes=10,
        output_dim=1024
    )

    num_params = count_model_parameters(base_model, exclude_embeddings=True)

    print(f"\nModel: {num_params:,} non-embedding parameters")
    print("\nRecommended dataset sizes (GOOD quality):")

    tasks = {
        'classification': 'Part classification (10 classes)',
        'property': 'Property prediction',
        'captioning': 'Part description generation',
        'similarity': 'Similarity search'
    }

    for task_type, description in tasks.items():
        recs = suggest_dataset_size(base_model, task_type=task_type, quality_level='good')
        print(f"  {description:40s}: {recs['recommended']:>10,} samples")

    # =========================================================================
    # Key Takeaways
    # =========================================================================
    print_section("Key Takeaways")

    print("""
1. MODEL SIZE MATTERS:
   - Larger models need more data (parameter × multiplier)
   - Count non-embedding parameters for scaling laws
   - Chinchilla: ~20 tokens per parameter is optimal

2. TASK COMPLEXITY MATTERS:
   - Classification: 1K-5K samples per class
   - Property prediction: 10K-100K samples
   - Captioning: 50K-500K samples
   - More complex tasks need exponentially more data

3. QUALITY VS QUANTITY:
   - Start with minimum viable dataset
   - Scale up gradually while monitoring performance
   - High-quality data > large noisy dataset

4. TRANSFER LEARNING:
   - Fine-tuning pretrained models reduces requirements 5-10×
   - With pretrained encoder: 100-500 samples often sufficient
   - From scratch: Need 10-100× more data

5. SCALING LAWS:
   - Performance improves as power law with data size
   - Doubling data ≠ doubling performance
   - Diminishing returns are predictable
    """)

    # =========================================================================
    # Next Steps
    # =========================================================================
    print_section("Next Steps")

    print("""
To get more precise estimates for YOUR specific dataset:

1. QUICK ESTIMATE (what we just did):
   - Use suggest_dataset_size() for ballpark numbers
   - Based on literature and model size
   - Takes <1 second, no training required

2. EMPIRICAL ANALYSIS (recommended for production):
   - Generate learning curves with your actual data
   - Train on varying dataset sizes (10%, 20%, 50%, etc.)
   - Fit power law to observed performance
   - Extrapolate to predict requirements
   - See: examples/analyze_data_requirements.py

3. ITERATIVE COLLECTION:
   - Start with minimum viable dataset
   - Train and evaluate
   - Use learning curves to decide if more data needed
   - Collect in batches, re-analyze after each batch

Example commands:
    # Quick estimate (no training)
    python demo_data_requirements.py

    # Full analysis with your data
    python examples/analyze_data_requirements.py --mode full

    # Read the documentation
    cat DATA_REQUIREMENTS_ANALYSIS.md
    """)

    print_section("Demo Complete")

    print("\nFor more information:")
    print("  - Full documentation: DATA_REQUIREMENTS_ANALYSIS.md")
    print("  - Example script: examples/analyze_data_requirements.py")
    print("  - Task guidelines: DATA_REQUIREMENTS.md")


if __name__ == '__main__':
    main()
