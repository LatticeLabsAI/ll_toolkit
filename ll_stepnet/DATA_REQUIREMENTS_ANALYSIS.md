# Data Requirements Analysis Tool for LL-STEPNet

This document explains how to use the data requirements analysis tool to determine optimal training dataset sizes for STEP neural network models.

## Overview

The data requirements tool implements **transformer-specific scaling laws** and **empirical learning curve methods** to help you:

1. **Predict** how much training data you need for target performance
2. **Estimate** model performance at different dataset sizes
3. **Optimize** data collection efforts
4. **Understand** scaling behavior of your STEP models

## Theoretical Background

### Transformer Scaling Laws

Modern transformer models follow predictable power-law scaling relationships:

**OpenAI Scaling Law** (Kaplan et al., 2020):
```
L(D) = (D_c / D)^α_D
```

Where:
- `L(D)` = Loss given dataset size D
- `D_c` = Data scaling constant (~5.4 × 10^13 for GPT)
- `α_D` = Data scaling exponent (~0.095 for language models)

**Key Insight**: Loss decreases as a power law with dataset size. Doubling data doesn't double performance - gains diminish predictably.

**Chinchilla Scaling Laws** (Hoffmann et al., 2022):
- Optimal training: **~20-25 tokens per model parameter**
- 8× larger model requires only ~5× more data
- Transformers are more data-efficient than traditional networks

### Learning Curves

Learning curves plot model performance (y-axis) against training dataset size (x-axis). By training on multiple dataset sizes, we can:

1. Fit empirical power laws to observed data
2. Extrapolate to predict performance at larger sizes
3. Determine required dataset size for target accuracy

## Quick Start

### 1. Quick Estimate (No Training Required)

Get instant recommendations based on model size and literature:

```python
from stepnet import STEPForClassification, suggest_dataset_size, count_model_parameters

# Create your model
model = STEPForClassification(
    vocab_size=50000,
    num_classes=10,
    output_dim=1024
)

# Analyze parameters
num_params = count_model_parameters(model, exclude_embeddings=True)

# Get recommendations
recommendations = suggest_dataset_size(
    model,
    task_type='classification',
    quality_level='good'
)

print(f"Minimum: {recommendations['minimum']:,} samples")
print(f"Recommended: {recommendations['recommended']:,} samples")
print(f"Excellent: {recommendations['excellent']:,} samples")
```

**Output:**
```
Total parameters: 14,566,410
Embedding parameters: 12,800,000
Non-embedding parameters: 1,766,410

Dataset Size Recommendations for classification:
  Model parameters (non-embedding): 1,766,410
  Chinchilla-optimal tokens: 35,328,200
  Suggested minimum: 1,766 samples
  Recommended: 5,299 samples
  Excellent: 17,664 samples
```

### 2. Full Learning Curve Analysis

Generate empirical learning curves by training on varying dataset sizes:

```python
from stepnet import (
    STEPForClassification,
    STEPLearningCurveGenerator,
    STEPScalingLawAnalyzer,
    plot_learning_curve_with_scaling_law
)
from stepnet.data import STEPDataset

# Create datasets
train_dataset = STEPDataset(
    file_paths=train_files,
    labels=train_labels,
    max_length=2048
)

val_dataset = STEPDataset(
    file_paths=val_files,
    labels=val_labels,
    max_length=2048
)

# Initialize learning curve generator
generator = STEPLearningCurveGenerator(
    model_class=STEPForClassification,
    model_kwargs={
        'vocab_size': 50000,
        'num_classes': 10,
        'output_dim': 1024
    },
    train_kwargs={
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'batch_size': 8
    }
)

# Generate learning curve
results = generator.generate_learning_curve(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    sample_fractions=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    n_iterations=3,
    save_dir='learning_curves'
)

# Analyze scaling laws
analyzer = STEPScalingLawAnalyzer()

sample_sizes = results['sample_sizes']
val_losses = results['val_losses']
mean_val_losses = np.mean(val_losses, axis=1)

# Fit power law
fitted_params = analyzer.fit_power_law(
    sample_sizes,
    mean_val_losses,
    law_type='openai'
)

# Predict required samples for target loss
target_loss = 0.2
required_samples = analyzer.predict_required_samples(target_loss)

print(f"To achieve loss of {target_loss:.4f}:")
print(f"  Estimated samples needed: {required_samples:,}")

# Visualize
plot_learning_curve_with_scaling_law(
    sample_sizes=sample_sizes,
    val_losses=val_losses,
    val_accuracies=results['val_accuracies'],
    fitted_params=fitted_params,
    fit_type='openai',
    extrapolate_to=required_samples,
    save_path='learning_curves/scaling_law_plot.png'
)
```

## API Reference

### STEPLearningCurveGenerator

Generates learning curves by training models on varying dataset sizes.

**Methods:**

#### `__init__(model_class, model_kwargs, train_kwargs=None, device='cuda')`

Initialize the generator.

**Args:**
- `model_class`: STEP model class (e.g., `STEPForClassification`)
- `model_kwargs`: Model initialization arguments
- `train_kwargs`: Training configuration (epochs, lr, batch_size)
- `device`: Device to train on

#### `generate_learning_curve(train_dataset, val_dataset, sample_fractions, n_iterations=3, save_dir=None)`

Generate learning curve data.

**Args:**
- `train_dataset`: Full training dataset
- `val_dataset`: Validation dataset
- `sample_fractions`: List of fractions of data to use (e.g., [0.1, 0.5, 1.0])
- `n_iterations`: Number of training runs per size
- `save_dir`: Optional directory to save results

**Returns:**
- Dictionary with `sample_sizes`, `train_losses`, `val_losses`, `val_accuracies`, `train_times`

### STEPScalingLawAnalyzer

Analyzes scaling laws and predicts data requirements.

**Methods:**

#### `fit_power_law(sample_sizes, losses, law_type='openai')`

Fit power law to learning curve data.

**Args:**
- `sample_sizes`: Array of dataset sizes
- `losses`: Corresponding validation losses
- `law_type`: `'openai'` or `'standard'`

**Returns:**
- Dictionary of fitted parameters

#### `predict_required_samples(target_loss)`

Predict dataset size needed for target loss.

**Args:**
- `target_loss`: Desired target loss

**Returns:**
- Estimated required sample size

#### `extrapolate_performance(target_size)`

Predict performance at a given dataset size.

**Args:**
- `target_size`: Dataset size to predict for

**Returns:**
- Predicted loss at target_size

### Utility Functions

#### `suggest_dataset_size(model, task_type='classification', quality_level='good')`

Get dataset size recommendations based on model and task.

**Args:**
- `model`: STEP model
- `task_type`: `'classification'`, `'property'`, `'captioning'`, `'similarity'`
- `quality_level`: `'minimum'`, `'good'`, or `'excellent'`

**Returns:**
- Dictionary with recommended dataset sizes

#### `count_model_parameters(model, exclude_embeddings=True)`

Count model parameters (excluding embeddings).

**Args:**
- `model`: PyTorch model
- `exclude_embeddings`: Whether to exclude embedding parameters

**Returns:**
- Number of parameters

#### `chinchilla_optimal_tokens(num_params)`

Estimate optimal training tokens based on Chinchilla scaling laws.

**Args:**
- `num_params`: Number of model parameters

**Returns:**
- Recommended number of training tokens (~20 tokens/param)

## Task-Specific Guidelines

### Classification

**Typical Requirements:**
- **Minimum:** 100-500 samples per class
- **Good:** 1,000-5,000 samples per class
- **Excellent:** 10,000+ samples per class

**Fine-tuning with pretrained model:**
- Can work with 100-500 samples total
- Transfer learning reduces requirements by 5-10×

**Example:**
```python
recommendations = suggest_dataset_size(
    model,
    task_type='classification',
    quality_level='good'
)
# Returns: ~1,000 samples per class minimum
```

### Property Prediction

**Typical Requirements:**
- **Minimum:** 1,000 samples
- **Good:** 10,000+ samples
- **Excellent:** 100,000+ samples

Property prediction (volume, mass, etc.) requires more data than classification due to continuous outputs and higher variance.

### Captioning

**Typical Requirements:**
- **Minimum:** 5,000 samples
- **Good:** 50,000+ samples
- **Excellent:** 500,000+ samples

Captioning is the most data-intensive task, requiring diverse examples to learn language generation.

### Similarity Learning

**Typical Requirements:**
- **Minimum:** 10,000 pairs
- **Good:** 100,000+ pairs
- **Excellent:** 1,000,000+ pairs

Self-supervised approaches can work with unlabeled data, significantly reducing annotation burden.

## Interpreting Results

### Learning Curve Shape

**Underfitting (High Bias):**
- Both training and validation losses are high
- Curves plateau at poor performance
- **Solution:** Use more complex model or better features

**Overfitting (High Variance):**
- Large gap between training and validation losses
- Training loss low, validation loss high
- **Solution:** Collect more data, add regularization, or simplify model

**Good Fit:**
- Small gap between curves
- Both losses plateau at acceptable level
- Diminishing returns from more data

### Scaling Law Parameters

**α_D (Data scaling exponent):**
- Typical range: 0.05 - 0.15
- Smaller α_D = slower improvement with more data
- GPT models: ~0.095
- If your α_D > 0.2, model may be too small for the task

**D_c (Data constant):**
- Large D_c indicates task requires substantial data
- Compare with Chinchilla-optimal estimates

## Example Workflow

### Complete Data Requirements Analysis

```python
#!/usr/bin/env python
"""
Complete workflow for analyzing data requirements.
"""

from stepnet import *
import numpy as np

# 1. Load your data
train_files, train_labels = load_dataset_from_directory('data', 'train', 'labels.json')
val_files, val_labels = load_dataset_from_directory('data', 'val', 'labels.json')

# 2. Create model
model = STEPForClassification(num_classes=10)

# 3. Quick estimate
print("=== QUICK ESTIMATE ===")
recommendations = suggest_dataset_size(model, task_type='classification')
print(f"Recommended dataset: {recommendations['recommended']:,} samples")

# 4. Generate learning curve (if you have time and compute)
print("\n=== GENERATING LEARNING CURVE ===")

train_dataset = STEPDataset(train_files, train_labels)
val_dataset = STEPDataset(val_files, val_labels)

generator = STEPLearningCurveGenerator(
    model_class=STEPForClassification,
    model_kwargs={'num_classes': 10}
)

results = generator.generate_learning_curve(
    train_dataset, val_dataset,
    sample_fractions=[0.2, 0.4, 0.6, 0.8, 1.0],
    n_iterations=3
)

# 5. Fit scaling law
analyzer = STEPScalingLawAnalyzer()
fitted_params = analyzer.fit_power_law(
    results['sample_sizes'],
    np.mean(results['val_losses'], axis=1)
)

# 6. Predict requirements
target_accuracy = 0.95
target_loss = 1 - target_accuracy
required = analyzer.predict_required_samples(target_loss)

print(f"\n=== PREDICTIONS ===")
print(f"For {target_accuracy:.1%} accuracy, need ~{required:,} samples")

# 7. Visualize
plot_learning_curve_with_scaling_law(
    sample_sizes=results['sample_sizes'],
    val_losses=results['val_losses'],
    val_accuracies=results['val_accuracies'],
    fitted_params=fitted_params,
    save_path='analysis_results.png'
)
```

## Tips and Best Practices

### Learning Curve Generation

1. **Use enough sample points**: Test at least 5-7 different dataset sizes
2. **Use log spacing**: `[0.05, 0.1, 0.2, 0.4, 0.8, 1.0]` covers range efficiently
3. **Multiple iterations**: Run 3-5 iterations per size for reliable statistics
4. **Early stopping**: Use patience to avoid overfitting on small datasets

### Scaling Law Fitting

1. **Check fit quality**: Plot fitted curve over data points
2. **Don't extrapolate too far**: Predictions beyond 2-3× largest size are unreliable
3. **Compare with literature**: Your α_D should be similar to related models
4. **Consider task complexity**: Some tasks inherently require more data

### Dataset Collection Strategy

1. **Start with quick estimates**: Get ballpark numbers before collecting data
2. **Collect in batches**: Gather data incrementally and re-analyze
3. **Prioritize quality**: 1,000 high-quality samples > 10,000 noisy ones
4. **Balance classes**: Equal samples per class for classification
5. **Use augmentation**: Can effectively increase dataset size 2-5×

## Common Issues

### "Fitting failed"

**Causes:**
- Too few data points (need 5+ sample sizes)
- Non-monotonic learning curve (increase n_iterations)
- Insufficient variation in dataset sizes

**Solutions:**
- Use more sample points
- Increase training iterations for stability
- Ensure sample sizes span 1-2 orders of magnitude

### "Predicted sample size is unrealistic"

**Causes:**
- Target performance exceeds model capacity
- Power law extrapolation beyond reliable range
- Poor fit to initial learning curve

**Solutions:**
- Check if target is achievable (compare with Bayes error)
- Don't extrapolate beyond 3× largest tested size
- Improve learning curve fit by testing more sizes

### "Learning curve is flat"

**Causes:**
- Model too simple for task
- Features not informative enough
- Already at Bayes error rate

**Solutions:**
- Use larger model architecture
- Improve feature engineering
- Verify labels are correct and consistent

## References

### Research Papers

1. **OpenAI Scaling Laws** (Kaplan et al., 2020)
   "Scaling Laws for Neural Language Models"
   https://arxiv.org/abs/2001.08361

2. **Chinchilla Scaling Laws** (Hoffmann et al., 2022)
   "Training Compute-Optimal Large Language Models"
   https://arxiv.org/abs/2203.15556

3. **BERT Fine-tuning** (Devlin et al., 2019)
   Sample requirements for transformer fine-tuning
   https://arxiv.org/abs/1810.04805

### Implementation Resources

- Keras learning curve examples: https://keras.io/examples/keras_recipes/sample_size_estimate/
- Scikit-learn learning curves: https://scikit-learn.org/stable/modules/learning_curve.html
- PyTorch scaling: https://pytorch.org/blog/scaling-multimodal-foundation-models/

## Support

For questions or issues with the data requirements tool:

1. Check this documentation
2. Review example scripts in `examples/analyze_data_requirements.py`
3. See `DATA_REQUIREMENTS.md` for task-specific guidelines
4. Open an issue on GitHub with your learning curve plot

---

**Remember:** Data requirements analysis is a tool to guide decision-making, not an exact science. Use it to establish reasonable ranges, then validate with empirical testing on your specific dataset and task.
