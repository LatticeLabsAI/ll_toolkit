"""
Data Requirement Determination Tool for STEP Models.

This module implements transformer-specific scaling laws and learning curve analysis
to determine training data requirements for STEP neural network models.

Based on:
- OpenAI Scaling Laws (Kaplan et al., 2020)
- Chinchilla Scaling Laws (Hoffmann et al., 2022)
- Empirical learning curve methods

Integrates with ll_stepnet training infrastructure.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np

# Lazy imports for optional heavy dependencies
plt = None  # matplotlib.pyplot, imported lazily
curve_fit = None  # scipy.optimize.curve_fit, imported lazily


def _ensure_matplotlib():
    """Lazily import matplotlib.pyplot."""
    global plt
    if plt is None:
        import matplotlib.pyplot as _plt
        plt = _plt
    return plt


def _ensure_scipy():
    """Lazily import scipy.optimize.curve_fit."""
    global curve_fit
    if curve_fit is None:
        from scipy.optimize import curve_fit as _curve_fit
        curve_fit = _curve_fit
    return curve_fit
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
from tqdm import tqdm
import json
from datetime import datetime

from .trainer import STEPTrainer
from .data import STEPDataset, STEPCollator, create_dataloader


# ============================================================================
# Scaling Law Functions
# ============================================================================

def power_law_loss(D: np.ndarray, D_c: float, alpha_D: float) -> np.ndarray:
    """
    OpenAI-style power law for loss as a function of dataset size.

    L(D) = (D_c / D)^alpha_D

    Args:
        D: Dataset size (number of samples or tokens)
        D_c: Data scaling constant
        alpha_D: Data scaling exponent (~0.095 for language models)

    Returns:
        Loss values
    """
    return np.power(D_c / D, alpha_D)


def power_law_error(n: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Power law for error as a function of dataset size.

    Error(n) = a * n^(-b) + c

    Args:
        n: Dataset size
        a: Scaling coefficient
        b: Scaling exponent
        c: Irreducible error (Bayes error rate)

    Returns:
        Error values
    """
    return a * np.power(n, -b) + c


def inverse_power_law_accuracy(D: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Inverse power law for accuracy.

    Acc(D) = c - a * D^(-b)

    Args:
        D: Dataset size
        a: Scaling coefficient
        b: Scaling exponent
        c: Maximum achievable accuracy (asymptote)

    Returns:
        Accuracy values
    """
    return c - a * np.power(D, -b)


def chinchilla_optimal_tokens(num_params: int) -> int:
    """
    Estimate optimal number of training tokens based on Chinchilla scaling laws.

    Rule: ~20-25 tokens per parameter for compute-optimal training.

    Args:
        num_params: Number of model parameters (non-embedding)

    Returns:
        Recommended number of training tokens/samples
    """
    tokens_per_param = 20  # Conservative estimate
    return int(num_params * tokens_per_param)


# ============================================================================
# Learning Curve Generator
# ============================================================================

class STEPLearningCurveGenerator:
    """
    Generate learning curves for STEP models to determine data requirements.

    This class trains models on varying dataset sizes and measures performance
    to establish empirical scaling relationships.
    """

    def __init__(
        self,
        model_class: type,
        model_kwargs: Dict,
        train_kwargs: Optional[Dict] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model_class: STEP model class (e.g., STEPForClassification)
            model_kwargs: Model initialization arguments
            train_kwargs: Training configuration (epochs, lr, etc.)
            device: Device to train on
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.device = device

        # Default training configuration
        self.train_kwargs = train_kwargs or {
            'num_epochs': 10,
            'learning_rate': 1e-4,
            'batch_size': 8,
            'patience': 3  # Early stopping patience
        }

        self.results = {
            'sample_sizes': [],
            'train_losses': [],
            'val_losses': [],
            'val_accuracies': [],
            'train_times': []
        }

    def generate_learning_curve(
        self,
        train_dataset: STEPDataset,
        val_dataset: STEPDataset,
        sample_fractions: List[float],
        n_iterations: int = 3,
        save_dir: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate learning curve by training on varying dataset sizes.

        Args:
            train_dataset: Full training dataset
            val_dataset: Validation dataset
            sample_fractions: List of fractions of training data to use (e.g., [0.1, 0.2, 0.5, 1.0])
            n_iterations: Number of training runs per sample size
            save_dir: Optional directory to save checkpoints

        Returns:
            Dictionary with learning curve data
        """
        total_samples = len(train_dataset)

        for fraction in sample_fractions:
            n_samples = int(total_samples * fraction)
            self.results['sample_sizes'].append(n_samples)

            print(f"\n{'='*70}")
            print(f"Training with {n_samples} samples ({fraction*100:.1f}% of data)")
            print(f"{'='*70}")

            iter_train_losses = []
            iter_val_losses = []
            iter_val_accs = []
            iter_times = []

            for iteration in range(n_iterations):
                print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")

                # Create random subset
                indices = np.random.choice(total_samples, size=n_samples, replace=False)
                subset = Subset(train_dataset, indices.tolist())

                # Create dataloaders
                collate_fn = STEPCollator()
                train_loader = DataLoader(
                    subset,
                    batch_size=self.train_kwargs['batch_size'],
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True,
                    persistent_workers=True,
                    collate_fn=collate_fn
                )

                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.train_kwargs['batch_size'],
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True,
                    persistent_workers=True,
                    collate_fn=collate_fn
                )

                # Create fresh model
                model = self.model_class(**self.model_kwargs).to(self.device)

                # Train
                import time
                start_time = time.time()

                train_loss, val_loss, val_acc = self._train_model(
                    model, train_loader, val_loader
                )

                elapsed_time = time.time() - start_time

                # Record results
                iter_train_losses.append(train_loss)
                iter_val_losses.append(val_loss)
                iter_val_accs.append(val_acc)
                iter_times.append(elapsed_time)

                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                if val_acc is not None:
                    print(f"  Val Accuracy: {val_acc:.4f}")
                print(f"  Time: {elapsed_time:.1f}s")

            # Store iteration results
            self.results['train_losses'].append(iter_train_losses)
            self.results['val_losses'].append(iter_val_losses)
            self.results['val_accuracies'].append(iter_val_accs)
            self.results['train_times'].append(iter_times)

        # Convert to numpy arrays
        results_np = {
            'sample_sizes': np.array(self.results['sample_sizes']),
            'train_losses': np.array(self.results['train_losses']),
            'val_losses': np.array(self.results['val_losses']),
            'val_accuracies': np.array(self.results['val_accuracies']),
            'train_times': np.array(self.results['train_times'])
        }

        # Save results
        if save_dir:
            self._save_results(results_np, save_dir)

        return results_np

    def _train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Tuple[float, float, Optional[float]]:
        """
        Train a single model and return final metrics.

        Returns:
            (train_loss, val_loss, val_accuracy)
        """
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.train_kwargs['learning_rate']
        )

        # Determine loss function based on model type
        model_name = model.__class__.__name__
        if 'Classification' in model_name:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        num_epochs = self.train_kwargs['num_epochs']
        patience = self.train_kwargs.get('patience', 3)
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                token_ids = batch['token_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                topology_data = batch.get('topology_data', None)

                optimizer.zero_grad()
                outputs = model(token_ids, topology_data=topology_data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            val_loss, val_acc = self._validate(model, val_loader, criterion)
            print(f"  Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}", flush=True)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

        return train_loss, best_val_loss, val_acc

    @torch.no_grad()
    def _validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, Optional[float]]:
        """
        Validate model and return metrics.

        Returns:
            (val_loss, val_accuracy)
        """
        model.eval()
        val_loss = 0.0
        total_correct = 0
        total_samples = 0

        model_name = model.__class__.__name__
        is_classification = 'Classification' in model_name

        for batch in val_loader:
            token_ids = batch['token_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            topology_data = batch.get('topology_data', None)

            outputs = model(token_ids, topology_data=topology_data)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            if is_classification:
                preds = torch.argmax(outputs, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = total_correct / total_samples if is_classification else None

        return val_loss, val_acc

    def _save_results(self, results: Dict, save_dir: str):
        """Save learning curve results to disk."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as numpy arrays
        np.savez(
            save_path / f'learning_curve_{timestamp}.npz',
            **results
        )

        # Save summary as JSON
        summary = {
            'sample_sizes': results['sample_sizes'].tolist(),
            'mean_val_losses': np.mean(results['val_losses'], axis=1).tolist(),
            'std_val_losses': np.std(results['val_losses'], axis=1).tolist()
        }

        with open(save_path / f'learning_curve_summary_{timestamp}.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {save_path}")


# ============================================================================
# Scaling Law Analyzer
# ============================================================================

class STEPScalingLawAnalyzer:
    """
    Analyze scaling laws for STEP models and predict data requirements.

    Fits power law relationships to learning curve data and extrapolates
    to estimate required dataset sizes for target performance.
    """

    def __init__(self):
        self.fitted_params = None
        self.fit_type = None

    def fit_power_law(
        self,
        sample_sizes: np.ndarray,
        losses: np.ndarray,
        law_type: str = 'openai'
    ) -> Dict[str, float]:
        """
        Fit power law to learning curve data.

        Args:
            sample_sizes: Array of dataset sizes
            losses: Corresponding validation losses
            law_type: 'openai' for L(D) = (D_c/D)^alpha_D or 'standard' for Error = a*n^(-b) + c

        Returns:
            Dictionary of fitted parameters
        """
        self.fit_type = law_type

        try:
            _curve_fit = _ensure_scipy()
            if law_type == 'openai':
                # Fit: L(D) = (D_c / D)^alpha_D
                params, _ = _curve_fit(
                    power_law_loss,
                    sample_sizes,
                    losses,
                    p0=[1e10, 0.1],
                    bounds=([1e5, 0.01], [1e15, 0.5]),
                    maxfev=10000
                )

                self.fitted_params = {
                    'D_c': params[0],
                    'alpha_D': params[1]
                }

                print(f"\nFitted OpenAI Scaling Law:")
                print(f"  D_c (data constant) = {params[0]:.2e}")
                print(f"  α_D (data exponent) = {params[1]:.4f}")
                print(f"  (Reference: GPT α_D ≈ 0.095)")

            elif law_type == 'standard':
                # Fit: Error(n) = a * n^(-b) + c
                params, _ = _curve_fit(
                    power_law_error,
                    sample_sizes,
                    losses,
                    p0=[1.0, 0.3, 0.01],
                    bounds=([0, 0, 0], [np.inf, 1, 1]),
                    maxfev=10000
                )

                self.fitted_params = {
                    'a': params[0],
                    'b': params[1],
                    'c': params[2]
                }

                print(f"\nFitted Standard Power Law:")
                print(f"  a (scaling coefficient) = {params[0]:.4f}")
                print(f"  b (scaling exponent) = {params[1]:.4f}")
                print(f"  c (irreducible error) = {params[2]:.4f}")

            else:
                raise ValueError(f"Unknown law_type: {law_type}")

            return self.fitted_params

        except Exception as e:
            print(f"Power law fitting failed: {e}")
            return None

    def predict_required_samples(
        self,
        target_loss: float,
        current_sizes: Optional[np.ndarray] = None,
        current_losses: Optional[np.ndarray] = None
    ) -> int:
        """
        Predict number of samples needed to achieve target loss.

        Args:
            target_loss: Desired target loss
            current_sizes: Optional array of current dataset sizes (for fitting)
            current_losses: Optional array of current losses (for fitting)

        Returns:
            Estimated required sample size
        """
        # Fit if needed
        if self.fitted_params is None:
            if current_sizes is None or current_losses is None:
                raise ValueError("Must fit power law first or provide current data")
            self.fit_power_law(current_sizes, current_losses)

        if self.fitted_params is None:
            return None

        try:
            if self.fit_type == 'openai':
                # Solve: target_loss = (D_c / D)^alpha_D
                # D = D_c / (target_loss^(1/alpha_D))
                D_c = self.fitted_params['D_c']
                alpha_D = self.fitted_params['alpha_D']

                required_samples = D_c / np.power(target_loss, 1 / alpha_D)

            elif self.fit_type == 'standard':
                # Solve: target_loss = a * n^(-b) + c
                # n = ((target_loss - c) / a)^(-1/b)
                a = self.fitted_params['a']
                b = self.fitted_params['b']
                c = self.fitted_params['c']

                if target_loss <= c:
                    print(f"Warning: Target loss {target_loss:.4f} may not be achievable")
                    print(f"Minimum error (Bayes rate): {c:.4f}")
                    return None

                required_samples = np.power((target_loss - c) / a, -1 / b)

            print(f"\nTo achieve loss of {target_loss:.4f}:")
            print(f"  Estimated samples needed: {int(required_samples):,}")

            return int(required_samples)

        except Exception as e:
            print(f"Sample prediction failed: {e}")
            return None

    def extrapolate_performance(
        self,
        target_size: int,
        current_sizes: Optional[np.ndarray] = None,
        current_losses: Optional[np.ndarray] = None
    ) -> float:
        """
        Predict performance at a given dataset size.

        Args:
            target_size: Dataset size to predict for
            current_sizes: Optional current dataset sizes (for fitting)
            current_losses: Optional current losses (for fitting)

        Returns:
            Predicted loss at target_size
        """
        # Fit if needed
        if self.fitted_params is None:
            if current_sizes is None or current_losses is None:
                raise ValueError("Must fit power law first or provide current data")
            self.fit_power_law(current_sizes, current_losses)

        if self.fitted_params is None:
            return None

        if self.fit_type == 'openai':
            D_c = self.fitted_params['D_c']
            alpha_D = self.fitted_params['alpha_D']
            predicted_loss = power_law_loss(target_size, D_c, alpha_D)

        elif self.fit_type == 'standard':
            a = self.fitted_params['a']
            b = self.fitted_params['b']
            c = self.fitted_params['c']
            predicted_loss = power_law_error(target_size, a, b, c)

        print(f"\nPredicted loss at {target_size:,} samples: {predicted_loss:.4f}")

        return predicted_loss


# ============================================================================
# Visualization
# ============================================================================

def plot_learning_curve_with_scaling_law(
    sample_sizes: np.ndarray,
    val_losses: np.ndarray,
    val_accuracies: Optional[np.ndarray] = None,
    fitted_params: Optional[Dict] = None,
    fit_type: str = 'openai',
    extrapolate_to: Optional[int] = None,
    save_path: Optional[str] = None
):
    """
    Plot learning curves with fitted scaling law.

    Args:
        sample_sizes: Array of dataset sizes
        val_losses: Validation losses [num_sizes, num_iterations]
        val_accuracies: Optional validation accuracies
        fitted_params: Fitted power law parameters
        fit_type: 'openai' or 'standard'
        extrapolate_to: Optional target size for extrapolation
        save_path: Optional path to save figure
    """
    # Calculate statistics
    loss_mean = np.mean(val_losses, axis=1)
    loss_std = np.std(val_losses, axis=1)

    # Ensure matplotlib is available
    _plt = _ensure_matplotlib()

    # Create figure
    if val_accuracies is not None:
        fig, (ax1, ax2) = _plt.subplots(1, 2, figsize=(16, 6))
    else:
        fig, ax1 = _plt.subplots(1, 1, figsize=(10, 6))
        ax2 = None

    # Plot 1: Loss vs Dataset Size
    ax1.errorbar(
        sample_sizes, loss_mean, yerr=loss_std,
        fmt='o-', capsize=5, linewidth=2, markersize=8,
        label='Measured Loss', color='steelblue'
    )

    # Plot fitted curve
    if fitted_params is not None:
        if extrapolate_to is not None:
            x_smooth = np.linspace(sample_sizes[0], extrapolate_to, 200)
        else:
            x_smooth = np.linspace(sample_sizes[0], sample_sizes[-1] * 1.5, 200)

        if fit_type == 'openai':
            D_c = fitted_params['D_c']
            alpha_D = fitted_params['alpha_D']
            y_smooth = power_law_loss(x_smooth, D_c, alpha_D)

            ax1.plot(
                x_smooth, y_smooth, '--', linewidth=2, alpha=0.7,
                label=f'L(D) = ({D_c:.1e}/D)^{alpha_D:.3f}',
                color='coral'
            )

            if extrapolate_to is not None:
                pred_loss = power_law_loss(extrapolate_to, D_c, alpha_D)
                ax1.plot(
                    extrapolate_to, pred_loss, 'r*',
                    markersize=20, label=f'Predicted at {extrapolate_to:,}',
                    zorder=5
                )

        elif fit_type == 'standard':
            a = fitted_params['a']
            b = fitted_params['b']
            c = fitted_params['c']
            y_smooth = power_law_error(x_smooth, a, b, c)

            ax1.plot(
                x_smooth, y_smooth, '--', linewidth=2, alpha=0.7,
                label=f'Error = {a:.2f}·D^(-{b:.2f}) + {c:.2f}',
                color='coral'
            )

    ax1.set_xlabel('Training Dataset Size', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Validation Loss', fontsize=13, fontweight='bold')
    ax1.set_title('STEP Model: Loss vs Dataset Size\n(Transformer Scaling Law)',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Plot 2: Accuracy vs Dataset Size (if provided)
    if ax2 is not None and val_accuracies is not None:
        acc_mean = np.mean(val_accuracies, axis=1)
        acc_std = np.std(val_accuracies, axis=1)

        ax2.errorbar(
            sample_sizes, acc_mean, yerr=acc_std,
            fmt='s-', capsize=5, linewidth=2, markersize=8,
            label='Measured Accuracy', color='forestgreen'
        )

        # Fit inverse power law to accuracy
        try:
            _curve_fit = _ensure_scipy()
            params, _ = _curve_fit(
                inverse_power_law_accuracy,
                sample_sizes,
                acc_mean,
                p0=[0.5, 0.1, 1.0],
                bounds=([0, 0, 0.5], [2, 1, 1.0]),
                maxfev=10000
            )

            a, b, c = params

            if extrapolate_to is not None:
                x_smooth_acc = np.linspace(sample_sizes[0], extrapolate_to, 200)
            else:
                x_smooth_acc = np.linspace(sample_sizes[0], sample_sizes[-1] * 1.5, 200)

            y_smooth_acc = inverse_power_law_accuracy(x_smooth_acc, a, b, c)

            ax2.plot(
                x_smooth_acc, y_smooth_acc, '--', linewidth=2, alpha=0.7,
                label=f'Acc = {c:.3f} - {a:.3f}·D^(-{b:.3f})',
                color='coral'
            )

            print(f"\nAccuracy Scaling Law:")
            print(f"  Maximum achievable accuracy: {c:.4f}")

        except Exception as e:
            print(f"Accuracy fit failed: {e}")

        ax2.set_xlabel('Training Dataset Size', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Validation Accuracy', fontsize=13, fontweight='bold')
        ax2.set_title('STEP Model: Accuracy vs Dataset Size',
                      fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')

    _plt.tight_layout()

    if save_path:
        _plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to {save_path}")

    _plt.show()


# ============================================================================
# Convenience Functions
# ============================================================================

def estimate_data_requirements(
    model: nn.Module,
    target_accuracy: float,
    sample_sizes: np.ndarray,
    val_accuracies: np.ndarray
) -> int:
    """
    Convenience function to estimate required dataset size for target accuracy.

    Args:
        model: STEP model
        target_accuracy: Desired accuracy
        sample_sizes: Current dataset sizes tested
        val_accuracies: Corresponding validation accuracies

    Returns:
        Estimated required dataset size
    """
    # Convert accuracy to loss
    val_losses = 1 - np.mean(val_accuracies, axis=1)
    target_loss = 1 - target_accuracy

    analyzer = STEPScalingLawAnalyzer()
    analyzer.fit_power_law(sample_sizes, val_losses, law_type='openai')

    required_samples = analyzer.predict_required_samples(target_loss)

    return required_samples


def count_model_parameters(model: nn.Module, exclude_embeddings: bool = True) -> int:
    """
    Count model parameters (excluding embeddings for scaling law analysis).

    Args:
        model: PyTorch model
        exclude_embeddings: Whether to exclude embedding parameters

    Returns:
        Number of parameters
    """
    total_params = sum(p.numel() for p in model.parameters())

    if exclude_embeddings:
        embedding_params = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                embedding_params += sum(p.numel() for p in module.parameters())

        non_embedding_params = total_params - embedding_params

        print(f"Total parameters: {total_params:,}")
        print(f"Embedding parameters: {embedding_params:,}")
        print(f"Non-embedding parameters: {non_embedding_params:,}")

        return non_embedding_params

    return total_params


def suggest_dataset_size(
    model: nn.Module,
    task_type: str = 'classification',
    quality_level: str = 'good'
) -> Dict[str, int]:
    """
    Suggest dataset size based on model size and task complexity.

    Based on research guidelines:
    - Classification: 1,000-5,000 samples per class
    - Property prediction: 10,000-100,000 samples
    - Fine-tuning: 100-1,000 samples (with pretrained model)

    Args:
        model: STEP model
        task_type: 'classification', 'property', 'captioning', etc.
        quality_level: 'minimum', 'good', or 'excellent'

    Returns:
        Dictionary with recommended dataset sizes
    """
    num_params = count_model_parameters(model, exclude_embeddings=True)

    # Chinchilla-optimal tokens
    optimal_tokens = chinchilla_optimal_tokens(num_params)

    # Task-specific recommendations
    recommendations = {
        'classification': {
            'minimum': 100,
            'good': 1000,
            'excellent': 5000
        },
        'property': {
            'minimum': 1000,
            'good': 10000,
            'excellent': 100000
        },
        'captioning': {
            'minimum': 5000,
            'good': 50000,
            'excellent': 500000
        },
        'similarity': {
            'minimum': 10000,
            'good': 100000,
            'excellent': 1000000
        }
    }

    base_size = recommendations.get(task_type, recommendations['classification'])[quality_level]

    # Adjust based on model size (larger models need more data)
    param_multiplier = max(1.0, num_params / 1e6)  # Scale up for models > 1M params

    suggested_size = int(base_size * param_multiplier)

    print(f"\nDataset Size Recommendations for {task_type}:")
    print(f"  Model parameters (non-embedding): {num_params:,}")
    print(f"  Chinchilla-optimal tokens: {optimal_tokens:,}")
    print(f"  Suggested minimum: {suggested_size:,} samples")
    print(f"  Recommended: {int(suggested_size * 3):,} samples")
    print(f"  Excellent: {int(suggested_size * 10):,} samples")

    return {
        'minimum': suggested_size,
        'recommended': int(suggested_size * 3),
        'excellent': int(suggested_size * 10),
        'chinchilla_optimal': optimal_tokens
    }
