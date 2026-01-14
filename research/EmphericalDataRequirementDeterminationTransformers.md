Now I'll create a comprehensive guide with transformer-specific implementations.

# Empirical Learning Curves for Transformers: Complete Guide with PyTorch

## Overview

Transformers have unique scaling properties compared to traditional neural networks. This guide provides **transformer-specific** implementations for determining training data requirements using empirical learning curves and scaling laws.

## Transformer Scaling Laws: The Mathematics

### OpenAI Scaling Laws [GPT](1)[2][3]

**Loss as a function of parameters** (\(N\)):

\[
L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}
\]

Where:

- \(N\) = non-embedding parameters
- \(N_c \approx 8.8 \times 10^{13}\) (scaling constant)
- \(\alpha_N \approx 0.076\) (parameter scaling exponent)

**Loss as a function of data** (\(D\)):

\[
L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}
\]

Where:

- \(D\) = number of tokens in training dataset
- \(D_c \approx 5.4 \times 10^{13}\) (data scaling constant)
- \(\alpha_D \approx 0.095\) (data scaling exponent)

**Combined scaling law**:[3][1]

\[
L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\frac{\alpha_N}{\alpha_D}} + \frac{D_c}{D}\right]^{\alpha_D}
\]

### Chinchilla Scaling Laws[2][4][5]

**Compute-optimal training**: For a given compute budget \(C\), the optimal model size and dataset size scale as:

\[
N_{opt} \propto C^{0.5} \quad \text{and} \quad D_{opt} \propto C^{0.5}
\]

**Practical ratio**: Approximately **20-25 tokens per parameter** for optimal training.[4][6][7]

### Key Insight for Transformers

**8× model size requires only ~5× more data** to avoid overfitting penalties, demonstrating transformers are more **data-efficient** than traditional networks.[2][3]

## Implementation 1: Transformer Learning Curves with Hugging Face

### Fine-tuning BERT with Varying Dataset Sizes

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

class TransformerLearningCurve:
    """
    Generate learning curves for transformer models (BERT, GPT, etc.)
    using Hugging Face Transformers.
    """
    
    def __init__(self, model_name, num_labels, task_name="classification"):
        """
        Args:
            model_name: HuggingFace model identifier (e.g., 'bert-base-uncased')
            num_labels: Number of output classes
            task_name: Task identifier for logging
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.task_name = task_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def tokenize_function(self, examples, text_column='text'):
        """Tokenize text examples."""
        return self.tokenizer(
            examples[text_column],
            padding='max_length',
            truncation=True,
            max_length=128
        )
    
    def compute_metrics(self, eval_pred):
        """Compute accuracy and F1 score."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1': f1
        }
    
    def train_model(self, train_dataset, eval_dataset, 
                   num_epochs=3, batch_size=16, learning_rate=2e-5):
        """
        Train a fresh model on the given dataset.
        
        Args:
            train_dataset: HuggingFace Dataset for training
            eval_dataset: HuggingFace Dataset for evaluation
            num_epochs: Training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            
        Returns:
            Dictionary with train and eval metrics
        """
        # Initialize fresh model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f'./results_{self.task_name}',
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            evaluation_strategy='epoch',
            save_strategy='no',  # Don't save checkpoints
            load_best_model_at_end=False,
            disable_tqdm=True,
            report_to='none'
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics
        )
        
        # Train
        train_result = trainer.train()
        
        # Evaluate
        eval_result = trainer.evaluate()
        
        return {
            'train_loss': train_result.training_loss,
            'eval_accuracy': eval_result['eval_accuracy'],
            'eval_f1': eval_result['eval_f1'],
            'eval_loss': eval_result['eval_loss']
        }
    
    def generate_learning_curve(self, full_dataset, eval_dataset,
                               sample_sizes, n_iterations=3,
                               num_epochs=3, batch_size=16):
        """
        Generate learning curve by training on varying dataset sizes.
        
        Args:
            full_dataset: Complete tokenized training dataset
            eval_dataset: Tokenized evaluation dataset
            sample_sizes: List of dataset sizes (can be fractions or absolutes)
            n_iterations: Number of training runs per sample size
            num_epochs: Epochs per training run
            batch_size: Batch size
            
        Returns:
            Tuple of (sample_sizes, train_losses, eval_accuracies, eval_losses)
        """
        total_samples = len(full_dataset)
        
        results = {
            'sample_sizes': [],
            'train_losses': [],
            'eval_accuracies': [],
            'eval_f1s': [],
            'eval_losses': []
        }
        
        for size in sample_sizes:
            # Convert fraction to absolute size
            if size <= 1.0:
                n_samples = int(total_samples * size)
            else:
                n_samples = int(size)
            
            results['sample_sizes'].append(n_samples)
            
            iter_train_losses = []
            iter_eval_accs = []
            iter_eval_f1s = []
            iter_eval_losses = []
            
            print(f"\n{'='*60}")
            print(f"Training with {n_samples} samples ({n_samples/total_samples*100:.1f}%)")
            print(f"{'='*60}")
            
            for i in range(n_iterations):
                print(f"\nIteration {i+1}/{n_iterations}")
                
                # Random sample from full dataset
                indices = np.random.choice(total_samples, size=n_samples, replace=False)
                train_subset = full_dataset.select(indices.tolist())
                
                # Train and evaluate
                metrics = self.train_model(
                    train_subset,
                    eval_dataset,
                    num_epochs=num_epochs,
                    batch_size=batch_size
                )
                
                iter_train_losses.append(metrics['train_loss'])
                iter_eval_accs.append(metrics['eval_accuracy'])
                iter_eval_f1s.append(metrics['eval_f1'])
                iter_eval_losses.append(metrics['eval_loss'])
                
                print(f"  Train Loss: {metrics['train_loss']:.4f}")
                print(f"  Eval Accuracy: {metrics['eval_accuracy']:.4f}")
                print(f"  Eval F1: {metrics['eval_f1']:.4f}")
            
            results['train_losses'].append(iter_train_losses)
            results['eval_accuracies'].append(iter_eval_accs)
            results['eval_f1s'].append(iter_eval_f1s)
            results['eval_losses'].append(iter_eval_losses)
        
        return (
            np.array(results['sample_sizes']),
            np.array(results['train_losses']),
            np.array(results['eval_accuracies']),
            np.array(results['eval_losses'])
        )

# Power law functions for fitting
def power_law_loss(D, D_c, alpha_D):
    """OpenAI-style power law: L(D) = (D_c / D)^alpha_D"""
    return np.power(D_c / D, alpha_D)

def inverse_power_law_accuracy(D, a, b, c):
    """Inverse power law for accuracy: Acc(D) = c - a * D^(-b)"""
    return c - a * np.power(D, -b)

def fit_transformer_scaling_law(sample_sizes, losses):
    """
    Fit transformer scaling law to learning curve data.
    
    Args:
        sample_sizes: Array of dataset sizes
        losses: Array of corresponding losses
        
    Returns:
        Fitted parameters and predictions
    """
    try:
        # Fit power law: L(D) = (D_c / D)^alpha_D
        params, _ = curve_fit(
            power_law_loss,
            sample_sizes,
            losses,
            p0=[1e10, 0.1],  # Initial guess for D_c and alpha_D
            bounds=([1e5, 0.01], [1e15, 0.5]),
            maxfev=10000
        )
        
        D_c, alpha_D = params
        print(f"\nFitted Scaling Law Parameters:")
        print(f"  D_c (data constant) = {D_c:.2e}")
        print(f"  α_D (data exponent) = {alpha_D:.4f}")
        print(f"  (OpenAI found α_D ≈ 0.095 for GPT)")
        
        return params
    except Exception as e:
        print(f"Power law fitting failed: {e}")
        return None

def plot_transformer_learning_curve(sample_sizes, eval_losses, eval_accuracies,
                                   extrapolate_to=None, model_name="Transformer"):
    """
    Plot learning curves with transformer scaling law fit.
    
    Args:
        sample_sizes: Array of dataset sizes
        eval_losses: Evaluation losses (2D: sizes × iterations)
        eval_accuracies: Evaluation accuracies (2D: sizes × iterations)
        extrapolate_to: Optional target size for extrapolation
        model_name: Model name for plot title
    """
    # Calculate statistics
    loss_mean = np.mean(eval_losses, axis=1)
    loss_std = np.std(eval_losses, axis=1)
    acc_mean = np.mean(eval_accuracies, axis=1)
    acc_std = np.std(eval_accuracies, axis=1)
    
    # Fit scaling law to loss
    params = fit_transformer_scaling_law(sample_sizes, loss_mean)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Loss vs Dataset Size
    ax1.errorbar(sample_sizes, loss_mean, yerr=loss_std,
                fmt='o-', capsize=5, linewidth=2, markersize=8,
                label='Measured Loss', color='steelblue')
    
    if params is not None:
        D_c, alpha_D = params
        
        # Generate smooth curve
        if extrapolate_to is not None:
            x_smooth = np.linspace(sample_sizes[0], extrapolate_to, 200)
        else:
            x_smooth = np.linspace(sample_sizes[0], sample_sizes[-1] * 1.5, 200)
        
        y_smooth = power_law_loss(x_smooth, D_c, alpha_D)
        
        ax1.plot(x_smooth, y_smooth, '--', linewidth=2, alpha=0.7,
                label=f'Power Law Fit: L(D) = ({D_c:.1e}/D)^{alpha_D:.3f}',
                color='coral')
        
        # Mark extrapolation point
        if extrapolate_to is not None:
            predicted_loss = power_law_loss(extrapolate_to, D_c, alpha_D)
            ax1.plot(extrapolate_to, predicted_loss, 'r*',
                    markersize=20, label=f'Predicted at {extrapolate_to}',
                    zorder=5)
            print(f"\nPredicted loss at {extrapolate_to} samples: {predicted_loss:.4f}")
    
    ax1.set_xlabel('Training Dataset Size (tokens/samples)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Evaluation Loss', fontsize=13, fontweight='bold')
    ax1.set_title(f'{model_name}: Loss vs Dataset Size\n(Transformer Scaling Law)',
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot 2: Accuracy vs Dataset Size
    ax2.errorbar(sample_sizes, acc_mean, yerr=acc_std,
                fmt='s-', capsize=5, linewidth=2, markersize=8,
                label='Measured Accuracy', color='forestgreen')
    
    # Fit inverse power law to accuracy
    try:
        acc_params, _ = curve_fit(
            inverse_power_law_accuracy,
            sample_sizes,
            acc_mean,
            p0=[0.5, 0.1, 1.0],
            bounds=([0, 0, 0.5], [2, 1, 1.0]),
            maxfev=10000
        )
        
        a, b, c = acc_params
        
        if extrapolate_to is not None:
            x_smooth_acc = np.linspace(sample_sizes[0], extrapolate_to, 200)
        else:
            x_smooth_acc = np.linspace(sample_sizes[0], sample_sizes[-1] * 1.5, 200)
        
        y_smooth_acc = inverse_power_law_accuracy(x_smooth_acc, a, b, c)
        
        ax2.plot(x_smooth_acc, y_smooth_acc, '--', linewidth=2, alpha=0.7,
                label=f'Power Law Fit: Acc = {c:.3f} - {a:.3f}·D^(-{b:.3f})',
                color='coral')
        
        if extrapolate_to is not None:
            predicted_acc = inverse_power_law_accuracy(extrapolate_to, a, b, c)
            ax2.plot(extrapolate_to, predicted_acc, 'r*',
                    markersize=20, label=f'Predicted at {extrapolate_to}',
                    zorder=5)
            print(f"Predicted accuracy at {extrapolate_to} samples: {predicted_acc:.4f}")
            print(f"Maximum achievable accuracy (asymptote): {c:.4f}")
    
    except Exception as e:
        print(f"Accuracy fit failed: {e}")
    
    ax2.set_xlabel('Training Dataset Size (tokens/samples)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Evaluation Accuracy', fontsize=13, fontweight='bold')
    ax2.set_title(f'{model_name}: Accuracy vs Dataset Size',
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.show()

# Example Usage
def run_bert_learning_curve_example():
    """
    Complete example: Generate learning curves for BERT fine-tuning
    on a text classification task.
    """
    print("Loading dataset...")
    # Load a dataset (using IMDb as example)
    from datasets import load_dataset
    dataset = load_dataset('imdb')
    
    # Take subset for faster experimentation
    train_data = dataset['train'].shuffle(seed=42).select(range(5000))
    test_data = dataset['test'].shuffle(seed=42).select(range=1000))
    
    # Initialize learning curve generator
    lc_generator = TransformerLearningCurve(
        model_name='bert-base-uncased',
        num_labels=2,  # Binary classification
        task_name='imdb_sentiment'
    )
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    train_tokenized = train_data.map(
        lambda x: lc_generator.tokenize_function(x, text_column='text'),
        batched=True,
        remove_columns=['text']
    )
    test_tokenized = test_data.map(
        lambda x: lc_generator.tokenize_function(x, text_column='text'),
        batched=True,
        remove_columns=['text']
    )
    
    # Set format for PyTorch
    train_tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Generate learning curve with different sample sizes
    sample_sizes = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    sizes, train_losses, eval_accs, eval_losses = lc_generator.generate_learning_curve(
        train_tokenized,
        test_tokenized,
        sample_sizes=sample_sizes,
        n_iterations=3,
        num_epochs=3,
        batch_size=16
    )
    
    # Plot results with extrapolation
    plot_transformer_learning_curve(
        sizes, eval_losses, eval_accs,
        extrapolate_to=len(train_data),
        model_name='BERT (IMDb Sentiment)'
    )
    
    return sizes, train_losses, eval_accs, eval_losses

# Run the example (uncomment to execute)
# results = run_bert_learning_curve_example()
```

## Implementation 2: GPT-Style Transformer with Scaling Laws

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class SimpleGPT(nn.Module):
    """
    Simplified GPT-style transformer for demonstrating scaling laws.
    """
    
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=1024, max_seq_len=512, num_classes=None):
        super().__init__()
        
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output head
        if num_classes:
            # Classification head
            self.output = nn.Linear(d_model, num_classes)
        else:
            # Language modeling head
            self.output = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(0.1)
        
        # Count non-embedding parameters
        self.non_embedding_params = self._count_non_embedding_params()
    
    def _count_non_embedding_params(self):
        """Count parameters excluding embeddings (for scaling laws)."""
        total = sum(p.numel() for p in self.parameters())
        embedding = sum(p.numel() for p in self.token_embedding.parameters())
        embedding += sum(p.numel() for p in self.position_embedding.parameters())
        return total - embedding
    
    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape
        
        # Create position indices
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(token_emb + pos_emb)
        
        # Transformer
        # Create causal mask for autoregressive generation
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Note: TransformerDecoder expects (tgt, memory)
        # For decoder-only, we use same input as both
        x = self.transformer(x, x, tgt_mask=mask)
        
        # Output projection
        logits = self.output(x)
        
        # For classification, use last token
        if self.num_classes:
            logits = logits[:, -1, :]
        
        return logits

class GPTScalingExperiment:
    """
    Run scaling law experiments for GPT-style transformers.
    """
    
    def __init__(self, vocab_size=5000, max_seq_len=128, num_classes=None):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def create_model(self, d_model, nhead, num_layers):
        """Create model with specified architecture."""
        model = SimpleGPT(
            vocab_size=self.vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_seq_len=self.max_seq_len,
            num_classes=self.num_classes
        ).to(self.device)
        
        print(f"Model created: d_model={d_model}, nhead={nhead}, layers={num_layers}")
        print(f"  Non-embedding params (N): {model.non_embedding_params:,}")
        print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    def train_and_evaluate(self, model, train_loader, val_loader,
                          epochs=10, lr=3e-4):
        """Train model and return final validation loss."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                
                if self.num_classes:
                    loss = criterion(output, target)
                else:
                    # Language modeling: predict next token
                    loss = criterion(output.reshape(-1, self.vocab_size),
                                   target.reshape(-1))
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    
                    if self.num_classes:
                        loss = criterion(output, target)
                    else:
                        loss = criterion(output.reshape(-1, self.vocab_size),
                                       target.reshape(-1))
                    
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            if (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        return avg_val_loss
    
    def run_data_scaling_experiment(self, model_config, dataset,
                                   data_fractions, batch_size=32):
        """
        Measure how loss scales with dataset size D.
        
        Args:
            model_config: Dict with 'd_model', 'nhead', 'num_layers'
            dataset: Full dataset
            data_fractions: List of fractions of data to use
            
        Returns:
            (dataset_sizes, losses)
        """
        dataset_sizes = []
        losses = []
        
        for fraction in data_fractions:
            print(f"\n{'='*60}")
            print(f"Training with {fraction*100:.0f}% of data")
            print(f"{'='*60}")
            
            # Sample data
            n_samples = int(len(dataset) * fraction)
            indices = np.random.choice(len(dataset), size=n_samples, replace=False)
            subset = torch.utils.data.Subset(dataset, indices)
            
            # Split into train/val
            n_train = int(0.9 * len(subset))
            n_val = len(subset) - n_train
            train_subset, val_subset = torch.utils.data.random_split(
                subset, [n_train, n_val]
            )
            
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size)
            
            # Create and train model
            model = self.create_model(**model_config)
            val_loss = self.train_and_evaluate(model, train_loader, val_loader)
            
            dataset_sizes.append(n_samples)
            losses.append(val_loss)
            
            print(f"Final validation loss: {val_loss:.4f}")
        
        return np.array(dataset_sizes), np.array(losses)
    
    def run_model_scaling_experiment(self, model_configs, dataset,
                                    batch_size=32, use_full_data=True):
        """
        Measure how loss scales with model size N.
        
        Args:
            model_configs: List of dicts with model architectures
            dataset: Training dataset
            batch_size: Batch size
            use_full_data: Whether to use full dataset
            
        Returns:
            (param_counts, losses)
        """
        param_counts = []
        losses = []
        
        # Prepare data
        if use_full_data:
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            train_subset, val_subset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size)
        
        for config in model_configs:
            print(f"\n{'='*60}")
            print(f"Training model: {config}")
            print(f"{'='*60}")
            
            model = self.create_model(**config)
            val_loss = self.train_and_evaluate(model, train_loader, val_loader)
            
            param_counts.append(model.non_embedding_params)
            losses.append(val_loss)
            
            print(f"Final validation loss: {val_loss:.4f}")
        
        return np.array(param_counts), np.array(losses)

def plot_scaling_laws(dataset_sizes=None, data_losses=None,
                     param_counts=None, model_losses=None):
    """
    Plot transformer scaling laws.
    
    Args:
        dataset_sizes: Array of dataset sizes (for L(D) plot)
        data_losses: Corresponding losses
        param_counts: Array of parameter counts (for L(N) plot)
        model_losses: Corresponding losses
    """
    n_plots = sum([dataset_sizes is not None, param_counts is not None])
    
    if n_plots == 0:
        print("No data to plot")
        return
    
    fig, axes = plt.subplots(1, n_plots, figsize=(8*n_plots, 6))
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot L(D) - Loss vs Dataset Size
    if dataset_sizes is not None:
        ax = axes[plot_idx]
        plot_idx += 1
        
        ax.loglog(dataset_sizes, data_losses, 'o', markersize=10,
                 label='Measured Loss', color='steelblue')
        
        # Fit power law
        try:
            params, _ = curve_fit(
                power_law_loss,
                dataset_sizes,
                data_losses,
                p0=[1e10, 0.1],
                bounds=([1e5, 0.01], [1e15, 0.5])
            )
            
            D_c, alpha_D = params
            
            x_smooth = np.logspace(np.log10(dataset_sizes[0]),
                                  np.log10(dataset_sizes[-1]*2), 100)
            y_smooth = power_law_loss(x_smooth, D_c, alpha_D)
            
            ax.loglog(x_smooth, y_smooth, '--', linewidth=2,
                     label=f'L(D) = ({D_c:.1e}/D)^{alpha_D:.3f}',
                     color='coral')
            
            ax.text(0.05, 0.95, f'α_D = {alpha_D:.4f}\n(OpenAI: ~0.095)',
                   transform=ax.transAxes, fontsize=11,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        except Exception as e:
            print(f"Data scaling fit failed: {e}")
        
        ax.set_xlabel('Dataset Size (D)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Validation Loss', fontsize=13, fontweight='bold')
        ax.set_title('Transformer Scaling Law: L(D)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, which='both')
    
    # Plot L(N) - Loss vs Model Size
    if param_counts is not None:
        ax = axes[plot_idx]
        
        ax.loglog(param_counts, model_losses, 's', markersize=10,
                 label='Measured Loss', color='forestgreen')
        
        # Fit power law
        try:
            params, _ = curve_fit(
                power_law_loss,
                param_counts,
                model_losses,
                p0=[1e13, 0.08],
                bounds=([1e10, 0.01], [1e15, 0.5])
            )
            
            N_c, alpha_N = params
            
            x_smooth = np.logspace(np.log10(param_counts[0]),
                                  np.log10(param_counts[-1]*2), 100)
            y_smooth = power_law_loss(x_smooth, N_c, alpha_N)
            
            ax.loglog(x_smooth, y_smooth, '--', linewidth=2,
                     label=f'L(N) = ({N_c:.1e}/N)^{alpha_N:.3f}',
                     color='coral')
            
            ax.text(0.05, 0.95, f'α_N = {alpha_N:.4f}\n(OpenAI: ~0.076)',
                   transform=ax.transAxes, fontsize=11,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        except Exception as e:
            print(f"Model scaling fit failed: {e}")
        
        ax.set_xlabel('Non-embedding Parameters (N)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Validation Loss', fontsize=13, fontweight='bold')
        ax.set_title('Transformer Scaling Law: L(N)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.show()
```

## Implementation 3: Sample Size Estimation for Transformer Fine-tuning

```python
def estimate_required_samples_transformer(current_sizes, current_losses,
                                        target_loss, model_name="Transformer"):
    """
    Estimate the number of samples needed to reach a target loss for transformers.
    
    Args:
        current_sizes: Array of dataset sizes already tested
        current_losses: Corresponding validation losses
        target_loss: Desired target loss
        model_name: Model identifier for display
        
    Returns:
        Estimated required sample size
    """
    # Fit scaling law
    try:
        params, _ = curve_fit(
            power_law_loss,
            current_sizes,
            current_losses,
            p0=[1e10, 0.1],
            bounds=([1e5, 0.01], [1e15, 0.5])
        )
        
        D_c, alpha_D = params
        
        print(f"\n{model_name} Scaling Law Fit:")
        print(f"  D_c = {D_c:.2e}")
        print(f"  α_D = {alpha_D:.4f}")
        
        # Solve for D: target_loss = (D_c / D)^alpha_D
        # D = D_c / (target_loss^(1/alpha_D))
        required_samples = D_c / np.power(target_loss, 1/alpha_D)
        
        print(f"\nTo achieve loss of {target_loss:.4f}:")
        print(f"  Estimated samples needed: {int(required_samples):,}")
        
        # Check if achievable
        min_loss = power_law_loss(current_sizes[-1] * 10, D_c, alpha_D)
        if target_loss < min_loss * 0.9:
            print(f"  ⚠️  Warning: Target may be difficult to achieve")
            print(f"     Extrapolated minimum loss (~10x data): {min_loss:.4f}")
        
        return int(required_samples)
    
    except Exception as e:
        print(f"Estimation failed: {e}")
        return None

# Example usage function
def transformer_sample_size_example():
    """
    Example: Estimate required samples for BERT fine-tuning.
    """
    # Simulated data from learning curve experiments
    # These would come from actual experiments
    sample_sizes = np.array([250, 500, 1000, 2000, 4000])
    eval_losses = np.array([1.2, 0.8, 0.55, 0.42, 0.35])
    
    # Target: achieve loss of 0.25
    target_loss = 0.25
    
    required = estimate_required_samples_transformer(
        sample_sizes,
        eval_losses,
        target_loss,
        model_name="BERT-base"
    )
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.loglog(sample_sizes, eval_losses, 'o-', markersize=10,
              linewidth=2, label='Observed', color='steelblue')
    
    if required:
        plt.loglog(required, target_loss, 'r*', markersize=20,
                  label=f'Estimated Required: {required:,} samples', zorder=5)
        plt.axhline(y=target_loss, color='red', linestyle='--',
                   alpha=0.5, label=f'Target Loss: {target_loss}')
    
    plt.xlabel('Training Dataset Size', fontsize=13, fontweight='bold')
    plt.ylabel('Validation Loss', fontsize=13, fontweight='bold')
    plt.title('Sample Size Estimation for Transformer', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()

# Run example
# transformer_sample_size_example()
```

## Transformer-Specific Insights

### Fine-tuning Requirements[8][9][10][11]

**BERT fine-tuning**:

- **Minimum**: ~100-500 samples for simple classification tasks[12][10][8]
- **Recommended**: 1,000+ samples for robust performance[9][11][8]
- **Optimal**: 2,000-5,000 samples per class for complex tasks[8]

**Sample efficiency by task**:[8]

- Named Entity Recognition (NER): **439-527 sentences** (threshold of diminishing returns)
- Sentiment analysis: **1,000-2,000 samples** minimum
- Domain-specific tasks: Often require **5,000-10,000 samples**

### Transfer Learning Benefits[13][9]

Fine-tuning pre-trained transformers reduces data requirements by **5-10×** compared to training from scratch:[9][13]

- BERT/GPT with transfer learning: **hundreds to thousands** of samples
- Training from scratch: **millions** of samples required

### Critical Batch Sizes[1]

For transformers, critical batch size scales with loss:

\[
B_{crit}(L) \approx 2.2 \times 10^7 \cdot L^{-4.26}
\]

Training above this batch size wastes compute without improving performance.[1]

### Optimal Training Configuration[5][2]

**Chinchilla-optimal** (training cost minimization):[4][5][2]

- 20-25 tokens per parameter
- Example: 7B parameter model → 140-175B tokens

**Overtraining** (inference cost minimization):[14][5][2]

- Modern practice uses 100-1,000+ tokens per parameter
- Llama 3 (8B params): **15 trillion tokens** (1,875 tokens/param)
- Reduces inference costs for models with high usage

## Key Takeaways

1. **Transformers follow power-law scaling** with both data (\(\alpha_D \approx 0.095\)) and parameters (\(\alpha_N \approx 0.076\))[3][2][1]

2. **8× larger models need only ~5× more data**, making transformers more sample-efficient than traditional networks[2][3]

3. **Fine-tuning is highly sample-efficient**: Pre-trained transformers can achieve good performance with 100-1,000 samples[10][9][8]

4. **Scaling laws enable prediction**: Train small models to predict large model performance with high accuracy[2]

5. **Data quality matters**: High-quality, diverse data reduces requirements significantly[15][16]

6. **Domain adaptation requires less data** than training from scratch [typically 10-50× less](13)[8]

This transformer-specific guide provides the mathematical foundations and practical implementations needed to determine optimal training data requirements for modern language models.

[1](https://github.com/shehper/scaling_laws)
[2](https://cameronrwolfe.substack.com/p/llm-scaling-laws)
[3](https://arxiv.org/pdf/2001.08361.pdf)
[4](https://mbrenndoerfer.com/writing/chinchilla-scaling-laws-compute-optimal-training-resource-allocation)
[5](https://www.databricks.com/blog/how-long-should-you-train-your-language-model)
[6](https://epoch.ai/publications/chinchilla-scaling-a-replication-attempt)
[7](https://lifearchitect.ai/chinchilla/)
[8](https://pmc.ncbi.nlm.nih.gov/articles/PMC11140272/)
[9](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)
[10](https://discuss.huggingface.co/t/using-extremely-small-dataset-to-finetune-bert/8847)
[11](https://discuss.huggingface.co/t/thoughts-on-quantity-of-training-data-for-fine-tuning/14886)
[12](https://pmc.ncbi.nlm.nih.gov/articles/PMC10399128/)
[13](https://pmc.ncbi.nlm.nih.gov/articles/PMC6484664/)
[14](https://arxiv.org/html/2401.00448v3)
[15](https://www.pccube.com/en/data-quality-vs-data-quantity/)
[16](https://www.reddit.com/r/llmsupport/comments/1k237gh/data_quality_vs_quantity_whats_your_approach_to/)
[17](https://www.machinelearningmastery.com/plotting-the-training-and-validation-loss-curves-for-the-transformer-model/)
[18](https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/)
[19](https://jax-ml.github.io/scaling-book/inference/)
[20](https://developer.nvidia.com/blog/scale-biology-transformer-models-with-pytorch-and-nvidia-bionemo-recipes/)
[21](https://social-media-lab.net/processing/bert_classification.html)
[22](https://arxiv.org/abs/2209.00588)
[23](https://pytorch.org/blog/paretoq-scaling-laws-in-extremely-low-bit-llm-quantization/)
[24](https://mbrenndoerfer.com/writing/bert-pretraining-mlm-nsp-training-guide)
[25](https://www.ijournalse.org/index.php/ESJ/article/view/1883)
[26](https://www.youtube.com/watch?v=nQAwL1xu058)
[27](https://towardsdatascience.com/a-complete-guide-to-bert-with-code-9f87602e4a11/)
[28](https://openreview.net/forum?id=C9uv8qR7RX)
[29](https://arxiv.org/html/2411.05050v1)
[30](https://meta-learn.github.io/2020/papers/32_paper.pdf)
[31](https://www.thoughtspot.com/data-trends/ai/what-is-transformer-architecture-chatgpt)
[32](https://www.reddit.com/r/MachineLearning/comments/xp9uoc/p_efficient_fewshot_learning_with_sentence/)
[33](https://learnopencv.com/fine-tuning-bert/)
[34](https://poloclub.github.io/transformer-explainer/)
[35](https://arxiv.org/abs/2301.02419)
[36](https://www.datacamp.com/tutorial/how-transformers-work)
[37](https://github.com/Yangzhangcst/Transformer-in-Computer-Vision/blob/main/main/few-shot-learning.md)
[38](https://stackoverflow.com/questions/61962710/how-to-fine-tune-bert-on-unlabeled-data)
[39](https://www.geeksforgeeks.org/artificial-intelligence/introduction-to-generative-pre-trained-transformer-gpt/)
