Now I'll create a comprehensive guide with implementations.

# Empirical Learning Curves: Complete Guide with PyTorch Implementation

## What Are Learning Curves?

Learning curves are empirical methods for determining **how much training data you need** by plotting model performance (y-axis) against training dataset size (x-axis). Unlike theoretical formulas, this approach provides **actual, problem-specific** data requirements.

## How It Works

### The Core Algorithm

1. **Train multiple models** with increasing dataset sizes
2. **Measure performance** (accuracy, loss, etc.) on a validation set for each size
3. **Plot the relationship** between dataset size and performance
4. **Fit a curve** (typically power law or exponential) to the data
5. **Extrapolate** to predict performance at larger dataset sizes
6. **Determine required sample size** based on target performance

### The Mathematics

**Power Law Formula** (most common for neural networks):[1][2][3][4]

\[
\text{Error}(n) = a \cdot n^{-b} + c
\]

Where:

- \(n\) = training dataset size
- \(a\) = scaling coefficient
- \(b\) = scaling exponent [typically 0.05-0.5 for neural networks](3)[4]
- \(c\) = irreducible error (Bayes error rate)

**Alternative: Exponential Formula** (used in Keras examples):[5]

\[
\text{Accuracy}(n) = a \cdot n^b
\]

**Inverse Power Law** (OpenAI scaling laws):[4]

\[
L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}
\]

Where:

- \(L(D)\) = loss given dataset size \(D\)
- \(D_c\) = dataset size constant
- \(\alpha_D\) = data scaling exponent [~0.095 for language models](4)

## Complete PyTorch Implementation

### Method 1: Using Scikit-Learn + Skorch (Recommended)

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from sklearn.model_selection import learning_curve
from sklearn.datasets import make_classification

# Define PyTorch model
class SimpleNet(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=50, num_classes=2):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create synthetic dataset
X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

# Wrap PyTorch model for sklearn compatibility
model = NeuralNetClassifier(
    SimpleNet,
    max_epochs=50,
    lr=0.01,
    batch_size=64,
    optimizer=torch.optim.Adam,
    criterion=nn.CrossEntropyLoss,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbose=0
)

# Generate learning curve data
# train_sizes: fractions or absolute numbers of training samples
train_sizes, train_scores, val_scores = learning_curve(
    estimator=model,
    X=X,
    y=y,
    train_sizes=np.linspace(0.1, 1.0, 10),  # 10%, 20%, ..., 100%
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,  # use all CPU cores
    verbose=1
)

# Calculate mean and std
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot learning curves
plt.figure(figsize=(12, 6))
plt.plot(train_sizes, train_mean, 'o-', label='Training score', linewidth=2)
plt.fill_between(train_sizes, train_mean - train_std, 
                 train_mean + train_std, alpha=0.2)
plt.plot(train_sizes, val_mean, 'o-', label='Validation score', linewidth=2)
plt.fill_between(train_sizes, val_mean - val_std, 
                 val_mean + val_std, alpha=0.2)
plt.xlabel('Training Dataset Size', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Learning Curve: Model Performance vs Dataset Size', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Training sizes: {train_sizes}")
print(f"Validation scores (mean): {val_mean}")
```

### Method 2: Framework-Agnostic Manual Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def power_law_error(n, a, b, c):
    """
    Power law for error: Error(n) = a * n^(-b) + c
    
    Args:
        n: dataset size
        a: scaling coefficient
        b: scaling exponent
        c: irreducible error (Bayes rate)
    """
    return a * np.power(n, -b) + c

def exponential_accuracy(n, a, b):
    """
    Exponential function for accuracy: Accuracy(n) = a * n^b
    
    Args:
        n: dataset size
        a: amplitude
        b: growth rate
    """
    return a * np.power(n, b)

def generate_learning_curve(model, X, y, train_sizes, 
                           n_iterations=5, test_size=0.2,
                           random_state=42):
    """
    Generate learning curve data by training model on varying dataset sizes.
    
    Args:
        model: Model with fit() and predict() methods
        X: Feature matrix
        y: Target vector
        train_sizes: List/array of dataset sizes to test
        n_iterations: Number of times to repeat training per size
        test_size: Fraction of data to use for validation
        random_state: Random seed
    
    Returns:
        train_sizes: Actual training sizes used
        train_scores: Training scores for each size and iteration
        val_scores: Validation scores for each size and iteration
    """
    X_train_full, X_val, y_train_full, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    train_scores_list = []
    val_scores_list = []
    actual_train_sizes = []
    
    for size in train_sizes:
        # Convert fraction to actual size if needed
        if size <= 1.0:
            n_samples = int(len(X_train_full) * size)
        else:
            n_samples = int(size)
        
        actual_train_sizes.append(n_samples)
        
        iteration_train_scores = []
        iteration_val_scores = []
        
        for i in range(n_iterations):
            # Random subset for this iteration
            indices = np.random.choice(
                len(X_train_full), 
                size=n_samples, 
                replace=False
            )
            X_subset = X_train_full[indices]
            y_subset = y_train_full[indices]
            
            # Train model
            model.fit(X_subset, y_subset)
            
            # Evaluate
            train_pred = model.predict(X_subset)
            val_pred = model.predict(X_val)
            
            train_score = accuracy_score(y_subset, train_pred)
            val_score = accuracy_score(y_val, val_pred)
            
            iteration_train_scores.append(train_score)
            iteration_val_scores.append(val_score)
        
        train_scores_list.append(iteration_train_scores)
        val_scores_list.append(iteration_val_scores)
    
    return (np.array(actual_train_sizes), 
            np.array(train_scores_list), 
            np.array(val_scores_list))

def fit_power_law_and_extrapolate(train_sizes, scores, target_size):
    """
    Fit power law to learning curve and extrapolate to target size.
    
    Args:
        train_sizes: Array of training sizes
        scores: Array of corresponding scores (use validation scores)
        target_size: Dataset size to extrapolate to
    
    Returns:
        predicted_score: Predicted score at target_size
        params: Fitted parameters (a, b, c)
    """
    # Convert accuracy to error for power law fitting
    errors = 1 - scores
    
    # Fit power law: error = a * n^(-b) + c
    try:
        params, _ = curve_fit(
            power_law_error,
            train_sizes,
            errors,
            p0=[1.0, 0.3, 0.01],  # initial guess
            bounds=([0, 0, 0], [np.inf, 1, 1]),  # parameter bounds
            maxfev=10000
        )
        
        # Predict error at target size
        predicted_error = power_law_error(target_size, *params)
        predicted_score = 1 - predicted_error
        
        return predicted_score, params
    except Exception as e:
        print(f"Power law fitting failed: {e}")
        return None, None

def fit_exponential_and_extrapolate(train_sizes, scores, target_size):
    """
    Fit exponential function to learning curve and extrapolate.
    
    Args:
        train_sizes: Array of training sizes
        scores: Array of corresponding scores
        target_size: Dataset size to extrapolate to
    
    Returns:
        predicted_score: Predicted score at target_size
        params: Fitted parameters (a, b)
    """
    try:
        params, _ = curve_fit(
            exponential_accuracy,
            train_sizes,
            scores,
            p0=[0.5, 0.1],  # initial guess
            maxfev=10000
        )
        
        predicted_score = exponential_accuracy(target_size, *params)
        
        return predicted_score, params
    except Exception as e:
        print(f"Exponential fitting failed: {e}")
        return None, None

def plot_learning_curve_with_extrapolation(train_sizes, train_scores, 
                                          val_scores, target_size=None,
                                          fit_type='power_law'):
    """
    Plot learning curve with optional extrapolation.
    
    Args:
        train_sizes: Array of training sizes
        train_scores: Training scores (2D: sizes × iterations)
        val_scores: Validation scores (2D: sizes × iterations)
        target_size: Optional target size for extrapolation
        fit_type: 'power_law' or 'exponential'
    """
    # Calculate means and standard deviations
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(14, 7))
    
    # Plot actual data
    plt.errorbar(train_sizes, train_mean, yerr=train_std, 
                fmt='o-', label='Training Score', capsize=5, linewidth=2)
    plt.errorbar(train_sizes, val_mean, yerr=val_std,
                fmt='s-', label='Validation Score', capsize=5, linewidth=2)
    
    # Fit curve and extrapolate if target_size provided
    if target_size is not None:
        if fit_type == 'power_law':
            predicted_score, params = fit_power_law_and_extrapolate(
                train_sizes, val_mean, target_size
            )
            if params is not None:
                # Generate smooth curve
                x_smooth = np.linspace(train_sizes[0], target_size, 200)
                errors_smooth = power_law_error(x_smooth, *params)
                y_smooth = 1 - errors_smooth
                
                plt.plot(x_smooth, y_smooth, '--', 
                        label=f'Power Law Fit (a={params[0]:.3f}, b={params[1]:.3f})',
                        linewidth=2, alpha=0.7)
                plt.plot(target_size, predicted_score, 'r*', 
                        markersize=15, 
                        label=f'Predicted at {target_size}: {predicted_score:.4f}')
        
        elif fit_type == 'exponential':
            predicted_score, params = fit_exponential_and_extrapolate(
                train_sizes, val_mean, target_size
            )
            if params is not None:
                x_smooth = np.linspace(train_sizes[0], target_size, 200)
                y_smooth = exponential_accuracy(x_smooth, *params)
                
                plt.plot(x_smooth, y_smooth, '--',
                        label=f'Exponential Fit (a={params[0]:.3f}, b={params[1]:.3f})',
                        linewidth=2, alpha=0.7)
                plt.plot(target_size, predicted_score, 'r*',
                        markersize=15,
                        label=f'Predicted at {target_size}: {predicted_score:.4f}')
    
    plt.xlabel('Training Dataset Size', fontsize=13)
    plt.ylabel('Score', fontsize=13)
    plt.title('Learning Curve with Extrapolation', fontsize=15, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Example usage with sklearn model (framework-agnostic)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=2000, n_features=20, 
                          n_informative=15, random_state=42)

# Create model
model = RandomForestClassifier(n_estimators=50, random_state=42)

# Generate learning curve
train_sizes_frac = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
train_sizes, train_scores, val_scores = generate_learning_curve(
    model, X, y, train_sizes_frac, n_iterations=5
)

print(f"Training sizes: {train_sizes}")
print(f"Validation scores (mean): {np.mean(val_scores, axis=1)}")

# Plot with extrapolation
plot_learning_curve_with_extrapolation(
    train_sizes, train_scores, val_scores,
    target_size=2000,  # extrapolate to full dataset
    fit_type='power_law'
)
```

### Method 3: Pure PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt

class PyTorchLearningCurve:
    """Generate learning curves for PyTorch models."""
    
    def __init__(self, model_class, model_kwargs, train_kwargs):
        """
        Args:
            model_class: PyTorch nn.Module class
            model_kwargs: Dict of kwargs for model initialization
            train_kwargs: Dict with 'epochs', 'lr', 'batch_size', 'device'
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.train_kwargs = train_kwargs
        self.device = train_kwargs.get('device', 'cpu')
    
    def train_model(self, train_loader, val_loader):
        """Train a fresh model and return train/val scores."""
        model = self.model_class(**self.model_kwargs).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.train_kwargs['lr'])
        
        # Training loop
        for epoch in range(self.train_kwargs['epochs']):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        train_acc = self.evaluate(model, train_loader)
        val_acc = self.evaluate(model, val_loader)
        
        return train_acc, val_acc
    
    def evaluate(self, model, loader):
        """Calculate accuracy on a dataset."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        return correct / total
    
    def generate_curve(self, X, y, train_sizes, n_iterations=3, val_split=0.2):
        """
        Generate learning curve data.
        
        Args:
            X: Feature tensor
            y: Target tensor
            train_sizes: List of training sizes (fractions or absolute)
            n_iterations: Repetitions per size
            val_split: Validation set fraction
        
        Returns:
            train_sizes_actual, train_scores, val_scores
        """
        # Convert to tensors if needed
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        if not isinstance(y, torch.Tensor):
            y = torch.LongTensor(y)
        
        # Split into train and validation
        n_samples = len(X)
        n_val = int(n_samples * val_split)
        n_train_total = n_samples - n_val
        
        indices = torch.randperm(n_samples)
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        # Create validation set
        val_dataset = TensorDataset(X[val_indices], y[val_indices])
        val_loader = DataLoader(val_dataset, 
                               batch_size=self.train_kwargs['batch_size'])
        
        train_scores_list = []
        val_scores_list = []
        actual_sizes = []
        
        for size in train_sizes:
            # Convert to actual size
            if size <= 1.0:
                n_use = int(n_train_total * size)
            else:
                n_use = int(size)
            
            actual_sizes.append(n_use)
            
            iter_train_scores = []
            iter_val_scores = []
            
            for i in range(n_iterations):
                print(f"Training with {n_use} samples, iteration {i+1}/{n_iterations}")
                
                # Random subset
                subset_indices = train_indices[torch.randperm(len(train_indices))[:n_use]]
                train_dataset = TensorDataset(X[subset_indices], y[subset_indices])
                train_loader = DataLoader(train_dataset,
                                        batch_size=self.train_kwargs['batch_size'],
                                        shuffle=True)
                
                # Train and evaluate
                train_acc, val_acc = self.train_model(train_loader, val_loader)
                
                iter_train_scores.append(train_acc)
                iter_val_scores.append(val_acc)
                
                print(f"  Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            train_scores_list.append(iter_train_scores)
            val_scores_list.append(iter_val_scores)
        
        return (np.array(actual_sizes),
                np.array(train_scores_list),
                np.array(val_scores_list))

# Example: Using the PyTorch learning curve class
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

# Generate synthetic data
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=3000, n_features=50,
                          n_informative=30, n_classes=3,
                          random_state=42)

# Setup learning curve generator
lc_generator = PyTorchLearningCurve(
    model_class=SimpleClassifier,
    model_kwargs={'input_dim': 50, 'hidden_dim': 100, 'num_classes': 3},
    train_kwargs={
        'epochs': 30,
        'lr': 0.001,
        'batch_size': 32,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
)

# Generate learning curve
train_sizes = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
sizes, train_scores, val_scores = lc_generator.generate_curve(
    X, y, train_sizes, n_iterations=3
)

# Plot using the previous plotting function
plot_learning_curve_with_extrapolation(
    sizes, train_scores, val_scores,
    target_size=len(X),
    fit_type='power_law'
)
```

## Practical Usage Guide

### Step 1: Determine Training Sizes

```python
# Option A: Use fractions
train_sizes = np.linspace(0.1, 1.0, 10)  # 10% to 100%

# Option B: Use log spacing for better coverage
train_sizes = np.logspace(np.log10(100), np.log10(10000), 10)

# Option C: Manual specification
train_sizes = [50, 100, 250, 500, 1000, 2000, 5000]
```

### Step 2: Interpret the Curves

**Underfitting** (High Bias):[6][7]

- Both training and validation scores are low
- Curves converge to poor performance
- **Solution**: Use more complex model or better features

**Overfitting** (High Variance):[7][6]

- Large gap between training and validation scores
- Training score high, validation score low
- **Solution**: Get more data, regularization, simpler model

**Good Fit**:

- Small gap between curves
- Both scores plateau at acceptable level
- No significant improvement with more data

### Step 3: Predict Required Sample Size

```python
def find_required_sample_size(train_sizes, val_scores, target_accuracy):
    """
    Find the dataset size needed to reach target accuracy.
    
    Returns:
        required_size: Estimated sample size needed
    """
    val_mean = np.mean(val_scores, axis=1)
    
    # Fit power law
    errors = 1 - val_mean
    params, _ = curve_fit(
        power_law_error,
        train_sizes,
        errors,
        p0=[1.0, 0.3, 0.01],
        bounds=([0, 0, 0], [np.inf, 1, 1])
    )
    
    # Solve for n: target_error = a * n^(-b) + c
    target_error = 1 - target_accuracy
    a, b, c = params
    
    if target_error <= c:
        print(f"Target accuracy {target_accuracy} may not be achievable.")
        print(f"Minimum error (Bayes rate): {c:.4f}")
        return None
    
    # n = (target_error - c) / a)^(-1/b)
    required_size = np.power((target_error - c) / a, -1/b)
    
    return int(required_size)

# Example usage
target_acc = 0.95
required_samples = find_required_sample_size(sizes, val_scores, target_acc)
print(f"Estimated samples needed for {target_acc:.1%} accuracy: {required_samples}")
```

## Key Insights from Research

**Power Law Exponents**:[3][4]

- Language models: \(\alpha_D \approx 0.095\) (data scaling exponent)
- Computer vision: typically 0.1-0.3
- Smaller exponent = slower improvement with more data

**Sample Complexity**:[8][9]

- Minimum sample size often 50× number of model parameters for neural networks
- With transfer learning, requirements drop 5-10×[10][11][12]

**Curve Reliability**:[13][14]

- Power laws work well in "high-shot" regime (>50 samples/class)
- May fail in "few-shot" regime [<10 samples/class](14)
- Use piecewise power laws for better extrapolation[14]

**Data Quality Matters**:[15][16][17]

- High-quality data can reduce requirements by 2-5×
- Cleaning noisy labels more effective than collecting more data

## Common Pitfalls

1. **Insufficient training sizes**: Use at least 5-10 different sizes[5][6]
2. **Too few iterations**: Repeat 3-5 times per size for reliable statistics[5]
3. **Ignoring computational cost**: Learning curves are expensive for large models[13]
4. **Extrapolating too far**: Don't predict beyond 2-3× your largest training size[18][14]
5. **Wrong curve type**: Try both power law and exponential fits[18][5]

## When to Use Learning Curves

**Use when**:[19][6]

- Uncertain about data requirements
- Deciding whether to collect more data
- Comparing model architectures
- Diagnosing bias/variance issues

**Skip when**:

- Computational resources are limited
- Dataset is fixed and small
- Using well-established architectures with known requirements

## Summary

Learning curves provide the **most reliable empirical method** for determining training data requirements. While computationally expensive, they offer problem-specific insights that theoretical formulas cannot match. The power law relationship \(Error(n) = a \cdot n^{-b} + c\) accurately models neural network performance across most domains, enabling extrapolation to predict required sample sizes for target performance levels.[20][1][6][19][3][4]

[1](https://towardsdatascience.com/scaling-law-of-language-models-5759de7f830c/)
[2](https://cameronrwolfe.substack.com/p/llm-scaling-laws)
[3](https://mbrenndoerfer.com/writing/power-laws-deep-learning-neural-network-scaling)
[4](https://arxiv.org/pdf/2001.08361.pdf)
[5](https://keras.io/examples/keras_recipes/sample_size_estimate/)
[6](https://www.machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/)
[7](https://scikit-learn.org/stable/modules/learning_curve.html)
[8](https://pmc.ncbi.nlm.nih.gov/articles/PMC11851979/)
[9](https://pmc.ncbi.nlm.nih.gov/articles/PMC11005022/)
[10](https://pmc.ncbi.nlm.nih.gov/articles/PMC6484664/)
[11](https://www.reddit.com/r/learnmachinelearning/comments/vrch72/how_many_images_would_i_need_to_make_my_own_image/)
[12](https://www.lightly.ai/blog/transfer-learning)
[13](https://research.nvidia.com/labs/toronto-ai/estimatingrequirements/)
[14](https://openaccess.thecvf.com/content/CVPR2023/papers/Jain_A_Meta-Learning_Approach_to_Predicting_Performance_and_Data_Requirements_CVPR_2023_paper.pdf)
[15](https://www.pccube.com/en/data-quality-vs-data-quantity/)
[16](https://www.reddit.com/r/llmsupport/comments/1k237gh/data_quality_vs_quantity_whats_your_approach_to/)
[17](https://www.owkin.com/a-z-of-ai-for-healthcare/quality-vs-quantity-of-data)
[18](https://studenttheses.universiteitleiden.nl/access/item:3665258/view)
[19](https://www.machinelearningmastery.com/much-training-data-required-machine-learning/)
[20](https://unidata.pro/blog/how-much-training-data-is-needed-for-machine-learning/)
[21](https://www.machinelearningmastery.com/use-pytorch-deep-learning-models-with-scikit-learn/)
[22](https://sebastianraschka.com/books/machine-learning-with-pytorch-and-scikit-learn/)
[23](https://www.nature.com/articles/s41746-022-00728-0)
[24](https://github.com/AlexGose/learning-curve)
[25](https://arxiv.org/html/2501.18812v2)
[26](https://www.reddit.com/r/deeplearning/comments/eur76w/pytorch_and_tensorflow_wrappers_for_scikitlearn/)
[27](https://github.com/kr-colab/disperseNN)
[28](https://discuss.pytorch.org/t/calibration-curve-of-a-multiclass-logistic-regression/175561)
[29](https://www.sciencedirect.com/science/article/abs/pii/S157495412030087X)
[30](https://stackoverflow.com/questions/77792831/fitting-a-power-law)
[31](https://pubmed.ncbi.nlm.nih.gov/38803150/)
[32](https://proceedings.neurips.cc/paper_files/paper/2023/file/3f1a5e8bfcc3005724d246abe454c1e5-Paper-Conference.pdf)
[33](https://neptune.ai/blog/keras-metrics)
[34](https://towardsdatascience.com/detecting-power-laws-in-real-world-data-with-python-b464190fade6/)
[35](https://www.jmlr.org/papers/volume2/meek02a/meek02a.pdf)
[36](https://www.machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/)
[37](https://mathai2022.github.io/papers/16.pdf)
[38](https://www.tensorflow.org/guide/keras/training_with_built_in_methods)
[39](https://github.com/shehper/scaling_laws)
[40](https://ada.liacs.nl/papers/KieEtAl24.pdf)
[41](https://keras.io/examples/)
