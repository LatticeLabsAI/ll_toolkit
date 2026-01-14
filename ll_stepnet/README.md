# LL-STEPNet

Neural network package for processing STEP/B-Rep CAD files with clean separation of concerns.

## Architecture

The package follows PyTorch best practices with separated modules:

1. **Tokenizer** (`stepnet.tokenizer`): Converts STEP text to token IDs
2. **Feature Extractor** (`stepnet.features`): Extracts geometric properties
3. **Topology Builder** (`stepnet.topology`): Constructs entity reference graphs
4. **Encoder** (`stepnet.encoder`): Neural network combining all representations
5. **Task Models** (`stepnet.tasks`): Task-specific prediction heads
6. **Data Utilities** (`stepnet.data`): Dataset and DataLoader helpers
7. **Trainer** (`stepnet.trainer`): Training loop implementation

## Installation

```bash
cd ll_stepnet
pip install -e .
```

## Quick Start

### Basic Tokenization

```python
from stepnet import STEPTokenizer

tokenizer = STEPTokenizer()

step_text = "#31=CONICAL_SURFACE('',#1837,2.6797,0.7854);"
token_ids = tokenizer.encode(step_text)

print(f"Tokens: {tokenizer.tokenize(step_text)}")
print(f"Token IDs: {token_ids}")
```

### Feature Extraction

```python
from stepnet import STEPFeatureExtractor

extractor = STEPFeatureExtractor()

features = extractor.extract_geometric_features(step_text)
print(f"Entity ID: {features['entity_id']}")
print(f"Entity Type: {features['entity_type']}")
print(f"Numeric params: {features['numeric_params']}")
print(f"References: {features['references']}")
```

### Topology Building

```python
from stepnet import STEPTopologyBuilder

# Extract features from multiple entities
features_list = extractor.extract_features_from_chunk(chunk_text)

# Build topology graph
builder = STEPTopologyBuilder()
topology = builder.build_complete_topology(features_list)

print(f"Nodes: {topology['num_nodes']}")
print(f"Edges: {topology['num_edges']}")
print(f"Adjacency matrix shape: {topology['adjacency_matrix'].shape}")
```

### Complete Encoding

```python
import torch
from stepnet import STEPEncoder, STEPTokenizer, STEPFeatureExtractor, STEPTopologyBuilder

# Initialize components
tokenizer = STEPTokenizer()
extractor = STEPFeatureExtractor()
builder = STEPTopologyBuilder()
encoder = STEPEncoder()

# Process STEP chunk
token_ids = torch.tensor([tokenizer.encode(chunk_text)])
features_list = extractor.extract_features_from_chunk(chunk_text)
topology = builder.build_complete_topology(features_list)

# Encode
output = encoder(token_ids, topology_data=topology)
print(f"Encoding shape: {output.shape}")  # [1, 1024]
```

## Task-Specific Models

LL-STEPNet provides pre-built models for common CAD analysis tasks:

### Part Classification

```python
from stepnet import STEPForClassification

model = STEPForClassification(
    vocab_size=50000,
    num_classes=10,  # bracket, housing, shaft, gear, etc.
    output_dim=1024
)

# Classify STEP file
logits = model(token_ids, topology_data=topology)
predicted_class = torch.argmax(logits, dim=1)
```

### Property Prediction

```python
from stepnet import STEPForPropertyPrediction

model = STEPForPropertyPrediction(
    vocab_size=50000,
    num_properties=6,  # volume, mass, surface_area, bbox dims
    output_dim=1024
)

# Predict physical properties
properties = model(token_ids, topology_data=topology)
# Returns: [volume, surface_area, mass, bbox_x, bbox_y, bbox_z]
```

### Similarity Search

```python
from stepnet import STEPForSimilarity

model = STEPForSimilarity(
    vocab_size=50000,
    embedding_dim=512
)

# Get embeddings for similarity search
embedding1 = model(token_ids_1, topology_data=topology_1)
embedding2 = model(token_ids_2, topology_data=topology_2)

# Compute cosine similarity (embeddings are L2-normalized)
similarity = torch.matmul(embedding1, embedding2.T)
```

### Captioning

```python
from stepnet import STEPForCaptioning

model = STEPForCaptioning(
    vocab_size=50000,
    decoder_vocab_size=50000,
    max_caption_length=128
)

# Generate description
logits = model(token_ids, caption_ids=target_captions, topology_data=topology)
```

### Question Answering

```python
from stepnet import STEPForQA

model = STEPForQA(
    step_vocab_size=50000,
    text_vocab_size=50000,
    output_dim=1024
)

# Answer questions about CAD part
answer_logits = model(
    step_token_ids=token_ids,
    question_token_ids=question_ids,
    answer_token_ids=answer_ids,  # For training
    topology_data=topology
)
```

## Training

### Using the Built-in Trainer

```python
from stepnet import STEPForClassification, create_dataloader, STEPTrainer

# Create data loaders
train_loader = create_dataloader(
    file_paths=train_files,
    labels=train_labels,
    batch_size=8,
    use_topology=True
)

val_loader = create_dataloader(
    file_paths=val_files,
    labels=val_labels,
    batch_size=8,
    use_topology=True
)

# Initialize model
model = STEPForClassification(num_classes=10)

# Train
trainer = STEPTrainer(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    checkpoint_dir='checkpoints'
)

trainer.train(num_epochs=10, save_every=2)
```

### Example Training Scripts

See the `examples/` directory for complete training scripts:

- `train_classification.py` - Part classification training
- `train_property_prediction.py` - Property prediction training
- `inference_example.py` - Using trained models for inference

## Testing

Run tests with actual STEP files:

```bash
cd ll_stepnet
python tests/test_with_step_files.py
```

This will test all components on the STEP files in `data/test_files/`.

## Data Format

### Dataset Structure

```
data/
├── train/
│   ├── part1.step
│   ├── part2.step
│   └── ...
├── val/
│   ├── part1.step
│   └── ...
└── labels.json
```

### Labels File (labels.json)

For classification:
```json
{
  "part1.step": 0,
  "part2.step": 1
}
```

For property prediction:
```json
{
  "part1.step": [100.5, 250.3, 45.2, 10.0, 15.0, 8.0],
  "part2.step": [200.1, 400.5, 90.3, 20.0, 25.0, 12.0]
}
```

## Design Philosophy

- **Separation of Concerns**: Each module has a single responsibility
- **Standard PyTorch**: Follows nn.Module conventions
- **No Over-Engineering**: Simple, clean implementations
- **Flexible**: Can use components independently or together
- **Production Ready**: Includes training, evaluation, and inference utilities

## Model Architecture

```
STEP File
    ↓
Tokenizer → Token IDs [batch, seq_len]
    ↓
Transformer Encoder → Token Features [batch, seq_len, dim]
    ↓
Mean Pooling → Token Embedding [batch, dim]

Feature Extractor → Geometric Features
    ↓
Topology Builder → Graph (adjacency matrix, node features)
    ↓
Graph Neural Network → Graph Features [num_nodes, dim]
    ↓
Graph Pooling → Graph Embedding [batch, dim]

Token Embedding + Graph Embedding → Fusion Layer → Final Encoding [batch, output_dim]
    ↓
Task-Specific Head → Predictions
```

## License

MIT
