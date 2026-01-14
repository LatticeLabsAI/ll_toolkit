# CAD Segmentation Model Training

Complete training infrastructure for CAD segmentation models with HuggingFace streaming support.

## 🎯 Overview

This directory provides:
- **Data Loaders**: HuggingFace dataset loaders for CAD datasets (MFCAD++, MFInstSeg, ABC, Fusion 360)
- **Streaming Pipeline**: Efficient data streaming without full dataset download
- **Training Loop**: Complete training infrastructure with mixed precision, checkpointing, early stopping
- **Dataset Builders**: Tools to convert local datasets to HuggingFace format

## 📁 Structure

```
training/
├── README.md                      # This file
├── __init__.py                    # Module exports
├── data_loaders.py                # HuggingFace dataset loaders
├── streaming_pipeline.py          # Streaming data pipeline
├── train.py                       # Training loop
└── dataset_builders/              # Dataset conversion tools
    ├── __init__.py
    ├── mfcad_builder.py          # MFCAD++ HF dataset builder
    └── create_hf_dataset.py      # Conversion script
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torch-geometric datasets huggingface_hub
```

### 2. Stream a Dataset (No Download)

```python
from cadling.models.segmentation.training import get_data_loader

# Stream MFCAD++ without downloading full dataset
loader = get_data_loader(
    dataset_name="path/to/mfcad",
    dataset_type="mfcad",
    streaming=True,
    split="train"
)

# Iterate over samples
for sample in loader.take(10):
    print(f"File: {sample['file_name']}")
    print(f"Faces: {sample['num_faces']}")
    print(f"Labels: {sample['face_labels'][:5]}")
```

### 3. Create Streaming Pipeline

```python
from cadling.models.segmentation.training import create_streaming_pipeline
from cadling.models.segmentation.training.streaming_pipeline import build_brep_graph

# Create pipeline with on-the-fly graph construction
pipeline = create_streaming_pipeline(
    dataset_name="path/to/mfcad",
    graph_builder=build_brep_graph,
    dataset_type="mfcad",
    batch_size=16,
    streaming=True,
    cache_graphs=True
)

# Iterate over batches
for batch in pipeline.take(5):
    print(f"Batch: {batch.num_graphs} graphs, {batch.num_nodes} nodes")
```

### 4. Train a Model

```python
from cadling.models.segmentation.architectures import HybridGATTransformer
from cadling.models.segmentation.training import (
    SegmentationTrainer,
    TrainingConfig,
    create_streaming_pipeline,
)
from cadling.models.segmentation.training.streaming_pipeline import build_brep_graph

# Create model
model = HybridGATTransformer(
    in_dim=24,
    num_classes=24,
    gat_hidden_dim=256,
    transformer_hidden_dim=512
)

# Create pipelines
train_pipeline = create_streaming_pipeline(
    dataset_name="path/to/mfcad",
    graph_builder=build_brep_graph,
    split="train",
    batch_size=16,
    streaming=True
)

val_pipeline = create_streaming_pipeline(
    dataset_name="path/to/mfcad",
    graph_builder=build_brep_graph,
    split="val",
    batch_size=16,
    streaming=True
)

# Configure training
config = TrainingConfig(
    num_epochs=50,
    learning_rate=1e-3,
    batch_size=16,
    checkpoint_dir="checkpoints/brep_seg",
    early_stopping_patience=10,
    mixed_precision=True  # Use AMP on GPU
)

# Train
trainer = SegmentationTrainer(model, config)
history = trainer.train(train_pipeline, val_pipeline)

print(f"Training complete! Best val loss: {trainer.best_val_loss:.4f}")
```

## 📊 Data Loaders

### Available Loaders

#### MFCADDataLoader
- **Dataset**: MFCAD++ (15,488 STEP files)
- **Features**: 24 manufacturing feature classes
- **Size**: ~50GB
- **Annotations**: Face-level labels + instance IDs + parameters

```python
from cadling.models.segmentation.training import MFCADDataLoader

loader = MFCADDataLoader(
    dataset_name="path/to/mfcad",
    streaming=True,
    split="train"
)

for sample in loader.take(1):
    print(f"File: {sample['file_name']}")
    print(f"STEP content: {len(sample['step_content'])} chars")
    print(f"Face labels: {sample['face_labels']}")
    print(f"Instances: {sample['face_instances']}")
    print(f"Feature classes: {sample['feature_class_names']}")
```

#### MFInstSegDataLoader
- **Dataset**: MFInstSeg (60,000+ STEP files)
- **Features**: Instance segmentation masks
- **Size**: ~200GB

```python
from cadling.models.segmentation.training import MFInstSegDataLoader

loader = MFInstSegDataLoader(
    dataset_name="path/to/mfinstseg",
    streaming=True
)
```

#### ABCDataLoader
- **Dataset**: ABC (1M+ CAD models)
- **Features**: Weak supervision from assembly hierarchies
- **Size**: ~1TB
- **Best with**: Streaming (avoid full download)

```python
from cadling.models.segmentation.training import ABCDataLoader

loader = ABCDataLoader(
    dataset_name="abc-dataset/abc-meshes",
    streaming=True,
    split="train"
)
```

#### Fusion360DataLoader
- **Dataset**: Fusion 360 Gallery (8,625 assemblies)
- **Features**: B-Rep topology, assembly structure
- **Size**: ~350GB

```python
from cadling.models.segmentation.training import Fusion360DataLoader

loader = Fusion360DataLoader(
    dataset_name="fusion360-gallery/assembly-dataset",
    streaming=True
)
```

### Auto-Detection

```python
from cadling.models.segmentation.training import get_data_loader

# Auto-detect dataset type from name
loader = get_data_loader(
    dataset_name="path/to/mfcad",
    dataset_type="auto",  # Auto-detect from name
    streaming=True
)
```

## 🌊 Streaming Pipeline

The streaming pipeline converts dataset samples to PyTorch Geometric graphs on-the-fly.

### Custom Graph Builder

```python
import torch
from torch_geometric.data import Data

def my_graph_builder(sample: dict) -> Data:
    """Convert sample to PyG graph."""
    # Extract features from sample
    num_faces = sample['num_faces']
    face_labels = sample['face_labels']

    # Create graph
    graph = Data(
        x=torch.randn(num_faces, 24),  # Node features
        edge_index=torch.randint(0, num_faces, (2, num_faces * 3)),
        edge_attr=torch.randn(num_faces * 3, 8),
        y=torch.tensor(face_labels, dtype=torch.long)
    )

    return graph

# Use custom builder
pipeline = create_streaming_pipeline(
    dataset_name="path/to/dataset",
    graph_builder=my_graph_builder,
    batch_size=16
)
```

### Built-in Graph Builders

```python
from cadling.models.segmentation.training.streaming_pipeline import (
    build_brep_graph,      # For B-Rep/STEP data
    build_mesh_graph,      # For mesh/STL data
    build_instance_seg_graph,  # For instance segmentation
)
```

### Pipeline Configuration

```python
pipeline = create_streaming_pipeline(
    dataset_name="path/to/mfcad",
    graph_builder=build_brep_graph,
    dataset_type="mfcad",
    split="train",
    batch_size=16,
    streaming=True,          # Avoid full download
    cache_graphs=True,       # Cache constructed graphs
    max_cache_size=1000,     # Max graphs in cache
    shuffle=True,            # Shuffle dataset
    shuffle_buffer_size=10000,  # Shuffle buffer (streaming)
    num_workers=0,           # Multi-process loading
)
```

## 🏋️ Training

### Training Configuration

```python
from cadling.models.segmentation.training import TrainingConfig

config = TrainingConfig(
    num_epochs=100,
    learning_rate=1e-3,
    weight_decay=1e-4,
    batch_size=16,
    gradient_clip=1.0,          # Gradient clipping
    lr_scheduler="plateau",     # 'plateau', 'cosine', or None
    early_stopping_patience=10, # Early stopping
    checkpoint_dir="checkpoints",
    checkpoint_frequency=5,     # Save every 5 epochs
    log_frequency=10,           # Log every 10 batches
    device="auto",              # 'cuda', 'cpu', or 'auto'
    mixed_precision=True,       # Use AMP (GPU only)
)
```

### Custom Loss Function

```python
import torch.nn as nn

# Multi-task loss
class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.semantic_loss = nn.CrossEntropyLoss()
        self.instance_loss = discriminative_loss

    def forward(self, semantic_logits, instance_embeds, y, instance_labels):
        loss_sem = self.semantic_loss(semantic_logits, y)
        loss_inst = self.instance_loss(instance_embeds, instance_labels)
        return loss_sem + 0.5 * loss_inst

# Use custom loss
trainer = SegmentationTrainer(
    model=model,
    config=config,
    criterion=MultiTaskLoss()
)
```

### Resume from Checkpoint

```python
trainer = SegmentationTrainer(model, config)

# Load checkpoint
trainer.load_checkpoint("checkpoints/best_model.pt")

# Continue training
trainer.train(train_pipeline, val_pipeline)
```

## 🔧 Dataset Builders

### Create HuggingFace Dataset from Local Files

```bash
# Convert MFCAD++ to HF format
python -m cadling.models.segmentation.training.dataset_builders.create_hf_dataset \
    --input_dir /path/to/MFCAD \
    --output_dir ./mfcad_hf \
    --dataset_type mfcad \
    --dataset_name mfcad-plus-plus

# Upload to HuggingFace Hub
python -m cadling.models.segmentation.training.dataset_builders.create_hf_dataset \
    --input_dir /path/to/MFCAD \
    --output_dir ./mfcad_hf \
    --dataset_type mfcad \
    --dataset_name mfcad-plus-plus \
    --upload \
    --hf_username your-username
```

### Use Custom Dataset Builder

```python
from datasets import load_dataset

# Load from builder script
dataset = load_dataset(
    "cadling/models/segmentation/training/dataset_builders/mfcad_builder.py",
    data_dir="/path/to/MFCAD",
    streaming=True,
    split="train"
)

for sample in dataset.take(5):
    print(f"File: {sample['file_name']}")
    print(f"Faces: {len(sample['faces'])}")
```

## 📈 Storage Requirements

### Local Storage (Full Download)

| Dataset | Size | Recommended Storage |
|---------|------|---------------------|
| MFCAD++ | 50GB | SSD |
| MFInstSeg | 200GB | SSD |
| ABC | 1TB | HDD acceptable |
| Fusion 360 | 350GB | SSD |

### Streaming (Cache Only)

| Dataset | Cache Size | Network I/O | Speed |
|---------|-----------|-------------|-------|
| MFCAD++ | ~5GB | Low | 1.0x |
| MFInstSeg | ~20GB | Medium | 0.67x |
| ABC | ~50GB | High | 0.4x |

**Recommendation**: Use streaming for experimentation, local SSD for final training.

## 🎓 Training Strategies

### Phase 1: Pretrain on ABC (Large Scale)

```python
# Pretrain on 1M models for general CAD understanding
train_pipeline = create_streaming_pipeline(
    dataset_name="abc-dataset/abc-meshes",
    graph_builder=build_mesh_graph,
    split="train",
    batch_size=32,
    streaming=True  # Essential for 1TB dataset
)

config = TrainingConfig(
    num_epochs=30,
    learning_rate=1e-3,
    batch_size=32
)

trainer = SegmentationTrainer(model, config)
trainer.train(train_pipeline)
```

### Phase 2: Fine-tune on MFCAD++ (Supervised)

```python
# Fine-tune on labeled data
train_pipeline = create_streaming_pipeline(
    dataset_name="path/to/mfcad",
    graph_builder=build_brep_graph,
    split="train",
    batch_size=16,
    streaming=False  # Can download (50GB)
)

config = TrainingConfig(
    num_epochs=50,
    learning_rate=5e-4,  # Lower LR for fine-tuning
    early_stopping_patience=10
)

trainer = SegmentationTrainer(model, config)
trainer.train(train_pipeline, val_pipeline)
```

### Phase 3: Fine-tune on MFInstSeg (Instance Seg)

```python
# Fine-tune for instance segmentation
train_pipeline = create_streaming_pipeline(
    dataset_name="path/to/mfinstseg",
    graph_builder=build_instance_seg_graph,
    split="train",
    batch_size=8,
    streaming=True  # 200GB - use streaming
)

# Multi-task loss
criterion = MultiTaskLoss()

trainer = SegmentationTrainer(model, config, criterion=criterion)
trainer.train(train_pipeline, val_pipeline)
```

## 🔍 Troubleshooting

### Out of Memory (GPU)

Reduce batch size or enable chunking:
```python
config = TrainingConfig(
    batch_size=8,  # Reduce from 16
    mixed_precision=True  # Use AMP to save memory
)
```

### Slow Streaming

Enable graph caching:
```python
pipeline = create_streaming_pipeline(
    dataset_name="path/to/dataset",
    graph_builder=build_brep_graph,
    cache_graphs=True,
    max_cache_size=5000  # Increase cache
)
```

### Dataset Not Found

Ensure correct directory structure:
```
MFCAD/
├── train/
│   ├── part_00001.step
│   ├── part_00001.json
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

## 📚 See Also

- [DATASETS.md](./DATASETS.md) - Dataset details and download instructions
- [../README.md](../README.md) - Main segmentation models documentation
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/) - Streaming docs

## 📄 License

Part of the cadling project. See main LICENSE file.
