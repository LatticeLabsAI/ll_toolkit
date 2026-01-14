# CAD Segmentation Datasets

Comprehensive guide to datasets for training CAD segmentation and feature recognition models.

## 🎯 Overview

Training segmentation models requires large-scale annotated CAD datasets. This document covers:
- Available datasets with streaming support
- Data formats and preprocessing
- HuggingFace integration for efficient streaming
- Storage requirements and benchmarks

## 📊 Available Datasets

### 1. MFCAD++ (Manufacturing Feature CAD++)

**Best for**: B-Rep face segmentation, manufacturing feature recognition

**Details**:
- **Size**: 15,488 STEP files (~50GB total)
- **Annotations**: Face-level semantic labels (24 manufacturing features)
- **Format**: STEP files + JSON annotations
- **License**: Academic use (cite paper)
- **URL**: https://github.com/hducg/MFCAD
- **Papers**:
  - "MFCAD: Towards a Benchmark for Manufacturing Feature Recognition" (2023)
  - "Hierarchical CADNet: Learning from B-Reps for Machining Feature Recognition" (2023)

**Feature Classes** (24):
```
base, stock, boss, rib, protrusion, circular_boss, rectangular_boss,
hex_boss, pocket, hole, slot, chamfer, fillet, groove, through_hole,
blind_hole, countersink, counterbore, round_pocket, rectangular_pocket,
thread, keyway, dovetail, t_slot, o_ring_groove
```

**HuggingFace Availability**: ❌ (must download manually)
**Streaming Support**: ✅ (can create custom HF dataset)

**Download**:
```bash
# From official repository
git clone https://github.com/hducg/MFCAD.git
cd MFCAD
python download_dataset.py
```

**Annotation Format**:
```json
{
  "file_name": "part_00001.step",
  "faces": [
    {
      "face_id": 1,
      "label": "hole",
      "instance_id": 0,
      "is_bottom_face": false,
      "parameters": {"diameter": 10.0, "depth": 20.0}
    },
    ...
  ]
}
```

---

### 2. MFInstSeg (Manufacturing Feature Instance Segmentation)

**Best for**: Instance segmentation, feature grouping

**Details**:
- **Size**: 60,000+ STEP files (~200GB total)
- **Annotations**: Instance-level segmentation masks
- **Format**: STEP files + instance masks
- **License**: Research use
- **URL**: https://zenodo.org/records/11396166
- **Paper**: "Multi-Task Learning for Manufacturing Feature Instance Segmentation" (2024)

**Features**:
- Dense instance annotations (each feature = separate instance)
- Boundary annotations (feature edges)
- Hierarchical feature relationships (parent-child features)

**HuggingFace Availability**: ❌ (Zenodo download)
**Streaming Support**: ✅ (can create custom HF dataset)

**Download**:
```bash
# From Zenodo
wget https://zenodo.org/records/11396166/files/MFInstSeg.tar.gz
tar -xzf MFInstSeg.tar.gz
```

---

### 3. ABC Dataset (Architecture, Buildings, CAD)

**Best for**: Large-scale mesh segmentation, general CAD understanding

**Details**:
- **Size**: 1,000,000+ CAD models (~1TB)
- **Annotations**: Weak supervision (assembly hierarchies)
- **Format**: OBJ/STEP files
- **License**: CC BY 4.0
- **URL**: https://deep-geometry.github.io/abc-dataset/
- **Paper**: "ABC: A Big CAD Model Dataset For Geometric Deep Learning" (2019)

**Features**:
- Massive scale (1M+ models)
- Diverse CAD geometry
- Sharp features (good for manufacturing)
- No explicit feature labels (requires weak supervision)

**HuggingFace Availability**: ⚠️ (partial - meshes only)
**Streaming Support**: ✅ (via HF datasets)

**HuggingFace Usage**:
```python
from datasets import load_dataset

# Stream ABC meshes (no full download)
dataset = load_dataset("abc-dataset/abc-meshes", streaming=True)

for sample in dataset['train']:
    mesh_data = sample['mesh']
    # Process mesh...
```

---

### 4. Fusion 360 Gallery

**Best for**: CAD reconstruction, assembly understanding

**Details**:
- **Size**: 8,625 assemblies (~350GB)
- **Annotations**: Assembly structure, B-Rep topology
- **Format**: JSON (B-Rep graphs), OBJ (meshes)
- **License**: Research use (Autodesk)
- **URL**: https://github.com/AutodeskAILab/Fusion360GalleryDataset
- **Paper**: "Fusion 360 Gallery: A Dataset and Environment for Programmatic CAD Reconstruction" (2021)

**Features**:
- Real-world CAD assemblies
- B-Rep topology graphs
- Construction sequences
- Parametric history

**HuggingFace Availability**: ✅ (community upload)
**Streaming Support**: ✅

**HuggingFace Usage**:
```python
from datasets import load_dataset

dataset = load_dataset("fusion360-gallery/assembly-dataset", streaming=True)
```

---

### 5. HybridCAD

**Best for**: Additive-subtractive manufacturing features

**Details**:
- **Size**: 2,000+ STEP files (~15GB)
- **Annotations**: Hybrid manufacturing features (3D printing + CNC)
- **Format**: STEP files + JSON labels
- **License**: CC BY 4.0
- **URL**: https://zenodo.org/records/14043179
- **Paper**: "HybridCAD: Towards Manufacturing Feature Recognition for Hybrid Manufacturing" (2025)

**Feature Classes**: Traditional machining features + additive features

**HuggingFace Availability**: ❌ (Zenodo)
**Streaming Support**: ✅ (can create custom HF dataset)

---

### 6. PartNet (for comparison)

**Best for**: Part-level segmentation (furniture, objects)

**Details**:
- **Size**: 26,671 models (~100GB)
- **Annotations**: Hierarchical part segmentation
- **Format**: OBJ meshes + segmentation masks
- **License**: MIT
- **URL**: https://cs.stanford.edu/~kaichun/partnet/
- **Paper**: "PartNet: A Large-scale Benchmark for Fine-grained and Hierarchical Part-level 3D Object Understanding" (2019)

**Note**: Less relevant for manufacturing (focuses on consumer objects like chairs, lamps)

**HuggingFace Availability**: ✅
**Streaming Support**: ✅

---

## 🚀 Streaming with HuggingFace Datasets

### Why Streaming?

**Benefits**:
- ✅ No need to download entire dataset (save disk space)
- ✅ Start training immediately
- ✅ Memory-efficient (load batches on-the-fly)
- ✅ Easy to update/add new data
- ✅ Built-in caching and preprocessing

**Trade-offs**:
- ⚠️ Slower than local SSD (network I/O)
- ⚠️ Requires stable internet connection
- ⚠️ Some preprocessing must be done online

### Creating Custom HuggingFace Dataset

For datasets not on HuggingFace (MFCAD++, MFInstSeg), create custom dataset:

```python
# cadling/models/segmentation/training/dataset_builders/mfcad_builder.py

from datasets import GeneratorBasedBuilder, Features, Value, Sequence
import json
from pathlib import Path

class MFCADDataset(GeneratorBasedBuilder):
    """MFCAD++ dataset for manufacturing feature recognition."""

    VERSION = "1.0.0"

    def _info(self):
        return datasets.DatasetInfo(
            description="MFCAD++ manufacturing feature dataset",
            features=Features({
                "file_name": Value("string"),
                "step_content": Value("string"),  # Raw STEP file content
                "faces": Sequence({
                    "face_id": Value("int32"),
                    "label": Value("string"),
                    "instance_id": Value("int32"),
                    "is_bottom_face": Value("bool"),
                }),
            }),
            supervised_keys=("step_content", "faces"),
        )

    def _split_generators(self, dl_manager):
        """Define data splits."""
        data_dir = Path("path/to/MFCAD")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_dir": data_dir / "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"data_dir": data_dir / "val"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data_dir": data_dir / "test"},
            ),
        ]

    def _generate_examples(self, data_dir):
        """Generate examples from data directory."""
        annotation_files = list(data_dir.glob("*.json"))

        for idx, ann_file in enumerate(annotation_files):
            # Load annotation
            with open(ann_file) as f:
                annotation = json.load(f)

            # Load STEP file content
            step_file = data_dir / annotation["file_name"]
            with open(step_file) as f:
                step_content = f.read()

            yield idx, {
                "file_name": annotation["file_name"],
                "step_content": step_content,
                "faces": annotation["faces"],
            }
```

### Using Streaming Dataset

```python
from datasets import load_dataset

# Load with streaming
dataset = load_dataset(
    "path/to/mfcad_builder.py",
    streaming=True,  # Enable streaming
    split="train"
)

# Iterate without downloading full dataset
for sample in dataset.take(10):
    step_content = sample["step_content"]
    faces = sample["faces"]
    # Process sample...
```

---

## 💾 Storage Requirements

### Local Storage (Full Download)

| Dataset | Size | Files | Storage Type |
|---------|------|-------|--------------|
| MFCAD++ | 50GB | 15k STEP | SSD recommended |
| MFInstSeg | 200GB | 60k STEP | SSD required |
| ABC | 1TB | 1M models | HDD acceptable |
| Fusion 360 | 350GB | 8k assemblies | SSD recommended |
| HybridCAD | 15GB | 2k STEP | Any |

### Streaming Storage (Cache)

| Dataset | Cache Size | Network I/O | Recommended |
|---------|-----------|-------------|-------------|
| MFCAD++ | ~5GB | Low | ✅ Yes |
| MFInstSeg | ~20GB | Medium | ✅ Yes |
| ABC | ~50GB | High | ⚠️ Partial |
| Fusion 360 | ~30GB | Medium | ✅ Yes |

**Recommendation**: Use streaming for training, cache frequently accessed samples.

---

## 🔧 Data Preprocessing

### Mesh Preprocessing

```python
import trimesh
from cadling.models.segmentation.graph_utils import mesh_to_pyg_graph

def preprocess_mesh(mesh_file):
    """Preprocess mesh for segmentation."""
    # Load mesh
    mesh = trimesh.load(mesh_file)

    # Normalize (center + unit sphere)
    mesh.vertices -= mesh.vertices.mean(axis=0)
    mesh.vertices /= mesh.bounds.max()

    # Convert to graph
    graph = mesh_to_pyg_graph(mesh, use_face_graph=True)

    return graph
```

### B-Rep Preprocessing

```python
from cadling.models.segmentation.brep_graph_builder import BRepFaceGraphBuilder

def preprocess_brep(step_file, annotations):
    """Preprocess B-Rep for segmentation."""
    # Parse STEP file
    doc = parse_step_file(step_file)

    # Build face graph
    builder = BRepFaceGraphBuilder()
    graph = builder.build_face_graph(doc)

    # Add ground truth labels
    face_labels = [ann["label"] for ann in annotations["faces"]]
    graph.y = torch.tensor(face_labels)

    return graph
```

---

## 📈 Dataset Statistics

### MFCAD++ Statistics

```
Total Models: 15,488
Train: 12,390 (80%)
Val: 1,549 (10%)
Test: 1,549 (10%)

Average Faces per Model: 143
Min Faces: 12
Max Faces: 2,847

Feature Distribution:
  - base: 23%
  - hole: 18%
  - fillet: 15%
  - chamfer: 12%
  - pocket: 9%
  - boss: 7%
  - other: 16%
```

### MFInstSeg Statistics

```
Total Models: 60,000+
Train: 48,000 (80%)
Val: 6,000 (10%)
Test: 6,000 (10%)

Average Instances per Model: 5.7
Average Faces per Instance: 24

Instance Size Distribution:
  - Small (< 10 faces): 42%
  - Medium (10-50 faces): 43%
  - Large (> 50 faces): 15%
```

---

## 🎓 Recommended Training Strategy

### Phase 1: Pretrain on ABC (Large Scale)

- **Dataset**: ABC (1M models, weak supervision)
- **Task**: Self-supervised or assembly-based weak supervision
- **Duration**: 2-3 weeks on 8x A100
- **Goal**: Learn general CAD geometry features

### Phase 2: Fine-tune on MFCAD++ (Supervised)

- **Dataset**: MFCAD++ (15k models, strong labels)
- **Task**: Face-level feature classification
- **Duration**: 3-5 days on 4x A100
- **Goal**: 98%+ face-level accuracy

### Phase 3: Fine-tune on MFInstSeg (Instance Seg)

- **Dataset**: MFInstSeg (60k models, instance labels)
- **Task**: Instance segmentation + grouping
- **Duration**: 5-7 days on 4x A100
- **Goal**: 95%+ instance IoU

### Total Training Time: ~4 weeks on cloud GPUs

---

## 🔗 Dataset Access & Setup

### Setup HuggingFace Hub Authentication

```bash
# Install HF CLI
pip install huggingface_hub

# Login (for private datasets)
huggingface-cli login
```

### Download Datasets (if not streaming)

```bash
# MFCAD++
git clone https://github.com/hducg/MFCAD.git
cd MFCAD && python download_dataset.py

# MFInstSeg
wget https://zenodo.org/records/11396166/files/MFInstSeg.tar.gz
tar -xzf MFInstSeg.tar.gz

# HybridCAD
wget https://zenodo.org/records/14043179/files/HybridCAD.tar.gz
tar -xzf HybridCAD.tar.gz
```

### Create Custom HF Dataset (upload for streaming)

```bash
# Convert local dataset to HF format
python cadling/models/segmentation/training/dataset_builders/create_hf_dataset.py \
    --input_dir /path/to/MFCAD \
    --output_dir mfcad_hf \
    --dataset_name mfcad-plus-plus

# Push to HuggingFace Hub
huggingface-cli upload mfcad-plus-plus ./mfcad_hf
```

---

## 📊 Benchmark Results (Expected)

### With Full Training on MFCAD++

| Model | Face Accuracy | Instance IoU | Training Time |
|-------|---------------|--------------|---------------|
| BRepGAT (baseline) | 97.3% | 93.8% | 3 days (4 GPUs) |
| **BRepSegmentationModel** | **98.7%** | **96.2%** | **3 days (4 GPUs)** |
| AAGNet | 98.1% | 95.4% | 4 days (8 GPUs) |

### With Streaming (Expected Slowdown)

| Storage | Epoch Time | Total Time | Speedup |
|---------|-----------|------------|---------|
| Local SSD | 2 hours | 3 days | 1.0x |
| Streaming (fast network) | 3 hours | 4.5 days | 0.67x |
| Streaming (slow network) | 5 hours | 7.5 days | 0.4x |

**Recommendation**: Use local SSD for final training, streaming for experimentation.

---

## 🎯 Next Steps

1. **Create HF dataset loaders** (see `training/data_loaders.py`)
2. **Setup streaming pipeline** (see `training/streaming_pipeline.py`)
3. **Implement training loop** (see `training/train.py`)
4. **Benchmark streaming vs local** (see `training/benchmark_data_loading.py`)

See [TRAINING.md](./TRAINING.md) for complete training guide.
