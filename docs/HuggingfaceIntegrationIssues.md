# HuggingFace CAD Integration: Plan vs Implementation Discrepancies

A systematic comparison of the integration plan (`huggingface_cad_integration_plan.md`) against the actual codebase implementation. This document identifies gaps, mismatches, and deviations to guide remediation.

---

## Executive Summary

| Category | Status | Critical Issues |
|----------|--------|-----------------|
| **Schemas** | ⚠️ Partial | Wrong data types (int32 vs int8/16), missing normalization metadata |
| **Streaming Pipeline** | ✅ Complete | Minor feature gaps (predicate pushdown) |
| **HF Builders** | ⚠️ Partial | Not true ArrowBasedBuilder subclasses |
| **Dataset Loaders** | ⚠️ Partial | Local-only, no Hub streaming |
| **Trainers** | ❌ Incomplete | Only VAE has streaming trainer |
| **Collators** | ⚠️ Partial | Missing label shifting for autoregressive |
| **CLI** | ✅ Complete | Minor option gaps |
| **Hub Publisher** | ✅ Complete | Works as designed |

---

## 1. Schema Discrepancies

### 1.1 Command Sequence Schema

| Field | Planned | Implemented | Impact |
|-------|---------|-------------|--------|
| `command_types` | `pa.list_(pa.int8(), 60)` fixed-length | `pa.list_(pa.int32())` variable | 4x memory overhead; variable length breaks batch stacking |
| `parameters` | `pa.list_(pa.int16(), 60*16)` 2D | `pa.list_(pa.int32())` flat | 2x memory overhead; requires reshape on load |
| `mask` | `pa.list_(pa.float32(), 60)` | `pa.list_(pa.float32())` | Minor - variable vs fixed |
| `normalization_center` | `pa.list_(pa.float32(), 3)` | **Missing** | Cannot dequantize without this |
| `normalization_scale` | `pa.float32()` | **Missing** | Cannot dequantize without this |

**File:** `cadling/cadling/data/schemas.py`

```python
# PLANNED (efficient)
pa.field("command_types", pa.list_(pa.int8(), 60), nullable=False)
pa.field("parameters", pa.list_(pa.int16(), 60 * 16), nullable=False)

# IMPLEMENTED (inefficient)
pa.field("command_types", pa.list_(pa.int32()), nullable=False)
pa.field("parameters", pa.list_(pa.int32()), nullable=False)
```

### 1.2 Text-CAD Schema

| Field | Planned | Implemented | Impact |
|-------|---------|-------------|--------|
| `text_abstract` | Separate field | Not present | Missing annotation level |
| `text_intermediate` | Separate field | Not present | Missing annotation level |
| `text_detailed` | Separate field | ✅ `text_detailed` | OK |
| `text_expert` | Separate field | Not present | Missing annotation level |
| `image_embedding` | `pa.list_(pa.float32(), 768)` | **Missing** | Cannot do image-conditioned generation |

**Impact:** Text2CAD dataset uses 4 complexity levels. Current schema only supports 2, limiting training data granularity.

### 1.3 B-Rep Graph Schema

| Field | Planned | Implemented | Impact |
|-------|---------|-------------|--------|
| Core fields | All present | ✅ All present | OK |
| `uv_points` for UV-Net | Optional field | ✅ Optional field | OK |
| `coedge_sequence` | Optional field | ✅ Optional field | OK |

**Status:** ✅ B-Rep schema matches plan.

---

## 2. Streaming Pipeline Discrepancies

### 2.1 CADStreamingConfig

| Parameter | Planned | Implemented | Impact |
|-----------|---------|-------------|--------|
| `dataset_id` | ✅ | ✅ | OK |
| `streaming` | ✅ | ✅ | OK |
| `columns` | ✅ | ✅ | OK - column projection works |
| `filters` | For predicate pushdown | **Missing** | Cannot skip row groups based on min/max stats |
| `world_size` | Standard DDP naming | Uses `num_shards` | Non-standard naming |
| `rank` | Standard DDP naming | Uses `shard_index` | Non-standard naming |

**File:** `cadling/cadling/data/streaming.py`

```python
# PLANNED
class CADStreamingConfig:
    filters: Optional[List[Tuple]] = None  # e.g., [("num_commands", ">=", 10)]
    world_size: int = 1
    rank: int = 0

# IMPLEMENTED
class CADStreamingConfig:
    # filters NOT present
    num_shards: int = 1      # Non-standard naming
    shard_index: int = 0     # Non-standard naming
```

### 2.2 Distributed Training Support

| Feature | Planned | Implemented | Status |
|---------|---------|-------------|--------|
| `split_dataset_by_node()` | Use HF's distributed utility | ✅ Used with fallback | OK |
| Shard-based distribution | `num_shards ≥ world_size × num_workers` | ✅ Documented | OK |
| `set_epoch()` for shuffle | Reset shuffle seed per epoch | ✅ Implemented | OK |

---

## 3. HuggingFace Builder Discrepancies

### 3.1 Architecture Pattern

| Aspect | Planned | Implemented | Impact |
|--------|---------|-------------|--------|
| Base class | `datasets.ArrowBasedBuilder` | Standalone class | Cannot use `load_dataset("path/to/builder.py")` |
| `_generate_tables()` | Yields `(key, pa.Table)` incrementally | Loads all, then converts | Memory-bound for large datasets |
| `_split_generators()` | HF-native split handling | Manual split iteration | Less integration with HF ecosystem |

**File:** `cadling/cadling/data/hf_builders/cad_dataset_builder.py`

```python
# PLANNED - True HF Builder
class CADCommandSequenceBuilder(datasets.ArrowBasedBuilder):
    def _generate_tables(self, split):
        for batch in self._stream_batches():
            yield batch_key, pa.Table.from_pydict(batch)

# IMPLEMENTED - Standalone class
class CADCommandSequenceBuilder:
    def to_arrow_tables(self):
        samples = list(self.generate_samples(split))  # Loads ALL into memory
        return samples_to_table(samples, self._schema)
```

### 3.2 Missing WebDataset Builder

| Component | Planned | Implemented |
|-----------|---------|-------------|
| `WebDatasetBuilder` for STEP TAR shards | Yes - 1-2GB TAR files with .step/.json/.png | **Not implemented** |

---

## 4. Dataset Loader Discrepancies

### 4.1 Hub Streaming vs Local-Only

| Loader | Planned | Implemented | Impact |
|--------|---------|-------------|--------|
| `DeepCADLoader` | Stream from Hub via `load_dataset()` | Local directory only | Must download 178K files first |
| `ABCLoader` | Stream from Hub | Local directory only | Must download 1M files first |
| `Text2CADLoader` | Stream from Hub | Local directory only | Must download 660K files first |
| `SketchGraphsLoader` | Stream from Hub | Local directory only | Must download 15M files first |

**File:** `cadling/cadling/data/datasets/deepcad_loader.py`

```python
# PLANNED - Hub streaming
class DeepCADLoader:
    def __init__(self, hub_id: str = "latticelabs/deepcad", ...):
        self.dataset = load_dataset(hub_id, streaming=True)

# IMPLEMENTED - Local only
class DeepCADLoader:
    def __init__(self, root_dir: str, ...):  # Requires local path
        split_dir = self.root_dir / self.split
        json_files = sorted(split_dir.glob("*.json"))  # Reads from disk
```

### 4.2 Auto-Download

| Feature | Planned | Implemented |
|---------|---------|-------------|
| `download=True` triggers HF download | Auto-download from Hub | Logs manual instructions only |

---

## 5. Trainer Discrepancies

### 5.1 Streaming Trainer Coverage

| Trainer | Planned | Implemented | File |
|---------|---------|-------------|------|
| `StreamingVAETrainer` | Step-based, mid-stream checkpointing | ✅ Complete | `streaming_vae_trainer.py` |
| `StreamingDiffusionTrainer` | Step-based diffusion training | ❌ **Missing** | N/A |
| `StreamingGANTrainer` | Step-based GAN training | ❌ **Missing** | N/A |

**Impact:** Cannot train BrepGen-style diffusion models on streaming datasets. Current `DiffusionTrainer` requires epoch-based `DataLoader`:

```python
# CURRENT - Epoch-based, requires len(dataset)
class DiffusionTrainer:
    def __init__(self, train_dataloader: DataLoader, ...):  # Not IterableDataset
        for epoch in range(num_epochs):
            for batch in self.train_dataloader:  # Assumes finite length
```

### 5.2 Missing Features in StreamingVAETrainer

| Feature | Planned | Implemented |
|---------|---------|-------------|
| Core training loop | ✅ | ✅ |
| KL warmup on steps | ✅ | ✅ |
| Mid-stream checkpointing | ✅ | ✅ |
| `set_epoch()` integration | ✅ | ✅ |
| Gradient accumulation | ✅ | ✅ |

**Status:** ✅ StreamingVAETrainer is complete.

---

## 6. Collator Discrepancies

### 6.1 CADCommandCollator

| Feature | Planned | Implemented | Impact |
|---------|---------|-------------|--------|
| Dynamic padding | ✅ | ✅ | OK |
| Attention mask generation | ✅ | ✅ | OK |
| Parameter reshaping | ✅ | ✅ | OK |
| **Label shifting** | `labels = input_ids[1:]` for autoregressive | **Missing** | Teacher forcing won't work correctly |

**File:** `cadling/cadling/data/collators.py`

```python
# PLANNED - Shift labels for autoregressive training
def __call__(self, batch):
    ...
    # Create labels shifted by 1 for next-token prediction
    collated["labels"] = collated["command_types"][:, 1:].clone()
    collated["command_types"] = collated["command_types"][:, :-1]

# IMPLEMENTED - No label shifting
def __call__(self, batch):
    ...
    # Labels not created; command_types not shifted
```

### 6.2 CADMultiModalCollator

| Feature | Planned | Implemented |
|---------|---------|-------------|
| Text tokenization | ✅ | ✅ |
| Command collation | ✅ | ✅ |
| Pre-tokenized text handling | ✅ | ✅ |

---

## 7. CLI Discrepancies

### 7.1 Hub Commands

| Command | Planned | Implemented | Gap |
|---------|---------|-------------|-----|
| `cadling hub publish` | ✅ | ✅ | OK |
| `cadling hub preview` | ✅ | ✅ | OK |
| `cadling hub build` | ✅ | ✅ | OK |
| `--schema` option | `command_sequences\|brep_graphs\|text_cad` | **Missing** | Cannot specify schema type |

**File:** `cadling/cadling/cli/hub.py`

```python
# PLANNED
@click.option("--schema", type=click.Choice(["command_sequences", "brep_graphs", "text_cad"]))

# IMPLEMENTED - Only pattern-based file selection
@click.option("--pattern", default="*.parquet")
```

---

## 8. Hub Publisher Discrepancies

| Feature | Planned | Implemented | Status |
|---------|---------|-------------|--------|
| Auto-sharding to ~500MB | ✅ | ✅ | OK |
| Shard naming pattern | `data/{split}-{idx:05d}.parquet` | ✅ | OK |
| Schema enforcement | ✅ | ✅ | OK |
| Dataset card generation | ✅ | ✅ | OK |
| Incremental uploads | Append additional shards | ⚠️ Partial | Works via `upload_file()` but no explicit append API |

---

## Priority Remediation Roadmap

### P0 - Critical (Blocks Training)

1. **Create StreamingDiffusionTrainer**
   - File: `ll_stepnet/stepnet/training/streaming_diffusion_trainer.py`
   - Copy pattern from `StreamingVAETrainer`
   - Required for BrepGen-style training

2. **Fix Schema Data Types**
   - Change `int32` → `int8` for command_types
   - Change `int32` → `int16` for parameters
   - Add fixed-length arrays for efficient batch stacking

3. **Add Label Shifting to Collator**
   - Essential for autoregressive training
   - ~10 lines of code in `CADCommandCollator.__call__()`

### P1 - Important (Limits Functionality)

1. **Add Normalization Metadata to Schema**
   - `normalization_center`, `normalization_scale` fields
   - Required for dequantization during inference

2. **Enable Hub Streaming in Dataset Loaders**
   - Add `hub_id` parameter to `DeepCADLoader`, etc.
   - Use `load_dataset(hub_id, streaming=True)` internally

3. **Add Predicate Pushdown**
   - Add `filters` param to `CADStreamingConfig`
   - Pass through to `load_dataset(..., filters=...)`

### P2 - Nice to Have

1. **Implement WebDataset Builder**
   - For raw STEP file TAR shards
   - Lower priority if Parquet path works

2. **Add 4-Level Text Annotations**
   - Extend schema with abstract/intermediate/detailed/expert
   - Matches Text2CAD dataset structure

3. **Convert Builders to True ArrowBasedBuilder**
   - Enables `load_dataset("latticelabs/builder.py")`
   - Better HF ecosystem integration

---

## Files Requiring Changes

| File | Changes Needed | Priority |
|------|----------------|----------|
| `cadling/data/schemas.py` | Fix types, add normalization fields | P0 |
| `cadling/data/collators.py` | Add label shifting | P0 |
| `ll_stepnet/stepnet/training/streaming_diffusion_trainer.py` | **Create new file** | P0 |
| `cadling/data/streaming.py` | Add `filters` param | P1 |
| `cadling/data/datasets/*.py` | Add Hub streaming support | P1 |
| `cadling/cli/hub.py` | Add `--schema` option | P2 |
| `cadling/data/hf_builders/*.py` | Inherit from ArrowBasedBuilder | P2 |
