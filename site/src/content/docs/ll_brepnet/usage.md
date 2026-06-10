---
title: ll_brepnet — Usage
description: Prepare the Fusion 360 dataset, train the segmentation model, and run inference on STEP files.
sidebar:
  label: Usage
  order: 3
---

The pipeline is **STEP → `.npz` records → train → segment**. Everything below
uses the real Fusion 360 Gallery segmentation dataset (s2.0.0).

## 1. Get the dataset

```bash
curl https://fusion-360-gallery-dataset.s3-us-west-2.amazonaws.com/segmentation/s2.0.0/s2.0.0.zip -o s2.0.0.zip
unzip s2.0.0.zip   # → s2.0.0/{breps/step, breps/seg, train_test.json, segment_names.json}
```

## 2. Prepare records + manifest

`quickstart` extracts each STEP file to a `.npz`, copies the matching `.seg`
labels, and writes a `dataset.json` (official train/test split, train-only
feature standardization, the 8 class names). Use `--train-subset` /
`--test-subset` to work on a subset, or omit them for the full 35,680 solids.

```bash
python -m ll_brepnet.pipelines.quickstart \
  --dataset-dir ./s2.0.0 \
  --output-dir  ./processed \
  --num-workers 12 \
  --train-subset 4000 --test-subset 800
```

## 3. Train

```bash
python -m ll_brepnet.train \
  --dataset-file ./processed/dataset.json \
  --dataset-dir  ./processed \
  --label-dir    ./processed \
  --max-epochs 30 --num-layers 4 --hidden-dim 128 \
  --learning-rate 0.002 --accelerator auto --log-dir ./logs
```

Checkpoints (best by validation mIoU) and TensorBoard logs land under
`--log-dir`. `--accelerator auto` uses an Apple-Silicon GPU (MPS) or CUDA when
available. **On the full official split (no `--*-subset`), this produced test
mIoU ≈ 0.828 / accuracy ≈ 0.947** on the 5,366-solid test split (see
[Overview](/ll_toolkit/ll_brepnet/overview/)).

## 4. Segment STEP files

```bash
python -m ll_brepnet.eval.evaluate \
  --step-dir ./s2.0.0/breps/step \
  --model ./logs/ll_brepnet/version_0/checkpoints/best.ckpt \
  --dataset-file ./processed/dataset.json \
  --output-dir ./predictions
```

Each input solid gets a `<stem>.logits` file: one row per face, one column per
class. The predicted segment of face *i* is the `argmax` of row *i*.

## Python API

```python
from ll_brepnet import (
    extract_brepnet_data_from_step,  # STEP folder → .npz records
    prepare_fusion360,               # Fusion 360 dir → records + dataset.json
    BRepDataModule,                  # LightningDataModule
    LLBRepNet,                       # the model (a LightningModule)
    evaluate_folder,                 # checkpoint + STEP folder → per-face logits
)

# Inference from an existing checkpoint:
written = evaluate_folder(
    step_dir="./s2.0.0/breps/step",
    model_ckpt="./logs/.../best.ckpt",
    dataset_file="./processed/dataset.json",
    output_dir="./predictions",
    num_workers=8,
)
```

The model also exposes `predict_logits(batch)` → `[(file_stem, per-face softmax)]`
for programmatic, per-solid inference.

## Alternate input: precomputed JSON topology

If you already have the coedge topology + features from another tool, skip the
OpenCascade extraction and feed JSON directly:

```bash
python -m ll_brepnet.pipelines.extract_brepnet_data_from_json \
  --json my_solid.json --output my_solid.npz
```

See `BRepJsonExtractor` for the schema (coedge incidence arrays + per-face/edge
feature matrices; UV-grids optional).
