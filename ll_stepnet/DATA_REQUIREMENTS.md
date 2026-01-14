# Training Data Requirements for LL-STEPNet

This document explains what data you need to train each model in the LL-STEPNet package.

## Overview

All models require **STEP files** (.step or .stp format) as input. The difference is in the **labels/annotations** needed for supervised learning.

---

## 1. Classification Model (STEPForClassification)

### What it predicts
Part categories (e.g., bracket, housing, shaft, gear, plate, connector)

### Required Data

**Input:** STEP files
**Labels:** Integer class IDs

### Dataset Structure
```
data/classification/
├── train/
│   ├── bracket_001.step
│   ├── bracket_002.step
│   ├── housing_001.step
│   ├── shaft_001.step
│   └── ...
├── val/
│   ├── bracket_050.step
│   └── ...
└── labels.json
```

### labels.json Format
```json
{
  "bracket_001.step": 0,
  "bracket_002.step": 0,
  "housing_001.step": 1,
  "shaft_001.step": 2,
  "gear_001.step": 3
}
```

### Class ID Mapping Example
```json
{
  "classes": {
    "0": "bracket",
    "1": "housing",
    "2": "shaft",
    "3": "gear",
    "4": "plate",
    "5": "connector",
    "6": "fastener",
    "7": "bearing",
    "8": "cover",
    "9": "frame"
  }
}
```

### Recommended Dataset Size
- **Minimum:** 100-500 files per class
- **Good:** 1,000-5,000 files per class
- **Excellent:** 10,000+ files per class

### How to Get Labels
1. Use existing folder structure (folder name = class)
2. Parse filename patterns (e.g., "bracket_*.step" → class 0)
3. Manual labeling (time-consuming but accurate)
4. Extract from PDM/PLM metadata

---

## 2. Property Prediction Model (STEPForPropertyPrediction)

### What it predicts
Physical properties: volume, surface area, mass, bounding box dimensions

### Required Data

**Input:** STEP files
**Labels:** 6 numeric values per file

### labels.json Format
```json
{
  "part_001.step": [100.5, 250.3, 45.2, 10.0, 15.0, 8.0],
  "part_002.step": [200.1, 400.5, 90.3, 20.0, 25.0, 12.0]
}
```

### Property Order
1. **Volume** (mm³)
2. **Surface area** (mm²)
3. **Mass** (g) - requires material density assumption
4. **Bounding box X** (mm)
5. **Bounding box Y** (mm)
6. **Bounding box Z** (mm)

### Recommended Dataset Size
- **Minimum:** 1,000 files
- **Good:** 10,000+ files
- **Excellent:** 100,000+ files

### How to Compute Properties

#### Option 1: Using pythonOCC (Python)
```python
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add

def compute_properties(step_file):
    # Load STEP file
    reader = STEPControl_Reader()
    reader.ReadFile(step_file)
    reader.TransferRoots()
    shape = reader.Shape()

    # Volume
    vol_props = GProp_GProps()
    brepgprop_VolumeProperties(shape, vol_props)
    volume = vol_props.Mass()

    # Surface area
    surf_props = GProp_GProps()
    brepgprop_SurfaceProperties(shape, surf_props)
    surface_area = surf_props.Mass()

    # Mass (assuming density = 1.0 g/mm³)
    density = 1.0
    mass = volume * density

    # Bounding box
    bbox = Bnd_Box()
    brepbndlib_Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    bbox_x = xmax - xmin
    bbox_y = ymax - ymin
    bbox_z = zmax - zmin

    return [volume, surface_area, mass, bbox_x, bbox_y, bbox_z]
```

#### Option 2: Using FreeCAD (Python)
```python
import FreeCAD
import Part

def compute_properties(step_file):
    doc = FreeCAD.newDocument()
    Part.insert(step_file, doc.Name)
    shape = doc.Objects[0].Shape

    volume = shape.Volume
    surface_area = shape.Area
    mass = volume * 1.0  # Density

    bbox = shape.BoundBox
    bbox_x = bbox.XLength
    bbox_y = bbox.YLength
    bbox_z = bbox.ZLength

    return [volume, surface_area, mass, bbox_x, bbox_y, bbox_z]
```

#### Option 3: Batch Processing Script
```bash
# Process all STEP files in directory
for file in train/*.step; do
    python compute_props.py "$file" >> labels.json
done
```

---

## 3. Similarity Model (STEPForSimilarity)

### What it predicts
Embedding vectors for similarity search (find similar CAD parts)

### Required Data

You have **3 options** for training:

#### Option A: Supervised Pairs (Recommended for labeled data)

```json
{
  "similar_pairs": [
    ["bracket_001.step", "bracket_002.step"],
    ["bracket_001.step", "bracket_003.step"],
    ["shaft_010.step", "shaft_011.step"]
  ],
  "dissimilar_pairs": [
    ["bracket_001.step", "gear_050.step"],
    ["housing_020.step", "fastener_100.step"]
  ]
}
```

**Training method:** Contrastive loss (push similar close, dissimilar apart)

#### Option B: Triplet Format

```json
{
  "triplets": [
    {
      "anchor": "part_001.step",
      "positive": "part_001_variant.step",
      "negative": "different_part.step"
    }
  ]
}
```

**Training method:** Triplet loss

#### Option C: Self-Supervised (NO LABELS NEEDED!)

No labels required! Use contrastive learning:
- Same file = positive pair
- Different files = negative pair
- Works well with data augmentation

**Recommended:** Start with Option C, then fine-tune with Option A if you have labels

### Recommended Dataset Size
- **Minimum:** 10,000 pairs
- **Good:** 100,000+ pairs
- **Self-supervised:** As many STEP files as possible

---

## 4. Captioning Model (STEPForCaptioning)

### What it predicts
Natural language descriptions of CAD parts

### Required Data

**Input:** STEP files
**Labels:** Text captions

### captions.json Format
```json
{
  "bracket_001.step": "L-shaped mounting bracket with 4 bolt holes and chamfered edges",
  "housing_042.step": "Cylindrical housing with threaded interior and mounting flange",
  "shaft_015.step": "Steel shaft with keyway and shoulder for bearing mounting",
  "gear_033.step": "Spur gear with 24 teeth and central bore"
}
```

### Caption Guidelines
- **Length:** 10-30 words
- **Content:** Describe shape, features, purpose
- **Include:** Material (if known), dimensions (if important), distinctive features
- **Avoid:** Overly technical jargon, file-specific info

### Caption Examples by Part Type

**Brackets:**
- "L-shaped bracket with 4 mounting holes"
- "Corner bracket with reinforcement ribs"
- "Adjustable mounting bracket with slotted holes"

**Housings:**
- "Rectangular housing with removable cover"
- "Cylindrical motor housing with cooling fins"
- "Split housing with bolt-on lid"

**Shafts:**
- "Stepped shaft with keyway"
- "Threaded shaft with hex head"
- "Hollow shaft with internal splines"

**Complex Parts:**
- "Gearbox housing with integrated mounting lugs and oil drain port"
- "Multi-tier bearing assembly with press-fit sleeves"

### Recommended Dataset Size
- **Minimum:** 5,000 files
- **Good:** 50,000+ files
- **Excellent:** 500,000+ files

### How to Generate Captions

1. **Manual annotation** (most accurate, most expensive)
   - Engineering interns/students
   - Crowdsourcing (Amazon Mechanical Turk)

2. **Extract from CAD metadata**
   - Part description fields
   - PDM/PLM systems
   - Title blocks in drawings

3. **LLM-assisted generation**
   - Use GPT-4/Claude with screenshots
   - Review and correct manually
   - Faster than fully manual

4. **Template-based**
   - Detect features automatically
   - Fill templates: "{shape} {part_type} with {features}"

---

## 5. Question Answering Model (STEPForQA)

### What it predicts
Answers to questions about CAD parts

### Required Data

**Input:** STEP files
**Labels:** Question-answer pairs

### qa_data.json Format
```json
{
  "bracket_001.step": [
    {
      "question": "How many holes does this part have?",
      "answer": "4"
    },
    {
      "question": "What is the material thickness?",
      "answer": "5mm"
    },
    {
      "question": "What type of part is this?",
      "answer": "mounting bracket"
    },
    {
      "question": "Does this part have any threads?",
      "answer": "no"
    }
  ]
}
```

### Question Types

**Counting questions:**
- "How many holes are there?"
- "How many faces does this part have?"
- "How many chamfers are present?"

**Measurement questions:**
- "What is the diameter of the main bore?"
- "What is the overall length?"
- "What is the wall thickness?"

**Classification questions:**
- "What type of part is this?"
- "What is the primary function?"
- "What material is this likely made from?"

**Feature detection questions:**
- "Does this have threads?"
- "Are there any fillets?"
- "Is this part symmetric?"

**Comparative questions:**
- "Is this larger than 100mm?"
- "Is the bore diameter greater than 10mm?"

### Recommended Dataset Size
- **Minimum:** 10,000 QA pairs across 2,000+ files
- **Good:** 100,000+ QA pairs
- **Excellent:** 1,000,000+ QA pairs

### Answer Format
- Keep answers **short** (1-10 words)
- For yes/no: use "yes" or "no"
- For numbers: include units "5mm", "10 holes"
- For categories: lowercase "bracket", "shaft"

---

## Data Sources

### Where to Get STEP Files

#### Public Sources
1. **GrabCAD** - https://grabcad.com/library
   - 5+ million CAD models
   - Community uploads
   - Mixed quality

2. **TraceParts** - https://www.traceparts.com/
   - Manufacturer CAD catalogs
   - High quality, standardized parts

3. **McMaster-Carr** - https://www.mcmaster.com/
   - CAD downloads for all products
   - High quality industrial parts

4. **Thingiverse** - https://www.thingiverse.com/
   - Mostly STL, but some STEP
   - Hobbyist/maker parts

5. **ABC Dataset** (Research)
   - 1 million CAD models
   - For academic use

#### Industrial Sources
- Your company's CAD library
- PDM/PLM system exports
- Supplier CAD models
- Engineering archives

#### Generate Synthetic Data
```python
# Example: Generate parametric brackets with FreeCAD
import FreeCAD
import Part

for width in range(20, 200, 10):
    for height in range(20, 200, 10):
        # Create bracket geometry
        doc = FreeCAD.newDocument()
        # ... parametric modeling ...
        Part.export(doc.Objects, f"bracket_{width}x{height}.step")
```

---

## Quick Start: Creating Your First Dataset

### 1. Collect 100-500 STEP files

### 2. Organize into folders
```bash
mkdir -p data/my_dataset/{train,val}
# Copy 80% to train/, 20% to val/
```

### 3. Create labels

For classification:
```python
import json
from pathlib import Path

labels = {}
for file in Path('data/my_dataset/train').glob('*.step'):
    # Manual classification or use filename patterns
    if 'bracket' in file.name:
        labels[file.name] = 0
    elif 'housing' in file.name:
        labels[file.name] = 1
    # ... etc

with open('data/my_dataset/labels.json', 'w') as f:
    json.dump(labels, f, indent=2)
```

### 4. Train!
```bash
python examples/train_classification.py
```

---

## Data Quality Tips

1. **Balance classes** - Similar number of examples per class
2. **Diverse geometries** - Simple to complex parts
3. **Realistic parts** - Use real engineering parts, not synthetic toys
4. **Clean STEP files** - Validate files open correctly
5. **Consistent units** - All files in same unit system (mm recommended)
6. **Remove duplicates** - Check for identical or near-identical files
7. **Split properly** - Don't leak validation data (same part in train/val)

---

## Minimal Dataset to Get Started

Want to try it out quickly? Here's the absolute minimum:

- **Files:** 50-100 STEP files
- **Classes:** 3-5 categories
- **Split:** 80% train, 20% validation
- **Labels:** Simple integer IDs

This won't give great performance, but will let you:
- Test the training pipeline
- Verify data format is correct
- Get a baseline model
- Identify issues early

Then gradually scale up to thousands of files for production use.

---

## Need Help?

- See `examples/` directory for training scripts
- Check `test_results/` for examples of model outputs
- Review `README.md` for usage examples
