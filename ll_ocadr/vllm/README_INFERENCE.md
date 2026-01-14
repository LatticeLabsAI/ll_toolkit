# LL-OCADR Inference Guide

This guide explains how to run inference with the LL-OCADR model on CAD and mesh files.

## Quick Start

### Single File Inference

Process a single STEP, STL, OBJ, or PLY file:

```bash
python run_ll_ocadr.py \
    --mesh-file example.step \
    --prompt "<mesh>\nDescribe this CAD model."
```

### Batch Evaluation

Process multiple files in a directory:

```bash
python run_ll_ocadr_eval_batch.py \
    --data-dir /path/to/cad/files \
    --output-file results.jsonl
```

## Installation

### Prerequisites

```bash
# Install core dependencies
pip install torch transformers

# Install 3D geometry processing libraries
pip install trimesh open3d numpy scipy

# Install pythonocc-core for STEP file support (recommended via conda)
conda install -c conda-forge pythonocc-core

# Optional: Install vLLM for faster inference
pip install vllm
```

### Using the Environment

```bash
# Create conda environment
conda env create -f environment.yml
conda activate ll-ocadr

# Or install with pip
pip install -r requirements.txt
```

## Usage

### Single File Runner (`run_ll_ocadr.py`)

#### Basic Usage

```bash
python run_ll_ocadr.py --mesh-file model.step
```

#### With Custom Prompt

```bash
python run_ll_ocadr.py \
    --mesh-file bracket.stl \
    --prompt "<mesh>\nWhat is this part? List dimensions and features."
```

#### From Prompt File

```bash
python run_ll_ocadr.py \
    --mesh-file assembly.obj \
    --prompt-file my_prompt.txt
```

#### Save Output

```bash
python run_ll_ocadr.py \
    --mesh-file part.step \
    --output-file description.txt
```

#### Model Selection

```bash
# Use different model size
python run_ll_ocadr.py \
    --mesh-file model.stl \
    --model-size 1.8b  # Options: 1.8b, 7b, 14b

# Use custom checkpoint
python run_ll_ocadr.py \
    --mesh-file model.step \
    --model-path /path/to/checkpoint.pt
```

#### Generation Parameters

```bash
python run_ll_ocadr.py \
    --mesh-file model.obj \
    --temperature 0.7 \
    --max-tokens 512 \
    --top-p 0.9
```

#### Device Selection

```bash
# Use CUDA GPU (default if available)
python run_ll_ocadr.py --mesh-file model.step --device cuda

# Use Apple Silicon MPS
python run_ll_ocadr.py --mesh-file model.step --device mps

# Use CPU
python run_ll_ocadr.py --mesh-file model.step --device cpu
```

#### Disable vLLM

```bash
# Use native PyTorch instead of vLLM
python run_ll_ocadr.py --mesh-file model.step --no-vllm
```

### Batch Evaluation Runner (`run_ll_ocadr_eval_batch.py`)

#### Basic Usage

```bash
python run_ll_ocadr_eval_batch.py \
    --data-dir /path/to/meshes \
    --output-file results.jsonl
```

#### Filter by Extension

```bash
# Only process STEP files
python run_ll_ocadr_eval_batch.py \
    --data-dir /path/to/meshes \
    --extensions .step,.stp \
    --output-file step_results.jsonl
```

#### Limit Number of Files

```bash
# Process first 10 files only
python run_ll_ocadr_eval_batch.py \
    --data-dir /path/to/meshes \
    --max-files 10 \
    --output-file test_results.jsonl
```

#### Custom Prompts Per File

Create a JSON file mapping filenames to prompts:

```json
{
  "bracket.step": "Describe this bracket and its mounting holes.",
  "gear.stl": "Analyze this gear. Count teeth and estimate module.",
  "housing.obj": "What is the purpose of this housing?"
}
```

Then run:

```bash
python run_ll_ocadr_eval_batch.py \
    --data-dir /path/to/meshes \
    --prompt-file prompts.json \
    --output-file results.jsonl
```

#### With Reference Texts

For evaluation with ground truth, create a references JSON:

```json
{
  "bracket.step": "L-shaped mounting bracket with two M6 holes...",
  "gear.stl": "Spur gear with 24 teeth, module 2.5mm..."
}
```

Then run:

```bash
python run_ll_ocadr_eval_batch.py \
    --data-dir /path/to/meshes \
    --reference-file references.json \
    --output-file results.jsonl
```

#### Example Output

The batch runner saves results as JSONL (one JSON per line):

```json
{"mesh_file": "/path/bracket.step", "prompt": "<mesh>\nDescribe...", "generated_text": "This is a mounting bracket...", "processing_time": 2.34, "num_tokens": 87, "error": null}
{"mesh_file": "/path/gear.stl", "prompt": "<mesh>\nDescribe...", "generated_text": "This is a spur gear...", "processing_time": 1.98, "num_tokens": 72, "error": null}
```

And prints a summary:

```
===========================================================
EVALUATION SUMMARY
===========================================================
Total files:        100
Successful:         98
Failed:             2
Total time:         245.67s
Avg time per file:  2.46s
Total tokens:       8432
Avg tokens per file:86.0
===========================================================
```

## Supported File Formats

- **STEP** (.step, .stp): CAD files with B-Rep geometry
- **STL** (.stl): Triangle mesh files (ASCII or binary)
- **OBJ** (.obj): Wavefront OBJ mesh files
- **PLY** (.ply): Polygon mesh files

## Prompt Format

Prompts should include the `<mesh>` placeholder where the mesh embeddings will be inserted:

```
<mesh>\nDescribe this CAD model and list its key features.
```

If you omit `<mesh>`, it will be automatically prepended to your prompt.

## Model Sizes

Three model sizes are available:

- **1.8B**: Fastest, lowest memory (8GB VRAM)
- **7B**: Balanced performance (24GB VRAM) - **recommended**
- **14B**: Best quality, highest memory (48GB VRAM)

## Performance Tips

### Speed

1. **Use vLLM**: 2-3x faster than native PyTorch
2. **Use GPU**: CUDA >> MPS > CPU
3. **Batch Processing**: Process multiple files together
4. **Smaller Models**: 1.8B is much faster than 14B

### Memory

1. **Reduce max_tokens**: Lower maximum generation length
2. **Use FP16**: Automatically enabled on CUDA
3. **Smaller Model**: Use 1.8B instead of 7B/14B
4. **Disable Chunking**: For simple meshes, can reduce memory

### Quality

1. **Higher Temperature**: More creative (0.7-0.9)
2. **Lower Temperature**: More factual (0.1-0.3)
3. **Larger Model**: 14B > 7B > 1.8B
4. **Better Prompts**: Be specific about what you want

## Troubleshooting

### CUDA Out of Memory

```bash
# Use smaller model
python run_ll_ocadr.py --mesh-file large.step --model-size 1.8b

# Reduce max tokens
python run_ll_ocadr.py --mesh-file large.step --max-tokens 256

# Use CPU (slow but no memory limit)
python run_ll_ocadr.py --mesh-file large.step --device cpu
```

### File Not Found Error

```bash
# Check file exists
ls -lh model.step

# Use absolute path
python run_ll_ocadr.py --mesh-file /full/path/to/model.step
```

### STEP Loading Error

```bash
# Ensure pythonocc-core is installed
conda install -c conda-forge pythonocc-core

# Or convert STEP to STL first
# (use FreeCAD, Blender, or online converter)
```

### vLLM Import Error

```bash
# Install vLLM
pip install vllm

# Or disable vLLM
python run_ll_ocadr.py --mesh-file model.step --no-vllm
```

## Examples

### Describe a Bracket

```bash
python run_ll_ocadr.py \
    --mesh-file bracket.step \
    --prompt "<mesh>\nDescribe this bracket. What is it used for?"
```

### Analyze Gear Geometry

```bash
python run_ll_ocadr.py \
    --mesh-file gear.stl \
    --prompt "<mesh>\nAnalyze this gear. Count the teeth and describe the profile." \
    --temperature 0.3
```

### Generate Assembly Instructions

```bash
python run_ll_ocadr.py \
    --mesh-file assembly.obj \
    --prompt "<mesh>\nGenerate assembly instructions for this part." \
    --max-tokens 1024
```

### Batch Process Dataset

```bash
python run_ll_ocadr_eval_batch.py \
    --data-dir ./cad_dataset \
    --default-prompt "<mesh>\nProvide a detailed technical description." \
    --output-file dataset_descriptions.jsonl \
    --max-files 1000
```

## API Usage

You can also use LL-OCADR programmatically:

```python
from run_ll_ocadr import LLOCADRInference

# Initialize
inference = LLOCADRInference(
    model_path="latticelabs/ll-ocadr-7b",
    model_size="7b",
    device="cuda"
)

# Generate (synchronous)
result = inference.generate(
    mesh_file="model.step",
    prompt="<mesh>\nDescribe this model.",
    temperature=0.7,
    max_tokens=512
)

print(result)

# Generate (asynchronous with vLLM)
import asyncio

async def main():
    result = await inference.generate_async(
        mesh_file="model.step",
        prompt="<mesh>\nDescribe this model.",
        temperature=0.7,
        max_tokens=512
    )
    print(result)

asyncio.run(main())
```

## Citation

If you use LL-OCADR in your research, please cite:

```bibtex
@software{ll_ocadr_2024,
  title={LL-OCADR: Lattice Labs Optical CAD Recognition},
  author={Lattice Labs},
  year={2024},
  url={https://github.com/latticelabs/ll-ocadr}
}
```

## License

See LICENSE file for details.

## Support

For issues and questions:
- GitHub Issues: https://github.com/latticelabs/ll-ocadr/issues
- Documentation: https://docs.latticelabs.com/ll-ocadr
