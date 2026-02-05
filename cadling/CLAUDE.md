# CADling Project Guidelines

## Architecture Decisions

### PyTorch Installation: Conda-Only (Not PyPI)

**Decision**: All PyTorch-related packages (pytorch, torchvision, torchaudio, pytorch-geometric) MUST be installed via conda-forge, NOT via pip/PyPI.

**Why This Matters**:

On macOS, mixing PyPI and conda installations of PyTorch causes a fatal OpenMP library conflict:
```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
Fatal Python error: Aborted
```

**Root Cause**:
- **PyPI PyTorch** bundles its own `libomp.dylib` in `site-packages/torch/lib/`
- **Conda packages** (scipy, numpy, etc.) use the system-wide `llvm-openmp` from conda-forge
- When both are loaded, Python crashes because two different OpenMP runtimes are initialized

**The Problem We Had**:
```
Environment before fix:
├── torch 2.9.1 (PyPI) ← bundled libomp.dylib (737KB)
├── torchvision 0.24.1 (PyPI)
├── libtorch 2.8.0 (conda) ← uses conda's libomp.dylib (824KB)
├── torchaudio 2.8.0 (conda)
└── scipy 1.16.3 (conda) ← uses conda's libomp.dylib

Result: Two different OpenMP libraries loaded → CRASH
```

**The Solution**:
Use conda-forge for ALL PyTorch packages:
```
Environment after fix:
├── pytorch 2.x (conda-forge) ← uses conda's libomp.dylib
├── torchvision (conda-forge)
├── torchaudio (conda-forge)
├── pytorch-geometric (conda-forge)
├── scipy (conda-forge)
└── llvm-openmp (conda-forge) ← single OpenMP runtime

Result: One unified OpenMP library → NO CONFLICTS ✓
```

**Benefits**:
- ✅ No OpenMP conflicts on macOS
- ✅ Better integration with conda scientific stack (scipy, numpy)
- ✅ Consistent library versions across all dependencies
- ✅ Easier reproducibility across different systems
- ✅ Conda manages complex binary dependencies automatically

**Tradeoffs**:
- ⚠️ Conda-forge PyTorch may lag slightly behind PyPI releases (typically days, not weeks)
- ⚠️ Must use `conda install` instead of `pip install` for PyTorch

**Configuration Files**:
- `environment.yml`: Contains ALL PyTorch packages (authoritative source)
- `requirements.txt`: Excludes PyTorch (has comment explaining why)
- `pyproject.toml`: Excludes PyTorch from `ml` dependencies (has comment explaining why)

**Installation Instructions**:

1. **Create environment from scratch**:
   ```bash
   conda env create -f environment.yml
   conda activate cadling
   ```

2. **Update existing environment**:
   ```bash
   # Remove PyPI torch first
   pip uninstall torch torchvision torchaudio -y

   # Install via conda
   conda env update -f environment.yml --prune
   ```

3. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'Location: {torch.__file__}')"
   # Should show: .../envs/cadling/lib/python3.XX/site-packages/torch/...
   # NOT: .../site-packages/torch/... (PyPI location)
   ```

**Never Do This**:
```bash
❌ pip install torch  # Will cause OpenMP conflicts!
❌ pip install torch torchvision  # Will cause OpenMP conflicts!
❌ pip install -r requirements.txt  # Safe - torch excluded
```

**Always Do This**:
```bash
✅ conda install pytorch torchvision torchaudio pytorch-geometric -c conda-forge
✅ conda env update -f environment.yml
```

---

## Development Guidelines

### General Principles

- **If something is called but missing, it should be implemented, not removed** - Unused imports, variables, or methods are intentionally included and should be used appropriately as they are critical to operations.

- **All commands need to be provided** - This is not a production server; all setup and configuration commands must be explicitly documented and runnable.

### Testing

- All tests must pass on macOS (Apple Silicon and Intel)
- Use pytest with proper mocking to avoid requiring external services
- Timeout critical tests to prevent hanging (use `@pytest.mark.timeout(5)`)

### Dependencies

- **CAD Processing**: `pythonocc-core` can only be installed via conda (not available on PyPI)
- **ML/PyTorch**: Must use conda-forge (see above)
- **Everything Else**: Prefer pip for pure Python packages, use conda for system-level dependencies
