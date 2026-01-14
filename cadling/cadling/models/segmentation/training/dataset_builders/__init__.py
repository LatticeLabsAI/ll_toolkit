"""Dataset builders for creating HuggingFace datasets from local CAD data.

Provides builders for converting:
- MFCAD++ → HuggingFace dataset
- MFInstSeg → HuggingFace dataset
- HybridCAD → HuggingFace dataset
- Custom STEP datasets → HuggingFace dataset
"""

from .mfcad_builder import MFCADDatasetBuilder
from .create_hf_dataset import create_hf_dataset_from_local

__all__ = [
    "MFCADDatasetBuilder",
    "create_hf_dataset_from_local",
]
