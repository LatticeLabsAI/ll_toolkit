"""LL-BRepNet: a B-Rep face-graph neural network for CAD solid-model segmentation.

``ll_brepnet`` operates directly on the boundary representation (B-Rep) of a CAD
solid: faces and edges are connected through *coedges* (oriented half-edges that
carry next/previous/mate/parent-face/parent-edge adjacency). A coedge
message-passing encoder fuses topological adjacency with UV-grid surface/curve
geometry to predict a semantic segment label for every face.

The package is organised into four subpackages:

- :mod:`ll_brepnet.pipelines` -- STEP/JSON -> coedge graph + geometry -> ``.npz``
  extraction and dataset-manifest building.
- :mod:`ll_brepnet.dataloaders` -- ``BRepDataset`` / ``BRepDataModule`` plus the
  multi-solid collation and face-count batch sampler.
- :mod:`ll_brepnet.models` -- the UV-Net geometry encoders and the
  ``LLBRepNet`` LightningModule with its per-face segmentation head.
- :mod:`ll_brepnet.eval` -- folder/checkpoint inference producing per-face logits.

This is an independent MIT implementation built on the LatticeLabs toolkit's own
B-Rep machinery. It is *inspired by* BRepNet (Lambourne et al., CVPR 2021,
arXiv:2104.00706) and UV-Net, but contains no code from those projects; see
``ATTRIBUTION.md``.
"""

from __future__ import annotations

__version__ = "0.1.0"

import importlib

# Public API. Resolved lazily (PEP 562) so ``import ll_brepnet`` stays cheap and
# does not eagerly pull in torch / pythonocc / cadling: the heavy module is only
# imported the first time the attribute is accessed.
_LAZY_EXPORTS = {
    "LLBRepNet": "ll_brepnet.models.ll_brepnet",
    "UVNetSurfaceEncoder": "ll_brepnet.models.uvnet_encoders",
    "UVNetCurveEncoder": "ll_brepnet.models.uvnet_encoders",
    "BRepDataset": "ll_brepnet.dataloaders.brep_dataset",
    "BRepDataModule": "ll_brepnet.dataloaders.brep_dataset",
    "BRepBatch": "ll_brepnet.dataloaders.brep_dataset",
    "brep_collate_fn": "ll_brepnet.dataloaders.brep_dataset",
    "MaxNumFacesSampler": "ll_brepnet.dataloaders.max_num_faces_loader",
    "BRepDataExtractor": "ll_brepnet.pipelines.extract_brepnet_data_from_step",
    "extract_brepnet_data_from_step": "ll_brepnet.pipelines.extract_brepnet_data_from_step",
    "extract_step_files": "ll_brepnet.pipelines.extract_brepnet_data_from_step",
    "extract_brepnet_data_from_json": "ll_brepnet.pipelines.extract_brepnet_data_from_json",
    "build_dataset_file": "ll_brepnet.pipelines.build_dataset_file",
    "prepare_fusion360": "ll_brepnet.pipelines.quickstart",
    "evaluate_folder": "ll_brepnet.eval.evaluate",
    "do_training": "ll_brepnet.train",
}

__all__ = ["__version__", *sorted(_LAZY_EXPORTS)]


def __getattr__(name: str):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module 'll_brepnet' has no attribute {name!r}")
    return getattr(importlib.import_module(module_path), name)


def __dir__():
    return sorted([*globals().keys(), *_LAZY_EXPORTS])
