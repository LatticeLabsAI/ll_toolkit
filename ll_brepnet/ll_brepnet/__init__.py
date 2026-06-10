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

# Public symbols are re-exported here as each subpackage is implemented so that
# ``from ll_brepnet import <X>`` mirrors the documented API. They are imported
# lazily inside the subpackages (which pull in torch / pythonocc) to keep
# ``import ll_brepnet`` cheap and dependency-light.
__all__ = ["__version__"]
