"""Research dataset loaders for CAD generation training.

Provides HuggingFace-compatible dataset loaders for the major
research datasets used in neural CAD generation:

- **DeepCAD** — 178K sketch-and-extrude command sequences
- **ABC** — 1M STEP B-rep models
- **Text2CAD** — 660K text-annotated CAD models
- **SketchGraphs** — 15M geometric constraint graphs

All loaders support both local files and HuggingFace Hub streaming,
and produce tokenized outputs compatible with geotoken's
``CommandSequenceTokenizer``.
"""
from ll_gen.datasets.deepcad_loader import DeepCADDataset, load_deepcad
from ll_gen.datasets.abc_loader import ABCDataset, load_abc
from ll_gen.datasets.text2cad_loader import Text2CADDataset, load_text2cad
from ll_gen.datasets.sketchgraphs_loader import SketchGraphsDataset, load_sketchgraphs

__all__ = [
    "DeepCADDataset",
    "load_deepcad",
    "ABCDataset",
    "load_abc",
    "Text2CADDataset",
    "load_text2cad",
    "SketchGraphsDataset",
    "load_sketchgraphs",
]
