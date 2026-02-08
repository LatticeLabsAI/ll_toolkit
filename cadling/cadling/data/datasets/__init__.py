"""Individual dataset loader implementations for CAD research datasets.

Provides standardized access to major CAD generation research datasets:
- DeepCAD (178K models): Sketch-and-extrude command sequences
- ABC (1M STEP files): Face adjacency graphs from B-Rep models
- Text2CAD (660K annotations): Multi-level text annotations paired with CAD
- SketchGraphs (15M sketches): Constraint graphs for parametric sketches
"""
from __future__ import annotations

from .base_loader import BaseCADDataset
from .deepcad_loader import DeepCADLoader
from .abc_loader import ABCLoader
from .text2cad_loader import Text2CADLoader
from .sketchgraphs_loader import SketchGraphsLoader

__all__ = [
    "ABCLoader",
    "BaseCADDataset",
    "DeepCADLoader",
    "SketchGraphsLoader",
    "Text2CADLoader",
]
