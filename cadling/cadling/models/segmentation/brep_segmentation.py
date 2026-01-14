"""B-Rep face segmentation model using hybrid GAT+Transformer.

This module provides ML models for semantic segmentation of B-Rep faces in STEP files.
Uses hybrid Graph Attention Network + Transformer architecture (BRepGAT/BRepFormer-inspired).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

from cadling.models.base_model import EnrichmentModel

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADItem, CADlingDocument
    from cadling.datamodel.step import STEPEntityItem

_log = logging.getLogger(__name__)


class BRepSegmentationModel(EnrichmentModel):
    """B-Rep face segmentation using hybrid GAT+Transformer.

    Segments B-Rep faces into manufacturing feature classes using a two-stage architecture:
    - Stage 1: Graph Attention Network (3 layers, 8 heads) for local face relationships
    - Stage 2: Transformer Encoder (4 layers) for global part context
    - Stage 3: Per-face classification head

    Architecture follows BRepGAT and BRepFormer papers, achieving 98%+ accuracy.

    Node features (24 dims): surface type (one-hot), area, curvature, normal, centroid
    Edge features (8 dims): edge type (concave/convex), dihedral angle, edge length

    Manufacturing Feature Classes (25 total):
        Base: base, stock
        Additive: boss, rib, protrusion, circular_boss, rectangular_boss, hex_boss
        Subtractive: pocket, hole, slot, chamfer, fillet, groove, through_hole, blind_hole,
                    countersink, counterbore, round_pocket, rectangular_pocket
        Advanced: thread, keyway, dovetail, t_slot, o_ring_groove

    Example:
        model = BRepSegmentationModel(
            artifacts_path=Path("models/brep_seg.pt"),
            num_classes=25
        )
        result = converter.convert(
            "part.step",
            pipeline_options=PipelineOptions(enrichment_models=[model])
        )
        for item in result.document.items:
            if "brep_segments" in item.properties:
                print(f"Found {item.properties['brep_segments']['num_segments']} face segments")

    Attributes:
        model: HybridGATTransformer model instance
        feature_classes: List of manufacturing feature class names
        artifacts_path: Path to model checkpoint
    """

    def __init__(
        self,
        artifacts_path: Optional[Path] = None,
        num_classes: int = 25,
        gat_hidden_dim: int = 256,
        gat_num_heads: int = 8,
        gat_num_layers: int = 3,
        transformer_hidden_dim: int = 512,
        transformer_num_layers: int = 4,
    ):
        """Initialize B-Rep segmentation model.

        Args:
            artifacts_path: Path to model checkpoint (.pt file)
            num_classes: Number of manufacturing feature classes (default 25)
            gat_hidden_dim: Hidden dimension for GAT layers
            gat_num_heads: Number of attention heads in GAT
            gat_num_layers: Number of GAT layers
            transformer_hidden_dim: Hidden dimension for Transformer
            transformer_num_layers: Number of Transformer layers
        """
        super().__init__()

        self.artifacts_path = artifacts_path
        self.num_classes = num_classes

        # Manufacturing feature classes (25 total)
        self.feature_classes = [
            # Base features
            "base", "stock",
            # Additive features
            "boss", "rib", "protrusion", "circular_boss", "rectangular_boss", "hex_boss",
            # Subtractive features
            "pocket", "hole", "slot", "chamfer", "fillet", "groove",
            "through_hole", "blind_hole", "countersink", "counterbore",
            "round_pocket", "rectangular_pocket",
            # Advanced features
            "thread", "keyway", "dovetail", "t_slot", "o_ring_groove",
        ]

        # Initialize graph builder
        try:
            from cadling.models.segmentation.brep_graph_builder import BRepFaceGraphBuilder

            self.graph_builder = BRepFaceGraphBuilder()
        except Exception as e:
            _log.error(f"Failed to initialize graph builder: {e}")
            self.graph_builder = None

        # Load model if artifacts_path provided
        if artifacts_path and artifacts_path.exists():
            try:
                from cadling.models.segmentation.architectures.gat_net import (
                    HybridGATTransformer,
                )

                # Create model instance
                self.model = HybridGATTransformer(
                    in_dim=24,  # Node feature dimension
                    gat_hidden_dim=gat_hidden_dim,
                    gat_num_heads=gat_num_heads,
                    gat_num_layers=gat_num_layers,
                    transformer_hidden_dim=transformer_hidden_dim,
                    transformer_num_layers=transformer_num_layers,
                    transformer_num_heads=8,
                    num_classes=num_classes,
                    dropout=0.1,
                    edge_dim=8,
                )

                # Load checkpoint
                checkpoint = torch.load(
                    str(artifacts_path), map_location="cpu", weights_only=False
                )

                if "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)

                self.model.eval()

                _log.info(f"Loaded B-Rep segmentation model from {artifacts_path}")
            except Exception as e:
                _log.error(f"Failed to load B-Rep segmentation model: {e}")
                self.model = None
        else:
            self.model = None
            if artifacts_path:
                _log.warning(f"Model path not found: {artifacts_path}")
            else:
                _log.info("B-Rep segmentation model path not provided")

    def __call__(
        self,
        doc: CADlingDocument,
        item_batch: list[CADItem],
    ) -> None:
        """Segment B-Rep faces.

        Args:
            doc: CADlingDocument being enriched
            item_batch: List of CADItem objects to segment
        """
        from cadling.datamodel.step import STEPEntityItem

        if not self.model:
            _log.debug("B-Rep segmentation skipped: model not available")
            return

        if not self.graph_builder:
            _log.debug("B-Rep segmentation skipped: graph builder not available")
            return

        for item in item_batch:
            # Only segment STEP entity items
            if not isinstance(item, STEPEntityItem):
                continue

            try:
                # Build face adjacency graph
                face_graph = self.graph_builder.build_face_graph(doc, item)

                if face_graph.num_nodes == 0:
                    _log.debug(f"No faces found in item {item.label.text if hasattr(item, 'label') else 'unknown'}")
                    continue

                # Run hybrid GAT+Transformer architecture
                with torch.no_grad():
                    logits = self.model(
                        x=face_graph.x,
                        edge_index=face_graph.edge_index,
                        edge_attr=face_graph.edge_attr,
                        batch=None,  # Single graph
                    )

                    # Get predictions
                    probabilities = torch.softmax(logits, dim=1)
                    face_labels = torch.argmax(probabilities, dim=1)
                    confidence = probabilities.max(dim=1)[0]

                # Store results in item.properties
                import numpy as np
                unique_labels = np.unique(face_labels.cpu().numpy())

                item.properties["brep_segments"] = {
                    "face_labels": face_labels.cpu().tolist(),
                    "label_names": self.feature_classes,
                    "num_segments": len(unique_labels),
                    "confidence": confidence.cpu().tolist(),
                    "face_ids": face_graph.face_ids if hasattr(face_graph, "face_ids") else None,
                }

                # Add provenance
                item.add_provenance(
                    component_type="enrichment_model",
                    component_name="BRepSegmentationModel",
                )

                _log.debug(
                    f"Segmented B-Rep with {face_graph.num_nodes} faces into {len(unique_labels)} feature classes"
                )

            except Exception as e:
                _log.error(f"B-Rep segmentation failed for item: {e}")

    def supports_batch_processing(self) -> bool:
        """B-Rep segmentation supports batch processing."""
        return True

    def get_batch_size(self) -> int:
        """Recommended batch size for B-Rep segmentation."""
        return 16

    def requires_gpu(self) -> bool:
        """B-Rep segmentation benefits from GPU."""
        return True
