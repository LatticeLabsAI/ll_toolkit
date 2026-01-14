"""Mesh segmentation model using graph neural networks.

This module provides ML models for semantic segmentation of triangle meshes (STL, OBJ, PLY).
Uses EdgeConv-based graph neural networks to classify mesh vertices/faces into semantic regions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import trimesh
from cadling.models.base_model import EnrichmentModel
from cadling.models.segmentation.architectures.edge_conv_net import MeshSegmentationGNN
from cadling.models.segmentation.graph_utils import mesh_to_pyg_graph

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADItem, CADlingDocument
    from cadling.datamodel.stl import MeshItem

_log = logging.getLogger(__name__)


class MeshSegmentationModel(EnrichmentModel):
    """Mesh segmentation using EdgeConv graph neural network.

    Segments triangle meshes into semantic regions using graph neural networks.
    Operates on face-level or vertex-level representations.

    Architecture:
    - Input: Trimesh mesh (vertices, faces, normals)
    - Graph: Face adjacency graph with geometric features
    - GNN: 5-layer EdgeConv with skip connections
    - Optional: Integration with GeometryNet (local) + ShapeNet (global) from ll_ocadr
    - Output: Per-vertex or per-face segment labels

    Semantic classes (12 default):
        base, boss, pocket, hole, fillet, chamfer, slot, rib, groove, step, thread, unknown

    Example:
        model = MeshSegmentationModel(
            artifacts_path=Path("models/mesh_seg.pt"),
            use_pretrained_encoders=True
        )
        result = converter.convert(
            "part.stl",
            pipeline_options=PipelineOptions(enrichment_models=[model])
        )
        for item in result.document.items:
            if "segments" in item.properties:
                print(f"Found {item.properties['segments']['num_segments']} segments")

    Attributes:
        model: EdgeConvNet model instance
        label_names: List of segment class names
        artifacts_path: Path to model checkpoint
    """

    def __init__(
        self,
        artifacts_path: Optional[Path] = None,
        num_classes: int = 12,
        use_pretrained_encoders: bool = False,
        hidden_dims: list[int] = None,
        chunk_large_meshes: bool = True,
        max_faces_per_chunk: int = 50000,
        use_face_graph: bool = True,
    ):
        """Initialize mesh segmentation model.

        Args:
            artifacts_path: Path to model checkpoint (.pt file)
            num_classes: Number of segmentation classes
            use_pretrained_encoders: Use ShapeNet/GeometryNet from ll_ocadr
            hidden_dims: Hidden dimensions for EdgeConv blocks
            chunk_large_meshes: Whether to chunk meshes exceeding max_faces_per_chunk
            max_faces_per_chunk: Maximum faces per chunk
            use_face_graph: If True, segments faces. If False, segments vertices.
        """
        super().__init__()

        self.artifacts_path = artifacts_path
        self.num_classes = num_classes
        self.use_pretrained_encoders = use_pretrained_encoders
        self.chunk_large_meshes = chunk_large_meshes
        self.max_faces_per_chunk = max_faces_per_chunk
        self.use_face_graph = use_face_graph

        # Default segment classes
        self.label_names = [
            "base",
            "boss",
            "pocket",
            "hole",
            "fillet",
            "chamfer",
            "slot",
            "rib",
            "groove",
            "step",
            "thread",
            "unknown",
        ]

        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512, 512]

        # Load model if artifacts_path provided
        if artifacts_path and artifacts_path.exists():
            try:
                # Create model instance
                self.model = MeshSegmentationGNN(
                    in_channels=7,  # [centroid/position (3) + normal (3) + area/curvature (1)]
                    num_classes=num_classes,
                    hidden_dims=hidden_dims,
                    use_pretrained_encoders=use_pretrained_encoders,
                    dropout=0.3,
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

                _log.info(f"Loaded mesh segmentation model from {artifacts_path}")
            except Exception as e:
                _log.error(f"Failed to load mesh segmentation model: {e}")
                self.model = None
        else:
            self.model = None
            if artifacts_path:
                _log.warning(f"Model path not found: {artifacts_path}")
            else:
                _log.info("Mesh segmentation model path not provided")

    def __call__(
        self,
        doc: CADlingDocument,
        item_batch: list[CADItem],
    ) -> None:
        """Segment mesh items.

        Args:
            doc: CADlingDocument being enriched
            item_batch: List of CADItem objects to segment
        """
        from cadling.datamodel.stl import MeshItem

        if not self.model:
            _log.debug("Mesh segmentation skipped: model not available")
            return

        for item in item_batch:
            # Only segment MeshItem objects
            if not isinstance(item, MeshItem):
                continue

            try:
                # Load mesh from item
                mesh = self._load_mesh_from_item(item)

                if mesh is None:
                    _log.warning(f"Could not load mesh from item {item.label.text}")
                    continue

                # Check if mesh is too large and needs chunking
                num_faces = len(mesh.faces)

                if self.chunk_large_meshes and num_faces > self.max_faces_per_chunk:
                    _log.info(f"Chunking large mesh with {num_faces} faces")
                    labels, confidence = self._segment_large_mesh(mesh)
                else:
                    labels, confidence = self._segment_mesh(mesh)

                # Store results in item.properties
                unique_labels = np.unique(labels)

                item.properties["segments"] = {
                    "vertex_labels" if not self.use_face_graph else "face_labels": labels.tolist(),
                    "label_names": self.label_names,
                    "num_segments": len(unique_labels),
                    "confidence": confidence.tolist() if confidence is not None else None,
                    "segmentation_type": "face" if self.use_face_graph else "vertex",
                }

                # Add provenance
                item.add_provenance(
                    component_type="enrichment_model",
                    component_name="MeshSegmentationModel",
                )

                _log.debug(
                    f"Segmented mesh {item.label.text} into {len(unique_labels)} regions"
                )

            except Exception as e:
                _log.error(f"Mesh segmentation failed for item {item.label.text}: {e}")

    def _load_mesh_from_item(self, item: "MeshItem") -> Optional[trimesh.Trimesh]:
        """Load trimesh mesh from MeshItem.

        Args:
            item: MeshItem to load mesh from

        Returns:
            Trimesh mesh or None if loading fails
        """
        try:
            # Extract vertices and faces from MeshItem
            vertices = np.array(item.vertices)  # [N, 3]
            facets = np.array(item.facets)  # [F, 3]

            # Create trimesh object
            mesh = trimesh.Trimesh(vertices=vertices, faces=facets, process=True)

            return mesh

        except Exception as e:
            _log.error(f"Failed to load mesh from item: {e}")
            return None

    def _segment_mesh(
        self, mesh: trimesh.Trimesh
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Segment a single mesh.

        Args:
            mesh: Trimesh mesh to segment

        Returns:
            labels: [N] segment labels per node
            confidence: [N] confidence scores per node (or None)
        """
        # Convert mesh to graph
        graph = mesh_to_pyg_graph(mesh, use_face_graph=self.use_face_graph)

        # Run model inference
        with torch.no_grad():
            # Extract coordinates and normals for pretrained encoders
            if self.use_face_graph:
                # For face graph, use face centroids and normals
                # Copy arrays to ensure they are writable for PyTorch
                coords = torch.from_numpy(
                    np.mean(mesh.vertices[mesh.faces], axis=1).copy()
                ).float()
                normals = torch.from_numpy(mesh.face_normals.copy()).float()
            else:
                # For vertex graph, use vertex positions and normals
                # Copy arrays to ensure they are writable for PyTorch
                coords = torch.from_numpy(mesh.vertices.copy()).float()
                normals = torch.from_numpy(mesh.vertex_normals.copy()).float()

            # Run model
            logits = self.model(
                x=graph.x,
                edge_index=graph.edge_index,
                batch=None,  # Single graph
                coords=coords if self.use_pretrained_encoders else None,
                normals=normals if self.use_pretrained_encoders else None,
            )

            # Get predictions
            probabilities = torch.softmax(logits, dim=1)
            labels = torch.argmax(probabilities, dim=1)
            confidence = probabilities.max(dim=1)[0]

        return labels.cpu().numpy(), confidence.cpu().numpy()

    def _segment_large_mesh(
        self, mesh: trimesh.Trimesh
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Segment large mesh by chunking.

        Uses OctreeChunker to split mesh into manageable chunks, segments each chunk,
        then merges results back together.

        Args:
            mesh: Large trimesh mesh

        Returns:
            labels: [N] segment labels per face
            confidence: [N] confidence scores per face
        """
        from cadling.chunker.mesh_chunker.mesh_chunker import MeshChunker
        from cadling.datamodel.mesh import MeshData

        _log.info(f"Chunking large mesh: {len(mesh.faces)} faces")

        # 1. Create chunker
        chunker = MeshChunker(strategy="octree", max_chunk_size=self.max_faces_per_chunk)

        # 2. Convert trimesh to MeshData for chunker
        mesh_data = MeshData(
            vertices=mesh.vertices,
            faces=mesh.faces,
            normals=mesh.face_normals if self.use_face_graph else mesh.vertex_normals,
        )

        # 3. Get chunks from chunker
        chunks = list(chunker.chunk_mesh_data(mesh_data, doc_name="large_mesh"))
        _log.info(f"Split into {len(chunks)} chunks")

        # 4. Process each chunk
        chunk_results = []
        for i, chunk in enumerate(chunks):
            # Extract face indices from chunk metadata
            face_indices = chunk.meta.properties.get("face_indices", [])

            if len(face_indices) == 0:
                _log.warning(f"Chunk {i} has no face_indices, skipping")
                continue

            _log.debug(f"Processing chunk {i}: {len(face_indices)} faces")

            # Create submesh from face indices
            try:
                submesh = mesh.submesh([face_indices], append=True)
            except Exception as e:
                _log.error(f"Failed to create submesh for chunk {i}: {e}")
                continue

            # Segment the submesh
            try:
                labels, confidence = self._segment_mesh(submesh)
                chunk_results.append((face_indices, labels, confidence))
            except Exception as e:
                _log.error(f"Failed to segment chunk {i}: {e}")
                continue

        if len(chunk_results) == 0:
            _log.error("All chunks failed to process!")
            # Return default labels
            num_faces = len(mesh.faces)
            return np.zeros(num_faces, dtype=np.int64), np.zeros(num_faces, dtype=np.float32)

        # 5. Merge results using confidence-based strategy
        final_labels, final_confidence = self._merge_chunk_results(
            chunk_results,
            num_nodes=len(mesh.faces),
        )

        _log.info(f"Merged {len(chunk_results)} chunk results")
        return final_labels, final_confidence

    def _merge_chunk_results(
        self,
        chunk_results: list[tuple[list[int], np.ndarray, np.ndarray]],
        num_nodes: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Merge segmentation results from multiple chunks.

        Uses confidence-based merging: for overlapping faces, keep prediction
        with highest confidence.

        Args:
            chunk_results: List of (face_indices, labels, confidence) tuples
            num_nodes: Total number of faces in original mesh

        Returns:
            merged_labels: [N] labels for all faces
            merged_confidence: [N] confidence scores for all faces
        """
        # Initialize output arrays
        merged_labels = np.zeros(num_nodes, dtype=np.int64)
        merged_confidence = np.zeros(num_nodes, dtype=np.float32)

        # For each chunk result
        for face_indices, labels, confidence in chunk_results:
            # Ensure labels and confidence have same length as face_indices
            min_len = min(len(face_indices), len(labels), len(confidence))

            for idx in range(min_len):
                face_idx = face_indices[idx]
                if face_idx >= num_nodes:
                    _log.warning(f"Face index {face_idx} out of range (max {num_nodes})")
                    continue

                # Use confidence-based merging: keep highest confidence prediction
                if confidence[idx] > merged_confidence[face_idx]:
                    merged_labels[face_idx] = labels[idx]
                    merged_confidence[face_idx] = confidence[idx]

        return merged_labels, merged_confidence

    def supports_batch_processing(self) -> bool:
        """Mesh segmentation supports batch processing."""
        return True

    def get_batch_size(self) -> int:
        """Recommended batch size for mesh segmentation."""
        return 8  # Conservative for memory

    def requires_gpu(self) -> bool:
        """Mesh segmentation benefits from GPU."""
        return True
