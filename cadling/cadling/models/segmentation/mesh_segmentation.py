"""Mesh segmentation model using graph neural networks.

This module provides ML models for semantic segmentation of triangle meshes (STL, OBJ, PLY).
Uses EdgeConv-based graph neural networks to classify mesh vertices/faces into semantic regions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh
from gguf import Any  # Re-exports typing.Any, used for chunk type hints

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
        artifacts_path: Path | None = None,
        num_classes: int = 12,
        use_pretrained_encoders: bool = False,
        hidden_dims: list[int] | None = None,
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

    def _load_mesh_from_item(self, item: MeshItem) -> trimesh.Trimesh | None:
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
            _log.error(f"Failed to load mesh from item: {e}", exc_info=True)
            return None

    def _segment_mesh(
        self, mesh: trimesh.Trimesh
    ) -> tuple[np.ndarray, np.ndarray | None]:
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

            # Run model (model is guaranteed non-None by caller check)
            assert self.model is not None, "Model must be loaded before inference"
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
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Segment large mesh by chunking.

        Uses OctreeChunker to split mesh into manageable chunks, segments each chunk,
        then merges results back together.

        Args:
            mesh: Large trimesh mesh

        Returns:
            labels: [N] segment labels per face
            confidence: [N] confidence scores per face
        """
        from cadling.chunker.mesh_chunker.mesh_chunker import MeshChunker, MeshData

        _log.info(f"Chunking large mesh: {len(mesh.faces)} faces")

        # 1. Create chunker
        chunker = MeshChunker(strategy="octree", max_faces_per_chunk=self.max_faces_per_chunk)

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
                # Try to recover face_indices from chunk's spatial metadata
                face_indices = self._recover_face_indices_from_chunk(mesh, chunk, i)
                if len(face_indices) == 0:
                    _log.warning(f"Chunk {i} has no face_indices and recovery failed, skipping")
                    continue
                _log.debug(f"Recovered {len(face_indices)} face indices for chunk {i}")

            _log.debug(f"Processing chunk {i}: {len(face_indices)} faces")

            # Create submesh from face indices
            try:
                submesh_result = mesh.submesh([face_indices], append=True)
                # submesh with append=True returns a single Trimesh, not a list
                if isinstance(submesh_result, list):
                    submesh = submesh_result[0] if submesh_result else None
                else:
                    submesh = submesh_result
                if submesh is None:
                    _log.warning(f"Empty submesh for chunk {i}, skipping")
                    continue
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

    def _recover_face_indices_from_chunk(
        self, mesh: trimesh.Trimesh, chunk: Any, chunk_idx: int
    ) -> list[int]:
        """Try to recover face_indices from chunk's spatial metadata.

        Uses multi-strategy approach:
        1. Primary: Use bounding box from chunk metadata
        2. Fallback: Use chunk_id pattern for spatial inference
        3. Fallback: Uniform distribution based on chunk index

        Args:
            mesh: Original mesh
            chunk: Chunk data with metadata
            chunk_idx: Chunk index for logging

        Returns:
            List of face indices within the chunk's region
        """
        # Strategy 1: Try to get bounding box from chunk metadata
        try:
            bbox = None
            if hasattr(chunk, "meta") and hasattr(chunk.meta, "properties"):
                bbox = chunk.meta.properties.get("bounding_box")
            elif hasattr(chunk, "bounding_box"):
                bbox = chunk.bounding_box

            if bbox is not None:
                # Parse bounding box
                if isinstance(bbox, dict):
                    min_pt = np.array([bbox.get("min_x", -np.inf),
                                       bbox.get("min_y", -np.inf),
                                       bbox.get("min_z", -np.inf)])
                    max_pt = np.array([bbox.get("max_x", np.inf),
                                       bbox.get("max_y", np.inf),
                                       bbox.get("max_z", np.inf)])
                elif hasattr(bbox, "min") and hasattr(bbox, "max"):
                    min_pt = np.array(bbox.min)
                    max_pt = np.array(bbox.max)
                else:
                    min_pt = max_pt = None

                if min_pt is not None and max_pt is not None:
                    # Compute face centroids
                    face_vertices = mesh.vertices[mesh.faces]  # [F, 3, 3]
                    face_centroids = face_vertices.mean(axis=1)  # [F, 3]

                    # Find faces within bounding box (with small epsilon for tolerance)
                    epsilon = 1e-6
                    within_bbox = np.all(
                        (face_centroids >= min_pt - epsilon) & (face_centroids <= max_pt + epsilon),
                        axis=1
                    )
                    face_indices = np.where(within_bbox)[0].tolist()

                    if face_indices:
                        _log.debug(
                            f"Recovered {len(face_indices)} faces from bbox for chunk {chunk_idx}"
                        )
                        return face_indices

        except Exception as e:
            _log.debug(f"Bbox recovery failed for chunk {chunk_idx}: {e}")

        # Strategy 2: Try chunk_id-based spatial inference
        try:
            chunk_id = getattr(chunk, "chunk_id", None)
            if chunk_id and isinstance(chunk_id, str):
                # Parse octree-style chunk_id like "0_1_3" (depth indices)
                parts = chunk_id.split("_")
                if len(parts) >= 2 and all(p.isdigit() for p in parts):
                    # Use chunk_id to infer spatial region
                    face_indices = self._infer_faces_from_chunk_id(mesh, parts)
                    if face_indices:
                        _log.debug(
                            f"Recovered {len(face_indices)} faces from chunk_id for chunk {chunk_idx}"
                        )
                        return face_indices
        except Exception as e:
            _log.debug(f"Chunk_id recovery failed for chunk {chunk_idx}: {e}")

        # Strategy 3: Uniform distribution fallback
        try:
            total_chunks = getattr(self, "_total_chunks", 8)  # Default assumption
            faces_per_chunk = max(1, len(mesh.faces) // total_chunks)
            start_idx = chunk_idx * faces_per_chunk
            end_idx = min(start_idx + faces_per_chunk, len(mesh.faces))

            if start_idx < len(mesh.faces):
                face_indices = list(range(start_idx, end_idx))
                _log.debug(
                    f"Using uniform distribution fallback: {len(face_indices)} faces for chunk {chunk_idx}"
                )
                return face_indices
        except Exception as e:
            _log.debug(f"Uniform distribution fallback failed: {e}")

        _log.warning(f"All face index recovery strategies failed for chunk {chunk_idx}")
        return []

    def _infer_faces_from_chunk_id(
        self, mesh: trimesh.Trimesh, chunk_id_parts: list[str]
    ) -> list[int]:
        """Infer face indices from octree-style chunk_id.

        Args:
            mesh: Original mesh
            chunk_id_parts: List of string indices from chunk_id

        Returns:
            List of face indices
        """
        # Get mesh bounds
        min_xyz = mesh.vertices.min(axis=0)
        max_xyz = mesh.vertices.max(axis=0)

        # Use first two parts to determine quadrant/octant
        if len(chunk_id_parts) >= 2:
            # Interpret as (depth, octant_index)
            octant_idx = int(chunk_id_parts[-1]) % 8

            mid_xyz = (min_xyz + max_xyz) / 2.0

            # Determine bounds for this octant
            x_low = min_xyz[0] if (octant_idx & 1) == 0 else mid_xyz[0]
            x_high = mid_xyz[0] if (octant_idx & 1) == 0 else max_xyz[0]
            y_low = min_xyz[1] if (octant_idx & 2) == 0 else mid_xyz[1]
            y_high = mid_xyz[1] if (octant_idx & 2) == 0 else max_xyz[1]
            z_low = min_xyz[2] if (octant_idx & 4) == 0 else mid_xyz[2]
            z_high = mid_xyz[2] if (octant_idx & 4) == 0 else max_xyz[2]

            min_pt = np.array([x_low, y_low, z_low])
            max_pt = np.array([x_high, y_high, z_high])

            # Find faces in this region
            face_vertices = mesh.vertices[mesh.faces]
            face_centroids = face_vertices.mean(axis=1)

            epsilon = 1e-6
            within_bounds = np.all(
                (face_centroids >= min_pt - epsilon) & (face_centroids <= max_pt + epsilon),
                axis=1
            )
            return np.where(within_bounds)[0].tolist()

        return []

    def supports_batch_processing(self) -> bool:
        """Mesh segmentation supports batch processing."""
        return True

    def get_batch_size(self) -> int:
        """Recommended batch size for mesh segmentation."""
        return 8  # Conservative for memory

    def requires_gpu(self) -> bool:
        """Mesh segmentation benefits from GPU."""
        return True
