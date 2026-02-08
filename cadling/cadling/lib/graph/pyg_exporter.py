"""PyTorch Geometric Export with UV-Grids

Exports CAD B-Rep graphs to PyTorch Geometric Data format with industry-standard
features following BRepNet, UV-Net, and AAGNet specifications:
- Node features: 48 dimensions with UV-grid statistics
- Edge features: 16 dimensions with geometric properties
- Face UV-grids: [num_faces, 10, 10, 7]
- Edge UV-grids: [num_edges, 10, 6]

Provides export, save, and validation functions for PyG Data objects.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    from torch_geometric.data import Data
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logging.warning("PyTorch or torch_geometric not available. PyG export will be disabled.")


logger = logging.getLogger(__name__)


def export_to_pyg_with_uvgrids(
    node_features: np.ndarray,
    edge_index: np.ndarray,
    edge_features: np.ndarray,
    face_uv_grids: Optional[Dict[int, np.ndarray]] = None,
    edge_uv_grids: Optional[Dict[int, np.ndarray]] = None,
    labels: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[Data]:
    """
    Export enhanced PyG Data object with UV-grids.

    Args:
        node_features: Node feature matrix [num_nodes, 48]
        edge_index: Edge connectivity [2, num_edges]
        edge_features: Edge feature matrix [num_edges, 16]
        face_uv_grids: Dict mapping face index -> UV-grid [10, 10, 7]
        edge_uv_grids: Dict mapping edge index -> UV-grid [10, 6]
        labels: Optional node labels [num_nodes] for supervised learning
        metadata: Optional metadata dictionary

    Returns:
        PyG Data object with:
        - x: [num_nodes, 48] node features (torch.FloatTensor)
        - edge_index: [2, num_edges] connectivity (torch.LongTensor)
        - edge_attr: [num_edges, 16] edge features (torch.FloatTensor)
        - face_uv_grids: [num_faces, 10, 10, 7] (torch.FloatTensor)
        - edge_uv_grids: [num_edges, 10, 6] (torch.FloatTensor)
        - y: Optional labels (torch.LongTensor)
        - metadata: Dict with model info

    Returns None if PyTorch not available.
    """
    if not HAS_TORCH:
        logger.error("PyTorch not available - cannot export to PyG format")
        return None

    try:
        num_nodes = node_features.shape[0]
        num_edges = edge_features.shape[0]

        # Convert to torch tensors
        x = torch.from_numpy(node_features).float()  # [num_nodes, 48]
        edge_index_tensor = torch.from_numpy(edge_index).long()  # [2, num_edges]
        edge_attr = torch.from_numpy(edge_features).float()  # [num_edges, 16]

        # Convert UV-grids to tensors
        # Face UV-grids: [num_faces, 10, 10, 7]
        if face_uv_grids is not None and len(face_uv_grids) > 0:
            face_uv_list = []
            for i in range(num_nodes):
                if i in face_uv_grids:
                    face_uv_list.append(face_uv_grids[i])
                else:
                    # Missing UV-grid - fill with zeros
                    face_uv_list.append(np.zeros((10, 10, 7), dtype=np.float32))
            face_uv_tensor = torch.from_numpy(np.stack(face_uv_list, axis=0)).float()
        else:
            # No UV-grids provided - create zeros
            face_uv_tensor = torch.zeros((num_nodes, 10, 10, 7), dtype=torch.float32)

        # Edge UV-grids: [num_edges, 10, 6]
        if edge_uv_grids is not None and len(edge_uv_grids) > 0:
            edge_uv_list = []
            for i in range(num_edges):
                if i in edge_uv_grids:
                    edge_uv_list.append(edge_uv_grids[i])
                else:
                    # Missing UV-grid - fill with zeros
                    edge_uv_list.append(np.zeros((10, 6), dtype=np.float32))
            edge_uv_tensor = torch.from_numpy(np.stack(edge_uv_list, axis=0)).float()
        else:
            # No UV-grids provided - create zeros
            edge_uv_tensor = torch.zeros((num_edges, 10, 6), dtype=torch.float32)

        # Convert labels if provided
        y_tensor = None
        if labels is not None:
            y_tensor = torch.from_numpy(labels).long()

        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index_tensor,
            edge_attr=edge_attr,
            face_uv_grids=face_uv_tensor,
            edge_uv_grids=edge_uv_tensor,
            y=y_tensor
        )

        # Add metadata as custom attribute
        if metadata is not None:
            data.metadata = metadata
        else:
            data.metadata = {}

        # Add shape info to metadata
        data.metadata.update({
            'num_nodes': int(num_nodes),
            'num_edges': int(num_edges),
            'node_feature_dim': int(node_features.shape[1]),
            'edge_feature_dim': int(edge_features.shape[1]),
            'face_uv_grid_shape': list(face_uv_tensor.shape),
            'edge_uv_grid_shape': list(edge_uv_tensor.shape),
            'has_labels': labels is not None
        })

        logger.info(f"Created PyG Data object: {num_nodes} nodes, {num_edges} edges, "
                   f"{node_features.shape[1]}-dim node features, "
                   f"{edge_features.shape[1]}-dim edge features")

        return data

    except Exception as e:
        logger.error(f"Failed to export to PyG format: {e}")
        return None


def save_pyg_data(
    data: Data,
    output_path: Path,
    include_metadata: bool = True
) -> None:
    """
    Save PyG Data object to .pt file.

    Args:
        data: PyG Data object to save
        output_path: Path to save .pt file
        include_metadata: If True, save separate metadata.json file

    Raises:
        RuntimeError: If save fails
    """
    if not HAS_TORCH:
        logger.error("PyTorch not available - cannot save PyG data")
        return

    try:
        # Ensure parent directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save PyG Data object
        torch.save(data, output_path)
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved PyG Data to: {output_path} ({file_size_mb:.2f} MB)")

        # Save metadata separately if requested
        if include_metadata and hasattr(data, 'metadata'):
            metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
            with open(metadata_path, 'w') as f:
                # Convert metadata to JSON-serializable format
                metadata_json = _make_json_serializable(data.metadata)
                json.dump(metadata_json, f, indent=2)
            logger.info(f"Saved metadata to: {metadata_path}")

    except Exception as e:
        logger.error(f"Failed to save PyG data: {e}")
        raise RuntimeError(f"Failed to save PyG data: {e}")


def validate_pyg_data(data: Data) -> List[str]:
    """
    Validate PyG Data structure matches industry standards.

    Checks:
    - x.shape[1] == 48 (node feature dimension)
    - edge_attr.shape[1] == 16 (edge feature dimension)
    - face_uv_grids.shape == [num_faces, 10, 10, 7]
    - edge_uv_grids.shape == [num_edges, 10, 6]
    - edge_index.shape == [2, num_edges]
    - All tensors are finite (no NaN/Inf)
    - Features are non-placeholder (>5% non-zero)
    - Edge index references valid node indices

    Args:
        data: PyG Data object to validate

    Returns:
        List of error messages (empty if valid)
    """
    if not HAS_TORCH:
        return ["PyTorch not available - cannot validate PyG data"]

    errors = []

    try:
        # Check node features
        if not hasattr(data, 'x') or data.x is None:
            errors.append("Missing node features (x)")
        else:
            if data.x.dim() != 2:
                errors.append(f"Node features should be 2D, got {data.x.dim()}D")
            elif data.x.shape[1] != 48:
                errors.append(f"Node features should be 48-dim, got {data.x.shape[1]}-dim")

            # Check for NaN/Inf
            if torch.isnan(data.x).any():
                errors.append("Node features contain NaN values")
            if torch.isinf(data.x).any():
                errors.append("Node features contain Inf values")

            # Check for placeholder data (all zeros or too many zeros)
            non_zero_pct = (data.x != 0).float().mean().item() * 100
            if non_zero_pct < 5.0:
                errors.append(f"Node features appear to be placeholder data (only {non_zero_pct:.1f}% non-zero)")

        # Check edge index
        if not hasattr(data, 'edge_index') or data.edge_index is None:
            errors.append("Missing edge index")
        else:
            if data.edge_index.dim() != 2:
                errors.append(f"Edge index should be 2D, got {data.edge_index.dim()}D")
            elif data.edge_index.shape[0] != 2:
                errors.append(f"Edge index first dim should be 2, got {data.edge_index.shape[0]}")

            # Check edge index references valid nodes
            if hasattr(data, 'x') and data.x is not None:
                max_node_idx = data.edge_index.max().item()
                num_nodes = data.x.shape[0]
                if max_node_idx >= num_nodes:
                    errors.append(f"Edge index references node {max_node_idx}, but only {num_nodes} nodes exist")

        # Check edge features
        if not hasattr(data, 'edge_attr') or data.edge_attr is None:
            errors.append("Missing edge features (edge_attr)")
        else:
            if data.edge_attr.dim() != 2:
                errors.append(f"Edge features should be 2D, got {data.edge_attr.dim()}D")
            elif data.edge_attr.shape[1] != 16:
                errors.append(f"Edge features should be 16-dim, got {data.edge_attr.shape[1]}-dim")

            # Check for NaN/Inf
            if torch.isnan(data.edge_attr).any():
                errors.append("Edge features contain NaN values")
            if torch.isinf(data.edge_attr).any():
                errors.append("Edge features contain Inf values")

            # Check for placeholder data
            non_zero_pct = (data.edge_attr != 0).float().mean().item() * 100
            if non_zero_pct < 5.0:
                errors.append(f"Edge features appear to be placeholder data (only {non_zero_pct:.1f}% non-zero)")

        # Check face UV-grids
        if not hasattr(data, 'face_uv_grids') or data.face_uv_grids is None:
            errors.append("Missing face UV-grids")
        else:
            expected_shape = (data.x.shape[0], 10, 10, 7) if hasattr(data, 'x') else None
            if data.face_uv_grids.dim() != 4:
                errors.append(f"Face UV-grids should be 4D, got {data.face_uv_grids.dim()}D")
            elif expected_shape and data.face_uv_grids.shape != expected_shape:
                errors.append(f"Face UV-grids shape should be {expected_shape}, got {tuple(data.face_uv_grids.shape)}")

            # Check channel dimensions
            if data.face_uv_grids.shape[-1] != 7:
                errors.append(f"Face UV-grids should have 7 channels, got {data.face_uv_grids.shape[-1]}")
            if data.face_uv_grids.shape[-2] != 10 or data.face_uv_grids.shape[-3] != 10:
                errors.append(f"Face UV-grids should be 10×10, got {data.face_uv_grids.shape[-3]}×{data.face_uv_grids.shape[-2]}")

        # Check edge UV-grids
        if not hasattr(data, 'edge_uv_grids') or data.edge_uv_grids is None:
            errors.append("Missing edge UV-grids")
        else:
            expected_shape = (data.edge_attr.shape[0], 10, 6) if hasattr(data, 'edge_attr') else None
            if data.edge_uv_grids.dim() != 3:
                errors.append(f"Edge UV-grids should be 3D, got {data.edge_uv_grids.dim()}D")
            elif expected_shape and data.edge_uv_grids.shape != expected_shape:
                errors.append(f"Edge UV-grids shape should be {expected_shape}, got {tuple(data.edge_uv_grids.shape)}")

            # Check dimensions
            if data.edge_uv_grids.shape[-1] != 6:
                errors.append(f"Edge UV-grids should have 6 channels, got {data.edge_uv_grids.shape[-1]}")
            if data.edge_uv_grids.shape[-2] != 10:
                errors.append(f"Edge UV-grids should have 10 samples, got {data.edge_uv_grids.shape[-2]}")

        # Check metadata
        if not hasattr(data, 'metadata'):
            errors.append("Missing metadata attribute")
        elif not isinstance(data.metadata, dict):
            errors.append(f"Metadata should be dict, got {type(data.metadata)}")

        # Log validation summary
        if len(errors) == 0:
            logger.info("PyG Data validation: PASSED")
        else:
            logger.warning(f"PyG Data validation: FAILED with {len(errors)} errors")
            for error in errors:
                logger.warning(f"  - {error}")

    except Exception as e:
        errors.append(f"Validation failed with exception: {e}")
        logger.error(f"PyG Data validation exception: {e}")

    return errors


def _make_json_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable format.

    Handles numpy arrays, torch tensors, and other common types.
    """
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif HAS_TORCH and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # Try to convert to string as fallback
        return str(obj)


def create_pyg_batch(data_list: List[Data]) -> Optional[Data]:
    """
    Create batched PyG Data object from list of Data objects.

    Args:
        data_list: List of PyG Data objects

    Returns:
        Batched PyG Data object, or None if batching fails
    """
    if not HAS_TORCH:
        logger.error("PyTorch not available - cannot create batch")
        return None

    try:
        from torch_geometric.data import Batch
        batch = Batch.from_data_list(data_list)
        logger.info(f"Created batch with {len(data_list)} graphs")
        return batch
    except Exception as e:
        logger.error(f"Failed to create batch: {e}")
        return None
