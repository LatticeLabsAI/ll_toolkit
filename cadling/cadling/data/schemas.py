"""PyArrow schemas for CAD-specific data types.

Defines schemas for efficient Parquet storage and HuggingFace Hub hosting
of CAD datasets. Supports command sequences (DeepCAD-style), B-Rep graphs
for GNN training, and text-conditioned generation data.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

_log = logging.getLogger(__name__)

# Lazy import PyArrow to avoid blocking at module load
_pa = None


def _ensure_pyarrow():
    """Lazily import PyArrow."""
    global _pa
    if _pa is None:
        try:
            import pyarrow as pa
            _pa = pa
        except ImportError:
            raise ImportError(
                "PyArrow is required for CAD schema definitions. "
                "Install via: pip install pyarrow>=14.0.0"
            )
    return _pa


# -----------------------------------------------------------------------------
# DeepCAD-style Command Sequence Schema
# -----------------------------------------------------------------------------

def get_command_sequence_schema(
    max_seq_len: int = 60,
    num_params: int = 16,
    include_text: bool = False,
    include_renders: bool = False,
) -> "pa.Schema":
    """Get PyArrow schema for DeepCAD-style command sequences.

    The schema stores quantized CAD command sequences suitable for
    autoregressive or VAE-based generation models.

    Args:
        max_seq_len: Maximum sequence length (default 60 per DeepCAD).
        num_params: Number of parameters per command (default 16).
        include_text: Whether to include text description field.
        include_renders: Whether to include multi-view render fields.

    Returns:
        PyArrow schema for command sequence data.
    """
    pa = _ensure_pyarrow()

    # Note: We use variable-length lists for flexibility.
    # The max_seq_len/num_params args are for documentation and metadata only.
    # Fixed-size lists (pa.list_(type, size)) require exact sizes which is inflexible.

    fields = [
        # Unique identifier for this sample
        pa.field("sample_id", pa.string(), nullable=False),

        # Command type indices (0=SOL, 1=Line, 2=Arc, 3=Circle, 4=Ext, 5=EOS)
        # Variable length to support different sequence lengths
        # int8 is sufficient for 6 command types (0-5), saves 4x memory
        pa.field(
            "command_types",
            pa.list_(pa.int8()),
            nullable=False,
        ),

        # Quantized parameters: stored as flat list
        # Reshape to (seq_len, num_params) during loading
        # int16 supports 65536 quantization levels, saves 2x memory
        pa.field(
            "parameters",
            pa.list_(pa.int16()),
            nullable=False,
        ),

        # Valid command mask (1=valid, 0=padding)
        pa.field(
            "mask",
            pa.list_(pa.float32()),
            nullable=False,
        ),

        # Actual command count before padding
        pa.field("num_commands", pa.int32(), nullable=False),

        # Original model ID for provenance
        pa.field("model_id", pa.string(), nullable=True),

        # Dataset source (e.g., "deepcad", "abc", "text2cad")
        pa.field("source", pa.string(), nullable=True),

        # Metadata as JSON string
        pa.field("metadata", pa.string(), nullable=True),

        # Normalization metadata for dequantization during inference
        # Center point used for centering the model before quantization
        pa.field("normalization_center", pa.list_(pa.float32()), nullable=True),
        # Scale factor applied during normalization
        pa.field("normalization_scale", pa.float32(), nullable=True),
    ]

    if include_text:
        fields.extend([
            # Natural language description
            pa.field("text_description", pa.string(), nullable=True),
            # Text embedding (optional, for retrieval)
            pa.field(
                "text_embedding",
                pa.list_(pa.float32()),
                nullable=True,
            ),
        ])

    if include_renders:
        fields.extend([
            # Multi-view renders as PNG bytes
            pa.field("render_front", pa.binary(), nullable=True),
            pa.field("render_top", pa.binary(), nullable=True),
            pa.field("render_iso", pa.binary(), nullable=True),
            # Optional segmentation masks
            pa.field("segmentation_mask", pa.binary(), nullable=True),
        ])

    return pa.schema(fields)


# Convenience constant for default schema
def COMMAND_SEQUENCE_SCHEMA() -> "pa.Schema":
    """Default command sequence schema without text or renders."""
    return get_command_sequence_schema()


# -----------------------------------------------------------------------------
# B-Rep Graph Schema for GNN Training
# -----------------------------------------------------------------------------

def get_brep_graph_schema(
    include_uv: bool = False,
    include_coedges: bool = False,
    max_faces: int = 1024,
    max_edges: int = 4096,
) -> "pa.Schema":
    """Get PyArrow schema for B-Rep graph data.

    The schema stores B-Rep topology as a graph suitable for GNN training:
    - Node features (face/edge properties)
    - Edge indices (adjacency relationships)
    - Optional UV-net style parameterization
    - Optional coedge sequence for autoregressive decoding

    Args:
        include_uv: Include UV-grid parameterization for faces.
        include_coedges: Include coedge sequence for BRepNet-style models.
        max_faces: Maximum number of faces (for padding).
        max_edges: Maximum number of edges (for padding).

    Returns:
        PyArrow schema for B-Rep graph data.
    """
    pa = _ensure_pyarrow()

    fields = [
        # Unique identifier
        pa.field("sample_id", pa.string(), nullable=False),

        # Number of actual faces/edges (before padding)
        pa.field("num_faces", pa.int32(), nullable=False),
        pa.field("num_edges", pa.int32(), nullable=False),

        # Face features: [num_faces, face_feat_dim]
        # Default features: surface_type, area, centroid(3), normal(3), curvatures(2)
        pa.field(
            "face_features",
            pa.list_(pa.float32()),
            nullable=False,
        ),
        pa.field("face_feature_dim", pa.int32(), nullable=False),

        # Edge features: [num_edges, edge_feat_dim]
        # Default features: curve_type, length, convexity, dihedral_angle
        pa.field(
            "edge_features",
            pa.list_(pa.float32()),
            nullable=False,
        ),
        pa.field("edge_feature_dim", pa.int32(), nullable=False),

        # Adjacency: edge_index [2, num_edges] as flat list
        pa.field(
            "edge_index",
            pa.list_(pa.int64()),
            nullable=False,
        ),

        # Face labels for segmentation tasks (optional)
        pa.field(
            "face_labels",
            pa.list_(pa.int32()),
            nullable=True,
        ),

        # Global graph features (optional)
        pa.field(
            "global_features",
            pa.list_(pa.float32()),
            nullable=True,
        ),

        # Source file path
        pa.field("source_path", pa.string(), nullable=True),

        # Dataset source
        pa.field("source", pa.string(), nullable=True),

        # Metadata as JSON string
        pa.field("metadata", pa.string(), nullable=True),
    ]

    if include_uv:
        fields.extend([
            # UV-grid samples per face: [num_faces, grid_size, grid_size, 3]
            # Stored flat, requires reshape
            pa.field(
                "uv_points",
                pa.list_(pa.float32()),
                nullable=True,
            ),
            pa.field("uv_grid_size", pa.int32(), nullable=True),
            # UV-grid normals: same shape as uv_points
            pa.field(
                "uv_normals",
                pa.list_(pa.float32()),
                nullable=True,
            ),
        ])

    if include_coedges:
        fields.extend([
            # Coedge sequence for autoregressive decoding
            # Each coedge: (face_id, edge_id, orientation)
            pa.field(
                "coedge_sequence",
                pa.list_(pa.int32()),
                nullable=True,
            ),
            pa.field("num_coedges", pa.int32(), nullable=True),
        ])

    return pa.schema(fields)


def BREP_GRAPH_SCHEMA() -> "pa.Schema":
    """Default B-Rep graph schema without UV or coedges."""
    return get_brep_graph_schema()


# -----------------------------------------------------------------------------
# Text-CAD Schema for Conditioned Generation
# -----------------------------------------------------------------------------

def get_text_cad_schema(
    max_seq_len: int = 60,
    num_params: int = 16,
    include_renders: bool = False,
) -> "pa.Schema":
    """Get PyArrow schema for text-conditioned CAD generation.

    Combines command sequences with natural language descriptions
    for text-to-CAD generation tasks (e.g., Text2CAD dataset).

    Args:
        max_seq_len: Maximum command sequence length.
        num_params: Number of parameters per command.
        include_renders: Include rendered views.

    Returns:
        PyArrow schema for text-CAD pairs.
    """
    pa = _ensure_pyarrow()

    fields = [
        # Unique identifier
        pa.field("sample_id", pa.string(), nullable=False),

        # Natural language description (required for text-CAD)
        pa.field("text", pa.string(), nullable=False),

        # 4-level text annotations for training curricula
        # High-level abstract description (~10 words)
        pa.field("text_abstract", pa.string(), nullable=True),
        # Medium detail intermediate description (~50 words)
        pa.field("text_intermediate", pa.string(), nullable=True),
        # Full detailed description (~200 words)
        pa.field("text_detailed", pa.string(), nullable=True),
        # Expert-level technical CAD terminology
        pa.field("text_expert", pa.string(), nullable=True),

        # Command sequence (same as COMMAND_SEQUENCE_SCHEMA)
        # Variable length for flexibility
        # int8 is sufficient for 6 command types (0-5), saves 4x memory
        pa.field(
            "command_types",
            pa.list_(pa.int8()),
            nullable=False,
        ),
        # int16 supports 65536 quantization levels, saves 2x memory
        pa.field(
            "parameters",
            pa.list_(pa.int16()),
            nullable=False,
        ),
        pa.field(
            "mask",
            pa.list_(pa.float32()),
            nullable=False,
        ),
        pa.field("num_commands", pa.int32(), nullable=False),

        # Text tokenization info (optional, for caching)
        pa.field(
            "text_token_ids",
            pa.list_(pa.int64()),
            nullable=True,
        ),
        pa.field(
            "text_attention_mask",
            pa.list_(pa.int32()),
            nullable=True,
        ),

        # Source info
        pa.field("model_id", pa.string(), nullable=True),
        pa.field("source", pa.string(), nullable=True),
        pa.field("metadata", pa.string(), nullable=True),

        # Normalization metadata for dequantization during inference
        # Center point used for centering the model before quantization
        pa.field("normalization_center", pa.list_(pa.float32()), nullable=True),
        # Scale factor applied during normalization
        pa.field("normalization_scale", pa.float32(), nullable=True),
    ]

    if include_renders:
        fields.extend([
            pa.field("render_front", pa.binary(), nullable=True),
            pa.field("render_top", pa.binary(), nullable=True),
            pa.field("render_iso", pa.binary(), nullable=True),
        ])

    return pa.schema(fields)


def TEXT_CAD_SCHEMA() -> "pa.Schema":
    """Default text-CAD schema without renders."""
    return get_text_cad_schema()


# -----------------------------------------------------------------------------
# Schema Validation Utilities
# -----------------------------------------------------------------------------

def validate_sample(
    sample: Dict[str, Any],
    schema: "pa.Schema",
    strict: bool = False,
) -> List[str]:
    """Validate a sample dictionary against a PyArrow schema.

    Args:
        sample: Dictionary with sample data.
        schema: PyArrow schema to validate against.
        strict: If True, raise on validation errors; else return error list.

    Returns:
        List of validation error messages (empty if valid).

    Raises:
        ValueError: If strict=True and validation fails.
    """
    pa = _ensure_pyarrow()

    errors: List[str] = []

    for field in schema:
        name = field.name
        nullable = field.nullable

        # Check required fields
        if name not in sample:
            if not nullable:
                errors.append(f"Missing required field: {name}")
            continue

        value = sample[name]

        # Check None values
        if value is None:
            if not nullable:
                errors.append(f"Field '{name}' cannot be None")
            continue

        # Type checking for common types
        if pa.types.is_string(field.type):
            if not isinstance(value, str):
                errors.append(
                    f"Field '{name}' should be string, got {type(value).__name__}"
                )
        elif pa.types.is_int32(field.type) or pa.types.is_int64(field.type):
            if not isinstance(value, (int, type(None))):
                errors.append(
                    f"Field '{name}' should be int, got {type(value).__name__}"
                )
        elif pa.types.is_list(field.type):
            if not isinstance(value, (list, tuple)):
                # Also accept numpy arrays
                try:
                    import numpy as np
                    if not isinstance(value, np.ndarray):
                        errors.append(
                            f"Field '{name}' should be list/array, "
                            f"got {type(value).__name__}"
                        )
                except ImportError:
                    errors.append(
                        f"Field '{name}' should be list, got {type(value).__name__}"
                    )

    if strict and errors:
        raise ValueError(
            f"Schema validation failed with {len(errors)} errors:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return errors


def sample_to_arrow(
    sample: Dict[str, Any],
    schema: "pa.Schema",
) -> "pa.RecordBatch":
    """Convert a sample dictionary to a PyArrow RecordBatch.

    Args:
        sample: Dictionary with sample data.
        schema: PyArrow schema defining the structure.

    Returns:
        PyArrow RecordBatch with the sample data.
    """
    pa = _ensure_pyarrow()

    # Build arrays for each field
    arrays = []
    for field in schema:
        name = field.name
        value = sample.get(name)

        if value is None:
            # Create null array
            arrays.append(pa.nulls(1, type=field.type))
        elif pa.types.is_list(field.type):
            # Handle list types
            if hasattr(value, "tolist"):
                value = value.tolist()
            arrays.append(pa.array([value], type=field.type))
        elif pa.types.is_binary(field.type):
            arrays.append(pa.array([value], type=field.type))
        else:
            arrays.append(pa.array([value], type=field.type))

    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def samples_to_table(
    samples: List[Dict[str, Any]],
    schema: "pa.Schema",
) -> "pa.Table":
    """Convert a list of samples to a PyArrow Table.

    Args:
        samples: List of sample dictionaries.
        schema: PyArrow schema defining the structure.

    Returns:
        PyArrow Table containing all samples.
    """
    pa = _ensure_pyarrow()

    if not samples:
        return pa.Table.from_pydict({f.name: [] for f in schema}, schema=schema)

    # Convert to columnar format
    columns: Dict[str, List[Any]] = {f.name: [] for f in schema}

    for sample in samples:
        for field in schema:
            name = field.name
            value = sample.get(name)

            # Convert numpy arrays to lists
            if hasattr(value, "tolist"):
                value = value.tolist()

            columns[name].append(value)

    return pa.Table.from_pydict(columns, schema=schema)


def infer_schema_from_sample(
    sample: Dict[str, Any],
    include_nullable: bool = True,
) -> "pa.Schema":
    """Infer a PyArrow schema from a sample dictionary.

    Useful for creating schemas from existing data. Handles numpy arrays
    and nested structures.

    Args:
        sample: Sample dictionary to infer schema from.
        include_nullable: Whether to mark fields as nullable.

    Returns:
        Inferred PyArrow schema.
    """
    pa = _ensure_pyarrow()

    fields = []

    for name, value in sample.items():
        if value is None:
            # Default to string for None values
            fields.append(pa.field(name, pa.string(), nullable=True))
            continue

        # Infer type from value
        if isinstance(value, str):
            fields.append(pa.field(name, pa.string(), nullable=include_nullable))
        elif isinstance(value, bool):
            fields.append(pa.field(name, pa.bool_(), nullable=include_nullable))
        elif isinstance(value, int):
            fields.append(pa.field(name, pa.int64(), nullable=include_nullable))
        elif isinstance(value, float):
            fields.append(pa.field(name, pa.float64(), nullable=include_nullable))
        elif isinstance(value, bytes):
            fields.append(pa.field(name, pa.binary(), nullable=include_nullable))
        elif isinstance(value, (list, tuple)):
            # Infer list element type
            if len(value) > 0:
                elem = value[0]
                if isinstance(elem, int):
                    elem_type = pa.int32()
                elif isinstance(elem, float):
                    elem_type = pa.float32()
                else:
                    elem_type = pa.string()
            else:
                elem_type = pa.float32()
            fields.append(
                pa.field(name, pa.list_(elem_type), nullable=include_nullable)
            )
        else:
            # Try numpy array
            try:
                import numpy as np
                if isinstance(value, np.ndarray):
                    if value.dtype in (np.int32, np.int64):
                        elem_type = pa.int32()
                    elif value.dtype in (np.float32, np.float64):
                        elem_type = pa.float32()
                    else:
                        elem_type = pa.float32()
                    fields.append(
                        pa.field(name, pa.list_(elem_type), nullable=include_nullable)
                    )
                    continue
            except ImportError:
                pass

            # Default to string
            fields.append(pa.field(name, pa.string(), nullable=include_nullable))

    return pa.schema(fields)


__all__ = [
    "get_command_sequence_schema",
    "get_brep_graph_schema",
    "get_text_cad_schema",
    "COMMAND_SEQUENCE_SCHEMA",
    "BREP_GRAPH_SCHEMA",
    "TEXT_CAD_SCHEMA",
    "validate_sample",
    "sample_to_arrow",
    "samples_to_table",
    "infer_schema_from_sample",
]
