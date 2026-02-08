"""Base data models for cadling.

This module provides the foundational data structures for CAD document processing,
adapted from docling's proven data models but extended for 3D geometry, topology,
and CAD-specific features.

Classes:
    InputFormat: Enum of supported CAD formats.
    CADDocumentOrigin: Metadata about the source CAD file.
    Segment: Semantic geometric segment for classification and RAG.
    CADItem: Base class for all CAD items (entities, meshes, assemblies).
    CADlingDocument: Central data structure for processed CAD files.
    ConversionResult: Wrapper for conversion results and status.

Example:
    doc = CADlingDocument(name="part.step")
    item = STEPEntityItem(entity_id=31, entity_type="CIRCLE")
    doc.add_item(item)

    # Add semantic segments
    segment = Segment(segment_id="seg_0", segment_type="feature")
    doc.add_segment(segment)

    json_output = doc.export_to_json()
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np

from pydantic import BaseModel, Field, field_validator

_log = logging.getLogger(__name__)


class InputFormat(str, Enum):
    """Supported CAD file formats.

    This enum defines all CAD formats that cadling can process, similar to
    docling's InputFormat but for CAD files instead of documents.
    """

    STEP = "step"
    STL = "stl"
    BREP = "brep"
    IGES = "iges"
    CAD_IMAGE = "cad_image"  # Rendered CAD image (for optical recognition)
    DXF = "dxf"  # 2D technical drawing (AutoCAD DXF)
    PDF_DRAWING = "pdf_drawing"  # Vector-based PDF engineering drawing
    PDF_RASTER = "pdf_raster"  # Scanned/raster PDF (needs VLM/OCR path)


class ConversionStatus(str, Enum):
    """Status of a conversion operation."""

    SUCCESS = "success"
    PARTIAL = "partial"  # Completed with errors
    FAILURE = "failure"


class CADDocumentOrigin(BaseModel):
    """Metadata about the source CAD file.

    Attributes:
        filename: Original filename.
        format: CAD file format.
        binary_hash: SHA256 hash of file content.
        mimetype: MIME type (optional).
        filesize: File size in bytes.
    """

    filename: str
    format: InputFormat
    binary_hash: str
    mimetype: Optional[str] = None
    filesize: Optional[int] = None


class BoundingBox3D(BaseModel):
    """3D bounding box for CAD geometry.

    Unlike docling's 2D BoundingBox, this represents a 3D axis-aligned
    bounding box (AABB) for CAD geometry.

    Attributes:
        x_min, y_min, z_min: Minimum coordinates.
        x_max, y_max, z_max: Maximum coordinates.
    """

    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float

    @property
    def center(self) -> Tuple[float, float, float]:
        """Center point of the bounding box."""
        return (
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
            (self.z_min + self.z_max) / 2,
        )

    @property
    def size(self) -> Tuple[float, float, float]:
        """Size of the bounding box (width, height, depth)."""
        return (
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            self.z_max - self.z_min,
        )

    @property
    def volume(self) -> float:
        """Volume of the bounding box."""
        size = self.size
        return size[0] * size[1] * size[2]


class Segment(BaseModel):
    """Semantic segment with geometric meaning.

    Unlike chunks (for LLMs), segments are semantic regions identified
    by segmentation models for classification, vision-text association,
    and RAG. Segments represent meaningful geometric features like holes,
    pockets, bosses, or other manufacturing features.

    Attributes:
        segment_id: Unique identifier for this segment.
        segment_type: Type of segment ("face", "feature", "component", "region").
        item_ids: List of CADItem IDs that belong to this segment.
        bbox: 3D bounding box of this segment (optional).
        properties: Predicted class, confidence, and other segment properties.
        visual_representation: Rendered image or point cloud (optional).
        embedding: Segment-level embedding vector (optional).
        parent_segment: ID of parent segment for hierarchical segmentation (optional).
        children_segments: IDs of child segments (for hierarchical segmentation).

    Example:
        segment = Segment(
            segment_id="seg_0",
            segment_type="feature",
            item_ids=["face_1", "face_2", "face_3"],
            properties={
                "manufacturing_feature": "hole",
                "confidence": 0.95,
                "diameter": 10.0
            }
        )
    """

    segment_id: str
    segment_type: str  # "face", "feature", "component", "region"
    item_ids: List[str] = Field(default_factory=list)
    bbox: Optional[BoundingBox3D] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    visual_representation: Optional[Any] = None  # Rendered image/point cloud
    embedding: Optional[List[float]] = None
    parent_segment: Optional[str] = None
    children_segments: List[str] = Field(default_factory=list)


class CADItemLabel(BaseModel):
    """Label for a CAD item.

    Similar to docling's DocItemLabel but adapted for CAD terminology.

    Attributes:
        text: Human-readable label text.
        entity_type: Type of entity (for STEP entities).
    """

    text: str
    entity_type: Optional[str] = None


class ProvenanceItem(BaseModel):
    """Provenance information for tracking processing steps.

    Attributes:
        component_type: Type of component that processed this item.
        component_name: Name of the component.
        timestamp: When processing occurred.
    """

    component_type: str
    component_name: str
    timestamp: datetime = Field(default_factory=datetime.now)


class CADItem(BaseModel):
    """Base class for all CAD items.

    This is the foundational class for all geometric elements, similar to
    docling's DocItem but adapted for CAD data.

    CADItem hierarchy:
        CADItem (base)
        ├── STEPEntityItem (STEP geometric entities)
        ├── MeshItem (mesh data from STL, OBJ)
        ├── AssemblyItem (multi-part assemblies)
        └── AnnotationItem (dimensions, tolerances)

    Attributes:
        item_type: Type identifier (e.g., "step_entity", "mesh").
        label: Human-readable label.
        text: Text representation of this item.
        bbox: 3D bounding box (optional).
        properties: Custom properties (volume, mass, etc.).
        parent: Parent item ID (for hierarchies).
        children: Child item IDs.
        prov: Provenance chain.
    """

    item_type: str
    label: CADItemLabel
    item_id: Optional[str] = None
    text: Optional[str] = None

    # Geometric properties
    bbox: Optional[BoundingBox3D] = None
    properties: Dict[str, Any] = Field(default_factory=dict)

    # Hierarchy
    parent: Optional[str] = None
    children: List[str] = Field(default_factory=list)

    # Provenance
    prov: List[ProvenanceItem] = Field(default_factory=list)

    def add_provenance(self, component_type: str, component_name: str):
        """Add provenance information.

        Args:
            component_type: Type of component.
            component_name: Name of component.
        """
        self.prov.append(
            ProvenanceItem(
                component_type=component_type,
                component_name=component_name,
            )
        )


class TopologyGraph(BaseModel):
    """Topology graph for CAD entity references.

    STEP files and other CAD formats have entities that reference each other
    (e.g., a FACE references EDGEs, an EDGE references VERTEXes). This graph
    structure captures those relationships.

    This is generated by ll_stepnet's TopologyBuilder.

    Attributes:
        num_nodes: Number of nodes (entities).
        num_edges: Number of edges (references).
        adjacency_list: Adjacency list representation.
        node_features: Feature matrix for nodes (optional).
        edge_features: Feature matrix for edges (optional).
    """

    num_nodes: int
    num_edges: int
    adjacency_list: Dict[int, List[int]] = Field(default_factory=dict)
    node_features: Optional[List[List[float]]] = None
    edge_features: Optional[List[List[float]]] = None

    def add_edge(self, from_node: int, to_node: int):
        """Add an edge to the graph.

        Args:
            from_node: Source node ID.
            to_node: Target node ID.
        """
        if from_node not in self.adjacency_list:
            self.adjacency_list[from_node] = []
        self.adjacency_list[from_node].append(to_node)
        self.num_edges += 1

    def get_neighbors(self, node_id: int) -> List[int]:
        """Get neighbors of a node.

        Args:
            node_id: Node ID.

        Returns:
            List of neighbor node IDs.
        """
        return self.adjacency_list.get(node_id, [])

    def to_numpy_node_features(self) -> Optional[np.ndarray]:
        """Return node features as a numpy array for GeoToken.

        Converts the internal list-of-lists storage to a numpy array
        matching the format expected by geotoken.GraphTokenizer.

        Returns:
            np.ndarray of shape (N, D), dtype float32, or None if no features.
        """
        if self.node_features is None:
            return None
        return np.array(self.node_features, dtype=np.float32)

    def to_numpy_edge_features(self) -> Optional[np.ndarray]:
        """Return edge features as a numpy array for GeoToken.

        Returns:
            np.ndarray of shape (M, D), dtype float32, or None if no features.
        """
        if self.edge_features is None:
            return None
        return np.array(self.edge_features, dtype=np.float32)

    def to_edge_index(self) -> np.ndarray:
        """Return edge index as (2, M) int64 array for GeoToken/PyG.

        Converts the adjacency list into the standard COO-format edge
        index used by PyTorch Geometric and geotoken.GraphTokenizer.

        Returns:
            np.ndarray of shape (2, M), dtype int64, where M is num_edges.
        """
        edges = []
        for src, targets in self.adjacency_list.items():
            for tgt in targets:
                edges.append([src, tgt])
        if not edges:
            return np.zeros((2, 0), dtype=np.int64)
        return np.array(edges, dtype=np.int64).T


class ProcessingStep(BaseModel):
    """Record of a processing step.

    Attributes:
        step_name: Name of the step.
        component: Component that executed this step.
        timestamp: When step was executed.
        duration_ms: Duration in milliseconds.
        status: Status of this step.
    """

    step_name: str
    component: str
    timestamp: datetime = Field(default_factory=datetime.now)
    duration_ms: Optional[float] = None
    status: ConversionStatus = ConversionStatus.SUCCESS


class CADlingDocument(BaseModel):
    """Central data structure for CAD documents.

    This is the equivalent of docling's DoclingDocument but adapted for CAD files.
    It contains hierarchical items (entities, meshes, assemblies), CAD-specific
    metadata (topology, embeddings), semantic segments, and methods for export.

    Attributes:
        name: Document name.
        format: CAD file format.
        origin: Source file metadata.
        hash: Document hash.
        items: Hierarchical list of CAD items.
        segments: List of semantic segments from segmentation models.
        segment_index: Fast lookup index for segments by ID.
        topology: Topology graph (entity references).
        embeddings: Neural embeddings from ll_stepnet (optional).
        bounding_box: Overall bounding box for the document.
        processing_history: History of processing steps.
    """

    name: str
    format: Optional[InputFormat] = None
    origin: Optional[CADDocumentOrigin] = None
    hash: Optional[str] = None

    # Content
    items: List[CADItem] = Field(default_factory=list)

    # Segmentation data (NEW)
    segments: List[Segment] = Field(default_factory=list)
    segment_index: Dict[str, Segment] = Field(default_factory=dict)

    # CAD-specific data
    topology: Optional[TopologyGraph] = None
    embeddings: Optional[List[List[float]]] = None  # ll_stepnet embeddings
    bounding_box: Optional[BoundingBox3D] = None

    # Metadata
    processing_history: List[ProcessingStep] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)  # Enrichment model results

    def add_item(self, item: CADItem):
        """Add a CAD item to the document.

        Args:
            item: CADItem to add.
        """
        self.items.append(item)
        _log.debug(f"Added {item.item_type} to document {self.name}")

    def add_segment(self, segment: Segment):
        """Add a semantic segment to the document.

        Segments are identified by segmentation models and represent
        meaningful geometric regions (features, components, etc.).

        Args:
            segment: Segment to add.

        Example:
            segment = Segment(
                segment_id="seg_hole_0",
                segment_type="feature",
                properties={"manufacturing_feature": "hole"}
            )
            doc.add_segment(segment)
        """
        self.segments.append(segment)
        self.segment_index[segment.segment_id] = segment
        _log.debug(
            f"Added segment {segment.segment_id} (type: {segment.segment_type}) "
            f"to document {self.name}"
        )

    def add_processing_step(
        self,
        step_name: str,
        component: str,
        duration_ms: Optional[float] = None,
        status: ConversionStatus = ConversionStatus.SUCCESS,
    ):
        """Add a processing step to history.

        Args:
            step_name: Name of the processing step.
            component: Component that executed this step.
            duration_ms: Duration in milliseconds.
            status: Status of this step.
        """
        self.processing_history.append(
            ProcessingStep(
                step_name=step_name,
                component=component,
                duration_ms=duration_ms,
                status=status,
            )
        )

    def export_to_json(self) -> Dict[str, Any]:
        """Export document to JSON format.

        Returns:
            Dictionary representation suitable for JSON serialization.

        Example:
            json_data = doc.export_to_json()
            with open("output.json", "w") as f:
                json.dump(json_data, f, indent=2)
        """
        return {
            "name": self.name,
            "format": self.format.value if self.format else None,
            "origin": self.origin.model_dump(mode='json') if self.origin else None,
            "hash": self.hash,
            "num_items": len(self.items),
            "items": [
                {
                    "type": item.item_type,
                    "label": item.label.text,
                    "text": item.text,
                    "bbox": item.bbox.model_dump(mode='json') if item.bbox else None,
                    "properties": item.properties,
                }
                for item in self.items
            ],
            "topology": {
                "num_nodes": self.topology.num_nodes,
                "num_edges": self.topology.num_edges,
                "adjacency_list": self.topology.adjacency_list,
            }
            if self.topology
            else None,
            "num_embeddings": len(self.embeddings) if self.embeddings else 0,
            "processing_history": [
                step.model_dump(mode='json') for step in self.processing_history
            ],
        }

    def export_to_markdown(self) -> str:
        """Export document to Markdown format.

        Returns:
            Markdown string representation.

        Example:
            markdown = doc.export_to_markdown()
            with open("output.md", "w") as f:
                f.write(markdown)
        """
        lines = []

        # Header
        lines.append(f"# {self.name}")
        lines.append("")

        if self.origin:
            lines.append("## Metadata")
            lines.append(f"- Format: {self.format.value if self.format else 'unknown'}")
            lines.append(f"- Filename: {self.origin.filename}")
            lines.append(f"- Hash: {self.hash[:16]}..." if self.hash else "")
            lines.append("")

        # Topology summary
        if self.topology:
            lines.append("## Topology")
            lines.append(f"- Nodes: {self.topology.num_nodes}")
            lines.append(f"- Edges: {self.topology.num_edges}")
            lines.append("")

        # Items
        lines.append("## Items")
        lines.append(f"Total items: {len(self.items)}")
        lines.append("")

        for i, item in enumerate(self.items):
            lines.append(f"### Item {i + 1}: {item.label.text}")
            lines.append(f"- Type: {item.item_type}")
            if item.text:
                lines.append(f"- Text: {item.text[:100]}...")
            if item.bbox:
                lines.append(f"- Bounding Box: {item.bbox.center}")
            if item.properties:
                lines.append("- Properties:")
                for key, value in item.properties.items():
                    lines.append(f"  - {key}: {value}")
            lines.append("")

        return "\n".join(lines)


class ErrorItem(BaseModel):
    """Error information for conversion failures.

    Attributes:
        component: Component where error occurred.
        error_message: Error message.
        item_id: ID of item that caused error (optional).
    """

    component: str
    error_message: str
    item_id: Optional[str] = None


class CADInputDocument(BaseModel):
    """Input document descriptor.

    Attributes:
        file: Path to the input file.
        format: Detected or specified format.
        document_hash: Hash of file content.
    """

    file: Path
    format: InputFormat
    document_hash: str
    _backend: Optional[Any] = None  # Backend instance (set during conversion)


class ConversionResult(BaseModel):
    """Result of a CAD file conversion.

    Similar to docling's ConversionResult but adapted for CAD conversion.

    Attributes:
        input: Input document descriptor.
        document: Converted CADlingDocument (if successful).
        status: Conversion status.
        errors: List of errors encountered.
    """

    model_config = {"arbitrary_types_allowed": True}

    input: CADInputDocument
    document: Optional[CADlingDocument] = None
    status: ConversionStatus = ConversionStatus.SUCCESS
    errors: List[ErrorItem] = Field(default_factory=list)

    def add_error(self, component: str, error_message: str, item_id: Optional[str] = None):
        """Add an error to the result.

        Args:
            component: Component where error occurred.
            error_message: Error message.
            item_id: ID of item that caused error.
        """
        self.errors.append(
            ErrorItem(
                component=component,
                error_message=error_message,
                item_id=item_id,
            )
        )
        _log.error(f"[{component}] {error_message}")
