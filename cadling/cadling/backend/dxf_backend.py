"""DXF 2D drawing backend.

This module provides a backend for parsing AutoCAD DXF files and extracting
2D geometric entities (lines, arcs, circles, polylines, etc.) as structured
Primitive2D models. The backend groups entities by layer into SketchProfiles
and returns a CADlingDocument containing Sketch2DItem instances.

DXF support is provided by the ``ezdxf`` library (optional dependency).

Data flow:
    DXF file → ezdxf.readfile() → modelspace entities
    → entity-to-Primitive2D mapping → group by layer
    → SketchProfile per layer → Sketch2DItem → CADlingDocument

Classes:
    DXFBackend: DeclarativeCADBackend for DXF 2D drawings.

Example:
    from cadling.backend.dxf_backend import DXFBackend
    from cadling.datamodel import CADInputDocument, InputFormat, DXFBackendOptions

    in_doc = CADInputDocument(
        file=Path("drawing.dxf"),
        format=InputFormat.DXF,
        document_hash="abc123",
    )
    options = DXFBackendOptions(target_layers=["0", "OUTLINE"])
    backend = DXFBackend(in_doc, Path("drawing.dxf"), options)
    if backend.is_valid():
        doc = backend.convert()
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from cadling.backend.abstract_backend import DeclarativeCADBackend
from cadling.datamodel.backend_options import BackendOptions, DXFBackendOptions
from cadling.datamodel.base_models import (
    CADDocumentOrigin,
    CADItem,
    CADItemLabel,
    CADlingDocument,
    InputFormat,
)
from cadling.datamodel.geometry_2d import (
    Arc2D,
    Circle2D,
    DimensionAnnotation,
    DimensionType,
    Ellipse2D,
    Line2D,
    Polyline2D,
    Primitive2D,
    Sketch2DItem,
    SketchProfile,
    Spline2D,
)

_log = logging.getLogger(__name__)

# Try to import ezdxf — it's an optional dependency
try:
    import ezdxf
    from ezdxf.entities import (
        Arc as DXFArc,
        Circle as DXFCircle,
        DXFGraphic,
        Ellipse as DXFEllipse,
        Line as DXFLine,
        LWPolyline as DXFLWPolyline,
        Spline as DXFSpline,
    )

    _HAS_EZDXF = True
except ImportError:
    _HAS_EZDXF = False
    _log.info(
        "ezdxf not available — DXF backend will not function. "
        "Install with: pip install ezdxf"
    )


class DXFBackend(DeclarativeCADBackend):
    """Backend for parsing AutoCAD DXF files into 2D sketch geometry.

    This backend reads DXF files using the ezdxf library and converts
    entities from the modelspace into structured Primitive2D models.
    Entities are grouped by layer into SketchProfiles, and the resulting
    document contains Sketch2DItem instances.

    The backend handles the following DXF entity types:
        - LINE → Line2D
        - ARC → Arc2D
        - CIRCLE → Circle2D
        - LWPOLYLINE → Polyline2D (with optional bulge→arc decomposition)
        - POLYLINE → Polyline2D
        - ELLIPSE → Ellipse2D
        - SPLINE → Spline2D
        - DIMENSION → DimensionAnnotation

    Block references (INSERT entities) are optionally inlined into
    modelspace coordinates when ``inline_blocks=True`` (the default).

    Attributes:
        _dxf_doc: Parsed ezdxf document (set after successful validation).
        _options: DXFBackendOptions controlling extraction behavior.
    """

    def __init__(
        self,
        in_doc: "CADInputDocument",
        path_or_stream: Union[Path, str, BytesIO],
        options: Optional[BackendOptions] = None,
    ):
        """Initialize DXF backend.

        Args:
            in_doc: Input document descriptor.
            path_or_stream: Path to DXF file or byte stream.
            options: DXFBackendOptions (defaults applied if None).

        Raises:
            ImportError: If ezdxf is not installed.
        """
        if not _HAS_EZDXF:
            raise ImportError(
                "ezdxf is required for DXF support. "
                "Install with: pip install 'cadling[drawings]'"
            )

        super().__init__(in_doc, path_or_stream, options)
        self._options: DXFBackendOptions = (
            options if isinstance(options, DXFBackendOptions)
            else DXFBackendOptions()
        )
        self._dxf_doc = None

    # ------------------------------------------------------------------
    # Class methods (interface contract)
    # ------------------------------------------------------------------

    @classmethod
    def supports_text_parsing(cls) -> bool:
        """DXF is a text-based format — always supports text parsing."""
        return True

    @classmethod
    def supports_rendering(cls) -> bool:
        """DXF backend does not support 3D rendering."""
        return False

    @classmethod
    def supported_formats(cls) -> Set[InputFormat]:
        """Return formats handled by this backend."""
        return {InputFormat.DXF}

    @classmethod
    def _get_default_options(cls) -> DXFBackendOptions:
        """Return default DXF backend options."""
        return DXFBackendOptions()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def is_valid(self) -> bool:
        """Validate that the input is a parseable DXF file.

        Attempts to read the file with ezdxf. If successful, caches the
        parsed document for use by ``convert()``.

        Returns:
            True if ezdxf can parse the file, False otherwise.
        """
        try:
            if isinstance(self.path_or_stream, BytesIO):
                self.path_or_stream.seek(0)
                self._dxf_doc = ezdxf.read(self.path_or_stream)
            else:
                self._dxf_doc = ezdxf.readfile(str(self.path_or_stream))

            _log.debug("DXF validation passed for %s", self.file.name)
            return True
        except Exception as exc:
            _log.warning("DXF validation failed for %s: %s", self.file.name, exc)
            self._dxf_doc = None
            return False

    # ------------------------------------------------------------------
    # Conversion (main entry point)
    # ------------------------------------------------------------------

    def convert(self) -> CADlingDocument:
        """Parse DXF file and convert to CADlingDocument.

        This method:
        1. Reads the DXF modelspace entities
        2. Converts each entity to a Primitive2D subclass
        3. Groups primitives by layer
        4. Builds a SketchProfile for each layer
        5. Creates Sketch2DItem instances
        6. Returns a fully populated CADlingDocument

        Returns:
            CADlingDocument containing Sketch2DItem instances with 2D geometry.

        Raises:
            RuntimeError: If file hasn't been validated or ezdxf parse fails.
        """
        # Ensure the file is loaded
        if self._dxf_doc is None:
            if not self.is_valid():
                raise RuntimeError(
                    f"Cannot convert invalid DXF file: {self.file.name}"
                )

        # Create document
        doc = CADlingDocument(
            name=self.file.name,
            format=InputFormat.DXF,
            origin=CADDocumentOrigin(
                filename=self.file.name,
                format=InputFormat.DXF,
                binary_hash=self.document_hash,
            ),
            hash=self.document_hash,
        )

        # Get modelspace
        msp = self._dxf_doc.modelspace()

        # Extract entities grouped by layer
        layer_primitives: Dict[str, List[Primitive2D]] = defaultdict(list)
        layer_annotations: Dict[str, List[DimensionAnnotation]] = defaultdict(list)
        entity_counts: Dict[str, int] = defaultdict(int)

        for entity in msp:
            entity_type = entity.dxftype()
            layer = entity.dxf.layer if hasattr(entity.dxf, "layer") else "0"

            # Apply layer filter if specified
            if (
                self._options.target_layers is not None
                and layer not in self._options.target_layers
            ):
                continue

            entity_counts[entity_type] += 1

            # Convert entity to Primitive2D or DimensionAnnotation
            if entity_type == "LINE":
                prim = self._convert_line(entity)
                if prim:
                    layer_primitives[layer].append(prim)

            elif entity_type == "ARC":
                prim = self._convert_arc(entity)
                if prim:
                    layer_primitives[layer].append(prim)

            elif entity_type == "CIRCLE":
                prim = self._convert_circle(entity)
                if prim:
                    layer_primitives[layer].append(prim)

            elif entity_type in ("LWPOLYLINE", "POLYLINE"):
                prim = self._convert_polyline(entity)
                if prim:
                    layer_primitives[layer].append(prim)

            elif entity_type == "ELLIPSE":
                prim = self._convert_ellipse(entity)
                if prim:
                    layer_primitives[layer].append(prim)

            elif entity_type == "SPLINE":
                prim = self._convert_spline(entity)
                if prim:
                    layer_primitives[layer].append(prim)

            elif entity_type == "DIMENSION" and self._options.extract_dimensions:
                annot = self._convert_dimension(entity)
                if annot:
                    layer_annotations[layer].append(annot)

            elif entity_type == "INSERT" and self._options.inline_blocks:
                # Inline block reference entities into the current layer
                block_prims, block_annots = self._inline_block(entity)
                layer_primitives[layer].extend(block_prims)
                layer_annotations[layer].extend(block_annots)

            else:
                _log.debug("Skipping unsupported DXF entity type: %s", entity_type)

        # Build SketchProfiles from layer groups
        all_layers = set(layer_primitives.keys()) | set(layer_annotations.keys())

        if self._options.merge_layers:
            # Merge all layers into a single profile
            all_prims = []
            all_annots = []
            for layer in sorted(all_layers):
                all_prims.extend(layer_primitives.get(layer, []))
                all_annots.extend(layer_annotations.get(layer, []))

            profile = SketchProfile(
                profile_id="merged_layers",
                primitives=all_prims,
                annotations=all_annots,
                closed=False,  # Let geometry extractor determine closure
            )
            profile.compute_bounds()

            item = Sketch2DItem(
                item_type="sketch_2d",
                label=CADItemLabel(text=f"{self.file.stem} - All Layers"),
                profiles=[profile],
                source_layer="merged",
            )
            doc.add_item(item)
        else:
            # One SketchProfile per layer
            for layer in sorted(all_layers):
                prims = layer_primitives.get(layer, [])
                annots = layer_annotations.get(layer, [])

                if not prims and not annots:
                    continue

                profile = SketchProfile(
                    profile_id=f"layer_{layer}",
                    primitives=prims,
                    annotations=annots,
                    closed=False,
                )
                profile.compute_bounds()

                item = Sketch2DItem(
                    item_type="sketch_2d",
                    label=CADItemLabel(text=f"{self.file.stem} - Layer {layer}"),
                    profiles=[profile],
                    source_layer=layer,
                )
                doc.add_item(item)

        # Store extraction metadata
        doc.properties["dxf_entity_counts"] = dict(entity_counts)
        doc.properties["dxf_layers"] = sorted(all_layers)
        doc.properties["total_primitives"] = sum(
            len(v) for v in layer_primitives.values()
        )
        doc.properties["total_annotations"] = sum(
            len(v) for v in layer_annotations.values()
        )

        _log.info(
            "Converted DXF %s: %d layers, %d primitives, %d annotations",
            self.file.name,
            len(all_layers),
            doc.properties["total_primitives"],
            doc.properties["total_annotations"],
        )

        return doc

    # ------------------------------------------------------------------
    # Entity conversion methods
    # ------------------------------------------------------------------

    def _get_entity_color(self, entity) -> Optional[Tuple[int, int, int]]:
        """Extract RGB color from a DXF entity.

        Args:
            entity: DXF entity with optional color attributes.

        Returns:
            (R, G, B) tuple or None if color cannot be determined.
        """
        try:
            if hasattr(entity.dxf, "true_color") and entity.dxf.true_color:
                tc = entity.dxf.true_color
                return ((tc >> 16) & 0xFF, (tc >> 8) & 0xFF, tc & 0xFF)
        except Exception:
            pass
        return None

    def _get_entity_handle(self, entity) -> Optional[str]:
        """Get the entity handle as a string identifier.

        Args:
            entity: DXF entity.

        Returns:
            Handle string or None.
        """
        try:
            return str(entity.dxf.handle) if hasattr(entity.dxf, "handle") else None
        except Exception:
            return None

    def _convert_line(self, entity) -> Optional[Line2D]:
        """Convert a DXF LINE entity to Line2D.

        Args:
            entity: DXF LINE entity.

        Returns:
            Line2D instance or None if conversion fails.
        """
        try:
            start = entity.dxf.start
            end = entity.dxf.end
            return Line2D(
                start=(float(start.x), float(start.y)),
                end=(float(end.x), float(end.y)),
                layer=entity.dxf.layer if hasattr(entity.dxf, "layer") else "0",
                color=self._get_entity_color(entity),
                source_entity_id=self._get_entity_handle(entity),
            )
        except Exception as exc:
            _log.debug("Failed to convert LINE entity: %s", exc)
            return None

    def _convert_arc(self, entity) -> Optional[Arc2D]:
        """Convert a DXF ARC entity to Arc2D.

        Args:
            entity: DXF ARC entity.

        Returns:
            Arc2D instance or None if conversion fails.
        """
        try:
            center = entity.dxf.center
            return Arc2D(
                center=(float(center.x), float(center.y)),
                radius=float(entity.dxf.radius),
                start_angle=float(entity.dxf.start_angle),
                end_angle=float(entity.dxf.end_angle),
                layer=entity.dxf.layer if hasattr(entity.dxf, "layer") else "0",
                color=self._get_entity_color(entity),
                source_entity_id=self._get_entity_handle(entity),
            )
        except Exception as exc:
            _log.debug("Failed to convert ARC entity: %s", exc)
            return None

    def _convert_circle(self, entity) -> Optional[Circle2D]:
        """Convert a DXF CIRCLE entity to Circle2D.

        Args:
            entity: DXF CIRCLE entity.

        Returns:
            Circle2D instance or None if conversion fails.
        """
        try:
            center = entity.dxf.center
            return Circle2D(
                center=(float(center.x), float(center.y)),
                radius=float(entity.dxf.radius),
                layer=entity.dxf.layer if hasattr(entity.dxf, "layer") else "0",
                color=self._get_entity_color(entity),
                source_entity_id=self._get_entity_handle(entity),
            )
        except Exception as exc:
            _log.debug("Failed to convert CIRCLE entity: %s", exc)
            return None

    def _convert_polyline(self, entity) -> Optional[Polyline2D]:
        """Convert a DXF LWPOLYLINE or POLYLINE entity to Polyline2D.

        Handles both lightweight polylines (LWPOLYLINE) and full polylines
        (POLYLINE with VERTEX sub-entities). Bulge values are preserved for
        arc-aware processing by the geometry extractor.

        Args:
            entity: DXF LWPOLYLINE or POLYLINE entity.

        Returns:
            Polyline2D instance or None if conversion fails.
        """
        try:
            entity_type = entity.dxftype()

            if entity_type == "LWPOLYLINE":
                # LWPOLYLINE stores (x, y, start_width, end_width, bulge) per vertex
                raw_points = list(entity.get_points(format="xyseb"))
                points = [(float(p[0]), float(p[1])) for p in raw_points]
                bulges = [float(p[4]) for p in raw_points]
                closed = entity.closed
            else:
                # POLYLINE with VERTEX sub-entities
                vertices = list(entity.vertices)
                points = [
                    (float(v.dxf.location.x), float(v.dxf.location.y))
                    for v in vertices
                ]
                bulges = [
                    float(v.dxf.bulge) if hasattr(v.dxf, "bulge") else 0.0
                    for v in vertices
                ]
                closed = entity.is_closed

            if len(points) < 2:
                return None

            return Polyline2D(
                points=points,
                closed=closed,
                bulges=bulges if any(b != 0 for b in bulges) else None,
                layer=entity.dxf.layer if hasattr(entity.dxf, "layer") else "0",
                color=self._get_entity_color(entity),
                source_entity_id=self._get_entity_handle(entity),
            )
        except Exception as exc:
            _log.debug("Failed to convert POLYLINE entity: %s", exc)
            return None

    def _convert_ellipse(self, entity) -> Optional[Ellipse2D]:
        """Convert a DXF ELLIPSE entity to Ellipse2D.

        Args:
            entity: DXF ELLIPSE entity.

        Returns:
            Ellipse2D instance or None if conversion fails.
        """
        try:
            center = entity.dxf.center
            major_axis = entity.dxf.major_axis
            return Ellipse2D(
                center=(float(center.x), float(center.y)),
                major_axis=(float(major_axis.x), float(major_axis.y)),
                ratio=float(entity.dxf.ratio),
                start_param=float(entity.dxf.start_param),
                end_param=float(entity.dxf.end_param),
                layer=entity.dxf.layer if hasattr(entity.dxf, "layer") else "0",
                color=self._get_entity_color(entity),
                source_entity_id=self._get_entity_handle(entity),
            )
        except Exception as exc:
            _log.debug("Failed to convert ELLIPSE entity: %s", exc)
            return None

    def _convert_spline(self, entity) -> Optional[Spline2D]:
        """Convert a DXF SPLINE entity to Spline2D.

        Args:
            entity: DXF SPLINE entity.

        Returns:
            Spline2D instance or None if conversion fails.
        """
        try:
            # ezdxf may return control points as Vec3 objects (with .x, .y)
            # or numpy arrays (with index-based access). Handle both.
            control_points = []
            for p in entity.control_points:
                try:
                    control_points.append((float(p.x), float(p.y)))
                except AttributeError:
                    control_points.append((float(p[0]), float(p[1])))
            knots = [float(k) for k in entity.knots]
            weights = (
                [float(w) for w in entity.weights]
                if entity.weights
                else None
            )

            if len(control_points) < 2:
                return None

            return Spline2D(
                control_points=control_points,
                degree=int(entity.dxf.degree),
                knots=knots,
                weights=weights,
                closed=bool(entity.closed),
                layer=entity.dxf.layer if hasattr(entity.dxf, "layer") else "0",
                color=self._get_entity_color(entity),
                source_entity_id=self._get_entity_handle(entity),
            )
        except Exception as exc:
            _log.debug("Failed to convert SPLINE entity: %s", exc)
            return None

    def _convert_dimension(self, entity) -> Optional[DimensionAnnotation]:
        """Convert a DXF DIMENSION entity to DimensionAnnotation.

        DXF DIMENSION entities carry measured values, dimension text, and
        attachment points for the dimension leaders.

        Args:
            entity: DXF DIMENSION entity.

        Returns:
            DimensionAnnotation instance or None if conversion fails.
        """
        try:
            # Determine dimension type from DXF dimtype
            dimtype_code = entity.dxf.dimtype if hasattr(entity.dxf, "dimtype") else 0
            dim_type_map = {
                0: DimensionType.LINEAR,    # Linear (horizontal/vertical/rotated)
                1: DimensionType.LINEAR,    # Aligned
                2: DimensionType.ANGULAR,   # Angular
                3: DimensionType.DIAMETER,  # Diameter
                4: DimensionType.RADIAL,    # Radius
                5: DimensionType.ANGULAR,   # Angular 3-point
                6: DimensionType.ORDINATE,  # Ordinate
            }
            dim_type = dim_type_map.get(dimtype_code & 0x0F, DimensionType.LINEAR)

            # Get measurement value
            value = 0.0
            if hasattr(entity, "measurement"):
                value = float(entity.measurement)

            # Get dimension text
            text = ""
            if hasattr(entity.dxf, "text") and entity.dxf.text:
                text = str(entity.dxf.text)

            # Get attachment points
            attachment_points = []
            if hasattr(entity.dxf, "defpoint"):
                p = entity.dxf.defpoint
                attachment_points.append((float(p.x), float(p.y)))
            if hasattr(entity.dxf, "defpoint2"):
                p = entity.dxf.defpoint2
                attachment_points.append((float(p.x), float(p.y)))
            if hasattr(entity.dxf, "defpoint3"):
                p = entity.dxf.defpoint3
                attachment_points.append((float(p.x), float(p.y)))

            layer = entity.dxf.layer if hasattr(entity.dxf, "layer") else "0"

            return DimensionAnnotation(
                dim_type=dim_type,
                value=value,
                text=text if text else f"{value:.3f}",
                attachment_points=attachment_points,
                layer=layer,
            )
        except Exception as exc:
            _log.debug("Failed to convert DIMENSION entity: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Block reference inlining
    # ------------------------------------------------------------------

    def _inline_block(
        self, insert_entity
    ) -> Tuple[List[Primitive2D], List[DimensionAnnotation]]:
        """Inline a block reference (INSERT entity) into modelspace.

        Transforms block entities into modelspace coordinates using the
        INSERT entity's position, rotation, and scale.

        Args:
            insert_entity: DXF INSERT entity referencing a block.

        Returns:
            Tuple of (primitives, annotations) from the inlined block.
        """
        primitives: List[Primitive2D] = []
        annotations: List[DimensionAnnotation] = []

        try:
            block_name = insert_entity.dxf.name
            if block_name not in self._dxf_doc.blocks:
                _log.debug("Block '%s' not found in DXF blocks", block_name)
                return primitives, annotations

            block = self._dxf_doc.blocks[block_name]
            insert_point = insert_entity.dxf.insert
            rotation = math.radians(
                insert_entity.dxf.rotation
                if hasattr(insert_entity.dxf, "rotation")
                else 0.0
            )
            x_scale = (
                insert_entity.dxf.xscale
                if hasattr(insert_entity.dxf, "xscale")
                else 1.0
            )
            y_scale = (
                insert_entity.dxf.yscale
                if hasattr(insert_entity.dxf, "yscale")
                else 1.0
            )

            # For simplicity, we extract entities from the block and transform
            # their coordinates. Full transformation is complex, so we handle
            # the common case (translation + uniform scale + rotation).
            for entity in block:
                entity_type = entity.dxftype()

                if entity_type == "LINE":
                    prim = self._convert_line(entity)
                    if prim:
                        prim.start = self._transform_point(
                            prim.start, insert_point, rotation, x_scale, y_scale
                        )
                        prim.end = self._transform_point(
                            prim.end, insert_point, rotation, x_scale, y_scale
                        )
                        primitives.append(prim)

                elif entity_type == "ARC":
                    prim = self._convert_arc(entity)
                    if prim:
                        prim.center = self._transform_point(
                            prim.center, insert_point, rotation, x_scale, y_scale
                        )
                        prim.radius *= (x_scale + y_scale) / 2  # Average scale
                        prim.start_angle += math.degrees(rotation)
                        prim.end_angle += math.degrees(rotation)
                        primitives.append(prim)

                elif entity_type == "CIRCLE":
                    prim = self._convert_circle(entity)
                    if prim:
                        prim.center = self._transform_point(
                            prim.center, insert_point, rotation, x_scale, y_scale
                        )
                        prim.radius *= (x_scale + y_scale) / 2
                        primitives.append(prim)

                elif entity_type in ("LWPOLYLINE", "POLYLINE"):
                    prim = self._convert_polyline(entity)
                    if prim:
                        prim.points = [
                            self._transform_point(
                                p, insert_point, rotation, x_scale, y_scale
                            )
                            for p in prim.points
                        ]
                        primitives.append(prim)

                elif entity_type == "DIMENSION" and self._options.extract_dimensions:
                    annot = self._convert_dimension(entity)
                    if annot:
                        annot.attachment_points = [
                            self._transform_point(
                                p, insert_point, rotation, x_scale, y_scale
                            )
                            for p in annot.attachment_points
                        ]
                        annotations.append(annot)

        except Exception as exc:
            _log.debug("Failed to inline block reference: %s", exc)

        return primitives, annotations

    @staticmethod
    def _transform_point(
        point: Tuple[float, float],
        insert: Any,
        rotation: float,
        x_scale: float,
        y_scale: float,
    ) -> Tuple[float, float]:
        """Transform a point from block-local to modelspace coordinates.

        Applies scale, rotation, then translation.

        Args:
            point: (x, y) point in block coordinates.
            insert: Insert point (ezdxf Vec3 or similar with .x, .y).
            rotation: Rotation angle in radians.
            x_scale: X scale factor.
            y_scale: Y scale factor.

        Returns:
            Transformed (x, y) point in modelspace coordinates.
        """
        # Scale
        x = point[0] * x_scale
        y = point[1] * y_scale

        # Rotate
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        rx = x * cos_r - y * sin_r
        ry = x * sin_r + y * cos_r

        # Translate
        return (rx + float(insert.x), ry + float(insert.y))
