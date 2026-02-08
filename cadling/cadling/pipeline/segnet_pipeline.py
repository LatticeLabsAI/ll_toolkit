"""SegNet pipeline: Segmentation → Tokenization (IR) → Reconstruction.

Three-stage orchestrator pipeline that wires together cadling's segmentation
models, geotoken's tokenizers, and cadling's reconstruction backend into a
single end-to-end flow:

1. **Segment** — Parse CAD file and apply segmentation models
   (BRepSegmentationModel / MeshSegmentationModel) via the standard
   Build → Assemble → Enrich stages inherited from :class:`BaseCADPipeline`.

2. **Tokenize** — Convert the segmented document into discrete token
   sequences (the *intermediate representation*).  Delegates to
   :class:`~cadling.backend.geotoken_integration.GeoTokenIntegration`
   which centralises all geotoken interactions for reuse by other
   pipelines.

3. **Reconstruct** — Execute token sequences through
   :class:`CommandExecutor` to rebuild validated CAD geometry.

The pipeline can be used standalone or composed with
:class:`GenerationPipeline` when the generation stage produces token
sequences that need reconstruction.

Example::

    from cadling.pipeline.segnet_pipeline import SegNetPipeline
    from cadling.datamodel.pipeline_options import PipelineOptions

    options = PipelineOptions(do_topology_analysis=True)
    pipeline = SegNetPipeline(options)

    # Full pipeline from a pre-parsed document
    result = pipeline.process_document(document)
    print(result.metrics)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from cadling.pipeline.base_pipeline import BaseCADPipeline

if TYPE_CHECKING:
    from cadling.datamodel.base_models import (
        CADItem,
        CADlingDocument,
        ConversionResult,
    )
    from cadling.datamodel.pipeline_options import PipelineOptions

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports — keep heavy dependencies optional
# ---------------------------------------------------------------------------



def _try_import_command_executor():
    """Lazily import cadling CommandExecutor.

    Returns:
        CommandExecutor class or None.
    """
    try:
        from cadling.generation.reconstruction.command_executor import (
            CommandExecutor,
        )
        return CommandExecutor
    except (ImportError, AttributeError):
        _log.debug("CommandExecutor not available; reconstruction disabled")
        return None


def _try_import_vertex_tools():
    """Lazily import geotoken vertex validation and clustering.

    Returns:
        Tuple of (VertexValidator, VertexClusterer, VertexMerger) or Nones.
    """
    try:
        from geotoken.vertex import VertexValidator, VertexClusterer, VertexMerger
        return VertexValidator, VertexClusterer, VertexMerger
    except ImportError:
        _log.debug("geotoken.vertex not available; vertex post-processing disabled")
        return None, None, None


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass
class SegNetTokenizationResult:
    """Result of the tokenization (IR) stage.

    Attributes:
        token_sequences: Mapping of item ID → TokenSequence.
            Each :class:`TokenSequence` may contain command tokens,
            graph tokens, and/or constraint tokens.
        vocabulary: The :class:`CADVocabulary` used for encoding.
        segment_map: Mapping of segment ID → list of item IDs
            that belong to that segment.
        metadata: Extra information (timing, counts, etc.).
    """
    token_sequences: Dict[str, Any] = field(default_factory=dict)
    vocabulary: Optional[Any] = None
    segment_map: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReconstructionResult:
    """Result of reconstructing a single item from tokens.

    Attributes:
        item_id: ID of the source item.
        success: Whether reconstruction succeeded.
        shape: Reconstructed geometry (OCC shape or trimesh) or ``None``.
        errors: List of error messages.
        validation_report: Optional dict of validation metrics.
    """
    item_id: str
    success: bool
    shape: Optional[Any] = None
    errors: List[str] = field(default_factory=list)
    validation_report: Optional[Dict[str, Any]] = None


@dataclass
class SegNetPipelineResult:
    """Full pipeline result from Segment → Tokenize → Reconstruct.

    Attributes:
        document: Segmented :class:`CADlingDocument`.
        tokenization: Tokenization stage output.
        reconstructions: Per-item reconstruction results.
        metrics: Aggregate quality metrics (timing, roundtrip fidelity).
    """
    document: Optional[Any] = None
    tokenization: Optional[SegNetTokenizationResult] = None
    reconstructions: List[ReconstructionResult] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SegNetPipeline
# ---------------------------------------------------------------------------


class SegNetPipeline(BaseCADPipeline):
    """Segmentation → Tokenization → Reconstruction pipeline.

    Inherits the Build → Assemble → Enrich workflow from
    :class:`BaseCADPipeline` for the segmentation stage, then adds two
    additional stages:

    - :meth:`tokenize` — convert segmented document to token sequences.
    - :meth:`reconstruct` — rebuild geometry from token IDs.

    The full flow is available via :meth:`process_document`.

    Args:
        pipeline_options: Standard pipeline configuration.
        include_graph_tokens: Whether to tokenize the topology graph.
        include_mesh_tokens: Whether to tokenize mesh geometry.
        include_command_tokens: Whether to tokenize command sequences.
        include_constraints: Whether to include constraint tokens.
        reconstruction_tolerance: Geometric tolerance for reconstruction.
        vertex_merge_distance: Distance threshold for vertex merging
            during post-processing (0 to disable).

    Example::

        options = PipelineOptions(
            enrichment_models=[BRepSegmentationModel(artifacts_path)],
            do_topology_analysis=True,
        )
        pipeline = SegNetPipeline(
            options,
            include_graph_tokens=True,
            include_command_tokens=True,
        )
        result = pipeline.process_document(document)
    """

    def __init__(
        self,
        pipeline_options: "PipelineOptions",
        include_graph_tokens: bool = True,
        include_mesh_tokens: bool = True,
        include_command_tokens: bool = True,
        include_constraints: bool = False,
        reconstruction_tolerance: float = 1e-6,
        vertex_merge_distance: float = 0.0,
    ) -> None:
        super().__init__(pipeline_options)
        self.include_graph_tokens = include_graph_tokens
        self.include_mesh_tokens = include_mesh_tokens
        self.include_command_tokens = include_command_tokens
        self.include_constraints = include_constraints
        self.reconstruction_tolerance = reconstruction_tolerance
        self.vertex_merge_distance = vertex_merge_distance

        _log.info(
            "SegNetPipeline: graph=%s, mesh=%s, commands=%s, "
            "constraints=%s, recon_tol=%.1e, merge_dist=%.1e",
            include_graph_tokens,
            include_mesh_tokens,
            include_command_tokens,
            include_constraints,
            reconstruction_tolerance,
            vertex_merge_distance,
        )

    # ------------------------------------------------------------------
    # BaseCADPipeline overrides (Stage 1: Segmentation)
    # ------------------------------------------------------------------

    @classmethod
    def get_default_options(cls) -> "PipelineOptions":
        """Return default options for the SegNet pipeline."""
        from cadling.datamodel.pipeline_options import PipelineOptions

        return PipelineOptions(
            do_topology_analysis=True,
            device="cpu",
        )

    def _build_document(self, conv_res: "ConversionResult") -> "ConversionResult":
        """Build: Parse CAD file via its attached backend.

        Delegates to the backend's ``convert()`` method, which is the
        standard cadling pattern used by STEPPipeline / STLPipeline.

        Args:
            conv_res: Conversion result with input document attached.

        Returns:
            Updated conversion result with document populated.
        """
        try:
            backend = conv_res.input._backend
            if backend is None:
                raise ValueError(
                    "No backend attached to input document. "
                    "Attach a STEPBackend or STLBackend before calling execute()."
                )

            _log.debug("SegNetPipeline: calling backend.convert()")
            document = backend.convert()
            conv_res.document = document

            _log.info(
                "Built document: %d items, topology=%s",
                len(document.items),
                document.topology is not None,
            )
            return conv_res

        except Exception as exc:
            _log.exception("Build stage failed: %s", exc)
            conv_res.add_error(
                component="SegNetPipeline._build_document",
                error_message=f"Failed to build document: {exc}",
            )
            raise

    def _assemble_document(self, conv_res: "ConversionResult") -> "ConversionResult":
        """Assemble: Prepare document for segmentation.

        For the SegNet pipeline, the assemble stage ensures that the
        document has the data structures needed by segmentation models
        (topology graph, mesh items, etc.).

        Args:
            conv_res: Conversion result to assemble.

        Returns:
            Updated conversion result.
        """
        if not conv_res.document:
            return conv_res

        doc = conv_res.document

        # Verify topology graph is present if requested
        if self.pipeline_options.do_topology_analysis and doc.topology is None:
            _log.warning(
                "do_topology_analysis is True but document has no topology graph. "
                "Graph tokenization will be skipped."
            )

        # Log segment state before enrichment (segmentation) runs
        _log.debug(
            "Pre-segmentation: %d items, %d existing segments",
            len(doc.items),
            len(doc.segments),
        )

        return conv_res

    # ------------------------------------------------------------------
    # Stage 2: Tokenization (Intermediate Representation)
    # ------------------------------------------------------------------

    def tokenize(
        self,
        doc: "CADlingDocument",
    ) -> SegNetTokenizationResult:
        """Convert a segmented document to token sequences.

        Delegates to :class:`~cadling.backend.geotoken_integration.GeoTokenIntegration`
        which centralises all geotoken interactions so other pipelines
        can reuse the same logic.

        **Step-by-step**:

        1. Instantiate the geotoken bridge (lazy imports).
        2. Call ``bridge.tokenize_document()`` with the pipeline flags.
        3. Build the segment → item mapping.
        4. Wrap the result in a :class:`SegNetTokenizationResult`.

        Args:
            doc: Segmented :class:`CADlingDocument`.

        Returns:
            :class:`SegNetTokenizationResult` containing per-item
            token sequences and metadata.
        """
        from cadling.backend.geotoken_integration import GeoTokenIntegration

        bridge = GeoTokenIntegration()

        if not bridge.available:
            _log.warning("geotoken not available; returning empty tokenization")
            return SegNetTokenizationResult(
                metadata={"error": "geotoken not installed"},
            )

        # Delegate tokenization to the bridge
        gt_result = bridge.tokenize_document(
            doc,
            include_mesh=self.include_mesh_tokens,
            include_graph=self.include_graph_tokens,
            include_commands=self.include_command_tokens,
            include_constraints=self.include_constraints,
        )

        # Build segment → item mapping from document segments
        segment_map: Dict[str, List[str]] = {}
        for seg in doc.segments:
            segment_map[seg.segment_id] = list(seg.item_ids)

        return SegNetTokenizationResult(
            token_sequences=gt_result.token_sequences,
            vocabulary=gt_result.vocabulary,
            segment_map=segment_map,
            metadata=gt_result.metadata,
        )

    def encode(
        self,
        tokenization_result: SegNetTokenizationResult,
    ) -> Dict[str, List[int]]:
        """Encode token sequences to integer IDs via CADVocabulary.

        Takes the :class:`SegNetTokenizationResult` from :meth:`tokenize`
        and converts each :class:`TokenSequence` into a flat list of
        integer token IDs using the vocabulary's ``encode_full_sequence``
        method.

        Args:
            tokenization_result: Output from :meth:`tokenize`.

        Returns:
            Dict mapping item_id → list of integer token IDs.
        """
        if tokenization_result.vocabulary is None:
            _log.warning("No vocabulary available; cannot encode")
            return {}

        vocab = tokenization_result.vocabulary
        encoded: Dict[str, List[int]] = {}

        for item_id, token_seq in tokenization_result.token_sequences.items():
            try:
                ids = vocab.encode_full_sequence(token_seq)
                encoded[item_id] = ids
            except Exception as exc:
                _log.warning(
                    "Failed to encode sequence for %s: %s", item_id, exc
                )

        _log.info("Encoded %d/%d token sequences",
                   len(encoded), len(tokenization_result.token_sequences))
        return encoded

    # ------------------------------------------------------------------
    # Stage 3: Reconstruction
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        token_ids: Dict[str, List[int]],
        vocabulary: Optional[Any] = None,
    ) -> List[ReconstructionResult]:
        """Reconstruct geometry from integer token IDs.

        For each item's token ID list:

        1. Decode IDs back to a :class:`TokenSequence` via the vocabulary.
        2. Execute the token sequence through :class:`CommandExecutor`.
        3. Optionally post-process vertices (validate, cluster, merge).
        4. Return per-item :class:`ReconstructionResult`.

        Args:
            token_ids: Dict mapping item_id → list of integer token IDs.
            vocabulary: Optional vocabulary for decoding.  If ``None``,
                reconstruction assumes the executor accepts raw IDs.

        Returns:
            List of :class:`ReconstructionResult` objects.
        """
        CommandExecutor = _try_import_command_executor()
        results: List[ReconstructionResult] = []

        if CommandExecutor is None:
            _log.warning(
                "CommandExecutor not available; skipping reconstruction"
            )
            for item_id in token_ids:
                results.append(ReconstructionResult(
                    item_id=item_id,
                    success=False,
                    errors=["CommandExecutor not installed"],
                ))
            return results

        try:
            executor = CommandExecutor()
        except Exception as exc:
            _log.error("Failed to create CommandExecutor: %s", exc)
            for item_id in token_ids:
                results.append(ReconstructionResult(
                    item_id=item_id,
                    success=False,
                    errors=[f"CommandExecutor init failed: {exc}"],
                ))
            return results

        for item_id, ids in token_ids.items():
            result = self._reconstruct_single(
                item_id=item_id,
                token_ids=ids,
                executor=executor,
                vocabulary=vocabulary,
            )
            results.append(result)

        successes = sum(1 for r in results if r.success)
        _log.info(
            "Reconstruction: %d/%d successful", successes, len(results)
        )
        return results

    def _reconstruct_single(
        self,
        item_id: str,
        token_ids: List[int],
        executor: Any,
        vocabulary: Optional[Any] = None,
    ) -> ReconstructionResult:
        """Reconstruct a single item from its token IDs.

        Args:
            item_id: Identifier for this item.
            token_ids: Integer token IDs.
            executor: CommandExecutor instance.
            vocabulary: Optional vocabulary for decoding.

        Returns:
            :class:`ReconstructionResult` for this item.
        """
        errors: List[str] = []
        shape = None
        validation_report: Optional[Dict[str, Any]] = None

        try:
            # Execute tokens → geometry
            shape = executor.execute_tokens(token_ids)

            # Optional vertex post-processing
            if self.vertex_merge_distance > 0 and shape is not None:
                shape, vtx_report = self._postprocess_vertices(shape)
                if vtx_report:
                    validation_report = vtx_report

        except Exception as exc:
            errors.append(f"Reconstruction failed: {exc}")
            _log.warning(
                "Reconstruction failed for %s: %s", item_id, exc
            )

        return ReconstructionResult(
            item_id=item_id,
            success=shape is not None and len(errors) == 0,
            shape=shape,
            errors=errors,
            validation_report=validation_report,
        )

    def _postprocess_vertices(
        self,
        shape: Any,
    ) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """Apply vertex validation and clustering post-processing.

        Extracts vertices/faces from the shape (if it is a trimesh),
        runs validation, clusters near-duplicate vertices, merges them,
        and returns the cleaned shape.

        Args:
            shape: Reconstructed geometry (trimesh or OCC shape).

        Returns:
            Tuple of (processed_shape, validation_report_dict).
        """
        VertexValidator, VertexClusterer, VertexMerger = _try_import_vertex_tools()

        if VertexValidator is None:
            return shape, None

        report: Dict[str, Any] = {}

        try:
            import trimesh
            if not isinstance(shape, trimesh.Trimesh):
                return shape, None
        except ImportError:
            return shape, None

        import numpy as np
        vertices = np.array(shape.vertices, dtype=np.float32)
        faces = np.array(shape.faces, dtype=np.int64)

        # Step 1: Validate
        validator = VertexValidator()
        val_report = validator.validate(vertices, faces)
        report["validation"] = {
            "valid": val_report.valid,
            "errors": val_report.errors,
            "warnings": val_report.warnings,
        }

        # Step 2: Cluster and merge
        if self.vertex_merge_distance > 0 and VertexClusterer is not None:
            clusterer = VertexClusterer(
                merge_distance=self.vertex_merge_distance,
            )
            clustering = clusterer.cluster(vertices)
            merged_verts, clean_faces = VertexMerger.merge(
                vertices, faces, clustering,
            )
            report["clustering"] = {
                "original_vertices": len(vertices),
                "merged_vertices": len(merged_verts),
                "vertices_removed": clustering.num_merged,
            }

            # Rebuild trimesh
            try:
                shape = trimesh.Trimesh(
                    vertices=merged_verts,
                    faces=clean_faces,
                    process=False,
                )
            except Exception as exc:
                _log.warning("Failed to rebuild trimesh after merge: %s", exc)

        return shape, report

    # ------------------------------------------------------------------
    # Full pipeline: process_document
    # ------------------------------------------------------------------

    def process_document(
        self,
        doc: "CADlingDocument",
    ) -> SegNetPipelineResult:
        """Run the full Segment → Tokenize → Reconstruct pipeline.

        This is the main entry point when you already have a parsed
        :class:`CADlingDocument` (e.g. from a previous ``execute()`` call
        or from another pipeline).

        **Step-by-step**:

        1. Run enrichment models (segmentation) on the document.
        2. Tokenize the segmented document.
        3. Encode token sequences to integer IDs.
        4. Reconstruct geometry from token IDs.
        5. Compute roundtrip quality metrics.

        Args:
            doc: Parsed :class:`CADlingDocument`.

        Returns:
            :class:`SegNetPipelineResult` with all stage outputs.
        """
        start_time = time.time()
        metrics: Dict[str, float] = {}

        # --- Stage 1: Segmentation (enrichment) ---
        seg_start = time.time()
        if self.enrichment_pipe:
            _log.info(
                "Running %d segmentation/enrichment models",
                len(self.enrichment_pipe),
            )
            items = doc.items
            for model in self.enrichment_pipe:
                try:
                    model(doc, items)
                except Exception as exc:
                    _log.error("Enrichment model failed: %s", exc)
        metrics["segmentation_ms"] = (time.time() - seg_start) * 1000

        # --- Stage 2: Tokenization ---
        tok_start = time.time()
        tokenization = self.tokenize(doc)
        metrics["tokenization_ms"] = (time.time() - tok_start) * 1000
        metrics["num_token_sequences"] = float(
            len(tokenization.token_sequences)
        )

        # --- Stage 2b: Encoding ---
        enc_start = time.time()
        encoded = self.encode(tokenization)
        metrics["encoding_ms"] = (time.time() - enc_start) * 1000
        metrics["num_encoded"] = float(len(encoded))

        # --- Stage 3: Reconstruction ---
        recon_start = time.time()
        reconstructions = self.reconstruct(
            encoded, vocabulary=tokenization.vocabulary,
        )
        metrics["reconstruction_ms"] = (time.time() - recon_start) * 1000
        metrics["num_reconstructed"] = float(
            sum(1 for r in reconstructions if r.success)
        )

        metrics["total_ms"] = (time.time() - start_time) * 1000

        result = SegNetPipelineResult(
            document=doc,
            tokenization=tokenization,
            reconstructions=reconstructions,
            metrics=metrics,
        )

        _log.info(
            "SegNet pipeline complete in %.1fms: "
            "%d tokens, %d encoded, %d reconstructed",
            metrics["total_ms"],
            len(tokenization.token_sequences),
            len(encoded),
            int(metrics["num_reconstructed"]),
        )

        return result

    # ------------------------------------------------------------------
    # Geometry extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_mesh_data(
        item: "CADItem",
    ) -> Optional[Tuple[Any, Any]]:
        """Extract (vertices, faces) from a CADItem if available.

        Checks for ``to_numpy()`` method (MeshItem) or mesh data stored
        in ``item.properties``.

        Args:
            item: A :class:`CADItem` that may contain mesh data.

        Returns:
            Tuple of ``(vertices, faces)`` numpy arrays or ``None``.
        """
        # MeshItem has .to_numpy() → (vertices, faces)
        if hasattr(item, "to_numpy"):
            try:
                return item.to_numpy()
            except Exception as exc:
                _log.debug("to_numpy() failed for %s: %s", item.item_id, exc)

        # Fallback: check properties
        props = getattr(item, "properties", {})
        if "vertices" in props and "faces" in props:
            import numpy as np
            return (
                np.asarray(props["vertices"], dtype=np.float32),
                np.asarray(props["faces"], dtype=np.int64),
            )

        return None

    @staticmethod
    def _extract_commands(
        item: "CADItem",
    ) -> Optional[List[Dict[str, Any]]]:
        """Extract command sequence from a CADItem if available.

        Checks for ``to_geotoken_commands()`` method (Sketch2DItem) or
        command data stored in ``item.properties['command_sequence']``.

        Args:
            item: A :class:`CADItem` that may contain command data.

        Returns:
            List of command dicts or ``None``.
        """
        # Sketch2DItem has .to_geotoken_commands()
        if hasattr(item, "to_geotoken_commands"):
            try:
                return item.to_geotoken_commands()
            except Exception as exc:
                _log.debug(
                    "to_geotoken_commands() failed for %s: %s",
                    item.item_id, exc,
                )

        # Fallback: check properties
        props = getattr(item, "properties", {})
        if "command_sequence" in props:
            return props["command_sequence"]

        return None

    @staticmethod
    def _extract_constraints(
        item: "CADItem",
    ) -> Optional[List[Dict[str, Any]]]:
        """Extract geometric constraints from a CADItem if available.

        Checks for ``to_geotoken_constraints()`` method or constraint
        data in ``item.properties['constraints']``.

        Args:
            item: A :class:`CADItem` that may contain constraint data.

        Returns:
            List of constraint dicts or ``None``.
        """
        if hasattr(item, "to_geotoken_constraints"):
            try:
                return item.to_geotoken_constraints()
            except Exception as exc:
                _log.debug(
                    "to_geotoken_constraints() failed for %s: %s",
                    item.item_id, exc,
                )

        props = getattr(item, "properties", {})
        if "constraints" in props:
            return props["constraints"]

        return None
