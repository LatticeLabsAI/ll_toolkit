"""Reusable bridge for cadling → geotoken tokenization.

Centralises all geotoken tokenizer interactions so that any cadling pipeline
(SegNet, STL, STEP, Generation, SDG) can tokenize a
:class:`~cadling.datamodel.base_models.CADlingDocument` through a single
entry-point.

The module follows the **lazy-import** pattern established by
``segnet_pipeline._try_import_geotoken()`` and
``step/stepnet_integration.py``.  When geotoken is not installed every public
method degrades gracefully and returns ``None`` or an empty result.

Classes:
    GeoTokenResult: Dataclass holding per-item token sequences + vocabulary.
    GeoTokenIntegration: Stateless (or config-driven) bridge.

Example::

    from cadling.backend.geotoken_integration import GeoTokenIntegration

    bridge = GeoTokenIntegration()
    if bridge.available:
        result = bridge.tokenize_document(doc)
        encoded = bridge.encode_sequences(result.token_sequences)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cadling.datamodel.base_models import (
        CADItem,
        CADlingDocument,
        TopologyGraph,
    )

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


def _try_import_tokenizers() -> tuple:
    """Lazily import geotoken tokenizer components.

    Returns:
        Tuple of (GeoTokenizer, GraphTokenizer, CommandSequenceTokenizer,
        CADVocabulary, TokenSequence) or tuple of Nones.
    """
    try:
        from geotoken.tokenizer.geo_tokenizer import GeoTokenizer
        from geotoken.tokenizer.graph_tokenizer import GraphTokenizer
        from geotoken.tokenizer.command_tokenizer import (
            CommandSequenceTokenizer,
        )
        from geotoken.tokenizer.vocabulary import CADVocabulary
        from geotoken.tokenizer.token_types import TokenSequence

        return (
            GeoTokenizer,
            GraphTokenizer,
            CommandSequenceTokenizer,
            CADVocabulary,
            TokenSequence,
        )
    except ImportError:
        _log.debug("geotoken not available; tokenization disabled")
        return (None,) * 5


def _try_import_vertex_tools() -> tuple:
    """Lazily import geotoken vertex validation and clustering.

    Returns:
        Tuple of (VertexValidator, VertexClusterer, VertexMerger) or Nones.
    """
    try:
        from geotoken.vertex import (
            VertexValidator,
            VertexClusterer,
            VertexMerger,
        )

        return VertexValidator, VertexClusterer, VertexMerger
    except ImportError:
        _log.debug("geotoken.vertex not available")
        return None, None, None


def _try_import_impact_analyzer() -> Any:
    """Lazily import geotoken QuantizationImpactAnalyzer.

    Returns:
        QuantizationImpactAnalyzer class or None.
    """
    try:
        from geotoken.impact.analyzer import QuantizationImpactAnalyzer

        return QuantizationImpactAnalyzer
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class GeoTokenResult:
    """Result of tokenising a :class:`CADlingDocument`.

    Attributes:
        token_sequences: Mapping of item/key → geotoken ``TokenSequence``.
            The special key ``"__graph__"`` holds the topology-graph
            token sequence (if generated).
        vocabulary: The :class:`CADVocabulary` instance used for encoding.
        metadata: Timing, counts, and error summary.
        errors: Per-item error messages (item_id → message).
    """

    token_sequences: dict[str, Any] = field(default_factory=dict)
    vocabulary: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main bridge
# ---------------------------------------------------------------------------


class GeoTokenIntegration:
    """Reusable bridge: CADlingDocument → geotoken TokenSequences.

    Extracts geometry from cadling data-model objects using their native
    export methods (``MeshItem.to_numpy()``, ``TopologyGraph.to_edge_index()``,
    ``Sketch2DItem.to_geotoken_commands()``) and calls the appropriate
    geotoken tokenizers.

    The class is intentionally **stateless** — every call re-imports lazily
    and constructs lightweight tokeniser instances on the fly so it is safe
    to share across threads or call sites.

    Args:
        config: Optional dict of overrides forwarded to tokenizer constructors.
            Supported keys:

            * ``source_format`` (str): Passed to ``CommandSequenceTokenizer``.
              Default ``"auto"``.
            * ``include_constraints`` (bool): Whether to tokenize geometric
              constraints alongside command sequences.  Default ``False``.

    Example::

        bridge = GeoTokenIntegration()
        if bridge.available:
            result = bridge.tokenize_document(doc)
            encoded = bridge.encode_sequences(result.token_sequences)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

        # Probe availability once
        imports = _try_import_tokenizers()
        self._GeoTokenizer = imports[0]
        self._GraphTokenizer = imports[1]
        self._CommandSequenceTokenizer = imports[2]
        self._CADVocabulary = imports[3]
        self._TokenSequence = imports[4]

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """Whether the geotoken package is installed and importable."""
        return self._GeoTokenizer is not None

    # ------------------------------------------------------------------
    # Document-level tokenization
    # ------------------------------------------------------------------

    def tokenize_document(
        self,
        doc: CADlingDocument,
        *,
        include_mesh: bool = True,
        include_graph: bool = True,
        include_commands: bool = True,
        include_constraints: bool = False,
    ) -> GeoTokenResult:
        """Tokenize an entire :class:`CADlingDocument` in one call.

        Iterates over the document's items and topology graph, calling
        the appropriate geotoken tokenizer for each data kind.

        Args:
            doc: A segmented/enriched :class:`CADlingDocument`.
            include_mesh: Tokenize mesh items via ``GeoTokenizer``.
            include_graph: Tokenize the topology graph via ``GraphTokenizer``.
            include_commands: Tokenize command sequences via
                ``CommandSequenceTokenizer``.
            include_constraints: Include geometric constraint tokens
                alongside command sequences.

        Returns:
            :class:`GeoTokenResult` with per-item token sequences,
            vocabulary, and metadata.
        """
        if not self.available:
            _log.warning("geotoken not available; returning empty result")
            return GeoTokenResult(
                metadata={"error": "geotoken not installed"},
            )

        start = time.time()

        token_sequences: dict[str, Any] = {}
        errors: dict[str, str] = {}
        counts = {
            "mesh_tokenized": 0,
            "graph_tokenized": 0,
            "command_tokenized": 0,
            "items_skipped": 0,
        }

        # --- Mesh tokenization ---
        if include_mesh:
            geo_tokenizer = self._GeoTokenizer()
            for item in doc.items:
                item_id = item.item_id or f"item_{id(item)}"
                mesh_data = self._extract_mesh_data(item)
                if mesh_data is not None:
                    try:
                        vertices, faces = mesh_data
                        ts = geo_tokenizer.tokenize(
                            vertices=vertices, faces=faces,
                        )
                        token_sequences[item_id] = ts
                        counts["mesh_tokenized"] += 1
                    except Exception as exc:
                        msg = f"Mesh tokenization failed: {exc}"
                        _log.warning("%s for %s", msg, item_id)
                        errors[item_id] = msg
                        counts["items_skipped"] += 1

        # --- Graph tokenization ---
        if include_graph and doc.topology is not None:
            try:
                graph_ts = self._tokenize_topology(doc.topology)
                if graph_ts is not None:
                    token_sequences["__graph__"] = graph_ts
                    counts["graph_tokenized"] += 1
            except Exception as exc:
                msg = f"Graph tokenization failed: {exc}"
                _log.warning(msg)
                errors["__graph__"] = msg

        # --- Command sequence tokenization ---
        if include_commands:
            use_constraints = include_constraints or self._config.get(
                "include_constraints", False,
            )
            cmd_tokenizer = self._CommandSequenceTokenizer()
            for item in doc.items:
                item_id = item.item_id or f"item_{id(item)}"
                commands = self._extract_commands(item)
                if commands is not None:
                    try:
                        constraints = (
                            self._extract_constraints(item)
                            if use_constraints
                            else None
                        )
                        cmd_ts = cmd_tokenizer.tokenize(
                            construction_history=commands,
                            constraints=constraints,
                        )
                        key = f"{item_id}__cmd"
                        token_sequences[key] = cmd_ts
                        counts["command_tokenized"] += 1
                    except Exception as exc:
                        msg = f"Command tokenization failed: {exc}"
                        _log.warning("%s for %s", msg, item_id)
                        errors[item_id] = msg
                        counts["items_skipped"] += 1

        # Build vocabulary
        vocabulary = self._create_vocabulary()

        duration_ms = (time.time() - start) * 1000

        _log.info(
            "Tokenization complete in %.1fms: %d mesh, %d graph, "
            "%d command, %d skipped",
            duration_ms,
            counts["mesh_tokenized"],
            counts["graph_tokenized"],
            counts["command_tokenized"],
            counts["items_skipped"],
        )

        return GeoTokenResult(
            token_sequences=token_sequences,
            vocabulary=vocabulary,
            metadata={"duration_ms": duration_ms, **counts},
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Per-item tokenizers
    # ------------------------------------------------------------------

    def tokenize_mesh(
        self,
        item: CADItem,
        return_metadata: bool = False,
    ) -> Any | tuple[Any, dict] | None:
        """Tokenize a single mesh item.

        Extracts ``(vertices, faces)`` via ``item.to_numpy()`` and
        runs :class:`GeoTokenizer.tokenize`.

        Args:
            item: A CADItem (typically :class:`MeshItem`) with mesh data.
            return_metadata: If True, return ``(result, metadata)`` tuple.

        Returns:
            A geotoken ``TokenSequence`` or ``None`` if extraction fails.
            If ``return_metadata=True``, returns ``(result, metadata)`` tuple.
        """
        metadata = {"degraded": False, "method": "geotoken", "warning": None}

        if not self.available:
            metadata["degraded"] = True
            metadata["warning"] = "geotoken not available"
            if return_metadata:
                return None, metadata
            return None

        mesh_data = self._extract_mesh_data(item)
        if mesh_data is None:
            metadata["degraded"] = True
            metadata["warning"] = "no mesh data available"
            if return_metadata:
                return None, metadata
            return None

        vertices, faces = mesh_data

        try:
            geo_tokenizer = self._GeoTokenizer()
            result = geo_tokenizer.tokenize(vertices=vertices, faces=faces)
            if return_metadata:
                return result, metadata
            return result
        except Exception as exc:
            _log.warning("tokenize_mesh failed: %s", exc)
            metadata["degraded"] = True
            metadata["warning"] = str(exc)
            if return_metadata:
                return None, metadata
            return None

    def tokenize_topology(
        self,
        topology: TopologyGraph,
        return_metadata: bool = False,
    ) -> Any | tuple[Any, dict] | None:
        """Tokenize a topology graph.

        Extracts node features, edge index, and edge features via the
        topology's ``to_numpy_*()`` methods and runs
        :class:`GraphTokenizer.tokenize`.

        Args:
            topology: A :class:`TopologyGraph` from a CADlingDocument.
            return_metadata: If True, return ``(result, metadata)`` tuple.

        Returns:
            A geotoken ``TokenSequence`` or ``None``.
            If ``return_metadata=True``, returns ``(result, metadata)`` tuple.
        """
        metadata = {"degraded": False, "method": "geotoken", "warning": None}

        if not self.available:
            metadata["degraded"] = True
            metadata["warning"] = "geotoken not available"
            if return_metadata:
                return None, metadata
            return None

        try:
            result = self._tokenize_topology(topology)
            if result is None:
                metadata["degraded"] = True
                metadata["warning"] = "topology tokenization returned None"
            if return_metadata:
                return result, metadata
            return result
        except Exception as exc:
            _log.warning("tokenize_topology failed: %s", exc)
            metadata["degraded"] = True
            metadata["warning"] = str(exc)
            if return_metadata:
                return None, metadata
            return None

    def tokenize_sketch(
        self,
        item: CADItem,
        *,
        include_constraints: bool = False,
        return_metadata: bool = False,
    ) -> Any | tuple[Any, dict] | None:
        """Tokenize a sketch item's command sequence.

        Extracts commands via ``item.to_geotoken_commands()`` and
        optionally constraints via ``item.to_geotoken_constraints()``.

        Args:
            item: A CADItem (typically :class:`Sketch2DItem`).
            include_constraints: Whether to parse geometric constraints.
            return_metadata: If True, return ``(result, metadata)`` tuple.

        Returns:
            A geotoken ``TokenSequence`` or ``None``.
            If ``return_metadata=True``, returns ``(result, metadata)`` tuple.
        """
        metadata = {"degraded": False, "method": "geotoken", "warning": None}

        if not self.available:
            metadata["degraded"] = True
            metadata["warning"] = "geotoken not available"
            if return_metadata:
                return None, metadata
            return None

        commands = self._extract_commands(item)
        if commands is None:
            metadata["degraded"] = True
            metadata["warning"] = "no command sequence available"
            if return_metadata:
                return None, metadata
            return None

        try:
            cmd_tokenizer = self._CommandSequenceTokenizer()
            constraints = (
                self._extract_constraints(item) if include_constraints else None
            )
            result = cmd_tokenizer.tokenize(
                construction_history=commands, constraints=constraints,
            )
            if return_metadata:
                return result, metadata
            return result
        except Exception as exc:
            _log.warning("tokenize_sketch failed: %s", exc)
            metadata["degraded"] = True
            metadata["warning"] = str(exc)
            if return_metadata:
                return None, metadata
            return None

    def tokenize_with_embeddings(
        self,
        item: CADItem,
        embeddings: list[list[float]] | Any | None = None,
        return_metadata: bool = False,
    ) -> Any | tuple[Any, dict] | None:
        """Tokenize mesh with ll_stepnet embeddings.

        The embeddings are quantized using :class:`FeatureVectorQuantizer`
        and stored in the ``TokenSequence.metadata`` for downstream use.

        This method provides integration between ll_stepnet neural embeddings
        and geotoken's quantization system, allowing embeddings to be
        serialized alongside geometric tokens.

        Args:
            item: CADItem with mesh data.
            embeddings: Per-vertex or per-face embeddings from ll_stepnet.
                Can be a list of float lists or a numpy array.
            return_metadata: If True, return ``(result, metadata)`` tuple.

        Returns:
            ``TokenSequence`` with embedding tokens in metadata, or ``None``.
            If ``return_metadata=True``, returns ``(result, metadata)`` tuple.
        """
        import numpy as np

        metadata = {
            "degraded": False,
            "method": "geotoken",
            "warning": None,
            "embeddings_included": False,
        }

        if not self.available:
            metadata["degraded"] = True
            metadata["warning"] = "geotoken not available"
            if return_metadata:
                return None, metadata
            return None

        mesh_data = self._extract_mesh_data(item)
        if mesh_data is None:
            metadata["degraded"] = True
            metadata["warning"] = "no mesh data available"
            if return_metadata:
                return None, metadata
            return None

        vertices, faces = mesh_data

        try:
            # Import FeatureVectorQuantizer for embeddings
            from geotoken.quantization.feature_quantizer import (
                FeatureVectorQuantizer,
            )

            geo_tokenizer = self._GeoTokenizer()
            ts = geo_tokenizer.tokenize(vertices=vertices, faces=faces)

            # Quantize embeddings if provided
            if embeddings is not None:
                emb_array = np.asarray(embeddings, dtype=np.float32)
                quantizer = FeatureVectorQuantizer(bits=8)
                params = quantizer.fit(emb_array)
                quantized_emb = quantizer.quantize(emb_array, params)

                # Store in token sequence metadata
                ts.metadata["embeddings"] = quantized_emb.tolist()
                ts.metadata["embedding_params"] = {
                    "dim": params.dim,
                    "min_vals": params.min_vals.tolist(),
                    "max_vals": params.max_vals.tolist(),
                }
                metadata["embeddings_included"] = True

            if return_metadata:
                return ts, metadata
            return ts

        except Exception as exc:
            _log.warning("tokenize_with_embeddings failed: %s", exc)
            metadata["degraded"] = True
            metadata["warning"] = str(exc)
            if return_metadata:
                return None, metadata
            return None

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------

    def encode_sequences(
        self,
        token_sequences: dict[str, Any],
    ) -> dict[str, list[int]]:
        """Encode token sequences to integer IDs via :class:`CADVocabulary`.

        Args:
            token_sequences: Mapping of item_id → ``TokenSequence``.

        Returns:
            Mapping of item_id → list of integer token IDs.
            Items that fail encoding are silently skipped.
        """
        vocab = self._create_vocabulary()
        if vocab is None:
            _log.warning("No vocabulary available; cannot encode")
            return {}

        encoded: dict[str, list[int]] = {}
        for item_id, ts in token_sequences.items():
            try:
                encoded[item_id] = vocab.encode_full_sequence(ts)
            except Exception as exc:
                _log.warning(
                    "Failed to encode sequence for %s: %s", item_id, exc,
                )

        _log.info(
            "Encoded %d/%d token sequences",
            len(encoded),
            len(token_sequences),
        )
        return encoded

    def decode_token_ids(
        self,
        token_ids: list[int],
    ) -> Any | None:
        """Decode integer IDs back to a geotoken ``TokenSequence``.

        Reverses the encoding performed by :meth:`encode_sequences`.
        Useful in the generation pipeline where a model predicts token
        IDs that need conversion back to geometry.

        The method first decodes IDs → ``list[CommandToken]`` via
        :meth:`CADVocabulary.decode`, then wraps them in a
        ``TokenSequence`` for downstream consumers.

        Args:
            token_ids: Flat list of integer token IDs.

        Returns:
            A geotoken ``TokenSequence`` or ``None``.
        """
        vocab = self._create_vocabulary()
        if vocab is None:
            return None

        try:
            command_tokens = vocab.decode(token_ids)

            # Wrap in a TokenSequence
            if self._TokenSequence is not None:
                ts = self._TokenSequence()
                ts.command_tokens = command_tokens
                return ts

            # If TokenSequence class unavailable, return raw list
            return command_tokens
        except Exception as exc:
            _log.warning("Token ID decoding failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Roundtrip validation
    # ------------------------------------------------------------------

    def validate_roundtrip(
        self,
        item: CADItem,
    ) -> dict[str, Any]:
        """Tokenize → detokenize a mesh and report quality metrics.

        Runs the full quantization roundtrip and reports Hausdorff
        distance, vertex validation, and feature-loss metrics.

        Args:
            item: A CADItem (typically :class:`MeshItem`) with mesh data.

        Returns:
            Dict of quality metrics.  Empty dict if validation is not
            possible (missing data, missing geotoken.impact, etc.).
        """
        if not self.available:
            return {}

        mesh_data = self._extract_mesh_data(item)
        if mesh_data is None:
            return {}

        vertices, faces = mesh_data
        report: dict[str, Any] = {}

        # Impact analysis (Hausdorff distance, relationship preservation)
        ImpactAnalyzer = _try_import_impact_analyzer()
        if ImpactAnalyzer is not None:
            try:
                analyzer = ImpactAnalyzer()
                impact = analyzer.analyze(vertices, faces)
                report["hausdorff_distance"] = impact.hausdorff_distance
                report["mean_error"] = impact.mean_error
                report["max_error"] = impact.max_error
                report["relationship_preservation"] = (
                    impact.relationship_preservation_rate
                )
            except Exception as exc:
                _log.debug("Impact analysis failed: %s", exc)

        # Vertex validation
        VertexValidator, _, _ = _try_import_vertex_tools()
        if VertexValidator is not None:
            try:
                val_report = VertexValidator().validate(vertices, faces)
                report["vertex_validation"] = {
                    "valid": val_report.valid,
                    "errors": val_report.errors,
                    "warnings": val_report.warnings,
                }
            except Exception as exc:
                _log.debug("Vertex validation failed: %s", exc)

        return report

    # ------------------------------------------------------------------
    # Vocabulary helper
    # ------------------------------------------------------------------

    def _create_vocabulary(self) -> Any | None:
        """Instantiate a :class:`CADVocabulary` or return ``None``."""
        if self._CADVocabulary is None:
            return None
        try:
            return self._CADVocabulary()
        except Exception as exc:
            _log.warning("Failed to create CADVocabulary: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Internal: topology tokenization
    # ------------------------------------------------------------------

    def _tokenize_topology(self, topology: TopologyGraph) -> Any | None:
        """Tokenize a :class:`TopologyGraph` via :class:`GraphTokenizer`.

        Args:
            topology: Topology graph with node/edge feature arrays.

        Returns:
            ``TokenSequence`` or ``None``.
        """
        graph_tokenizer = self._GraphTokenizer()
        node_features = topology.to_numpy_node_features()
        edge_index = topology.to_edge_index()
        edge_features = topology.to_numpy_edge_features()

        return graph_tokenizer.tokenize(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
        )

    # ------------------------------------------------------------------
    # Geometry extraction helpers
    # ------------------------------------------------------------------
    # These are intentionally static so they can be called without an
    # instance (e.g. by SegNetPipeline during the refactor transition).

    @staticmethod
    def _extract_mesh_data(
        item: CADItem,
    ) -> tuple[Any, Any] | None:
        """Extract ``(vertices, faces)`` from a CADItem if available.

        Checks for ``to_numpy()`` method (:class:`MeshItem`) or mesh
        data stored in ``item.properties``.

        Returns:
            Tuple of ``(vertices, faces)`` numpy arrays or ``None``.
        """
        if hasattr(item, "to_numpy"):
            try:
                return item.to_numpy()
            except Exception as exc:
                _log.debug(
                    "to_numpy() failed for %s: %s",
                    getattr(item, "item_id", "?"),
                    exc,
                )

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
        item: CADItem,
    ) -> list[dict[str, Any]] | None:
        """Extract command sequence from a CADItem if available.

        Checks for ``to_geotoken_commands()`` method
        (:class:`Sketch2DItem`) or command data in
        ``item.properties['command_sequence']``.

        Returns:
            List of command dicts or ``None``.
        """
        if hasattr(item, "to_geotoken_commands"):
            try:
                return item.to_geotoken_commands()
            except Exception as exc:
                _log.debug(
                    "to_geotoken_commands() failed for %s: %s",
                    getattr(item, "item_id", "?"),
                    exc,
                )

        props = getattr(item, "properties", {})
        if "command_sequence" in props:
            return props["command_sequence"]

        return None

    @staticmethod
    def _extract_constraints(
        item: CADItem,
    ) -> list[dict[str, Any]] | None:
        """Extract geometric constraints from a CADItem if available.

        Checks for ``to_geotoken_constraints()`` method or constraint
        data in ``item.properties['constraints']``.

        Returns:
            List of constraint dicts or ``None``.
        """
        if hasattr(item, "to_geotoken_constraints"):
            try:
                return item.to_geotoken_constraints()
            except Exception as exc:
                _log.debug(
                    "to_geotoken_constraints() failed for %s: %s",
                    getattr(item, "item_id", "?"),
                    exc,
                )

        props = getattr(item, "properties", {})
        if "constraints" in props:
            return props["constraints"]

        return None
