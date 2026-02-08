"""Sequence annotator for paired (text, command_sequence) training data.

This module provides the SequenceAnnotator class that pairs multi-level text
annotations (from TextCADAnnotator) with tokenized CAD command sequences,
producing training data for Text-to-CAD generative models.

Supported input formats:
- STEP files with optional construction history
- DeepCAD JSON format (command sequences + sketches)
- Pre-tokenized command sequences

The output is JSONL files where each line contains:
    {
        "text_abstract": "...",
        "text_intermediate": "...",
        "text_detailed": "...",
        "text_expert": "...",
        "command_tokens": [int, int, ...],
        "metadata": {...}
    }

Classes:
    SequenceAnnotator: Pair text annotations with command token sequences

Example:
    from cadling.sdg.qa.text_cad_annotator import TextCADAnnotator
    from cadling.sdg.qa.sequence_annotator import SequenceAnnotator

    text_annotator = TextCADAnnotator(api_provider="openai")
    seq_annotator = SequenceAnnotator(text_annotator=text_annotator)

    # From STEP file
    result = seq_annotator.annotate("part.step")

    # Export for training
    seq_annotator.export_training_pairs(
        [result], "training_data.jsonl", format="jsonl"
    )
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from cadling.sdg.qa.base import AnnotationLevel

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy import: geotoken command tokenizer + vocabulary
# ---------------------------------------------------------------------------


def _try_import_geotoken_command() -> tuple:
    """Lazily import geotoken command tokenization.

    Returns:
        Tuple of (CommandSequenceTokenizer, CADVocabulary,
        BOS_TOKEN_ID, EOS_TOKEN_ID, SEP_TOKEN_ID, PAD_TOKEN_ID) or Nones.
    """
    try:
        from geotoken.tokenizer.command_tokenizer import (
            CommandSequenceTokenizer,
        )
        from geotoken.tokenizer.vocabulary import (
            CADVocabulary,
            BOS_TOKEN_ID,
            EOS_TOKEN_ID,
            SEP_TOKEN_ID,
            PAD_TOKEN_ID,
        )

        return (
            CommandSequenceTokenizer,
            CADVocabulary,
            BOS_TOKEN_ID,
            EOS_TOKEN_ID,
            SEP_TOKEN_ID,
            PAD_TOKEN_ID,
        )
    except ImportError:
        _log.debug("geotoken not available; using built-in tokenizer")
        return (None,) * 6


# ---------------------------------------------------------------------------
# Fallback: built-in DeepCAD command mapping
# ---------------------------------------------------------------------------

# DeepCAD command type mapping (used when geotoken is not installed)
_DEEPCAD_COMMAND_TYPES = {
    "Line": 0,
    "Arc": 1,
    "Circle": 2,
    "EXT": 3,       # Extrude
    "CUT": 4,       # Cut extrude
    "FILLET": 5,
    "CHAMFER": 6,
    "REV": 7,       # Revolve
    "SOL": 8,       # Solid boolean
    "SKETCH": 9,
    "CLOSE": 10,
    "END": 11,
}

# Special token IDs for command sequence framing (fallback values)
_CMD_PAD_ID = 0
_CMD_BOS_ID = 1       # Beginning of sequence
_CMD_EOS_ID = 2       # End of sequence
_CMD_SEP_ID = 3       # Separator between operations
_CMD_SKETCH_START = 4  # Start of sketch context
_CMD_SKETCH_END = 5    # End of sketch context
_CMD_EXTRUDE_START = 6
_CMD_EXTRUDE_END = 7
_CMD_OFFSET = 10       # Offset for command type tokens


class SequenceAnnotator:
    """Pair text annotations with tokenized CAD command sequences.

    Combines TextCADAnnotator output (multi-level descriptions) with
    tokenized CAD construction sequences to produce training data for
    text-conditioned CAD generation models.

    Attributes:
        text_annotator: TextCADAnnotator instance for generating descriptions
        command_tokenizer: Optional external tokenizer (STEPTokenizer or GeoTokenizer)

    Example:
        from cadling.sdg.qa.text_cad_annotator import TextCADAnnotator
        from cadling.sdg.qa.sequence_annotator import SequenceAnnotator

        annotator = SequenceAnnotator(
            text_annotator=TextCADAnnotator(api_provider="openai"),
        )
        result = annotator.annotate("bracket.step")
        print(result["text_abstract"])
        print(result["command_tokens"][:20])
    """

    def __init__(
        self,
        text_annotator: Any = None,
        command_tokenizer: Any = None,
    ):
        """Initialize SequenceAnnotator.

        Args:
            text_annotator: TextCADAnnotator instance for text annotation.
                If None, one will be created lazily on first use with
                default settings.
            command_tokenizer: External tokenizer for STEP command sequences.
                Accepts any object with an ``encode(text) -> list[int]``
                method (e.g., ``STEPTokenizer`` from ll_stepnet or
                ``GeoTokenizer`` from geotoken). If None, the built-in
                DeepCAD-style tokenizer is used for JSON inputs and the
                STEPTokenizer is used for STEP files.
        """
        self._text_annotator = text_annotator
        self.command_tokenizer = command_tokenizer

        # Try to initialise geotoken-based tokenization
        self._use_geotoken = False
        self._geotoken_tokenizer: Any = None
        self._geotoken_vocab: Any = None
        self._init_geotoken_tokenizer()

        tokenizer_label = "geotoken" if self._use_geotoken else "builtin"
        if command_tokenizer:
            tokenizer_label = "external"

        _log.info(
            "Initialized SequenceAnnotator "
            f"(text_annotator={'provided' if text_annotator else 'deferred'}, "
            f"tokenizer={tokenizer_label})"
        )

    def _init_geotoken_tokenizer(self) -> None:
        """Try to initialise geotoken command tokenizer and vocabulary.

        If geotoken is installed, creates a
        :class:`CommandSequenceTokenizer` and :class:`CADVocabulary`
        so that construction-history tokenization produces IDs from
        the unified geotoken vocabulary instead of the legacy
        ``_DEEPCAD_COMMAND_TYPES`` mapping.

        When geotoken is not available the flag ``_use_geotoken``
        stays ``False`` and all tokenization falls back to the
        built-in implementation.
        """
        imports = _try_import_geotoken_command()
        CmdTokenizer = imports[0]
        CadVocab = imports[1]

        if CmdTokenizer is None or CadVocab is None:
            return

        try:
            self._geotoken_tokenizer = CmdTokenizer()
            self._geotoken_vocab = CadVocab()
            self._use_geotoken = True
            _log.info(
                "geotoken command tokenizer available — using unified "
                "CADVocabulary (~%d tokens)",
                getattr(self._geotoken_vocab, "vocab_size", -1),
            )
        except Exception as exc:
            _log.warning("Failed to init geotoken tokenizer: %s", exc)
            self._use_geotoken = False

    # ------------------------------------------------------------------
    # Lazy accessor for text annotator
    # ------------------------------------------------------------------

    @property
    def text_annotator(self) -> Any:
        """Get or lazily create the TextCADAnnotator.

        Returns:
            TextCADAnnotator instance.
        """
        if self._text_annotator is None:
            from cadling.sdg.qa.text_cad_annotator import TextCADAnnotator

            self._text_annotator = TextCADAnnotator()
            _log.info("Lazily created default TextCADAnnotator")
        return self._text_annotator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate(
        self,
        step_file_path: str,
        construction_history: Optional[dict] = None,
        num_views: int = 4,
    ) -> dict:
        """Annotate a STEP file with paired (text, command_tokens) data.

        If ``construction_history`` is provided, it is tokenized to produce
        the command sequence. Otherwise the STEP file text is tokenized
        directly using the STEPTokenizer.

        Args:
            step_file_path: Path to the STEP file.
            construction_history: Optional construction history dict with
                keys like 'operations', 'sketches', 'extrusions'. If None
                the raw STEP text is tokenized instead.
            num_views: Number of rendered views for text annotation.

        Returns:
            Dictionary with keys:
                - 'text_abstract', 'text_intermediate',
                  'text_detailed', 'text_expert': Multi-level descriptions
                - 'command_tokens': List of integer token IDs
                - 'metadata': Provenance and statistics

        Raises:
            FileNotFoundError: If STEP file does not exist.
        """
        step_path = Path(step_file_path)
        if not step_path.exists():
            raise FileNotFoundError(f"STEP file not found: {step_path}")

        start_time = time.time()
        annotation_id = str(uuid.uuid4())

        _log.info(f"Annotating '{step_path.name}' with command sequence")

        # Generate text annotations via TextCADAnnotator
        text_result = self.text_annotator.annotate(
            str(step_path), num_views=num_views
        )

        # Generate command tokens
        if construction_history is not None:
            command_tokens = self._tokenize_construction_history(
                construction_history
            )
            tokenizer_type = "construction_history"
        else:
            command_tokens = self._tokenize_step_file(str(step_path))
            tokenizer_type = "step_tokenizer"

        elapsed = time.time() - start_time

        result = {
            "text_abstract": text_result.get(AnnotationLevel.ABSTRACT.value, ""),
            "text_intermediate": text_result.get(
                AnnotationLevel.INTERMEDIATE.value, ""
            ),
            "text_detailed": text_result.get(AnnotationLevel.DETAILED.value, ""),
            "text_expert": text_result.get(AnnotationLevel.EXPERT.value, ""),
            "command_tokens": command_tokens,
            "vlm_description": text_result.get("vlm_description", ""),
            "metadata": {
                "annotation_id": annotation_id,
                "source_file": str(step_path),
                "source_name": step_path.name,
                "tokenizer_type": tokenizer_type,
                "num_command_tokens": len(command_tokens),
                "num_views": num_views,
                "vlm_model": text_result.get("metadata", {}).get(
                    "vlm_model", ""
                ),
                "llm_model": text_result.get("metadata", {}).get(
                    "llm_model", ""
                ),
                "time_seconds": round(elapsed, 3),
            },
        }

        _log.info(
            f"Sequence annotation complete for '{step_path.name}': "
            f"{len(command_tokens)} command tokens, {elapsed:.2f}s"
        )

        return result

    def annotate_from_deepcad_json(
        self,
        json_path: str,
        num_views: int = 4,
    ) -> dict:
        """Annotate from a DeepCAD JSON file.

        DeepCAD JSON files contain command sequences (sketch profiles,
        extrusions, booleans) that can be directly tokenized. Text
        annotations are generated either from:
        - An associated STEP file (if ``step_file`` key exists in JSON), or
        - Rendered views of the shape described by the JSON commands.

        Args:
            json_path: Path to the DeepCAD JSON file.
            num_views: Number of views for text annotation (if STEP
                file is available).

        Returns:
            Dictionary with paired text and command tokens.

        Raises:
            FileNotFoundError: If JSON file does not exist.
            ValueError: If JSON format is invalid.
        """
        json_file = Path(json_path)
        if not json_file.exists():
            raise FileNotFoundError(f"JSON file not found: {json_file}")

        start_time = time.time()
        annotation_id = str(uuid.uuid4())

        _log.info(f"Annotating DeepCAD JSON: '{json_file.name}'")

        # Parse DeepCAD JSON
        with open(json_file, "r") as f:
            deepcad_data = json.load(f)

        # Validate minimal structure
        if not isinstance(deepcad_data, dict):
            raise ValueError(
                f"Expected JSON object, got {type(deepcad_data).__name__}"
            )

        # Tokenize command sequence from JSON
        command_tokens = self._tokenize_deepcad_json(deepcad_data)

        # Generate text annotations
        step_file = deepcad_data.get("step_file")
        if step_file and Path(step_file).exists():
            # Use associated STEP file for rendering + VLM
            text_result = self.text_annotator.annotate(
                step_file, num_views=num_views
            )
        else:
            # Generate text description from the JSON command structure
            text_result = self._text_from_deepcad_commands(deepcad_data)

        elapsed = time.time() - start_time

        result = {
            "text_abstract": text_result.get(AnnotationLevel.ABSTRACT.value, ""),
            "text_intermediate": text_result.get(
                AnnotationLevel.INTERMEDIATE.value, ""
            ),
            "text_detailed": text_result.get(AnnotationLevel.DETAILED.value, ""),
            "text_expert": text_result.get(AnnotationLevel.EXPERT.value, ""),
            "command_tokens": command_tokens,
            "vlm_description": text_result.get("vlm_description", ""),
            "metadata": {
                "annotation_id": annotation_id,
                "source_file": str(json_file),
                "source_name": json_file.name,
                "source_format": "deepcad_json",
                "tokenizer_type": "deepcad",
                "num_command_tokens": len(command_tokens),
                "num_operations": len(
                    deepcad_data.get("sequence", deepcad_data.get("ops", []))
                ),
                "time_seconds": round(elapsed, 3),
            },
        }

        _log.info(
            f"DeepCAD annotation complete: {len(command_tokens)} tokens, "
            f"{elapsed:.2f}s"
        )

        return result

    def annotate_batch(
        self,
        file_pairs: list[dict],
        num_views: int = 4,
    ) -> list[dict]:
        """Batch-annotate multiple files.

        Each entry in ``file_pairs`` should be a dict with at least one of:
        - 'step_file': Path to a STEP file
        - 'json_file': Path to a DeepCAD JSON file
        - 'construction_history': Dict of construction operations

        Args:
            file_pairs: List of dicts describing files to annotate.
            num_views: Number of views per file.

        Returns:
            List of annotation result dictionaries.
        """
        results: list[dict] = []
        total = len(file_pairs)

        _log.info(f"Starting batch sequence annotation of {total} files")
        batch_start = time.time()

        for idx, pair in enumerate(file_pairs):
            _log.info(f"[{idx + 1}/{total}] Processing pair")

            try:
                if "json_file" in pair:
                    result = self.annotate_from_deepcad_json(
                        pair["json_file"], num_views=num_views
                    )
                elif "step_file" in pair:
                    result = self.annotate(
                        pair["step_file"],
                        construction_history=pair.get("construction_history"),
                        num_views=num_views,
                    )
                else:
                    _log.warning(
                        f"Pair {idx} has no 'step_file' or 'json_file' key, "
                        "skipping"
                    )
                    continue

                results.append(result)

            except Exception as e:
                error_msg = f"Failed to annotate pair {idx}: {e}"
                _log.error(error_msg)
                results.append({
                    "text_abstract": "",
                    "text_intermediate": "",
                    "text_detailed": "",
                    "text_expert": "",
                    "command_tokens": [],
                    "vlm_description": "",
                    "metadata": {
                        "annotation_id": str(uuid.uuid4()),
                        "source_file": pair.get(
                            "step_file", pair.get("json_file", "unknown")
                        ),
                        "error": str(e),
                    },
                })

        batch_elapsed = time.time() - batch_start
        num_ok = sum(
            1 for r in results if "error" not in r.get("metadata", {})
        )

        _log.info(
            f"Batch sequence annotation complete: {num_ok}/{total} succeeded, "
            f"{batch_elapsed:.2f}s total"
        )

        return results

    def export_training_pairs(
        self,
        annotations: list[dict],
        output_path: str,
        format: str = "jsonl",
    ) -> None:
        """Export annotated pairs as training data.

        Writes each annotation as a single JSONL line containing the four
        text levels plus the command token sequence.

        Args:
            annotations: List of annotation dicts from annotate() or
                annotate_batch().
            output_path: Path for the output file.
            format: Output format. Currently supports 'jsonl' and 'json'.

        Raises:
            ValueError: If format is not supported.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        valid_formats = ("jsonl", "json")
        if format not in valid_formats:
            raise ValueError(
                f"Unsupported format: '{format}'. Use one of {valid_formats}"
            )

        # Filter out failed annotations (empty command tokens)
        valid_annotations = [
            ann for ann in annotations
            if ann.get("command_tokens") and any(
                ann.get(f"text_{level.value}", "")
                for level in AnnotationLevel
            )
        ]

        skipped = len(annotations) - len(valid_annotations)
        if skipped > 0:
            _log.warning(
                f"Skipping {skipped} annotations with empty tokens or text"
            )

        if format == "jsonl":
            self._export_jsonl(valid_annotations, output)
        elif format == "json":
            self._export_json(valid_annotations, output)

        _log.info(
            f"Exported {len(valid_annotations)} training pairs to "
            f"{output} ({format})"
        )

    # ------------------------------------------------------------------
    # STEP tokenization
    # ------------------------------------------------------------------

    def _tokenize_step_file(self, step_path: str) -> list[int]:
        """Tokenize a STEP file into command token IDs.

        Uses the external command_tokenizer if provided, otherwise
        falls back to STEPTokenizer from ll_stepnet.

        Args:
            step_path: Path to the STEP file.

        Returns:
            List of integer token IDs.
        """
        # If external tokenizer provided, use it
        if self.command_tokenizer is not None:
            return self._tokenize_with_external(step_path)

        # Fallback: use STEPTokenizer from ll_stepnet
        try:
            from stepnet.tokenizer import STEPTokenizer

            tokenizer = STEPTokenizer()
        except ImportError:
            _log.warning(
                "STEPTokenizer not available from ll_stepnet, "
                "using basic STEP entity tokenization"
            )
            return self._tokenize_step_basic(step_path)

        # Read STEP file content
        with open(step_path, "r", errors="ignore") as f:
            step_text = f.read()

        # Tokenize the STEP text
        token_ids = tokenizer.encode(step_text)

        _log.debug(
            f"Tokenized STEP file '{Path(step_path).name}': "
            f"{len(token_ids)} tokens"
        )

        return token_ids

    def _tokenize_with_external(self, step_path: str) -> list[int]:
        """Tokenize using the externally provided tokenizer.

        The tokenizer must have an ``encode`` method accepting a string
        and returning a list of ints.

        Args:
            step_path: Path to the STEP file.

        Returns:
            List of integer token IDs.
        """
        with open(step_path, "r", errors="ignore") as f:
            step_text = f.read()

        # Support both encode(text) -> list[int]
        # and tokenize(vertices, faces) -> TokenSequence patterns
        if hasattr(self.command_tokenizer, "encode"):
            token_ids = self.command_tokenizer.encode(step_text)
        elif hasattr(self.command_tokenizer, "tokenize"):
            # GeoTokenizer pattern: needs vertices/faces, not raw text
            # Fall back to basic tokenization
            _log.warning(
                "External tokenizer has tokenize() but not encode(); "
                "using basic STEP tokenization as fallback"
            )
            return self._tokenize_step_basic(step_path)
        else:
            raise TypeError(
                f"command_tokenizer {type(self.command_tokenizer).__name__} "
                "must have an 'encode' or 'tokenize' method"
            )

        if isinstance(token_ids, dict):
            # batch_encode returns dict with 'token_ids' key
            token_ids = token_ids.get("token_ids", [[]])[0]

        return list(token_ids)

    def _tokenize_step_basic(self, step_path: str) -> list[int]:
        """Basic STEP tokenization when no external tokenizer is available.

        Extracts entity types and hashes them to token IDs.

        Args:
            step_path: Path to the STEP file.

        Returns:
            List of integer token IDs.
        """
        with open(step_path, "r", errors="ignore") as f:
            content = f.read()

        token_ids: list[int] = [_CMD_BOS_ID]

        # Extract STEP entity lines
        entity_pattern = re.compile(
            r"#(\d+)\s*=\s*([A-Z_]+)\(([^;]*)\);", re.DOTALL
        )

        for match in entity_pattern.finditer(content):
            entity_id = int(match.group(1))
            entity_type = match.group(2)
            entity_args = match.group(3)

            # Encode entity type as token
            type_token = _CMD_OFFSET + (hash(entity_type) % 49990)
            token_ids.append(type_token)

            # Encode numeric parameters
            numbers = re.findall(
                r"-?\d+\.?\d*(?:[Ee][+-]?\d+)?", entity_args
            )
            for num_str in numbers[:8]:  # Limit parameters per entity
                try:
                    val = float(num_str)
                    # Quantize to integer token range
                    quantized = int(val * 1000) % 50000
                    token_ids.append(quantized)
                except ValueError:
                    pass

            # Add separator between entities
            token_ids.append(_CMD_SEP_ID)

        token_ids.append(_CMD_EOS_ID)

        _log.debug(
            f"Basic STEP tokenization: {len(token_ids)} tokens from "
            f"'{Path(step_path).name}'"
        )

        return token_ids

    # ------------------------------------------------------------------
    # Construction history tokenization
    # ------------------------------------------------------------------

    def _tokenize_construction_history(
        self,
        history: dict,
    ) -> list[int]:
        """Tokenize a construction history dictionary into command tokens.

        Expected history format::

            {
                "operations": [
                    {"type": "sketch", "plane": "XY", "profiles": [...]},
                    {"type": "extrude", "distance": 50.0, "direction": "positive"},
                    {"type": "fillet", "radius": 2.0, "edges": [...]},
                    ...
                ]
            }

        Tokenization priority:

        1. **External** ``command_tokenizer`` (if provided).
        2. **geotoken** ``CommandSequenceTokenizer`` + ``CADVocabulary``
           (if geotoken is installed).
        3. **Built-in** DeepCAD-style fallback mapping.

        Args:
            history: Construction history dictionary.

        Returns:
            List of integer token IDs.
        """
        # --- Priority 1: external tokenizer ---
        if self.command_tokenizer is not None and hasattr(
            self.command_tokenizer, "encode"
        ):
            history_text = json.dumps(history, separators=(",", ":"))
            token_ids = self.command_tokenizer.encode(history_text)
            if isinstance(token_ids, dict):
                token_ids = token_ids.get("token_ids", [[]])[0]
            return list(token_ids)

        # --- Priority 2: geotoken command tokenizer ---
        if self._use_geotoken:
            return self._tokenize_history_via_geotoken(history)

        # --- Priority 3: built-in fallback ---
        operations = history.get("operations", history.get("ops", []))
        token_ids: list[int] = [_CMD_BOS_ID]

        for op in operations:
            op_type = op.get("type", "unknown").upper()

            # Map operation type to token
            type_id = _DEEPCAD_COMMAND_TYPES.get(
                op_type, hash(op_type) % 256
            )
            token_ids.append(_CMD_OFFSET + type_id)

            # Encode operation parameters
            param_tokens = self._encode_operation_params(op)
            token_ids.extend(param_tokens)

            # Separator
            token_ids.append(_CMD_SEP_ID)

        token_ids.append(_CMD_EOS_ID)

        _log.debug(
            f"Tokenized construction history: {len(operations)} ops -> "
            f"{len(token_ids)} tokens"
        )

        return token_ids

    def _tokenize_history_via_geotoken(self, history: dict) -> list[int]:
        """Tokenize construction history using geotoken's vocabulary.

        Converts the history dict into the command-list format expected
        by :class:`CommandSequenceTokenizer`, tokenizes, then encodes the
        resulting :class:`TokenSequence` to integer IDs via
        :class:`CADVocabulary`.

        Args:
            history: Construction history dict with ``operations`` key.

        Returns:
            List of integer token IDs from the unified CADVocabulary.
        """
        operations = history.get("operations", history.get("ops", []))

        # Build a list of command dicts in geotoken format
        commands: list[dict[str, Any]] = []
        for op in operations:
            cmd: dict[str, Any] = {"type": op.get("type", "unknown")}

            # Carry through all numeric / list parameters
            for key, val in op.items():
                if key == "type":
                    continue
                cmd[key] = val

            commands.append(cmd)

        try:
            # Tokenize commands → TokenSequence
            token_seq = self._geotoken_tokenizer.tokenize(construction_history=commands)

            # Encode TokenSequence → integer IDs
            encoded = self._geotoken_vocab.encode_full_sequence(token_seq)

            _log.debug(
                "geotoken tokenized construction history: %d ops -> "
                "%d tokens",
                len(operations),
                len(encoded),
            )
            return encoded

        except Exception as exc:
            _log.warning(
                "geotoken tokenization failed, falling back to builtin: %s",
                exc,
            )
            # Fall back to built-in
            return self._tokenize_history_builtin(history)

    def _tokenize_history_builtin(self, history: dict) -> list[int]:
        """Built-in DeepCAD-style construction history tokenization.

        Called as a fallback when geotoken tokenization fails.

        Args:
            history: Construction history dict.

        Returns:
            List of integer token IDs.
        """
        operations = history.get("operations", history.get("ops", []))
        token_ids: list[int] = [_CMD_BOS_ID]

        for op in operations:
            op_type = op.get("type", "unknown").upper()
            type_id = _DEEPCAD_COMMAND_TYPES.get(
                op_type, hash(op_type) % 256
            )
            token_ids.append(_CMD_OFFSET + type_id)
            param_tokens = self._encode_operation_params(op)
            token_ids.extend(param_tokens)
            token_ids.append(_CMD_SEP_ID)

        token_ids.append(_CMD_EOS_ID)
        return token_ids

    def _encode_operation_params(self, operation: dict) -> list[int]:
        """Encode operation parameters as integer tokens.

        Handles common CAD operation parameters: distances, angles,
        radii, point coordinates, boolean flags.

        Args:
            operation: Single operation dictionary.

        Returns:
            List of parameter token IDs.
        """
        tokens: list[int] = []
        op_type = operation.get("type", "").lower()

        # Extract numeric parameters
        numeric_keys = [
            "distance", "radius", "angle", "depth", "width", "height",
            "offset", "taper_angle", "draft_angle",
        ]
        for key in numeric_keys:
            if key in operation:
                val = operation[key]
                if isinstance(val, (int, float)):
                    # Quantize: multiply by 100 and take modulo
                    quantized = int(abs(val) * 100) % 50000
                    # Sign bit
                    if val < 0:
                        quantized = quantized | 0x8000
                    tokens.append(quantized)

        # Encode point coordinates if present
        for point_key in ("center", "start", "end", "origin", "point"):
            point = operation.get(point_key)
            if isinstance(point, (list, tuple)):
                for coord in point[:3]:
                    if isinstance(coord, (int, float)):
                        quantized = int(coord * 100) % 50000
                        tokens.append(quantized)

        # Encode direction/axis
        direction = operation.get("direction", operation.get("axis"))
        if isinstance(direction, str):
            dir_map = {
                "positive": 1, "negative": 2,
                "x": 3, "y": 4, "z": 5,
                "+x": 3, "+y": 4, "+z": 5,
                "-x": 6, "-y": 7, "-z": 8,
            }
            dir_token = dir_map.get(direction.lower(), 0)
            if dir_token:
                tokens.append(dir_token)
        elif isinstance(direction, (list, tuple)):
            for coord in direction[:3]:
                if isinstance(coord, (int, float)):
                    quantized = int(coord * 100) % 50000
                    tokens.append(quantized)

        # Boolean/enum parameters
        boolean_keys = [
            "symmetric", "through_all", "reversed", "midplane",
        ]
        for key in boolean_keys:
            if key in operation:
                tokens.append(1 if operation[key] else 0)

        # Sketch profiles (nested structure)
        if op_type == "sketch" and "profiles" in operation:
            profile_tokens = self._encode_sketch_profiles(
                operation["profiles"]
            )
            tokens.extend(profile_tokens)

        return tokens

    def _encode_sketch_profiles(
        self,
        profiles: list,
    ) -> list[int]:
        """Encode sketch profile curves into tokens.

        Args:
            profiles: List of sketch profile definitions, each containing
                curves with type and parameters.

        Returns:
            List of profile token IDs.
        """
        tokens: list[int] = [_CMD_SKETCH_START]

        for profile in profiles:
            curves = profile if isinstance(profile, list) else profile.get(
                "curves", profile.get("segments", [])
            )

            for curve in curves:
                if isinstance(curve, dict):
                    curve_type = curve.get("type", "Line")
                    type_id = _DEEPCAD_COMMAND_TYPES.get(curve_type, 0)
                    tokens.append(_CMD_OFFSET + type_id)

                    # Encode curve parameters (start/end points, radius, etc.)
                    for key in ("start", "end", "center", "mid"):
                        point = curve.get(key)
                        if isinstance(point, (list, tuple)):
                            for coord in point[:2]:  # 2D sketch
                                if isinstance(coord, (int, float)):
                                    quantized = int(coord * 100) % 50000
                                    tokens.append(quantized)

                    if "radius" in curve:
                        r = curve["radius"]
                        if isinstance(r, (int, float)):
                            tokens.append(int(abs(r) * 100) % 50000)

            tokens.append(_CMD_SEP_ID)

        tokens.append(_CMD_SKETCH_END)
        return tokens

    # ------------------------------------------------------------------
    # DeepCAD JSON tokenization
    # ------------------------------------------------------------------

    def _tokenize_deepcad_json(self, data: dict) -> list[int]:
        """Tokenize a DeepCAD JSON structure into command tokens.

        DeepCAD format::

            {
                "sequence": [
                    {"type": "SKETCH", "plane": {...}, "profiles": [...]},
                    {"type": "EXT", "extent_one": 0.5, ...},
                    ...
                ]
            }

        When geotoken is available, the JSON steps are converted to
        geotoken command dicts and tokenized via
        :class:`CommandSequenceTokenizer` + :class:`CADVocabulary`.
        Otherwise the built-in DeepCAD mapping is used.

        Args:
            data: Parsed DeepCAD JSON data.

        Returns:
            List of integer token IDs.
        """
        # --- geotoken path ---
        if self._use_geotoken:
            return self._tokenize_deepcad_via_geotoken(data)

        # --- built-in fallback ---
        return self._tokenize_deepcad_builtin(data)

    def _tokenize_deepcad_via_geotoken(self, data: dict) -> list[int]:
        """Tokenize DeepCAD JSON using geotoken vocabulary.

        Args:
            data: Parsed DeepCAD JSON data.

        Returns:
            List of integer token IDs from CADVocabulary.
        """
        sequence = data.get(
            "sequence", data.get("ops", data.get("commands", []))
        )

        # Convert each step into a geotoken-compatible command dict
        commands: list[dict[str, Any]] = []
        for step in sequence:
            cmd: dict[str, Any] = {
                "type": step.get("type", step.get("command", "UNKNOWN")),
            }
            for key, val in step.items():
                if key in ("type", "command"):
                    continue
                cmd[key] = val
            commands.append(cmd)

        try:
            token_seq = self._geotoken_tokenizer.tokenize(construction_history=commands)
            encoded = self._geotoken_vocab.encode_full_sequence(token_seq)

            _log.debug(
                "geotoken tokenized DeepCAD JSON: %d steps -> %d tokens",
                len(sequence),
                len(encoded),
            )
            return encoded

        except Exception as exc:
            _log.warning(
                "geotoken DeepCAD tokenization failed, falling back: %s",
                exc,
            )
            return self._tokenize_deepcad_builtin(data)

    def _tokenize_deepcad_builtin(self, data: dict) -> list[int]:
        """Built-in DeepCAD JSON tokenization (legacy fallback).

        Args:
            data: Parsed DeepCAD JSON data.

        Returns:
            List of integer token IDs using the built-in mapping.
        """
        sequence = data.get(
            "sequence", data.get("ops", data.get("commands", []))
        )

        token_ids: list[int] = [_CMD_BOS_ID]

        for step in sequence:
            step_type = step.get("type", step.get("command", "UNKNOWN"))

            # Map to token
            type_id = _DEEPCAD_COMMAND_TYPES.get(
                step_type, hash(step_type) % 256
            )
            token_ids.append(_CMD_OFFSET + type_id)

            # Handle sketch steps
            if step_type in ("SKETCH", "sketch"):
                profiles = step.get("profiles", step.get("loops", []))
                sketch_tokens = self._encode_sketch_profiles(profiles)
                token_ids.extend(sketch_tokens)

            # Handle extrusion steps
            elif step_type in ("EXT", "extrude", "CUT", "cut"):
                ext_tokens = self._encode_extrusion_params(step)
                token_ids.extend(ext_tokens)

            # Handle fillet/chamfer
            elif step_type in ("FILLET", "fillet", "CHAMFER", "chamfer"):
                radius = step.get("radius", step.get("distance", 0))
                if isinstance(radius, (int, float)):
                    token_ids.append(int(abs(radius) * 100) % 50000)

            # Handle revolution
            elif step_type in ("REV", "revolve"):
                angle = step.get("angle", 360.0)
                if isinstance(angle, (int, float)):
                    token_ids.append(int(abs(angle) * 10) % 50000)
                axis = step.get("axis")
                if isinstance(axis, (list, tuple)):
                    for coord in axis[:3]:
                        if isinstance(coord, (int, float)):
                            token_ids.append(int(coord * 100) % 50000)

            # Handle boolean operations
            elif step_type in ("SOL", "boolean"):
                bool_type = step.get("operation", "union")
                bool_map = {"union": 1, "subtract": 2, "intersect": 3}
                token_ids.append(bool_map.get(bool_type, 0))

            # Generic: encode all numeric values
            else:
                for key, val in step.items():
                    if key == "type":
                        continue
                    if isinstance(val, (int, float)):
                        token_ids.append(int(abs(val) * 100) % 50000)

            token_ids.append(_CMD_SEP_ID)

        token_ids.append(_CMD_EOS_ID)

        _log.debug(
            f"Tokenized DeepCAD JSON: {len(sequence)} steps -> "
            f"{len(token_ids)} tokens"
        )

        return token_ids

    def _encode_extrusion_params(self, step: dict) -> list[int]:
        """Encode extrusion parameters from a DeepCAD step.

        Args:
            step: Extrusion step dictionary.

        Returns:
            List of parameter tokens.
        """
        tokens: list[int] = [_CMD_EXTRUDE_START]

        # Extent values
        for key in ("extent_one", "extent_two", "distance", "depth"):
            val = step.get(key)
            if isinstance(val, (int, float)):
                tokens.append(int(abs(val) * 1000) % 50000)

        # Boolean operation type
        bool_type = step.get("boolean", step.get("operation", "new_body"))
        bool_map = {
            "new_body": 0, "join": 1, "cut": 2,
            "intersect": 3, "new": 0, "add": 1, "subtract": 2,
        }
        tokens.append(bool_map.get(str(bool_type).lower(), 0))

        # Direction
        direction = step.get("direction", step.get("extent_type", "one_side"))
        dir_map = {
            "one_side": 0, "two_sides": 1, "symmetric": 2,
            "positive": 0, "negative": 3, "both": 1,
        }
        tokens.append(dir_map.get(str(direction).lower(), 0))

        tokens.append(_CMD_EXTRUDE_END)
        return tokens

    # ------------------------------------------------------------------
    # Text generation from DeepCAD commands (no STEP file)
    # ------------------------------------------------------------------

    def _text_from_deepcad_commands(self, data: dict) -> dict:
        """Generate text annotations from DeepCAD command structure.

        When no STEP file or rendered views are available, constructs
        text descriptions by analyzing the command sequence structure.
        Uses the LLM to expand the structural summary into multi-level
        annotations.

        Args:
            data: Parsed DeepCAD JSON data.

        Returns:
            Dictionary compatible with TextCADAnnotator.annotate() output.
        """
        # Build a structural summary of the commands
        sequence = data.get(
            "sequence", data.get("ops", data.get("commands", []))
        )

        summary_lines: list[str] = []
        summary_lines.append(
            f"CAD construction sequence with {len(sequence)} operations:"
        )

        sketch_count = 0
        extrude_count = 0
        fillet_count = 0
        other_ops: list[str] = []

        for step in sequence:
            step_type = step.get("type", step.get("command", "UNKNOWN"))

            if step_type in ("SKETCH", "sketch"):
                sketch_count += 1
                profiles = step.get("profiles", step.get("loops", []))
                num_curves = sum(
                    len(p) if isinstance(p, list) else len(
                        p.get("curves", p.get("segments", []))
                    )
                    for p in profiles
                )
                summary_lines.append(
                    f"  - Sketch #{sketch_count}: {len(profiles)} profile(s), "
                    f"~{num_curves} curve segments"
                )

            elif step_type in ("EXT", "extrude", "CUT", "cut"):
                extrude_count += 1
                distance = step.get(
                    "extent_one", step.get("distance", "unknown")
                )
                boolean = step.get("boolean", step.get("operation", "new"))
                summary_lines.append(
                    f"  - Extrude #{extrude_count}: distance={distance}, "
                    f"boolean={boolean}"
                )

            elif step_type in ("FILLET", "fillet", "CHAMFER", "chamfer"):
                fillet_count += 1
                radius = step.get("radius", step.get("distance", "unknown"))
                summary_lines.append(
                    f"  - {step_type.capitalize()}: radius={radius}"
                )

            else:
                other_ops.append(step_type)

        if other_ops:
            summary_lines.append(
                f"  - Other operations: {', '.join(set(other_ops))}"
            )

        structural_description = "\n".join(summary_lines)

        # Use LLM to generate multi-level annotations from structure
        annotations = {}
        for level in AnnotationLevel:
            try:
                prompt = self._build_deepcad_annotation_prompt(
                    structural_description, level
                )
                raw_response = self.text_annotator.llm_agent.ask(
                    prompt, max_tokens=1024
                )
                annotations[level.value] = raw_response.strip()
            except Exception as e:
                _log.error(
                    f"Failed to generate {level.value} from DeepCAD: {e}"
                )
                annotations[level.value] = ""

        return {
            AnnotationLevel.ABSTRACT.value: annotations.get(
                AnnotationLevel.ABSTRACT.value, ""
            ),
            AnnotationLevel.INTERMEDIATE.value: annotations.get(
                AnnotationLevel.INTERMEDIATE.value, ""
            ),
            AnnotationLevel.DETAILED.value: annotations.get(
                AnnotationLevel.DETAILED.value, ""
            ),
            AnnotationLevel.EXPERT.value: annotations.get(
                AnnotationLevel.EXPERT.value, ""
            ),
            "vlm_description": structural_description,
        }

    def _build_deepcad_annotation_prompt(
        self,
        structural_description: str,
        level: AnnotationLevel,
    ) -> str:
        """Build annotation prompt from DeepCAD command structure.

        Args:
            structural_description: Text summary of the command sequence.
            level: Target annotation level.

        Returns:
            Formatted prompt string.
        """
        level_guidance = {
            AnnotationLevel.ABSTRACT: (
                "Describe the likely overall shape in plain, high-level "
                "language. What does this object probably look like?"
            ),
            AnnotationLevel.INTERMEDIATE: (
                "Describe the features created by this construction sequence "
                "(e.g., 'extruded base with cut pockets and filleted edges'). "
                "Name specific feature types."
            ),
            AnnotationLevel.DETAILED: (
                "Provide a detailed engineering description with the "
                "dimensions, feature counts, and operation parameters "
                "listed in the construction sequence."
            ),
            AnnotationLevel.EXPERT: (
                "Provide a full parametric specification listing every "
                "operation in order with exact parameters, sketch profile "
                "descriptions, and boolean combinations."
            ),
        }

        guidance = level_guidance.get(level, "")

        prompt = (
            "You are an expert CAD engineer. Given the following "
            "construction sequence summary, produce a "
            f"{level.value.upper()}-level text description of the "
            "resulting 3D shape.\n\n"
            f"=== Construction Sequence ===\n{structural_description}\n\n"
            f"=== Level Guidance ===\n{guidance}\n\n"
            "Output ONLY the description text, no labels or preamble.\n\n"
            "Description:"
        )

        return prompt

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _export_jsonl(
        self,
        annotations: list[dict],
        output: Path,
    ) -> None:
        """Export annotations as JSONL (one JSON object per line).

        Args:
            annotations: List of annotation dictionaries.
            output: Output file path.
        """
        # Record which vocabulary was used for this export
        vocab_info = "geotoken" if self._use_geotoken else "builtin"
        vocab_size = (
            getattr(self._geotoken_vocab, "vocab_size", None)
            if self._use_geotoken
            else len(_DEEPCAD_COMMAND_TYPES) + _CMD_OFFSET
        )

        with open(output, "w") as f:
            for ann in annotations:
                record = {
                    "text_abstract": ann.get("text_abstract", ""),
                    "text_intermediate": ann.get("text_intermediate", ""),
                    "text_detailed": ann.get("text_detailed", ""),
                    "text_expert": ann.get("text_expert", ""),
                    "command_tokens": ann.get("command_tokens", []),
                }

                # Include metadata fields that are useful for training
                metadata = ann.get("metadata", {})
                if metadata:
                    record["source_file"] = metadata.get("source_file", "")
                    record["num_command_tokens"] = metadata.get(
                        "num_command_tokens", len(record["command_tokens"])
                    )
                    record["tokenizer_type"] = metadata.get(
                        "tokenizer_type", ""
                    )

                # Vocabulary provenance (important for downstream consumers)
                record["vocabulary"] = vocab_info
                record["vocab_size"] = vocab_size

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _export_json(
        self,
        annotations: list[dict],
        output: Path,
    ) -> None:
        """Export annotations as a single JSON array.

        Args:
            annotations: List of annotation dictionaries.
            output: Output file path.
        """
        records = []
        for ann in annotations:
            record = {
                "text_abstract": ann.get("text_abstract", ""),
                "text_intermediate": ann.get("text_intermediate", ""),
                "text_detailed": ann.get("text_detailed", ""),
                "text_expert": ann.get("text_expert", ""),
                "command_tokens": ann.get("command_tokens", []),
                "metadata": ann.get("metadata", {}),
            }
            records.append(record)

        with open(output, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
