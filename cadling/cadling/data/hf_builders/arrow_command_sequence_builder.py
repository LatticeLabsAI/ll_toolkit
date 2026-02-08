"""True ArrowBasedBuilder for CAD command sequences.

Properly inherits from datasets.ArrowBasedBuilder with _generate_tables()
for efficient HuggingFace Hub hosting and streaming.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

_log = logging.getLogger(__name__)

# Lazy imports
_datasets = None
_pa = None


def _ensure_deps():
    """Lazily import datasets and pyarrow."""
    global _datasets, _pa
    if _datasets is None:
        try:
            import datasets
            import pyarrow as pa
            _datasets = datasets
            _pa = pa
        except ImportError as e:
            raise ImportError(
                "datasets and pyarrow are required for ArrowBasedBuilder. "
                "Install via: pip install datasets>=2.16.0 pyarrow>=14.0.0"
            ) from e
    return _datasets, _pa


@dataclass
class CADCommandSequenceConfig:
    """Configuration for CAD command sequence datasets.

    Attributes:
        name: Configuration name (e.g., 'default', 'with_text').
        version: Dataset version string.
        description: Human-readable description.
        max_seq_len: Maximum command sequence length.
        num_params: Number of parameters per command.
        quantization_bits: Bits for parameter quantization.
        include_text: Whether to include text descriptions.
        include_renders: Whether to include rendered views.
        include_normalization: Whether to include normalization metadata.
        data_files: Mapping of split name to file paths.
    """
    name: str = "default"
    version: str = "1.0.0"
    description: str = ""
    max_seq_len: int = 60
    num_params: int = 16
    quantization_bits: int = 8
    include_text: bool = False
    include_renders: bool = False
    include_normalization: bool = True
    data_files: Optional[Dict[str, List[str]]] = None

    def __post_init__(self):
        if not self.description:
            self.description = f"CAD command sequences ({self.name})"


# Pre-defined configurations
DEFAULT_CONFIG = CADCommandSequenceConfig(
    name="default",
    description="Standard DeepCAD-style command sequences",
)

WITH_TEXT_CONFIG = CADCommandSequenceConfig(
    name="with_text",
    include_text=True,
    description="Command sequences with 4-level text annotations",
)

WITH_RENDERS_CONFIG = CADCommandSequenceConfig(
    name="with_renders",
    include_renders=True,
    description="Command sequences with multi-view renders",
)

FULL_CONFIG = CADCommandSequenceConfig(
    name="full",
    include_text=True,
    include_renders=True,
    description="Complete dataset with text and renders",
)


def _get_builder_class():
    """Get the ArrowBasedBuilder class lazily to avoid import errors."""
    datasets, pa = _ensure_deps()

    class ArrowCADCommandSequenceBuilder(datasets.ArrowBasedBuilder):
        """True ArrowBasedBuilder for CAD command sequences.

        Inherits from datasets.ArrowBasedBuilder and implements
        the required _info(), _split_generators(), and _generate_tables()
        methods for proper HuggingFace ecosystem integration.

        This enables:
        - Efficient streaming from HuggingFace Hub
        - Automatic Arrow format handling
        - Native integration with datasets library features

        Usage:
            >>> # Load from Hub
            >>> ds = datasets.load_dataset(
            ...     "latticelabs/deepcad-sequences",
            ...     streaming=True,
            ... )

            >>> # Build locally
            >>> builder = ArrowCADCommandSequenceBuilder(
            ...     config=DEFAULT_CONFIG,
            ...     data_files={"train": ["train/*.json"]},
            ... )
            >>> builder.download_and_prepare()
            >>> ds = builder.as_dataset()
        """

        VERSION = datasets.Version("1.0.0")
        BUILDER_CONFIGS = [
            datasets.BuilderConfig(name="default", version=VERSION),
            datasets.BuilderConfig(name="with_text", version=VERSION),
            datasets.BuilderConfig(name="with_renders", version=VERSION),
            datasets.BuilderConfig(name="full", version=VERSION),
        ]
        DEFAULT_CONFIG_NAME = "default"

        # Command type mapping
        COMMAND_TYPES = {"SOL": 0, "Line": 1, "Arc": 2, "Circle": 3, "Ext": 4, "EOS": 5}
        NUM_COMMAND_TYPES = 6

        def __init__(
            self,
            config: Optional[CADCommandSequenceConfig] = None,
            data_files: Optional[Dict[str, List[str]]] = None,
            **kwargs,
        ):
            """Initialize the builder.

            Args:
                config: CADCommandSequenceConfig or use default.
                data_files: Mapping of split names to file paths.
                **kwargs: Additional arguments passed to ArrowBasedBuilder.
            """
            self._cad_config = config or DEFAULT_CONFIG
            self._data_files = data_files or self._cad_config.data_files

            # Set up BuilderConfig for parent class
            if "config" not in kwargs:
                builder_config = datasets.BuilderConfig(
                    name=self._cad_config.name,
                    version=datasets.Version(self._cad_config.version),
                    description=self._cad_config.description,
                )
                kwargs["config"] = builder_config

            super().__init__(**kwargs)

        def _info(self) -> datasets.DatasetInfo:
            """Return dataset metadata including features schema."""
            features_dict = {
                # Unique identifier
                "sample_id": datasets.Value("string"),

                # Command types: int8 for 6 types (0-5)
                "command_types": datasets.Sequence(datasets.Value("int8")),

                # Parameters: int16 for 65536 quantization levels
                "parameters": datasets.Sequence(datasets.Value("int16")),

                # Valid command mask
                "mask": datasets.Sequence(datasets.Value("float32")),

                # Actual command count
                "num_commands": datasets.Value("int32"),

                # Provenance
                "model_id": datasets.Value("string"),
                "source": datasets.Value("string"),
                "metadata": datasets.Value("string"),
            }

            # Normalization metadata for dequantization
            if self._cad_config.include_normalization:
                features_dict["normalization_center"] = datasets.Sequence(
                    datasets.Value("float32")
                )
                features_dict["normalization_scale"] = datasets.Value("float32")

            # 4-level text annotations
            if self._cad_config.include_text:
                features_dict["text_description"] = datasets.Value("string")
                features_dict["text_abstract"] = datasets.Value("string")
                features_dict["text_intermediate"] = datasets.Value("string")
                features_dict["text_detailed"] = datasets.Value("string")
                features_dict["text_expert"] = datasets.Value("string")
                features_dict["text_embedding"] = datasets.Sequence(
                    datasets.Value("float32")
                )

            # Multi-view renders
            if self._cad_config.include_renders:
                features_dict["render_front"] = datasets.Value("binary")
                features_dict["render_top"] = datasets.Value("binary")
                features_dict["render_iso"] = datasets.Value("binary")
                features_dict["segmentation_mask"] = datasets.Value("binary")

            return datasets.DatasetInfo(
                description=self._cad_config.description,
                features=datasets.Features(features_dict),
                homepage="https://github.com/latticelabs/cadling",
                license="Apache-2.0",
                version=datasets.Version(self._cad_config.version),
            )

        def _split_generators(
            self, dl_manager: datasets.DownloadManager
        ) -> List[datasets.SplitGenerator]:
            """Download data and return split generators."""
            if not self._data_files:
                raise ValueError(
                    "data_files must be provided to build dataset. "
                    "Pass data_files={'train': [...], 'val': [...]} to constructor."
                )

            # Download and extract if URLs provided
            data_files = dl_manager.download_and_extract(self._data_files)

            splits = []
            for split_name, files in data_files.items():
                if isinstance(files, str):
                    files = [files]

                # Map to datasets.Split enum
                split_enum = getattr(
                    datasets.Split,
                    split_name.upper(),
                    datasets.Split.TRAIN,
                )

                splits.append(
                    datasets.SplitGenerator(
                        name=split_enum,
                        gen_kwargs={"files": files},
                    )
                )

            return splits

        def _generate_tables(
            self, files: List[str]
        ) -> Iterator[Tuple[str, pa.Table]]:
            """Generate PyArrow tables from source files.

            This is the core method that ArrowBasedBuilder requires.
            Yields (key, pa.Table) tuples.

            Args:
                files: List of file paths to process.

            Yields:
                Tuple of (unique_key, PyArrow Table).
            """
            batch_size = 100  # Batch for efficiency
            batch = []
            batch_idx = 0

            for file_path in files:
                file_path = Path(file_path)

                if file_path.suffix == ".json":
                    sample = self._process_json_file(file_path)
                    if sample is not None:
                        batch.append(sample)

                        if len(batch) >= batch_size:
                            table = self._samples_to_table(batch)
                            yield str(batch_idx), table
                            batch = []
                            batch_idx += 1

                elif file_path.suffix == ".parquet":
                    # Already in Arrow format, read directly
                    import pyarrow.parquet as pq
                    table = pq.read_table(file_path)
                    yield str(file_path.stem), self._cast_table(table)

            # Yield remaining samples
            if batch:
                table = self._samples_to_table(batch)
                yield str(batch_idx), table

        def _cast_table(self, table: pa.Table) -> pa.Table:
            """Cast table to match expected schema."""
            from datasets.table import table_cast

            if self.info.features is not None:
                schema = self.info.features.arrow_schema
                try:
                    table = table_cast(table, schema)
                except Exception as e:
                    _log.warning("Failed to cast table: %s", e)
            return table

        def _samples_to_table(self, samples: List[Dict[str, Any]]) -> pa.Table:
            """Convert list of samples to PyArrow Table."""
            if not samples:
                return pa.table({})

            # Build columnar data
            columns: Dict[str, List[Any]] = {}
            for key in samples[0].keys():
                columns[key] = [s.get(key) for s in samples]

            return pa.Table.from_pydict(columns)

        def _process_json_file(
            self, json_path: Path, source: str = "deepcad"
        ) -> Optional[Dict[str, Any]]:
            """Process a single DeepCAD JSON file into a sample dict."""
            try:
                with open(json_path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                _log.debug("Failed to load %s: %s", json_path, e)
                return None

            # Extract command sequence
            sequence = data.get("sequence", data.get("commands", []))
            if not sequence:
                return None

            commands = self._parse_commands(sequence)
            if not commands or len(commands) < 2:
                return None

            # Normalize and quantize
            commands = self._normalize_sketches(commands)
            cmd_types, params = self._quantize(commands)

            # Convert to lists for Arrow
            if hasattr(cmd_types, "tolist"):
                cmd_types = cmd_types.tolist()
            if hasattr(params, "tolist"):
                params = params.tolist()

            num_commands = len(cmd_types)
            max_seq_len = self._cad_config.max_seq_len
            num_params = self._cad_config.num_params

            # Create mask
            mask = [1.0] * min(num_commands, max_seq_len)

            # Pad/truncate
            if num_commands > max_seq_len:
                cmd_types = cmd_types[:max_seq_len]
                params = params[:max_seq_len]
                num_commands = max_seq_len
            elif num_commands < max_seq_len:
                pad_len = max_seq_len - num_commands
                cmd_types = cmd_types + [self.COMMAND_TYPES["EOS"]] * pad_len
                params = params + [[0] * num_params] * pad_len
                mask = mask + [0.0] * pad_len

            # Flatten params
            params_flat = []
            for p in params:
                params_flat.extend(p[:num_params])
                if len(p) < num_params:
                    params_flat.extend([0] * (num_params - len(p)))

            sample = {
                "sample_id": json_path.stem,
                "command_types": cmd_types,
                "parameters": params_flat,
                "mask": mask,
                "num_commands": num_commands,
                "model_id": json_path.stem,
                "source": source,
                "metadata": json.dumps({"original_file": json_path.name}),
            }

            # Add normalization metadata
            if self._cad_config.include_normalization:
                sample["normalization_center"] = [0.0, 0.0, 0.0]
                sample["normalization_scale"] = 1.0

            # Add text fields if configured
            if self._cad_config.include_text:
                sample["text_description"] = data.get("description", None)
                sample["text_abstract"] = data.get("text_abstract", None)
                sample["text_intermediate"] = data.get("text_intermediate", None)
                sample["text_detailed"] = data.get("text_detailed", None)
                sample["text_expert"] = data.get("text_expert", None)
                sample["text_embedding"] = None

            # Add render fields if configured
            if self._cad_config.include_renders:
                sample["render_front"] = None
                sample["render_top"] = None
                sample["render_iso"] = None
                sample["segmentation_mask"] = None

            return sample

        def _parse_commands(
            self, sequence: List[Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
            """Parse raw command dicts."""
            commands = []
            for cmd in sequence:
                cmd_type = cmd.get("type", cmd.get("command", ""))
                if cmd_type not in self.COMMAND_TYPES:
                    continue
                params = cmd.get("params", cmd.get("parameters", []))
                if isinstance(params, dict):
                    params = list(params.values())
                params = [float(p) for p in params]
                num_params = self._cad_config.num_params
                params = params[:num_params] + [0.0] * max(0, num_params - len(params))
                commands.append({"type": cmd_type, "params": params})
            return commands

        def _normalize_sketches(
            self, commands: List[Dict[str, Any]], normalization_range: float = 2.0
        ) -> List[Dict[str, Any]]:
            """Normalize 2D sketch coordinates."""
            try:
                import numpy as np
            except ImportError:
                return commands

            points = []
            for cmd in commands:
                if cmd["type"] in ("Line", "Arc", "Circle"):
                    points.extend(cmd["params"][:6])

            if not points:
                return commands

            points_arr = np.array(points)
            max_abs = np.max(np.abs(points_arr)) + 1e-8
            scale = (normalization_range / 2.0) / max_abs

            for cmd in commands:
                if cmd["type"] in ("Line", "Arc", "Circle"):
                    cmd["params"] = [
                        p * scale if i < 6 else p
                        for i, p in enumerate(cmd["params"])
                    ]
            return commands

        def _quantize(
            self,
            commands: List[Dict[str, Any]],
            normalization_range: float = 2.0,
        ) -> Tuple[List[int], List[List[int]]]:
            """Quantize continuous parameters to discrete levels."""
            num_levels = 2 ** self._cad_config.quantization_bits
            half_range = normalization_range / 2.0
            cmd_types_list = []
            params_list = []

            for cmd in commands:
                cmd_types_list.append(self.COMMAND_TYPES[cmd["type"]])
                q_params = []
                for p in cmd["params"]:
                    p_clamped = max(-half_range, min(half_range, p))
                    normalized = (p_clamped + half_range) / normalization_range
                    q = int(round(normalized * (num_levels - 1)))
                    q = max(0, min(num_levels - 1, q))
                    q_params.append(q)
                params_list.append(q_params)

            return cmd_types_list, params_list

    return ArrowCADCommandSequenceBuilder


def get_arrow_builder(**kwargs):
    """Get an instance of the ArrowBasedBuilder."""
    BuilderClass = _get_builder_class()
    return BuilderClass(**kwargs)


__all__ = [
    "CADCommandSequenceConfig",
    "DEFAULT_CONFIG",
    "WITH_TEXT_CONFIG",
    "WITH_RENDERS_CONFIG",
    "FULL_CONFIG",
    "get_arrow_builder",
]
