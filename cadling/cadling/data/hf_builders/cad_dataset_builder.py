"""HuggingFace Dataset Builder for CAD command sequences.

Provides ArrowBasedBuilder for converting DeepCAD-style command
sequence data to Parquet format on HuggingFace Hub.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional

_log = logging.getLogger(__name__)

# Lazy imports for heavy dependencies
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
                "datasets and pyarrow are required for HF builders. "
                "Install via: pip install datasets>=2.16.0 pyarrow>=14.0.0"
            ) from e
    return _datasets, _pa


class CADCommandSequenceConfig:
    """Configuration for CADCommandSequenceBuilder.

    Attributes:
        name: Configuration name (e.g., 'default', 'with_text', 'with_renders').
        max_seq_len: Maximum command sequence length.
        num_params: Number of parameters per command.
        quantization_bits: Bits for parameter quantization.
        include_text: Whether to include text descriptions.
        include_renders: Whether to include rendered views.
        version: Dataset version string.
        description: Human-readable description.
    """

    def __init__(
        self,
        name: str = "default",
        max_seq_len: int = 60,
        num_params: int = 16,
        quantization_bits: int = 8,
        include_text: bool = False,
        include_renders: bool = False,
        version: str = "1.0.0",
        description: str = "",
    ) -> None:
        self.name = name
        self.max_seq_len = max_seq_len
        self.num_params = num_params
        self.quantization_bits = quantization_bits
        self.include_text = include_text
        self.include_renders = include_renders
        self.version = version
        self.description = description or f"CAD command sequences ({name})"


# Pre-defined configurations
DEFAULT_CONFIG = CADCommandSequenceConfig(
    name="default",
    description="Standard DeepCAD-style command sequences",
)

WITH_TEXT_CONFIG = CADCommandSequenceConfig(
    name="with_text",
    include_text=True,
    description="Command sequences with natural language descriptions",
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


class CADCommandSequenceBuilder:
    """Builder for CAD command sequence datasets.

    Converts DeepCAD-style JSON data to Parquet format suitable for
    HuggingFace Hub hosting. Supports multiple configurations:
    - default: Just command sequences
    - with_text: Adds natural language descriptions
    - with_renders: Adds multi-view rendered images

    Usage:
        >>> builder = CADCommandSequenceBuilder(
        ...     source_dir="/path/to/deepcad",
        ...     config=DEFAULT_CONFIG,
        ... )
        >>> # Build to local Parquet files
        >>> builder.build("/path/to/output")
        >>>
        >>> # Or get as HuggingFace Dataset
        >>> dataset = builder.to_dataset()

    Args:
        source_dir: Directory containing source JSON files.
        config: Configuration specifying dataset options.
        splits: List of splits to process (e.g., ['train', 'val', 'test']).
    """

    # DeepCAD command type mapping
    COMMAND_TYPES = {"SOL": 0, "Line": 1, "Arc": 2, "Circle": 3, "Ext": 4, "EOS": 5}
    NUM_COMMAND_TYPES = 6

    def __init__(
        self,
        source_dir: Optional[str] = None,
        config: Optional[CADCommandSequenceConfig] = None,
        splits: Optional[List[str]] = None,
    ) -> None:
        _ensure_deps()

        self.source_dir = Path(source_dir) if source_dir else None
        self.config = config or DEFAULT_CONFIG
        self.splits = splits or ["train", "val", "test"]

        # Schema from our schemas module
        from ..schemas import get_command_sequence_schema
        self._schema = get_command_sequence_schema(
            max_seq_len=self.config.max_seq_len,
            num_params=self.config.num_params,
            include_text=self.config.include_text,
            include_renders=self.config.include_renders,
        )

    def _get_features(self) -> "datasets.Features":
        """Get HuggingFace Features from PyArrow schema."""
        datasets, pa = _ensure_deps()

        features_dict = {}

        for field in self._schema:
            name = field.name

            if pa.types.is_string(field.type):
                features_dict[name] = datasets.Value("string")
            elif pa.types.is_int32(field.type):
                features_dict[name] = datasets.Value("int32")
            elif pa.types.is_int64(field.type):
                features_dict[name] = datasets.Value("int64")
            elif pa.types.is_float32(field.type):
                features_dict[name] = datasets.Value("float32")
            elif pa.types.is_binary(field.type):
                features_dict[name] = datasets.Value("binary")
            elif pa.types.is_list(field.type):
                # Get element type
                elem_type = field.type.value_type
                if pa.types.is_int32(elem_type):
                    features_dict[name] = datasets.Sequence(
                        datasets.Value("int32")
                    )
                elif pa.types.is_int64(elem_type):
                    features_dict[name] = datasets.Sequence(
                        datasets.Value("int64")
                    )
                elif pa.types.is_float32(elem_type):
                    features_dict[name] = datasets.Sequence(
                        datasets.Value("float32")
                    )
                else:
                    features_dict[name] = datasets.Sequence(
                        datasets.Value("string")
                    )
            else:
                features_dict[name] = datasets.Value("string")

        return datasets.Features(features_dict)

    def _parse_commands(
        self, sequence: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Parse raw DeepCAD command dicts."""
        commands: List[Dict[str, Any]] = []
        for cmd in sequence:
            cmd_type = cmd.get("type", cmd.get("command", ""))
            if cmd_type not in self.COMMAND_TYPES:
                continue
            params = cmd.get("params", cmd.get("parameters", []))
            if isinstance(params, dict):
                params = list(params.values())
            params = [float(p) for p in params]
            # Pad params to num_params
            num_params = self.config.num_params
            params = params[:num_params] + [0.0] * max(0, num_params - len(params))
            commands.append({"type": cmd_type, "params": params})
        return commands

    def _normalize_sketches(
        self, commands: List[Dict[str, Any]], normalization_range: float = 2.0
    ) -> List[Dict[str, Any]]:
        """Normalize 2D sketch coordinates to [-1, 1] range."""
        try:
            import numpy as np
        except ImportError:
            return commands

        points: List[float] = []
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
    ) -> tuple:
        """Quantize continuous parameters to discrete levels."""
        try:
            import numpy as np
        except ImportError:
            # Fallback without numpy
            num_levels = 2 ** self.config.quantization_bits
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

        num_levels = 2 ** self.config.quantization_bits
        half_range = normalization_range / 2.0
        cmd_types_list: List[int] = []
        params_list: List[List[int]] = []

        for cmd in commands:
            cmd_types_list.append(self.COMMAND_TYPES[cmd["type"]])
            q_params: List[int] = []
            for p in cmd["params"]:
                p_clamped = max(-half_range, min(half_range, p))
                normalized = (p_clamped + half_range) / normalization_range
                q = int(round(normalized * (num_levels - 1)))
                q = max(0, min(num_levels - 1, q))
                q_params.append(q)
            params_list.append(q_params)

        return np.array(cmd_types_list), np.array(params_list)

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

        # Normalize sketches
        commands = self._normalize_sketches(commands)

        # Quantize parameters
        cmd_types, params = self._quantize(commands)

        # Convert to lists if numpy arrays
        if hasattr(cmd_types, "tolist"):
            cmd_types = cmd_types.tolist()
        if hasattr(params, "tolist"):
            params = params.tolist()

        num_commands = len(cmd_types)
        max_seq_len = self.config.max_seq_len
        num_params = self.config.num_params

        # Create mask (1=valid, 0=padding)
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

        # Flatten params to [max_seq_len * num_params]
        params_flat = []
        for p in params:
            params_flat.extend(p[:num_params])
            # Pad if needed
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

        # Add text if configured
        if self.config.include_text:
            sample["text_description"] = data.get(
                "description", data.get("text", None)
            )
            sample["text_embedding"] = None

        # Add renders if configured
        if self.config.include_renders:
            sample["render_front"] = None
            sample["render_top"] = None
            sample["render_iso"] = None
            sample["segmentation_mask"] = None

        return sample

    def generate_samples(
        self, split: str = "train"
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate samples for a given split.

        Yields sample dictionaries suitable for conversion to Parquet.

        Args:
            split: Dataset split ('train', 'val', 'test').

        Yields:
            Sample dictionaries with command sequence data.
        """
        if self.source_dir is None:
            raise ValueError("source_dir must be set to generate samples")

        split_dir = self.source_dir / split
        if not split_dir.exists():
            _log.warning("Split directory not found: %s", split_dir)
            return

        json_files = sorted(split_dir.glob("*.json"))
        _log.info("Processing %d JSON files from %s", len(json_files), split_dir)

        for json_path in json_files:
            sample = self._process_json_file(json_path)
            if sample is not None:
                yield sample

    def to_arrow_tables(
        self,
    ) -> Dict[str, "pa.Table"]:
        """Convert all splits to PyArrow Tables.

        Returns:
            Dictionary mapping split names to PyArrow Tables.
        """
        datasets, pa = _ensure_deps()
        from ..schemas import samples_to_table

        tables = {}

        for split in self.splits:
            samples = list(self.generate_samples(split))
            if samples:
                tables[split] = samples_to_table(samples, self._schema)
                _log.info(
                    "Built %s table with %d samples", split, len(samples)
                )

        return tables

    def build(
        self,
        output_dir: str,
        compression: str = "zstd",
        row_group_size: int = 10000,
    ) -> Dict[str, Path]:
        """Build Parquet files for all splits.

        Args:
            output_dir: Directory to write Parquet files.
            compression: Compression codec ('zstd', 'snappy', 'gzip').
            row_group_size: Rows per row group (affects streaming performance).

        Returns:
            Dictionary mapping split names to output file paths.
        """
        datasets, pa = _ensure_deps()
        import pyarrow.parquet as pq

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        output_files = {}
        tables = self.to_arrow_tables()

        for split, table in tables.items():
            file_path = output_path / f"{split}.parquet"
            pq.write_table(
                table,
                file_path,
                compression=compression,
                row_group_size=row_group_size,
            )
            output_files[split] = file_path
            _log.info("Wrote %s to %s", split, file_path)

        return output_files

    def to_dataset(self) -> "datasets.DatasetDict":
        """Convert to HuggingFace DatasetDict.

        Returns:
            DatasetDict with splits as keys.
        """
        datasets_lib, pa = _ensure_deps()

        dataset_dict = {}
        features = self._get_features()

        for split in self.splits:
            samples = list(self.generate_samples(split))
            if samples:
                dataset_dict[split] = datasets_lib.Dataset.from_list(
                    samples, features=features
                )

        return datasets_lib.DatasetDict(dataset_dict)

    def push_to_hub(
        self,
        repo_id: str,
        private: bool = False,
        token: Optional[str] = None,
        commit_message: str = "Upload CAD command sequence dataset",
    ) -> str:
        """Push dataset to HuggingFace Hub.

        Args:
            repo_id: Repository ID (e.g., 'username/dataset-name').
            private: Whether to make the repository private.
            token: HuggingFace API token.
            commit_message: Commit message for the upload.

        Returns:
            URL of the uploaded dataset.
        """
        dataset = self.to_dataset()

        dataset.push_to_hub(
            repo_id,
            private=private,
            token=token,
            commit_message=commit_message,
        )

        return f"https://huggingface.co/datasets/{repo_id}"


__all__ = [
    "CADCommandSequenceBuilder",
    "CADCommandSequenceConfig",
    "DEFAULT_CONFIG",
    "WITH_TEXT_CONFIG",
    "WITH_RENDERS_CONFIG",
    "FULL_CONFIG",
]
