"""DeepCAD dataset loader.

Loads and preprocesses DeepCAD's 178K sketch-and-extrude command
sequences. Parses JSON format, normalizes sketches, quantizes
parameters, and pads to fixed length for transformer consumption.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .base_loader import BaseCADDataset

_log = logging.getLogger(__name__)


class DeepCADLoader(BaseCADDataset):
    """Dataset loader for DeepCAD's sketch-and-extrude sequences.

    Each sample is a fixed-length command sequence with quantized
    parameters, ready for training autoregressive or VAE-based
    generation models.

    Supports two modes:
    1. Local mode: Load from JSON files in root_dir
    2. Hub mode: Stream from HuggingFace Hub (e.g., "latticelabs/deepcad")

    Args:
        root_dir: Path to the DeepCAD dataset directory (optional if hub_id provided).
        split: Dataset split ('train', 'val', 'test').
        max_seq_len: Maximum command sequence length (60 per DeepCAD).
        quantization_bits: Bits for parameter quantization (8 = 256 levels).
        normalization_range: Bounding cube size for normalization.
        transform: Optional transform applied to each sample dict.
        download: Download dataset if not present.
        hub_id: HuggingFace Hub dataset ID (e.g., "latticelabs/deepcad-sequences").
        streaming: Whether to stream from Hub (default True for Hub mode).
    """

    # DeepCAD command type mapping
    COMMAND_TYPES = {"SOL": 0, "Line": 1, "Arc": 2, "Circle": 3, "Ext": 4, "EOS": 5}
    NUM_COMMAND_TYPES = 6
    NUM_PARAMS = 16

    def __init__(
        self,
        root_dir: Optional[str] = None,
        split: str = "train",
        max_seq_len: int = 60,
        quantization_bits: int = 8,
        normalization_range: float = 2.0,
        transform: Optional[Callable] = None,
        download: bool = False,
        hub_id: Optional[str] = None,
        streaming: bool = True,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.quantization_bits = quantization_bits
        self.num_levels = 2 ** quantization_bits
        self.normalization_range = normalization_range

        self._samples: list[dict[str, Any]] = []

        super().__init__(root_dir, split, transform, download, hub_id, streaming)

        # Load samples if data exists (local mode only)
        if not self._use_hub and self._verify_integrity():
            self._load_samples()

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a sample dict with tokenized command sequence.

        Returns:
            Dict with:
                'command_types': [max_seq_len] int array of command type indices.
                'parameters': [max_seq_len, NUM_PARAMS] int array of quantized params.
                'mask': [max_seq_len] bool array (1=valid, 0=padding).
                'num_commands': int, actual command count before padding.
        """
        sample = self._samples[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def download(self) -> None:
        """Download DeepCAD dataset (178K sketch-and-extrude sequences).

        Attempts automated download in order of preference:

        1. **HuggingFace Hub** via ``huggingface_hub.snapshot_download``.
        2. **Direct GitHub release** tarball from the DeepCAD repository.
        3. **Manual fallback** — logs the URL and expected directory layout.

        After download, the dataset is placed in ``root_dir`` with structure::

            root_dir/
                train/
                    *.json
                val/
                    *.json
                test/
                    *.json
        """
        if self._verify_integrity():
            _log.info("DeepCAD dataset already present at %s", self.root_dir)
            return

        self.root_dir.mkdir(parents=True, exist_ok=True)

        # --- Strategy 1: HuggingFace Hub ---
        try:
            from huggingface_hub import snapshot_download

            _log.info("Downloading DeepCAD dataset from HuggingFace Hub...")
            snapshot_download(
                repo_id="Wenchao/DeepCAD",
                repo_type="dataset",
                local_dir=str(self.root_dir),
                allow_patterns=["*.json", "*.jsonl"],
            )

            if self._verify_integrity():
                _log.info("DeepCAD dataset downloaded via HuggingFace Hub")
                return

            # Some Hub layouts nest data under a data/ prefix
            data_dir = self.root_dir / "data"
            if data_dir.exists():
                import shutil

                for split_name in ("train", "val", "test"):
                    src = data_dir / split_name
                    dst = self.root_dir / split_name
                    if src.exists() and not dst.exists():
                        shutil.copytree(str(src), str(dst))
                if self._verify_integrity():
                    _log.info("DeepCAD restructured from data/ prefix")
                    return

        except ImportError:
            _log.debug("huggingface_hub not installed; trying direct download")
        except Exception as exc:
            _log.warning("HuggingFace Hub download failed: %s", exc)

        # --- Strategy 2: Direct GitHub release tarball ---
        try:
            import os
            import tarfile
            import tempfile
            import urllib.request

            url = (
                "https://github.com/ChrisWu1997/DeepCAD/releases/download/"
                "v1.0/DeepCAD_data.tar.gz"
            )
            _log.info("Downloading DeepCAD from GitHub: %s", url)

            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                tmp_path = tmp.name
                urllib.request.urlretrieve(url, tmp_path)

            with tarfile.open(tmp_path, "r:gz") as tar:
                tar.extractall(path=str(self.root_dir))
            os.unlink(tmp_path)

            if self._verify_integrity():
                _log.info("DeepCAD extracted from GitHub release")
                return

        except Exception as exc:
            _log.warning("Direct download failed: %s", exc)

        # --- Strategy 3: Manual instructions ---
        _log.info(
            "Automated download unsuccessful. Please download manually "
            "from https://github.com/ChrisWu1997/DeepCAD and place JSON "
            "files in %s/{train,val,test}/",
            self.root_dir,
        )

    def _verify_integrity(self) -> bool:
        """Check if split directory exists with JSON files."""
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            return False
        json_files = list(split_dir.glob("*.json"))
        return len(json_files) > 0

    def _load_samples(self) -> None:
        """Load and preprocess all JSON files for this split."""
        split_dir = self.root_dir / self.split
        json_files = sorted(split_dir.glob("*.json"))

        _log.info("Loading %d DeepCAD files from %s", len(json_files), split_dir)

        for json_path in json_files:
            try:
                with open(json_path) as f:
                    data = json.load(f)
                sample = self._process_deepcad_json(data, json_path.stem)
                if sample is not None:
                    self._samples.append(sample)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                _log.debug("Skipping %s: %s", json_path.name, e)

        _log.info("Loaded %d valid samples", len(self._samples))

    def _process_deepcad_json(
        self, data: dict[str, Any], model_id: str
    ) -> Optional[dict[str, Any]]:
        """Process a single DeepCAD JSON file into a training sample.

        Pipeline: parse → normalize → quantize → pad.
        """
        # Extract command sequence
        sequence = data.get("sequence", data.get("commands", []))
        if not sequence:
            return None

        commands = self._parse_commands(sequence)
        if not commands or len(commands) < 2:
            return None

        # Normalize sketch coordinates
        commands = self._normalize_sketches(commands)

        # Normalize extrusion parameters
        commands = self._normalize_extrusions(commands)

        # Quantize parameters
        cmd_types, params = self._quantize(commands)

        # Create mask (1=valid, 0=padding)
        num_commands = len(cmd_types)
        mask = np.ones(self.max_seq_len, dtype=np.float32)

        # Pad/truncate
        if num_commands > self.max_seq_len:
            cmd_types = cmd_types[:self.max_seq_len]
            params = params[:self.max_seq_len]
            num_commands = self.max_seq_len
        elif num_commands < self.max_seq_len:
            pad_len = self.max_seq_len - num_commands
            cmd_types = np.concatenate([
                cmd_types, np.full(pad_len, self.COMMAND_TYPES["EOS"])
            ])
            params = np.concatenate([
                params, np.zeros((pad_len, self.NUM_PARAMS), dtype=np.int64)
            ])
            mask[num_commands:] = 0.0

        return {
            "command_types": cmd_types.astype(np.int64),
            "parameters": params.astype(np.int64),
            "mask": mask,
            "num_commands": num_commands,
            "model_id": model_id,
        }

    def _parse_commands(
        self, sequence: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Parse raw DeepCAD command dicts."""
        commands: list[dict[str, Any]] = []
        for cmd in sequence:
            cmd_type = cmd.get("type", cmd.get("command", ""))
            if cmd_type not in self.COMMAND_TYPES:
                continue
            params = cmd.get("params", cmd.get("parameters", []))
            if isinstance(params, dict):
                params = list(params.values())
            params = [float(p) for p in params]
            # Pad params to NUM_PARAMS
            params = params[:self.NUM_PARAMS] + [0.0] * max(0, self.NUM_PARAMS - len(params))
            commands.append({"type": cmd_type, "params": params})
        return commands

    def _normalize_sketches(
        self, commands: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Normalize 2D sketch coordinates to [-1, 1] range."""
        points: list[float] = []
        for cmd in commands:
            if cmd["type"] in ("Line", "Arc", "Circle"):
                points.extend(cmd["params"][:6])

        if not points:
            return commands

        points_arr = np.array(points)
        max_abs = np.max(np.abs(points_arr)) + 1e-8
        scale = (self.normalization_range / 2.0) / max_abs

        for cmd in commands:
            if cmd["type"] in ("Line", "Arc", "Circle"):
                cmd["params"] = [p * scale if i < 6 else p
                                 for i, p in enumerate(cmd["params"])]
        return commands

    def _normalize_extrusions(
        self, commands: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Normalize extrusion parameters to [-1, 1] range."""
        ext_vals: list[float] = []
        for cmd in commands:
            if cmd["type"] == "Ext":
                ext_vals.extend([abs(p) for p in cmd["params"] if p != 0])

        if not ext_vals:
            return commands

        max_val = max(ext_vals) + 1e-8
        scale = (self.normalization_range / 2.0) / max_val

        for cmd in commands:
            if cmd["type"] == "Ext":
                cmd["params"] = [p * scale for p in cmd["params"]]
        return commands

    def _quantize(
        self, commands: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Quantize continuous parameters to discrete levels."""
        cmd_types_list: list[int] = []
        params_list: list[list[int]] = []
        half_range = self.normalization_range / 2.0

        for cmd in commands:
            cmd_types_list.append(self.COMMAND_TYPES[cmd["type"]])
            q_params: list[int] = []
            for p in cmd["params"]:
                p_clamped = max(-half_range, min(half_range, p))
                normalized = (p_clamped + half_range) / self.normalization_range
                q = int(round(normalized * (self.num_levels - 1)))
                q = max(0, min(self.num_levels - 1, q))
                q_params.append(q)
            params_list.append(q_params)

        return np.array(cmd_types_list), np.array(params_list)

    def _process_hub_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Process a sample from Hub format to loader format.

        Hub format is pre-quantized with:
        - command_types: [seq_len] int8 array
        - parameters: [seq_len * num_params] int16 array (flat)
        - mask: [seq_len] float32 array
        - num_commands: int
        - model_id: str (optional)
        """
        # Convert lists to numpy arrays if needed
        cmd_types = sample.get("command_types", [])
        params = sample.get("parameters", [])
        mask = sample.get("mask", [])
        num_commands = sample.get("num_commands", len(cmd_types))

        if isinstance(cmd_types, list):
            cmd_types = np.array(cmd_types, dtype=np.int64)
        if isinstance(params, list):
            params = np.array(params, dtype=np.int64)
        if isinstance(mask, list):
            mask = np.array(mask, dtype=np.float32)

        # Reshape flat parameters to [seq_len, num_params]
        if params.ndim == 1:
            seq_len = len(cmd_types)
            if len(params) == seq_len * self.NUM_PARAMS:
                params = params.reshape(seq_len, self.NUM_PARAMS)

        return {
            "command_types": cmd_types,
            "parameters": params,
            "mask": mask,
            "num_commands": num_commands,
            "model_id": sample.get("model_id", sample.get("sample_id", "")),
        }
