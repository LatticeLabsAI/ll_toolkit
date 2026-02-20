"""Text2CAD dataset loader.

Loads Text2CAD's 660K text-annotated CAD models with multi-level
annotations (abstract, intermediate, detailed, expert) paired with
command sequences for training text-conditioned CAD generation.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .base_loader import BaseCADDataset

_log = logging.getLogger(__name__)


class Text2CADLoader(BaseCADDataset):
    """Dataset loader for Text2CAD's text-annotated command sequences.

    Each sample contains a text description at one or more annotation
    levels paired with the command sequence that produces the CAD model.

    Supports two modes:
    1. Local mode: Load from JSON/JSONL files in root_dir
    2. Hub mode: Stream from HuggingFace Hub (e.g., "latticelabs/text2cad")

    Args:
        root_dir: Path to Text2CAD dataset directory (optional if hub_id provided).
        split: Dataset split ('train', 'val', 'test').
        annotation_level: Which level(s) to load ('abstract', 'intermediate',
            'detailed', 'expert', or 'all').
        max_seq_len: Maximum command sequence length.
        quantization_bits: Bits for parameter quantization.
        max_text_len: Maximum text token length.
        transform: Optional transform for each sample.
        download: Download if not present.
        hub_id: HuggingFace Hub dataset ID (e.g., "latticelabs/text2cad").
        streaming: Whether to stream from Hub (default True for Hub mode).
    """

    ANNOTATION_LEVELS = ("abstract", "intermediate", "detailed", "expert")

    def __init__(
        self,
        root_dir: Optional[str] = None,
        split: str = "train",
        annotation_level: str = "all",
        max_seq_len: int = 60,
        quantization_bits: int = 8,
        max_text_len: int = 128,
        transform: Optional[Callable] = None,
        download: bool = False,
        hub_id: Optional[str] = None,
        streaming: bool = True,
    ) -> None:
        self.annotation_level = annotation_level
        self.max_seq_len = max_seq_len
        self.quantization_bits = quantization_bits
        self.num_levels = 2 ** quantization_bits
        self.max_text_len = max_text_len

        self._samples: list[dict[str, Any]] = []

        super().__init__(root_dir, split, transform, download, hub_id, streaming)

        # Local mode only
        if not self._use_hub and self._verify_integrity():
            self._load_samples()

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a text-CAD pair sample.

        Returns:
            Dict with:
                'text': str, text description at requested level.
                'text_abstract': str, abstract-level annotation.
                'text_intermediate': str, intermediate-level annotation.
                'text_detailed': str, detailed-level annotation.
                'text_expert': str, expert-level annotation.
                'command_types': [max_seq_len] int array.
                'parameters': [max_seq_len, 16] int array.
                'mask': [max_seq_len] bool array.
                'annotation_level': str.
                'model_id': str.
        """
        sample = self._samples[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def download(self) -> None:
        """Download Text2CAD dataset (660K text-annotated CAD models).

        Attempts automated download in order of preference:

        1. **HuggingFace Hub** via ``huggingface_hub.snapshot_download``.
        2. **Manual fallback** — logs the URL and expected directory layout.

        Text2CAD data consists of JSONL files pairing four levels of text
        annotations (abstract, intermediate, detailed, expert) with
        command sequences.  After download the expected layout is::

            root_dir/
                train/
                    annotations.jsonl
                val/
                    annotations.jsonl
                test/
                    annotations.jsonl
        """
        if self._verify_integrity():
            _log.info("Text2CAD dataset already present at %s", self.root_dir)
            return

        self.root_dir.mkdir(parents=True, exist_ok=True)

        # --- Strategy 1: HuggingFace Hub ---
        try:
            from huggingface_hub import snapshot_download

            _log.info("Downloading Text2CAD dataset from HuggingFace Hub...")
            snapshot_download(
                repo_id="SadilKhan/Text2CAD",
                repo_type="dataset",
                local_dir=str(self.root_dir),
                allow_patterns=["*.jsonl", "*.json", "*.csv"],
            )

            if self._verify_integrity():
                _log.info("Text2CAD dataset downloaded via HuggingFace Hub")
                return

            # Handle data/ prefix layout
            data_dir = self.root_dir / "data"
            if data_dir.exists():
                import shutil

                for split_name in ("train", "val", "test"):
                    src = data_dir / split_name
                    dst = self.root_dir / split_name
                    if src.exists() and not dst.exists():
                        shutil.copytree(str(src), str(dst))
                if self._verify_integrity():
                    _log.info("Text2CAD restructured from data/ prefix")
                    return

        except ImportError:
            _log.debug("huggingface_hub not installed")
        except Exception as exc:
            _log.warning("HuggingFace Hub download failed: %s", exc)

        # --- Strategy 2: Manual instructions ---
        _log.info(
            "Automated download unsuccessful. Please download from "
            "https://github.com/SadilKhan/Text2CAD or HuggingFace Hub "
            "and place JSONL files in %s/{train,val,test}/",
            self.root_dir,
        )

    def _verify_integrity(self) -> bool:
        """Check if annotation files exist."""
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            return False
        # Look for JSONL or JSON annotation files
        return (
            any(split_dir.glob("*.jsonl"))
            or any(split_dir.glob("*.json"))
        )

    def _load_samples(self) -> None:
        """Load text-CAD paired annotations."""
        split_dir = self.root_dir / self.split

        # Try JSONL format first
        jsonl_files = sorted(split_dir.glob("*.jsonl"))
        if jsonl_files:
            for jsonl_path in jsonl_files:
                self._load_jsonl(jsonl_path)
        else:
            # Fall back to individual JSON files
            json_files = sorted(split_dir.glob("*.json"))
            for json_path in json_files:
                self._load_json(json_path)

        _log.info(
            "Loaded %d Text2CAD samples (level=%s)",
            len(self._samples), self.annotation_level,
        )

    def _load_jsonl(self, path: Path) -> None:
        """Load samples from a JSONL file."""
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    sample = self._process_annotation(data)
                    if sample is not None:
                        self._samples.append(sample)
                except (json.JSONDecodeError, KeyError) as e:
                    _log.debug("Skipping line: %s", e)

    def _load_json(self, path: Path) -> None:
        """Load samples from a JSON file."""
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    sample = self._process_annotation(item)
                    if sample is not None:
                        self._samples.append(sample)
            else:
                sample = self._process_annotation(data)
                if sample is not None:
                    self._samples.append(sample)
        except (json.JSONDecodeError, KeyError) as e:
            _log.debug("Skipping %s: %s", path.name, e)

    def _process_annotation(
        self, data: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Process a single text-CAD annotation entry."""
        # Extract text annotations
        texts = {}
        for level in self.ANNOTATION_LEVELS:
            texts[level] = data.get(f"text_{level}", data.get(level, ""))

        # Filter by annotation level
        if self.annotation_level != "all":
            if not texts.get(self.annotation_level, ""):
                return None

        # Extract command sequence
        commands = data.get("commands", data.get("sequence", []))
        if not commands:
            return None

        # Quantize commands
        cmd_types, params = self._quantize_commands(commands)
        num_commands = len(cmd_types)

        # Pad/truncate
        mask = np.ones(self.max_seq_len, dtype=np.float32)
        if num_commands > self.max_seq_len:
            cmd_types = cmd_types[:self.max_seq_len]
            params = params[:self.max_seq_len]
            num_commands = self.max_seq_len
        elif num_commands < self.max_seq_len:
            pad_len = self.max_seq_len - num_commands
            cmd_types = np.concatenate([cmd_types, np.full(pad_len, 5)])  # 5=EOS
            params = np.concatenate([params, np.zeros((pad_len, 16), dtype=np.int64)])
            mask[num_commands:] = 0.0

        # Select primary text
        primary_text = ""
        if self.annotation_level == "all":
            # Use the most detailed available
            for level in reversed(self.ANNOTATION_LEVELS):
                if texts[level]:
                    primary_text = texts[level]
                    break
        else:
            primary_text = texts.get(self.annotation_level, "")

        return {
            "text": primary_text,
            "text_abstract": texts.get("abstract", ""),
            "text_intermediate": texts.get("intermediate", ""),
            "text_detailed": texts.get("detailed", ""),
            "text_expert": texts.get("expert", ""),
            "command_types": cmd_types.astype(np.int64),
            "parameters": params.astype(np.int64),
            "mask": mask,
            "annotation_level": self.annotation_level,
            "model_id": data.get("model_id", data.get("id", "")),
        }

    def _quantize_commands(
        self, commands: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Quantize command parameters to discrete levels."""
        type_map = {"SOL": 0, "Line": 1, "Arc": 2, "Circle": 3, "Ext": 4, "EOS": 5}
        half_range = 1.0  # Assume pre-normalized

        cmd_types_list: list[int] = []
        params_list: list[list[int]] = []

        for cmd in commands:
            cmd_type_str = cmd.get("type", cmd.get("command", "EOS"))
            cmd_type_id = type_map.get(cmd_type_str, 5)
            cmd_types_list.append(cmd_type_id)

            raw_params = cmd.get("params", cmd.get("parameters", []))
            if isinstance(raw_params, dict):
                raw_params = list(raw_params.values())
            raw_params = [float(p) for p in raw_params]
            raw_params = raw_params[:16] + [0.0] * max(0, 16 - len(raw_params))

            q_params: list[int] = []
            for p in raw_params:
                p_clamped = max(-half_range, min(half_range, p))
                normalized = (p_clamped + half_range) / (2 * half_range)
                q = int(round(normalized * (self.num_levels - 1)))
                q_params.append(max(0, min(self.num_levels - 1, q)))
            params_list.append(q_params)

        return np.array(cmd_types_list), np.array(params_list)

    def _process_hub_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Process a sample from Hub format to loader format.

        Hub format has:
        - text: str (primary text)
        - text_abstract: str
        - text_intermediate: str
        - text_detailed: str
        - text_expert: str
        - command_types: [seq_len] int8 array
        - parameters: [seq_len * num_params] int16 array (flat)
        - mask: [seq_len] float32 array
        - num_commands: int
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
            if len(params) == seq_len * 16:
                params = params.reshape(seq_len, 16)

        # Get text at requested level
        primary_text = sample.get("text", "")
        if self.annotation_level != "all" and self.annotation_level in self.ANNOTATION_LEVELS:
            primary_text = sample.get(f"text_{self.annotation_level}", primary_text)

        return {
            "text": primary_text,
            "text_abstract": sample.get("text_abstract", ""),
            "text_intermediate": sample.get("text_intermediate", ""),
            "text_detailed": sample.get("text_detailed", ""),
            "text_expert": sample.get("text_expert", ""),
            "command_types": cmd_types,
            "parameters": params,
            "mask": mask,
            "annotation_level": self.annotation_level,
            "model_id": sample.get("model_id", sample.get("sample_id", "")),
        }
