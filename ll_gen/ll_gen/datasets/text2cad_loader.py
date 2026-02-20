from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["Text2CADDataset", "load_text2cad"]

logger = logging.getLogger(__name__)

# Lazy imports
_torch = None
_datasets = None
_geotoken = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _get_datasets():
    global _datasets
    if _datasets is None:
        import datasets
        _datasets = datasets
    return _datasets


def _get_geotoken():
    global _geotoken
    if _geotoken is None:
        import geotoken
        _geotoken = geotoken
    return _geotoken


# Special token IDs
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2

# Command type IDs
COMMAND_TYPE_IDS = {
    "SOL": 6,
    "LINE": 7,
    "ARC": 8,
    "CIRCLE": 9,
    "EXTRUDE": 10,
    "EOS": 11,
}

# Parameter tokens start at offset 12
PARAM_OFFSET = 12

# Annotation levels
ANNOTATION_LEVELS = {
    "abstract": "text_abstract",
    "detailed": "text_detailed",
    "expert": "text_expert",
}


class Text2CADDataset:
    """PyTorch Dataset for Text2CAD annotated CAD sequences.

    This dataset loads JSON files containing text annotations and CAD
    sequences, supporting multiple levels of text detail (abstract,
    detailed, expert).

    Attributes:
        data_dir: Directory containing JSON files organized by split.
        split: Dataset split ("train", "val", "test").
        annotation_level: Level of text annotation to use.
        max_commands: Maximum number of commands to include.
        quantization_bits: Number of bits for parameter quantization.
        normalization_range: Range for normalizing parameters.
        max_samples: Maximum number of samples to load.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        annotation_level: str = "detailed",
        max_commands: int = 60,
        quantization_bits: int = 8,
        normalization_range: float = 2.0,
        max_samples: Optional[int] = None,
    ):
        """Initialize the Text2CAD dataset.

        Args:
            data_dir: Path to directory containing Text2CAD data.
            split: Dataset split to load ("train", "val", "test").
            annotation_level: Text annotation level to use
                ("abstract", "detailed", "expert").
            max_commands: Maximum number of commands per sequence.
            quantization_bits: Bits for parameter quantization.
            normalization_range: Range for parameter normalization.
            max_samples: Limit number of samples loaded.

        Raises:
            FileNotFoundError: If split directory does not exist.
            ValueError: If no JSON files found or invalid annotation level.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.annotation_level = annotation_level
        self.max_commands = max_commands
        self.quantization_bits = quantization_bits
        self.normalization_range = normalization_range
        self.max_samples = max_samples

        if annotation_level not in ANNOTATION_LEVELS:
            raise ValueError(
                f"Invalid annotation_level: {annotation_level}. "
                f"Must be one of {list(ANNOTATION_LEVELS.keys())}"
            )

        self.text_key = ANNOTATION_LEVELS[annotation_level]

        # Scan for JSON files
        split_dir = self.data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.json_files = sorted(split_dir.glob("**/*.json"))
        if not self.json_files:
            raise ValueError(f"No JSON files found in {split_dir}")

        if max_samples is not None:
            self.json_files = self.json_files[:max_samples]

        logger.info(
            f"Loaded {len(self.json_files)} Text2CAD samples from {split} "
            f"with annotation level '{annotation_level}'"
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.json_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary containing:
                - text: Text annotation string
                - token_ids: List of CAD sequence token IDs
                - command_tokens: List of command dictionaries
                - attention_mask: List of attention mask values
                - num_commands: Number of commands in sequence
                - metadata: Dictionary with file path and other info
        """
        json_file = self.json_files[idx]

        with open(json_file, "r") as f:
            data = json.load(f)

        # Get text annotation
        text = data.get(self.text_key, "")

        sequence = data.get("sequence", [])
        token_ids = [BOS_TOKEN_ID]
        command_tokens = []
        attention_mask = [1]

        # Quantization levels
        quantization_levels = 2**self.quantization_bits

        for i, cmd in enumerate(sequence):
            if i >= self.max_commands:
                break

            cmd_type = cmd.get("type", "")
            params = cmd.get("params", [])

            # Add command type token
            if cmd_type in COMMAND_TYPE_IDS:
                cmd_type_id = COMMAND_TYPE_IDS[cmd_type]
                token_ids.append(cmd_type_id)
                attention_mask.append(1)

                # Process parameters
                quantized_params = []
                param_mask = []

                for param in params:
                    # Normalize parameter to [0, 1]
                    try:
                        normalized = float(param) / self.normalization_range
                        normalized = max(0.0, min(1.0, normalized))
                    except (ValueError, TypeError):
                        normalized = 0.0

                    # Quantize to discrete levels
                    quantized = round(
                        normalized * (quantization_levels - 1)
                    )
                    quantized = max(0, min(quantization_levels - 1, quantized))
                    quantized_params.append(quantized)
                    param_mask.append(1)

                    # Add parameter token
                    param_token_id = PARAM_OFFSET + quantized
                    token_ids.append(param_token_id)
                    attention_mask.append(1)

                command_tokens.append({
                    "command_type": cmd_type_id,
                    "parameters": quantized_params,
                    "parameter_mask": param_mask,
                })

        # Add EOS token
        token_ids.append(EOS_TOKEN_ID)
        attention_mask.append(1)

        num_commands = len(command_tokens)

        # Pad to max_commands
        while len(token_ids) < (self.max_commands * 10):
            token_ids.append(PAD_TOKEN_ID)
            attention_mask.append(0)

        token_ids = token_ids[: self.max_commands * 10]
        attention_mask = attention_mask[: self.max_commands * 10]

        return {
            "text": text,
            "token_ids": token_ids,
            "command_tokens": command_tokens,
            "attention_mask": attention_mask,
            "num_commands": num_commands,
            "metadata": {
                "file_path": str(json_file),
                "split": self.split,
                "annotation_level": self.annotation_level,
            },
        }


def _tokenize_text2cad_sample(
    sample: Dict[str, Any],
    annotation_level: str = "detailed",
    max_commands: int = 60,
    quantization_bits: int = 8,
    normalization_range: float = 2.0,
) -> Dict[str, Any]:
    """Tokenize a single Text2CAD sample from HuggingFace.

    Args:
        sample: Sample dictionary with text and sequence.
        annotation_level: Text annotation level to use.
        max_commands: Maximum number of commands.
        quantization_bits: Bits for quantization.
        normalization_range: Normalization range.

    Returns:
        Dictionary with tokenized output.
    """
    text_key = ANNOTATION_LEVELS.get(
        annotation_level, "text_detailed"
    )
    text = sample.get(text_key, "")

    sequence = sample.get("sequence", [])
    token_ids = [BOS_TOKEN_ID]
    command_tokens = []
    attention_mask = [1]

    quantization_levels = 2**quantization_bits

    for i, cmd in enumerate(sequence):
        if i >= max_commands:
            break

        cmd_type = cmd.get("type", "")
        params = cmd.get("params", [])

        if cmd_type in COMMAND_TYPE_IDS:
            cmd_type_id = COMMAND_TYPE_IDS[cmd_type]
            token_ids.append(cmd_type_id)
            attention_mask.append(1)

            quantized_params = []
            param_mask = []

            for param in params:
                try:
                    normalized = float(param) / normalization_range
                    normalized = max(0.0, min(1.0, normalized))
                except (ValueError, TypeError):
                    normalized = 0.0

                quantized = round(normalized * (quantization_levels - 1))
                quantized = max(0, min(quantization_levels - 1, quantized))
                quantized_params.append(int(quantized))
                param_mask.append(1)

                param_token_id = PARAM_OFFSET + quantized
                token_ids.append(param_token_id)
                attention_mask.append(1)

            command_tokens.append({
                "command_type": cmd_type_id,
                "parameters": quantized_params,
                "parameter_mask": param_mask,
            })

    token_ids.append(EOS_TOKEN_ID)
    attention_mask.append(1)

    num_commands = len(command_tokens)

    while len(token_ids) < (max_commands * 10):
        token_ids.append(PAD_TOKEN_ID)
        attention_mask.append(0)

    token_ids = token_ids[: max_commands * 10]
    attention_mask = attention_mask[: max_commands * 10]

    return {
        "text": text,
        "token_ids": token_ids,
        "command_tokens": command_tokens,
        "attention_mask": attention_mask,
        "num_commands": num_commands,
    }


def load_text2cad(
    path: str = "latticelabs/text2cad",
    split: str = "train",
    streaming: bool = True,
    annotation_level: str = "detailed",
    max_commands: int = 60,
    quantization_bits: int = 8,
    normalization_range: float = 2.0,
    max_samples: Optional[int] = None,
) -> Any:
    """Load the Text2CAD dataset.

    Loads either from a local directory or from the HuggingFace Hub,
    returning a PyTorch Dataset or HuggingFace IterableDataset.

    Args:
        path: Path to local directory or HuggingFace Hub ID.
        split: Dataset split to load ("train", "val", "test").
        streaming: Whether to stream from HuggingFace (if using Hub).
        annotation_level: Text annotation level ("abstract", "detailed", "expert").
        max_commands: Maximum number of commands per sequence.
        quantization_bits: Bits for parameter quantization.
        normalization_range: Range for parameter normalization.
        max_samples: Maximum number of samples to load.

    Returns:
        PyTorch Dataset if path is local directory,
        HuggingFace IterableDataset if path is Hub ID.

    Raises:
        FileNotFoundError: If local path does not exist.
        ValueError: If no data found or invalid annotation level.
    """
    # Check if path is a local directory
    if os.path.isdir(path):
        logger.info(f"Loading Text2CAD from local directory: {path}")
        return Text2CADDataset(
            data_dir=path,
            split=split,
            annotation_level=annotation_level,
            max_commands=max_commands,
            quantization_bits=quantization_bits,
            normalization_range=normalization_range,
            max_samples=max_samples,
        )
    else:
        # Load from HuggingFace Hub
        logger.info(f"Loading Text2CAD from HuggingFace Hub: {path}")
        datasets = _get_datasets()

        hf_dataset = datasets.load_dataset(
            path, split=split, streaming=streaming
        )

        if max_samples is not None:
            hf_dataset = hf_dataset.take(max_samples)

        tokenized_dataset = hf_dataset.map(
            lambda sample: _tokenize_text2cad_sample(
                sample,
                annotation_level=annotation_level,
                max_commands=max_commands,
                quantization_bits=quantization_bits,
                normalization_range=normalization_range,
            ),
            remove_columns=hf_dataset.column_names,
        )

        return tokenized_dataset
