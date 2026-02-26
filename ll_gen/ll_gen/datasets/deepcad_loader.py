from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["DeepCADDataset", "load_deepcad"]

_log = logging.getLogger(__name__)

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


from ll_gen.datasets._tokenization import (
    PAD_TOKEN_ID,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    COMMAND_TYPE_IDS,
    PARAM_OFFSET,
    quantize_parameter,
    tokenize_command_sequence,
    validate_token_space,
)


class DeepCADDataset:
    """PyTorch Dataset for DeepCAD sketch-and-extrude sequences.

    This dataset loads JSON files containing sequences of CAD operations
    (SOL, LINE, ARC, CIRCLE, EXTRUDE, EOS) and tokenizes them for use
    with geotoken vocabulary.

    Attributes:
        data_dir: Directory containing JSON files organized by split.
        split: Dataset split ("train", "val", "test").
        max_commands: Maximum number of commands to include in a sequence.
        quantization_bits: Number of bits for quantizing parameters.
        normalization_range: Range for normalizing parameters (typically 2.0).
        max_samples: Maximum number of samples to load (None for all).
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_commands: int = 60,
        quantization_bits: int = 8,
        normalization_range: float = 2.0,
        max_samples: Optional[int] = None,
    ):
        """Initialize the DeepCAD dataset.

        Args:
            data_dir: Path to the directory containing DeepCAD data.
            split: Dataset split to load ("train", "val", "test").
            max_commands: Maximum number of commands per sequence.
            quantization_bits: Bits for parameter quantization.
            normalization_range: Range for parameter normalization.
            max_samples: Limit number of samples loaded.

        Raises:
            FileNotFoundError: If the split directory does not exist.
            ValueError: If no JSON files are found in the split directory.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_commands = max_commands
        self.quantization_bits = quantization_bits
        self.normalization_range = normalization_range
        self.max_samples = max_samples

        # Scan for JSON files
        split_dir = self.data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.json_files = sorted(split_dir.glob("**/*.json"))
        if not self.json_files:
            raise ValueError(f"No JSON files found in {split_dir}")

        if max_samples is not None:
            self.json_files = self.json_files[:max_samples]

        _log.info(
            f"Loaded {len(self.json_files)} DeepCAD samples from {split}"
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
                - token_ids: List of token IDs (int)
                - command_tokens: List of command dictionaries
                - attention_mask: List of attention mask values
                - num_commands: Number of commands in sequence
                - metadata: Dictionary with file path and other info
        """
        json_file = self.json_files[idx]

        with open(json_file, "r") as f:
            data = json.load(f)

        sequence = data.get("sequence", [])
        result = tokenize_command_sequence(
            sequence,
            quantization_bits=self.quantization_bits,
            normalization_range=self.normalization_range,
            max_commands=self.max_commands,
        )
        result["metadata"] = {
            "file_path": str(json_file),
            "split": self.split,
        }
        return result


def _tokenize_deepcad_sample(
    sample: Dict[str, Any],
    max_commands: int = 60,
    quantization_bits: int = 8,
    normalization_range: float = 2.0,
) -> Dict[str, Any]:
    """Tokenize a single DeepCAD sample from HuggingFace.

    Args:
        sample: Sample dictionary with 'sequence' key.
        max_commands: Maximum number of commands.
        quantization_bits: Bits for quantization.
        normalization_range: Normalization range.

    Returns:
        Dictionary with tokenized output.
    """
    sequence = sample.get("sequence", [])
    return tokenize_command_sequence(
        sequence,
        quantization_bits=quantization_bits,
        normalization_range=normalization_range,
        max_commands=max_commands,
    )


def load_deepcad(
    path: str = "latticelabs/deepcad",
    split: str = "train",
    streaming: bool = True,
    max_samples: Optional[int] = None,
    max_commands: int = 60,
    quantization_bits: int = 8,
    normalization_range: float = 2.0,
) -> Any:
    """Load the DeepCAD dataset.

    Loads either from a local directory or from the HuggingFace Hub,
    returning a PyTorch Dataset or HuggingFace IterableDataset.

    Args:
        path: Path to local directory or HuggingFace Hub ID.
        split: Dataset split to load ("train", "val", "test").
        streaming: Whether to stream from HuggingFace (if using Hub).
        max_samples: Maximum number of samples to load.
        max_commands: Maximum number of commands per sequence.
        quantization_bits: Bits for parameter quantization.
        normalization_range: Range for parameter normalization.

    Returns:
        PyTorch Dataset if path is local directory,
        HuggingFace IterableDataset if path is Hub ID.

    Raises:
        FileNotFoundError: If local path does not exist.
        ValueError: If no data found in specified location.
    """
    # Check if path is a local directory
    if os.path.isdir(path):
        _log.info(f"Loading DeepCAD from local directory: {path}")
        return DeepCADDataset(
            data_dir=path,
            split=split,
            max_commands=max_commands,
            quantization_bits=quantization_bits,
            normalization_range=normalization_range,
            max_samples=max_samples,
        )
    else:
        # Load from HuggingFace Hub
        _log.info(f"Loading DeepCAD from HuggingFace Hub: {path}")
        datasets = _get_datasets()

        hf_dataset = datasets.load_dataset(
            path, split=split, streaming=streaming
        )

        if max_samples is not None:
            hf_dataset = hf_dataset.take(max_samples)

        tokenized_dataset = hf_dataset.map(
            lambda sample: _tokenize_deepcad_sample(
                sample,
                max_commands=max_commands,
                quantization_bits=quantization_bits,
                normalization_range=normalization_range,
            ),
            remove_columns=hf_dataset.column_names,
        )

        return tokenized_dataset
