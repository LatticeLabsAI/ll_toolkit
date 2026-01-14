"""
Data loading utilities for STEP files.
Handles dataset creation and batching.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

from .tokenizer import STEPTokenizer
from .features import STEPFeatureExtractor
from .topology import STEPTopologyBuilder


class STEPDataset(Dataset):
    """
    PyTorch Dataset for STEP files.
    Loads and preprocesses STEP files on-the-fly.
    """

    def __init__(
        self,
        file_paths: List[str],
        labels: Optional[List] = None,
        tokenizer: Optional[STEPTokenizer] = None,
        max_length: int = 2048,
        use_topology: bool = True
    ):
        """
        Args:
            file_paths: List of paths to STEP files
            labels: Optional labels for supervised learning
            tokenizer: STEPTokenizer instance (creates default if None)
            max_length: Maximum sequence length
            use_topology: Whether to build topology graphs
        """
        self.file_paths = file_paths
        self.labels = labels
        self.max_length = max_length
        self.use_topology = use_topology

        # Initialize processors
        self.tokenizer = tokenizer or STEPTokenizer()
        self.feature_extractor = STEPFeatureExtractor()
        self.topology_builder = STEPTopologyBuilder() if use_topology else None

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and process a single STEP file.

        Returns:
            Dictionary with:
                - token_ids: [seq_len] token sequence
                - attention_mask: [seq_len] mask for padding
                - topology_data: Optional topology dictionary
                - label: Optional label
        """
        file_path = self.file_paths[idx]

        # Read STEP file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Find DATA section
        if 'DATA;' in content and 'ENDSEC;' in content:
            data_start = content.index('DATA;') + 5
            data_end = content.index('ENDSEC;', data_start)
            chunk_text = content[data_start:data_end].strip()
        else:
            chunk_text = content

        # Tokenize
        token_ids = self.tokenizer.encode(chunk_text)

        # Truncate or pad to max_length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]

        attention_mask = [1] * len(token_ids)

        # Pad if needed
        padding_length = self.max_length - len(token_ids)
        if padding_length > 0:
            token_ids.extend([0] * padding_length)
            attention_mask.extend([0] * padding_length)

        result = {
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }

        # Extract features and build topology
        if self.use_topology:
            features_list = self.feature_extractor.extract_features_from_chunk(chunk_text)
            topology_data = self.topology_builder.build_complete_topology(features_list)
            result['topology_data'] = topology_data

        # Add label if provided
        if self.labels is not None:
            result['label'] = torch.tensor(self.labels[idx])

        return result


class STEPCollator:
    """
    Collate function for batching STEP data.
    Handles variable-length sequences and topology graphs.
    """

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of STEP samples.

        Args:
            batch: List of samples from STEPDataset

        Returns:
            Batched tensors
        """
        # Stack token sequences
        token_ids = torch.stack([item['token_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])

        result = {
            'token_ids': token_ids,
            'attention_mask': attention_mask
        }

        # Handle topology data (can't batch easily, return as list)
        if 'topology_data' in batch[0]:
            result['topology_data'] = [item['topology_data'] for item in batch]

        # Handle labels
        if 'label' in batch[0]:
            result['labels'] = torch.stack([item['label'] for item in batch])

        return result


def create_dataloader(
    file_paths: List[str],
    labels: Optional[List] = None,
    batch_size: int = 8,
    max_length: int = 2048,
    use_topology: bool = True,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create DataLoader for STEP files.

    Args:
        file_paths: List of STEP file paths
        labels: Optional labels
        batch_size: Batch size
        max_length: Maximum sequence length
        use_topology: Whether to build topology graphs
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes

    Returns:
        DataLoader instance
    """
    dataset = STEPDataset(
        file_paths=file_paths,
        labels=labels,
        max_length=max_length,
        use_topology=use_topology
    )

    collator = STEPCollator()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator
    )


def load_dataset_from_directory(
    data_dir: str,
    split: str = 'train',
    label_file: Optional[str] = None
) -> Tuple[List[str], Optional[List]]:
    """
    Load dataset from directory structure.

    Expected structure:
        data_dir/
            train/
                *.step, *.stp
            val/
                *.step, *.stp
            labels.json  # Optional

    Args:
        data_dir: Root data directory
        split: 'train' or 'val'
        label_file: Optional JSON file with labels

    Returns:
        Tuple of (file_paths, labels)
    """
    data_path = Path(data_dir) / split

    # Find all STEP files
    file_paths = []
    for ext in ['*.step', '*.stp']:
        file_paths.extend(str(p) for p in data_path.glob(ext))

    file_paths.sort()

    # Load labels if provided
    labels = None
    if label_file:
        label_path = Path(data_dir) / label_file
        if label_path.exists():
            with open(label_path, 'r') as f:
                label_data = json.load(f)

            # Map file names to labels
            labels = []
            for fp in file_paths:
                file_name = Path(fp).name
                labels.append(label_data.get(file_name, 0))

    return file_paths, labels
