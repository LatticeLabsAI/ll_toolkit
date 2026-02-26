"""
Data loading utilities for STEP files.
Handles dataset creation and batching.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

from .tokenizer import STEPTokenizer
from .features import STEPFeatureExtractor
from .topology import STEPTopologyBuilder
from .config import STEPReserializationConfig, STEPAnnotationConfig
from .reserialization import STEPEntityGraph, STEPDFSSerializer
from .annotations import STEPStructuralAnnotator


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
        use_topology: bool = True,
        use_reserialization: bool = False,
        use_annotations: bool = False,
        reserialization_config: Optional[STEPReserializationConfig] = None,
        annotation_config: Optional[STEPAnnotationConfig] = None,
    ):
        """
        Args:
            file_paths: List of paths to STEP files
            labels: Optional labels for supervised learning
            tokenizer: STEPTokenizer instance (creates default if None)
            max_length: Maximum sequence length
            use_topology: Whether to build topology graphs
            use_reserialization: Whether to apply DFS reserialization to entity text
            use_annotations: Whether to prepend structural annotations
            reserialization_config: Configuration for DFS reserialization
            annotation_config: Configuration for structural annotations
        """
        self.file_paths = file_paths
        self.labels = labels
        self.max_length = max_length
        self.use_topology = use_topology
        self.use_reserialization = use_reserialization
        self.use_annotations = use_annotations

        # Initialize processors
        self.tokenizer = tokenizer or STEPTokenizer()
        self.feature_extractor = STEPFeatureExtractor()
        self.topology_builder = STEPTopologyBuilder() if use_topology else None

        # Initialize reserialization and annotation processors
        self.reserialization_config = reserialization_config or STEPReserializationConfig()
        self.annotation_config = annotation_config or STEPAnnotationConfig()
        self.serializer = STEPDFSSerializer(self.reserialization_config) if use_reserialization else None
        self.annotator = STEPStructuralAnnotator(self.annotation_config) if use_annotations else None

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
        if 'DATA;' in content:
            data_start = content.index('DATA;') + 5
            endsec_pos = content.find('ENDSEC;', data_start)
            if endsec_pos != -1:
                chunk_text = content[data_start:endsec_pos].strip()
            else:
                chunk_text = content[data_start:].strip()
        else:
            chunk_text = content

        # Apply DFS reserialization if enabled
        if self.use_reserialization and self.serializer is not None:
            graph = STEPEntityGraph.parse(chunk_text)
            reserialized = self.serializer.serialize(graph)

            # Apply structural annotations if enabled
            if self.use_annotations and self.annotator is not None:
                annotated = self.annotator.annotate(graph, reserialized)
                chunk_text = annotated.format()
            else:
                chunk_text = reserialized.text

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


class GeoTokenDataset(Dataset):
    """PyTorch Dataset for geotoken TokenSequence objects.

    Consumes geotoken TokenSequences directly — no format conversion
    needed because ll_stepnet's CommandType enum and PARAMETER_MASKS
    match geotoken's natively.

    The integer index of each geotoken CommandType enum member maps
    directly to ll_stepnet's CommandType IntEnum:
        SOL=0, LINE=1, ARC=2, CIRCLE=3, EXTRUDE=4, EOS=5

    Each item is a dictionary containing:
        - command_types: [seq_len] integer command type IDs
        - parameters: [seq_len, 16] quantized parameter values
        - parameter_mask: [seq_len, 16] boolean active-parameter mask
        - attention_mask: [seq_len] validity mask (1=real, 0=padding)

    When ``encode_graph_tokens=True`` and the TokenSequence has graph
    tokens, the item also contains:
        - graph_token_ids: [variable] integer IDs from CADVocabulary

    When ``encode_constraint_tokens=True`` and the TokenSequence has
    constraint tokens, the item also contains:
        - constraint_token_ids: [variable] integer IDs from CADVocabulary

    Args:
        token_sequences: List of geotoken TokenSequence objects.
        max_commands: Maximum command sequence length (pad/truncate). Default 60.
        labels: Optional labels for supervised learning.
        encode_graph_tokens: Encode graph tokens via CADVocabulary. Default False.
        encode_constraint_tokens: Encode constraint tokens via CADVocabulary. Default False.
    """

    # Canonical ordering of geotoken CommandType enum members.
    # This matches ll_stepnet's CommandType IntEnum exactly.
    _GEOTOKEN_TYPE_ORDER = ["SOL", "LINE", "ARC", "CIRCLE", "EXTRUDE", "EOS"]

    def __init__(
        self,
        token_sequences: List,
        max_commands: int = 60,
        labels: Optional[List] = None,
        encode_graph_tokens: bool = False,
        encode_constraint_tokens: bool = False,
    ) -> None:
        self.token_sequences = token_sequences
        self.max_commands = max_commands
        self.labels = labels
        self.encode_graph_tokens = encode_graph_tokens
        self.encode_constraint_tokens = encode_constraint_tokens
        # Build lookup for geotoken string → integer index
        self._type_to_int = {name: i for i, name in enumerate(self._GEOTOKEN_TYPE_ORDER)}
        # Cached vocab instance (lazy-initialised on first use)
        self._vocab = None

    def _get_vocab(self):
        """Lazy-import and cache a CADVocabulary instance."""
        if self._vocab is None:
            from geotoken.tokenizer.vocabulary import CADVocabulary
            self._vocab = CADVocabulary()
        return self._vocab

    def __len__(self) -> int:
        return len(self.token_sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.token_sequences[idx]
        cmd_tokens = getattr(seq, 'command_tokens', [])

        cmd_types = []
        params = []
        param_masks = []

        for ct in cmd_tokens[:self.max_commands]:
            # Direct integer index — geotoken enum order == ll_stepnet enum order
            type_name = ct.command_type.value if hasattr(ct.command_type, 'value') else str(ct.command_type)
            cmd_types.append(self._type_to_int.get(type_name, 5))

            p = list(ct.parameters[:16])
            p.extend([0] * (16 - len(p)))
            params.append(p)

            m = list(ct.parameter_mask[:16])
            m.extend([False] * (16 - len(m)))
            param_masks.append(m)

        actual_len = len(cmd_types)
        while len(cmd_types) < self.max_commands:
            cmd_types.append(0)
            params.append([0] * 16)
            param_masks.append([False] * 16)

        attention_mask = [1] * actual_len + [0] * (self.max_commands - actual_len)

        result = {
            'command_types': torch.tensor(cmd_types, dtype=torch.long),
            'parameters': torch.tensor(params, dtype=torch.long),
            'parameter_mask': torch.tensor(param_masks, dtype=torch.bool),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }

        # --- Graph token encoding (optional) ---
        if self.encode_graph_tokens:
            graph_nodes = getattr(seq, 'graph_node_tokens', None) or []
            graph_struct = getattr(seq, 'graph_structure_tokens', None) or []
            if graph_nodes or graph_struct:
                vocab = self._get_vocab()
                ids = []
                if graph_struct:
                    ids.extend(vocab.encode_graph_structure(graph_struct))
                if graph_nodes:
                    ids.extend(vocab.encode_graph_node_features(graph_nodes))
                # Edge tokens
                graph_edges = getattr(seq, 'graph_edge_tokens', None) or []
                if graph_edges:
                    ids.extend(vocab.encode_graph_edge_features(graph_edges))
                result['graph_token_ids'] = torch.tensor(ids, dtype=torch.long)

        # --- Constraint token encoding (optional) ---
        if self.encode_constraint_tokens:
            constraints = getattr(seq, 'constraint_tokens', None) or []
            if constraints:
                vocab = self._get_vocab()
                cids = vocab.encode_constraints(constraints)
                result['constraint_token_ids'] = torch.tensor(cids, dtype=torch.long)

        if self.labels is not None:
            result['label'] = torch.tensor(self.labels[idx])

        return result


class CadlingDataset(Dataset):
    """PyTorch Dataset for cadling Sketch2DItem objects.

    Accepts a list of cadling ``Sketch2DItem`` instances and converts
    each one to the same dict format as :class:`GeoTokenDataset` by
    calling ``item.to_geotoken_commands()`` to get the command sequence
    in geotoken-compatible format.

    This lets you train ll_stepnet's generative models (STEPVAE,
    SkexGenVQVAE, etc.) directly on cadling's in-memory geometry
    objects without writing them to disk first.

    Each ``__getitem__`` returns:
        - command_types: ``[max_commands]`` integer command type IDs
        - parameters: ``[max_commands, 16]`` parameter values
        - parameter_mask: ``[max_commands, 16]`` active-parameter mask
        - attention_mask: ``[max_commands]`` validity mask

    Args:
        sketch_items: List of cadling Sketch2DItem objects (or any object
            with a ``to_geotoken_commands()`` method).
        max_commands: Maximum command sequence length. Default 60.
        include_topology: If True, build topology and include in output.
        labels: Optional labels for supervised learning.
    """

    _GEOTOKEN_TYPE_ORDER = ["SOL", "LINE", "ARC", "CIRCLE", "EXTRUDE", "EOS"]

    # Active parameter indices per command type — matches PARAMETER_MASKS
    _ACTIVE_INDICES = {
        "SOL": {0, 1},
        "LINE": {0, 1, 2, 3},
        "ARC": {0, 1, 2, 3, 4, 5},
        "CIRCLE": {0, 1, 2},
        "EXTRUDE": {0, 1, 2, 3, 4, 5, 6, 7},
        "EOS": set(),
    }

    def __init__(
        self,
        sketch_items: List,
        max_commands: int = 60,
        include_topology: bool = False,
        labels: Optional[List] = None,
    ) -> None:
        self.sketch_items = sketch_items
        self.max_commands = max_commands
        self.include_topology = include_topology
        self.labels = labels
        self._type_to_int = {name: i for i, name in enumerate(self._GEOTOKEN_TYPE_ORDER)}

    def __len__(self) -> int:
        return len(self.sketch_items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.sketch_items[idx]

        # Get command sequence via cadling's native method
        commands = item.to_geotoken_commands()  # List[Dict] with {type, params}

        cmd_types = []
        params = []
        param_masks = []

        for cmd in commands[:self.max_commands]:
            cmd_type_str = cmd.get("type", "EOS")
            cmd_types.append(self._type_to_int.get(cmd_type_str, 5))

            # Extract parameters (padded to 16)
            p = list(cmd.get("params", []))[:16]
            p.extend([0] * (16 - len(p)))
            # Convert to int (quantised values expected)
            params.append([int(v) if isinstance(v, (int, float)) else 0 for v in p])

            # Build parameter mask from known active indices
            active = self._ACTIVE_INDICES.get(cmd_type_str, set())
            param_masks.append([i in active for i in range(16)])

        actual_len = len(cmd_types)
        while len(cmd_types) < self.max_commands:
            cmd_types.append(0)
            params.append([0] * 16)
            param_masks.append([False] * 16)

        attention_mask = [1] * actual_len + [0] * (self.max_commands - actual_len)

        result = {
            'command_types': torch.tensor(cmd_types, dtype=torch.long),
            'parameters': torch.tensor(params, dtype=torch.long),
            'parameter_mask': torch.tensor(param_masks, dtype=torch.bool),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }

        # Optionally build topology from the item's topology data
        if self.include_topology:
            topo_graph = getattr(item, 'topology_graph', None)
            if topo_graph is not None:
                # Use STEPEncoder.prepare_topology_data for direct conversion
                from .encoder import STEPEncoder
                result['topology_data'] = STEPEncoder.prepare_topology_data(topo_graph)

        if self.labels is not None:
            result['label'] = torch.tensor(self.labels[idx])

        return result


class GeoTokenCollator:
    """Collate function for batching GeoTokenDataset samples.

    Handles variable-length graph and constraint token sequences by
    padding to the maximum length in the batch.

    Args:
        pad_token_id: Token ID for padding. Default 0.
    """

    def __init__(self, pad_token_id: int = 0) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        result = {
            'command_types': torch.stack([item['command_types'] for item in batch]),
            'parameters': torch.stack([item['parameters'] for item in batch]),
            'parameter_mask': torch.stack([item['parameter_mask'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        }

        # Pad variable-length graph token IDs
        if 'graph_token_ids' in batch[0]:
            max_graph_len = max(item['graph_token_ids'].shape[0] for item in batch)
            padded = []
            for item in batch:
                ids = item['graph_token_ids']
                pad_len = max_graph_len - ids.shape[0]
                if pad_len > 0:
                    ids = torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
                padded.append(ids)
            result['graph_token_ids'] = torch.stack(padded)

        # Pad variable-length constraint token IDs
        if 'constraint_token_ids' in batch[0]:
            max_cstr_len = max(item['constraint_token_ids'].shape[0] for item in batch)
            padded = []
            for item in batch:
                ids = item['constraint_token_ids']
                pad_len = max_cstr_len - ids.shape[0]
                if pad_len > 0:
                    ids = torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
                padded.append(ids)
            result['constraint_token_ids'] = torch.stack(padded)

        # Handle labels
        if 'label' in batch[0]:
            result['labels'] = torch.stack([item['label'] for item in batch])

        return result
