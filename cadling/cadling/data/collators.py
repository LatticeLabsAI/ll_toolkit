"""Collators for batching CAD data.

Provides specialized collate functions for different CAD data types:
- CADCommandCollator: Dynamic padding and attention masks for sequences
- CADMultiModalCollator: Text + geometry batching for conditioned generation
- CADGraphCollator: PyTorch Geometric graph batching
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

_log = logging.getLogger(__name__)

# Lazy imports
_torch = None
_pyg = None


def _ensure_torch():
    """Lazily import torch."""
    global _torch
    if _torch is None:
        try:
            import torch
            _torch = torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for collators. "
                "Install via: conda install pytorch -c conda-forge"
            )
    return _torch


def _ensure_pyg():
    """Lazily import torch_geometric."""
    global _pyg
    if _pyg is None:
        try:
            import torch_geometric
            _pyg = torch_geometric
        except ImportError:
            _pyg = False
    return _pyg


@dataclass
class CADCollatorConfig:
    """Configuration for CAD collators.

    Attributes:
        pad_token_id: Token ID used for padding (default 0).
        max_seq_len: Maximum sequence length for truncation.
        num_params: Number of parameters per command.
        pad_to_max: Whether to pad all sequences to max_seq_len.
        return_tensors: Output format ('pt' for PyTorch).
        include_position_ids: Whether to generate position IDs.
        create_labels: Whether to create shifted labels for autoregressive training.
            When True, creates 'labels' and 'input_ids' with proper shifting.
    """

    pad_token_id: int = 0
    max_seq_len: int = 60
    num_params: int = 16
    pad_to_max: bool = False
    return_tensors: str = "pt"
    include_position_ids: bool = False
    create_labels: bool = False


class CADCommandCollator:
    """Collator for CAD command sequence batching.

    Handles dynamic padding and attention mask generation for
    variable-length command sequences. Compatible with HuggingFace
    Trainer and PyTorch DataLoader.

    Usage:
        >>> collator = CADCommandCollator(pad_token_id=0)
        >>> batch = collator([sample1, sample2, sample3])
        >>> print(batch["command_types"].shape)  # [batch_size, seq_len]
        >>> print(batch["attention_mask"].shape)  # [batch_size, seq_len]

    Args:
        config: Collator configuration.
    """

    def __init__(
        self,
        config: Optional[CADCollatorConfig] = None,
        pad_token_id: int = 0,
        max_seq_len: int = 60,
        num_params: int = 16,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = CADCollatorConfig(
                pad_token_id=pad_token_id,
                max_seq_len=max_seq_len,
                num_params=num_params,
            )

    def __call__(
        self, batch: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Collate a batch of samples.

        Args:
            batch: List of sample dictionaries.

        Returns:
            Batched dictionary with padded tensors.
        """
        torch = _ensure_torch()

        if not batch:
            return {}

        # Find max sequence length in batch
        if self.config.pad_to_max:
            max_len = self.config.max_seq_len
        else:
            max_len = 0
            for sample in batch:
                if "command_types" in sample:
                    cmd_types = sample["command_types"]
                    if isinstance(cmd_types, torch.Tensor):
                        max_len = max(max_len, cmd_types.numel())
                    elif isinstance(cmd_types, (list, tuple)):
                        max_len = max(max_len, len(cmd_types))
                elif "num_commands" in sample:
                    max_len = max(max_len, sample["num_commands"])

            max_len = min(max_len, self.config.max_seq_len)

        # Collate each field
        collated = {}

        # Command types
        cmd_types_list = []
        for sample in batch:
            cmd_types = self._get_tensor(sample, "command_types")
            if cmd_types is not None:
                cmd_types_list.append(
                    self._pad_sequence(cmd_types, max_len, self.config.pad_token_id)
                )

        if cmd_types_list:
            collated["command_types"] = torch.stack(cmd_types_list)

        # Parameters
        params_list = []
        for sample in batch:
            params = self._get_tensor(sample, "parameters")
            if params is not None:
                # Handle flat or 2D parameters
                if params.dim() == 1:
                    # Reshape to [seq_len, num_params]
                    num_params = self.config.num_params
                    seq_len = params.numel() // num_params
                    params = params[:seq_len * num_params].view(seq_len, num_params)

                # Pad sequence dimension
                params_padded = self._pad_2d_sequence(
                    params, max_len, self.config.num_params, 0
                )
                params_list.append(params_padded)

        if params_list:
            collated["parameters"] = torch.stack(params_list)

        # Attention mask
        mask_list = []
        for sample in batch:
            mask = self._get_tensor(sample, "mask")
            if mask is None:
                # Create mask from command types
                if "command_types" in sample:
                    cmd_types = self._get_tensor(sample, "command_types")
                    if cmd_types is not None:
                        mask = (cmd_types != self.config.pad_token_id).float()

            if mask is not None:
                mask_padded = self._pad_sequence(mask, max_len, 0.0)
                mask_list.append(mask_padded)

        if mask_list:
            collated["mask"] = torch.stack(mask_list)
            collated["attention_mask"] = (collated["mask"] > 0).long()

        # Create shifted labels for autoregressive training (teacher forcing)
        # Labels are the target tokens, shifted by 1 for next-token prediction
        if self.config.create_labels and "command_types" in collated:
            cmd_types = collated["command_types"]
            # For autoregressive: input_ids = tokens[:-1], labels = tokens[1:]
            collated["labels"] = cmd_types[:, 1:].clone()
            collated["input_ids"] = cmd_types[:, :-1].clone()
            # Also shift attention mask to match
            if "attention_mask" in collated:
                collated["attention_mask"] = collated["attention_mask"][:, :-1]
            # Shift parameters if present
            if "parameters" in collated:
                collated["input_parameters"] = collated["parameters"][:, :-1, :].clone()
                collated["target_parameters"] = collated["parameters"][:, 1:, :].clone()

        # Position IDs
        if self.config.include_position_ids:
            batch_size = len(batch)
            position_ids = torch.arange(max_len).unsqueeze(0).expand(batch_size, -1)
            collated["position_ids"] = position_ids

        # Num commands
        num_cmds_list = []
        for sample in batch:
            num_cmds = sample.get("num_commands")
            if num_cmds is not None:
                if isinstance(num_cmds, torch.Tensor):
                    num_cmds_list.append(num_cmds.item() if num_cmds.numel() == 1 else num_cmds[0].item())
                else:
                    num_cmds_list.append(int(num_cmds))

        if num_cmds_list:
            collated["num_commands"] = torch.tensor(num_cmds_list, dtype=torch.long)

        # Model IDs (as list, not tensor)
        model_ids = [sample.get("model_id") for sample in batch if "model_id" in sample]
        if model_ids:
            collated["model_ids"] = model_ids

        # Sample IDs
        sample_ids = [sample.get("sample_id") for sample in batch if "sample_id" in sample]
        if sample_ids:
            collated["sample_ids"] = sample_ids

        return collated

    def _get_tensor(
        self, sample: Dict[str, Any], key: str
    ) -> Optional["torch.Tensor"]:
        """Extract a tensor from a sample, converting if needed."""
        torch = _ensure_torch()

        value = sample.get(key)
        if value is None:
            return None

        if isinstance(value, torch.Tensor):
            return value

        if isinstance(value, (list, tuple)):
            try:
                return torch.tensor(value)
            except (ValueError, TypeError):
                return None

        if hasattr(value, "numpy"):
            return torch.from_numpy(value.numpy())

        return None

    def _pad_sequence(
        self,
        tensor: "torch.Tensor",
        target_len: int,
        pad_value: Union[int, float],
    ) -> "torch.Tensor":
        """Pad or truncate a 1D tensor to target length."""
        torch = _ensure_torch()

        tensor = tensor.flatten()
        current_len = tensor.numel()

        if current_len >= target_len:
            return tensor[:target_len]

        # Pad
        padding = torch.full(
            (target_len - current_len,),
            pad_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat([tensor, padding])

    def _pad_2d_sequence(
        self,
        tensor: "torch.Tensor",
        target_len: int,
        target_dim: int,
        pad_value: Union[int, float],
    ) -> "torch.Tensor":
        """Pad a 2D tensor [seq_len, dim] to target dimensions."""
        torch = _ensure_torch()

        if tensor.dim() != 2:
            tensor = tensor.view(-1, target_dim)

        seq_len, dim = tensor.shape

        # Truncate/pad dimension
        if dim > target_dim:
            tensor = tensor[:, :target_dim]
        elif dim < target_dim:
            padding = torch.full(
                (seq_len, target_dim - dim),
                pad_value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            tensor = torch.cat([tensor, padding], dim=1)

        # Truncate/pad sequence
        if seq_len >= target_len:
            return tensor[:target_len, :]

        padding = torch.full(
            (target_len - seq_len, target_dim),
            pad_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat([tensor, padding], dim=0)


class CADMultiModalCollator(CADCommandCollator):
    """Collator for multi-modal CAD data (text + geometry).

    Extends CADCommandCollator with text tokenization and
    batching for text-conditioned generation models.

    Usage:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> collator = CADMultiModalCollator(tokenizer=tokenizer)
        >>> batch = collator([sample1, sample2])
        >>> print(batch["text_input_ids"].shape)
        >>> print(batch["command_types"].shape)

    Args:
        tokenizer: HuggingFace tokenizer for text processing.
        config: Collator configuration.
        max_text_len: Maximum text sequence length.
    """

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        config: Optional[CADCollatorConfig] = None,
        max_text_len: int = 128,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

    def __call__(
        self, batch: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Collate a batch of multi-modal samples.

        Args:
            batch: List of sample dictionaries with 'text' field.

        Returns:
            Batched dictionary with both text and CAD tensors.
        """
        torch = _ensure_torch()

        # First, collate CAD data
        collated = super().__call__(batch)

        # Extract and tokenize text
        texts = []
        for sample in batch:
            text = sample.get("text") or sample.get("text_description") or ""
            texts.append(text)

        if self.tokenizer is not None and any(texts):
            # Tokenize all texts
            text_encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt",
            )

            collated["text_input_ids"] = text_encoded["input_ids"]
            collated["text_attention_mask"] = text_encoded["attention_mask"]

            if "token_type_ids" in text_encoded:
                collated["text_token_type_ids"] = text_encoded["token_type_ids"]
        else:
            # Store raw texts
            collated["texts"] = texts

        # Handle pre-tokenized text
        text_token_ids = []
        text_attention_masks = []

        for sample in batch:
            if "text_token_ids" in sample:
                ids = self._get_tensor(sample, "text_token_ids")
                if ids is not None:
                    text_token_ids.append(ids)

            if "text_attention_mask" in sample:
                mask = self._get_tensor(sample, "text_attention_mask")
                if mask is not None:
                    text_attention_masks.append(mask)

        if text_token_ids and "text_input_ids" not in collated:
            # Pad pre-tokenized text
            max_text_len = max(t.numel() for t in text_token_ids)
            max_text_len = min(max_text_len, self.max_text_len)

            padded_ids = []
            padded_masks = []

            for i, ids in enumerate(text_token_ids):
                padded_ids.append(self._pad_sequence(ids, max_text_len, 0))
                if i < len(text_attention_masks):
                    padded_masks.append(
                        self._pad_sequence(text_attention_masks[i], max_text_len, 0)
                    )
                else:
                    padded_masks.append(
                        self._pad_sequence((ids != 0).long(), max_text_len, 0)
                    )

            collated["text_input_ids"] = torch.stack(padded_ids)
            collated["text_attention_mask"] = torch.stack(padded_masks)

        return collated


class CADGraphCollator:
    """Collator for B-Rep graph batching with PyTorch Geometric.

    Batches graph data using PyG's Batch.from_data_list() for
    efficient message passing in GNN training.

    Usage:
        >>> collator = CADGraphCollator()
        >>> batch = collator([graph1, graph2, graph3])
        >>> print(batch.x.shape)  # [total_nodes, feat_dim]
        >>> print(batch.edge_index.shape)  # [2, total_edges]
        >>> print(batch.batch.shape)  # [total_nodes]

    Args:
        follow_batch: List of keys to track batch membership.
        exclude_keys: List of keys to exclude from batching.
    """

    def __init__(
        self,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ) -> None:
        self.follow_batch = follow_batch or []
        self.exclude_keys = exclude_keys or []

    def __call__(
        self, batch: List[Any]
    ) -> Any:
        """Collate a batch of graph samples.

        Args:
            batch: List of PyG Data objects or dicts.

        Returns:
            PyG Batch object if torch_geometric available,
            otherwise a dict with stacked tensors.
        """
        torch = _ensure_torch()
        pyg = _ensure_pyg()

        if not batch:
            return {}

        # Check if we have PyG Data objects
        if pyg and hasattr(pyg, "data"):
            from torch_geometric.data import Data, Batch

            # Convert dicts to Data objects if needed
            data_list = []
            for item in batch:
                if isinstance(item, Data):
                    data_list.append(item)
                elif isinstance(item, dict):
                    data = self._dict_to_data(item)
                    if data is not None:
                        data_list.append(data)

            if data_list:
                return Batch.from_data_list(
                    data_list,
                    follow_batch=self.follow_batch,
                    exclude_keys=self.exclude_keys,
                )

        # Fallback: manual batching
        return self._manual_batch(batch)

    def _dict_to_data(
        self, sample: Dict[str, Any]
    ) -> Optional[Any]:
        """Convert a dictionary to a PyG Data object."""
        pyg = _ensure_pyg()
        torch = _ensure_torch()

        if not pyg:
            return None

        from torch_geometric.data import Data

        # Extract graph components
        x = self._get_tensor(sample, "face_features")
        edge_index = self._get_tensor(sample, "edge_index")
        edge_attr = self._get_tensor(sample, "edge_features")
        y = self._get_tensor(sample, "face_labels")

        num_faces = sample.get("num_faces", 0)
        face_feat_dim = sample.get("face_feature_dim", 10)
        edge_feat_dim = sample.get("edge_feature_dim", 4)

        # Reshape if needed
        if x is not None and x.dim() == 1 and num_faces > 0:
            x = x[:num_faces * face_feat_dim].view(num_faces, face_feat_dim)

        if edge_index is not None and edge_index.dim() == 1:
            num_edges = edge_index.numel() // 2
            edge_index = edge_index[:num_edges * 2].view(2, num_edges)

        if edge_attr is not None and edge_attr.dim() == 1:
            num_edges = sample.get("num_edges", edge_attr.numel() // edge_feat_dim)
            if num_edges > 0:
                edge_attr = edge_attr[:num_edges * edge_feat_dim].view(
                    num_edges, edge_feat_dim
                )

        return Data(
            x=x.float() if x is not None else None,
            edge_index=edge_index.long() if edge_index is not None else None,
            edge_attr=edge_attr.float() if edge_attr is not None else None,
            y=y if y is not None else None,
            num_nodes=num_faces,
        )

    def _get_tensor(
        self, sample: Dict[str, Any], key: str
    ) -> Optional["torch.Tensor"]:
        """Extract a tensor from a sample."""
        torch = _ensure_torch()

        value = sample.get(key)
        if value is None:
            return None

        if isinstance(value, torch.Tensor):
            return value

        if isinstance(value, (list, tuple)):
            try:
                return torch.tensor(value)
            except (ValueError, TypeError):
                return None

        return None

    def _manual_batch(
        self, batch: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Manual batching fallback without PyG."""
        torch = _ensure_torch()

        collated = {}

        # Collect all node features
        all_x = []
        all_edge_index = []
        all_edge_attr = []
        batch_indices = []
        node_offset = 0

        for batch_idx, sample in enumerate(batch):
            if isinstance(sample, dict):
                x = self._get_tensor(sample, "face_features")
                edge_index = self._get_tensor(sample, "edge_index")
                edge_attr = self._get_tensor(sample, "edge_features")
                num_faces = sample.get("num_faces", 0)

                if x is not None:
                    all_x.append(x)
                    batch_indices.extend([batch_idx] * (x.numel() // 10))

                if edge_index is not None:
                    # Offset edge indices
                    edge_index = edge_index + node_offset
                    all_edge_index.append(edge_index)

                if edge_attr is not None:
                    all_edge_attr.append(edge_attr)

                node_offset += num_faces

        if all_x:
            collated["x"] = torch.cat(all_x)
        if all_edge_index:
            collated["edge_index"] = torch.cat(all_edge_index, dim=-1)
        if all_edge_attr:
            collated["edge_attr"] = torch.cat(all_edge_attr)
        if batch_indices:
            collated["batch"] = torch.tensor(batch_indices, dtype=torch.long)

        return collated


__all__ = [
    "CADCollatorConfig",
    "CADCommandCollator",
    "CADMultiModalCollator",
    "CADGraphCollator",
]
