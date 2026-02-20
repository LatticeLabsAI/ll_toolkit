"""Hybrid Transformer + GNN encoder for shape understanding.

Fuses conditioning embeddings (text/image) processed through a transformer
branch with B-Rep topology features processed through a GNN branch.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)


class HybridShapeEncoder:
    """Fused Transformer + GNN encoder for shape understanding.

    Combines conditioning embeddings (text/image) processed through
    a transformer branch with B-Rep topology features processed through
    a GNN branch (lazy-imported from cadling's BRepNetEncoder or UVNetEncoder).

    The two branches produce fixed-dimension embeddings that are fused
    via concatenation + linear projection.

    Args:
        input_dim: Dimension of input conditioning embeddings. Defaults to 768.
        transformer_dim: Hidden dimension for transformer layers. Defaults to 512.
        graph_dim: Hidden dimension for GNN branch. Defaults to 256.
        output_dim: Dimension of final fused embedding. Defaults to 512.
        num_transformer_layers: Number of transformer encoder layers. Defaults to 3.
        num_heads: Number of attention heads. Defaults to 8.
        dropout: Dropout probability. Defaults to 0.1.
        graph_encoder_type: Type of graph encoder ("brep_net" or "uv_net").
            Defaults to "brep_net".
        device: Target device ("cpu" or "cuda"). Defaults to "cpu".

    Example::

        encoder = HybridShapeEncoder(
            input_dim=768,
            output_dim=512,
            device="cuda"
        )

        # Encode conditioning only
        cond_embedding = encoder.encode_conditioning_only(cond_array)

        # Encode with graph
        output = encoder.forward(cond_array, graph_data=graph_dict)
    """

    def __init__(
        self,
        input_dim: int = 768,
        transformer_dim: int = 512,
        graph_dim: int = 256,
        output_dim: int = 512,
        num_transformer_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        graph_encoder_type: str = "brep_net",
        device: str = "cpu",
    ) -> None:
        """Initialize the hybrid encoder.

        Args:
            input_dim: Dimension of input conditioning embeddings.
            transformer_dim: Hidden dimension for transformer.
            graph_dim: Hidden dimension for GNN.
            output_dim: Output embedding dimension.
            num_transformer_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            graph_encoder_type: Type of graph encoder.
            device: Target device.
        """
        self.input_dim = input_dim
        self.transformer_dim = transformer_dim
        self.graph_dim = graph_dim
        self.output_dim = output_dim
        self.num_transformer_layers = num_transformer_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.graph_encoder_type = graph_encoder_type
        self.device = device

        # Lazy-initialized torch components
        self._torch = None
        self._nn = None
        self._input_projection = None
        self._transformer_encoder = None
        self._transformer_output_projection = None
        self._fusion_projection = None
        self._gnn_encoder = None
        self._has_gnn = False

        self._initialize_torch_components()

    def _initialize_torch_components(self) -> None:
        """Initialize torch components (lazy import).

        Sets up transformer and GNN branches. GNN branch is optional and
        only initialized if cadling is available.
        """
        try:
            import torch
            import torch.nn as nn

            self._torch = torch
            self._nn = nn
        except ImportError as exc:
            raise ImportError(
                "torch is required for HybridShapeEncoder; install with "
                "'conda install pytorch::pytorch -c conda-forge'"
            ) from exc

        # Input projection
        self._input_projection = self._nn.Linear(self.input_dim, self.transformer_dim)

        # Transformer encoder
        encoder_layer = self._nn.TransformerEncoderLayer(
            d_model=self.transformer_dim,
            nhead=self.num_heads,
            dim_feedforward=self.transformer_dim * 4,
            dropout=self.dropout,
            batch_first=True,
        )
        self._transformer_encoder = self._nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_transformer_layers
        )

        # Determine output projection size based on graph availability
        has_graph = self._try_initialize_gnn()
        transformer_out_dim = self.output_dim // 2 if has_graph else self.output_dim
        self._transformer_output_projection = self._nn.Linear(
            self.transformer_dim, transformer_out_dim
        )

        # Fusion projection (only if graph is available)
        if has_graph:
            self._fusion_projection = self._nn.Linear(
                transformer_out_dim + self.graph_dim, self.output_dim
            )

        # Move to device
        self.to(self.device)

    def _try_initialize_gnn(self) -> bool:
        """Try to initialize GNN encoder from cadling.

        Returns:
            True if GNN encoder was successfully initialized, False otherwise.
        """
        try:
            from cadling.models.geometry.brep_net import BRepNetEncoder
        except ImportError:
            try:
                from cadling.models.geometry.uv_net import UVNetEncoder
            except ImportError:
                _log.debug(
                    "cadling geometry encoders not available; GNN branch disabled"
                )
                self._has_gnn = False
                return False

        # Initialize the selected GNN encoder
        try:
            if self.graph_encoder_type == "brep_net":
                from cadling.models.geometry.brep_net import BRepNetEncoder

                self._gnn_encoder = BRepNetEncoder(
                    output_dim=self.graph_dim, dropout=self.dropout
                )
            elif self.graph_encoder_type == "uv_net":
                from cadling.models.geometry.uv_net import UVNetEncoder

                self._gnn_encoder = UVNetEncoder(
                    output_dim=self.graph_dim, dropout=self.dropout
                )
            else:
                _log.warning(
                    f"Unknown graph encoder type: {self.graph_encoder_type}; "
                    "GNN branch disabled"
                )
                self._has_gnn = False
                return False

            self._has_gnn = True
            _log.debug(f"GNN encoder initialized: {self.graph_encoder_type}")
            return True
        except Exception as exc:
            _log.debug(
                f"Failed to initialize GNN encoder: {exc}; GNN branch disabled"
            )
            self._has_gnn = False
            return False

    def forward(
        self,
        conditioning_embeddings: Any,
        graph_data: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Encode conditioning + optional graph to shape embedding.

        Args:
            conditioning_embeddings: Input embeddings, either numpy array
                (seq_len, input_dim) or torch tensor. If 1D, will be expanded
                to (1, input_dim).
            graph_data: Optional dict with keys:
                - node_features: (num_nodes, feat_dim) tensor or array
                - edge_index: (2, num_edges) edge indices
                - edge_attr: Optional (num_edges, edge_feat_dim) edge attributes

        Returns:
            numpy array of shape (output_dim,) containing the fused embedding.
        """
        # Convert input to torch tensor
        if isinstance(conditioning_embeddings, np.ndarray):
            cond_tensor = self._torch.from_numpy(conditioning_embeddings).float()
        else:
            cond_tensor = conditioning_embeddings.float()

        # Ensure 3D for transformer (batch, seq_len, dim)
        if cond_tensor.ndim == 1:
            cond_tensor = cond_tensor.unsqueeze(0).unsqueeze(0)
        elif cond_tensor.ndim == 2:
            cond_tensor = cond_tensor.unsqueeze(0)

        cond_tensor = cond_tensor.to(self.device)

        # Transformer path
        projected = self._input_projection(cond_tensor)
        transformer_out = self._transformer_encoder(projected)
        # Mean pooling over sequence dimension
        transformer_pooled = transformer_out.mean(dim=1)  # (batch, transformer_dim)
        transformer_embedding = self._transformer_output_projection(
            transformer_pooled
        )  # (batch, transformer_out_dim)

        # GNN path (if available and graph_data provided)
        if self._has_gnn and graph_data is not None:
            gnn_embedding = self._encode_graph(graph_data)  # (graph_dim,)
            # Fuse: concatenate + project
            fused = self._torch.cat(
                [transformer_embedding[0], gnn_embedding.unsqueeze(0)], dim=1
            )  # (batch=1, transformer_out_dim + graph_dim)
            output = self._fusion_projection(fused)  # (batch=1, output_dim)
        else:
            output = transformer_embedding  # (batch=1, transformer_out_dim)

        # Return as numpy (batch dimension = 1)
        return output.detach().cpu().numpy()[0]

    def _encode_graph(self, graph_data: dict[str, Any]) -> Any:
        """Encode graph data using GNN encoder.

        Args:
            graph_data: Dictionary with node_features, edge_index, edge_attr.

        Returns:
            Graph embedding tensor of shape (graph_dim,).

        Raises:
            RuntimeError: If GNN encoder is not available.
        """
        if not self._has_gnn or self._gnn_encoder is None:
            raise RuntimeError(
                "GNN encoder is not available; cannot encode graph data"
            )

        # Extract graph components
        node_features = graph_data.get("node_features")
        edge_index = graph_data.get("edge_index")
        edge_attr = graph_data.get("edge_attr")

        # Convert to torch tensors if needed
        if isinstance(node_features, np.ndarray):
            node_features = self._torch.from_numpy(node_features).float()
        if isinstance(edge_index, np.ndarray):
            edge_index = self._torch.from_numpy(edge_index).long()
        if edge_attr is not None and isinstance(edge_attr, np.ndarray):
            edge_attr = self._torch.from_numpy(edge_attr).float()

        # Move to device
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(self.device)

        # Forward through GNN
        graph_out = self._gnn_encoder(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        # Pool to single embedding
        graph_embedding = graph_out.mean(dim=0)  # Average pooling
        return graph_embedding

    def encode_conditioning_only(self, conditioning_embeddings: Any) -> np.ndarray:
        """Encode conditioning embeddings through transformer branch only.

        Useful for text/image inputs without graph data.

        Args:
            conditioning_embeddings: Input embeddings (numpy array or torch tensor).

        Returns:
            numpy array of shape (output_dim,) or (transformer_out_dim,) if graph
            is available but not used.
        """
        # Convert input
        if isinstance(conditioning_embeddings, np.ndarray):
            cond_tensor = self._torch.from_numpy(conditioning_embeddings).float()
        else:
            cond_tensor = conditioning_embeddings.float()

        # Ensure 3D
        if cond_tensor.ndim == 1:
            cond_tensor = cond_tensor.unsqueeze(0).unsqueeze(0)
        elif cond_tensor.ndim == 2:
            cond_tensor = cond_tensor.unsqueeze(0)

        cond_tensor = cond_tensor.to(self.device)

        # Transformer only
        projected = self._input_projection(cond_tensor)
        transformer_out = self._transformer_encoder(projected)
        transformer_pooled = transformer_out.mean(dim=1)
        embedding = self._transformer_output_projection(transformer_pooled)

        return embedding.detach().cpu().numpy()[0]

    def encode_graph_only(self, graph_data: dict[str, Any]) -> np.ndarray:
        """Encode graph data through GNN branch only.

        Args:
            graph_data: Dictionary with node_features, edge_index, edge_attr.

        Returns:
            numpy array of shape (graph_dim,).

        Raises:
            RuntimeError: If GNN encoder is not available.
        """
        if not self._has_gnn or self._gnn_encoder is None:
            raise RuntimeError(
                "GNN encoder is not available; cannot encode graph data"
            )

        graph_embedding = self._encode_graph(graph_data)
        return graph_embedding.detach().cpu().numpy()

    def to(self, device: str) -> HybridShapeEncoder:
        """Move all parameters to device.

        Args:
            device: Target device ("cpu" or "cuda").

        Returns:
            Self for chaining.
        """
        self.device = device
        if self._input_projection is not None:
            self._input_projection = self._input_projection.to(device)
        if self._transformer_encoder is not None:
            self._transformer_encoder = self._transformer_encoder.to(device)
        if self._transformer_output_projection is not None:
            self._transformer_output_projection = (
                self._transformer_output_projection.to(device)
            )
        if self._fusion_projection is not None:
            self._fusion_projection = self._fusion_projection.to(device)
        if self._gnn_encoder is not None:
            self._gnn_encoder = self._gnn_encoder.to(device)
        return self

    def eval(self) -> None:
        """Set all submodules to evaluation mode."""
        if self._input_projection is not None:
            self._input_projection.eval()
        if self._transformer_encoder is not None:
            self._transformer_encoder.eval()
        if self._transformer_output_projection is not None:
            self._transformer_output_projection.eval()
        if self._fusion_projection is not None:
            self._fusion_projection.eval()
        if self._gnn_encoder is not None:
            self._gnn_encoder.eval()

    def train(self, mode: bool = True) -> None:
        """Set all submodules to training mode.

        Args:
            mode: Whether to enable training mode. Defaults to True.
        """
        if self._input_projection is not None:
            self._input_projection.train(mode)
        if self._transformer_encoder is not None:
            self._transformer_encoder.train(mode)
        if self._transformer_output_projection is not None:
            self._transformer_output_projection.train(mode)
        if self._fusion_projection is not None:
            self._fusion_projection.train(mode)
        if self._gnn_encoder is not None:
            self._gnn_encoder.train(mode)

    def parameters(self):
        """Yield all trainable parameters.

        Yields:
            torch.nn.Parameter instances.
        """
        if self._input_projection is not None:
            yield from self._input_projection.parameters()
        if self._transformer_encoder is not None:
            yield from self._transformer_encoder.parameters()
        if self._transformer_output_projection is not None:
            yield from self._transformer_output_projection.parameters()
        if self._fusion_projection is not None:
            yield from self._fusion_projection.parameters()
        if self._gnn_encoder is not None:
            yield from self._gnn_encoder.parameters()

    def state_dict(self) -> dict[str, Any]:
        """Return state dictionary for all components.

        Returns:
            Dictionary mapping component names to their state dicts.
        """
        state = {}
        if self._input_projection is not None:
            state["input_projection"] = self._input_projection.state_dict()
        if self._transformer_encoder is not None:
            state["transformer_encoder"] = self._transformer_encoder.state_dict()
        if self._transformer_output_projection is not None:
            state["transformer_output_projection"] = (
                self._transformer_output_projection.state_dict()
            )
        if self._fusion_projection is not None:
            state["fusion_projection"] = self._fusion_projection.state_dict()
        if self._gnn_encoder is not None:
            state["gnn_encoder"] = self._gnn_encoder.state_dict()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dictionary.

        Args:
            state_dict: Dictionary mapping component names to state dicts.
        """
        if "input_projection" in state_dict and self._input_projection is not None:
            self._input_projection.load_state_dict(state_dict["input_projection"])
        if "transformer_encoder" in state_dict and self._transformer_encoder is not None:
            self._transformer_encoder.load_state_dict(
                state_dict["transformer_encoder"]
            )
        if (
            "transformer_output_projection" in state_dict
            and self._transformer_output_projection is not None
        ):
            self._transformer_output_projection.load_state_dict(
                state_dict["transformer_output_projection"]
            )
        if "fusion_projection" in state_dict and self._fusion_projection is not None:
            self._fusion_projection.load_state_dict(state_dict["fusion_projection"])
        if "gnn_encoder" in state_dict and self._gnn_encoder is not None:
            self._gnn_encoder.load_state_dict(state_dict["gnn_encoder"])
