"""
Configuration for LatticeLabs OCADR model.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class LLOCADRConfig:
    """Configuration for LL-OCADR model."""

    # Model architecture
    model_type: str = "ll_ocadr"
    model_name: str = "latticelabs/ll-ocadr-7b"

    # Language model
    language_model_name: str = "Qwen/Qwen2-7B"  # Base LLM
    n_embed: int = 1280  # LLM embedding dimension

    # 3D encoders
    geometry_embed_dim: int = 256  # GeometryNet output dimension
    shape_embed_dim: int = 768  # ShapeNet output dimension
    shape_depth: int = 12  # Number of transformer layers in ShapeNet
    shape_num_heads: int = 12  # Number of attention heads in ShapeNet

    # Projector
    projector_type: str = "linear"  # "linear" or "mlp_gelu"
    input_dim: int = 1024  # 256 (GeometryNet) + 768 (ShapeNet)
    projector_depth: int = 1  # For mlp_gelu type
    projector_mlp_ratio: int = 1  # For mlp_gelu type

    # Mesh processing
    min_chunk_size: int = 1000  # Minimum faces per chunk
    max_chunks: int = 27  # Maximum number of chunks (3x3x3)
    target_global_faces: int = 4096  # Target faces for global view
    mesh_token: str = "<mesh>"  # Placeholder token for meshes
    mesh_token_id: Optional[int] = None  # Will be set from tokenizer

    # Training
    freeze_vision_model: bool = False
    freeze_language_model: bool = False

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value like a dict."""
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "language_model_name": self.language_model_name,
            "n_embed": self.n_embed,
            "geometry_embed_dim": self.geometry_embed_dim,
            "shape_embed_dim": self.shape_embed_dim,
            "shape_depth": self.shape_depth,
            "shape_num_heads": self.shape_num_heads,
            "projector_type": self.projector_type,
            "input_dim": self.input_dim,
            "projector_depth": self.projector_depth,
            "min_chunk_size": self.min_chunk_size,
            "max_chunks": self.max_chunks,
            "target_global_faces": self.target_global_faces,
            "mesh_token": self.mesh_token,
            "freeze_vision_model": self.freeze_vision_model,
            "freeze_language_model": self.freeze_language_model,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LLOCADRConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


def get_default_config() -> LLOCADRConfig:
    """Get default LL-OCADR configuration."""
    return LLOCADRConfig()


def get_config_for_model(model_size: str = "7b") -> LLOCADRConfig:
    """
    Get configuration for specific model size.

    Args:
        model_size: "1.8b", "7b", or "14b"

    Returns:
        LLOCADRConfig
    """
    base_config = LLOCADRConfig()

    if model_size == "1.8b":
        base_config.language_model_name = "Qwen/Qwen2-1.8B"
        base_config.n_embed = 1024
        base_config.shape_depth = 6
        base_config.shape_num_heads = 8
    elif model_size == "7b":
        base_config.language_model_name = "Qwen/Qwen2-7B"
        base_config.n_embed = 1280
        base_config.shape_depth = 12
        base_config.shape_num_heads = 12
    elif model_size == "14b":
        base_config.language_model_name = "Qwen/Qwen2-14B"
        base_config.n_embed = 1536
        base_config.shape_depth = 16
        base_config.shape_num_heads = 16
    else:
        raise ValueError(f"Unknown model size: {model_size}")

    base_config.input_dim = base_config.geometry_embed_dim + base_config.shape_embed_dim

    return base_config
