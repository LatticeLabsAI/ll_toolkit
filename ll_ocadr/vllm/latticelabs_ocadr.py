"""
LatticeLabs OCADR - Main model implementation.
Integrates GeometryNet, ShapeNet, and LLM for 3D CAD/Mesh understanding.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .lattice_encoder.build_linear import MlpProjector
from .lattice_encoder.geometry_net import build_geometry_net
from .lattice_encoder.shape_net import build_shape_net


class LatticelabsOCADRForCausalLM(nn.Module):
    """
    Main model class integrating 3D geometry processing with LLM.
    Mirrors DeepseekOCRForCausalLM structure but for 3D meshes.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Initialize 3D encoders
        self.geometry_model = build_geometry_net()  # Local geometry features
        self.shape_model = build_shape_net(
            embed_dim=config.shape_embed_dim,
            depth=config.shape_depth,
            num_heads=config.shape_num_heads
        )  # Global shape features

        # MLP Projector: concatenated features -> LLM embedding space
        self.projector = MlpProjector(config)

        # Language model
        self.language_model = AutoModelForCausalLM.from_pretrained(
            config.language_model_name
        )

        # Special tokens (learnable parameters)
        self.mesh_boundary = nn.Parameter(
            torch.randn(1, config.n_embed)
        )  # Separates chunk layers
        self.view_separator = nn.Parameter(
            torch.randn(1, config.n_embed)
        )  # Separates local from global

        # Token IDs
        self.mesh_token_id = config.mesh_token_id

    def _mesh_to_embedding(
        self,
        vertex_coords: torch.Tensor,
        vertex_normals: torch.Tensor,
        chunks_coords: Optional[torch.Tensor] = None,
        chunks_normals: Optional[torch.Tensor] = None,
        mesh_spatial_partition: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Core 3D encoding pipeline. Mirrors _pixel_values_to_embedding from DeepSeek-OCR.

        Args:
            vertex_coords: [B, N, 3] global mesh vertices
            vertex_normals: [B, N, 3] global normals
            chunks_coords: [B, num_chunks, M, 3] local chunk vertices
            chunks_normals: [B, num_chunks, M, 3] local normals
            mesh_spatial_partition: [B, 3] - (nx, ny, nz) grid

        Returns:
            List of [num_tokens, n_embed] embeddings per mesh
        """
        # Only disable gradients during inference; during training,
        # encoder gradients must flow for fine-tuning convergence.
        context = torch.no_grad() if not self.training else torch.enable_grad()
        with context:
            batch_size = vertex_coords.shape[0]
            device = vertex_coords.device
            embeddings_list = []

            for batch_idx in range(batch_size):
                # Get data for this mesh
                v_coords = vertex_coords[batch_idx]  # [N, 3]
                v_normals = vertex_normals[batch_idx]  # [N, 3]

                # ===== GLOBAL FEATURES (from full mesh) =====
                # Add batch dimension
                v_coords_batch = v_coords.unsqueeze(0)  # [1, N, 3]
                v_normals_batch = v_normals.unsqueeze(0)  # [1, N, 3]

                # Encode with GeometryNet
                global_feat_geom = self.geometry_model(
                    v_coords_batch, v_normals_batch
                )  # [1, 128, 256]

                # Encode with ShapeNet
                global_feat_shape = self.shape_model(
                    v_coords_batch, v_normals_batch
                )  # [1, 257, 768]

                # Skip CLS token, concatenate features
                global_feat_shape_no_cls = global_feat_shape[:, 1:]  # [1, 256, 768]

                # Pad/align geometry features to match shape features
                geom_tokens = global_feat_geom.shape[1]  # 128
                shape_tokens = global_feat_shape_no_cls.shape[1]  # 256

                if geom_tokens < shape_tokens:
                    # Pad geometry features
                    padding = torch.zeros(
                        1, shape_tokens - geom_tokens, 256,
                        device=device
                    )
                    global_feat_geom_padded = torch.cat([global_feat_geom, padding], dim=1)
                else:
                    global_feat_geom_padded = global_feat_geom[:, :shape_tokens]

                # Concatenate: [1, 256, 768] + [1, 256, 256] = [1, 256, 1024]
                global_features = torch.cat([
                    global_feat_shape_no_cls,
                    global_feat_geom_padded
                ], dim=-1)  # [1, 256, 1024]

                # Project to LLM space
                global_features = self.projector(global_features)  # [1, 256, n_embed]
                global_features = global_features.squeeze(0)  # [256, n_embed]

                # ===== LOCAL FEATURES (from chunks) =====
                if chunks_coords is not None and chunks_normals is not None and mesh_spatial_partition is not None and torch.sum(chunks_coords[batch_idx]) != 0:
                    chunks_c = chunks_coords[batch_idx]  # [num_chunks, M, 3]
                    chunks_n = chunks_normals[batch_idx]  # [num_chunks, M, 3]
                    num_chunks = chunks_c.shape[0]

                    # Process all chunks in parallel (batched encoder calls)
                    chunk_feat_geom = self.geometry_model(chunks_c, chunks_n)  # [num_chunks, 128, 256]
                    chunk_feat_shape = self.shape_model(chunks_c, chunks_n)  # [num_chunks, 257, 768]

                    # Skip CLS token
                    chunk_feat_shape_no_cls = chunk_feat_shape[:, 1:]  # [num_chunks, 256, 768]

                    # Align dimensions
                    geom_tokens = chunk_feat_geom.shape[1]
                    shape_tokens = chunk_feat_shape_no_cls.shape[1]

                    if geom_tokens < shape_tokens:
                        padding = torch.zeros(num_chunks, shape_tokens - geom_tokens, 256, device=device)
                        chunk_feat_geom = torch.cat([chunk_feat_geom, padding], dim=1)
                    else:
                        chunk_feat_geom = chunk_feat_geom[:, :shape_tokens]

                    # Concatenate and project
                    chunk_features = torch.cat([
                        chunk_feat_shape_no_cls,
                        chunk_feat_geom
                    ], dim=-1)  # [num_chunks, 256, 1024]

                    local_features = self.projector(chunk_features)  # [num_chunks, 256, n_embed]

                    # Flatten chunks and add boundaries between layers
                    nx, ny, nz = mesh_spatial_partition[batch_idx].tolist()
                    formatted_local = self._format_chunk_grid(local_features, nx, ny, nz)

                    # Concatenate: [local, global, separator]
                    combined = torch.cat([
                        formatted_local,
                        global_features,
                        self.view_separator.squeeze(0).unsqueeze(0)
                    ], dim=0)
                else:
                    # No chunking, just global + separator
                    combined = torch.cat([
                        global_features,
                        self.view_separator.squeeze(0).unsqueeze(0)
                    ], dim=0)

                embeddings_list.append(combined)

            return embeddings_list

    def _format_chunk_grid(self, local_features: torch.Tensor,
                          nx: int, ny: int, nz: int) -> torch.Tensor:
        """
        Format chunk features into spatial grid with boundary tokens.

        Args:
            local_features: [num_chunks, num_tokens, n_embed]
            nx, ny, nz: grid dimensions

        Returns:
            formatted: [total_tokens, n_embed] with boundaries inserted
        """
        num_chunks, num_tokens, n_embed = local_features.shape

        # Flatten all chunk tokens
        flattened = local_features.view(-1, n_embed)  # [num_chunks*num_tokens, n_embed]

        # Add boundary tokens between z-layers
        formatted_parts = []
        tokens_per_layer = nx * ny * num_tokens

        for z in range(nz):
            start_idx = z * tokens_per_layer
            end_idx = (z + 1) * tokens_per_layer

            if end_idx <= flattened.shape[0]:
                layer_tokens = flattened[start_idx:end_idx]
                formatted_parts.append(layer_tokens)

                # Add boundary token between layers (except last)
                if z < nz - 1:
                    formatted_parts.append(self.mesh_boundary.squeeze(0).unsqueeze(0))

        return torch.cat(formatted_parts, dim=0) if formatted_parts else flattened

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Merge mesh embeddings with text embeddings.
        Identical logic to DeepSeek-OCR's implementation.

        Args:
            input_ids: [batch, seq_len] with mesh_token_id placeholders
            multimodal_embeddings: List of [num_mesh_tokens, n_embed] tensors

        Returns:
            inputs_embeds: [batch, seq_len, n_embed] merged embeddings
        """
        # Get text embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        # [batch, seq_len, n_embed]

        if multimodal_embeddings is not None:
            # Replace mesh_token_id positions with actual mesh embeddings
            batch_size, seq_len = input_ids.shape

            for batch_idx in range(batch_size):
                # Find positions of mesh tokens
                mesh_positions = (input_ids[batch_idx] == self.mesh_token_id).nonzero(as_tuple=False).squeeze(-1)

                if len(mesh_positions) > 0 and batch_idx < len(multimodal_embeddings):
                    mesh_emb = multimodal_embeddings[batch_idx]
                    num_mesh_tokens = mesh_emb.shape[0]

                    # Replace tokens
                    if len(mesh_positions) >= num_mesh_tokens:
                        inputs_embeds[batch_idx, mesh_positions[:num_mesh_tokens]] = mesh_emb
                    else:
                        inputs_embeds[batch_idx, mesh_positions] = mesh_emb[:len(mesh_positions)]

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        vertex_coords: Optional[torch.Tensor] = None,
        vertex_normals: Optional[torch.Tensor] = None,
        chunks_coords: Optional[torch.Tensor] = None,
        chunks_normals: Optional[torch.Tensor] = None,
        mesh_spatial_partition: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Full inference pipeline integrating 3D geometry + language.

        Args:
            input_ids: [batch, seq_len] with mesh_token_id placeholders
            attention_mask: [batch, seq_len]
            vertex_coords: [batch, N, 3]
            vertex_normals: [batch, N, 3]
            chunks_coords: [batch, num_chunks, M, 3]
            chunks_normals: [batch, num_chunks, M, 3]
            mesh_spatial_partition: [batch, 3]

        Returns:
            Language model outputs
        """
        # Process mesh data if provided
        if vertex_coords is not None:
            mesh_embeddings = self._mesh_to_embedding(
                vertex_coords=vertex_coords,
                vertex_normals=vertex_normals,
                chunks_coords=chunks_coords,
                chunks_normals=chunks_normals,
                mesh_spatial_partition=mesh_spatial_partition
            )
        else:
            mesh_embeddings = None

        # Merge mesh embeddings with text
        inputs_embeds = self.get_input_embeddings(
            input_ids=input_ids,
            multimodal_embeddings=mesh_embeddings
        )

        # Pass to language model
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        vertex_coords: Optional[torch.Tensor] = None,
        vertex_normals: Optional[torch.Tensor] = None,
        chunks_coords: Optional[torch.Tensor] = None,
        chunks_normals: Optional[torch.Tensor] = None,
        mesh_spatial_partition: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Autoregressive generation with 3D mesh conditioning.

        Processes mesh inputs through the 3D encoders, merges the resulting
        embeddings with text token embeddings, then delegates to the inner
        language model's generate() (which inherits from GenerationMixin).

        Accepts all keyword arguments supported by
        ``transformers.GenerationMixin.generate`` (e.g. max_new_tokens,
        temperature, top_p, do_sample).

        Returns:
            Generated token IDs from the language model.
        """
        # Process mesh data if provided
        if vertex_coords is not None:
            mesh_embeddings = self._mesh_to_embedding(
                vertex_coords=vertex_coords,
                vertex_normals=vertex_normals,
                chunks_coords=chunks_coords,
                chunks_normals=chunks_normals,
                mesh_spatial_partition=mesh_spatial_partition,
            )
        else:
            mesh_embeddings = None

        # Merge mesh embeddings with text token embeddings
        inputs_embeds = self.get_input_embeddings(
            input_ids=input_ids,
            multimodal_embeddings=mesh_embeddings,
        )

        # Delegate to the language model's generate(), passing inputs_embeds
        # instead of input_ids so the mesh tokens are already resolved.
        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )


def build_ll_ocadr_model(config):
    """Build LatticeLabs OCADR model."""
    return LatticelabsOCADRForCausalLM(config)


# =============================================================================
# vLLM Integration Classes
# =============================================================================

class LLOCADRProcessingInfo:
    """
    Metadata about mesh processing for vLLM.
    Provides token count calculations for KV cache allocation.
    """

    def __init__(self, config):
        self.config = config
        from .process.mesh_process import MeshLoader
        self.loader = MeshLoader()

    def get_num_mesh_tokens(self, mesh_file: str, chunking: bool = True) -> int:
        """
        Calculate token count based on mesh complexity.
        Critical for vLLM's KV cache allocation.

        Args:
            mesh_file: Path to mesh file
            chunking: Whether to use spatial chunking

        Returns:
            Total number of tokens this mesh will generate
        """
        # Quick mesh analysis
        mesh = self.loader.load(mesh_file)
        num_faces = mesh.num_faces

        # Determine chunking grid based on complexity
        if not chunking or num_faces <= self.config.min_chunk_size:
            subdivision = (1, 1, 1)
        elif num_faces <= 8000:
            subdivision = (2, 2, 2)  # 8 chunks
        else:
            subdivision = (3, 3, 3)  # 27 chunks

        # Calculate tokens
        # Global view: ~256 tokens (from shape encoder)
        global_tokens = 256

        num_chunks = subdivision[0] * subdivision[1] * subdivision[2]
        if num_chunks > 1:
            # Local tokens: 256 tokens per chunk (from encoders)
            local_tokens = num_chunks * 256
            # Boundary tokens: one per z-layer
            boundary_tokens = subdivision[2]
            # Separator token
            total_tokens = local_tokens + global_tokens + boundary_tokens + 1
        else:
            # Just global + separator
            total_tokens = global_tokens + 1

        return total_tokens


class LLOCADRMultiModalProcessor:
    """
    vLLM integration layer for LL-OCADR.
    Mirrors DeepseekOCRMultiModalProcessor structure.
    """

    def __init__(self, config):
        self.config = config
        self.info = LLOCADRProcessingInfo(config)

        # Cache tokenizer once — AutoTokenizer.from_pretrained is expensive
        self._tokenizer = AutoTokenizer.from_pretrained(config.language_model_name)

        # Add mesh token to tokenizer if not present
        if config.mesh_token not in self._tokenizer.get_vocab():
            self._tokenizer.add_tokens([config.mesh_token])
            config.mesh_token_id = self._tokenizer.vocab[config.mesh_token]

        # Cache processor once — avoid re-instantiation on every call
        from .process.mesh_process import LLOCADRProcessor

        self._processor = LLOCADRProcessor(
            tokenizer=self._tokenizer,
            mesh_token_id=self.config.mesh_token_id
        )

    def _call_hf_processor(self, prompt: str, mm_data: Dict, mm_kwargs: Dict):
        """
        Call LLOCADRProcessor to preprocess meshes.

        Args:
            prompt: Text prompt with <mesh> placeholders
            mm_data: {"mesh": [mesh_files]}
            mm_kwargs: Additional processing kwargs

        Returns:
            Processed tensors
        """

        mesh_files = mm_data.get("mesh", [])
        return self._processor.tokenize_with_meshes(
            mesh_files=mesh_files,
            conversation=prompt,
            cropping=mm_kwargs.get("cropping", True)
        )

    def _get_mm_fields_config(self) -> Dict:
        """
        Declare which tensor fields are multimodal.
        Used by vLLM for batch handling.

        Returns:
            Dictionary mapping field names to multimodal configs
        """
        # This would use vLLM's MultiModalFieldConfig if available
        # For now, return field names
        return {
            "vertex_coords": "batched_mesh",
            "vertex_normals": "batched_mesh",
            "chunks_coords": "batched_mesh",
            "chunks_normals": "batched_mesh",
            "mesh_spatial_partition": "batched_mesh",
        }

    def _get_prompt_updates(self, mm_items: List, hf_processor_mm_kwargs: Dict) -> List:
        """
        Calculate dynamic token count and create prompt replacements.

        Args:
            mm_items: List of mesh files
            hf_processor_mm_kwargs: Processing kwargs

        Returns:
            List of prompt replacement configs
        """
        def get_replacement_ll_ocadr(item_idx: int) -> List[int]:
            """Get token IDs to replace <mesh> placeholder."""
            mesh_file = mm_items[item_idx]
            num_tokens = self.info.get_num_mesh_tokens(
                mesh_file=mesh_file,
                chunking=hf_processor_mm_kwargs.get("cropping", True)
            )
            # Return that many mesh_token_ids
            return [self.config.mesh_token_id] * num_tokens

        # This would use vLLM's PromptReplacement if available
        # For now, return the function
        return [{
            "modality": "mesh",
            "target": [self.config.mesh_token_id],
            "replacement": get_replacement_ll_ocadr
        }]
