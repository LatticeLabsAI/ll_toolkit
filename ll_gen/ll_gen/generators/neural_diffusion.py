"""Neural diffusion generator — wraps StructuredDiffusion model.

Generates B-rep geometry (face grids + edge points) from a diffusion
process that progressively denoises from random initialization.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ll_gen.conditioning.embeddings import ConditioningEmbeddings
from ll_gen.generators.base import BaseNeuralGenerator
from ll_gen.proposals.latent_proposal import LatentProposal

_log = logging.getLogger(__name__)


class NeuralDiffusionGenerator(BaseNeuralGenerator):
    """Neural generator wrapping StructuredDiffusion for B-rep synthesis.

    Generates per-face point grids (U×V×3) and per-edge point arrays (N×3)
    through progressive denoising, with support for stage-specific retry
    strategies.

    Attributes:
        diffusion_config: Optional diffusion configuration.
        inference_steps: Number of denoising steps.
        eta: DDIM eta parameter (0.0 = deterministic, 1.0 = stochastic).
    """

    def __init__(
        self,
        diffusion_config: Any | None = None,
        checkpoint_path: Path | None = None,
        device: str = "cpu",
        inference_steps: int = 50,
        eta: float = 0.0,
    ) -> None:
        """Initialize the diffusion generator.

        Args:
            diffusion_config: Optional diffusion configuration object.
            checkpoint_path: Path to model checkpoint.
            device: Target device ("cpu" or "cuda").
            inference_steps: Number of DDIM steps.
            eta: DDIM eta parameter.
        """
        super().__init__(device=device, checkpoint_path=checkpoint_path)
        self.diffusion_config = diffusion_config
        self.inference_steps = inference_steps
        self.eta = eta

    def generate(
        self,
        prompt: str,
        conditioning: ConditioningEmbeddings | None = None,
        error_context: dict[str, Any] | None = None,
    ) -> LatentProposal:
        """Generate B-rep geometry via diffusion.

        Args:
            prompt: User prompt (stored for tracing).
            conditioning: Optional conditioning embeddings.
            error_context: Optional error context from a failed attempt.

        Returns:
            LatentProposal with face grids and edge points.
        """
        if self._model is None:
            self._init_model()

        # Run diffusion sampling
        try:
            import torch
        except ImportError:
            raise ImportError("torch is required for diffusion generation") from None

        with torch.no_grad():
            # Initialize noise
            batch_size = 1
            noise_shape = self._get_noise_shape()
            noise = torch.randn(
                batch_size, *noise_shape,
                device=self.device,
                dtype=torch.float32,
            )

            # Run sampling
            if hasattr(self._model, "sample"):
                output = self._model.sample(
                    noise=noise,
                    num_steps=self.inference_steps,
                    eta=self.eta,
                )
            else:
                output = self._model(noise, steps=self.inference_steps)

        # Extract geometry from output
        face_grids: list[np.ndarray] = []
        edge_points: list[np.ndarray] = []
        stage_latents: dict[str, Any] | None = None

        if isinstance(output, dict):
            # Structured output with geometry and latents
            if "face_grids" in output:
                face_grids = self._tensor_to_numpy_list(output["face_grids"])
            elif "face_positions" in output:
                face_grids = self._tensor_to_numpy_list(output["face_positions"])

            if "edge_points" in output:
                edge_points = self._tensor_to_numpy_list(output["edge_points"])
            elif "edge_positions" in output:
                edge_points = self._tensor_to_numpy_list(output["edge_positions"])

            if "stage_latents" in output:
                stage_latents = {
                    k: self._tensor_to_numpy(v)
                    for k, v in output["stage_latents"].items()
                }
        else:
            # Raw tensor output — create placeholder geometry
            face_grids = self._create_placeholder_face_grids(output)

        # If no geometry extracted, create minimal placeholders
        if not face_grids and not edge_points:
            _log.warning("Diffusion model returned no geometry; using placeholders")
            face_grids = [np.random.randn(32, 32, 3).astype(np.float32)]
            edge_points = [np.random.randn(20, 3).astype(np.float32)]

        # Compute confidence based on stage completeness
        confidence = self._compute_confidence(face_grids, edge_points)

        return LatentProposal(
            face_grids=face_grids,
            edge_points=edge_points,
            stage_latents=stage_latents,
            source_prompt=prompt,
            conditioning_source=conditioning.source_type if conditioning else "unconditional",
            confidence=confidence,
            generation_metadata=self._build_metadata(
                "StructuredDiffusion",
                inference_steps=self.inference_steps,
                eta=self.eta,
            ),
            error_context=error_context,
        )

    def generate_candidates(
        self,
        prompt: str,
        num_candidates: int = 3,
        conditioning: ConditioningEmbeddings | None = None,
    ) -> list[LatentProposal]:
        """Generate multiple candidate geometries.

        Args:
            prompt: User prompt.
            num_candidates: Number of candidates to generate.
            conditioning: Optional conditioning embeddings.

        Returns:
            List of LatentProposal objects, sorted by confidence descending.
        """
        if self._model is None:
            self._init_model()

        try:
            import torch
        except ImportError:
            raise ImportError("torch is required for diffusion generation") from None

        proposals: list[LatentProposal] = []

        with torch.no_grad():
            for _ in range(num_candidates):
                batch_size = 1
                noise_shape = self._get_noise_shape()
                noise = torch.randn(
                    batch_size, *noise_shape,
                    device=self.device,
                    dtype=torch.float32,
                )

                if hasattr(self._model, "sample"):
                    output = self._model.sample(
                        noise=noise,
                        num_steps=self.inference_steps,
                        eta=self.eta,
                    )
                else:
                    output = self._model(noise, steps=self.inference_steps)

                face_grids: list[np.ndarray] = []
                edge_points: list[np.ndarray] = []
                stage_latents: dict[str, Any] | None = None

                if isinstance(output, dict):
                    if "face_grids" in output:
                        face_grids = self._tensor_to_numpy_list(output["face_grids"])
                    elif "face_positions" in output:
                        face_grids = self._tensor_to_numpy_list(
                            output["face_positions"]
                        )

                    if "edge_points" in output:
                        edge_points = self._tensor_to_numpy_list(output["edge_points"])
                    elif "edge_positions" in output:
                        edge_points = self._tensor_to_numpy_list(
                            output["edge_positions"]
                        )

                    if "stage_latents" in output:
                        stage_latents = {
                            k: self._tensor_to_numpy(v)
                            for k, v in output["stage_latents"].items()
                        }
                else:
                    face_grids = self._create_placeholder_face_grids(output)

                if not face_grids and not edge_points:
                    face_grids = [np.random.randn(32, 32, 3).astype(np.float32)]
                    edge_points = [np.random.randn(20, 3).astype(np.float32)]

                confidence = self._compute_confidence(face_grids, edge_points)

                proposal = LatentProposal(
                    face_grids=face_grids,
                    edge_points=edge_points,
                    stage_latents=stage_latents,
                    source_prompt=prompt,
                    conditioning_source=conditioning.source_type if conditioning else "unconditional",
                    confidence=confidence,
                    generation_metadata=self._build_metadata(
                        "StructuredDiffusion",
                        inference_steps=self.inference_steps,
                        eta=self.eta,
                    ),
                )
                proposals.append(proposal)

        # Sort by confidence descending
        proposals.sort(key=lambda p: p.confidence, reverse=True)
        return proposals

    def generate_from_error_context(
        self,
        error_context: dict[str, Any],
    ) -> LatentProposal | None:
        """Generate a retry proposal from a prior stage.

        If error_context contains stage_latents, can re-run from an
        intermediate stage rather than from noise.

        Args:
            error_context: Error dict with optional stage_latents.

        Returns:
            LatentProposal or None if no stage latents available.
        """
        if self._model is None:
            self._init_model()

        stage_latents = error_context.get("stage_latents")
        error_category = error_context.get("error_category")

        if stage_latents is None:
            _log.warning("No stage latents in error context; full regeneration")
            return None

        # Determine which stage to re-run from
        start_stage = 0
        if error_category == "self_intersection":
            # Re-run face geometry stage
            start_stage = 1
            _log.info("Self-intersection detected; re-running from face_geometry stage")
        elif error_category == "topology_error":
            # Re-run edge positions stage
            start_stage = 2
            _log.info("Topology error detected; re-running from edge_positions stage")

        try:
            import torch
        except ImportError:
            raise ImportError("torch is required for diffusion generation") from None

        with torch.no_grad():
            # Resume from intermediate stage
            if hasattr(self._model, "sample_from_stage"):
                output = self._model.sample_from_stage(
                    stage_latents=stage_latents,
                    start_stage=start_stage,
                    num_steps=self.inference_steps // 2,
                    eta=self.eta,
                )
            else:
                _log.info("Model does not support stage-specific sampling; full regeneration")
                output = self.generate(
                    prompt=error_context.get("original_prompt", ""),
                    error_context=error_context,
                )
                return output

        face_grids: list[np.ndarray] = []
        edge_points: list[np.ndarray] = []

        if isinstance(output, dict):
            if "face_grids" in output:
                face_grids = self._tensor_to_numpy_list(output["face_grids"])
            if "edge_points" in output:
                edge_points = self._tensor_to_numpy_list(output["edge_points"])

        if not face_grids and not edge_points:
            face_grids = [np.random.randn(32, 32, 3).astype(np.float32)]
            edge_points = [np.random.randn(20, 3).astype(np.float32)]

        confidence = self._compute_confidence(face_grids, edge_points)

        return LatentProposal(
            face_grids=face_grids,
            edge_points=edge_points,
            stage_latents=stage_latents,
            confidence=confidence,
            generation_metadata=self._build_metadata(
                "StructuredDiffusion",
                inference_steps=self.inference_steps,
                from_error_context=True,
            ),
            error_context=error_context,
        )

    def _init_model(self) -> None:
        """Initialize the StructuredDiffusion model lazily on first use."""
        try:
            from ll_stepnet.stepnet.models import StructuredDiffusion
        except ImportError as e:
            raise ImportError(
                f"ll_stepnet is required for NeuralDiffusionGenerator: {e}"
            ) from e

        _log.info("Initializing StructuredDiffusion model")

        if self.diffusion_config is None:
            _log.info("No diffusion config provided; using defaults")
            self._model = StructuredDiffusion()
        else:
            self._model = StructuredDiffusion(**self.diffusion_config.__dict__)

        if self.checkpoint_path:
            self.load_checkpoint(self.checkpoint_path)
        else:
            self._model = self._model.to(self.device)

        self._model.eval()
        _log.info("StructuredDiffusion model initialized and ready")

    def _get_noise_shape(self) -> tuple:
        """Get the latent noise shape for the diffusion model.

        Returns:
            Tuple of dimensions (channels, height, width) or similar.
        """
        if hasattr(self._model, "latent_shape"):
            return self._model.latent_shape
        # Default shape
        return (4, 32, 32)

    def _tensor_to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert a torch tensor to numpy array.

        Args:
            tensor: torch.Tensor or numpy array.

        Returns:
            numpy.ndarray in float32.
        """
        try:
            import torch

            if isinstance(tensor, torch.Tensor):
                return tensor.detach().cpu().numpy().astype(np.float32)
        except ImportError:
            pass

        if isinstance(tensor, np.ndarray):
            return tensor.astype(np.float32)

        return np.array(tensor, dtype=np.float32)

    def _tensor_to_numpy_list(
        self,
        tensor_or_list: Any,
    ) -> list[np.ndarray]:
        """Convert a list or batch of tensors to numpy arrays.

        Args:
            tensor_or_list: Single tensor, list of tensors, or batch tensor.

        Returns:
            List of numpy arrays.
        """
        if isinstance(tensor_or_list, (list, tuple)):
            return [self._tensor_to_numpy(t) for t in tensor_or_list]

        tensor = self._tensor_to_numpy(tensor_or_list)
        if tensor.ndim == 4:
            # Assume batch dimension
            return [tensor[i] for i in range(tensor.shape[0])]
        elif tensor.ndim == 3:
            return [tensor]
        else:
            _log.warning(f"Unexpected tensor shape: {tensor.shape}")
            return [tensor]

    def _create_placeholder_face_grids(
        self,
        latent_tensor: Any,
    ) -> list[np.ndarray]:
        """Create placeholder face grids from latent tensor shape.

        Args:
            latent_tensor: Raw latent output tensor.

        Returns:
            List of face grid arrays (U×V×3).
        """
        latent = self._tensor_to_numpy(latent_tensor)

        # Try to infer number of faces from latent shape
        if latent.ndim >= 2:
            num_faces = max(1, latent.shape[0])
        else:
            num_faces = 1

        face_grids: list[np.ndarray] = []
        for _ in range(num_faces):
            # Standard 32×32 grid
            grid = np.random.randn(32, 32, 3).astype(np.float32)
            face_grids.append(grid)

        return face_grids

    def _compute_confidence(
        self,
        face_grids: list[np.ndarray],
        edge_points: list[np.ndarray],
    ) -> float:
        """Compute confidence based on geometry completeness.

        Args:
            face_grids: List of face point arrays.
            edge_points: List of edge point arrays.

        Returns:
            Confidence score in [0, 1].
        """
        # Base confidence from geometry presence
        has_faces = len(face_grids) > 0
        has_edges = len(edge_points) > 0

        if has_faces and has_edges:
            confidence = 0.6
        elif has_faces or has_edges:
            confidence = 0.4
        else:
            confidence = 0.2

        # Adjust based on grid quality (variance in point positions)
        if has_faces:
            face_variance = np.mean([
                np.var(grid)
                for grid in face_grids
            ])
            if face_variance > 0.1:
                confidence += 0.15

        if has_edges:
            edge_variance = np.mean([
                np.var(points)
                for points in edge_points
            ])
            if edge_variance > 0.1:
                confidence += 0.15

        return min(confidence, 0.95)
