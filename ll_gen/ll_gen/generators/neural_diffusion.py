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
from ll_gen.config import ErrorCategory
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

    def _extract_geometry_from_output(
        self,
        output: Any,
    ) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any] | None]:
        """Extract face grids, edge points, and stage latents from model output.

        Args:
            output: Raw model output (dict or tensor).

        Returns:
            Tuple of (face_grids, edge_points, stage_latents).
        """
        face_grids: list[np.ndarray] = []
        edge_points: list[np.ndarray] = []
        stage_latents: dict[str, Any] | None = None

        if isinstance(output, dict):
            # The diffusion model decodes its final stage tokens into batched
            # primitive-set tensors: face_grids [B, N_faces, U, V, 3] and
            # edge_points [B, N_edges, M, 3]. Split them into per-primitive
            # arrays ([U, V, 3] per face, [M, 3] per edge).
            if "face_grids" in output:
                face_grids = self._decoded_primitive_list(output["face_grids"], 3)
            elif "face_positions" in output:
                face_grids = self._tensor_to_numpy_list(output["face_positions"])

            if "edge_points" in output:
                edge_points = self._decoded_primitive_list(output["edge_points"], 2)
            elif "edge_positions" in output:
                edge_points = self._tensor_to_numpy_list(output["edge_positions"])

            if "stage_latents" in output:
                stage_latents = {
                    k: self._tensor_to_numpy(v)
                    for k, v in output["stage_latents"].items()
                }
        else:
            # Bare-tensor output (no stage dict): the sampled latent IS the
            # geometry representation this model produces — StructuredDiffusion
            # has no separate latent→grid decoder — so surface it directly,
            # exactly as the dict path does for its stage latents above.
            face_grids = self._latent_to_face_grids(output)

        if not face_grids and not edge_points:
            _log.warning(
                "Diffusion model returned no geometry; returning proposal "
                "with empty geometry (callers should handle None/empty)"
            )

        return face_grids, edge_points, stage_latents

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
            # StructuredDiffusion.sample() draws its own noise internally and
            # returns a {stage_name: latent [B, D]} dict.
            output = self._model.sample(batch_size=1, device=self.device)

        face_grids, edge_points, stage_latents = self._extract_geometry_from_output(
            output
        )
        confidence = self._compute_confidence(face_grids, edge_points)

        return LatentProposal(
            face_grids=face_grids,
            edge_points=edge_points,
            stage_latents=stage_latents,
            source_prompt=prompt,
            conditioning_source=(
                conditioning.source_type if conditioning else "unconditional"
            ),
            confidence=confidence,
            generation_metadata=self._build_metadata(
                "StructuredDiffusion",
                inference_steps=self.inference_steps,
                eta=self.eta,
            ),
            error_context=error_context,
        )

    def generate_for_training(
        self,
        prompt: str,
        conditioning: ConditioningEmbeddings | None = None,
        error_context: dict[str, Any] | None = None,
        target_dimensions: tuple[float, float, float] | None = None,
    ) -> LatentProposal:
        """Generate with a REAL DDPO policy-gradient signal for RL training.

        ``target_dimensions`` is accepted for trainer-call uniformity (diffusion
        does not yet condition on it).

        This runs ``StructuredDiffusion.sample_with_log_prob`` — a stochastic
        DDIM reverse process (DDPO; Black et al., 2023) executed **with
        gradients enabled**. The returned ``log_probs`` is the sum of the
        per-step Gaussian log-probabilities of the actual sampled denoising
        trajectory, whose transition means are produced by the denoiser
        network. It is therefore connected to the model parameters: the RL
        trainer's ``-advantage * log_probs`` REINFORCE update backpropagates
        into the diffusion denoisers and trains them. This replaces the former
        decoupled noise-prior stand-in, which produced a finite loss while
        updating zero parameters.

        A non-zero DDIM ``eta`` is required for a usable policy gradient (a
        deterministic trajectory has a degenerate policy); when ``self.eta`` is
        0 (the deterministic inference default) the training path uses
        ``eta = 1.0``.

        Args:
            prompt: User prompt.
            conditioning: Optional conditioning embeddings.
            error_context: Optional error context.
            target_dimensions: Accepted for call uniformity; unused here.

        Returns:
            LatentProposal whose ``log_probs`` is a differentiable scalar
            connected to the model parameters and whose ``entropy`` is the
            trajectory's summed Gaussian entropy.
        """
        if self._model is None:
            self._init_model()

        # Deterministic inference uses eta=0; RL needs a stochastic trajectory.
        train_eta = self.eta if self.eta and self.eta > 0.0 else 1.0

        # Stochastic DDIM sampling WITH gradients: the trajectory log-prob flows
        # through every denoiser's epsilon prediction into the model params.
        output, log_prob_batch, entropy_batch = self._model.sample_with_log_prob(
            batch_size=1,
            device=self.device,
            num_inference_steps=self.inference_steps,
            eta=train_eta,
        )

        # Sum over the (size-1) batch -> scalar trajectory log-prob (keeps grad).
        log_prob = log_prob_batch.sum()
        entropy_value = float(entropy_batch.sum().detach().cpu())

        face_grids, edge_points, stage_latents = self._extract_geometry_from_output(
            output
        )
        confidence = self._compute_confidence(face_grids, edge_points)

        return LatentProposal(
            face_grids=face_grids,
            edge_points=edge_points,
            stage_latents=stage_latents,
            source_prompt=prompt,
            conditioning_source=(
                conditioning.source_type if conditioning else "unconditional"
            ),
            confidence=confidence,
            generation_metadata=self._build_metadata(
                "StructuredDiffusion",
                inference_steps=self.inference_steps,
                eta=train_eta,
            ),
            error_context=error_context,
            log_probs=log_prob,
            entropy=entropy_value,
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
                output = self._model.sample(batch_size=1, device=self.device)

                face_grids, edge_points, stage_latents = (
                    self._extract_geometry_from_output(output)
                )
                confidence = self._compute_confidence(face_grids, edge_points)

                proposal = LatentProposal(
                    face_grids=face_grids,
                    edge_points=edge_points,
                    stage_latents=stage_latents,
                    source_prompt=prompt,
                    conditioning_source=(
                        conditioning.source_type if conditioning else "unconditional"
                    ),
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
        if error_category == ErrorCategory.SELF_INTERSECTION.value:
            # Re-run face geometry stage
            start_stage = 1
            _log.info("Self-intersection detected; re-running from face_geometry stage")
        elif error_category == ErrorCategory.TOPOLOGY_ERROR.value:
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
                _log.info(
                    "Model does not support stage-specific sampling; full regeneration"
                )
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
            _log.warning(
                "Diffusion model returned no geometry from error recovery; "
                "returning proposal with empty geometry"
            )

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
            from stepnet.diffusion import StructuredDiffusion
        except ImportError as e:
            raise ImportError(
                f"stepnet (ll-stepnet) is required for NeuralDiffusionGenerator: {e}"
            ) from e

        _log.info("Initializing StructuredDiffusion model")

        # StructuredDiffusion.__init__(self, config) takes a single config
        # object and reads its attributes via getattr — pass the object
        # directly rather than splatting it into keyword arguments.
        if self.diffusion_config is None:
            _log.info("No diffusion config provided; using defaults")
            self._model = StructuredDiffusion()
        else:
            self._model = StructuredDiffusion(config=self.diffusion_config)

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

    def _decoded_primitive_list(
        self,
        tensor: Any,
        primitive_ndim: int,
    ) -> list[np.ndarray]:
        """Split a decoded primitive-set tensor into per-primitive arrays.

        The diffusion codec returns batched sets — face grids as
        ``[B, N_faces, U, V, 3]`` and edge polylines as ``[B, N_edges, M, 3]``.
        ``primitive_ndim`` is the ndim of a *single* primitive (3 for a face
        grid ``[U, V, 3]``, 2 for an edge polyline ``[M, 3]``). Leading batch
        dimensions are dropped (generation uses ``batch_size == 1``) and the
        primitive dimension is split into one array per face/edge.

        Args:
            tensor: Decoded geometry tensor or array.
            primitive_ndim: ndim of one primitive's array.

        Returns:
            List of per-primitive ``np.ndarray`` (each with ndim
            ``primitive_ndim``).
        """
        arr = self._tensor_to_numpy(tensor)
        # Drop leading batch dim(s) until we have [N_primitives, *primitive].
        while arr.ndim > primitive_ndim + 1:
            arr = arr[0]
        if arr.ndim == primitive_ndim + 1:
            return [arr[i] for i in range(arr.shape[0])]
        return [arr]

    def _latent_to_face_grids(
        self,
        latent_tensor: Any,
    ) -> list[np.ndarray]:
        """Surface a bare-tensor diffusion output as face-grid arrays.

        ``StructuredDiffusion.sample()`` normally returns a per-stage dict
        whose ``face_positions`` / ``face_geometry`` latents are consumed by
        :meth:`_extract_geometry_from_output`. When a caller hands this method
        a bare tensor instead, that tensor *is* the model's geometry latent —
        the model has no separate latent→grid decoder — so it is converted to
        numpy and returned directly, identically to how the dict path handles
        its stage latents (see :meth:`_tensor_to_numpy_list`). A leading batch
        dimension is split into one array per sample.

        Args:
            latent_tensor: Raw latent output tensor from the diffusion model.

        Returns:
            List of latent arrays, one per face/sample.
        """
        return self._tensor_to_numpy_list(latent_tensor)

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
            face_variance = np.mean([np.var(grid) for grid in face_grids])
            if face_variance > 0.1:
                confidence += 0.15

        if has_edges:
            edge_variance = np.mean([np.var(points) for points in edge_points])
            if edge_variance > 0.1:
                confidence += 0.15

        return min(confidence, 0.95)
