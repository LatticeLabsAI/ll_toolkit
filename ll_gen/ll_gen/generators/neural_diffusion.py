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
        """Generate with a (decoupled) policy gradient signal for RL training.

        ``target_dimensions`` is accepted for trainer-call uniformity (diffusion
        does not yet condition on it).

        IMPORTANT — the REINFORCE signal returned here is currently DECOUPLED
        from the geometry that is actually produced. ``StructuredDiffusion``
        does not expose a noise-conditioned / log-prob sampling API: its
        ``sample()`` draws its own noise internally and runs the full
        (non-differentiable) denoising chain. The ``noise`` tensor whose
        Gaussian-prior log-prob we compute below is therefore an *independent*
        sample, not the latent that generated ``output``.

        Consequence: ``log_probs`` is an unbiased-but-high-variance stand-in
        policy gradient (the score of an N(0, I) draw), **not** a true DDPO
        gradient flowing through the denoising trajectory. It lets the RL loop
        run end-to-end, but the reward is not attached to the actual sampling
        path.

        Properly wiring the sampled noise through ``sample()`` — or adding a
        ``sample_with_log_prob`` DDPO path to ``StructuredDiffusion`` so the
        reward attaches to the real trajectory — is tracked as future work in
        the M2 training plan. Until then, treat the diffusion RL signal as
        approximate.

        Args:
            prompt: User prompt.
            conditioning: Optional conditioning embeddings.
            error_context: Optional error context.

        Returns:
            LatentProposal with ``log_probs`` (prior log-prob of an independent
            noise sample) and ``entropy`` set.
        """
        import torch

        if self._model is None:
            self._init_model()

        batch_size = 1
        noise_shape = self._get_noise_shape()

        # Independent N(0, I) sample whose prior log-prob is the (decoupled)
        # REINFORCE signal — see the method docstring. This is NOT threaded into
        # sample() below, which draws its own internal noise.
        noise = torch.randn(
            batch_size,
            *noise_shape,
            device=self.device,
            dtype=torch.float32,
            requires_grad=True,
        )

        # Log-prob of noise under N(0, I): -0.5 * sum(z^2) - 0.5*d*log(2*pi)
        d = noise.numel()
        log_prob = -0.5 * noise.pow(2).sum() - 0.5 * d * torch.log(
            torch.tensor(2.0 * torch.pi, device=self.device)
        )

        # Entropy of N(0,I) = 0.5 * d * (1 + log(2*pi))
        entropy_value = float(0.5 * d * (1.0 + np.log(2.0 * np.pi)))

        # sample() runs the full denoising chain with its OWN internal noise; the
        # geometry it returns is independent of `noise` above (see docstring).
        with torch.no_grad():
            output = self._model.sample(batch_size=batch_size, device=self.device)

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
