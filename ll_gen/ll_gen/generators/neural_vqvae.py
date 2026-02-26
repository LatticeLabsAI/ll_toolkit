"""Neural VQ-VAE generator — wraps VQVAEModel + CADGenerationPipeline.

Generates command sequences from quantized codebook-based VAE, with
support for selective codebook re-sampling on retries.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ll_gen.conditioning.embeddings import ConditioningEmbeddings
from ll_gen.generators.base import BaseNeuralGenerator
from ll_gen.proposals.command_proposal import CommandSequenceProposal

_log = logging.getLogger(__name__)


class NeuralVQVAEGenerator(BaseNeuralGenerator):
    """Neural generator wrapping VQVAEModel + CADGenerationPipeline.

    Uses discrete quantized codebooks instead of continuous VAE latent,
    enabling masked codebook re-sampling for targeted error recovery.

    Attributes:
        checkpoint_path: Path to model checkpoint.
        temperature: Sampling temperature.
        codebook_dim: Codebook vector dimension.
        max_seq_len: Maximum sequence length.
        _pipeline: Lazy-initialized CADGenerationPipeline.
    """

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        device: str = "cpu",
        temperature: float = 0.7,
        codebook_dim: int = 512,
        max_seq_len: int = 60,
    ) -> None:
        """Initialize the VQ-VAE generator.

        Args:
            checkpoint_path: Path to model checkpoint.
            device: Target device ("cpu" or "cuda").
            temperature: Sampling temperature.
            codebook_dim: Codebook embedding dimension.
            max_seq_len: Maximum sequence length.
        """
        super().__init__(device=device, checkpoint_path=checkpoint_path)
        self.temperature = temperature
        self.codebook_dim = codebook_dim
        self.max_seq_len = max_seq_len
        self._pipeline: Any | None = None

    def generate(
        self,
        prompt: str,
        conditioning: ConditioningEmbeddings | None = None,
        error_context: dict[str, Any] | None = None,
    ) -> CommandSequenceProposal:
        """Generate a command sequence from the VQ-VAE model.

        Args:
            prompt: User prompt (stored for tracing).
            conditioning: Optional conditioning embeddings.
            error_context: Optional error context from a failed attempt.

        Returns:
            CommandSequenceProposal with decoded token sequence.
        """
        if self._model is None:
            self._init_model()

        temp = self._adjust_temperature_for_error(self.temperature, error_context)

        # Generate from VQ-VAE
        result_list = self._pipeline.generate(
            num_samples=1,
            reconstruct=False,
            temperature=temp,
        )

        if not result_list:
            _log.warning("VQ-VAE pipeline returned empty result")
            return CommandSequenceProposal(
                source_prompt=prompt,
                conditioning_source=conditioning.source_type if conditioning else "unconditional",
                confidence=0.0,
                generation_metadata=self._build_metadata("VQVAE", temperature=temp),
                error_context=error_context,
            )

        result = result_list[0]

        # Extract command dicts
        command_dicts: list[dict[str, Any]] = []
        if "commands" in result and result["commands"]:
            command_dicts = result["commands"]

        # Extract token IDs from logits
        token_ids: list[int] = []
        command_logits = result.get("command_logits")
        param_logits = result.get("param_logits")

        if command_logits is not None and param_logits is not None:
            token_ids = self._logits_to_token_ids(
                command_logits, param_logits, max_seq_len=self.max_seq_len
            )

        # Extract codebook indices if available
        codebook_indices: np.ndarray | None = None
        if "codebook_indices" in result:
            try:
                import torch

                indices = result["codebook_indices"]
                if isinstance(indices, torch.Tensor):
                    codebook_indices = indices.detach().cpu().numpy()
                else:
                    codebook_indices = np.array(indices)
            except ImportError:
                pass

        # Compute confidence
        confidence = self._compute_confidence(command_logits, param_logits)

        return CommandSequenceProposal(
            token_ids=token_ids,
            command_dicts=command_dicts,
            source_prompt=prompt,
            conditioning_source=conditioning.source_type if conditioning else "unconditional",
            confidence=confidence,
            generation_metadata=self._build_metadata(
                "VQVAE",
                temperature=temp,
                has_codebook_indices=codebook_indices is not None,
                codebook_dim=self.codebook_dim,
                codebook_indices=codebook_indices,
            ),
            error_context=error_context,
        )

    def generate_candidates(
        self,
        prompt: str,
        num_candidates: int = 3,
        conditioning: ConditioningEmbeddings | None = None,
    ) -> list[CommandSequenceProposal]:
        """Generate multiple candidate sequences.

        Args:
            prompt: User prompt.
            num_candidates: Number of candidates to generate.
            conditioning: Optional conditioning embeddings.

        Returns:
            List of CommandSequenceProposal objects, sorted by confidence descending.
        """
        if self._model is None:
            self._init_model()

        result_list = self._pipeline.generate(
            num_samples=num_candidates,
            reconstruct=False,
            temperature=self.temperature,
        )

        proposals: list[CommandSequenceProposal] = []

        for result in result_list:
            command_dicts: list[dict[str, Any]] = []
            if "commands" in result and result["commands"]:
                command_dicts = result["commands"]

            token_ids: list[int] = []
            command_logits = result.get("command_logits")
            param_logits = result.get("param_logits")

            if command_logits is not None and param_logits is not None:
                token_ids = self._logits_to_token_ids(
                    command_logits, param_logits, max_seq_len=self.max_seq_len
                )

            confidence = self._compute_confidence(command_logits, param_logits)

            proposal = CommandSequenceProposal(
                token_ids=token_ids,
                command_dicts=command_dicts,
                source_prompt=prompt,
                conditioning_source=conditioning.source_type if conditioning else "unconditional",
                confidence=confidence,
                generation_metadata=self._build_metadata(
                    "VQVAE",
                    temperature=self.temperature,
                    codebook_dim=self.codebook_dim,
                ),
            )
            proposals.append(proposal)

        # Sort by confidence descending
        proposals.sort(key=lambda p: p.confidence, reverse=True)
        return proposals

    def generate_from_masked_codebooks(
        self,
        error_context: dict[str, Any],
    ) -> CommandSequenceProposal | None:
        """Generate a retry by re-sampling with masked codebooks.

        Excludes codebook entries that led to topology or geometry failures.

        Args:
            error_context: Error dict with optional masked_codebook_indices.

        Returns:
            CommandSequenceProposal or None if no mask available.
        """
        if self._model is None:
            self._init_model()

        masked_indices = error_context.get("masked_codebook_indices")
        if masked_indices is None:
            _log.warning("No masked codebook indices in error context")
            return None

        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError("torch is required for VQ-VAE generation") from None

        _log.info(
            f"Re-sampling with masked codebooks: {len(masked_indices)} entries excluded"
        )

        # Re-sample with mask applied
        # This would require pipeline support for masked sampling
        if hasattr(self._pipeline, "generate_masked"):
            result_list = self._pipeline.generate_masked(
                num_samples=1,
                masked_codebook_indices=masked_indices,
                temperature=self.temperature * 0.9,
            )
        else:
            _log.warning("Pipeline does not support masked sampling; standard re-generation")
            result_list = self._pipeline.generate(
                num_samples=1,
                temperature=self.temperature * 0.9,
            )

        if not result_list:
            return None

        result = result_list[0]

        command_dicts: list[dict[str, Any]] = []
        if "commands" in result and result["commands"]:
            command_dicts = result["commands"]

        token_ids: list[int] = []
        command_logits = result.get("command_logits")
        param_logits = result.get("param_logits")

        if command_logits is not None and param_logits is not None:
            token_ids = self._logits_to_token_ids(
                command_logits, param_logits, max_seq_len=self.max_seq_len
            )

        confidence = self._compute_confidence(command_logits, param_logits)

        return CommandSequenceProposal(
            token_ids=token_ids,
            command_dicts=command_dicts,
            source_prompt=error_context.get("original_prompt", ""),
            confidence=confidence,
            generation_metadata=self._build_metadata(
                "VQVAE",
                temperature=self.temperature,
                from_masked_resampling=True,
            ),
            error_context=error_context,
        )

    def _init_model(self) -> None:
        """Initialize the VQVAEModel lazily on first use."""
        try:
            from ll_stepnet.stepnet.models import VQVAEModel
            from ll_stepnet.stepnet.pipeline import CADGenerationPipeline
        except ImportError as e:
            raise ImportError(
                f"ll_stepnet is required for NeuralVQVAEGenerator: {e}"
            ) from e

        _log.info("Initializing VQVAEModel")

        self._model = VQVAEModel()

        if self.checkpoint_path:
            self.load_checkpoint(self.checkpoint_path)
        else:
            self._model = self._model.to(self.device)

        self._model.eval()

        # Initialize pipeline
        self._pipeline = CADGenerationPipeline(
            model=self._model,
            mode="vqvae",
            device=self.device,
        )

        _log.info("VQVAEModel initialized and ready")
