"""Neural VAE generator — wraps STEPVAE + CADGenerationPipeline.

Generates command sequences from VAE latent space, with support for
error-aware latent perturbation on retries.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ll_gen.conditioning.embeddings import ConditioningEmbeddings
from ll_gen.config import ErrorCategory
from ll_gen.generators.base import BaseNeuralGenerator
from ll_gen.proposals.command_proposal import CommandSequenceProposal

_log = logging.getLogger(__name__)


class NeuralVAEGenerator(BaseNeuralGenerator):
    """Neural generator wrapping STEPVAE + CADGenerationPipeline.

    Attributes:
        vae_config: Optional VAE configuration object.
        temperature: Sampling temperature (higher = more stochastic).
        max_seq_len: Maximum sequence length (default 60).
        _pipeline: Lazy-initialized CADGenerationPipeline.
    """

    def __init__(
        self,
        vae_config: Any | None = None,
        checkpoint_path: Path | None = None,
        device: str = "cpu",
        temperature: float = 0.8,
        max_seq_len: int = 60,
    ) -> None:
        """Initialize the VAE generator.

        Args:
            vae_config: Optional VAEConfig object from ll_stepnet.
            checkpoint_path: Path to model checkpoint.
            device: Target device ("cpu" or "cuda").
            temperature: Sampling temperature.
            max_seq_len: Maximum sequence length.
        """
        super().__init__(device=device, checkpoint_path=checkpoint_path)
        self.vae_config = vae_config
        self.temperature = temperature
        self.max_seq_len = max_seq_len
        self._pipeline: Any | None = None

    def _extract_pipeline_result(
        self,
        result: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[int], Any, Any, np.ndarray | None]:
        """Extract command dicts, token IDs, logits, and latent from a pipeline result.

        Args:
            result: Single result dict from CADGenerationPipeline.generate().

        Returns:
            Tuple of (command_dicts, token_ids, command_logits, param_logits, latent_vector).
        """
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

        latent_vector: np.ndarray | None = None
        if hasattr(self._model, "last_latent"):
            last_latent = self._model.last_latent
            if last_latent is not None:
                try:
                    import torch

                    if isinstance(last_latent, torch.Tensor):
                        latent_vector = last_latent.detach().cpu().numpy()
                    else:
                        latent_vector = np.array(last_latent)
                except (ImportError, AttributeError):
                    pass

        return command_dicts, token_ids, command_logits, param_logits, latent_vector

    def generate(
        self,
        prompt: str,
        conditioning: ConditioningEmbeddings | None = None,
        error_context: dict[str, Any] | None = None,
    ) -> CommandSequenceProposal:
        """Generate a command sequence from the VAE model.

        Args:
            prompt: User prompt (stored in proposal for tracing).
            conditioning: Optional conditioning embeddings.
            error_context: Optional error context from a failed attempt.

        Returns:
            CommandSequenceProposal with decoded token sequence.
        """
        if self._model is None:
            self._init_model()

        temp = self._adjust_temperature_for_error(self.temperature, error_context)

        result_list = self._pipeline.generate(
            num_samples=1,
            reconstruct=False,
            temperature=temp,
        )

        if not result_list:
            _log.warning("VAE pipeline returned empty result")
            return CommandSequenceProposal(
                source_prompt=prompt,
                conditioning_source=conditioning.source_type if conditioning else "unconditional",
                confidence=0.0,
                generation_metadata=self._build_metadata("STEPVAE", temperature=temp),
                error_context=error_context,
            )

        command_dicts, token_ids, command_logits, param_logits, latent_vector = (
            self._extract_pipeline_result(result_list[0])
        )
        confidence = self._compute_confidence(command_logits, param_logits)

        return CommandSequenceProposal(
            token_ids=token_ids,
            command_dicts=command_dicts,
            source_prompt=prompt,
            conditioning_source=conditioning.source_type if conditioning else "unconditional",
            confidence=confidence,
            generation_metadata=self._build_metadata("STEPVAE", temperature=temp),
            latent_vector=latent_vector,
            error_context=error_context,
        )

    def generate_for_training(
        self,
        prompt: str,
        conditioning: ConditioningEmbeddings | None = None,
        error_context: dict[str, Any] | None = None,
    ) -> CommandSequenceProposal:
        """Generate with gradients, computing log-probs on the sampled trajectory.

        Runs the VAE forward pass *directly* (bypassing
        ``CADGenerationPipeline.generate()`` which wraps everything in
        ``torch.no_grad``).  This keeps the computation graph alive so
        that the returned ``proposal.log_probs`` is a differentiable
        scalar usable in a REINFORCE loss.

        The key correctness property: each token is sampled **once** from
        the logits produced by the live forward pass, and log-probs are
        evaluated on that same sample — ensuring an unbiased REINFORCE
        estimator.

        Args:
            prompt: User prompt (stored for tracing).
            conditioning: Optional conditioning embeddings.
            error_context: Optional error context from a failed attempt.

        Returns:
            CommandSequenceProposal with ``log_probs`` and ``entropy`` set.
        """
        import torch
        import torch.nn.functional as functional

        if self._model is None:
            self._init_model()

        temp = self._adjust_temperature_for_error(self.temperature, error_context)

        # --- Forward pass WITHOUT no_grad so gradients flow ---
        # We bypass the pipeline (which uses torch.no_grad) and call the
        # model directly to get logits with a live computation graph.
        device = torch.device(self.device)
        z = torch.randn(1, self._model.latent_dim, device=device)

        # Store latent for downstream inspection
        self._model.last_latent = z

        hidden = self._model.decode(z, seq_len=self.max_seq_len)
        command_logits = self._model.command_head(hidden)       # [1, S, C]
        param_logits = [
            head(hidden) for head in self._model.param_heads    # 16 × [1, S, P]
        ]

        # Strip batch dim → [S, C] and [S, P] respectively
        command_logits = command_logits[0]
        param_logits_2d = [pl[0] for pl in param_logits]        # 16 × [S, P]

        # --- Single-pass sampling + log-prob accumulation ---
        from ll_gen.generators.base import (
            BOS_TOKEN_ID,
            CMD_TOKEN_MAP,
            EOS_CMD_TOKEN_ID,
            EOS_TOKEN_ID,
            PARAM_OFFSET,
            PARAMETER_MASKS,
        )

        seq_len = min(command_logits.shape[0], self.max_seq_len)
        log_probs_accum: list[torch.Tensor] = []
        entropy_accum: list[torch.Tensor] = []
        token_ids: list[int] = [BOS_TOKEN_ID]

        for pos in range(seq_len):
            cmd_logits_pos = command_logits[pos] / max(temp, 1e-8)
            cmd_log_probs = functional.log_softmax(cmd_logits_pos, dim=-1)

            # Sample command type (single draw)
            cmd_dist = torch.distributions.Categorical(logits=cmd_logits_pos)
            cmd_type_t = cmd_dist.sample()
            cmd_type = int(cmd_type_t.item())

            log_probs_accum.append(cmd_dist.log_prob(cmd_type_t))
            entropy_accum.append(cmd_dist.entropy())

            if cmd_type not in CMD_TOKEN_MAP:
                break

            cmd_token = CMD_TOKEN_MAP[cmd_type]
            token_ids.append(cmd_token)

            if cmd_token in (EOS_CMD_TOKEN_ID, EOS_TOKEN_ID):
                break

            # Sample parameters from the SAME logits (single draw each)
            self._sample_params_with_log_probs(
                param_logits_2d, pos, cmd_type, temp,
                token_ids, log_probs_accum, entropy_accum,
            )

        if token_ids[-1] not in (EOS_TOKEN_ID, EOS_CMD_TOKEN_ID):
            token_ids.append(EOS_TOKEN_ID)

        total_log_prob: torch.Tensor | None = None
        entropy_value = 0.0
        if log_probs_accum:
            total_log_prob = torch.stack(log_probs_accum).sum()
            entropy_value = float(
                torch.stack(entropy_accum).mean().detach().cpu()
            )

        # Extract latent vector for metadata
        latent_vector: np.ndarray | None = z.detach().cpu().numpy()

        # Confidence from the live logits (detached for scalar computation)
        confidence = self._compute_confidence(
            command_logits.detach(), [pl.detach() for pl in param_logits_2d]
        )

        return CommandSequenceProposal(
            token_ids=token_ids,
            command_dicts=[],
            source_prompt=prompt,
            conditioning_source=conditioning.source_type if conditioning else "unconditional",
            confidence=confidence,
            generation_metadata=self._build_metadata("STEPVAE", temperature=temp),
            latent_vector=latent_vector,
            error_context=error_context,
            log_probs=total_log_prob,
            entropy=entropy_value,
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

        proposals: list[CommandSequenceProposal] = []

        for _ in range(num_candidates):
            result_list = self._pipeline.generate(
                num_samples=1,
                reconstruct=False,
                temperature=self.temperature,
            )

            if not result_list:
                _log.warning("VAE pipeline returned empty result for candidate")
                continue

            command_dicts, token_ids, command_logits, param_logits, latent_vector = (
                self._extract_pipeline_result(result_list[0])
            )
            confidence = self._compute_confidence(command_logits, param_logits)

            proposal = CommandSequenceProposal(
                token_ids=token_ids,
                command_dicts=command_dicts,
                source_prompt=prompt,
                conditioning_source=conditioning.source_type if conditioning else "unconditional",
                confidence=confidence,
                generation_metadata=self._build_metadata(
                    "STEPVAE",
                    temperature=self.temperature,
                ),
                latent_vector=latent_vector,
            )
            proposals.append(proposal)

        # Sort by confidence descending
        proposals.sort(key=lambda p: p.confidence, reverse=True)
        return proposals

    def generate_from_error_context(
        self,
        error_context: dict[str, Any],
    ) -> CommandSequenceProposal | None:
        """Generate a retry proposal by perturbing a prior latent vector.

        If error_context contains 'previous_latent_vector' and the error
        category is known, perturb the latent and decode directly.

        Args:
            error_context: Error dict with optional 'previous_latent_vector'.

        Returns:
            CommandSequenceProposal or None if no prior latent available.
        """
        if self._model is None:
            self._init_model()

        latent_vector = error_context.get("previous_latent_vector")
        if latent_vector is None:
            _log.warning("No previous latent vector in error context")
            return None

        # Convert to numpy if needed
        try:
            import torch

            if isinstance(latent_vector, torch.Tensor):
                latent_vector = latent_vector.detach().cpu().numpy()
        except ImportError:
            pass

        latent_vector = np.array(latent_vector)

        # Determine perturbation scale based on error category
        error_category = error_context.get("error_category")
        perturbation_scale = 0.1

        if error_category == ErrorCategory.TOPOLOGY_ERROR.value:
            perturbation_scale = 0.05
        elif error_category == ErrorCategory.DEGENERATE_SHAPE.value:
            perturbation_scale = 0.2
        elif error_category == ErrorCategory.SELF_INTERSECTION.value:
            perturbation_scale = 0.08

        # Perturb latent
        noise = np.random.randn(*latent_vector.shape)
        perturbed_latent = latent_vector + noise * perturbation_scale
        perturbed_latent_norm = np.linalg.norm(perturbed_latent)

        if perturbed_latent_norm > 0:
            perturbed_latent = (
                perturbed_latent / perturbed_latent_norm
            ) * np.linalg.norm(latent_vector)

        # Decode perturbed latent directly
        _log.info("Decoding perturbed latent vector for retry generation")

        token_ids: list[int] = []
        confidence = 0.0

        try:
            import torch as torch_mod

            latent_tensor = torch_mod.from_numpy(
                perturbed_latent.astype(np.float32)
            ).unsqueeze(0).to(self.device)

            with torch_mod.no_grad():
                output = self._model.decode(latent_tensor)

            command_logits = output.get("command_logits") if isinstance(output, dict) else None
            param_logits = output.get("param_logits") if isinstance(output, dict) else None

            if command_logits is not None and param_logits is not None:
                token_ids = self._logits_to_token_ids(
                    command_logits, param_logits, max_seq_len=self.max_seq_len
                )
            confidence = self._compute_confidence(command_logits, param_logits)
        except (ImportError, AttributeError, RuntimeError) as exc:
            _log.warning(f"Failed to decode perturbed latent: {exc}")

        # Return a proposal with the perturbed latent
        return CommandSequenceProposal(
            token_ids=token_ids,
            latent_vector=perturbed_latent,
            confidence=confidence,
            generation_metadata=self._build_metadata(
                "STEPVAE",
                temperature=self.temperature,
                from_error_context=True,
            ),
            error_context=error_context,
        )

    def _init_model(self) -> None:
        """Initialize the STEPVAE model lazily on first use."""
        try:
            from ll_stepnet.stepnet.models import STEPVAE
            from ll_stepnet.stepnet.pipeline import CADGenerationPipeline
        except ImportError as e:
            raise ImportError(
                f"ll_stepnet is required for NeuralVAEGenerator: {e}"
            ) from e

        _log.info("Initializing STEPVAE model")

        # Create VAE model
        if self.vae_config is None:
            _log.info("No VAE config provided; using defaults")
            self._model = STEPVAE()
        else:
            from dataclasses import asdict

            self._model = STEPVAE(**asdict(self.vae_config))

        # Load checkpoint if provided
        if self.checkpoint_path:
            self.load_checkpoint(self.checkpoint_path)
        else:
            self._model = self._model.to(self.device)

        self._model.eval()

        # Initialize pipeline
        self._pipeline = CADGenerationPipeline(
            model=self._model,
            mode="vae",
            device=self.device,
        )

        _log.info("STEPVAE model initialized and ready")
