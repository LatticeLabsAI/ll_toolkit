"""Neural VAE generator — wraps a STEPVAE.

Generates command sequences from VAE latent space via a single shared decoder
(``_decode_and_sample``), with support for error-aware latent perturbation on
retries.
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
    """Neural generator wrapping a STEPVAE.

    All generation paths (generate, generate_for_training, generate_candidates,
    generate_from_error_context) decode through the shared ``_decode_and_sample``,
    so RL improvements to the policy are reflected at inference.

    Attributes:
        vae_config: Optional VAE configuration object.
        temperature: Sampling temperature (higher = more stochastic).
        max_seq_len: Maximum sequence length (default 60).
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
            vae_config: Optional VAEConfig object from stepnet.
            checkpoint_path: Path to model checkpoint.
            device: Target device ("cpu" or "cuda").
            temperature: Sampling temperature.
            max_seq_len: Maximum sequence length.
        """
        super().__init__(device=device, checkpoint_path=checkpoint_path)
        self.vae_config = vae_config
        self.temperature = temperature
        self.max_seq_len = max_seq_len

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

        import torch

        temp = self._adjust_temperature_for_error(self.temperature, error_context)

        # Decode the SAME sampled trajectory the RL path optimizes, under
        # no_grad. Using the identical decoder at inference (rather than the
        # separate CADGenerationPipeline decode) ensures the policy improvements
        # made by generate_for_training's REINFORCE updates show up when the
        # orchestrator calls generate().
        with torch.no_grad():
            token_ids, _log_probs, entropy_value, latent_vector, confidence = (
                self._decode_and_sample(temp)
            )

        return CommandSequenceProposal(
            token_ids=token_ids,
            command_dicts=[],
            source_prompt=prompt,
            conditioning_source=(
                conditioning.source_type if conditioning else "unconditional"
            ),
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
        target_dimensions: tuple[float, float, float] | None = None,
    ) -> CommandSequenceProposal:
        """Generate with gradients, computing log-probs on the sampled trajectory.

        Runs the shared ``_decode_and_sample`` with gradients live so the
        returned ``proposal.log_probs`` is a differentiable scalar usable in a
        REINFORCE loss. Each token is sampled once from the live forward pass and
        its log-prob evaluated on that same sample — an unbiased estimator.

        Args:
            prompt: User prompt (stored for tracing).
            conditioning: Optional conditioning embeddings.
            error_context: Optional error context from a failed attempt.
            target_dimensions: Optional ``(w, h, d)`` to condition generation on
                (shifts the latent via the trained dimension encoder).

        Returns:
            CommandSequenceProposal with ``log_probs`` and ``entropy`` set.
        """
        if self._model is None:
            self._init_model()

        temp = self._adjust_temperature_for_error(self.temperature, error_context)
        token_ids, total_log_prob, entropy_value, latent_vector, confidence = (
            self._decode_and_sample(temp, target_dimensions=target_dimensions)
        )

        return CommandSequenceProposal(
            token_ids=token_ids,
            command_dicts=[],
            source_prompt=prompt,
            conditioning_source=(
                conditioning.source_type if conditioning else "unconditional"
            ),
            confidence=confidence,
            generation_metadata=self._build_metadata("STEPVAE", temperature=temp),
            latent_vector=latent_vector,
            error_context=error_context,
            log_probs=total_log_prob,
            entropy=entropy_value,
        )

    def _decode_and_sample(
        self,
        temp: float,
        z: Any | None = None,
        target_dimensions: tuple[float, float, float] | None = None,
    ) -> tuple[list[int], Any, float, np.ndarray | None, float]:
        """Sample one command trajectory from a latent ``z`` (prior if None).

        The single decode path shared by ``generate`` (inference, under
        ``torch.no_grad``), ``generate_for_training`` (RL, with grad),
        ``generate_candidates``, and ``generate_from_error_context`` (which
        passes a perturbed ``z``). Each token is sampled once from the live
        decoder logits and its log-prob recorded on that same sample — an
        unbiased REINFORCE estimator when gradients are tracked. Unifying every
        path here is what makes RL improvements visible at inference time.

        Args:
            temp: Sampling temperature (already error-adjusted).
            z: Optional latent ``[1, latent_dim]`` tensor; sampled from the
                ``N(0, I)`` prior when None.

        Returns:
            ``(token_ids, total_log_prob, entropy_value, latent_vector,
            confidence)``. ``total_log_prob`` is a differentiable scalar tensor
            (or None if no tokens were sampled); under ``no_grad`` it carries no
            graph and inference callers ignore it.
        """
        import torch

        from ll_gen.generators.base import (
            BOS_TOKEN_ID,
            CMD_TOKEN_MAP,
            EOS_CMD_TOKEN_ID,
            EOS_TOKEN_ID,
        )

        device = torch.device(self.device)
        if z is None:
            z = torch.randn(1, self._model.latent_dim, device=device)
        # Dimension conditioning: shift the latent by a learned offset of the
        # requested (w, h, d). dim_encoder is zero-initialized, so this is a
        # no-op until the conditioner is trained.
        if target_dimensions is not None and hasattr(self._model, "dim_encoder"):
            dim_t = torch.tensor(
                [list(target_dimensions)], dtype=torch.float32, device=device
            )
            z = z + self._model.dim_encoder(dim_t)
        self._model.last_latent = z

        hidden = self._model.decode(z, seq_len=self.max_seq_len)
        command_logits = self._model.command_head(hidden)[0]  # [S, C]
        param_logits_2d = [head(hidden)[0] for head in self._model.param_heads]

        seq_len = min(command_logits.shape[0], self.max_seq_len)
        log_probs_accum: list[torch.Tensor] = []
        entropy_accum: list[torch.Tensor] = []
        token_ids: list[int] = [BOS_TOKEN_ID]

        for pos in range(seq_len):
            cmd_logits_pos = command_logits[pos] / max(temp, 1e-8)
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
                param_logits_2d,
                pos,
                cmd_type,
                temp,
                token_ids,
                log_probs_accum,
                entropy_accum,
            )

        if token_ids[-1] not in (EOS_TOKEN_ID, EOS_CMD_TOKEN_ID):
            token_ids.append(EOS_TOKEN_ID)

        total_log_prob: torch.Tensor | None = None
        entropy_value = 0.0
        if log_probs_accum:
            total_log_prob = torch.stack(log_probs_accum).sum()
            entropy_value = float(torch.stack(entropy_accum).mean().detach().cpu())

        latent_vector: np.ndarray | None = z.detach().cpu().numpy()
        confidence = self._compute_confidence(
            command_logits.detach(), [pl.detach() for pl in param_logits_2d]
        )
        return token_ids, total_log_prob, entropy_value, latent_vector, confidence

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

        import torch

        # Each candidate is an independent draw from the same decoder the RL
        # loop optimizes (under no_grad), so batch generation reflects training.
        proposals: list[CommandSequenceProposal] = []
        with torch.no_grad():
            for _ in range(num_candidates):
                token_ids, _log_probs, entropy_value, latent_vector, confidence = (
                    self._decode_and_sample(self.temperature)
                )
                proposals.append(
                    CommandSequenceProposal(
                        token_ids=token_ids,
                        command_dicts=[],
                        source_prompt=prompt,
                        conditioning_source=(
                            conditioning.source_type
                            if conditioning
                            else "unconditional"
                        ),
                        confidence=confidence,
                        generation_metadata=self._build_metadata(
                            "STEPVAE", temperature=self.temperature
                        ),
                        latent_vector=latent_vector,
                        entropy=entropy_value,
                    )
                )

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

        # Decode the perturbed latent through the shared sampler (the same
        # decoder the RL loop optimizes) instead of re-implementing decode.
        _log.info("Decoding perturbed latent vector for retry generation")

        token_ids: list[int] = []
        confidence = 0.0

        try:
            import torch as torch_mod

            latent_tensor = (
                torch_mod.from_numpy(perturbed_latent.astype(np.float32))
                .unsqueeze(0)
                .to(self.device)
            )

            with torch_mod.no_grad():
                token_ids, _log_probs, _entropy, _latent, confidence = (
                    self._decode_and_sample(self.temperature, z=latent_tensor)
                )
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
            from stepnet.config import STEPEncoderConfig
            from stepnet.vae import STEPVAE
        except ImportError as e:
            raise ImportError(
                f"stepnet (ll-stepnet) is required for NeuralVAEGenerator: {e}"
            ) from e

        _log.info("Initializing STEPVAE model")

        # STEPVAE requires a STEPEncoderConfig object plus scalar hyper-params.
        # ll_gen's optional vae_config (a stepnet VAEConfig or any object with
        # the matching attributes) carries encoder + VAE settings under
        # different field names, so map them explicitly rather than splatting.
        vc = self.vae_config
        if vc is None:
            _log.info("No VAE config provided; using defaults")
            encoder_config = STEPEncoderConfig()
        else:
            encoder_config = STEPEncoderConfig(
                vocab_size=getattr(vc, "encoder_vocab_size", 50000),
                token_embed_dim=getattr(vc, "encoder_embed_dim", 256),
                num_transformer_layers=getattr(vc, "encoder_layers", 6),
                dropout=getattr(vc, "dropout", 0.1),
            )

        vae_kwargs: dict[str, Any] = {"max_seq_len": self.max_seq_len}
        if vc is not None:
            for attr in (
                "latent_dim",
                "kl_weight",
                "num_command_types",
                "num_param_levels",
                "max_seq_len",
            ):
                if hasattr(vc, attr):
                    vae_kwargs[attr] = getattr(vc, attr)

        self._model = STEPVAE(encoder_config=encoder_config, **vae_kwargs)

        # Dimension conditioner: maps a target (w, h, d) to a latent-space
        # offset added to z before decode (M3 follow-up). Attached as a model
        # submodule so its parameters are optimized + checkpointed alongside the
        # VAE. Zero-initialized so it starts as a no-op — conditioning is learned
        # as a perturbation of the working unconditional model, not a random
        # shove off the learned latent manifold.
        import torch.nn as nn

        dim_encoder = nn.Linear(3, self._model.latent_dim)
        nn.init.zeros_(dim_encoder.weight)
        nn.init.zeros_(dim_encoder.bias)
        self._model.dim_encoder = dim_encoder

        # Load checkpoint if provided
        if self.checkpoint_path:
            self.load_checkpoint(self.checkpoint_path)
        else:
            self._model = self._model.to(self.device)

        self._model.eval()

        _log.info("STEPVAE model initialized and ready")
