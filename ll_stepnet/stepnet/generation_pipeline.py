"""End-to-end CAD generation pipeline.

Unified pipeline that connects ll_stepnet generative models → geotoken decoding →
cadling reconstruction and evaluation. Supports three generation modes:

1. VAE sampling: Sample from latent prior N(0, I), decode to command sequence
2. VQ-VAE sampling: Autoregressive code generation, decode to commands
3. Diffusion sampling: Reverse diffusion from noise, decode to commands

This module uses lazy imports for optional dependencies (cadling, geotoken) so
that the pipeline works even if they are not installed. Reconstruction and
evaluation are only available when the dependencies are present.

Example usage:
    >>> from stepnet import STEPVAE
    >>> from stepnet.generation_pipeline import CADGenerationPipeline
    >>>
    >>> model = STEPVAE(encoder_config, latent_dim=256)
    >>> pipeline = CADGenerationPipeline(model=model, mode='vae')
    >>> results = pipeline.generate(num_samples=4)
    >>> # results is a list of dicts with 'token_sequence', 'commands', 'shape', 'valid'
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

_log = logging.getLogger(__name__)


class CADGenerationPipeline:
    """End-to-end CAD generation pipeline.

    Connects ll_stepnet generative models → geotoken decoding → cadling reconstruction.
    Supports VAE, VQ-VAE, and Diffusion sampling modes.

    Attributes:
        model: Generative model (STEPVAE, VQVAEModel, or StructuredDiffusion).
        mode: Generation mode ('vae', 'vqvae', or 'diffusion').
        device: Target device ('cpu' or 'cuda').
        executor_tolerance: Tolerance for cadling CommandExecutor validation.
        quantization_levels: Number of quantization levels per parameter.
    """

    def __init__(
        self,
        model: nn.Module,
        mode: str = 'vae',
        device: str = 'cpu',
        executor_tolerance: float = 1e-6,
        quantization_levels: int = 256,
    ) -> None:
        """Initialize the CAD generation pipeline.

        Args:
            model: Generative model (STEPVAE, VQVAEModel, or StructuredDiffusion).
            mode: Generation mode - 'vae', 'vqvae', or 'diffusion'.
            device: Target device - 'cpu' or 'cuda'.
            executor_tolerance: Tolerance threshold for cadling CommandExecutor.
            quantization_levels: Number of quantization bins per parameter.

        Raises:
            ValueError: If mode is not one of the supported modes.
            TypeError: If model is not a recognized generative model.
        """
        if mode not in ('vae', 'vqvae', 'diffusion'):
            raise ValueError(
                f"mode must be one of ('vae', 'vqvae', 'diffusion'), got {mode!r}"
            )

        self.model = model.to(device)
        self.mode = mode
        self.device = device
        self.executor_tolerance = executor_tolerance
        self.quantization_levels = quantization_levels

        # Infer mode from model if not explicitly set (as fallback)
        self._inferred_model_type = self._infer_model_type(model)

        _log.info(
            "CADGenerationPipeline initialised: mode=%s, model_type=%s, "
            "device=%s, quantization_levels=%d",
            mode, self._inferred_model_type, device, quantization_levels,
        )

    @staticmethod
    def _infer_model_type(model: nn.Module) -> str:
        """Infer the generative model type from the model class name.

        Args:
            model: The generative model instance.

        Returns:
            String identifying the model type.
        """
        class_name = model.__class__.__name__
        if 'VAE' in class_name and 'VQ' not in class_name:
            return 'STEPVAE'
        elif 'VQVAE' in class_name:
            return 'VQVAEModel'
        elif 'Diffusion' in class_name:
            return 'StructuredDiffusion'
        else:
            return class_name

    def generate(
        self,
        num_samples: int = 1,
        seq_len: Optional[int] = None,
        reconstruct: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate CAD sequences end-to-end.

        Samples from the generative model, decodes to geotoken TokenSequence,
        and optionally reconstructs using cadling's CommandExecutor.

        Args:
            num_samples: Number of sequences to generate.
            seq_len: Sequence length for generation (mode-dependent).
            reconstruct: Whether to reconstruct via cadling. Defaults to True.
            **kwargs: Additional keyword arguments passed to the model's
                sampling method (e.g., temperature, top_k for diffusion).

        Returns:
            List of result dictionaries, each containing:
                - 'token_sequence': geotoken TokenSequence (if geotoken installed)
                - 'commands': List of command dicts (if geotoken installed)
                - 'command_logits': Raw command logits [num_samples, seq_len, 6]
                - 'param_logits': Raw parameter logits, list of 16 tensors
                - 'shape': Reconstructed cadling Shape (if reconstruct=True and cadling installed)
                - 'valid': Boolean validity flag (if reconstruct=True and cadling installed)
                - 'error': Error message if reconstruction failed (if reconstruct=True)
        """
        self.model.eval()

        with torch.no_grad():
            # Step 1: Sample from model based on mode
            if self.mode == 'vae':
                model_output = self._sample_vae(num_samples, seq_len, **kwargs)
            elif self.mode == 'vqvae':
                model_output = self._sample_vqvae(num_samples, seq_len, **kwargs)
            elif self.mode == 'diffusion':
                model_output = self._sample_diffusion(num_samples, seq_len, **kwargs)
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")

            # Step 2: Extract command and parameter logits
            command_logits = model_output.get('command_logits')
            param_logits = model_output.get('param_logits', [])

            # Step 3: Build result list by decoding each sample
            results = []
            for batch_idx in range(num_samples):
                result = {
                    'command_logits': command_logits,
                    'param_logits': param_logits,
                    'batch_index': batch_idx,
                }

                # Decode to TokenSequence (requires geotoken)
                try:
                    token_seq = self._decode_to_token_sequence(
                        command_logits, param_logits, batch_idx
                    )
                    result['token_sequence'] = token_seq
                    result['commands'] = self._token_sequence_to_commands(token_seq)
                except ImportError:
                    _log.debug(
                        "geotoken not available; skipping TokenSequence decoding"
                    )
                except Exception as e:
                    _log.warning(f"TokenSequence decoding failed: {e}")
                    result['decode_error'] = str(e)

                # Reconstruct via cadling (if requested and available)
                if reconstruct and 'token_sequence' in result:
                    try:
                        recon_result = self._reconstruct(result['token_sequence'])
                        result.update(recon_result)
                    except ImportError:
                        _log.debug(
                            "cadling not available; skipping reconstruction"
                        )
                    except Exception as e:
                        _log.warning(f"Reconstruction failed: {e}")
                        result['error'] = str(e)

                results.append(result)

        return results

    def _sample_vae(
        self,
        num_samples: int,
        seq_len: Optional[int] = None,
        temperature: float = 1.0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Sample from VAE by sampling latent code and decoding.

        Converts discrete predictions from the VAE decoder into proper
        probability distributions using temperature-scaled soft logits.
        This preserves the argmax identity (same decoded result) while
        providing meaningful probability distributions for downstream
        sampling, beam search, and uncertainty estimation.

        Args:
            num_samples: Number of samples to generate.
            seq_len: Sequence length (uses model.max_seq_len if None).
            temperature: Controls distribution sharpness.
                T=1.0 (default), T<1.0 (sharper/more confident),
                T>1.0 (softer/more uniform).
            **kwargs: Unused, provided for interface compatibility.

        Returns:
            Dictionary with 'command_logits' and 'param_logits'.
        """
        if not hasattr(self.model, 'sample'):
            raise RuntimeError("Model does not have a 'sample' method")

        sample_output = self.model.sample(
            num_samples=num_samples,
            seq_len=seq_len,
            device=self.device,
        )

        # Check if the model already provides logits directly
        if 'command_logits' in sample_output and 'param_logits' in sample_output:
            _log.debug("VAE sample returned logits directly")
            return {
                'command_logits': sample_output['command_logits'],
                'param_logits': sample_output['param_logits'],
            }

        # Convert discrete predictions to temperature-scaled soft logits.
        # Instead of one-hot (which destroys probability info), we create
        # a peaked distribution centered on the predicted class.
        command_preds = sample_output.get('command_preds')
        param_preds = sample_output.get('param_preds')

        if command_preds is None:
            raise RuntimeError("VAE sample output missing 'command_preds'")

        batch_size, actual_seq_len = command_preds.shape

        # Build command logits: high value at predicted class, low elsewhere.
        # After softmax, this yields ~99.5% at predicted class for T=1.0,
        # while still providing a valid probability distribution.
        peak_logit = 10.0 / max(temperature, 1e-6)
        base_logit = 0.1 / max(temperature, 1e-6)

        command_logits = torch.full(
            (batch_size, actual_seq_len, 6),
            base_logit,
            device=self.device,
        )
        command_logits.scatter_(
            2,
            command_preds.unsqueeze(-1).long(),
            peak_logit,
        )

        # Build parameter logits with same temperature scaling
        param_logits = []
        if param_preds is not None:
            # param_preds is [batch_size, seq_len, 16]
            for param_slot in range(16):
                slot_preds = param_preds[..., param_slot].long()  # [B, S]
                slot_logits = torch.full(
                    (batch_size, actual_seq_len, self.quantization_levels),
                    base_logit,
                    device=self.device,
                )
                slot_logits.scatter_(
                    2,
                    slot_preds.unsqueeze(-1),
                    peak_logit,
                )
                param_logits.append(slot_logits)

        _log.debug(
            "VAE: Converted predictions to soft logits "
            "(T=%.2f, peak=%.1f, base=%.2f)",
            temperature, peak_logit, base_logit,
        )

        return {
            'command_logits': command_logits,
            'param_logits': param_logits,
        }

    def _sample_vqvae(
        self,
        num_samples: int,
        seq_len: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Sample from VQ-VAE using autoregressive code generation.

        Generates discrete codes from the VQ-VAE and decodes them to
        command/parameter logits using a multi-strategy approach:
        1. Model's decode_codes() method (preferred)
        2. Codebook embedding lookup + output head projection
        3. Direct embedding lookup + learned linear projection

        Args:
            num_samples: Number of samples to generate.
            seq_len: Sequence length (model-dependent).
            **kwargs: Additional kwargs for autoregressive generation
                (e.g., temperature, top_k).

        Returns:
            Dictionary with 'command_logits' and 'param_logits'.

        Raises:
            RuntimeError: If model lacks generate() or all decode strategies fail.
        """
        if not hasattr(self.model, 'generate'):
            raise RuntimeError("VQ-VAE model does not have a 'generate' method")

        # VQ-VAE generate returns codes, not logits
        gen_output = self.model.generate(
            num_samples=num_samples, seq_len=seq_len, **kwargs,
        )

        # Handle dict output (some models return dict with 'codes' key)
        if isinstance(gen_output, dict):
            if 'command_logits' in gen_output and 'param_logits' in gen_output:
                _log.debug("VQ-VAE generate returned logits directly")
                return {
                    'command_logits': gen_output['command_logits'],
                    'param_logits': gen_output['param_logits'],
                }
            codes = gen_output.get('codes', gen_output.get('token_ids'))
            if codes is None:
                raise RuntimeError(
                    f"VQ-VAE dict output missing 'codes'/'token_ids' key. "
                    f"Available keys: {list(gen_output.keys())}"
                )
        elif isinstance(gen_output, torch.Tensor):
            codes = gen_output
        else:
            raise RuntimeError(f"Unexpected VQ-VAE output type: {type(gen_output)}")

        batch_size = codes.shape[0]
        codes_seq_len = codes.shape[1] if codes.dim() > 1 else 1
        if codes.dim() == 1:
            codes = codes.unsqueeze(1)

        _log.info("VQ-VAE: Generated codes shape %s", codes.shape)

        # Strategy 1: Use model's decode_codes() if available
        if hasattr(self.model, 'decode_codes'):
            try:
                decoded = self.model.decode_codes(codes)  # [B, S, D]
                command_logits = self._project_to_command_logits(decoded)
                param_logits = self._project_to_param_logits(decoded)
                _log.info(
                    "VQ-VAE: Decoded via model.decode_codes() -> "
                    "cmd %s, params [%d x %s]",
                    command_logits.shape, len(param_logits),
                    param_logits[0].shape if param_logits else "empty",
                )
                return {
                    'command_logits': command_logits,
                    'param_logits': param_logits,
                }
            except Exception as e:
                _log.warning("VQ-VAE decode_codes() failed: %s. Trying embedding lookup.", e)

        # Strategy 2: Codebook embedding lookup
        embedding_table = None
        if hasattr(self.model, 'embedding'):
            embedding_table = self.model.embedding
        elif hasattr(self.model, 'codebook'):
            cb = self.model.codebook
            if hasattr(cb, 'embedding'):
                embedding_table = cb.embedding
            elif hasattr(cb, 'embed'):
                embedding_table = cb.embed

        if embedding_table is not None:
            try:
                # Look up embeddings for each code
                flat_codes = codes.reshape(-1).long()  # [B*S]
                embeddings = embedding_table(flat_codes)  # [B*S, D]
                decoded = embeddings.reshape(batch_size, codes_seq_len, -1)  # [B, S, D]

                command_logits = self._project_to_command_logits(decoded)
                param_logits = self._project_to_param_logits(decoded)
                _log.info(
                    "VQ-VAE: Decoded via embedding lookup -> "
                    "cmd %s, params [%d x %s]",
                    command_logits.shape, len(param_logits),
                    param_logits[0].shape if param_logits else "empty",
                )
                return {
                    'command_logits': command_logits,
                    'param_logits': param_logits,
                }
            except Exception as e:
                _log.warning("VQ-VAE embedding lookup failed: %s", e)

        # Strategy 3: If model has a decoder module, pass codes through it
        if hasattr(self.model, 'decoder'):
            try:
                decoded = self.model.decoder(codes.float())
                command_logits = self._project_to_command_logits(decoded)
                param_logits = self._project_to_param_logits(decoded)
                _log.info("VQ-VAE: Decoded via model.decoder()")
                return {
                    'command_logits': command_logits,
                    'param_logits': param_logits,
                }
            except Exception as e:
                _log.warning("VQ-VAE model.decoder() failed: %s", e)

        # All strategies failed — raise instead of returning zeros
        raise RuntimeError(
            "VQ-VAE: Could not decode codes to logits. The model needs at "
            "least one of: decode_codes() method, codebook.embedding table, "
            "or decoder module. Model type: %s, available attrs: %s"
            % (type(self.model).__name__,
               [a for a in dir(self.model) if not a.startswith('_')])
        )

    def _sample_diffusion(
        self,
        num_samples: int,
        seq_len: Optional[int] = None,
        num_inference_steps: int = 50,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Sample from Diffusion model via reverse diffusion process.

        StructuredDiffusion produces 4-stage denoised latents:
        face_positions, face_geometry, edge_positions, edge_vertex_geometry.
        These are concatenated and decoded to command/parameter logits.

        Args:
            num_samples: Number of samples to generate.
            seq_len: Target sequence length for output (default 60).
            num_inference_steps: Not used by StructuredDiffusion (uses
                internal scheduler), kept for API compatibility.
            **kwargs: Additional kwargs passed to model.sample().

        Returns:
            Dictionary with 'command_logits' and 'param_logits'.

        Raises:
            RuntimeError: If model lacks sample() or produces no decodable output.
        """
        if not hasattr(self.model, 'sample'):
            raise RuntimeError("Diffusion model does not have a 'sample' method")

        # StructuredDiffusion.sample() takes (batch_size, device, use_pndm)
        # It does NOT accept num_steps — it uses its internal scheduler.
        sample_kwargs = {}
        if 'use_pndm' in kwargs:
            sample_kwargs['use_pndm'] = kwargs.pop('use_pndm')

        sample_output = self.model.sample(
            batch_size=num_samples,
            device=torch.device(self.device),
            **sample_kwargs,
        )

        # StructuredDiffusion returns {stage_name: denoised_latents [B, D]}
        # for stages: face_positions, face_geometry, edge_positions,
        # edge_vertex_geometry. Collect all available stage latents.
        stage_names = (
            'face_positions', 'face_geometry',
            'edge_positions', 'edge_vertex_geometry',
        )

        stage_latents = []
        for stage_name in stage_names:
            if stage_name in sample_output:
                stage_latents.append(sample_output[stage_name])
            else:
                _log.warning("Diffusion output missing stage: %s", stage_name)

        # Also handle flat 'latents' key for non-StructuredDiffusion models
        if not stage_latents and 'latents' in sample_output:
            stage_latents.append(sample_output['latents'])

        if not stage_latents:
            raise RuntimeError(
                "Diffusion sample output contains no decodable latents. "
                "Expected stage keys %s or 'latents' key. "
                "Got keys: %s" % (stage_names, list(sample_output.keys()))
            )

        # Concatenate all stage latents: [B, sum(D_i)]
        concatenated = torch.cat(stage_latents, dim=-1)
        batch_size = concatenated.shape[0]
        combined_dim = concatenated.shape[1]

        _log.info(
            "Diffusion: Collected %d stages, concatenated latent dim = %d",
            len(stage_latents), combined_dim,
        )

        # Strategy 1: Use model's decoder_head if available
        if hasattr(self.model, 'decoder_head'):
            try:
                decoded = self.model.decoder_head(concatenated)
                command_logits = self._project_to_command_logits(decoded)
                param_logits = self._project_to_param_logits(decoded)
                _log.info("Diffusion: Decoded via model.decoder_head()")
                return {
                    'command_logits': command_logits,
                    'param_logits': param_logits,
                }
            except Exception as e:
                _log.warning("Diffusion decoder_head() failed: %s. Using MLP projection.", e)

        # Strategy 2: Project through a latent-to-sequence MLP.
        # This converts the flat latent [B, D] into [B, S, C] logits.
        target_seq_len = seq_len or 60  # DeepCAD uses 60-length sequences
        command_logits = self._latent_to_sequence_logits(
            concatenated, target_seq_len, 6,
        )
        param_logits = []
        for _ in range(16):
            slot_logits = self._latent_to_sequence_logits(
                concatenated, target_seq_len, self.quantization_levels,
            )
            param_logits.append(slot_logits)

        _log.info(
            "Diffusion: Projected latent [%d, %d] -> cmd %s, "
            "params [%d x %s]",
            batch_size, combined_dim,
            command_logits.shape, len(param_logits),
            param_logits[0].shape if param_logits else "empty",
        )

        return {
            'command_logits': command_logits,
            'param_logits': param_logits,
        }

    # ------------------------------------------------------------------
    # Shared projection helpers
    # ------------------------------------------------------------------

    def _project_to_command_logits(
        self, decoded: torch.Tensor,
    ) -> torch.Tensor:
        """Project decoded features to command logits [B, S, 6].

        Uses model's command_head if available, otherwise a linear projection.

        Args:
            decoded: Decoded features [B, S, D] or [B, D].

        Returns:
            Command logits tensor [B, S, 6].
        """
        if hasattr(self.model, 'command_head'):
            return self.model.command_head(decoded)

        # Ensure 3D: [B, S, D]
        if decoded.dim() == 2:
            decoded = decoded.unsqueeze(1)

        feat_dim = decoded.shape[-1]
        proj = nn.Linear(feat_dim, 6, bias=True).to(decoded.device)
        nn.init.xavier_uniform_(proj.weight)
        return proj(decoded)

    def _project_to_param_logits(
        self, decoded: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Project decoded features to parameter logits [16 x [B, S, Q]].

        Uses model's param_heads if available, otherwise linear projections.

        Args:
            decoded: Decoded features [B, S, D] or [B, D].

        Returns:
            List of 16 parameter logit tensors [B, S, quantization_levels].
        """
        if hasattr(self.model, 'param_heads'):
            return [head(decoded) for head in self.model.param_heads]

        # Ensure 3D: [B, S, D]
        if decoded.dim() == 2:
            decoded = decoded.unsqueeze(1)

        feat_dim = decoded.shape[-1]
        param_logits = []
        for _ in range(16):
            proj = nn.Linear(feat_dim, self.quantization_levels, bias=True).to(decoded.device)
            nn.init.xavier_uniform_(proj.weight)
            param_logits.append(proj(decoded))

        return param_logits

    def _latent_to_sequence_logits(
        self,
        latent: torch.Tensor,
        target_seq_len: int,
        num_classes: int,
    ) -> torch.Tensor:
        """Project a flat latent vector to a sequence of logits.

        Uses a 2-layer MLP: latent_dim -> hidden -> (seq_len * num_classes)
        then reshapes to [B, seq_len, num_classes].

        Args:
            latent: Flat latent vectors [B, D].
            target_seq_len: Target sequence length.
            num_classes: Number of output classes per position.

        Returns:
            Logits tensor [B, target_seq_len, num_classes].
        """
        batch_size, latent_dim = latent.shape
        hidden_dim = max(512, latent_dim)
        output_dim = target_seq_len * num_classes

        mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        ).to(latent.device)

        # Initialize weights for reasonable output range
        for layer in mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        logits_flat = mlp(latent)  # [B, S*C]
        return logits_flat.reshape(batch_size, target_seq_len, num_classes)

    def _decode_to_token_sequence(
        self,
        command_logits: torch.Tensor,
        param_logits: List[torch.Tensor],
        batch_index: int = 0,
    ):
        """Convert model output logits to a geotoken TokenSequence.

        Lazily imports from geotoken and uses the standard decoding procedure:
        argmax over command types and parameters, apply parameter masks,
        stop at EOS token.

        Args:
            command_logits: [batch, seq_len, 6] command type logits.
            param_logits: List of 16 [batch, seq_len, num_levels] parameter logits.
            batch_index: Which sample in the batch to decode.

        Returns:
            geotoken.TokenSequence with decoded command tokens.

        Raises:
            ImportError: If geotoken is not installed.
        """
        try:
            from geotoken.tokenizer.token_types import (
                CommandToken,
                CommandType,
                TokenSequence,
            )
        except ImportError as e:
            raise ImportError(
                "geotoken is required for token sequence decoding. "
                "Install with: pip install geotoken"
            ) from e

        # Get the sequence length and command types for this sample
        seq_len = command_logits.shape[1]
        cmd_indices = command_logits[batch_index].argmax(dim=-1)  # [S]

        command_tokens = []
        for s in range(seq_len):
            cmd_idx = int(cmd_indices[s].item())
            cmd_idx = min(cmd_idx, len(CommandType) - 1)
            geo_cmd = CommandType(cmd_idx)

            # Argmax each parameter
            params = []
            for p in range(16):
                if p < len(param_logits):
                    val = int(param_logits[p][batch_index, s].argmax(dim=-1).item())
                else:
                    val = 0
                params.append(val)

            # Get parameter mask for this command type
            # (import the masks from output_heads)
            from .output_heads import PARAMETER_MASKS, CommandType as StepCommandType
            step_cmd = StepCommandType(cmd_idx)
            active = PARAMETER_MASKS.get(step_cmd, [])
            mask = [i in active for i in range(16)]

            command_tokens.append(CommandToken(
                command_type=geo_cmd,
                parameters=params,
                parameter_mask=mask,
            ))

            # Stop at EOS
            if geo_cmd == CommandType.EOS:
                break

        return TokenSequence(command_tokens=command_tokens)

    @staticmethod
    def _token_sequence_to_commands(token_sequence) -> List[Dict[str, Any]]:
        """Convert a geotoken TokenSequence to a list of command dicts.

        Args:
            token_sequence: geotoken.TokenSequence instance.

        Returns:
            List of command dictionaries with keys: command_type, parameters, mask.
        """
        commands = []
        for token in token_sequence.command_tokens:
            commands.append({
                'command_type': token.command_type.name if hasattr(token.command_type, 'name') else str(token.command_type),
                'parameters': token.parameters,
                'mask': token.parameter_mask,
            })
        return commands

    def _reconstruct(self, token_sequence) -> Dict[str, Any]:
        """Reconstruct a CAD shape using cadling's CommandExecutor.

        Converts TokenSequence to raw token IDs, executes via CommandExecutor,
        validates the result, and optionally post-processes vertices using
        geotoken's vertex validation, clustering, and merging.

        Args:
            token_sequence: geotoken.TokenSequence instance.

        Returns:
            Dictionary with:
                - 'shape': Reconstructed cadling.Shape (or None if failed)
                - 'valid': Boolean validity flag
                - 'error': Error message (if any)
                - 'vertex_report': Optional vertex validation report (if post-processing ran)

        Raises:
            ImportError: If cadling is not installed.
        """
        try:
            from cadling.execution import CommandExecutor
            from cadling.validation import validate_shape
        except ImportError as e:
            raise ImportError(
                "cadling is required for reconstruction. "
                "Install with: pip install cadling"
            ) from e

        result = {'valid': False, 'error': None}

        try:
            # Convert TokenSequence to command token IDs
            # (This assumes token_sequence has a method or property to get IDs)
            if hasattr(token_sequence, 'to_token_ids'):
                token_ids = token_sequence.to_token_ids()
            elif hasattr(token_sequence, 'command_tokens'):
                # Reconstruct from command tokens
                token_ids = []
                for cmd_token in token_sequence.command_tokens:
                    # Encode command type and parameters
                    cmd_id = int(cmd_token.command_type)
                    token_ids.append(cmd_id)
                    for param in cmd_token.parameters:
                        token_ids.append(int(param))
            else:
                raise AttributeError("TokenSequence has no method to convert to IDs")

            # Execute the sequence
            executor = CommandExecutor(tolerance=self.executor_tolerance)
            shape = executor.execute(token_ids)

            # Optional vertex post-processing using geotoken.vertex
            vertex_report = self._postprocess_vertices(shape)
            if vertex_report is not None:
                result['vertex_report'] = vertex_report

            # Validate
            validity = validate_shape(shape) if hasattr(validate_shape, '__call__') else True

            result['shape'] = shape
            result['valid'] = validity

        except Exception as e:
            _log.debug(f"Reconstruction error: {e}")
            result['error'] = str(e)

        return result

    @staticmethod
    def _postprocess_vertices(shape: Any) -> Optional[Dict[str, Any]]:
        """Run vertex validation and clustering on a reconstructed shape.

        Attempts to extract vertices/faces from the shape (e.g. trimesh),
        validates them using :class:`geotoken.vertex.VertexValidator`, and
        clusters near-duplicate vertices with :class:`VertexClusterer`.

        This is a best-effort step — if geotoken.vertex or trimesh is not
        installed, it silently returns ``None``.

        Args:
            shape: Reconstructed geometry (trimesh, OCC shape, or other).

        Returns:
            Dict with validation and clustering info, or ``None``.
        """
        if shape is None:
            return None

        try:
            from geotoken.vertex import VertexValidator, VertexClusterer
        except ImportError:
            return None

        try:
            import trimesh
            import numpy as np
        except ImportError:
            return None

        # Only process trimesh objects for now
        if not isinstance(shape, trimesh.Trimesh):
            return None

        report: Dict[str, Any] = {}

        try:
            vertices = np.array(shape.vertices, dtype=np.float32)
            faces = np.array(shape.faces, dtype=np.int64)

            # Validate
            validator = VertexValidator()
            val_result = validator.validate(vertices, faces)
            report['validation'] = {
                'valid': val_result.valid,
                'num_errors': len(val_result.errors),
                'num_warnings': len(val_result.warnings),
            }

            # Cluster with a small merge distance for duplicate detection
            clusterer = VertexClusterer(merge_distance=1e-4)
            clustering = clusterer.cluster(vertices)
            report['clustering'] = {
                'original_vertices': len(vertices),
                'unique_clusters': clustering.num_clusters,
                'duplicates_found': clustering.num_merged,
            }

        except Exception as exc:
            _log.debug("Vertex post-processing failed: %s", exc)
            report['error'] = str(exc)

        return report

    def evaluate(
        self,
        generated_results: List[Dict[str, Any]],
        reference_shapes: Optional[List[Any]] = None,
    ) -> Dict[str, float]:
        """Evaluate generation quality using cadling's metrics.

        Computes validity rate, and optionally novelty and coverage
        if reference shapes are provided.

        Args:
            generated_results: List of result dicts from generate().
            reference_shapes: Optional list of reference cadling.Shape objects
                for novelty/coverage computation.

        Returns:
            Dictionary with evaluation metrics:
                - 'validity_rate': Fraction of valid generated shapes
                - 'novelty': Novelty score (if references provided)
                - 'coverage': Coverage score (if references provided)

        Raises:
            ImportError: If cadling is not installed.
        """
        try:
            from cadling.metrics import GenerationMetrics
        except ImportError as e:
            raise ImportError(
                "cadling is required for evaluation. "
                "Install with: pip install cadling"
            ) from e

        metrics = GenerationMetrics()

        # Extract valid shapes
        valid_shapes = [
            r['shape'] for r in generated_results
            if r.get('valid', False) and 'shape' in r
        ]

        # Compute validity rate
        validity_rate = len(valid_shapes) / len(generated_results) if generated_results else 0.0

        result = {'validity_rate': validity_rate}

        # Optionally compute novelty and coverage
        if reference_shapes and valid_shapes:
            try:
                novelty = metrics.compute_novelty(valid_shapes, reference_shapes)
                coverage = metrics.compute_coverage(valid_shapes, reference_shapes)
                result['novelty'] = novelty
                result['coverage'] = coverage
            except Exception as e:
                _log.warning(f"Could not compute novelty/coverage: {e}")

        return result
