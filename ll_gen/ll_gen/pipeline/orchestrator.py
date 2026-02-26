"""GenerationOrchestrator — end-to-end text/image → STEP pipeline.

Implements the full state machine from the hybrid architecture:

.. code-block:: text

    START → Encode Input → Generate Proposal → Execute → Validate
        ↓ valid? → Introspect → matches intent? → Export → SUCCESS
        ↓ invalid? → Repair → still broken? → Build Feedback
        ↓ retries left? → back to Generate Proposal
        ↓ no retries? → FAIL

The orchestrator coordinates between:
- ``GenerationRouter`` — decides Path A vs Path B
- ``CadQueryProposer`` / ``OpenSCADProposer`` — code generation
- ``DisposalEngine`` — execution, validation, repair, export
- ``VisualVerifier`` — optional VLM-based semantic checking
- ``ll_stepnet`` generators — VAE/diffusion/VQ-VAE (lazy import)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ll_gen.codegen.cadquery_proposer import CadQueryProposer
from ll_gen.codegen.openscad_proposer import OpenSCADProposer
from ll_gen.config import (
    CodegenConfig,
    ConditioningConfig,
    DisposalConfig,
    ExportConfig,
    FeedbackConfig,
    GeneratorConfig,
    GenerationRoute,
    LLGenConfig,
)
from ll_gen.disposal.engine import DisposalEngine
from ll_gen.feedback.feedback_builder import build_code_feedback, build_neural_feedback
from ll_gen.proposals.base import BaseProposal
from ll_gen.proposals.code_proposal import CodeProposal
from ll_gen.proposals.command_proposal import CommandSequenceProposal
from ll_gen.proposals.disposal_result import DisposalResult
from ll_gen.proposals.latent_proposal import LatentProposal
from ll_gen.routing.router import GenerationRouter, RoutingDecision

_log = logging.getLogger(__name__)


@dataclass
class GenerationHistory:
    """Tracks all attempts across the retry loop.

    Attributes:
        routing_decision: How the route was chosen.
        attempts: List of (proposal, disposal_result) pairs.
        total_time_ms: Wall-clock time for the entire generation.
        final_result: The best DisposalResult from all attempts.
    """

    routing_decision: Optional[RoutingDecision] = None
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    total_time_ms: float = 0.0
    final_result: Optional[DisposalResult] = None


class GenerationOrchestrator:
    """End-to-end generation orchestrator.

    Ties together routing, proposal generation, disposal, feedback,
    and retry into a single ``generate()`` call.

    Args:
        config: Top-level ll_gen configuration.

    Example::

        orchestrator = GenerationOrchestrator()
        result = orchestrator.generate(
            "A mounting bracket with 4 bolt holes, 80mm wide, 3mm thick"
        )
        if result.is_valid:
            print(f"STEP file: {result.step_path}")
    """

    def __init__(self, config: Optional[LLGenConfig] = None) -> None:
        self.config = config or LLGenConfig()

        self.router = GenerationRouter(self.config.routing)
        self.disposal_engine = DisposalEngine(
            disposal_config=self.config.disposal,
            export_config=self.config.export,
            feedback_config=self.config.feedback,
            output_dir=self.config.output_dir,
        )

        # Lazy-initialized proposers (created on first use)
        self._cadquery_proposer: Optional[CadQueryProposer] = None
        self._openscad_proposer: Optional[OpenSCADProposer] = None

        # Lazy-initialized neural generators (created on first use)
        self._vae_generator = None
        self._diffusion_generator = None
        self._vqvae_generator = None
        self._conditioner = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        image_path: Optional[Path] = None,
        force_route: Optional[GenerationRoute] = None,
        max_retries: Optional[int] = None,
        export: bool = True,
        render: bool = False,
    ) -> DisposalResult:
        """Generate CAD geometry from a text prompt.

        Full pipeline:

        1. **Route** — Analyze the prompt to decide Code (Path A)
           vs Neural (Path B).
        2. **Propose** — Generate a typed proposal via the selected
           path.
        3. **Dispose** — Execute, validate, repair through the
           disposal engine.
        4. **Feedback** — If invalid, build structured feedback and
           retry with error context.
        5. **Export** — Write STEP/STL for valid results.

        Args:
            prompt: Text description of the desired geometry.
            image_path: Optional image for conditioning.
            force_route: Override automatic routing.
            max_retries: Override default max retry count.
            export: Whether to export valid shapes.
            render: Whether to generate multi-view renders.

        Returns:
            ``DisposalResult`` from the best attempt.
        """
        start_time = time.monotonic()
        retries = max_retries if max_retries is not None else self.config.max_retries
        history = GenerationHistory()

        # --- Step 1: Route ---
        decision = self.router.route(
            prompt=prompt,
            has_image=image_path is not None,
            force_route=force_route,
        )
        history.routing_decision = decision
        _log.info(
            "Routed to %s (confidence=%.2f)",
            decision.route.value,
            decision.confidence,
        )

        # --- Step 2-4: Propose → Dispose → Retry loop ---
        best_result: Optional[DisposalResult] = None
        error_context: Optional[Dict[str, Any]] = None

        for attempt in range(1, retries + 1):
            _log.info(
                "Attempt %d/%d (route=%s)",
                attempt, retries, decision.route.value,
            )

            # Generate proposal
            try:
                proposal = self._propose(
                    route=decision.route,
                    prompt=prompt,
                    image_path=image_path,
                    error_context=error_context,
                    attempt=attempt,
                )
            except Exception as exc:
                _log.error("Proposal generation failed: %s", exc)
                history.attempts.append({
                    "attempt": attempt,
                    "error": str(exc),
                    "stage": "propose",
                })
                continue

            # Dispose
            result = self.disposal_engine.dispose(
                proposal=proposal,
                export=export and (attempt == retries),
                render=render and (attempt == retries),
            )

            history.attempts.append({
                "attempt": attempt,
                "proposal_id": proposal.proposal_id,
                "proposal_type": type(proposal).__name__,
                "is_valid": result.is_valid,
                "reward_signal": result.reward_signal,
                "error_category": (
                    result.error_category.value if result.error_category else None
                ),
                "execution_time_ms": result.execution_time_ms,
            })

            # Track best result (highest reward)
            if best_result is None or result.reward_signal > best_result.reward_signal:
                best_result = result

            # Success — stop retrying
            if result.is_valid:
                _log.info(
                    "Generation succeeded on attempt %d/%d",
                    attempt, retries,
                )
                break

            # Build feedback for next attempt
            if attempt < retries:
                error_context = self._build_feedback(
                    result, proposal, decision.route
                )
                _log.info(
                    "Building feedback for retry: %s",
                    result.error_category.value if result.error_category else "unknown",
                )

        # --- Step 5: Visual verification (optional) ---
        if (
            best_result is not None
            and best_result.is_valid
            and render
            and best_result.render_paths
        ):
            try:
                from ll_gen.pipeline.verification import VisualVerifier

                verifier = VisualVerifier()
                verification = verifier.verify(
                    best_result.render_paths, prompt
                )
                if not verification.matches_intent:
                    _log.warning(
                        "Visual verification failed: %s",
                        verification.issues,
                    )
            except Exception as exc:
                _log.debug("Visual verification skipped: %s", exc)

        # Finalize
        history.total_time_ms = (time.monotonic() - start_time) * 1000
        history.final_result = best_result

        if best_result is None:
            _log.error(
                "All %d attempts failed for prompt: %s",
                retries, prompt[:100],
            )
            best_result = DisposalResult(
                error_message=f"All {retries} generation attempts failed.",
                suggestion="Try simplifying the prompt or using a different route.",
            )

        _log.info(
            "Generation complete: valid=%s, attempts=%d, time=%.0fms",
            best_result.is_valid,
            len(history.attempts),
            history.total_time_ms,
        )

        return best_result

    def generate_batch(
        self,
        prompt: str,
        num_candidates: int = 3,
        image_path: Optional[Path] = None,
        force_route: Optional[GenerationRoute] = None,
    ) -> List[DisposalResult]:
        """Generate multiple candidate shapes and return all results.

        Unlike ``generate()`` which retries on failure, this method
        generates ``num_candidates`` independent proposals and returns
        all disposal results sorted by reward signal (best first).

        Args:
            prompt: Text description.
            num_candidates: Number of candidates to generate.
            image_path: Optional image conditioning.
            force_route: Override automatic routing.

        Returns:
            List of DisposalResults, sorted by reward (best first).
        """
        decision = self.router.route(
            prompt=prompt,
            has_image=image_path is not None,
            force_route=force_route,
        )

        results: List[DisposalResult] = []

        for i in range(num_candidates):
            try:
                proposal = self._propose(
                    route=decision.route,
                    prompt=prompt,
                    image_path=image_path,
                    attempt=1,
                )
                result = self.disposal_engine.dispose(proposal, export=True)
                results.append(result)
            except Exception as exc:
                _log.warning("Candidate %d failed: %s", i + 1, exc)

        # Sort by reward (best first)
        results.sort(key=lambda r: r.reward_signal, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Proposal generation
    # ------------------------------------------------------------------

    def _propose(
        self,
        route: GenerationRoute,
        prompt: str,
        image_path: Optional[Path] = None,
        error_context: Optional[Dict[str, Any]] = None,
        attempt: int = 1,
    ) -> BaseProposal:
        """Generate a proposal via the selected route.

        Args:
            route: Generation route to use.
            prompt: Text prompt.
            image_path: Optional image.
            error_context: Error context from previous failed attempt.
            attempt: Current attempt number.

        Returns:
            Typed proposal (CodeProposal, CommandSequenceProposal, etc.)

        Raises:
            RuntimeError: If the generation path is not available.
        """
        if route == GenerationRoute.CODE_CADQUERY:
            return self._propose_cadquery(prompt, image_path, error_context, attempt)

        elif route == GenerationRoute.CODE_OPENSCAD:
            return self._propose_openscad(prompt, image_path, error_context, attempt)

        elif route == GenerationRoute.NEURAL_VAE:
            return self._propose_neural_vae(prompt, error_context)

        elif route == GenerationRoute.NEURAL_DIFFUSION:
            return self._propose_neural_diffusion(prompt, error_context)

        elif route == GenerationRoute.NEURAL_VQVAE:
            return self._propose_neural_vqvae(prompt, error_context)

        else:
            raise RuntimeError(f"Unsupported generation route: {route}")

    def _propose_cadquery(
        self,
        prompt: str,
        image_path: Optional[Path],
        error_context: Optional[Dict[str, Any]],
        attempt: int,
    ) -> CodeProposal:
        """Generate a CadQuery CodeProposal."""
        if self._cadquery_proposer is None:
            self._cadquery_proposer = CadQueryProposer(self.config.codegen)
        return self._cadquery_proposer.propose(
            prompt=prompt,
            image_path=image_path,
            error_context=error_context,
            attempt=attempt,
        )

    def _propose_openscad(
        self,
        prompt: str,
        image_path: Optional[Path],
        error_context: Optional[Dict[str, Any]],
        attempt: int,
    ) -> CodeProposal:
        """Generate an OpenSCAD CodeProposal."""
        if self._openscad_proposer is None:
            self._openscad_proposer = OpenSCADProposer(self.config.codegen)
        return self._openscad_proposer.propose(
            prompt=prompt,
            image_path=image_path,
            error_context=error_context,
            attempt=attempt,
        )

    def _propose_neural_vae(
        self,
        prompt: str,
        error_context: Optional[Dict[str, Any]],
    ) -> CommandSequenceProposal:
        """Generate a CommandSequenceProposal via VAE sampling."""
        if self._vae_generator is None:
            try:
                from ll_gen.generators.neural_vae import NeuralVAEGenerator
            except ImportError as exc:
                raise RuntimeError(
                    "ll_gen.generators requires ll_stepnet. "
                    f"Import error: {exc}"
                ) from exc

            gen_config = getattr(self.config, 'generators', None)
            self._vae_generator = NeuralVAEGenerator(
                checkpoint_path=(
                    Path(gen_config.vae_checkpoint)
                    if gen_config and gen_config.vae_checkpoint
                    else None
                ),
                device=self.config.device,
                temperature=gen_config.default_temperature if gen_config else 0.8,
                max_seq_len=gen_config.max_seq_len if gen_config else 60,
            )

        conditioning = self._get_conditioning(prompt)
        return self._vae_generator.generate(
            prompt=prompt,
            conditioning=conditioning,
            error_context=error_context,
        )

    def _propose_neural_diffusion(
        self,
        prompt: str,
        error_context: Optional[Dict[str, Any]],
    ) -> LatentProposal:
        """Generate a LatentProposal via structured diffusion."""
        if self._diffusion_generator is None:
            try:
                from ll_gen.generators.neural_diffusion import NeuralDiffusionGenerator
            except ImportError as exc:
                raise RuntimeError(
                    "ll_gen.generators requires ll_stepnet. "
                    f"Import error: {exc}"
                ) from exc

            gen_config = getattr(self.config, 'generators', None)
            self._diffusion_generator = NeuralDiffusionGenerator(
                checkpoint_path=(
                    Path(gen_config.diffusion_checkpoint)
                    if gen_config and gen_config.diffusion_checkpoint
                    else None
                ),
                device=self.config.device,
                inference_steps=(
                    gen_config.diffusion_inference_steps if gen_config else 50
                ),
                eta=gen_config.diffusion_eta if gen_config else 0.0,
            )

        conditioning = self._get_conditioning(prompt)
        return self._diffusion_generator.generate(
            prompt=prompt,
            conditioning=conditioning,
            error_context=error_context,
        )

    def _propose_neural_vqvae(
        self,
        prompt: str,
        error_context: Optional[Dict[str, Any]],
    ) -> CommandSequenceProposal:
        """Generate a CommandSequenceProposal via VQ-VAE codebooks."""
        if self._vqvae_generator is None:
            try:
                from ll_gen.generators.neural_vqvae import NeuralVQVAEGenerator
            except ImportError as exc:
                raise RuntimeError(
                    "ll_gen.generators requires ll_stepnet. "
                    f"Import error: {exc}"
                ) from exc

            gen_config = getattr(self.config, 'generators', None)
            self._vqvae_generator = NeuralVQVAEGenerator(
                checkpoint_path=(
                    Path(gen_config.vqvae_checkpoint)
                    if gen_config and gen_config.vqvae_checkpoint
                    else None
                ),
                device=self.config.device,
                temperature=gen_config.default_temperature if gen_config else 0.7,
                codebook_dim=(
                    gen_config.vqvae_codebook_dim if gen_config else 512
                ),
            )

        conditioning = self._get_conditioning(prompt)
        return self._vqvae_generator.generate(
            prompt=prompt,
            conditioning=conditioning,
            error_context=error_context,
        )

    def _get_conditioning(self, prompt: str, image_path: Optional[Path] = None):
        """Get conditioning embeddings for the prompt.

        Lazy-initializes the MultiModalConditioner on first use.
        Returns None if conditioning is not available.

        Args:
            prompt: Text prompt.
            image_path: Optional image path.

        Returns:
            ConditioningEmbeddings or None.
        """
        try:
            if self._conditioner is None:
                from ll_gen.conditioning.multimodal import MultiModalConditioner

                cond_config = getattr(self.config, 'conditioning', None)
                self._conditioner = MultiModalConditioner(
                    text_model=(
                        cond_config.text_model if cond_config else "bert-base-uncased"
                    ),
                    image_model=(
                        cond_config.image_model if cond_config else "dino_vits16"
                    ),
                    conditioning_dim=(
                        cond_config.conditioning_dim if cond_config else 768
                    ),
                    fusion_method=(
                        cond_config.fusion_method if cond_config else "concat"
                    ),
                    device=self.config.device,
                )
            return self._conditioner.encode(prompt, image_path)
        except Exception as exc:
            _log.debug("Conditioning unavailable: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Feedback construction
    # ------------------------------------------------------------------

    def _build_feedback(
        self,
        result: DisposalResult,
        proposal: BaseProposal,
        route: GenerationRoute,
    ) -> Dict[str, Any]:
        """Build structured feedback for the next retry attempt.

        For code proposals: builds an LLM-readable error message.
        For neural proposals: builds a structured error dict.
        """
        if isinstance(proposal, CodeProposal):
            feedback_text = build_code_feedback(result, proposal)
            return {
                "type": "code_feedback",
                "error_message": feedback_text,
                "original_code": proposal.code,
                "error_category": (
                    result.error_category.value if result.error_category else None
                ),
            }
        else:
            return build_neural_feedback(result)
