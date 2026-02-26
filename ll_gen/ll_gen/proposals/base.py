"""Base proposal dataclass shared by all proposal types.

Every proposal carries:
- A unique ID for tracing through the pipeline.
- A confidence score from the neural generator.
- Attempt tracking for the retry loop.
- Source prompt and conditioning metadata.
- An optional list of alternative proposals generated in the same batch.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class BaseProposal:
    """Common fields shared by all neural proposal types.

    Subclasses (CodeProposal, CommandSequenceProposal, LatentProposal)
    add their domain-specific payloads on top of these shared fields.

    Attributes:
        proposal_id: Unique identifier for this proposal instance.
        confidence: Neural generator's self-assessed confidence in [0, 1].
            For LLM code generation this comes from explicit self-rating
            or calibrated token probability.  For VAE/diffusion it is
            derived from reconstruction loss or denoising confidence.
        attempt: Current retry attempt (1-indexed).
        max_attempts: Maximum retries before giving up.
        source_prompt: The original user prompt that triggered generation.
        conditioning_source: Description of the conditioning input type
            ("text", "image", "text+image", "unconditional", etc.).
        generation_metadata: Arbitrary key-value metadata from the
            generator (e.g. model name, latent vector norm, temperature).
        alternatives: Sibling proposals generated in the same batch.
            The orchestrator may try alternatives if this proposal fails.
        timestamp: UTC creation time.
        error_context: Structured error feedback from a prior failed
            attempt, used by the generator to correct on retry.
    """

    proposal_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    confidence: float = 0.0
    attempt: int = 1
    max_attempts: int = 3
    source_prompt: str = ""
    conditioning_source: Optional[str] = None
    generation_metadata: Dict[str, Any] = field(default_factory=dict)
    alternatives: List[Any] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    error_context: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def should_retry(self) -> bool:
        """Whether the retry budget allows another attempt.

        Returns:
            True if ``attempt < max_attempts``.
        """
        return self.attempt < self.max_attempts

    def next_attempt(self) -> int:
        """Return the next attempt number.

        Returns:
            ``attempt + 1``, clamped to ``max_attempts``.
        """
        return min(self.attempt + 1, self.max_attempts)

    def with_error_context(self, error: Dict[str, Any]) -> BaseProposal:
        """Create a shallow copy with updated error context and incremented attempt.

        This is used by the orchestrator when building a retry proposal:
        the error context from the failed disposal is attached so the
        generator can condition on it.

        Args:
            error: Structured error dict from DisposalResult.

        Returns:
            New BaseProposal (same type) with updated fields.
            Subclasses should override to preserve their own fields.
        """
        import copy

        new = copy.deepcopy(self)
        new.proposal_id = uuid.uuid4().hex
        new.attempt = self.next_attempt()
        new.error_context = error
        new.timestamp = datetime.now(timezone.utc).isoformat()
        return new

    @property
    def is_first_attempt(self) -> bool:
        """Whether this is the first generation attempt."""
        return self.attempt == 1

    @property
    def is_retry(self) -> bool:
        """Whether this proposal is a retry (attempt > 1)."""
        return self.attempt > 1

    def summary(self) -> Dict[str, Any]:
        """Return a compact summary dict for logging / serialization.

        Returns:
            Dict with key identification and status fields.
        """
        return {
            "proposal_id": self.proposal_id,
            "type": type(self).__name__,
            "confidence": self.confidence,
            "attempt": f"{self.attempt}/{self.max_attempts}",
            "source_prompt_len": len(self.source_prompt),
            "has_error_context": self.error_context is not None,
            "num_alternatives": len(self.alternatives),
        }
