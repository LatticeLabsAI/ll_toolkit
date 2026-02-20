"""Generation path router — decides Code (Path A) vs Neural (Path B).

The router uses a multi-signal scoring system:

1. **Keyword matching** — scans the prompt for mechanical vs freeform
   vocabulary and scores each route proportionally.
2. **Image presence** — if an image is provided, biases toward neural
   generation (diffusion or VLM-conditioned code gen).
3. **Dimensional cues** — if the prompt contains explicit dimensions
   (e.g. "80mm wide, 3mm thick"), biases toward code generation
   (these are easy to parameterize in CadQuery).
4. **Explicit overrides** — the user can force a specific route.

The router returns a ``RoutingDecision`` containing the selected
route, confidence score, and an explanation of why that route was
chosen.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ll_gen.config import GenerationRoute, RoutingConfig

_log = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Result of the routing analysis.

    Attributes:
        route: Selected generation route.
        confidence: Confidence in [0, 1] that this is the right route.
        scores: Per-route scores from the analysis.
        reasons: Human-readable reasons for the decision.
        forced: Whether the route was user-forced (override).
    """

    route: GenerationRoute = GenerationRoute.CODE_CADQUERY
    confidence: float = 0.5
    scores: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    forced: bool = False


class GenerationRouter:
    """Automatic generation path router.

    Analyzes the user prompt and optional context to decide between
    code generation (Path A) and neural generation (Path B).

    Args:
        config: Routing configuration with keyword lists and thresholds.

    Example::

        router = GenerationRouter()
        decision = router.route("A mounting bracket with 4 bolt holes")
        print(decision.route)  # GenerationRoute.CODE_CADQUERY
        print(decision.confidence)  # 0.85
    """

    def __init__(self, config: Optional[RoutingConfig] = None) -> None:
        self.config = config or RoutingConfig()

        # Precompile keyword sets (lowercased) for fast lookup
        self._mechanical = set(
            kw.lower() for kw in self.config.mechanical_keywords
        )
        self._openscad = set(
            kw.lower() for kw in self.config.openscad_keywords
        )
        self._freeform = set(
            kw.lower() for kw in self.config.freeform_keywords
        )
        self._exploration = set(
            kw.lower() for kw in self.config.exploration_keywords
        )
        self._codebook = set(
            kw.lower() for kw in self.config.codebook_keywords
        )

        # Regex for dimensional cues (e.g. "80mm", "3.5 inches", "100x200")
        self._dim_pattern = re.compile(
            r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m|in|inch|inches|ft|feet)\b"
            r"|\b\d+\s*[xX×]\s*\d+(?:\s*[xX×]\s*\d+)?\b",
            re.IGNORECASE,
        )

    def route(
        self,
        prompt: str,
        has_image: bool = False,
        has_reference_geometry: bool = False,
        force_route: Optional[GenerationRoute] = None,
    ) -> RoutingDecision:
        """Analyze a prompt and decide the generation route.

        Args:
            prompt: User's text description of the desired geometry.
            has_image: Whether an image is provided as conditioning.
            has_reference_geometry: Whether reference geometry (STEP/STL)
                is provided.
            force_route: If set, override the analysis and use this
                route directly.

        Returns:
            ``RoutingDecision`` with selected route and explanation.
        """
        # Handle forced override
        if force_route is not None:
            return RoutingDecision(
                route=force_route,
                confidence=1.0,
                scores={force_route.value: 1.0},
                reasons=[f"Route forced to {force_route.value} by user."],
                forced=True,
            )

        prompt_lower = prompt.lower()
        words = set(re.findall(r"\b\w+\b", prompt_lower))

        # Initialize score accumulators
        scores: Dict[GenerationRoute, float] = {
            GenerationRoute.CODE_CADQUERY: 0.0,
            GenerationRoute.CODE_OPENSCAD: 0.0,
            GenerationRoute.NEURAL_VAE: 0.0,
            GenerationRoute.NEURAL_DIFFUSION: 0.0,
            GenerationRoute.NEURAL_VQVAE: 0.0,
        }
        reasons: List[str] = []

        # --- Signal 1: Keyword matching ---
        mechanical_hits = words.intersection(self._mechanical)
        openscad_hits = words.intersection(self._openscad)
        freeform_hits = words.intersection(self._freeform)
        exploration_hits = words.intersection(self._exploration)
        codebook_hits = words.intersection(self._codebook)

        if mechanical_hits:
            weight = min(len(mechanical_hits) * 0.15, 0.8)
            scores[GenerationRoute.CODE_CADQUERY] += weight
            reasons.append(
                f"Mechanical keywords matched ({len(mechanical_hits)}): "
                f"{', '.join(sorted(mechanical_hits)[:5])}"
            )

        if openscad_hits:
            weight = min(len(openscad_hits) * 0.3, 0.9)
            scores[GenerationRoute.CODE_OPENSCAD] += weight
            reasons.append(
                f"OpenSCAD keywords matched: {', '.join(sorted(openscad_hits))}"
            )

        if freeform_hits:
            weight = min(len(freeform_hits) * 0.2, 0.7)
            scores[GenerationRoute.NEURAL_DIFFUSION] += weight
            reasons.append(
                f"Freeform keywords matched ({len(freeform_hits)}): "
                f"{', '.join(sorted(freeform_hits)[:5])}"
            )

        if exploration_hits:
            weight = min(len(exploration_hits) * 0.25, 0.7)
            scores[GenerationRoute.NEURAL_VAE] += weight
            reasons.append(
                f"Exploration keywords matched: {', '.join(sorted(exploration_hits))}"
            )

        if codebook_hits:
            weight = min(len(codebook_hits) * 0.3, 0.6)
            scores[GenerationRoute.NEURAL_VQVAE] += weight
            reasons.append(
                f"Codebook keywords matched: {', '.join(sorted(codebook_hits))}"
            )

        # --- Signal 2: Dimensional cues ---
        dim_matches = self._dim_pattern.findall(prompt)
        if dim_matches:
            scores[GenerationRoute.CODE_CADQUERY] += 0.25
            reasons.append(
                f"Dimensional cues found ({len(dim_matches)}): "
                f"favors parameterized code generation."
            )

        # --- Signal 3: Image presence ---
        if has_image:
            scores[GenerationRoute.NEURAL_DIFFUSION] += 0.3
            scores[GenerationRoute.CODE_CADQUERY] += 0.1  # VLM code gen
            reasons.append(
                "Image provided: biases toward neural diffusion "
                "or VLM-conditioned code generation."
            )

        # --- Signal 4: Reference geometry ---
        if has_reference_geometry:
            scores[GenerationRoute.NEURAL_DIFFUSION] += 0.15
            scores[GenerationRoute.NEURAL_VAE] += 0.15
            reasons.append(
                "Reference geometry provided: enables latent-space "
                "conditioning."
            )

        # --- Signal 5: Prompt length heuristic ---
        # Short, specific prompts → code gen; long, vague → neural
        word_count = len(prompt.split())
        if word_count <= 15:
            scores[GenerationRoute.CODE_CADQUERY] += 0.1
        elif word_count >= 50:
            scores[GenerationRoute.NEURAL_DIFFUSION] += 0.1
            reasons.append(
                "Long/detailed prompt: may benefit from neural "
                "interpolation."
            )

        # --- Default bias ---
        # CadQuery gets a small default bonus because it's the most
        # reliable path (90%+ success rate)
        scores[GenerationRoute.CODE_CADQUERY] += 0.1

        # --- Select best route ---
        best_route = max(scores, key=scores.get)
        best_score = scores[best_route]

        # Normalize scores to [0, 1]
        total = sum(scores.values())
        if total > 0:
            confidence = best_score / total
        else:
            confidence = 0.0

        # Fall back to default if confidence is too low
        if confidence < self.config.confidence_threshold:
            best_route = self.config.default_route
            reasons.append(
                f"Confidence {confidence:.2f} below threshold "
                f"{self.config.confidence_threshold}; using default route."
            )
            confidence = self.config.confidence_threshold

        # If no keywords matched at all, use default
        if all(s < 0.15 for s in scores.values()):
            best_route = self.config.default_route
            reasons.append(
                "No strong signals detected; using default route "
                f"({self.config.default_route.value})."
            )

        return RoutingDecision(
            route=best_route,
            confidence=round(confidence, 3),
            scores={k.value: round(v, 3) for k, v in scores.items()},
            reasons=reasons,
        )

    def explain(self, decision: RoutingDecision) -> str:
        """Generate a human-readable explanation of a routing decision.

        Args:
            decision: A RoutingDecision to explain.

        Returns:
            Multi-line string describing the decision rationale.
        """
        lines = [
            f"Selected route: {decision.route.value}",
            f"Confidence: {decision.confidence:.1%}",
            f"Forced: {decision.forced}",
            "",
            "Score breakdown:",
        ]
        for route, score in sorted(
            decision.scores.items(), key=lambda x: -x[1]
        ):
            bar = "█" * int(score * 20)
            lines.append(f"  {route:25s} {score:.3f} {bar}")

        if decision.reasons:
            lines.append("")
            lines.append("Reasons:")
            for reason in decision.reasons:
                lines.append(f"  • {reason}")

        return "\n".join(lines)
