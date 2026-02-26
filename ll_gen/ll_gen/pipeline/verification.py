"""Visual verification — check generated geometry against intent.

Provides two verification strategies:

1. **VLM-based** — renders the generated shape and feeds the renders
   + original prompt to a vision-language model (CLIP, or an LLM with
   vision) to assess whether the geometry matches the description.

2. **Dimensional** — extracts numeric dimensions from the prompt
   (e.g. "80mm wide", "3mm thick") and compares them against the
   shape's bounding box.

The VLM approach is optional; dimensional checking is always available
when a GeometryReport is present.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ll_gen.proposals.disposal_result import GeometryReport

_log = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Outcome of visual / dimensional verification.

    Attributes:
        matches_intent: Whether the shape appears to match the prompt.
        confidence: Confidence score in [0, 1].
        method: Verification method used ("vlm", "dimensional", "both").
        dimension_checks: Per-dimension pass/fail results.
        issues: List of detected mismatches.
        vlm_response: Raw VLM response text (if VLM was used).
    """

    matches_intent: bool = True
    confidence: float = 0.5
    method: str = "dimensional"
    dimension_checks: List[Dict[str, Any]] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    vlm_response: Optional[str] = None


class VisualVerifier:
    """Verify generated geometry against the original prompt.

    Args:
        dimension_tolerance: Fractional tolerance for dimensional
            matching (0.15 = 15%).
        vlm_backend: Optional VLM backend name for vision-based
            verification. Supported: ``"clip"``, ``"llm"``.
            If None, only dimensional checking is used.

    Example::

        verifier = VisualVerifier()
        result = verifier.verify(
            render_paths=[Path("view_front.png"), ...],
            prompt="A box 100mm × 50mm × 20mm",
            geometry_report=report,
        )
        print(result.matches_intent)  # True/False
    """

    def __init__(
        self,
        dimension_tolerance: float = 0.15,
        vlm_backend: Optional[str] = None,
    ) -> None:
        self.dimension_tolerance = dimension_tolerance
        self.vlm_backend = vlm_backend
        self._clip_model = None
        self._clip_processor = None

    def verify(
        self,
        render_paths: Optional[List[Path]] = None,
        prompt: str = "",
        geometry_report: Optional[GeometryReport] = None,
    ) -> VerificationResult:
        """Run verification against the prompt.

        Combines dimensional checking (if GeometryReport available)
        with optional VLM verification (if renders available).

        Args:
            render_paths: Paths to rendered images of the shape.
            prompt: Original text prompt.
            geometry_report: GeometryReport from introspection.

        Returns:
            VerificationResult with pass/fail and details.
        """
        result = VerificationResult()
        methods_used: List[str] = []

        # --- Dimensional verification ---
        if geometry_report is not None and prompt:
            dim_result = self._verify_dimensions(prompt, geometry_report)
            result.dimension_checks = dim_result["checks"]
            if dim_result["issues"]:
                result.issues.extend(dim_result["issues"])
                result.matches_intent = False
            methods_used.append("dimensional")

        # --- Feature count verification ---
        if geometry_report is not None and prompt:
            feature_result = self._verify_features(prompt, geometry_report)
            if feature_result["issues"]:
                result.issues.extend(feature_result["issues"])
                result.matches_intent = False
            methods_used.append("feature_count")

        # --- VLM verification ---
        if (
            self.vlm_backend is not None
            and render_paths
            and len(render_paths) > 0
        ):
            try:
                vlm_result = self._verify_vlm(render_paths, prompt)
                result.vlm_response = vlm_result.get("response", "")
                if not vlm_result.get("matches", True):
                    result.issues.extend(vlm_result.get("issues", []))
                    result.matches_intent = False
                methods_used.append("vlm")
            except Exception as exc:
                _log.warning("VLM verification failed: %s", exc)

        # Compute confidence
        if not methods_used:
            result.confidence = 0.0
            result.method = "none"
        else:
            result.method = "+".join(methods_used)
            # Confidence is higher if more methods agree
            if result.matches_intent:
                result.confidence = min(0.5 + 0.2 * len(methods_used), 1.0)
            else:
                result.confidence = max(0.8 - 0.1 * len(result.issues), 0.1)

        return result

    # ------------------------------------------------------------------
    # Dimensional verification
    # ------------------------------------------------------------------

    def _verify_dimensions(
        self,
        prompt: str,
        report: GeometryReport,
    ) -> Dict[str, Any]:
        """Extract dimensions from the prompt and compare to geometry.

        Recognizes patterns like:
        - "80mm wide" → width ~= 80mm
        - "3mm thick" → height/depth ~= 3mm
        - "100 x 50 x 20" → overall dims
        - "diameter 10mm" → circle with ø10
        - "radius 5" → circle with r=5

        Returns:
            Dict with "checks" (list of per-dim results) and
            "issues" (list of mismatch descriptions).
        """
        checks: List[Dict[str, Any]] = []
        issues: List[str] = []

        if report.bounding_box is None:
            return {"checks": checks, "issues": issues}

        dims = report.bbox_dimensions
        if dims is None:
            return {"checks": checks, "issues": issues}

        # Sort dims descending for matching (largest = width, etc.)
        sorted_dims = sorted(dims, reverse=True)

        # Pattern 1: "NNmm wide/long/tall/thick/deep/high"
        dim_patterns = [
            (r"(\d+(?:\.\d+)?)\s*(?:mm|cm|m|in)\s*(?:wide|width)", "width"),
            (r"(\d+(?:\.\d+)?)\s*(?:mm|cm|m|in)\s*(?:long|length)", "length"),
            (r"(\d+(?:\.\d+)?)\s*(?:mm|cm|m|in)\s*(?:tall|height|high)", "height"),
            (r"(\d+(?:\.\d+)?)\s*(?:mm|cm|m|in)\s*(?:thick|thickness|deep|depth)", "thickness"),
            (r"(?:wide|width)\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(?:mm|cm|m|in)", "width"),
            (r"(?:diameter|dia|ø)\s*(\d+(?:\.\d+)?)\s*(?:mm|cm|m|in)?", "diameter"),
            (r"(?:radius|r)\s*(\d+(?:\.\d+)?)\s*(?:mm|cm|m|in)?", "radius"),
        ]

        extracted: List[Tuple[float, str]] = []
        for pattern, name in dim_patterns:
            for match in re.finditer(pattern, prompt, re.IGNORECASE):
                value = float(match.group(1))
                extracted.append((value, name))

        # Pattern 2: "N x N x N" overall dimensions
        multi_dim = re.search(
            r"(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)"
            r"(?:\s*[xX×]\s*(\d+(?:\.\d+)?))?",
            prompt,
        )
        if multi_dim:
            vals = [float(multi_dim.group(i)) for i in range(1, 4) if multi_dim.group(i)]
            vals.sort(reverse=True)
            for val, actual in zip(vals, sorted_dims):
                check = {
                    "name": "multi_dim",
                    "expected": val,
                    "actual": actual,
                    "tolerance": self.dimension_tolerance,
                    "passed": self._within_tolerance(actual, val),
                }
                checks.append(check)
                if not check["passed"]:
                    issues.append(
                        f"Dimension mismatch: expected ~{val:.1f}, "
                        f"got {actual:.1f} (tolerance={self.dimension_tolerance:.0%})"
                    )

        # Check extracted single dimensions
        for value, name in extracted:
            if name == "width":
                actual = sorted_dims[0]
            elif name == "length":
                actual = sorted_dims[0] if len(sorted_dims) > 0 else 0
            elif name == "height":
                actual = sorted_dims[1] if len(sorted_dims) > 1 else sorted_dims[0]
            elif name == "thickness":
                actual = sorted_dims[-1]
            elif name == "diameter":
                # Check against any dimension
                actual = min(sorted_dims, key=lambda d: abs(d - value))
            elif name == "radius":
                value = value * 2  # Compare diameter
                actual = min(sorted_dims, key=lambda d: abs(d - value))
            else:
                continue

            check = {
                "name": name,
                "expected": value,
                "actual": actual,
                "tolerance": self.dimension_tolerance,
                "passed": self._within_tolerance(actual, value),
            }
            checks.append(check)
            if not check["passed"]:
                issues.append(
                    f"{name.capitalize()} mismatch: expected ~{value:.1f}, "
                    f"got {actual:.1f}"
                )

        return {"checks": checks, "issues": issues}

    # ------------------------------------------------------------------
    # Feature verification
    # ------------------------------------------------------------------

    def _verify_features(
        self,
        prompt: str,
        report: GeometryReport,
    ) -> Dict[str, Any]:
        """Check feature counts mentioned in the prompt.

        Recognizes patterns like "4 bolt holes", "6 fins",
        "2 mounting holes".

        Returns:
            Dict with "issues" list.
        """
        issues: List[str] = []

        # Extract feature counts
        hole_pattern = re.search(
            r"(\d+)\s*(?:bolt\s*)?holes?", prompt, re.IGNORECASE
        )
        if hole_pattern:
            expected_holes = int(hole_pattern.group(1))
            # Each through-hole typically adds 2 cylindrical faces
            cylinder_count = report.surface_types.get("Cylinder", 0)
            estimated_holes = cylinder_count // 2

            if estimated_holes < expected_holes:
                issues.append(
                    f"Expected {expected_holes} holes but found "
                    f"~{estimated_holes} (based on {cylinder_count} "
                    f"cylindrical faces)."
                )

        return {"issues": issues}

    # ------------------------------------------------------------------
    # VLM verification
    # ------------------------------------------------------------------

    def _verify_vlm(
        self,
        render_paths: List[Path],
        prompt: str,
    ) -> Dict[str, Any]:
        """Use a vision-language model to verify the geometry.

        Supported backends:
        - "clip": CLIP-based similarity scoring
        - "llm": LLM with vision capability (requires API key)

        Returns:
            Dict with "matches" (bool), "response" (str), "issues" (list).
        """
        if self.vlm_backend == "clip":
            return self._verify_clip(render_paths, prompt)
        elif self.vlm_backend == "llm":
            return self._verify_llm_vision(render_paths, prompt)
        else:
            _log.warning("Unknown VLM backend: %s", self.vlm_backend)
            return {"matches": True, "response": "", "issues": []}

    def _verify_clip(
        self,
        render_paths: List[Path],
        prompt: str,
    ) -> Dict[str, Any]:
        """CLIP-based similarity between renders and text prompt.

        Computes cosine similarity between the CLIP text embedding
        of the prompt and CLIP image embeddings of each render.
        """
        try:
            import torch
            from PIL import Image
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            _log.warning("CLIP verification requires transformers+PIL: %s", exc)
            return {"matches": True, "response": "", "issues": []}

        if self._clip_model is None:
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = self._clip_model
        processor = self._clip_processor

        # Load images
        images = []
        for rp in render_paths:
            if rp.exists():
                try:
                    img = Image.open(rp).convert("RGB")
                    images.append(img)
                except Exception:
                    continue

        if not images:
            return {"matches": True, "response": "No valid renders", "issues": []}

        # Compute similarities
        inputs = processor(
            text=[prompt],
            images=images,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            outputs = model(**inputs)
            similarities = outputs.logits_per_text.softmax(dim=-1)

        avg_similarity = similarities.mean().item()
        matches = avg_similarity > 0.25  # CLIP threshold

        response = f"CLIP similarity: {avg_similarity:.3f}"
        issues = []
        if not matches:
            issues.append(
                f"CLIP similarity ({avg_similarity:.3f}) below threshold (0.25). "
                f"The shape may not match the description."
            )

        return {
            "matches": matches,
            "response": response,
            "issues": issues,
        }

    def _verify_llm_vision(
        self,
        render_paths: List[Path],
        prompt: str,
    ) -> Dict[str, Any]:
        """LLM-with-vision verification.

        Sends renders to an LLM (via cadling's ChatAgent) and asks
        whether the geometry matches the description.
        """
        try:
            from cadling.generation.codegen.cadquery_generator import CadQueryGenerator
        except ImportError:
            _log.warning("cadling not available for LLM vision verification")
            return {"matches": True, "response": "", "issues": []}

        # Use the first available render
        image_path = None
        for rp in render_paths:
            if rp.exists():
                image_path = rp
                break

        if image_path is None:
            return {"matches": True, "response": "No renders available", "issues": []}

        # Create a temporary generator to use the ChatAgent
        gen = CadQueryGenerator()
        verification_prompt = (
            f"I asked for: \"{prompt}\"\n\n"
            f"The attached image shows the generated 3D CAD geometry. "
            f"Does this shape match the description? Answer with:\n"
            f"- MATCH: if the shape reasonably matches\n"
            f"- MISMATCH: if the shape does not match, explain why\n"
            f"Be brief (1-2 sentences)."
        )

        try:
            response = gen.generate(
                verification_prompt,
                image_path=image_path,
            )
            response_lower = response.lower() if response else ""
            matches = "mismatch" not in response_lower
            issues = []
            if not matches:
                issues.append(f"VLM says: {response}")

            return {
                "matches": matches,
                "response": response or "",
                "issues": issues,
            }
        except Exception as exc:
            _log.warning("LLM vision verification failed: %s", exc)
            return {"matches": True, "response": str(exc), "issues": []}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _within_tolerance(self, actual: float, expected: float) -> bool:
        """Check if actual is within tolerance of expected."""
        if expected == 0:
            return abs(actual) < self.dimension_tolerance
        return abs(actual - expected) / abs(expected) <= self.dimension_tolerance
