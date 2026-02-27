"""Constraint predictor — extracts geometric constraints from prompts and embeddings.

This module provides constraint detection from natural language text, as well as
a learned predictor that can extract constraints from embeddings. Constraints
guide the generation process and inform reward weighting in RL training.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ll_gen.conditioning.embeddings import ConditioningEmbeddings

_log = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _log.debug("torch not available; learned constraints unavailable")


class ConstraintType(str, Enum):
    """Enumeration of geometric constraint types.

    Attributes:
        BOUNDING_BOX: Dimensional constraints (width, height, depth, etc.).
        SYMMETRY: Mirror or rotational symmetry requirements.
        PLANARITY: Surfaces must be flat/planar.
        SMOOTHNESS: Continuous/smooth surface transitions required.
        CONNECTIVITY: Parts must connect or be joined.
        MANIFOLD: Closed, watertight topology.
        REGULARITY: Regular patterns (arrays, grids, evenly spaced).
        WATERTIGHT: Closed volume suitable for 3D printing.
    """

    BOUNDING_BOX = "bounding_box"
    SYMMETRY = "symmetry"
    PLANARITY = "planarity"
    SMOOTHNESS = "smoothness"
    CONNECTIVITY = "connectivity"
    MANIFOLD = "manifold"
    REGULARITY = "regularity"
    WATERTIGHT = "watertight"


@dataclass
class ConstraintPrediction:
    """Prediction of a single geometric constraint.

    Attributes:
        constraint_type: Type of constraint (from ConstraintType enum).
        confidence: Confidence score in [0.0, 1.0].
        parameters: Type-specific parameters (e.g., dimensions, axis).
        source: Origin of prediction ("keyword", "dimension_regex", or "learned").
    """

    constraint_type: ConstraintType
    confidence: float
    parameters: dict[str, Any]
    source: str

    def __post_init__(self) -> None:
        """Validate confidence is in [0.0, 1.0]."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")


class ConstraintPredictor:
    """Extracts geometric constraints from prompts and embeddings.

    Uses keyword patterns, regular expressions for dimensions, and optional
    learned MLP for constraint prediction from embeddings.

    Attributes:
        device: Torch device for learned model.
        embedding_dim: Dimension of embeddings (default 768).
    """

    def __init__(self, device: str = "cpu", embedding_dim: int = 768) -> None:
        """Initialize ConstraintPredictor.

        Args:
            device: Torch device ("cpu" or "cuda:*").
            embedding_dim: Embedding dimension for learned model input.
        """
        self.device = device
        self.embedding_dim = embedding_dim
        self._learned_model = None
        self._learned_model_initialized = False
        self.use_learned_model = False

    def predict_from_prompt(self, prompt: str) -> list[ConstraintPrediction]:
        """Extract constraints from natural language prompt.

        Parses prompt for:
        - Dimensional patterns (mm, cm, m, in, inches)
        - Symmetry keywords (symmetric, mirror, symmetrical)
        - Smoothness keywords (smooth, fillet, round, continuous)
        - Regularity keywords (pattern, array, grid, repeated, evenly spaced)
        - Connectivity keywords (connected, joined, attached, assembled)
        - Watertight/manifold keywords (solid, closed, watertight, manifold, printable)
        - Planarity keywords (flat, planar, plane)

        Always adds MANIFOLD constraint with confidence 0.5 (default expectation).

        Args:
            prompt: Natural language prompt.

        Returns:
            List of ConstraintPrediction objects.
        """
        predictions: list[ConstraintPrediction] = []
        prompt_lower = prompt.lower()
        keyword_count = 0

        # Dimensional constraints
        dim_pattern = r"(\d+(?:\.\d+)?)\s*(?:mm|cm|m|in|inch|inches|wide|tall|thick|deep|long|x)"
        dim_matches = re.findall(dim_pattern, prompt_lower, re.IGNORECASE)
        if dim_matches:
            dimensions = [float(d) for d in dim_matches]
            predictions.append(
                ConstraintPrediction(
                    constraint_type=ConstraintType.BOUNDING_BOX,
                    confidence=min(0.9, 0.7 + 0.1 * len(dimensions)),
                    parameters={
                        "dimensions": dimensions,
                        "count": len(dimensions),
                    },
                    source="dimension_regex",
                )
            )
            keyword_count += 1

        # Symmetry constraints
        symmetry_keywords = ["symmetric", "mirror", "symmetrical", "reflection"]
        if any(kw in prompt_lower for kw in symmetry_keywords):
            predictions.append(
                ConstraintPrediction(
                    constraint_type=ConstraintType.SYMMETRY,
                    confidence=0.85,
                    parameters={"keywords": [kw for kw in symmetry_keywords if kw in prompt_lower]},
                    source="keyword",
                )
            )
            keyword_count += 1

        # Smoothness constraints
        smoothness_keywords = ["smooth", "fillet", "round", "continuous", "blend"]
        if any(kw in prompt_lower for kw in smoothness_keywords):
            predictions.append(
                ConstraintPrediction(
                    constraint_type=ConstraintType.SMOOTHNESS,
                    confidence=0.80,
                    parameters={"keywords": [kw for kw in smoothness_keywords if kw in prompt_lower]},
                    source="keyword",
                )
            )
            keyword_count += 1

        # Regularity constraints
        regularity_keywords = [
            "pattern",
            "array",
            "grid",
            "repeated",
            "evenly spaced",
            "regular",
        ]
        if any(kw in prompt_lower for kw in regularity_keywords):
            predictions.append(
                ConstraintPrediction(
                    constraint_type=ConstraintType.REGULARITY,
                    confidence=0.80,
                    parameters={"keywords": [kw for kw in regularity_keywords if kw in prompt_lower]},
                    source="keyword",
                )
            )
            keyword_count += 1

        # Connectivity constraints
        connectivity_keywords = ["connected", "joined", "attached", "assembled", "union"]
        if any(kw in prompt_lower for kw in connectivity_keywords):
            predictions.append(
                ConstraintPrediction(
                    constraint_type=ConstraintType.CONNECTIVITY,
                    confidence=0.80,
                    parameters={"keywords": [kw for kw in connectivity_keywords if kw in prompt_lower]},
                    source="keyword",
                )
            )
            keyword_count += 1

        # Watertight/Manifold constraints
        watertight_keywords = ["solid", "closed", "watertight", "manifold", "printable"]
        if any(kw in prompt_lower for kw in watertight_keywords):
            predictions.append(
                ConstraintPrediction(
                    constraint_type=ConstraintType.WATERTIGHT,
                    confidence=0.85,
                    parameters={"keywords": [kw for kw in watertight_keywords if kw in prompt_lower]},
                    source="keyword",
                )
            )
            keyword_count += 1

        # Planarity constraints
        planarity_keywords = ["flat", "planar", "plane"]
        if any(kw in prompt_lower for kw in planarity_keywords):
            predictions.append(
                ConstraintPrediction(
                    constraint_type=ConstraintType.PLANARITY,
                    confidence=0.80,
                    parameters={"keywords": [kw for kw in planarity_keywords if kw in prompt_lower]},
                    source="keyword",
                )
            )
            keyword_count += 1

        # Always add default MANIFOLD constraint
        # (expectation for CAD solids, even without explicit mention)
        predictions.append(
            ConstraintPrediction(
                constraint_type=ConstraintType.MANIFOLD,
                confidence=0.5,
                parameters={"default": True},
                source="keyword",
            )
        )

        return predictions

    def predict_from_embeddings(
        self,
        embeddings: ConditioningEmbeddings,
    ) -> list[ConstraintPrediction]:
        """Predict constraints from embeddings using learned model.

        If no learned model has been trained/initialized, returns empty list.
        Requires torch to be available.

        Args:
            embeddings: ConditioningEmbeddings instance.

        Returns:
            List of ConstraintPrediction objects from learned model.
        """
        if not _TORCH_AVAILABLE:
            _log.debug("torch not available; cannot use learned constraints")
            return []

        if self._learned_model is None:
            return []

        if embeddings.pooled_embedding is None:
            _log.warning("embeddings.pooled_embedding is None; cannot predict constraints")
            return []

        try:
            with torch.no_grad():
                # Convert to tensor
                emb_tensor = torch.from_numpy(embeddings.pooled_embedding).float()
                if emb_tensor.dim() == 1:
                    emb_tensor = emb_tensor.unsqueeze(0)
                emb_tensor = emb_tensor.to(self.device)

                # Run through model
                logits = self._learned_model(emb_tensor)
                probs = torch.sigmoid(logits)
                probs_np = probs.detach().cpu().numpy()[0]

            # Convert to predictions
            constraint_types = list(ConstraintType)
            predictions = []
            for i, ctype in enumerate(constraint_types):
                confidence = float(probs_np[i])
                if confidence > 0.1:  # Only include non-negligible predictions
                    predictions.append(
                        ConstraintPrediction(
                            constraint_type=ctype,
                            confidence=confidence,
                            parameters={},
                            source="learned",
                        )
                    )

            return predictions
        except Exception as e:
            _log.error(f"Error predicting constraints from embeddings: {e}")
            return []

    def to_loss_weights(
        self,
        predictions: list[ConstraintPrediction],
    ) -> dict[str, float]:
        """Convert constraint predictions to loss weights.

        Maps constraint types to confidence scores for use in RL training
        reward weighting.

        Args:
            predictions: List of constraint predictions.

        Returns:
            Dictionary mapping constraint type names to confidence weights.
        """
        weights: dict[str, float] = {}
        for pred in predictions:
            constraint_name = pred.constraint_type.value
            if constraint_name in weights:
                weights[constraint_name] = max(weights[constraint_name], pred.confidence)
            else:
                weights[constraint_name] = pred.confidence

        return weights

    def initialize_learned_model(self) -> None:
        """Public entry point for initializing the learned constraint model.

        Delegates to ``_init_learned_model`` which creates the MLP.
        """
        self._init_learned_model()

    def _init_learned_model(self) -> None:
        """Initialize the learned constraint prediction MLP.

        Creates a small feedforward network:
        - Input: embedding_dim
        - Hidden: 128 with ReLU
        - Output: len(ConstraintType) with sigmoid
        """
        if not _TORCH_AVAILABLE:
            _log.warning("torch not available; cannot initialize learned model")
            return

        try:
            num_constraints = len(ConstraintType)
            self._learned_model = nn.Sequential(
                nn.Linear(self.embedding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, num_constraints),
            )
            self._learned_model = self._learned_model.to(self.device)
            self._learned_model_initialized = True
            _log.info(
                f"Initialized learned constraint model: "
                f"{self.embedding_dim} -> 128 -> {num_constraints}"
            )
        except Exception as e:
            _log.error(f"Failed to initialize learned model: {e}")
            self._learned_model_initialized = False

    def set_learned_model(self, model: Any | None) -> None:
        """Set a pre-trained learned constraint model.

        Args:
            model: Torch nn.Module or None to clear the model.
        """
        self._learned_model = model
        if model is not None:
            self._learned_model = model.to(self.device)
            self._learned_model_initialized = True
        else:
            self._learned_model_initialized = False

    def get_learned_model(self) -> Any | None:
        """Get the current learned constraint model.

        Returns:
            The learned model or None.
        """
        return self._learned_model
