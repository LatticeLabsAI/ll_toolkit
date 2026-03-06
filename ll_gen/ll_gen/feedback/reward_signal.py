"""Convert disposal outcomes to scalar rewards for RL training.

The reward signal is a composite score (floored at −1.0) built from
multiple tiers:

==========================  =======  ===================================
Component                   Weight   Condition
==========================  =======  ===================================
Base validity               +0.8     Shape passes all BRepCheck tests
Shape constructed           +0.16    TopoDS_Shape exists (even if invalid)
Repairable                  +0.0     ShapeFix repaired it successfully
Per-tier bonus              +0.16    Each passing validation tier
Semantic match              +0.2     Bbox matches target dimensions
Critical error penalty      −0.1     Per critical-severity finding
==========================  =======  ===================================

Weights are configurable via ``FeedbackConfig``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from ll_gen.config import ErrorCategory, ErrorSeverity, FeedbackConfig
from ll_gen.proposals.disposal_result import DisposalResult, GeometryReport

_log = logging.getLogger(__name__)


def compute_reward(
    result: DisposalResult,
    config: Optional[FeedbackConfig] = None,
    target_dimensions: Optional[Tuple[float, float, float]] = None,
    target_volume: Optional[float] = None,
) -> float:
    """Compute a composite scalar reward from a DisposalResult.

    The reward is built up from independent components, each
    contributing a signed delta.  The components are:

    1. **Base validity** (``config.validity_reward``):
       +0.8 if the shape is completely valid, else 0.0.

    2. **Shape constructed** (``config.shape_constructed_reward``):
       +0.16 if a TopoDS_Shape was produced at all (even if invalid).

    3. **Repairable** (``config.repairable_reward``):
       +0.0 if deterministic repair succeeded.

    4. **Per-tier bonus** (``config.per_tier_reward``):
       +0.16 for each passing validation tier:
       manifold, watertight, euler, no-self-intersection.

    5. **Semantic match** (``config.semantic_match_reward``):
       +0.2 if bounding box dimensions match the target within
       ``config.dimension_tolerance_pct``, plus an additional +0.1
       (50% of ``semantic_match_reward``) if the volume also matches.
       Combined semantic reward is capped at 1.5x ``semantic_match_reward``.

    6. **Critical error penalty** (``config.critical_error_penalty``):
       −0.1 per critical-severity finding.

    The final reward is floored at −1.0 but has no upper clamp, so
    the RL trainer receives full gradient signal for semantic match.

    Args:
        result: Disposal result to score.
        config: Feedback config with reward weights.  Uses defaults
            if None.
        target_dimensions: Expected ``(w, h, d)`` bounding box dims.
            If provided, enables semantic match scoring.
        target_volume: Expected volume.  Used as supplementary
            semantic check (±10% tolerance).

    Returns:
        Scalar reward ≥ −1.0.
    """
    if config is None:
        config = FeedbackConfig()

    reward = 0.0

    # 1. Base validity
    if result.is_valid:
        reward += config.validity_reward

    # 2. Shape constructed
    if result.has_shape:
        reward += config.shape_constructed_reward

    # 3. Repairable
    if result.repair_attempted and result.repair_succeeded:
        reward += config.repairable_reward

    # 4. Per-tier bonus (only if not fully valid — avoid double-counting)
    if not result.is_valid:
        tiers_passed = _count_passing_tiers(result)
        reward += config.per_tier_reward * tiers_passed

    # 5. Semantic match (dimension + volume share a single reward budget)
    semantic_reward = 0.0
    if target_dimensions is not None and result.geometry_report is not None:
        if result.geometry_report.matches_dimensions(
            target_dimensions, config.dimension_tolerance_pct
        ):
            semantic_reward += config.semantic_match_reward

    if target_volume is not None and result.geometry_report is not None:
        if result.geometry_report.volume is not None:
            vol_err = abs(
                result.geometry_report.volume - target_volume
            ) / max(abs(target_volume), 1e-10)
            if vol_err <= config.dimension_tolerance_pct:
                semantic_reward += config.semantic_match_reward * 0.5

    # Budget = dimension bonus (1.0x) + volume bonus (0.5x) = 1.5x
    semantic_budget = config.semantic_match_reward * 1.5
    reward += min(semantic_reward, semantic_budget)

    # 6. Critical error penalty
    num_critical = sum(
        1 for f in result.error_details
        if f.severity == ErrorSeverity.CRITICAL
    )
    reward += config.critical_error_penalty * num_critical

    # Clamp lower bound only — upper bound is uncapped so the RL trainer
    # can distinguish valid shapes that also match target dimensions.
    reward = max(-1.0, reward)

    return round(reward, 4)


def compute_batch_rewards(
    results: List[DisposalResult],
    config: Optional[FeedbackConfig] = None,
    target_dimensions: Optional[Tuple[float, float, float]] = None,
    target_volume: Optional[float] = None,
) -> List[float]:
    """Compute rewards for a batch of DisposalResults.

    Convenience wrapper that calls ``compute_reward`` on each result.

    Args:
        results: List of disposal results.
        config: Shared feedback config.
        target_dimensions: Shared target dimensions.
        target_volume: Shared target volume.

    Returns:
        List of scalar rewards, same length as ``results``.
    """
    return [
        compute_reward(r, config, target_dimensions, target_volume)
        for r in results
    ]


def compute_batch_rewards_tensor(
    results: List[DisposalResult],
    config: Optional[FeedbackConfig] = None,
    target_dimensions: Optional[Tuple[float, float, float]] = None,
    target_volume: Optional[float] = None,
) -> Any:
    """Compute rewards as a PyTorch tensor.

    Requires torch to be installed.

    Args:
        results: List of disposal results.
        config: Shared feedback config.
        target_dimensions: Shared target dimensions.
        target_volume: Shared target volume.

    Returns:
        ``torch.FloatTensor`` of shape ``(len(results),)``.

    Raises:
        ImportError: If torch is not installed.
    """
    import torch

    rewards = compute_batch_rewards(
        results, config, target_dimensions, target_volume
    )
    return torch.tensor(rewards, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Internal tier counting
# ---------------------------------------------------------------------------

def _count_passing_tiers(result: DisposalResult) -> int:
    """Count how many validation tiers pass.

    Tiers checked:
    1. Manifold — no ``InvalidMultiConnexity`` or ``FreeEdge``
    2. Watertight — no ``NotClosed`` or ``NotConnected``
    3. Euler valid — V − E + F = 2
    4. No self-intersection — no ``SelfIntersectingWire`` etc.

    Args:
        result: DisposalResult to check.

    Returns:
        Number of passing tiers (0–4).
    """
    error_codes = {f.error_code for f in result.error_details}

    tiers_passed = 0

    # Manifold check
    manifold_codes = {"BRepCheck_InvalidMultiConnexity", "BRepCheck_FreeEdge"}
    if not error_codes.intersection(manifold_codes):
        tiers_passed += 1

    # Watertight check
    watertight_codes = {"BRepCheck_NotClosed", "BRepCheck_NotConnected"}
    if not error_codes.intersection(watertight_codes):
        tiers_passed += 1

    # Euler check
    if result.geometry_report is not None:
        ec = result.geometry_report.euler_characteristic
        if ec is not None and ec == 2:
            tiers_passed += 1
    else:
        # Missing geometry report means shape failed — no benefit of the doubt
        pass

    # Self-intersection check
    si_codes = {
        "BRepCheck_SelfIntersectingWire",
        "BRepCheck_IntersectingWires",
        "BOPAlgo_AlertSelfInterferingShape",
    }
    if not error_codes.intersection(si_codes):
        tiers_passed += 1

    return tiers_passed
