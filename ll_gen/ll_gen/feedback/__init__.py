"""Feedback layer — error mapping, structured feedback, and reward signals.

This package bridges the gap between OpenCASCADE's 37 low-level
``BRepCheck_Status`` error codes and the 6 neural-interpretable
error categories that LLMs and RL reward functions can act on.

Modules
-------
error_mapper
    Maps each OCC BRepCheck code to an ``ErrorCategory``, severity,
    human-readable description, and correction suggestion.
feedback_builder
    Builds structured feedback for LLM code retry and neural re-generation.
reward_signal
    Converts disposal outcomes to scalar rewards for RL training.
"""
from ll_gen.feedback.error_mapper import (
    categorize_errors,
    map_brep_errors,
    map_single_error,
    OCC_ERROR_MAP,
)
from ll_gen.feedback.feedback_builder import (
    build_code_feedback,
    build_neural_feedback,
    build_training_feedback,
)
from ll_gen.feedback.reward_signal import compute_batch_rewards, compute_reward

__all__ = [
    "OCC_ERROR_MAP",
    "categorize_errors",
    "map_brep_errors",
    "map_single_error",
    "build_code_feedback",
    "build_neural_feedback",
    "build_training_feedback",
    "compute_batch_rewards",
    "compute_reward",
]
