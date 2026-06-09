"""Deterministic tests for the validity-evaluation harness (SPEC-1 M3, T3.2).

These exercise the harness's aggregation, error-handling, and checkpoint-format
logic without torch, pythonocc, or cadquery: the generator is a fake that
returns a sentinel proposal and the disposal step is an injected ``dispose_fn``
returning canned ``DisposalResult`` objects.  This isolates the harness logic
(counting valid/total, seeding, exception accounting) from the OCC kernel,
which is measured separately by the real baseline run.
"""

from __future__ import annotations

from typing import Any

import pytest

from ll_gen.generators.base import BaseNeuralGenerator
from ll_gen.proposals.base import BaseProposal
from ll_gen.proposals.disposal_result import DisposalResult, GeometryReport
from ll_gen.training.evaluate_validity import (
    _extract_state_dict,
    _normalize_prompts,
    evaluate_validity,
    load_generator_checkpoint,
)


class _FakeGenerator(BaseNeuralGenerator):
    """Minimal generator: no model, returns a sentinel proposal per call.

    Records the prompts it was asked to generate for so tests can assert the
    harness drives the full prompt x n_samples grid.
    """

    def __init__(self) -> None:
        super().__init__(device="cpu")
        self.seen_prompts: list[str] = []

    def generate(
        self,
        prompt: str,
        conditioning: Any = None,
        error_context: Any = None,
    ) -> BaseProposal:
        self.seen_prompts.append(prompt)
        return BaseProposal()

    def generate_candidates(
        self,
        prompt: str,
        num_candidates: int = 3,
        conditioning: Any = None,
    ) -> list[BaseProposal]:
        return [self.generate(prompt) for _ in range(num_candidates)]


def _valid_result() -> DisposalResult:
    # has_shape is derived from `shape is not None`; a truthy sentinel marks a
    # constructed, valid shape. The reward now requires a closed solid for full
    # credit, so a "valid result" fixture carries a solid geometry report.
    return DisposalResult(
        shape=object(),
        is_valid=True,
        geometry_report=GeometryReport(solid_count=1, is_solid=True),
    )


def _invalid_result() -> DisposalResult:
    return DisposalResult(shape=None, is_valid=False)


# ---------------------------------------------------------------------------
# Prompt normalization
# ---------------------------------------------------------------------------


def test_normalize_accepts_strings_and_dicts() -> None:
    records = _normalize_prompts(
        ["a bracket", {"prompt": "a flange"}, {"caption": "a gear"}]
    )
    assert [r["prompt"] for r in records] == ["a bracket", "a flange", "a gear"]


def test_normalize_carries_target_dimensions() -> None:
    records = _normalize_prompts([{"prompt": "box", "target_dimensions": [10, 20, 30]}])
    assert records[0]["target_dimensions"] == (10, 20, 30)


def test_normalize_rejects_non_str_non_dict() -> None:
    with pytest.raises(TypeError):
        _normalize_prompts([123])


# ---------------------------------------------------------------------------
# Core validity-rate behavior (the TDD fixtures from the plan)
# ---------------------------------------------------------------------------


def test_always_valid_generator_rate_is_one() -> None:
    gen = _FakeGenerator()
    metrics = evaluate_validity(
        gen,
        ["p1", "p2", "p3"],
        n_samples=2,
        dispose_fn=lambda _proposal: _valid_result(),
    )
    assert metrics.validity_rate == 1.0
    assert metrics.num_samples == 6
    assert metrics.num_valid == 6
    assert metrics.compile_rate == 1.0


def test_always_invalid_generator_rate_is_zero() -> None:
    gen = _FakeGenerator()
    metrics = evaluate_validity(
        gen,
        ["p1", "p2"],
        n_samples=4,
        dispose_fn=lambda _proposal: _invalid_result(),
    )
    assert metrics.validity_rate == 0.0
    assert metrics.num_samples == 8
    assert metrics.num_valid == 0


def test_mixed_validity_rate_matches_fraction() -> None:
    # Valid iff the prompt contains "good"; 1 of 2 prompts -> 0.5.
    gen = _FakeGenerator()

    def dispose(_proposal: BaseProposal) -> DisposalResult:
        # The most recent prompt the fake generated for.
        return _valid_result() if "good" in gen.seen_prompts[-1] else _invalid_result()

    metrics = evaluate_validity(
        gen,
        ["good part", "bad part"],
        n_samples=10,
        dispose_fn=dispose,
    )
    assert metrics.validity_rate == pytest.approx(0.5)
    assert metrics.num_valid == 10
    assert metrics.num_samples == 20


def test_n_samples_drives_full_grid() -> None:
    gen = _FakeGenerator()
    evaluate_validity(
        gen, ["a", "b"], n_samples=5, dispose_fn=lambda _p: _valid_result()
    )
    assert len(gen.seen_prompts) == 10
    assert gen.seen_prompts.count("a") == 5
    assert gen.seen_prompts.count("b") == 5


def test_reward_signal_is_computed_for_valid_results() -> None:
    gen = _FakeGenerator()
    metrics = evaluate_validity(
        gen, ["p"], n_samples=1, dispose_fn=lambda _p: _valid_result()
    )
    # Valid + has_shape -> validity_reward (0.8) + shape_constructed (0.16).
    assert metrics.mean_reward == pytest.approx(0.96)


# ---------------------------------------------------------------------------
# Honest error accounting
# ---------------------------------------------------------------------------


def test_dispose_exception_counts_as_invalid_not_dropped() -> None:
    gen = _FakeGenerator()

    def boom(_proposal: BaseProposal) -> DisposalResult:
        raise RuntimeError("pythonocc not available")

    metrics = evaluate_validity(gen, ["p1", "p2"], n_samples=3, dispose_fn=boom)
    # 6 samples all errored -> counted as 6 invalid results, not a 0-length set.
    assert metrics.num_samples == 6
    assert metrics.num_valid == 0
    assert metrics.validity_rate == 0.0


def test_partial_errors_still_counted() -> None:
    gen = _FakeGenerator()
    calls = {"n": 0}

    def flaky(_proposal: BaseProposal) -> DisposalResult:
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise RuntimeError("transient disposal failure")
        return _valid_result()

    metrics = evaluate_validity(gen, ["p"], n_samples=4, dispose_fn=flaky)
    assert metrics.num_samples == 4
    assert metrics.num_valid == 2
    assert metrics.validity_rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_empty_prompts_raises() -> None:
    gen = _FakeGenerator()
    with pytest.raises(ValueError):
        evaluate_validity(gen, [], dispose_fn=lambda _p: _valid_result())


def test_zero_samples_raises() -> None:
    gen = _FakeGenerator()
    with pytest.raises(ValueError):
        evaluate_validity(
            gen, ["p"], n_samples=0, dispose_fn=lambda _p: _valid_result()
        )


# ---------------------------------------------------------------------------
# Checkpoint-format bridging (nested trainer format vs flat state dict)
# ---------------------------------------------------------------------------


def test_extract_state_dict_unwraps_trainer_checkpoint() -> None:
    flat = {"layer.weight": "W", "layer.bias": "B"}
    nested = {"model_state_dict": flat, "optimizer_state_dict": {}, "step_count": 5}
    assert _extract_state_dict(nested) is flat


def test_extract_state_dict_passes_through_flat() -> None:
    flat = {"layer.weight": "W"}
    assert _extract_state_dict(flat) is flat


def test_load_generator_checkpoint_missing_file_raises(tmp_path) -> None:
    gen = _FakeGenerator()

    # _FakeGenerator has no _init_model and _model stays None, so the loader
    # raises before touching torch — assert it surfaces the missing model
    # rather than silently no-op'ing.
    with pytest.raises((FileNotFoundError, RuntimeError, ImportError)):
        load_generator_checkpoint(gen, tmp_path / "does_not_exist.pt")
