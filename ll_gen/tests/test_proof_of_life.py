"""Tests for the proof-of-life driver's pure helpers (SPEC-1 M3, T3.4/T3.5).

The end-to-end before/after RL run is exercised as a measured experiment (it
needs torch + the OCC dispose kernel); these cover the prompt-handling and
summary helpers that shape its report without those heavy deps.
"""

from __future__ import annotations

import json

import pytest

from ll_gen.training.proof_of_life import _load_prompts, _prompt_text


def test_prompt_text_from_string() -> None:
    assert _prompt_text("a bracket") == "a bracket"


def test_prompt_text_from_dict_prompt_then_caption() -> None:
    assert _prompt_text({"prompt": "a flange"}) == "a flange"
    assert _prompt_text({"caption": "a gear"}) == "a gear"


def test_prompt_text_coerces_other() -> None:
    assert _prompt_text(42) == "42"


def test_load_prompts_jsonl_and_raw_lines(tmp_path) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text('{"prompt": "a box"}\n\nplain string prompt\n')
    prompts = _load_prompts(str(path))
    assert prompts == [{"prompt": "a box"}, "plain string prompt"]


def test_load_prompts_empty_raises(tmp_path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n  \n")
    with pytest.raises(ValueError):
        _load_prompts(str(path))
