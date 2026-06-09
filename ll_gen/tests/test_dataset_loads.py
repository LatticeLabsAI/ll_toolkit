"""Dataset loading smoke tests for the DeepCAD subset (SPEC-1 M3, T3.1).

Three layers, increasingly dependent on external resources:

1. ``DeepCADDataset`` local-directory mechanics, using a tiny fixture written to
   a temp dir (no network, always runs).
2. The materialized local subset under ``data/deepcad_dsl`` (skipped until
   ``scripts/download_deepcad_subset.py`` has been run).
3. Live streaming from the ``palapav/DeepCAD-DSL`` Hub dataset (integration;
   skipped when ``datasets`` is absent or the Hub is unreachable).

Per SPEC-1 M3 the dataset is real (no synthetic CI fallback), so layers 2 and 3
skip rather than fabricate data when the resource is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ll_gen.datasets.deepcad_loader import DeepCADDataset, load_deepcad

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_SUBSET = _REPO_ROOT / "data" / "deepcad_dsl"

# token_ids / attention_mask are padded to max_commands * 10.
_MAX_COMMANDS = 60
_EXPECTED_LEN = _MAX_COMMANDS * 10


def _write_fixture(root: Path, split: str, count: int) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    sequence = [
        {"type": "SOL", "params": []},
        {"type": "LINE", "params": [0.0, 0.0, 0.1, 0.0]},
        {"type": "LINE", "params": [0.1, 0.0, 0.1, 0.1]},
        {"type": "LINE", "params": [0.1, 0.1, 0.0, 0.1]},
        {"type": "LINE", "params": [0.0, 0.1, 0.0, 0.0]},
        {"type": "EXTRUDE", "params": [0.05]},
        {"type": "EOS", "params": []},
    ]
    for i in range(count):
        (split_dir / f"{i:04d}.json").write_text(json.dumps({"sequence": sequence}))


def _assert_sample_shape(sample: dict) -> None:
    assert set(sample).issuperset(
        {"token_ids", "command_tokens", "attention_mask", "num_commands"}
    )
    assert len(sample["token_ids"]) == _EXPECTED_LEN
    assert len(sample["attention_mask"]) == _EXPECTED_LEN
    assert sample["num_commands"] >= 1
    assert sample["token_ids"][0] == 1  # BOS
    first_cmd = sample["command_tokens"][0]
    assert {"command_type", "parameters", "parameter_mask"} <= set(first_cmd)


# ---------------------------------------------------------------------------
# Layer 1 — local-directory mechanics (no network)
# ---------------------------------------------------------------------------


def test_local_directory_loader_counts_and_shapes(tmp_path) -> None:
    _write_fixture(tmp_path, "train", count=5)
    ds = load_deepcad(path=str(tmp_path), split="train", max_commands=_MAX_COMMANDS)
    assert isinstance(ds, DeepCADDataset)
    assert len(ds) == 5
    _assert_sample_shape(ds[0])


def test_local_directory_max_samples_caps_count(tmp_path) -> None:
    _write_fixture(tmp_path, "train", count=10)
    ds = load_deepcad(path=str(tmp_path), split="train", max_samples=3)
    assert len(ds) == 3


def test_missing_split_dir_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_deepcad(path=str(tmp_path), split="train")


# ---------------------------------------------------------------------------
# Layer 2 — materialized real subset (skip until downloaded)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not (_LOCAL_SUBSET / "train").exists(),
    reason="run scripts/download_deepcad_subset.py to materialize the subset",
)
def test_materialized_subset_loads() -> None:
    ds = DeepCADDataset(data_dir=str(_LOCAL_SUBSET), split="train")
    assert len(ds) > 0
    _assert_sample_shape(ds[0])
    # Real DeepCAD sketches have multiple commands.
    assert ds[0]["num_commands"] >= 2


# ---------------------------------------------------------------------------
# Layer 3 — live Hub streaming (integration)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
def test_streaming_from_hub_yields_real_samples() -> None:
    pytest.importorskip("datasets")
    try:
        ds = load_deepcad(
            split="train", streaming=True, max_samples=3, max_commands=_MAX_COMMANDS
        )
        samples = list(ds)
    except Exception as exc:  # network / hub unavailable
        pytest.skip(f"Hub dataset unreachable: {exc}")

    assert len(samples) == 3
    for sample in samples:
        _assert_sample_shape(sample)
