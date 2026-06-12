"""Conformance: the LatticeLabs Toolkit consumes a Three Indexer (SPEC-2) dataset.

Drives the REAL cadling consumer — ``CADStreamingDataset`` (license-family cohort filter +
per-sample transform) and ``BRepGraphBuilder`` (raw STEP → B-Rep face graph) — against a
committed fixture laid out exactly as the Three Indexer publishes a ``cad/`` config (MeshFolder
+ ``metadata.parquet``). It proves the producer→consumer dataset contract from the *consumer*
side: SPEC-2 §4.1 (B-Rep bridge), §4.2 (renders/caption), §5 (license cohort), §6.1 (Mode-A
streaming), §10 (acceptance).

There is no runtime dependency on the Three Indexer — the contract crosses as data only. The
``REQUIRED_COLUMNS`` tuple below is the consumer's own copy of the SPEC-2 §3.2 column set; if the
producer ever drops or renames a load-bearing column, this test fails.

Run (from the LatticeLabs_toolkit repo root):
  PYTHONPATH=cadling /Users/ryanoboyle/miniforge3/envs/cadling/bin/python -m pytest \
      cadling/tests/test_indexer_contract_conformance.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "indexer_cad_repo"
META = FIXTURE / "cad" / "metadata.parquet"

# Consumer's copy of the SPEC-2 §3.2 load-bearing columns the producer must emit on every row.
REQUIRED_COLUMNS = (
    "file_name", "asset_id", "source", "source_id", "domain", "license_id",
    "license_family", "requires_attribution", "sha256",
    "original_file_name", "pointcloud_file_name",
    "render_file_names", "thumbnail_file_name", "split",
)
# Clean (redistributable, commercial-OK) families — the complement of SPEC-1's private routing.
PERMISSIVE = {"cc0", "public-domain", "public_domain", "cc-by", "cc-by-4.0", "permissive"}

pytestmark = pytest.mark.skipif(not META.exists(), reason="Three Indexer fixture missing")


def _require_consumer():
    """Import the real cadling consumer stack; skip cleanly if its deps are absent."""
    pytest.importorskip("datasets")
    pytest.importorskip("pyarrow")
    pytest.importorskip("OCC")
    from cadling.data.hf_builders.brep_graph_builder import BRepGraphBuilder
    from cadling.data.schemas import get_brep_graph_schema
    from cadling.data.streaming import CADStreamingConfig, CADStreamingDataset

    return CADStreamingConfig, CADStreamingDataset, BRepGraphBuilder, get_brep_graph_schema


def _load_rows() -> list[dict]:
    """Mode-A: stream the metadata.parquet contract table (the documented consumer path)."""
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files={"train": str(META)}, split="train", streaming=True)
    return list(ds)


def test_required_columns_present_and_lineage_intact() -> None:
    pytest.importorskip("datasets")
    rows = _load_rows()
    assert len(rows) == 1
    row = rows[0]
    missing = [c for c in REQUIRED_COLUMNS if c not in row]
    assert missing == [], f"producer dropped contract column(s): {missing}"
    # SPEC-2 §5 — lineage reconstructable, routing predicate populated.
    assert row["asset_id"] and row["source"] and row["sha256"]
    assert row["license_family"]


def test_turnkey_cadstreamingdataset_consumes_indexer_repo() -> None:
    """SPEC-2 §6.1 / §10.1 — the canonical entry point works against a faithful Indexer repo.

    The published HF card declares ``data_files`` path ``<config>/*.parquet``, so
    ``load_dataset`` (and therefore ``CADStreamingDataset(dataset_id=...)``) streams the
    metadata table directly instead of choking on the GLB/STEP/PNG media. Single-config repo ⇒
    no ``name`` needed; the README ``configs:`` block resolves it.
    """
    from datasets import load_dataset

    CADStreamingConfig, CADStreamingDataset, *_ = _require_consumer()
    root = str(FIXTURE)

    # turnkey datasets path, honoring the repo's own README configs block
    assert len(list(load_dataset(root, name="cad", split="train", streaming=True))) == 1
    # turnkey cadling path — the REAL class, dataset_id only, streamed end-to-end
    samples = list(CADStreamingDataset(
        CADStreamingConfig(dataset_id=root, split="train", streaming=True, shuffle=False)))
    assert len(samples) == 1
    assert samples[0]["asset_id"]  # streamed + transformed through the real consumer


def test_streaming_license_filter_carves_clean_cohort() -> None:
    """SPEC-2 §5 / §10.3 — the REAL CADStreamingDataset, run end-to-end via dataset_id, applies
    its license filter and excludes non-permissive rows (defense-in-depth atop repo routing)."""
    CADStreamingConfig, CADStreamingDataset, *_ = _require_consumer()
    root = str(FIXTURE)

    kept = list(CADStreamingDataset(CADStreamingConfig(
        dataset_id=root, split="train", streaming=True, shuffle=False,
        filters=[("license_family", "==", "public-domain")])))
    assert len(kept) == 1
    assert all(r["license_family"] in PERMISSIVE for r in kept)  # no copyleft / NC leak
    assert kept[0]["asset_id"]  # lineage passthrough survives the transform

    # a non-matching predicate yields an empty (still-clean) cohort — no false positives.
    empty = list(CADStreamingDataset(CADStreamingConfig(
        dataset_id=root, split="train", streaming=True, shuffle=False,
        filters=[("license_family", "==", "odbl")])))
    assert empty == []


def test_raw_step_becomes_brep_graph() -> None:
    """SPEC-2 §4.1 — the raw STEP at ``original_file_name`` feeds the real BRepGraphBuilder."""
    _, _, BRepGraphBuilder, get_brep_graph_schema = _require_consumer()
    row = _load_rows()[0]

    step = FIXTURE / "cad" / row["original_file_name"]
    assert step.exists() and step.suffix == ".step"

    record = BRepGraphBuilder()._process_step_file_pythonocc(step)
    assert record is not None
    assert len(record["faces"]) >= 1 and len(record["edges"]) >= 1  # real B-Rep topology
    assert "edge_index" in record

    # the consumer's GNN schema (what the graph populates) has the expected node/edge fields.
    schema_fields = {f.name for f in get_brep_graph_schema()}
    assert {"face_features", "edge_features", "edge_index", "num_faces"} <= schema_fields


def test_renders_addressable_for_vision_handoff() -> None:
    """SPEC-2 §4.2 — renders + caption resolve from metadata (the ll-ocadr vision handoff)."""
    pytest.importorskip("datasets")
    row = _load_rows()[0]

    renders = row["render_file_names"]
    assert isinstance(renders, list) and renders
    for r in renders:
        assert (FIXTURE / "cad" / r).exists()
    assert (FIXTURE / "cad" / row["thumbnail_file_name"]).exists()
    assert row["caption"] and row["caption"].strip()
