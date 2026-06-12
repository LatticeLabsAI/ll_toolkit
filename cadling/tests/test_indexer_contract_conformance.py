"""Conformance: the LatticeLabs Toolkit consumes a Three Indexer (SPEC-2) dataset.

Drives the REAL toolkit consumers against a faithful, multi-config fixture laid out exactly as
the Three Indexer publishes a repo (MeshFolder + per-config ``metadata.parquet``, card
``data_files`` = ``<config>/*.parquet``), proving every producer→consumer bridge from the
consumer side with no runtime dependency on the Indexer:

- **turnkey + multi-config** (§6.1 / §10.1): ``CADStreamingDataset(dataset_id=<repo>,
  config_name=<config>)`` streams each of "3d"/"cad"/"geo" directly.
- **license cohort** (§5): the streaming filter excludes non-permissive rows.
- **cad B-Rep bridge** (§4.1): ``BRepGraphBuilder`` reads the raw STEP (``original_file_name``).
- **mesh bridge** (§4.3): the GLB (``file_name``) feeds ``ll_clouds.io.sample_from_mesh`` and
  ``geotoken.GeoTokenizer.tokenize``.
- **point-cloud bridge** (§4.4): the COPC/LAZ (``file_name``) feeds ``ll_clouds.io.read_point_cloud``.

The ``REQUIRED_COLUMNS`` tuple is the consumer's own copy of SPEC-2 §3.2; a dropped/renamed
producer column fails this test.

Run (from the LatticeLabs_toolkit repo root):
  PYTHONPATH=cadling:ll_clouds:geotoken \
    /Users/ryanoboyle/miniforge3/envs/cadling/bin/python -m pytest \
      cadling/tests/test_indexer_contract_conformance.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "indexer_repo"
CONFIGS = ("3d", "cad", "geo")

# Consumer's copy of the SPEC-2 §3.2 load-bearing columns the producer must emit on every row.
REQUIRED_COLUMNS = (
    "file_name", "asset_id", "source", "source_id", "domain", "license_id",
    "license_family", "requires_attribution", "sha256",
    "original_file_name", "pointcloud_file_name",
    "render_file_names", "thumbnail_file_name",
    "render_front", "render_top", "render_iso", "split",
)
PERMISSIVE = {"cc0", "public-domain", "public_domain", "cc-by", "cc-by-4.0", "permissive"}

pytestmark = pytest.mark.skipif(
    not (FIXTURE / "cad" / "metadata.parquet").exists(), reason="Three Indexer fixture missing"
)


def _meta(config: str) -> Path:
    return FIXTURE / config / "metadata.parquet"


def _rows(config: str) -> list[dict]:
    """Mode-A: stream a config's metadata.parquet (the documented consumer path)."""
    from datasets import load_dataset

    return list(load_dataset("parquet", data_files={"train": str(_meta(config))},
                             split="train", streaming=True))


def _streaming():
    pytest.importorskip("datasets")
    pytest.importorskip("pyarrow")
    from cadling.data.streaming import CADStreamingConfig, CADStreamingDataset

    return CADStreamingConfig, CADStreamingDataset


# --------------------------------------------------------------------------- contract columns
def test_required_columns_present_all_configs() -> None:
    pytest.importorskip("datasets")
    for config in CONFIGS:
        rows = _rows(config)
        assert rows, f"{config}: no rows"
        for row in rows:
            missing = [c for c in REQUIRED_COLUMNS if c not in row]
            assert missing == [], f"{config}: producer dropped column(s): {missing}"
            assert row["asset_id"] and row["source"] and row["sha256"]  # §5 lineage
            assert row["license_family"]
            assert row["domain"] == ({"3d": "mesh"}.get(config, config))


# --------------------------------------------------------------------------- turnkey + config_name
def test_turnkey_multiconfig_selection() -> None:
    """SPEC-2 §6.1 / §10.1 — CADStreamingDataset(dataset_id=<repo>, config_name=<c>) selects the
    right config from a multi-config Indexer repo, end-to-end through the real class."""
    from datasets import load_dataset

    CADStreamingConfig, CADStreamingDataset = _streaming()
    root = str(FIXTURE)
    for config in CONFIGS:
        # turnkey datasets path with explicit config name
        assert len(list(load_dataset(root, name=config, split="train", streaming=True))) >= 1
        # turnkey cadling path — the real class, dataset_id + config_name only
        samples = list(CADStreamingDataset(CADStreamingConfig(
            dataset_id=root, config_name=config, split="train", streaming=True, shuffle=False)))
        assert len(samples) >= 1
        assert samples[0]["asset_id"]


def test_streaming_license_filter_carves_clean_cohort() -> None:
    """SPEC-2 §5 / §10.3 — the real CADStreamingDataset filter excludes non-permissive rows."""
    CADStreamingConfig, CADStreamingDataset = _streaming()
    root = str(FIXTURE)

    kept = list(CADStreamingDataset(CADStreamingConfig(
        dataset_id=root, config_name="cad", split="train", streaming=True, shuffle=False,
        filters=[("license_family", "==", "public-domain")])))
    assert len(kept) == 1 and all(r["license_family"] in PERMISSIVE for r in kept)
    assert kept[0]["asset_id"]

    empty = list(CADStreamingDataset(CADStreamingConfig(
        dataset_id=root, config_name="cad", split="train", streaming=True, shuffle=False,
        filters=[("license_family", "==", "odbl")])))
    assert empty == []


def test_license_tier_configs_select_subsets() -> None:
    """SPEC-2 §7 / OQ-2 (E3) — per-license-tier HF configs (cad-public-domain / cad-permissive)
    select the right subset of the umbrella config, via the real CADStreamingDataset."""
    from datasets import load_dataset

    CADStreamingConfig, CADStreamingDataset = _streaming()
    root = str(FIXTURE)

    # umbrella "cad" = both tiers (public-domain NIST + permissive MIT)
    assert len(list(load_dataset(root, name="cad", split="train", streaming=True))) == 2

    pd_rows = list(CADStreamingDataset(CADStreamingConfig(
        dataset_id=root, config_name="cad-public-domain", split="train", streaming=True, shuffle=False)))
    assert len(pd_rows) == 1 and pd_rows[0]["license_family"] == "public-domain"

    perm_rows = list(CADStreamingDataset(CADStreamingConfig(
        dataset_id=root, config_name="cad-permissive", split="train", streaming=True, shuffle=False)))
    assert len(perm_rows) == 1 and perm_rows[0]["license_family"] == "mit"
    assert perm_rows[0]["requires_attribution"] is True  # permissive tier carries attribution


# --------------------------------------------------------------------------- cad B-Rep bridge
def test_cad_bridge_raw_step_becomes_brep_graph() -> None:
    """SPEC-2 §4.1 — the raw STEP at original_file_name feeds the real BRepGraphBuilder."""
    pytest.importorskip("datasets")
    pytest.importorskip("OCC")
    from cadling.data.hf_builders.brep_graph_builder import BRepGraphBuilder
    from cadling.data.schemas import get_brep_graph_schema

    row = _rows("cad")[0]
    step = FIXTURE / "cad" / row["original_file_name"]
    assert step.exists() and step.suffix == ".step"

    record = BRepGraphBuilder()._process_step_file_pythonocc(step)
    assert record is not None
    assert len(record["faces"]) >= 1 and len(record["edges"]) >= 1 and "edge_index" in record

    schema_fields = {f.name for f in get_brep_graph_schema()}
    assert {"face_features", "edge_features", "edge_index", "num_faces"} <= schema_fields


# --------------------------------------------------------------------------- mesh (3d) bridge
def test_mesh_bridge_glb_to_points_and_tokens() -> None:
    """SPEC-2 §4.3 — the 3d GLB (file_name) feeds ll_clouds.sample_from_mesh + geotoken."""
    pytest.importorskip("datasets")
    pytest.importorskip("trimesh")
    io = pytest.importorskip("ll_clouds.io")
    geotoken = pytest.importorskip("geotoken")
    import numpy as np
    import trimesh

    row = _rows("3d")[0]
    glb = FIXTURE / "3d" / row["file_name"]
    assert glb.exists() and glb.suffix == ".glb"

    # ll_clouds: GLB -> sampled point cloud
    pc = io.sample_from_mesh(str(glb), 256)
    assert pc.num_points == 256 and pc.points.shape == (256, 3)

    # geotoken: GLB -> (vertices, faces) -> token sequence
    mesh = trimesh.load(str(glb), force="mesh")
    tokens = geotoken.GeoTokenizer().tokenize(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    assert tokens is not None and type(tokens).__name__ == "TokenSequence"


# --------------------------------------------------------------------------- geo point-cloud bridge
def test_geo_bridge_copc_to_point_cloud() -> None:
    """SPEC-2 §4.4 — the geo COPC/LAZ (file_name) feeds ll_clouds.io.read_point_cloud.

    Exercises the LAS/LAZ reader added to ll_clouds.io so the toolkit can consume the Three
    Indexer's geospatial point-cloud artifacts directly (laspy[lazrs] backend)."""
    pytest.importorskip("datasets")
    pytest.importorskip("laspy")
    io = pytest.importorskip("ll_clouds.io")

    row = _rows("geo")[0]
    assert row["crs"] and row["point_count"] == 512
    cloud = FIXTURE / "geo" / row["file_name"]
    assert cloud.exists() and cloud.name.endswith(".copc.laz")

    pc = io.read_point_cloud(str(cloud))
    assert pc.num_points == 512 and pc.points.shape == (512, 3)


# --------------------------------------------------------------------------- renders / vision
def test_renders_addressable_for_vision_handoff() -> None:
    """SPEC-2 §4.2 — cad + 3d renders + caption resolve from metadata; geo has none."""
    pytest.importorskip("datasets")
    for config in ("cad", "3d"):
        row = _rows(config)[0]
        renders = row["render_file_names"]
        assert isinstance(renders, list) and renders            # turntable bag (§4.2 E1)
        for r in renders:
            assert (FIXTURE / config / r).exists()
        assert (FIXTURE / config / row["thumbnail_file_name"]).exists()
        # canonical fixed views (OQ-3) — render_front/top/iso resolve to real files
        for col in ("render_front", "render_top", "render_iso"):
            assert row[col], f"{config}: missing {col}"
            assert (FIXTURE / config / row[col]).exists()
        assert row["caption"] and row["caption"].strip()
    # geo point clouds aren't rendered — turntable empty, canonical views null.
    geo = _rows("geo")[0]
    assert geo["render_file_names"] == []
    assert geo["render_front"] is None and geo["render_top"] is None and geo["render_iso"] is None
