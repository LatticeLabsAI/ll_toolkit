"""Tests for `cadling hub build`, including the ``--type brep_graphs`` route.

Regression guard: the ``build`` command previously hardcoded
``CADCommandSequenceBuilder`` and never routed ``--type brep_graphs`` to
``BRepGraphBuilder``, so real B-Rep graph datasets could not be built from the
CLI at all. These tests pin the wiring.
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest
from click.testing import CliRunner

from cadling.cli.hub import build, hub


def _type_option_choices():
    for param in build.params:
        if getattr(param, "name", None) == "dataset_type":
            return list(param.type.choices)
    return None


class TestBuildTypeOption:
    """OCC-free: the option must exist and offer the brep_graphs route."""

    def test_build_exposes_dataset_type_option(self):
        choices = _type_option_choices()
        assert choices is not None, "`build` is missing the --type/dataset_type option"
        assert "brep_graphs" in choices
        assert "command_sequences" in choices

    def test_brep_graphs_routes_to_brep_builder(self, monkeypatch):
        """`--type brep_graphs` must construct BRepGraphBuilder, not the command builder."""
        import cadling.data.hf_builders.brep_graph_builder as bg

        constructed = {}

        class _FakeBuilder:
            def __init__(self, source_dir=None, config=None, splits=None, **kw):
                constructed["cls"] = "BRepGraphBuilder"
                constructed["config"] = config

            def build(self, output, compression="zstd"):
                return {}

        monkeypatch.setattr(bg, "BRepGraphBuilder", _FakeBuilder)

        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as out:
            res = CliRunner().invoke(
                hub,
                ["build", src, "-o", out, "--type", "brep_graphs", "--splits", "train"],
            )
        assert res.exit_code == 0, res.output
        assert constructed.get("cls") == "BRepGraphBuilder"


@pytest.mark.requires_pythonocc
class TestBuildBrepGraphsEndToEnd:
    def test_build_writes_real_graph_parquet(self):
        try:
            from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
            from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
        except ImportError:
            pytest.skip("pythonocc not available")
        pa_pq = pytest.importorskip("pyarrow.parquet")

        with tempfile.TemporaryDirectory() as src_s, tempfile.TemporaryDirectory() as out_s:
            src = pathlib.Path(src_s)
            out = pathlib.Path(out_s)
            (src / "train").mkdir()
            writer = STEPControl_Writer()
            writer.Transfer(BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape(), STEPControl_AsIs)
            writer.Write(str(src / "train" / "box.step"))

            res = CliRunner().invoke(
                hub,
                ["build", str(src), "-o", str(out), "--type", "brep_graphs", "--splits", "train"],
            )
            assert res.exit_code == 0, res.output

            parquet = out / "train.parquet"
            assert parquet.exists(), "brep_graphs build produced no parquet"

            rows = pa_pq.read_table(parquet).to_pylist()
            assert len(rows) == 1
            row = rows[0]
            assert row["num_faces"] == 6  # a box
            # Real box adjacency: 12 undirected shared edges -> 24 directed ->
            # flattened src+dst = 48 entries. The old index placeholder would not
            # yield the box's exact degree-4 topology.
            assert len(row["edge_index"]) == 48
