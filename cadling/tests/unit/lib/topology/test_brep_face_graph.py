"""Regression tests for the real B-Rep face-adjacency graph builder.

These guard against the previous fabrication, in which three dataset builders
(``data/hf_builders/brep_graph_builder.py``, ``arrow_brep_builder.py`` and
``data/webdataset.py``) connected each face to the next four faces by array
index (``range(i + 1, min(i + 5, num_faces))``) and emitted zeroed
normals/curvatures/convexity/dihedral. ``build_brep_face_graph`` replaces that
with real shared-edge topology (``MapShapesAndAncestors``) and real geometric
features, and all three builders now delegate to it.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from cadling.lib.topology.brep_face_graph import (
    HAS_OCC,
    build_brep_face_graph,
)

# Source files that must never reintroduce the index-order placeholder.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_BUILDER_FILES = [
    _REPO_ROOT / "cadling" / "data" / "hf_builders" / "brep_graph_builder.py",
    _REPO_ROOT / "cadling" / "data" / "hf_builders" / "arrow_brep_builder.py",
    _REPO_ROOT / "cadling" / "data" / "webdataset.py",
]


def _make_box(dx: float = 10.0, dy: float = 20.0, dz: float = 30.0):
    """Build a unit-ish OCC box solid (6 faces, 12 edges, all convex edges)."""
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

    return BRepPrimAPI_MakeBox(dx, dy, dz).Shape()


def _make_notched_solid():
    """Box with a corner octant removed -> introduces concave (reentrant) edges."""
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.gp import gp_Pnt

    big = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 30.0, 30.0, 30.0).Shape()
    notch = BRepPrimAPI_MakeBox(gp_Pnt(15, 15, 15), 30.0, 30.0, 30.0).Shape()
    return BRepAlgoAPI_Cut(big, notch).Shape()


def _make_box_with_through_hole():
    """Plate with a cylindrical through-hole -> curved faces, CONVEX hole rim (90° material)."""
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt

    plate = BRepPrimAPI_MakeBox(gp_Pnt(-10, -10, 0), 20.0, 20.0, 10.0).Shape()
    drill = BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(0, 0, -1), gp_Dir(0, 0, 1)), 4.0, 12.0
    ).Shape()
    return BRepAlgoAPI_Cut(plate, drill).Shape()


def _make_blind_pocket():
    """Plate with a blind cylindrical pocket -> CONCAVE floor ring (curved face)."""
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt

    plate = BRepPrimAPI_MakeBox(gp_Pnt(-10, -10, 0), 20.0, 20.0, 10.0).Shape()
    pocket = BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(0, 0, 5), gp_Dir(0, 0, 1)), 4.0, 10.0
    ).Shape()
    return BRepAlgoAPI_Cut(plate, pocket).Shape()


class TestSourceHasNoFabricatedAdjacency:
    """OCC-free static guards — run everywhere, even without pythonocc."""

    def test_builders_delegate_to_shared_helper(self):
        for path in _BUILDER_FILES:
            text = path.read_text()
            assert "build_brep_face_graph" in text, (
                f"{path.name} no longer delegates to the shared face-graph helper"
            )

    def test_no_index_order_placeholder_remains(self):
        # The fabrication was `range(i + 1, min(i + 5, num_faces))`. Match it
        # tolerant to whitespace so a reformat can't smuggle it back in.
        import re

        pattern = re.compile(r"min\(\s*i\s*\+\s*5\s*,")
        for path in _BUILDER_FILES:
            assert not pattern.search(path.read_text()), (
                f"{path.name} reintroduced index-order face adjacency (min(i+5, ...))"
            )


def test_null_shape_raises():
    if not HAS_OCC:
        pytest.skip("pythonocc not available")
    from OCC.Core.TopoDS import TopoDS_Shape

    with pytest.raises(ValueError):
        build_brep_face_graph(TopoDS_Shape())  # default-constructed -> IsNull()


@pytest.mark.requires_pythonocc
class TestBoxTopology:
    """A box is the canonical check: known faces, edges, and adjacency."""

    def setup_method(self):
        if not HAS_OCC:
            pytest.skip("pythonocc not available")
        self.graph = build_brep_face_graph(_make_box())

    def test_schema_keys(self):
        g = self.graph
        assert set(g) == {"faces", "edges", "edge_index"}
        face = g["faces"][0]
        assert set(face) == {"idx", "surface_type", "area", "centroid", "normal", "curvatures"}
        edge = g["edges"][0]
        assert set(edge) == {"idx", "curve_type", "length", "convexity", "dihedral_angle"}

    def test_counts(self):
        assert len(self.graph["faces"]) == 6
        assert len(self.graph["edges"]) == 12

    def test_adjacency_is_real_box_topology(self):
        src, dst = self.graph["edge_index"]
        pairs = set(zip(src, dst))

        # 12 undirected shared edges -> 24 directed entries.
        assert len(pairs) == 24
        # Bidirectional.
        assert all((b, a) in pairs for (a, b) in pairs)
        # No self-loops.
        assert not [p for p in pairs if p[0] == p[1]]
        # Each face shares an edge with exactly 4 others (all but its opposite).
        degree = Counter(a for a, _ in pairs)
        assert set(degree.values()) == {4}
        assert len(degree) == 6

    def test_not_the_old_index_scheme(self):
        """Regression: the result must differ from the fabricated index adjacency."""
        old_pairs = set()
        for i in range(6):
            for j in range(i + 1, min(i + 5, 6)):
                old_pairs.add((i, j))
                old_pairs.add((j, i))
        new_pairs = set(zip(*self.graph["edge_index"]))
        assert new_pairs != old_pairs

    def test_face_normals_are_real(self):
        normals = [tuple(np.round(f["normal"], 3)) for f in self.graph["faces"]]
        # A box has 6 distinct outward normals; the old placeholder was [0,0,1] x6.
        assert len(set(normals)) == 6
        # Each is a unit vector.
        for f in self.graph["faces"]:
            assert np.isclose(np.linalg.norm(f["normal"]), 1.0, atol=1e-6)

    def test_box_edges_are_convex_with_right_angle(self):
        for e in self.graph["edges"]:
            assert e["convexity"] == 1.0  # every box edge is convex
            assert np.isclose(e["dihedral_angle"], np.pi / 2, atol=1e-3)
            assert e["length"] > 0.0


@pytest.mark.requires_pythonocc
class TestConcaveDetection:
    def test_notched_solid_has_concave_edges(self):
        if not HAS_OCC:
            pytest.skip("pythonocc not available")
        graph = build_brep_face_graph(_make_notched_solid())
        convex = [e["convexity"] for e in graph["edges"]]

        # Removing a corner octant creates exactly 3 reentrant (concave) edges
        # alongside the convex ones — the signed test must distinguish them.
        assert convex.count(0.0) == 3, "expected 3 concave edges on a corner notch"
        assert 1.0 in convex, "expected convex edges to remain"

        # Adjacency still well-formed.
        pairs = set(zip(*graph["edge_index"]))
        assert all((b, a) in pairs for (a, b) in pairs)
        assert not [p for p in pairs if p[0] == p[1]]


@pytest.mark.requires_pythonocc
class TestCurvedFaceConvexity:
    """Curved faces (holes, pockets) are the dominant CAD case — must be correct,
    not just the planar box. A normal-dot test or a face-centroid test both fail
    here; these guard the signed coedge implementation."""

    def setup_method(self):
        if not HAS_OCC:
            pytest.skip("pythonocc not available")

    def test_through_hole_rim_is_convex(self):
        graph = build_brep_face_graph(_make_box_with_through_hole())
        # There must be a cylindrical face and circular rim edges.
        assert any(f["surface_type"] == "cylinder" for f in graph["faces"])
        rim = [e for e in graph["edges"] if e["curve_type"] == "circle"]
        assert rim, "expected circular rim edges"
        # A through-hole rim is a 90° material edge => convex, NOT concave.
        # (This is exactly the case where the earlier centroid heuristic returned
        # a wrong/neutral value.)
        assert all(e["convexity"] == 1.0 for e in rim), (
            f"hole rim must be convex, got {[e['convexity'] for e in rim]}"
        )
        # No concave edges anywhere on a simple through-hole plate.
        assert 0.0 not in [e["convexity"] for e in graph["edges"]]

    def test_blind_pocket_floor_is_concave(self):
        graph = build_brep_face_graph(_make_blind_pocket())
        convex = [e["convexity"] for e in graph["edges"]]
        # The ring where the pocket floor meets the cylindrical wall is concave.
        assert 0.0 in convex, "expected a concave floor edge in a blind pocket"
        assert 1.0 in convex, "expected convex outer edges to remain"

    def test_cylinder_caps_are_convex(self):
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder

        graph = build_brep_face_graph(BRepPrimAPI_MakeCylinder(5.0, 20.0).Shape())
        circles = [e for e in graph["edges"] if e["curve_type"] == "circle"]
        # The two cap<->lateral circular edges are convex.
        assert circles and all(e["convexity"] == 1.0 for e in circles)
        # Dihedral at a cylinder cap is a right angle.
        for e in circles:
            assert np.isclose(e["dihedral_angle"], np.pi / 2, atol=1e-3)
