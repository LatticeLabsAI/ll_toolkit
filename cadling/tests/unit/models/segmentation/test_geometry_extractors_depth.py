"""Regression tests for measured (non-fabricated) hole depth.

Guards the H5 fix: ``HoleGeometryExtractor`` previously returned
``depth = diameter * 2.0`` at confidence 0.9 even on the OCC path where the
full cylinder geometry was available — a fabricated measurement dressed as a
computed feature parameter. The OCC path now measures the real depth from the
cylindrical face's axial (V) extent; the text-only fallback flags its estimate.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cadling.models.segmentation.geometry_extractors import (
    ChamferGeometryExtractor,
    HoleGeometryExtractor,
)


def _holed_plate(thickness: float = 10.0, radius: float = 4.0):
    """A `thickness`-thick plate with a coaxial cylindrical through-hole."""
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt

    plate = BRepPrimAPI_MakeBox(
        gp_Pnt(-10, -10, 0), 20.0, 20.0, thickness
    ).Shape()
    drill = BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(0, 0, -1), gp_Dir(0, 0, 1)), radius, thickness + 2.0
    ).Shape()
    return BRepAlgoAPI_Cut(plate, drill).Shape()


def _cylindrical_face_graph(shape):
    """Wrap a shape's faces as a minimal graph with ``_occ_face`` entities."""
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods

    faces = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        faces.append(SimpleNamespace(_occ_face=topods.Face(explorer.Current())))
        explorer.Next()
    return SimpleNamespace(faces=faces)


def _cylindrical_face_ids(graph):
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import GeomAbs_Cylinder

    return [
        i
        for i, fe in enumerate(graph.faces)
        if BRepAdaptor_Surface(fe._occ_face).GetType() == GeomAbs_Cylinder
    ]


@pytest.mark.requires_pythonocc
class TestMeasuredHoleDepth:
    def test_occ_depth_is_measured_not_two_times_diameter(self):
        extractor = HoleGeometryExtractor()
        if not extractor.has_pythonocc:
            pytest.skip("pythonocc not available")

        thickness, radius = 10.0, 4.0
        graph = _cylindrical_face_graph(_holed_plate(thickness, radius))
        cyl_ids = _cylindrical_face_ids(graph)
        assert cyl_ids, "expected a cylindrical hole-wall face"

        result = extractor._extract_from_occ_faces(cyl_ids, graph)
        assert result is not None

        # Diameter is the real cylinder diameter.
        assert result["diameter"] == pytest.approx(2 * radius, abs=1e-6)
        # Depth is the MEASURED plate thickness, NOT the fabricated 2*diameter.
        assert result["depth"] == pytest.approx(thickness, abs=1e-6)
        assert result["depth"] != pytest.approx(2 * (2 * radius), abs=1e-6)
        assert result["depth_estimated"] is False

    def test_depth_scales_with_real_thickness(self):
        """Different plate thicknesses must yield different measured depths
        (a fabricated 2*diameter would be identical for both)."""
        extractor = HoleGeometryExtractor()
        if not extractor.has_pythonocc:
            pytest.skip("pythonocc not available")

        depths = []
        for thickness in (6.0, 18.0):
            graph = _cylindrical_face_graph(_holed_plate(thickness, 4.0))
            cyl_ids = _cylindrical_face_ids(graph)
            depths.append(extractor._extract_from_occ_faces(cyl_ids, graph)["depth"])

        assert depths[0] == pytest.approx(6.0, abs=1e-6)
        assert depths[1] == pytest.approx(18.0, abs=1e-6)
        assert depths[0] != pytest.approx(depths[1])


@pytest.mark.requires_pythonocc
class TestMeasuredChamferDistance:
    """The chamfer OCC path previously hardcoded distance=2.0 at confidence 0.75.
    It must now measure the chamfer face width from real geometry."""

    def _chamfered_box(self, setback: float):
        from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeChamfer
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCC.Core.TopAbs import TopAbs_EDGE
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopoDS import topods

        box = BRepPrimAPI_MakeBox(20.0, 20.0, 20.0).Shape()
        chamfer = BRepFilletAPI_MakeChamfer(box)
        explorer = TopExp_Explorer(box, TopAbs_EDGE)
        chamfer.Add(setback, topods.Edge(explorer.Current()))
        return chamfer.Shape()

    def _chamfer_faces(self, shape):
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.GeomAbs import GeomAbs_Plane
        from OCC.Core.TopAbs import TopAbs_FACE
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopoDS import topods

        faces = []
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            faces.append(SimpleNamespace(_occ_face=topods.Face(explorer.Current())))
            explorer.Next()
        graph = SimpleNamespace(faces=faces)
        # The chamfer face is the tilted (non-axis-aligned) planar face.
        ids = []
        for i, fe in enumerate(faces):
            adaptor = BRepAdaptor_Surface(fe._occ_face)
            if adaptor.GetType() == GeomAbs_Plane:
                n = adaptor.Plane().Axis().Direction()
                v = np.array([n.X(), n.Y(), n.Z()])
                if np.count_nonzero(np.abs(v) > 0.3) >= 2:
                    ids.append(i)
        return graph, ids

    def test_distance_is_measured_and_scales(self):
        extractor = ChamferGeometryExtractor()
        if not extractor.has_pythonocc:
            pytest.skip("pythonocc not available")

        results = {}
        for setback in (2.0, 5.0):
            graph, ids = self._chamfer_faces(self._chamfered_box(setback))
            assert ids, "expected a chamfer face"
            res = extractor._extract_from_occ_geometry(ids, graph)
            assert res is not None
            assert res["distance_measured"] is True
            assert res["distance"] != pytest.approx(2.0)  # not the old default
            results[setback] = res["distance"]

        # A larger setback yields a measurably larger chamfer face width.
        assert results[5.0] > results[2.0] * 1.5


class TestTextFallbackDepthIsFlagged:
    def test_text_path_marks_depth_estimated(self):
        """The text-only fallback cannot measure depth; it must flag the
        estimate rather than present it as measured."""
        extractor = HoleGeometryExtractor()
        # The text parser keys on CYLINDRICAL_SURFACE('',#N,RADIUS).
        entity_text = "ADVANCED_FACE('',(#10),#20,.T.) CYLINDRICAL_SURFACE('',#30,5.0)"
        face_entities = [
            {"entity_id": 1, "entity_type": "ADVANCED_FACE", "text": entity_text}
        ]
        result = extractor._extract_from_step_text(face_entities)
        assert result is not None, "text parser should recover diameter from sample"
        assert result["diameter"] == pytest.approx(10.0, abs=1e-6)
        # Depth is heuristic here (no geometry) -> must be flagged estimated.
        assert result.get("depth_estimated") is True
