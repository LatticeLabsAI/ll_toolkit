"""Unit tests for ll_ocadr content chunkers + mesh loader — SPEC-1 M4, T4.3.

Covers the parts that run without conda-only deps (pythonocc/open3d):
- STL ASCII-vs-binary detection (the known 'solid'-prefix pitfall) + chunking.
- OBJ directive-level chunking (referenced-vertex inclusion behaviour).
- CADLoader / MeshData on a real trimesh STL.
STEP topology extraction is gated behind pythonocc and skips when it's absent.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ll_ocadr.vllm.process.file_content_chunker import (
    OBJContentChunker,
    STLContentChunker,
)

# Repo-root part.step shipped with the project (used by the STEP test).
_PART_STEP = Path(__file__).resolve().parents[3] / "part.step"


def _occ_available() -> bool:
    try:
        from OCC.Core.STEPControl import STEPControl_Reader  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.unit
class TestSTLContentChunker:
    def test_binary_stl_detected_as_binary(self, tiny_stl_file) -> None:
        """trimesh exports binary STL; the size heuristic must NOT call it ASCII
        even though binary headers can start with the bytes 'solid'."""
        chunker = STLContentChunker()
        assert chunker.is_ascii_stl(tiny_stl_file) is False

    def test_ascii_stl_detected_as_ascii(self, tmp_path) -> None:
        ascii_stl = tmp_path / "tri.stl"
        ascii_stl.write_text(
            "solid tri\n"
            "  facet normal 0 0 1\n"
            "    outer loop\n"
            "      vertex 0 0 0\n"
            "      vertex 1 0 0\n"
            "      vertex 0 1 0\n"
            "    endloop\n"
            "  endfacet\n"
            "endsolid tri\n"
        )
        chunker = STLContentChunker()
        assert chunker.is_ascii_stl(str(ascii_stl)) is True

    def test_chunk_stl_returns_facet_chunks(self, tiny_stl_file) -> None:
        chunker = STLContentChunker()
        chunks = chunker.chunk_stl(tiny_stl_file)
        assert isinstance(chunks, list) and len(chunks) > 0
        assert all(isinstance(c, dict) for c in chunks)
        assert all(c.get("format") == "binary_stl" for c in chunks)

    def test_chunk_ascii_stl_path(self, tmp_path) -> None:
        """Exercise the ASCII chunking logic (not just detection): chunk_stl
        auto-routes an ASCII file to chunk_ascii_stl, yielding ascii_stl chunks
        with parsed facets and non-empty raw content."""
        ascii_stl = tmp_path / "two_tri.stl"
        facet = (
            "  facet normal 0 0 1\n"
            "    outer loop\n"
            "      vertex 0 0 0\n"
            "      vertex 1 0 0\n"
            "      vertex 0 1 0\n"
            "    endloop\n"
            "  endfacet\n"
        )
        ascii_stl.write_text("solid tri\n" + facet + facet + "endsolid tri\n")

        chunks = STLContentChunker(chunk_size=1).chunk_stl(str(ascii_stl))
        assert isinstance(chunks, list) and len(chunks) == 2  # two facets, 1/chunk
        for chunk in chunks:
            assert chunk["format"] == "ascii_stl"
            assert chunk["raw_content"].strip()  # non-empty text
            assert len(chunk["facets"]) == 1
            assert "vertex" in chunk["raw_content"]


@pytest.mark.unit
class TestOBJContentChunker:
    def test_chunk_obj_reindexes_only_referenced_vertices(self, tmp_path) -> None:
        """chunk_obj emits one chunk per ``chunk_size`` faces, each carrying ONLY
        the vertices its faces reference (reindexed) — not the full vertex list.

        A unit quad has 4 vertices; each triangular face references 3 of them.
        With chunk_size=1 (one face per chunk) every chunk must report 3
        vertices, proving the per-chunk reindexing rather than dumping all 4.
        """
        obj = tmp_path / "quad.obj"
        obj.write_text(
            "v 0 0 0\n"
            "v 1 0 0\n"
            "v 1 1 0\n"
            "v 0 1 0\n"
            "vn 0 0 1\n"
            "f 1//1 2//1 3//1\n"
            "f 1//1 3//1 4//1\n"
        )
        chunks = OBJContentChunker(chunk_size=1).chunk_obj(str(obj))
        assert isinstance(chunks, list) and len(chunks) == 2
        assert all(c.get("format") == "obj" for c in chunks)
        # Reindexing: each single-face chunk references exactly 3 vertices.
        assert all(c["num_vertices"] == 3 for c in chunks)
        # The reindexed payload re-emits the referenced 'v '/'f ' directives.
        for c in chunks:
            content = c.get("raw_content", "")
            assert "f " in content and "v " in content


@pytest.mark.requires_trimesh
@pytest.mark.unit
class TestCADLoaderMesh:
    def test_load_stl_returns_meshdata(self, tiny_stl_file) -> None:
        from ll_ocadr.vllm.process.mesh_process import CADLoader, MeshData

        data = CADLoader().load(tiny_stl_file)
        assert isinstance(data, MeshData)
        # A trimesh box: 8 vertices, 12 triangular faces.
        assert data.num_vertices == 8
        assert data.num_faces == 12
        assert data.vertices.shape == (8, 3)
        assert data.normals.shape[0] == data.num_vertices
        assert data.bbox_volume > 0.0

    def test_meshloader_alias_is_cadloader(self) -> None:
        from ll_ocadr.vllm.process.mesh_process import CADLoader, MeshLoader

        assert MeshLoader is CADLoader

    def test_unsupported_extension_raises(self) -> None:
        from ll_ocadr.vllm.process.mesh_process import CADLoader

        with pytest.raises(ValueError, match="Unsupported file format"):
            CADLoader().load("model.unknownext")


@pytest.mark.requires_pythonocc
@pytest.mark.skipif(not _occ_available(), reason="pythonocc-core not installed (conda-only)")
@pytest.mark.skipif(not _PART_STEP.exists(), reason="part.step fixture not found")
class TestSTEPProcessor:
    def test_step_topology_extraction(self) -> None:
        from ll_ocadr.vllm.process.step_process import STEPProcessor

        proc = STEPProcessor()
        assert proc.validate_step_file(str(_PART_STEP)) is True
        # load_step_file tessellates the B-Rep to (vertices, faces, normals, bbox).
        vertices, faces, normals, bbox = proc.load_step_file(str(_PART_STEP))
        assert vertices.shape[1] == 3 and len(vertices) > 0
        assert faces.shape[1] == 3 and len(faces) > 0
        assert normals.shape == vertices.shape
        bbox_min, bbox_max = bbox
        assert (bbox_max >= bbox_min).all()
