"""Shared fixtures for ll_gen test suite.

Provides reusable test fixtures for:
- Configuration instances at various levels
- Typed proposal instances (Code, Command, Latent)
- DisposalResult instances (valid, invalid, repaired)
- GeometryReport instances with realistic data
- Mock objects for optional dependencies (pythonocc, torch, cadling)

All fixtures are designed to work WITHOUT optional heavy dependencies
(pythonocc, torch, cadquery, etc.) by using mock objects and realistic
synthetic data.
"""
from __future__ import annotations

import copy
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ll_gen.config import (
    CodegenConfig,
    CodeLanguage,
    DatasetConfig,
    DisposalConfig,
    ErrorCategory,
    ErrorSeverity,
    ExportConfig,
    FeedbackConfig,
    GenerationRoute,
    LLGenConfig,
    RoutingConfig,
    StepSchema,
    get_ll_gen_config,
)
from ll_gen.proposals.base import BaseProposal
from ll_gen.proposals.code_proposal import CodeProposal
from ll_gen.proposals.command_proposal import CommandSequenceProposal
from ll_gen.proposals.disposal_result import (
    DisposalResult,
    GeometryReport,
    RepairAction,
    ValidationFinding,
)
from ll_gen.proposals.latent_proposal import LatentProposal


# ---------------------------------------------------------------------------
# Helper: skip if OCC not available
# ---------------------------------------------------------------------------

def _occ_available() -> bool:
    """Check whether pythonocc-core is importable."""
    try:
        from OCC.Core.TopoDS import TopoDS_Shape  # noqa: F401
        return True
    except ImportError:
        return False


requires_occ = pytest.mark.skipif(
    not _occ_available(),
    reason="pythonocc-core not installed",
)


def _torch_available() -> bool:
    """Check whether torch is importable."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


requires_torch = pytest.mark.skipif(
    not _torch_available(),
    reason="torch not installed",
)


def _cadquery_available() -> bool:
    """Check whether cadquery is importable."""
    try:
        import cadquery  # noqa: F401
        return True
    except ImportError:
        return False


requires_cadquery = pytest.mark.skipif(
    not _cadquery_available(),
    reason="cadquery not installed",
)


def _geotoken_available() -> bool:
    """Check whether geotoken is importable."""
    try:
        from geotoken.tokenizer.token_types import TokenSequence  # noqa: F401
        return True
    except ImportError:
        return False


requires_geotoken = pytest.mark.skipif(
    not _geotoken_available(),
    reason="geotoken not installed",
)


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def routing_config() -> RoutingConfig:
    """Default routing config."""
    return RoutingConfig()


@pytest.fixture
def codegen_config() -> CodegenConfig:
    """Default codegen config."""
    return CodegenConfig()


@pytest.fixture
def disposal_config() -> DisposalConfig:
    """Default disposal config."""
    return DisposalConfig()


@pytest.fixture
def export_config() -> ExportConfig:
    """Default export config."""
    return ExportConfig()


@pytest.fixture
def feedback_config() -> FeedbackConfig:
    """Default feedback config."""
    return FeedbackConfig()


@pytest.fixture
def dataset_config() -> DatasetConfig:
    """Default dataset config."""
    return DatasetConfig()


@pytest.fixture
def ll_gen_config() -> LLGenConfig:
    """Top-level config with all defaults."""
    return LLGenConfig()


# ---------------------------------------------------------------------------
# Proposal fixtures
# ---------------------------------------------------------------------------

CADQUERY_BOX_CODE = """\
# Using pre-imported cq (cadquery.Workplane) from sandbox namespace
result = cq("XY").box(100, 50, 20)
result.val()
"""

CADQUERY_BRACKET_CODE = """\
# Using pre-imported cq (cadquery.Workplane) from sandbox namespace
base = cq("XY").box(80, 50, 10)
result = (
    base
    .faces(">Z")
    .workplane()
    .rarray(xSpacing=30, ySpacing=30, nRows=2, nCols=2)
    .hole(4.5)
)
result.val()
"""

CADQUERY_SYNTAX_ERROR_CODE = """\
# Syntax error: missing commas and closing paren
result = cq("XY").box(100 50 20
result.val()
"""

OPENSCAD_BOX_CODE = """\
difference() {
  cube([100, 50, 20], center=true);
  translate([0, 0, 0]) cylinder(h=22, r=5, center=true);
}
"""

OPENSCAD_SYNTAX_ERROR_CODE = """\
difference() {
  cube([100, 50, 20], center=true;
  translate([0, 0, 0]) cylinder(h=22, r=5, center=true);

"""


@pytest.fixture
def base_proposal() -> BaseProposal:
    """Minimal base proposal."""
    return BaseProposal(
        proposal_id="test_base_0001",
        confidence=0.75,
        attempt=1,
        max_attempts=3,
        source_prompt="A simple test shape",
        conditioning_source="text",
    )


@pytest.fixture
def code_proposal_cadquery() -> CodeProposal:
    """Valid CadQuery code proposal."""
    return CodeProposal(
        proposal_id="test_code_cq_01",
        confidence=0.85,
        source_prompt="A box 100mm × 50mm × 20mm",
        conditioning_source="text",
        code=CADQUERY_BOX_CODE,
        language=CodeLanguage.CADQUERY,
    )


@pytest.fixture
def code_proposal_bracket() -> CodeProposal:
    """CadQuery bracket with bolt holes."""
    return CodeProposal(
        proposal_id="test_code_bracket",
        confidence=0.80,
        source_prompt="A mounting bracket 80mm wide with 4 bolt holes",
        conditioning_source="text",
        code=CADQUERY_BRACKET_CODE,
        language=CodeLanguage.CADQUERY,
    )


@pytest.fixture
def code_proposal_syntax_error() -> CodeProposal:
    """CadQuery proposal with a Python syntax error."""
    return CodeProposal(
        proposal_id="test_code_bad_syntax",
        confidence=0.3,
        source_prompt="A box",
        code=CADQUERY_SYNTAX_ERROR_CODE,
        language=CodeLanguage.CADQUERY,
    )


@pytest.fixture
def code_proposal_openscad() -> CodeProposal:
    """Valid OpenSCAD code proposal."""
    return CodeProposal(
        proposal_id="test_code_scad_01",
        confidence=0.80,
        source_prompt="A box with a hole",
        code=OPENSCAD_BOX_CODE,
        language=CodeLanguage.OPENSCAD,
    )


@pytest.fixture
def code_proposal_openscad_bad() -> CodeProposal:
    """OpenSCAD proposal with unbalanced braces."""
    return CodeProposal(
        proposal_id="test_code_scad_bad",
        confidence=0.2,
        source_prompt="A box with a hole",
        code=OPENSCAD_SYNTAX_ERROR_CODE,
        language=CodeLanguage.OPENSCAD,
    )


@pytest.fixture
def command_proposal() -> CommandSequenceProposal:
    """Realistic command sequence proposal with a simple rectangle extrusion.

    Commands:  SOL → LINE × 4 → EXTRUDE → EOS
    Token IDs use the convention: PAD=0, BOS=1, EOS=2, SOL=6, LINE=7,
    ARC=8, CIRCLE=9, EXTRUDE=10, EOS_CMD=11, params≥12.
    """
    # Simple rectangle sketch + extrude: 1 sketch loop, 4 lines, 1 extrude
    command_dicts = [
        {
            "command_type": "SOL",
            "parameters": [0] * 16,
            "parameter_mask": [False] * 16,
        },
        {
            "command_type": "LINE",
            "parameters": [0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "parameter_mask": [True, True, True, True] + [False] * 12,
        },
        {
            "command_type": "LINE",
            "parameters": [128, 0, 128, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "parameter_mask": [True, True, True, True] + [False] * 12,
        },
        {
            "command_type": "LINE",
            "parameters": [128, 128, 0, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "parameter_mask": [True, True, True, True] + [False] * 12,
        },
        {
            "command_type": "LINE",
            "parameters": [0, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "parameter_mask": [True, True, True, True] + [False] * 12,
        },
        {
            "command_type": "EXTRUDE",
            "parameters": [64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "parameter_mask": [True] + [False] * 15,
        },
        {
            "command_type": "EOS",
            "parameters": [0] * 16,
            "parameter_mask": [False] * 16,
        },
    ]

    return CommandSequenceProposal(
        proposal_id="test_cmd_rect_01",
        confidence=0.70,
        source_prompt="A rectangular prism",
        conditioning_source="text",
        command_dicts=command_dicts,
        quantization_bits=8,
        normalization_range=2.0,
        precision_tier="STANDARD",
        latent_vector=np.random.randn(256).astype(np.float32),
    )


@pytest.fixture
def command_proposal_token_ids() -> CommandSequenceProposal:
    """Command proposal using raw token IDs instead of dicts."""
    # BOS, SOL, LINE params, LINE params, EXTRUDE param, EOS
    token_ids = [
        1,   # BOS
        6,   # SOL
        7,   # LINE
        12,  # param: 0
        12,  # param: 0
        140, # param: 128
        12,  # param: 0
        7,   # LINE
        140, # param: 128
        12,  # param: 0
        140, # param: 128
        140, # param: 128
        10,  # EXTRUDE
        76,  # param: 64
        2,   # EOS
    ]
    return CommandSequenceProposal(
        proposal_id="test_cmd_tokens_01",
        confidence=0.65,
        source_prompt="A box from tokens",
        token_ids=token_ids,
        quantization_bits=8,
        normalization_range=2.0,
    )


@pytest.fixture
def latent_proposal() -> LatentProposal:
    """Latent proposal with synthetic face grids and edge points.

    Creates a simple cube-like proposal with 6 faces (32×32 grids)
    and 12 edges (64 points each), plus 8 vertex positions.
    """
    rng = np.random.RandomState(42)
    face_grids = []
    for _ in range(6):
        grid = rng.randn(32, 32, 3).astype(np.float32)
        face_grids.append(grid)

    edge_points = []
    for _ in range(12):
        points = rng.randn(64, 3).astype(np.float32)
        edge_points.append(points)

    vertices = rng.randn(8, 3).astype(np.float32)
    face_bboxes = rng.randn(6, 6).astype(np.float32)
    edge_bboxes = rng.randn(12, 6).astype(np.float32)

    adjacency = {
        0: [0, 1, 2, 3],
        1: [0, 4, 5, 8],
        2: [1, 4, 6, 9],
        3: [2, 5, 7, 10],
        4: [3, 6, 7, 11],
        5: [8, 9, 10, 11],
    }

    return LatentProposal(
        proposal_id="test_latent_cube_01",
        confidence=0.60,
        source_prompt="A smooth organic shape",
        conditioning_source="text",
        face_grids=face_grids,
        edge_points=edge_points,
        face_bboxes=face_bboxes,
        edge_bboxes=edge_bboxes,
        vertex_positions=vertices,
        face_edge_adjacency=adjacency,
    )


@pytest.fixture
def latent_proposal_minimal() -> LatentProposal:
    """Minimal latent proposal — 1 face, no edges."""
    grid = np.zeros((4, 4, 3), dtype=np.float32)
    # Flat square in XY plane
    for u in range(4):
        for v in range(4):
            grid[u, v] = [u * 10.0, v * 10.0, 0.0]

    return LatentProposal(
        proposal_id="test_latent_minimal",
        confidence=0.40,
        source_prompt="A flat square",
        face_grids=[grid],
    )


# ---------------------------------------------------------------------------
# GeometryReport fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def geometry_report_box() -> GeometryReport:
    """GeometryReport for a 100×50×20 box."""
    return GeometryReport(
        volume=100000.0,
        surface_area=2 * (100 * 50 + 100 * 20 + 50 * 20),
        bounding_box=(0.0, 0.0, 0.0, 100.0, 50.0, 20.0),
        center_of_mass=(50.0, 25.0, 10.0),
        face_count=6,
        edge_count=12,
        vertex_count=8,
        shell_count=1,
        solid_count=1,
        surface_types={"Plane": 6},
        curve_types={"Line": 12},
        euler_characteristic=2,
        is_solid=True,
    )


@pytest.fixture
def geometry_report_cylinder() -> GeometryReport:
    """GeometryReport for a cylinder r=10, h=30."""
    import math
    return GeometryReport(
        volume=math.pi * 10 ** 2 * 30,
        surface_area=2 * math.pi * 10 * 30 + 2 * math.pi * 10 ** 2,
        bounding_box=(-10.0, -10.0, 0.0, 10.0, 10.0, 30.0),
        center_of_mass=(0.0, 0.0, 15.0),
        face_count=3,
        edge_count=3,
        vertex_count=2,
        shell_count=1,
        solid_count=1,
        surface_types={"Plane": 2, "Cylinder": 1},
        curve_types={"Circle": 2, "Line": 1},
        euler_characteristic=2,
        is_solid=True,
    )


@pytest.fixture
def geometry_report_no_bbox() -> GeometryReport:
    """GeometryReport with no bounding box (failed introspection)."""
    return GeometryReport(
        volume=None,
        surface_area=None,
        bounding_box=None,
        face_count=0,
        edge_count=0,
        vertex_count=0,
    )


# ---------------------------------------------------------------------------
# DisposalResult fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def disposal_result_valid(geometry_report_box: GeometryReport) -> DisposalResult:
    """Fully valid disposal result with geometry."""
    return DisposalResult(
        shape="<mock_shape>",
        is_valid=True,
        error_details=[],
        geometry_report=geometry_report_box,
        repair_attempted=False,
        repair_succeeded=False,
        reward_signal=1.0,
        execution_time_ms=250.0,
        proposal_id="test_valid_01",
        proposal_type="CodeProposal",
        step_path=Path("/tmp/test.step"),
        stl_path=Path("/tmp/test.stl"),
    )


@pytest.fixture
def disposal_result_invalid() -> DisposalResult:
    """Invalid disposal result with topology errors."""
    findings = [
        ValidationFinding(
            entity_type="SHELL",
            entity_index=0,
            error_code="BRepCheck_NotClosed",
            error_category=ErrorCategory.TOPOLOGY_ERROR,
            severity=ErrorSeverity.CRITICAL,
            description="Shell is not closed (watertightness failure).",
            suggestion="Ensure all sketch loops are closed before extrusion.",
        ),
        ValidationFinding(
            entity_type="EDGE",
            entity_index=3,
            error_code="BRepCheck_FreeEdge",
            error_category=ErrorCategory.DEGENERATE_SHAPE,
            severity=ErrorSeverity.WARNING,
            description="Edge belongs to only one face.",
            suggestion="Close the shell by adding missing faces.",
        ),
    ]
    return DisposalResult(
        shape="<mock_invalid_shape>",
        is_valid=False,
        error_category=ErrorCategory.TOPOLOGY_ERROR,
        error_details=findings,
        geometry_report=GeometryReport(
            volume=None,
            surface_area=500.0,
            bounding_box=(0.0, 0.0, 0.0, 50.0, 50.0, 10.0),
            face_count=5,
            edge_count=11,
            vertex_count=8,
            euler_characteristic=2,
            is_solid=False,
        ),
        repair_attempted=True,
        repair_succeeded=False,
        repair_actions=[
            RepairAction(
                tool="ShapeFix_Shape",
                action="General shape repair",
                status="done",
                tolerance_used=1e-7,
            ),
            RepairAction(
                tool="ShapeFix_Wire",
                action="Fixed 2 wire issues",
                status="done",
                tolerance_used=1e-7,
                entities_affected=2,
            ),
        ],
        reward_signal=0.3,
        error_message="Shell is not closed — watertightness failure.",
        suggestion="Ensure all sketch loops are closed before extrusion.",
        execution_time_ms=450.0,
        proposal_id="test_invalid_01",
        proposal_type="CodeProposal",
    )


@pytest.fixture
def disposal_result_repaired() -> DisposalResult:
    """Disposal result that was invalid but repaired successfully."""
    return DisposalResult(
        shape="<mock_repaired_shape>",
        is_valid=True,
        error_details=[],
        geometry_report=GeometryReport(
            volume=50000.0,
            surface_area=8000.0,
            bounding_box=(0.0, 0.0, 0.0, 50.0, 50.0, 20.0),
            face_count=6,
            edge_count=12,
            vertex_count=8,
            euler_characteristic=2,
            is_solid=True,
        ),
        repair_attempted=True,
        repair_succeeded=True,
        repair_actions=[
            RepairAction(
                tool="ShapeFix_Shape",
                action="General shape repair",
                status="done",
                tolerance_used=1e-7,
            ),
        ],
        reward_signal=0.8,
        execution_time_ms=800.0,
        proposal_id="test_repaired_01",
        proposal_type="CodeProposal",
    )


@pytest.fixture
def disposal_result_no_shape() -> DisposalResult:
    """Disposal result where execution failed entirely."""
    return DisposalResult(
        shape=None,
        is_valid=False,
        error_category=ErrorCategory.INVALID_PARAMS,
        error_details=[],
        geometry_report=None,
        repair_attempted=False,
        repair_succeeded=False,
        reward_signal=0.0,
        error_message="Code execution timed out after 30 seconds.",
        suggestion="Simplify the geometry or reduce iteration count.",
        execution_time_ms=30000.0,
        proposal_id="test_no_shape_01",
        proposal_type="CodeProposal",
    )


@pytest.fixture
def disposal_result_self_intersection() -> DisposalResult:
    """Disposal result with self-intersection errors."""
    findings = [
        ValidationFinding(
            entity_type="WIRE",
            entity_index=0,
            error_code="BRepCheck_SelfIntersectingWire",
            error_category=ErrorCategory.SELF_INTERSECTION,
            severity=ErrorSeverity.CRITICAL,
            description="Wire crosses itself.",
            suggestion="Separate overlapping sketch segments.",
        ),
        ValidationFinding(
            entity_type="FACE",
            entity_index=1,
            error_code="BRepCheck_IntersectingWires",
            error_category=ErrorCategory.SELF_INTERSECTION,
            severity=ErrorSeverity.CRITICAL,
            description="Two wires on the same face intersect.",
            suggestion="Ensure inner wires don't overlap outer boundary.",
        ),
    ]
    return DisposalResult(
        shape="<mock_si_shape>",
        is_valid=False,
        error_category=ErrorCategory.SELF_INTERSECTION,
        error_details=findings,
        geometry_report=GeometryReport(
            face_count=6,
            edge_count=14,
            vertex_count=10,
            euler_characteristic=0,
            is_solid=False,
        ),
        repair_attempted=True,
        repair_succeeded=False,
        reward_signal=0.1,
        execution_time_ms=300.0,
        proposal_id="test_si_01",
        proposal_type="CodeProposal",
    )


# ---------------------------------------------------------------------------
# Temporary directory fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Temporary output directory for test exports."""
    out = tmp_path / "ll_gen_test_output"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Cadling integration fixtures
# ---------------------------------------------------------------------------

def _cadling_available() -> bool:
    """Check whether cadling is importable."""
    try:
        import cadling  # noqa: F401
        return True
    except ImportError:
        return False


requires_cadling = pytest.mark.skipif(
    not _cadling_available(),
    reason="cadling package not installed",
)


@pytest.fixture
def mock_cadquery_generator() -> MagicMock:
    """Mock CadQuery generator for testing without LLM calls."""
    mock_gen = MagicMock()
    mock_gen.generate = MagicMock(return_value="""\
from cadquery import Workplane as cq
result = cq("XY").box(100, 50, 20)
result.val()
""")
    return mock_gen


@pytest.fixture
def mock_openscad_generator() -> MagicMock:
    """Mock OpenSCAD generator for testing without LLM calls."""
    mock_gen = MagicMock()
    mock_gen.generate = MagicMock(return_value="""\
difference() {
  cube([100, 50, 20], center=true);
}
""")
    return mock_gen


# ---------------------------------------------------------------------------
# Neural model mock fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_stepvae() -> MagicMock:
    """Mock STEPVAE model for testing without torch."""
    mock_vae = MagicMock()
    mock_vae.encode = MagicMock(return_value=np.random.randn(1, 256).astype(np.float32))
    mock_vae.decode = MagicMock(return_value=np.random.randn(60, 17).astype(np.float32))
    mock_vae.sample = MagicMock(return_value=np.random.randn(1, 256).astype(np.float32))
    return mock_vae


@pytest.fixture
def mock_vqvae() -> MagicMock:
    """Mock VQ-VAE model for testing without torch."""
    mock_model = MagicMock()
    mock_model.encode = MagicMock(return_value=np.random.randint(0, 512, (60,)))
    mock_model.decode = MagicMock(return_value=np.random.randn(60, 17).astype(np.float32))
    mock_model.quantize = MagicMock(return_value=(
        np.random.randn(1, 256).astype(np.float32),
        np.random.randint(0, 512, (256,)),
    ))
    return mock_model


@pytest.fixture
def mock_diffusion() -> MagicMock:
    """Mock diffusion model for testing without torch."""
    mock_model = MagicMock()
    mock_model.sample = MagicMock(return_value={
        "face_grids": [np.random.randn(32, 32, 3).astype(np.float32) for _ in range(6)],
        "edge_points": [np.random.randn(64, 3).astype(np.float32) for _ in range(12)],
    })
    return mock_model


# ---------------------------------------------------------------------------
# Geotoken mock fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_token_sequence() -> MagicMock:
    """Mock geotoken TokenSequence for testing without geotoken."""
    mock_seq = MagicMock()
    mock_seq.token_ids = [1, 6, 7, 12, 12, 140, 12, 10, 76, 2]  # BOS, SOL, LINE, params, EXTRUDE, param, EOS
    mock_seq.tokens = []
    mock_seq.__len__ = MagicMock(return_value=10)
    return mock_seq


@pytest.fixture
def mock_cad_vocabulary() -> MagicMock:
    """Mock CADVocabulary for testing without geotoken."""
    mock_vocab = MagicMock()
    mock_vocab.pad_token_id = 0
    mock_vocab.bos_token_id = 1
    mock_vocab.eos_token_id = 2
    mock_vocab.encode = MagicMock(return_value=[1, 6, 7, 10, 2])
    mock_vocab.decode = MagicMock(return_value=["BOS", "SOL", "LINE", "EXTRUDE", "EOS"])
    return mock_vocab


# ---------------------------------------------------------------------------
# Test file path fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_step_file(tmp_path: Path) -> Optional[Path]:
    """Create a minimal STEP file for testing (if OCC available).

    Returns None if OCC is not available.
    """
    if not _occ_available():
        return None

    try:
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs

        shape = BRepPrimAPI_MakeBox(100, 50, 20).Shape()
        step_path = tmp_path / "test_sample.step"

        writer = STEPControl_Writer()
        writer.Transfer(shape, STEPControl_AsIs)
        writer.Write(str(step_path))

        return step_path
    except Exception:
        return None


@pytest.fixture
def sample_stl_file(tmp_path: Path) -> Optional[Path]:
    """Create a minimal STL file for testing (if OCC available).

    Returns None if OCC is not available.
    """
    if not _occ_available():
        return None

    try:
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCC.Core.StlAPI import StlAPI_Writer
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh

        shape = BRepPrimAPI_MakeBox(100, 50, 20).Shape()
        stl_path = tmp_path / "test_sample.stl"

        # Mesh the shape
        mesh = BRepMesh_IncrementalMesh(shape, 0.1)
        mesh.Perform()

        writer = StlAPI_Writer()
        writer.Write(shape, str(stl_path))

        return stl_path
    except Exception:
        return None


# ---------------------------------------------------------------------------
# OCC shape fixtures (for integration tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def occ_box_shape():
    """Create a simple OCC box shape for testing.

    Returns None if OCC is not available.
    """
    if not _occ_available():
        return None

    try:
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        return BRepPrimAPI_MakeBox(100, 50, 20).Shape()
    except Exception:
        return None


@pytest.fixture
def occ_cylinder_shape():
    """Create a simple OCC cylinder shape for testing.

    Returns None if OCC is not available.
    """
    if not _occ_available():
        return None

    try:
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
        return BRepPrimAPI_MakeCylinder(10, 30).Shape()
    except Exception:
        return None
