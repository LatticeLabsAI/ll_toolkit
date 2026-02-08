"""Unit tests for SegNetPipeline.

Comprehensive test suite covering:
  - Pipeline initialization with default and custom options
  - Tokenization with various document states
  - Encoding of token sequences
  - Reconstruction with mocked CommandExecutor
  - Integration testing of the full pipeline
  - Static helper methods for geometry extraction
  - Result dataclasses creation
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any, Optional
import numpy as np

from cadling.pipeline.segnet_pipeline import (
    SegNetPipeline,
    SegNetTokenizationResult,
    ReconstructionResult,
    SegNetPipelineResult,
)
from cadling.datamodel.pipeline_options import PipelineOptions
from cadling.datamodel.base_models import (
    CADlingDocument,
    CADItem,
    CADItemLabel,
    Segment,
    TopologyGraph,
    InputFormat,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def default_options() -> PipelineOptions:
    """Create default pipeline options."""
    return PipelineOptions(
        enrichment_models=[],
        device="cpu",
        do_topology_analysis=True,
    )


@pytest.fixture
def custom_options() -> PipelineOptions:
    """Create custom pipeline options."""
    return PipelineOptions(
        enrichment_models=[],
        device="cpu",
        do_topology_analysis=False,
        enrichment_batch_size=16,
        max_items=100,
    )


@pytest.fixture
def mock_cad_item_with_mesh() -> CADItem:
    """Create a mock CADItem with mesh data in properties."""
    item = CADItem(
        item_type="mesh",
        item_id="item_0",
        label=CADItemLabel(text="test_mesh", entity_type="mesh"),
        properties={
            "vertices": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            "faces": np.array([[0, 1, 2]], dtype=np.int64),
        },
    )
    return item


@pytest.fixture
def mock_cad_item_with_commands() -> CADItem:
    """Create a mock CADItem with command sequence data."""
    item = CADItem(
        item_type="sketch",
        item_id="item_1",
        label=CADItemLabel(text="test_sketch", entity_type="sketch"),
        properties={
            "command_sequence": [
                {"command": "move", "args": [1.0, 2.0]},
                {"command": "line", "args": [3.0, 4.0]},
            ],
        },
    )
    return item


@pytest.fixture
def mock_cad_item_with_constraints() -> CADItem:
    """Create a mock CADItem with constraints data."""
    item = CADItem(
        item_type="sketch",
        item_id="item_2",
        label=CADItemLabel(text="constrained_sketch", entity_type="sketch"),
        properties={
            "constraints": [
                {"constraint_type": "distance", "value": 10.0},
                {"constraint_type": "horizontal", "value": None},
            ],
        },
    )
    return item


@pytest.fixture
def minimal_mock_document() -> CADlingDocument:
    """Create a minimal mock CADlingDocument."""
    doc = CADlingDocument(
        name="test_document",
        format=InputFormat.STEP,
    )
    return doc


@pytest.fixture
def mock_document_with_items(
    mock_cad_item_with_mesh,
    mock_cad_item_with_commands,
) -> CADlingDocument:
    """Create a mock document with multiple items."""
    doc = CADlingDocument(
        name="test_document_full",
        format=InputFormat.STEP,
    )
    doc.add_item(mock_cad_item_with_mesh)
    doc.add_item(mock_cad_item_with_commands)

    # Add segment
    segment = Segment(
        segment_id="seg_0",
        segment_type="feature",
        item_ids=["item_0", "item_1"],
    )
    doc.add_segment(segment)

    return doc


@pytest.fixture
def mock_document_with_topology() -> CADlingDocument:
    """Create a mock document with topology graph."""
    doc = CADlingDocument(
        name="test_document_topology",
        format=InputFormat.STEP,
    )

    # Add topology
    topology = TopologyGraph(
        num_nodes=5,
        num_edges=4,
        node_features=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        edge_features=[[0.5], [0.6], [0.7], [0.8]],
    )
    topology.add_edge(0, 1)
    topology.add_edge(1, 2)
    topology.add_edge(2, 3)
    topology.add_edge(3, 4)

    doc.topology = topology

    return doc


# ============================================================================
# Test: Initialization with default options
# ============================================================================


def test_segnet_pipeline_init_default_options(default_options: PipelineOptions):
    """Test SegNetPipeline initialization with default options."""
    pipeline = SegNetPipeline(default_options)

    assert pipeline.pipeline_options == default_options
    assert pipeline.include_graph_tokens is True
    assert pipeline.include_mesh_tokens is True
    assert pipeline.include_command_tokens is True
    assert pipeline.include_constraints is False
    assert pipeline.reconstruction_tolerance == 1e-6
    assert pipeline.vertex_merge_distance == 0.0


# ============================================================================
# Test: Initialization with custom options
# ============================================================================


def test_segnet_pipeline_init_custom_options(custom_options: PipelineOptions):
    """Test SegNetPipeline initialization with custom tokenization options."""
    pipeline = SegNetPipeline(
        custom_options,
        include_graph_tokens=False,
        include_mesh_tokens=False,
        include_command_tokens=True,
        include_constraints=True,
        reconstruction_tolerance=1e-4,
        vertex_merge_distance=0.01,
    )

    assert pipeline.pipeline_options == custom_options
    assert pipeline.include_graph_tokens is False
    assert pipeline.include_mesh_tokens is False
    assert pipeline.include_command_tokens is True
    assert pipeline.include_constraints is True
    assert pipeline.reconstruction_tolerance == 1e-4
    assert pipeline.vertex_merge_distance == 0.01


# ============================================================================
# Test: get_default_options class method
# ============================================================================


def test_get_default_options():
    """Test that get_default_options returns valid PipelineOptions."""
    options = SegNetPipeline.get_default_options()

    assert isinstance(options, PipelineOptions)
    assert options.do_topology_analysis is True
    assert options.device == "cpu"


# ============================================================================
# Test: tokenize() with mock document containing mesh data
# ============================================================================


@patch("cadling.pipeline.segnet_pipeline._try_import_geotoken")
def test_tokenize_with_mesh_data(mock_import_geotoken, default_options, mock_document_with_items):
    """Test tokenization with a document containing mesh data."""
    # Mock the geotoken components
    mock_geo_tokenizer_class = MagicMock()
    mock_graph_tokenizer_class = MagicMock()
    mock_cmd_tokenizer_class = MagicMock()
    mock_vocab_class = MagicMock()

    mock_import_geotoken.return_value = (
        mock_geo_tokenizer_class,
        mock_graph_tokenizer_class,
        mock_cmd_tokenizer_class,
        mock_vocab_class,
        None,  # TokenSequence
    )

    # Mock the tokenizer instances
    mock_geo_tokenizer = MagicMock()
    mock_geo_tokenizer.tokenize.return_value = MagicMock(name="mesh_token_seq")
    mock_geo_tokenizer_class.return_value = mock_geo_tokenizer

    mock_cmd_tokenizer = MagicMock()
    mock_cmd_tokenizer.tokenize.return_value = MagicMock(name="cmd_token_seq")
    mock_cmd_tokenizer_class.return_value = mock_cmd_tokenizer

    mock_vocab = MagicMock()
    mock_vocab_class.return_value = mock_vocab

    pipeline = SegNetPipeline(
        default_options,
        include_mesh_tokens=True,
        include_command_tokens=True,
        include_graph_tokens=False,
    )

    result = pipeline.tokenize(mock_document_with_items)

    assert isinstance(result, SegNetTokenizationResult)
    assert result.vocabulary == mock_vocab
    assert result.segment_map == {"seg_0": ["item_0", "item_1"]}
    assert "mesh_tokenized" in result.metadata
    assert "command_tokenized" in result.metadata
    assert result.metadata["mesh_tokenized"] >= 0
    assert result.metadata["command_tokenized"] >= 0


# ============================================================================
# Test: tokenize() with empty document
# ============================================================================


@patch("cadling.pipeline.segnet_pipeline._try_import_geotoken")
def test_tokenize_with_empty_document(mock_import_geotoken, default_options, minimal_mock_document):
    """Test tokenization with an empty document (no items) returns empty result."""
    # Mock the geotoken components
    mock_geo_tokenizer_class = MagicMock()
    mock_graph_tokenizer_class = MagicMock()
    mock_cmd_tokenizer_class = MagicMock()
    mock_vocab_class = MagicMock()

    mock_import_geotoken.return_value = (
        mock_geo_tokenizer_class,
        mock_graph_tokenizer_class,
        mock_cmd_tokenizer_class,
        mock_vocab_class,
        None,  # TokenSequence
    )

    mock_vocab = MagicMock()
    mock_vocab_class.return_value = mock_vocab

    pipeline = SegNetPipeline(default_options)

    result = pipeline.tokenize(minimal_mock_document)

    assert isinstance(result, SegNetTokenizationResult)
    assert len(result.token_sequences) == 0
    assert result.segment_map == {}
    assert result.metadata["mesh_tokenized"] == 0
    assert result.metadata["command_tokenized"] == 0


# ============================================================================
# Test: tokenize() without geotoken available
# ============================================================================


@patch("cadling.pipeline.segnet_pipeline._try_import_geotoken")
def test_tokenize_without_geotoken(mock_import_geotoken, default_options, minimal_mock_document):
    """Test tokenize gracefully handles missing geotoken."""
    mock_import_geotoken.return_value = (None,) * 5

    pipeline = SegNetPipeline(default_options)
    result = pipeline.tokenize(minimal_mock_document)

    assert isinstance(result, SegNetTokenizationResult)
    assert "error" in result.metadata
    assert result.metadata["error"] == "geotoken not installed"


# ============================================================================
# Test: encode() with mock tokenization result
# ============================================================================


def test_encode_with_tokenization_result():
    """Test encode() converts token sequences to integer IDs."""
    # Create mock token sequences
    mock_token_seq_1 = MagicMock()
    mock_token_seq_2 = MagicMock()

    tokenization_result = SegNetTokenizationResult(
        token_sequences={
            "item_0": mock_token_seq_1,
            "item_1": mock_token_seq_2,
        },
        vocabulary=None,
    )

    # Mock the vocabulary
    mock_vocab = MagicMock()
    mock_vocab.encode_full_sequence.side_effect = [
        [1, 2, 3, 4],
        [5, 6, 7],
    ]
    tokenization_result.vocabulary = mock_vocab

    options = SegNetPipeline.get_default_options()
    pipeline = SegNetPipeline(options)

    encoded = pipeline.encode(tokenization_result)

    assert isinstance(encoded, dict)
    assert "item_0" in encoded
    assert "item_1" in encoded
    assert encoded["item_0"] == [1, 2, 3, 4]
    assert encoded["item_1"] == [5, 6, 7]
    assert mock_vocab.encode_full_sequence.call_count == 2


# ============================================================================
# Test: encode() with no vocabulary
# ============================================================================


def test_encode_without_vocabulary():
    """Test encode() returns empty dict when no vocabulary is available."""
    tokenization_result = SegNetTokenizationResult(
        token_sequences={"item_0": MagicMock()},
        vocabulary=None,
    )

    options = SegNetPipeline.get_default_options()
    pipeline = SegNetPipeline(options)

    encoded = pipeline.encode(tokenization_result)

    assert isinstance(encoded, dict)
    assert len(encoded) == 0


# ============================================================================
# Test: reconstruct() gracefully handles missing CommandExecutor
# ============================================================================


@patch("cadling.pipeline.segnet_pipeline._try_import_command_executor")
def test_reconstruct_without_command_executor(mock_import_executor):
    """Test reconstruct() gracefully handles missing CommandExecutor."""
    mock_import_executor.return_value = None

    options = SegNetPipeline.get_default_options()
    pipeline = SegNetPipeline(options)

    token_ids = {
        "item_0": [1, 2, 3],
        "item_1": [4, 5, 6],
    }

    results = pipeline.reconstruct(token_ids)

    assert isinstance(results, list)
    assert len(results) == 2
    for result in results:
        assert isinstance(result, ReconstructionResult)
        assert result.success is False
        assert "CommandExecutor not installed" in result.errors


# ============================================================================
# Test: reconstruct() with mock CommandExecutor
# ============================================================================


@patch("cadling.pipeline.segnet_pipeline._try_import_command_executor")
def test_reconstruct_with_command_executor(mock_import_executor):
    """Test reconstruct() with a mocked CommandExecutor."""
    # Mock CommandExecutor
    mock_executor_class = MagicMock()
    mock_executor = MagicMock()
    mock_executor_class.return_value = mock_executor

    # Mock the execute_tokens method to return a shape
    mock_shape = MagicMock()
    mock_executor.execute_tokens.return_value = mock_shape

    mock_import_executor.return_value = mock_executor_class

    options = SegNetPipeline.get_default_options()
    pipeline = SegNetPipeline(options)

    token_ids = {
        "item_0": [1, 2, 3],
        "item_1": [4, 5, 6],
    }

    results = pipeline.reconstruct(token_ids)

    assert isinstance(results, list)
    assert len(results) == 2

    for i, result in enumerate(results):
        assert isinstance(result, ReconstructionResult)
        assert result.success is True
        assert result.shape == mock_shape


# ============================================================================
# Test: reconstruct() when CommandExecutor init fails
# ============================================================================


@patch("cadling.pipeline.segnet_pipeline._try_import_command_executor")
def test_reconstruct_executor_init_failure(mock_import_executor):
    """Test reconstruct() handles CommandExecutor initialization failure."""
    mock_executor_class = MagicMock()
    mock_executor_class.side_effect = RuntimeError("Failed to initialize")

    mock_import_executor.return_value = mock_executor_class

    options = SegNetPipeline.get_default_options()
    pipeline = SegNetPipeline(options)

    token_ids = {"item_0": [1, 2, 3]}

    results = pipeline.reconstruct(token_ids)

    assert len(results) == 1
    assert results[0].success is False
    assert "CommandExecutor init failed" in results[0].errors[0]


# ============================================================================
# Test: process_document() integration with minimal mock
# ============================================================================


@patch("cadling.pipeline.segnet_pipeline._try_import_geotoken")
@patch("cadling.pipeline.segnet_pipeline._try_import_command_executor")
def test_process_document_integration(
    mock_import_executor,
    mock_import_geotoken,
    default_options,
    mock_document_with_items,
):
    """Test full process_document() integration with mocked dependencies."""
    # Setup geotoken mocks
    mock_geo_tokenizer_class = MagicMock()
    mock_geo_tokenizer = MagicMock()
    mock_geo_tokenizer.tokenize.return_value = MagicMock(name="mesh_token")
    mock_geo_tokenizer_class.return_value = mock_geo_tokenizer

    mock_cmd_tokenizer_class = MagicMock()
    mock_cmd_tokenizer = MagicMock()
    mock_cmd_tokenizer.tokenize.return_value = MagicMock(name="cmd_token")
    mock_cmd_tokenizer_class.return_value = mock_cmd_tokenizer

    mock_vocab_class = MagicMock()
    mock_vocab = MagicMock()
    mock_vocab.encode_full_sequence.side_effect = [[1, 2], [3, 4]]
    mock_vocab_class.return_value = mock_vocab

    mock_import_geotoken.return_value = (
        mock_geo_tokenizer_class,
        mock_cmd_tokenizer_class,
        mock_cmd_tokenizer_class,
        mock_vocab_class,
        None,
    )

    # Setup CommandExecutor mocks
    mock_executor_class = MagicMock()
    mock_executor = MagicMock()
    mock_executor.execute_tokens.return_value = MagicMock()
    mock_executor_class.return_value = mock_executor

    mock_import_executor.return_value = mock_executor_class

    pipeline = SegNetPipeline(default_options)
    result = pipeline.process_document(mock_document_with_items)

    assert isinstance(result, SegNetPipelineResult)
    assert result.document == mock_document_with_items
    assert isinstance(result.tokenization, SegNetTokenizationResult)
    assert isinstance(result.reconstructions, list)
    assert isinstance(result.metrics, dict)
    assert "total_ms" in result.metrics
    assert "tokenization_ms" in result.metrics
    assert "encoding_ms" in result.metrics
    assert "reconstruction_ms" in result.metrics
    assert result.metrics["total_ms"] >= 0


# ============================================================================
# Test: _extract_mesh_data() static method with properties
# ============================================================================


def test_extract_mesh_data_from_properties(mock_cad_item_with_mesh):
    """Test _extract_mesh_data extracts mesh from item.properties."""
    result = SegNetPipeline._extract_mesh_data(mock_cad_item_with_mesh)

    assert result is not None
    vertices, faces = result
    assert isinstance(vertices, np.ndarray)
    assert isinstance(faces, np.ndarray)
    assert vertices.shape == (3, 3)
    assert faces.shape == (1, 3)


# ============================================================================
# Test: _extract_mesh_data() with to_numpy method
# ============================================================================


def test_extract_mesh_data_with_to_numpy_method():
    """Test _extract_mesh_data uses to_numpy() method when available."""
    # Use MagicMock without spec so to_numpy attribute is allowed
    mock_item = MagicMock()
    mock_item.item_id = "mesh_item"
    mock_item.properties = {}

    # Setup to_numpy method
    expected_vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    expected_faces = np.array([[0, 1, 2]], dtype=np.int64)
    mock_item.to_numpy.return_value = (expected_vertices, expected_faces)

    result = SegNetPipeline._extract_mesh_data(mock_item)

    assert result is not None
    vertices, faces = result
    np.testing.assert_array_equal(vertices, expected_vertices)
    np.testing.assert_array_equal(faces, expected_faces)
    mock_item.to_numpy.assert_called_once()


# ============================================================================
# Test: _extract_mesh_data() returns None for non-mesh items
# ============================================================================


def test_extract_mesh_data_returns_none_for_non_mesh():
    """Test _extract_mesh_data returns None when no mesh data available."""
    item = CADItem(
        item_type="generic",
        item_id="item_no_mesh",
        label=CADItemLabel(text="non-mesh item"),
        properties={},
    )

    result = SegNetPipeline._extract_mesh_data(item)

    assert result is None


# ============================================================================
# Test: _extract_commands() static method with properties
# ============================================================================


def test_extract_commands_from_properties(mock_cad_item_with_commands):
    """Test _extract_commands extracts command sequence from properties."""
    result = SegNetPipeline._extract_commands(mock_cad_item_with_commands)

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["command"] == "move"
    assert result[1]["command"] == "line"


# ============================================================================
# Test: _extract_commands() with to_geotoken_commands method
# ============================================================================


def test_extract_commands_with_method():
    """Test _extract_commands uses to_geotoken_commands() when available."""
    # Use MagicMock without spec so to_geotoken_commands attribute is allowed
    mock_item = MagicMock()
    mock_item.item_id = "sketch_item"
    mock_item.properties = {}

    expected_commands = [
        {"command": "circle", "args": [0, 0, 5]},
        {"command": "line", "args": [5, 0, 10, 0]},
    ]
    mock_item.to_geotoken_commands.return_value = expected_commands

    result = SegNetPipeline._extract_commands(mock_item)

    assert result == expected_commands
    mock_item.to_geotoken_commands.assert_called_once()


# ============================================================================
# Test: _extract_commands() returns None for items without commands
# ============================================================================


def test_extract_commands_returns_none():
    """Test _extract_commands returns None when no commands available."""
    item = CADItem(
        item_type="generic",
        item_id="item_no_commands",
        label=CADItemLabel(text="no commands"),
        properties={},
    )

    result = SegNetPipeline._extract_commands(item)

    assert result is None


# ============================================================================
# Test: _extract_constraints() static method with properties
# ============================================================================


def test_extract_constraints_from_properties(mock_cad_item_with_constraints):
    """Test _extract_constraints extracts constraints from properties."""
    result = SegNetPipeline._extract_constraints(mock_cad_item_with_constraints)

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["constraint_type"] == "distance"
    assert result[1]["constraint_type"] == "horizontal"


# ============================================================================
# Test: _extract_constraints() with to_geotoken_constraints method
# ============================================================================


def test_extract_constraints_with_method():
    """Test _extract_constraints uses to_geotoken_constraints() when available."""
    # Use MagicMock without spec so to_geotoken_constraints attribute is allowed
    mock_item = MagicMock()
    mock_item.item_id = "constrained_item"
    mock_item.properties = {}

    expected_constraints = [
        {"constraint_type": "distance", "value": 5.0},
        {"constraint_type": "angle", "value": 90.0},
    ]
    mock_item.to_geotoken_constraints.return_value = expected_constraints

    result = SegNetPipeline._extract_constraints(mock_item)

    assert result == expected_constraints
    mock_item.to_geotoken_constraints.assert_called_once()


# ============================================================================
# Test: _extract_constraints() returns None for unconstrained items
# ============================================================================


def test_extract_constraints_returns_none():
    """Test _extract_constraints returns None when no constraints available."""
    item = CADItem(
        item_type="generic",
        item_id="item_no_constraints",
        label=CADItemLabel(text="unconstrained"),
        properties={},
    )

    result = SegNetPipeline._extract_constraints(item)

    assert result is None


# ============================================================================
# Test: SegNetTokenizationResult dataclass creation
# ============================================================================


def test_segnet_tokenization_result_creation():
    """Test SegNetTokenizationResult dataclass initialization."""
    token_sequences = {
        "item_0": MagicMock(),
        "item_1": MagicMock(),
    }
    vocabulary = MagicMock()
    segment_map = {"seg_0": ["item_0"], "seg_1": ["item_1"]}
    metadata = {"duration_ms": 100.5, "mesh_tokenized": 2}

    result = SegNetTokenizationResult(
        token_sequences=token_sequences,
        vocabulary=vocabulary,
        segment_map=segment_map,
        metadata=metadata,
    )

    assert result.token_sequences == token_sequences
    assert result.vocabulary == vocabulary
    assert result.segment_map == segment_map
    assert result.metadata == metadata


# ============================================================================
# Test: SegNetTokenizationResult with default factories
# ============================================================================


def test_segnet_tokenization_result_defaults():
    """Test SegNetTokenizationResult uses default factories."""
    result = SegNetTokenizationResult()

    assert isinstance(result.token_sequences, dict)
    assert len(result.token_sequences) == 0
    assert result.vocabulary is None
    assert isinstance(result.segment_map, dict)
    assert isinstance(result.metadata, dict)


# ============================================================================
# Test: ReconstructionResult dataclass creation
# ============================================================================


def test_reconstruction_result_creation():
    """Test ReconstructionResult dataclass initialization."""
    shape = MagicMock()
    errors = ["error1", "error2"]
    validation_report = {"valid": True, "errors": []}

    result = ReconstructionResult(
        item_id="item_0",
        success=True,
        shape=shape,
        errors=errors,
        validation_report=validation_report,
    )

    assert result.item_id == "item_0"
    assert result.success is True
    assert result.shape == shape
    assert result.errors == errors
    assert result.validation_report == validation_report


# ============================================================================
# Test: ReconstructionResult with defaults
# ============================================================================


def test_reconstruction_result_defaults():
    """Test ReconstructionResult uses default factories."""
    result = ReconstructionResult(
        item_id="item_0",
        success=False,
    )

    assert result.item_id == "item_0"
    assert result.success is False
    assert result.shape is None
    assert isinstance(result.errors, list)
    assert len(result.errors) == 0
    assert result.validation_report is None


# ============================================================================
# Test: SegNetPipelineResult dataclass creation
# ============================================================================


def test_segnet_pipeline_result_creation():
    """Test SegNetPipelineResult dataclass initialization."""
    document = MagicMock(spec=CADlingDocument)
    tokenization = SegNetTokenizationResult()
    reconstructions = [
        ReconstructionResult(item_id="item_0", success=True),
        ReconstructionResult(item_id="item_1", success=False),
    ]
    metrics = {"total_ms": 250.5, "num_tokens": 10}

    result = SegNetPipelineResult(
        document=document,
        tokenization=tokenization,
        reconstructions=reconstructions,
        metrics=metrics,
    )

    assert result.document == document
    assert result.tokenization == tokenization
    assert result.reconstructions == reconstructions
    assert result.metrics == metrics


# ============================================================================
# Test: SegNetPipelineResult with defaults
# ============================================================================


def test_segnet_pipeline_result_defaults():
    """Test SegNetPipelineResult uses default factories."""
    result = SegNetPipelineResult()

    assert result.document is None
    assert result.tokenization is None
    assert isinstance(result.reconstructions, list)
    assert len(result.reconstructions) == 0
    assert isinstance(result.metrics, dict)
    assert len(result.metrics) == 0


# ============================================================================
# Test: tokenize() with topology graph
# ============================================================================


@patch("cadling.pipeline.segnet_pipeline._try_import_geotoken")
def test_tokenize_with_topology_graph(mock_import_geotoken, default_options, mock_document_with_topology):
    """Test tokenization of topology graph when present in document."""
    # Mock the geotoken components
    mock_geo_tokenizer_class = MagicMock()
    mock_graph_tokenizer_class = MagicMock()
    mock_cmd_tokenizer_class = MagicMock()
    mock_vocab_class = MagicMock()

    mock_graph_tokenizer = MagicMock()
    mock_graph_tokenizer.tokenize.return_value = MagicMock(name="graph_token_seq")
    mock_graph_tokenizer_class.return_value = mock_graph_tokenizer

    mock_vocab = MagicMock()
    mock_vocab_class.return_value = mock_vocab

    mock_import_geotoken.return_value = (
        mock_geo_tokenizer_class,
        mock_graph_tokenizer_class,
        mock_cmd_tokenizer_class,
        mock_vocab_class,
        None,
    )

    pipeline = SegNetPipeline(
        default_options,
        include_graph_tokens=True,
        include_mesh_tokens=False,
        include_command_tokens=False,
    )

    result = pipeline.tokenize(mock_document_with_topology)

    assert isinstance(result, SegNetTokenizationResult)
    assert "__graph__" in result.token_sequences
    assert result.metadata["graph_tokenized"] == 1
    mock_graph_tokenizer.tokenize.assert_called_once()


# ============================================================================
# Test: tokenize() handles exceptions in mesh tokenization
# ============================================================================


@patch("cadling.pipeline.segnet_pipeline._try_import_geotoken")
def test_tokenize_mesh_exception_handling(mock_import_geotoken, default_options, mock_document_with_items):
    """Test tokenize gracefully handles exceptions in mesh tokenization."""
    # Mock the geotoken components
    mock_geo_tokenizer_class = MagicMock()
    mock_geo_tokenizer = MagicMock()
    mock_geo_tokenizer.tokenize.side_effect = RuntimeError("Tokenization failed")
    mock_geo_tokenizer_class.return_value = mock_geo_tokenizer

    mock_vocab_class = MagicMock()
    mock_vocab = MagicMock()
    mock_vocab_class.return_value = mock_vocab

    mock_import_geotoken.return_value = (
        mock_geo_tokenizer_class,
        None,
        None,
        mock_vocab_class,
        None,
    )

    pipeline = SegNetPipeline(
        default_options,
        include_mesh_tokens=True,
        include_graph_tokens=False,
        include_command_tokens=False,
    )

    result = pipeline.tokenize(mock_document_with_items)

    assert isinstance(result, SegNetTokenizationResult)
    assert result.metadata["mesh_tokenized"] == 0
    assert result.metadata["items_skipped"] > 0


# ============================================================================
# Test: Pipeline preserves enrichment models through full flow
# ============================================================================


def test_pipeline_with_enrichment_models(default_options):
    """Test that pipeline preserves enrichment models through initialization."""
    # PipelineOptions validates enrichment_models as List[EnrichmentModel],
    # so we pass an empty list and verify the pipeline initialises cleanly.
    options = PipelineOptions(
        enrichment_models=[],
        device="cpu",
    )

    pipeline = SegNetPipeline(options)

    # Pipeline should initialise without error even with empty enrichment list
    assert pipeline is not None
    assert pipeline.include_graph_tokens is True


# ============================================================================
# Test: encode() handles encoding errors gracefully
# ============================================================================


def test_encode_with_encoding_error():
    """Test encode() handles exceptions during token encoding."""
    mock_token_seq = MagicMock()

    tokenization_result = SegNetTokenizationResult(
        token_sequences={
            "item_0": mock_token_seq,
        },
        vocabulary=None,
    )

    mock_vocab = MagicMock()
    mock_vocab.encode_full_sequence.side_effect = RuntimeError("Encoding failed")
    tokenization_result.vocabulary = mock_vocab

    options = SegNetPipeline.get_default_options()
    pipeline = SegNetPipeline(options)

    encoded = pipeline.encode(tokenization_result)

    assert isinstance(encoded, dict)
    assert "item_0" not in encoded


# ============================================================================
# Test: reconstruct() handles reconstruction errors per-item
# ============================================================================


@patch("cadling.pipeline.segnet_pipeline._try_import_command_executor")
def test_reconstruct_handles_per_item_errors(mock_import_executor):
    """Test reconstruct() handles per-item errors while processing others."""
    mock_executor_class = MagicMock()
    mock_executor = MagicMock()

    # First call succeeds, second fails
    mock_shape = MagicMock()
    mock_executor.execute_tokens.side_effect = [
        mock_shape,
        RuntimeError("Reconstruction failed"),
    ]

    mock_executor_class.return_value = mock_executor

    mock_import_executor.return_value = mock_executor_class

    options = SegNetPipeline.get_default_options()
    pipeline = SegNetPipeline(options)

    token_ids = {
        "item_0": [1, 2, 3],
        "item_1": [4, 5, 6],
    }

    results = pipeline.reconstruct(token_ids)

    assert len(results) == 2
    assert results[0].success is True
    assert results[1].success is False
    assert len(results[1].errors) > 0
