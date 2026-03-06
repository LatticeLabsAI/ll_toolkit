"""Unit tests for CADGenerator.generate_from_document.

Tests the document-to-QA pipeline path, including the token_count
computation that was previously broken (always 0).
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cadling.chunker.base_chunker import CADChunk, CADChunkMeta
from cadling.sdg.qa.base import (
    CADGenerateOptions,
    CADQaChunk,
    LlmProvider,
    Status,
)
from cadling.sdg.qa.generate import CADGenerator


@pytest.fixture
def mock_agent():
    """Mock ChatAgent that returns plausible Q&A text."""
    with patch("cadling.sdg.qa.generate.ChatAgent") as mock_cls:
        agent = MagicMock()
        agent.ask.side_effect = [
            "What material is used for Part_A?",  # question
            "Part_A uses aluminum alloy 6061.",  # answer
        ]
        mock_cls.from_options.return_value = agent
        yield agent


@pytest.fixture
def generator_options(tmp_path):
    """Standard generator options pointing at a temp output file."""
    return CADGenerateOptions(
        provider=LlmProvider.OPENAI,
        model_id="test-model",
        generated_file=tmp_path / "generated.jsonl",
        max_qac=10,
    )


@pytest.fixture
def sample_cad_doc():
    """Minimal CADlingDocument mock with a name attribute."""
    doc = MagicMock()
    doc.name = "test_part.step"
    return doc


def _make_chunk(text: str, chunk_id: str = "c1") -> CADChunk:
    """Create a CADChunk with the given text."""
    return CADChunk(
        text=text,
        meta=CADChunkMeta(
            entity_types=["MANIFOLD_SOLID_BREP"],
            entity_ids=[42],
            properties={"material": "aluminum"},
        ),
        chunk_id=chunk_id,
        doc_name="test_part.step",
    )


class TestGenerateFromDocument:
    """Tests for CADGenerator.generate_from_document."""

    def test_token_count_computed_correctly(
        self, mock_agent, generator_options, sample_cad_doc
    ):
        """token_count on CADQaChunks must reflect actual text, not default 0."""
        text = "This is a MANIFOLD_SOLID_BREP entity with aluminum alloy 6061 material."
        chunk = _make_chunk(text)

        # Verify the raw CADChunk has token_count=0 (the default)
        assert chunk.token_count == 0

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [chunk]

        generator = CADGenerator(generator_options, seed=42)

        # Capture the qa_chunks passed to generate_from_chunks
        captured_chunks: list[CADQaChunk] = []
        original_gen = generator.generate_from_chunks

        def capture_gen(chunks_iter):
            chunks_list = list(chunks_iter)
            captured_chunks.extend(chunks_list)
            return original_gen(iter(chunks_list))

        generator.generate_from_chunks = capture_gen

        generator.generate_from_document(
            sample_cad_doc, chunker=mock_chunker
        )

        assert len(captured_chunks) == 1
        qa_chunk = captured_chunks[0]

        # The token_count must be computed, not 0
        expected = len(re.findall(r"\w+|[^\w\s]", text))
        assert qa_chunk.token_count == expected
        assert qa_chunk.token_count > 0

    def test_metadata_propagated(
        self, mock_agent, generator_options, sample_cad_doc
    ):
        """Entity types, IDs, and properties propagate from chunk meta."""
        chunk = _make_chunk("Some CAD entity description text here.")

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [chunk]

        generator = CADGenerator(generator_options, seed=42)

        captured: list[CADQaChunk] = []
        original_gen = generator.generate_from_chunks

        def capture_gen(chunks_iter):
            chunks_list = list(chunks_iter)
            captured.extend(chunks_list)
            return original_gen(iter(chunks_list))

        generator.generate_from_chunks = capture_gen

        generator.generate_from_document(
            sample_cad_doc, chunker=mock_chunker
        )

        assert len(captured) == 1
        assert captured[0].entity_types == ["MANIFOLD_SOLID_BREP"]
        assert captured[0].entity_ids == ["42"]
        assert captured[0].properties == {"material": "aluminum"}

    def test_empty_chunks_produce_failure(
        self, mock_agent, generator_options, sample_cad_doc
    ):
        """No chunks means no Q&A pairs → FAILURE status."""
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = []

        generator = CADGenerator(generator_options, seed=42)
        result = generator.generate_from_document(
            sample_cad_doc, chunker=mock_chunker
        )

        assert result.status == Status.FAILURE
        assert result.num_qac == 0

    def test_num_pairs_limits_sampling(
        self, mock_agent, generator_options, sample_cad_doc
    ):
        """num_pairs caps how many chunks are sampled."""
        chunks = [_make_chunk(f"Chunk number {i} text.", f"c{i}") for i in range(10)]

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = chunks

        generator = CADGenerator(generator_options, seed=42)

        captured: list[CADQaChunk] = []
        original_gen = generator.generate_from_chunks

        def capture_gen(chunks_iter):
            chunks_list = list(chunks_iter)
            captured.extend(chunks_list)
            return original_gen(iter(chunks_list))

        generator.generate_from_chunks = capture_gen

        generator.generate_from_document(
            sample_cad_doc, num_pairs=3, chunker=mock_chunker
        )

        assert len(captured) == 3

    def test_default_chunker_used_when_none(
        self, mock_agent, generator_options, sample_cad_doc
    ):
        """When no chunker is provided, CADHybridChunker is instantiated."""
        with patch(
            "cadling.chunker.hybrid_chunker.CADHybridChunker"
        ) as mock_hybrid_cls:
            mock_hybrid = MagicMock()
            mock_hybrid.chunk.return_value = []
            mock_hybrid_cls.return_value = mock_hybrid

            generator = CADGenerator(generator_options, seed=42)
            generator.generate_from_document(sample_cad_doc)

            mock_hybrid_cls.assert_called_once_with(max_tokens=512)
            mock_hybrid.chunk.assert_called_once_with(sample_cad_doc)

    def test_chunk_with_empty_meta(
        self, mock_agent, generator_options, sample_cad_doc
    ):
        """Chunks with empty meta should produce empty entity lists."""
        chunk = CADChunk(
            text="A chunk with empty metadata.",
            meta=CADChunkMeta(),
            chunk_id="c_empty_meta",
            doc_name="test_part.step",
        )

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [chunk]

        generator = CADGenerator(generator_options, seed=42)

        captured: list[CADQaChunk] = []
        original_gen = generator.generate_from_chunks

        def capture_gen(chunks_iter):
            chunks_list = list(chunks_iter)
            captured.extend(chunks_list)
            return original_gen(iter(chunks_list))

        generator.generate_from_chunks = capture_gen

        generator.generate_from_document(
            sample_cad_doc, chunker=mock_chunker
        )

        assert len(captured) == 1
        assert captured[0].entity_types == []
        assert captured[0].entity_ids == []
        assert captured[0].properties == {}
        assert captured[0].token_count > 0
