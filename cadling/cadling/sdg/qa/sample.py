"""CAD passage sampler for Q&A generation.

This module provides the CADPassageSampler class for sampling
passages (chunks) from CAD documents to use for Q&A generation.

Classes:
    CADPassageSampler: Sample passages from CAD documents
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from cadling.sdg.qa.base import (
    CADQaChunk,
    CADSampleOptions,
    ChunkerType,
    SampleResult,
    Status,
)
from cadling.sdg.qa.utils import count_tokens_simple, save_one_to_file

if TYPE_CHECKING:
    from cadling.chunker.base_chunker import CADChunk
    from cadling.datamodel.base_models import CADlingDocument

_log = logging.getLogger(__name__)


class CADPassageSampler:
    """Sample passages from CAD documents for Q&A generation.

    Extracts chunks from CAD documents using cadling chunkers and
    samples them for use in Q&A pair generation.

    Attributes:
        options: Sampling configuration options
        chunker: Initialized CAD chunker

    Example:
        from cadling.sdg.qa.sample import CADPassageSampler
        from cadling.sdg.qa.base import CADSampleOptions

        options = CADSampleOptions(
            sample_file=Path("samples.jsonl"),
            max_passages=100,
            chunker="hybrid",
        )

        sampler = CADPassageSampler(options)
        result = sampler.sample([Path("model.step")])
    """

    def __init__(self, options: CADSampleOptions):
        """Initialize passage sampler.

        Args:
            options: Sampling configuration options
        """
        self.options = options
        self.chunker = self._init_chunker()

        _log.info(
            f"Initialized CADPassageSampler (chunker={options.chunker}, "
            f"max_passages={options.max_passages})"
        )

    def _init_chunker(self):
        """Initialize the CAD chunker based on options.

        Returns:
            Initialized chunker instance
        """
        chunker_type = self.options.chunker

        if chunker_type == ChunkerType.HYBRID:
            from cadling.chunker.hybrid_chunker import CADHybridChunker
            return CADHybridChunker(max_tokens=self.options.max_tokens)

        elif chunker_type == ChunkerType.STEP:
            from cadling.chunker.step_chunker import STEPChunker
            return STEPChunker(max_tokens=self.options.max_tokens)

        elif chunker_type == ChunkerType.STL:
            from cadling.chunker.stl_chunker import STLChunker
            return STLChunker(max_tokens=self.options.max_tokens)

        elif chunker_type == ChunkerType.BREP:
            from cadling.chunker.brep_chunker import BRepChunker
            return BRepChunker(max_tokens=self.options.max_tokens)

        elif chunker_type == ChunkerType.TOPOLOGY:
            from cadling.chunker.topology_chunker import TopologyChunker
            return TopologyChunker(max_tokens=self.options.max_tokens)

        else:
            # Default to hybrid
            _log.warning(f"Unknown chunker type: {chunker_type}, using hybrid")
            from cadling.chunker.hybrid_chunker import CADHybridChunker
            return CADHybridChunker(max_tokens=self.options.max_tokens)

    def sample(self, sources: list[Path]) -> SampleResult:
        """Sample passages from source CAD files.

        Args:
            sources: List of CAD file paths to process

        Returns:
            SampleResult with statistics and output path
        """
        start_time = time.time()
        errors: list[str] = []
        all_chunks: list[CADQaChunk] = []

        _log.info(f"Sampling from {len(sources)} source files")

        # Process each source file
        for source in sources:
            try:
                chunks = self._process_source(source)
                all_chunks.extend(chunks)
                _log.debug(f"Extracted {len(chunks)} chunks from {source}")
            except Exception as e:
                error_msg = f"Failed to process {source}: {e}"
                _log.error(error_msg)
                errors.append(error_msg)

        # Random sample if we have more chunks than max_passages
        if len(all_chunks) > self.options.max_passages:
            random.seed(self.options.seed)
            all_chunks = random.sample(all_chunks, self.options.max_passages)
            _log.info(f"Sampled {len(all_chunks)} passages from total pool")

        # Save to file
        output_path = self.options.sample_file
        for chunk in all_chunks:
            save_one_to_file(chunk, output_path)

        elapsed = time.time() - start_time

        # Determine status
        if len(all_chunks) == 0:
            status = Status.FAILURE
        elif errors:
            status = Status.PARTIAL
        else:
            status = Status.SUCCESS

        result = SampleResult(
            status=status,
            time_taken=elapsed,
            output=output_path,
            num_passages=len(all_chunks),
            num_sources=len(sources),
            errors=errors,
        )

        _log.info(
            f"Sampling complete: {result.num_passages} passages in {elapsed:.2f}s"
        )

        return result

    def _process_source(self, source: Path) -> list[CADQaChunk]:
        """Process a single source file to extract chunks.

        Args:
            source: Path to CAD file

        Returns:
            List of CADQaChunk objects
        """
        # Convert CAD file to document
        from cadling.backend.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(source))

        if not result.document:
            raise ValueError(f"Failed to convert {source}")

        doc = result.document

        # Generate chunks
        chunks = list(self.chunker.chunk(doc))
        _log.debug(f"Generated {len(chunks)} raw chunks from {source}")

        # Filter and convert to CADQaChunk
        qa_chunks = []

        for chunk in chunks:
            # Check token count
            token_count = count_tokens_simple(chunk.text)

            if token_count < self.options.min_tokens:
                _log.debug(f"Skipping chunk with {token_count} tokens (below min)")
                continue

            # Check item types if specified
            if self.options.item_types:
                if not self._has_matching_types(chunk):
                    continue

            # Convert to CADQaChunk
            qa_chunk = self._convert_chunk(chunk, doc)
            qa_chunk.token_count = token_count
            qa_chunks.append(qa_chunk)

        return qa_chunks

    def _has_matching_types(self, chunk: CADChunk) -> bool:
        """Check if chunk has any of the desired entity types.

        Args:
            chunk: CAD chunk to check

        Returns:
            True if chunk has matching types
        """
        if not chunk.meta or not chunk.meta.entity_types:
            return False

        chunk_types = set(chunk.meta.entity_types)
        desired_types = set(self.options.item_types)

        return bool(chunk_types & desired_types)

    def _convert_chunk(
        self,
        chunk: CADChunk,
        doc: CADlingDocument,
    ) -> CADQaChunk:
        """Convert CADChunk to CADQaChunk.

        Args:
            chunk: Source CAD chunk
            doc: Source document

        Returns:
            CADQaChunk for Q&A generation
        """
        return CADQaChunk(
            text=chunk.text,
            chunk_id=chunk.chunk_id,
            doc_id=doc.id if hasattr(doc, "id") else "",
            doc_name=chunk.doc_name or doc.name,
            entity_types=chunk.meta.entity_types if chunk.meta else [],
            entity_ids=chunk.meta.entity_ids if chunk.meta else [],
            properties=chunk.meta.properties if chunk.meta else {},
            topology_subgraph=(
                chunk.meta.topology_subgraph
                if chunk.meta and self.options.include_topology
                else None
            ),
        )

    def sample_from_document(
        self,
        doc: CADlingDocument,
    ) -> Iterator[CADQaChunk]:
        """Sample passages from an already-loaded document.

        Args:
            doc: CADlingDocument to sample from

        Yields:
            CADQaChunk instances
        """
        # Generate chunks
        chunks = list(self.chunker.chunk(doc))

        # Filter
        valid_chunks = []
        for chunk in chunks:
            token_count = count_tokens_simple(chunk.text)

            if token_count < self.options.min_tokens:
                continue

            if self.options.item_types and not self._has_matching_types(chunk):
                continue

            qa_chunk = self._convert_chunk(chunk, doc)
            qa_chunk.token_count = token_count
            valid_chunks.append(qa_chunk)

        # Sample if needed
        if len(valid_chunks) > self.options.max_passages:
            random.seed(self.options.seed)
            valid_chunks = random.sample(valid_chunks, self.options.max_passages)

        yield from valid_chunks
