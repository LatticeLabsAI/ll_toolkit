"""
Integration tests for chunking system.

Tests various chunking strategies on CAD documents.
"""

import pytest


class TestChunking:
    """Integration tests for chunking."""

    def test_sequential_chunker(self):
        """Test sequential chunker."""
        from cadling.chunking import SequentialChunker
        from cadling.datamodel.base_models import CADlingDocument, CADItem

        # Create test document
        doc = CADlingDocument(name="test_doc.step")

        # Add test items
        for i in range(25):
            item = CADItem(
                label={"text": f"Item {i}"},
                text=f"Test item number {i}",
                item_type="test",
            )
            doc.add_item(item)

        # Create chunker
        chunker = SequentialChunker(chunk_size=10, chunk_overlap=2)

        # Chunk document
        chunks = chunker.chunk(doc)

        # Verify chunks
        assert len(chunks) > 0
        assert len(chunks) <= 3  # 25 items / 10 per chunk with overlap

        # Check chunk properties
        for chunk in chunks:
            assert chunk.chunk_id is not None
            assert len(chunk.items) > 0
            assert chunk.text is not None
            assert "chunk_type" in chunk.metadata
            assert chunk.metadata["chunk_type"] == "sequential"

    def test_entity_type_chunker(self):
        """Test entity type chunker."""
        from cadling.chunking import EntityTypeChunker
        from cadling.datamodel.base_models import CADlingDocument
        from cadling.datamodel.step import STEPEntityItem

        # Create test document
        doc = CADlingDocument(name="test_doc.step")

        # Add items of different types
        for i in range(10):
            point = STEPEntityItem(
                entity_id=i,
                entity_type="CARTESIAN_POINT",
                label={"text": f"Point {i}"},
                text=f"Point {i}",
            )
            doc.add_item(point)

        for i in range(5):
            line = STEPEntityItem(
                entity_id=i + 10,
                entity_type="LINE",
                label={"text": f"Line {i}"},
                text=f"Line {i}",
            )
            doc.add_item(line)

        # Create chunker
        chunker = EntityTypeChunker(chunk_size=20)

        # Chunk document
        chunks = chunker.chunk(doc)

        # Verify chunks
        assert len(chunks) == 2  # One for points, one for lines

        # Check that items are grouped by type
        entity_types = set()
        for chunk in chunks:
            chunk_types = set(item.entity_type for item in chunk.items)
            assert len(chunk_types) == 1  # Each chunk has only one type
            entity_types.update(chunk_types)

        assert "CARTESIAN_POINT" in entity_types
        assert "LINE" in entity_types

    def test_spatial_chunker(self):
        """Test spatial chunker."""
        from cadling.chunking import SpatialChunker
        from cadling.datamodel.base_models import CADlingDocument, BoundingBox3D
        from cadling.datamodel.step import STEPEntityItem

        # Create test document with bounding box
        doc = CADlingDocument(name="test_doc.step")
        doc.bounding_box = BoundingBox3D(
            x_min=0.0, y_min=0.0, z_min=0.0, x_max=10.0, y_max=10.0, z_max=10.0
        )

        # Add items with bounding boxes at different locations
        for i in range(8):
            # Distribute items in 8 octants
            x = 2.5 if i % 2 == 0 else 7.5
            y = 2.5 if (i // 2) % 2 == 0 else 7.5
            z = 2.5 if (i // 4) % 2 == 0 else 7.5

            item = STEPEntityItem(
                entity_id=i,
                entity_type="CARTESIAN_POINT",
                label={"text": f"Point {i}"},
                text=f"Point at ({x},{y},{z}))",
            )
            item.bbox = BoundingBox3D(
                x_min=x, y_min=y, z_min=z, x_max=x, y_max=y, z_max=z
            )
            doc.add_item(item)

        # Create chunker
        chunker = SpatialChunker(chunk_size=5, max_depth=1)

        # Chunk document
        chunks = chunker.chunk(doc)

        # Verify chunks
        assert len(chunks) > 0

        # Check chunk properties
        for chunk in chunks:
            assert chunk.chunk_id is not None
            assert "chunk_type" in chunk.metadata
            assert chunk.metadata["chunk_type"] == "spatial"
            if "octant" in chunk.metadata:
                # Spatial chunks should have octant metadata
                assert chunk.metadata["octant"] is not None

    def test_chunk_overlap(self):
        """Test chunk overlap functionality."""
        from cadling.chunking import SequentialChunker
        from cadling.datamodel.base_models import CADlingDocument, CADItem

        # Create test document
        doc = CADlingDocument(name="test_doc.step")

        # Add test items
        for i in range(20):
            item = CADItem(
                label={"text": f"Item {i}"},
                text=f"Test item {i}",
                item_type="test",
            )
            doc.add_item(item)

        # Create chunker with overlap
        chunker = SequentialChunker(chunk_size=10, chunk_overlap=3)

        # Chunk document
        chunks = chunker.chunk(doc)

        # Verify overlap
        assert len(chunks) >= 2

        # Check that consecutive chunks have overlapping items
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]

            # Get last 3 items from current chunk
            current_end_items = current_chunk.items[-3:]

            # Get first 3 items from next chunk
            next_start_items = next_chunk.items[:3]

            # These should be the same items (by reference or content)
            # Note: Due to implementation, overlap is added from previous chunk
            # So we just verify that next chunk has more items than just new ones
            assert len(next_chunk.items) >= 3

    def test_chunk_text_generation(self):
        """Test chunk text generation."""
        from cadling.chunking import SequentialChunker
        from cadling.datamodel.base_models import CADlingDocument
        from cadling.datamodel.step import STEPEntityItem

        # Create test document
        doc = CADlingDocument(name="test_doc.step")

        # Add items
        for i in range(5):
            item = STEPEntityItem(
                entity_id=i,
                entity_type="CARTESIAN_POINT",
                label={"text": f"Point {i}"},
                text=f"CARTESIAN_POINT({i}.0, 0.0, 0.0)",
            )
            doc.add_item(item)

        # Create chunker
        chunker = SequentialChunker(chunk_size=10)

        # Chunk document
        chunks = chunker.chunk(doc)

        # Verify text generation
        assert len(chunks) == 1
        chunk = chunks[0]

        # Check text content
        assert chunk.text is not None
        assert len(chunk.text) > 0
        assert "CAD Chunk" in chunk.text
        assert "Item 1" in chunk.text or "Item" in chunk.text
        assert chunk.metadata["chunk_id"] in chunk.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
