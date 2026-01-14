"""Chunk serialization utilities.

This module provides serializers for converting CAD chunks to various
output formats (JSON, JSONL, Markdown, HTML).

Classes:
    ChunkSerializer: Abstract serializer interface
    JSONSerializer: JSON format serializer
    JSONLSerializer: JSON Lines format serializer
    MarkdownSerializer: Markdown format serializer
    HTMLSerializer: HTML format serializer

Functions:
    get_serializer: Factory function to create serializer instances
"""

from cadling.chunker.serializer.serializer import (
    ChunkSerializer,
    HTMLSerializer,
    JSONLSerializer,
    JSONSerializer,
    MarkdownSerializer,
    get_serializer,
)

__all__ = [
    "ChunkSerializer",
    "JSONSerializer",
    "JSONLSerializer",
    "MarkdownSerializer",
    "HTMLSerializer",
    "get_serializer",
]
