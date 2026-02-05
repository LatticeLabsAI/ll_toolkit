"""Chunk serialization utilities.

This module provides serializers for converting CAD chunks to various
output formats (JSON, JSONL, Markdown, HTML, etc.).

Classes:
    ChunkSerializer: Abstract serializer interface
    JSONSerializer: JSON format serializer
    JSONLSerializer: JSON Lines format serializer
    MarkdownSerializer: Markdown format serializer
    HTMLSerializer: HTML format serializer
"""

from __future__ import annotations

import html
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

from cadling.chunker.base_chunker import CADChunk

_log = logging.getLogger(__name__)


class ChunkSerializer(ABC):
    """Abstract base class for chunk serializers."""

    @abstractmethod
    def serialize(self, chunks: List[CADChunk]) -> str:
        """Serialize chunks to string format.

        Args:
            chunks: List of CAD chunks

        Returns:
            Serialized string
        """
        pass

    @abstractmethod
    def serialize_one(self, chunk: CADChunk) -> str:
        """Serialize a single chunk.

        Args:
            chunk: CAD chunk

        Returns:
            Serialized string
        """
        pass

    def save(self, chunks: List[CADChunk], output_path: Union[str, Path]):
        """Save chunks to file.

        Args:
            chunks: List of CAD chunks
            output_path: Output file path
        """
        output_path = Path(output_path)
        serialized = self.serialize(chunks)
        output_path.write_text(serialized)
        _log.info(f"Saved {len(chunks)} chunks to {output_path}")


class JSONSerializer(ChunkSerializer):
    """JSON format serializer.

    Serializes chunks as a JSON array.
    """

    def __init__(self, pretty: bool = True, include_metadata: bool = True):
        """Initialize JSON serializer.

        Args:
            pretty: Pretty-print JSON
            include_metadata: Include chunk metadata
        """
        self.pretty = pretty
        self.include_metadata = include_metadata

    def serialize(self, chunks: List[CADChunk]) -> str:
        """Serialize chunks to JSON array.

        Args:
            chunks: List of CAD chunks

        Returns:
            JSON string
        """
        chunk_dicts = [self._chunk_to_dict(chunk) for chunk in chunks]

        if self.pretty:
            return json.dumps(chunk_dicts, indent=2)
        else:
            return json.dumps(chunk_dicts)

    def serialize_one(self, chunk: CADChunk) -> str:
        """Serialize single chunk to JSON.

        Args:
            chunk: CAD chunk

        Returns:
            JSON string
        """
        chunk_dict = self._chunk_to_dict(chunk)

        if self.pretty:
            return json.dumps(chunk_dict, indent=2)
        else:
            return json.dumps(chunk_dict)

    def _chunk_to_dict(self, chunk: CADChunk) -> dict:
        """Convert chunk to dictionary.

        Args:
            chunk: CAD chunk

        Returns:
            Dictionary representation
        """
        data = {
            "chunk_id": chunk.chunk_id,
            "doc_name": chunk.doc_name,
            "text": chunk.text,
        }

        if self.include_metadata and chunk.meta:
            data["metadata"] = {
                "entity_types": chunk.meta.entity_types,
                "entity_ids": chunk.meta.entity_ids,
                "properties": chunk.meta.properties,
            }

            if chunk.meta.topology_subgraph:
                data["metadata"]["topology_subgraph"] = chunk.meta.topology_subgraph

            if chunk.meta.embedding:
                data["metadata"]["embedding"] = chunk.meta.embedding

        return data


class JSONLSerializer(ChunkSerializer):
    """JSON Lines format serializer.

    Serializes each chunk as a JSON object on a separate line.
    Useful for streaming and large datasets.
    """

    def __init__(self, include_metadata: bool = True):
        """Initialize JSONL serializer.

        Args:
            include_metadata: Include chunk metadata
        """
        self.include_metadata = include_metadata

    def serialize(self, chunks: List[CADChunk]) -> str:
        """Serialize chunks to JSONL format.

        Args:
            chunks: List of CAD chunks

        Returns:
            JSONL string (one JSON object per line)
        """
        lines = [self.serialize_one(chunk) for chunk in chunks]
        return "\n".join(lines)

    def serialize_one(self, chunk: CADChunk) -> str:
        """Serialize single chunk to JSON line.

        Args:
            chunk: CAD chunk

        Returns:
            JSON string (single line)
        """
        chunk_dict = self._chunk_to_dict(chunk)
        return json.dumps(chunk_dict)

    def _chunk_to_dict(self, chunk: CADChunk) -> dict:
        """Convert chunk to dictionary.

        Args:
            chunk: CAD chunk

        Returns:
            Dictionary representation
        """
        data = {
            "chunk_id": chunk.chunk_id,
            "doc_name": chunk.doc_name,
            "text": chunk.text,
        }

        if self.include_metadata and chunk.meta:
            data["metadata"] = {
                "entity_types": chunk.meta.entity_types,
                "entity_ids": chunk.meta.entity_ids,
                "properties": chunk.meta.properties,
            }

            if chunk.meta.topology_subgraph:
                data["metadata"]["topology_subgraph"] = chunk.meta.topology_subgraph

            if chunk.meta.embedding:
                data["metadata"]["embedding"] = chunk.meta.embedding

        return data


class MarkdownSerializer(ChunkSerializer):
    """Markdown format serializer.

    Serializes chunks as a Markdown document.
    """

    def __init__(
        self,
        include_metadata: bool = True,
        include_toc: bool = True,
        heading_level: int = 2,
    ):
        """Initialize Markdown serializer.

        Args:
            include_metadata: Include chunk metadata as tables
            include_toc: Include table of contents
            heading_level: Heading level for chunks (1-6)
        """
        self.include_metadata = include_metadata
        self.include_toc = include_toc
        self.heading_level = max(1, min(6, heading_level))

    def serialize(self, chunks: List[CADChunk]) -> str:
        """Serialize chunks to Markdown.

        Args:
            chunks: List of CAD chunks

        Returns:
            Markdown string
        """
        lines = ["# CAD Chunks\n"]

        # Table of contents
        if self.include_toc and len(chunks) > 1:
            lines.append("## Table of Contents\n")
            for i, chunk in enumerate(chunks):
                chunk_title = chunk.chunk_id or f"Chunk {i + 1}"
                lines.append(
                    f"{i + 1}. [{chunk_title}](#{self._make_anchor(chunk_title)})"
                )
            lines.append("\n---\n")

        # Chunks
        for i, chunk in enumerate(chunks):
            lines.append(self.serialize_one(chunk))
            if i < len(chunks) - 1:
                lines.append("\n---\n")

        return "\n".join(lines)

    def serialize_one(self, chunk: CADChunk) -> str:
        """Serialize single chunk to Markdown.

        Args:
            chunk: CAD chunk

        Returns:
            Markdown string
        """
        lines = []

        # Heading
        heading_prefix = "#" * self.heading_level
        chunk_title = chunk.chunk_id or "Unnamed Chunk"
        lines.append(f"{heading_prefix} {chunk_title}\n")

        # Metadata table
        if self.include_metadata and chunk.meta:
            lines.append("**Metadata:**\n")
            lines.append("| Property | Value |")
            lines.append("|----------|-------|")

            if chunk.doc_name:
                lines.append(f"| Document | `{chunk.doc_name}` |")

            if chunk.meta.entity_types:
                entity_types_str = ", ".join(set(chunk.meta.entity_types))
                lines.append(f"| Entity Types | {entity_types_str} |")

            if chunk.meta.entity_ids:
                lines.append(f"| Entity Count | {len(chunk.meta.entity_ids)} |")

            if chunk.meta.properties:
                for key, value in chunk.meta.properties.items():
                    lines.append(f"| {key} | {value} |")

            lines.append("")

        # Content
        lines.append("**Content:**\n")
        lines.append("```")
        lines.append(chunk.text)
        lines.append("```\n")

        return "\n".join(lines)

    def _make_anchor(self, text: str) -> str:
        """Convert text to markdown anchor.

        Args:
            text: Heading text

        Returns:
            Anchor string
        """
        # Convert to lowercase, replace spaces with hyphens
        anchor = text.lower().replace(" ", "-")
        # Remove special characters
        anchor = "".join(c for c in anchor if c.isalnum() or c == "-")
        return anchor


class HTMLSerializer(ChunkSerializer):
    """HTML format serializer.

    Serializes chunks as an HTML document.
    """

    def __init__(
        self,
        include_metadata: bool = True,
        include_styles: bool = True,
        syntax_highlighting: bool = False,
    ):
        """Initialize HTML serializer.

        Args:
            include_metadata: Include chunk metadata
            include_styles: Include CSS styles
            syntax_highlighting: Enable syntax highlighting (requires pygments)
        """
        self.include_metadata = include_metadata
        self.include_styles = include_styles
        self.syntax_highlighting = syntax_highlighting

    def serialize(self, chunks: List[CADChunk]) -> str:
        """Serialize chunks to HTML.

        Args:
            chunks: List of CAD chunks

        Returns:
            HTML string
        """
        lines = ["<!DOCTYPE html>", "<html>", "<head>"]
        lines.append("<meta charset='utf-8'>")
        lines.append("<title>CAD Chunks</title>")

        if self.include_styles:
            lines.append(self._get_styles())

        lines.append("</head>")
        lines.append("<body>")
        lines.append("<h1>CAD Chunks</h1>")

        # Chunks
        for chunk in chunks:
            lines.append(self.serialize_one(chunk))

        lines.append("</body>")
        lines.append("</html>")

        return "\n".join(lines)

    def serialize_one(self, chunk: CADChunk) -> str:
        """Serialize single chunk to HTML.

        Args:
            chunk: CAD chunk

        Returns:
            HTML string
        """
        lines = ["<div class='chunk'>"]

        # Heading
        chunk_title = chunk.chunk_id or "Unnamed Chunk"
        lines.append(f"<h2>{self._escape_html(chunk_title)}</h2>")

        # Metadata
        if self.include_metadata and chunk.meta:
            lines.append("<div class='metadata'>")
            lines.append("<h3>Metadata</h3>")
            lines.append("<table>")

            if chunk.doc_name:
                lines.append(
                    f"<tr><td>Document</td><td><code>{self._escape_html(chunk.doc_name)}</code></td></tr>"
                )

            if chunk.meta.entity_types:
                entity_types_str = ", ".join(set(chunk.meta.entity_types))
                lines.append(
                    f"<tr><td>Entity Types</td><td>{self._escape_html(entity_types_str)}</td></tr>"
                )

            if chunk.meta.entity_ids:
                lines.append(
                    f"<tr><td>Entity Count</td><td>{len(chunk.meta.entity_ids)}</td></tr>"
                )

            if chunk.meta.properties:
                for key, value in chunk.meta.properties.items():
                    lines.append(
                        f"<tr><td>{self._escape_html(str(key))}</td><td>{self._escape_html(str(value))}</td></tr>"
                    )

            lines.append("</table>")
            lines.append("</div>")

        # Content
        lines.append("<div class='content'>")
        lines.append("<h3>Content</h3>")

        if self.syntax_highlighting:
            lines.append(self._highlight_code(chunk.text))
        else:
            lines.append(f"<pre><code>{self._escape_html(chunk.text)}</code></pre>")

        lines.append("</div>")
        lines.append("</div>")

        return "\n".join(lines)

    def _get_styles(self) -> str:
        """Get CSS styles.

        Returns:
            CSS style tag
        """
        return """
<style>
    body {
        font-family: Arial, sans-serif;
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
        background-color: #f5f5f5;
    }
    h1 {
        color: #333;
        border-bottom: 3px solid #007bff;
        padding-bottom: 10px;
    }
    .chunk {
        background-color: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chunk h2 {
        color: #007bff;
        margin-top: 0;
    }
    .metadata {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 15px;
    }
    .metadata h3 {
        margin-top: 0;
        color: #555;
    }
    table {
        width: 100%;
        border-collapse: collapse;
    }
    table td {
        padding: 8px;
        border-bottom: 1px solid #ddd;
    }
    table td:first-child {
        font-weight: bold;
        width: 200px;
    }
    .content pre {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 4px;
        overflow-x: auto;
    }
    .content code {
        font-family: 'Courier New', monospace;
        font-size: 14px;
    }
</style>
"""

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters.

        Args:
            text: Input text

        Returns:
            Escaped text
        """
        return html.escape(text, quote=True)

    def _highlight_code(self, text: str) -> str:
        """Apply syntax highlighting to code.

        Args:
            text: Code text

        Returns:
            Highlighted HTML
        """
        try:
            from pygments import highlight
            from pygments.lexers import guess_lexer
            from pygments.formatters import HtmlFormatter

            lexer = guess_lexer(text)
            formatter = HtmlFormatter(style="colorful")
            return highlight(text, lexer, formatter)
        except ImportError:
            _log.warning("pygments not installed, syntax highlighting disabled")
            return f"<pre><code>{self._escape_html(text)}</code></pre>"
        except Exception as e:
            _log.warning(f"Syntax highlighting failed: {e}")
            return f"<pre><code>{self._escape_html(text)}</code></pre>"


def get_serializer(format: str = "json", **kwargs) -> ChunkSerializer:
    """Factory function to get serializer instance.

    Args:
        format: Output format ("json", "jsonl", "markdown", "html")
        **kwargs: Additional arguments for serializer

    Returns:
        ChunkSerializer instance
    """
    format = format.lower()

    if format == "json":
        return JSONSerializer(**kwargs)
    elif format == "jsonl":
        return JSONLSerializer(**kwargs)
    elif format == "markdown" or format == "md":
        return MarkdownSerializer(**kwargs)
    elif format == "html":
        return HTMLSerializer(**kwargs)
    else:
        _log.warning(f"Unknown format: {format}, using JSON")
        return JSONSerializer(**kwargs)
