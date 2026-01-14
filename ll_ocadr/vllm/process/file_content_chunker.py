"""
File content chunker for CAD/Mesh files.
Processes the actual file format content (like OCR processes document text).
"""

import struct
from pathlib import Path
from typing import List, Tuple, Dict, Union
import numpy as np


class STLContentChunker:
    """
    Chunk STL file content at the format level.
    Processes actual facet definitions (ASCII) or binary records.
    """

    def __init__(self, chunk_size: int = 1000):
        """
        Args:
            chunk_size: Number of facets per chunk
        """
        self.chunk_size = chunk_size

    def is_ascii_stl(self, file_path: str) -> bool:
        """Check if STL file is ASCII or binary."""
        with open(file_path, 'rb') as f:
            header = f.read(80)
            return b'solid' in header[:5]

    def chunk_ascii_stl(self, file_path: str) -> List[Dict]:
        """
        Chunk ASCII STL file by facet definitions.

        Returns list of chunks, each containing:
        - raw_content: The actual text lines
        - facets: Parsed facet data
        - start_facet: Starting facet index
        - end_facet: Ending facet index
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()

        chunks = []
        current_chunk_lines = []
        current_chunk_facets = []
        facet_count = 0
        chunk_start = 0

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith('facet normal'):
                # Start of a new facet
                facet_lines = [lines[i]]
                facet_data = {'normal': self._parse_vector(line)}

                # Parse the facet (should be: outer loop, 3 vertices, endloop, endfacet)
                i += 1
                vertices = []
                while i < len(lines):
                    inner_line = lines[i].strip()
                    facet_lines.append(lines[i])

                    if inner_line.startswith('vertex'):
                        vertices.append(self._parse_vector(inner_line))
                    elif inner_line.startswith('endfacet'):
                        facet_data['vertices'] = vertices
                        break
                    i += 1

                current_chunk_lines.extend(facet_lines)
                current_chunk_facets.append(facet_data)
                facet_count += 1

                # Check if chunk is full
                if len(current_chunk_facets) >= self.chunk_size:
                    chunks.append({
                        'raw_content': ''.join(current_chunk_lines),
                        'facets': current_chunk_facets,
                        'start_facet': chunk_start,
                        'end_facet': chunk_start + len(current_chunk_facets),
                        'format': 'ascii_stl'
                    })
                    chunk_start += len(current_chunk_facets)
                    current_chunk_lines = []
                    current_chunk_facets = []

            i += 1

        # Add remaining facets
        if current_chunk_facets:
            chunks.append({
                'raw_content': ''.join(current_chunk_lines),
                'facets': current_chunk_facets,
                'start_facet': chunk_start,
                'end_facet': chunk_start + len(current_chunk_facets),
                'format': 'ascii_stl'
            })

        return chunks

    def chunk_binary_stl(self, file_path: str) -> List[Dict]:
        """
        Chunk binary STL file by facet records.

        Binary STL format:
        - 80 byte header
        - 4 byte facet count
        - For each facet (50 bytes):
            - 12 bytes: normal (3 floats)
            - 36 bytes: 3 vertices (9 floats)
            - 2 bytes: attribute byte count
        """
        with open(file_path, 'rb') as f:
            header = f.read(80)
            num_facets = struct.unpack('<I', f.read(4))[0]

            chunks = []
            chunk_start = 0

            while chunk_start < num_facets:
                chunk_end = min(chunk_start + self.chunk_size, num_facets)
                num_in_chunk = chunk_end - chunk_start

                # Read raw binary data for this chunk
                chunk_data = f.read(50 * num_in_chunk)

                # Parse facets
                facets = []
                for i in range(num_in_chunk):
                    offset = i * 50
                    facet_data = chunk_data[offset:offset + 50]

                    # Parse normal (3 floats)
                    normal = struct.unpack('<3f', facet_data[0:12])

                    # Parse 3 vertices (9 floats)
                    v1 = struct.unpack('<3f', facet_data[12:24])
                    v2 = struct.unpack('<3f', facet_data[24:36])
                    v3 = struct.unpack('<3f', facet_data[36:48])

                    facets.append({
                        'normal': normal,
                        'vertices': [v1, v2, v3]
                    })

                chunks.append({
                    'raw_content': chunk_data,  # Raw binary bytes
                    'facets': facets,
                    'start_facet': chunk_start,
                    'end_facet': chunk_end,
                    'format': 'binary_stl'
                })

                chunk_start = chunk_end

        return chunks

    def chunk_stl(self, file_path: str) -> List[Dict]:
        """Chunk STL file (auto-detect ASCII or binary)."""
        if self.is_ascii_stl(file_path):
            return self.chunk_ascii_stl(file_path)
        else:
            return self.chunk_binary_stl(file_path)

    def _parse_vector(self, line: str) -> Tuple[float, float, float]:
        """Parse vector from STL line (e.g., 'vertex 1.0 2.0 3.0')."""
        parts = line.split()
        return (float(parts[-3]), float(parts[-2]), float(parts[-1]))


class STEPContentChunker:
    """
    Chunk STEP file content at the entity level.
    Processes actual STEP entity definitions (the #123 = ... lines).
    """

    def __init__(self, chunk_size: int = 1000):
        """
        Args:
            chunk_size: Number of entities per chunk
        """
        self.chunk_size = chunk_size

    def chunk_step(self, file_path: str) -> List[Dict]:
        """
        Chunk STEP file by entity definitions.

        STEP format:
        - Header section: ISO-10303-21;HEADER;...ENDSEC;
        - Data section: DATA;#1=ENTITY(...);#2=...;ENDSEC;

        Returns list of chunks, each containing:
        - raw_content: The actual entity lines
        - entities: Parsed entity data
        - start_entity: Starting entity number
        - end_entity: Ending entity number
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Split into header and data sections
        parts = content.split('DATA;')
        if len(parts) < 2:
            raise ValueError("Invalid STEP file: no DATA section found")

        header = parts[0]
        data_section = parts[1].split('ENDSEC;')[0]

        # Extract entities (lines starting with #)
        # Preserve EXACT original formatting including newlines and whitespace
        entity_lines = []
        current_entity_lines = []

        for line in data_section.split('\n'):
            # Check if line starts with # (new entity) - don't strip yet
            stripped = line.strip()

            if stripped.startswith('#'):
                # Save previous entity with original formatting
                if current_entity_lines:
                    entity_lines.append('\n'.join(current_entity_lines))
                current_entity_lines = [line]
            elif stripped:  # Non-empty continuation line
                current_entity_lines.append(line)
            # Skip completely empty lines between entities

        # Add final entity
        if current_entity_lines:
            entity_lines.append('\n'.join(current_entity_lines))

        # Chunk entities
        chunks = []
        for i in range(0, len(entity_lines), self.chunk_size):
            chunk_entities = entity_lines[i:i + self.chunk_size]

            # Parse entities
            parsed_entities = []
            for entity_line in chunk_entities:
                parsed = self._parse_entity(entity_line)
                if parsed:
                    parsed_entities.append(parsed)

            chunks.append({
                'raw_content': '\n'.join(chunk_entities),
                'entities': parsed_entities,
                'start_entity': parsed_entities[0]['id'] if parsed_entities else i,
                'end_entity': parsed_entities[-1]['id'] if parsed_entities else i + len(chunk_entities),
                'format': 'step'
            })

        return chunks

    def _parse_entity(self, entity_text: str) -> Dict:
        """
        Parse STEP entity (may be multi-line).
        Example: #123 = CARTESIAN_POINT('', (1.0, 2.0, 3.0));
        Or multi-line:
            #24=(
            BOUNDED_CURVE()
            ...
            );

        Returns:
            {
                'id': 123,
                'type': 'CARTESIAN_POINT',
                'content': "('', (1.0, 2.0, 3.0))"
            }
        """
        # Get first line for ID and type extraction
        first_line = entity_text.split('\n')[0].strip()

        if not first_line.startswith('#'):
            return None

        try:
            # Extract entity ID
            id_end = first_line.index('=')
            entity_id = int(first_line[1:id_end].strip())

            # Extract entity type from entire entity text (not just first line)
            # Remove newlines and extra spaces for type detection
            full_text = entity_text.replace('\n', ' ')
            rest = full_text[full_text.index('=') + 1:].strip()

            if '(' in rest:
                type_end = rest.index('(')
                entity_type = rest[:type_end].strip()
                entity_content = rest[type_end:]
            else:
                entity_type = rest.rstrip(';')
                entity_content = ""

            return {
                'id': entity_id,
                'type': entity_type,
                'content': entity_content
            }
        except (ValueError, IndexError):
            return None


class OBJContentChunker:
    """
    Chunk OBJ file content at the directive level.
    Processes actual OBJ directives (v, vn, vt, f, etc.).
    """

    def __init__(self, chunk_size: int = 1000):
        """
        Args:
            chunk_size: Number of faces per chunk
        """
        self.chunk_size = chunk_size

    def chunk_obj(self, file_path: str) -> List[Dict]:
        """
        Chunk OBJ file by face definitions.

        OBJ format:
        - v x y z (vertex)
        - vn x y z (normal)
        - vt u v (texture coordinate)
        - f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 (face)

        Returns list of chunks with vertices and faces.
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # First pass: collect all vertices and normals (needed for faces)
        vertices = []
        normals = []
        texcoords = []

        for line in lines:
            line = line.strip()
            if line.startswith('v '):
                vertices.append(line)
            elif line.startswith('vn '):
                normals.append(line)
            elif line.startswith('vt '):
                texcoords.append(line)

        # Second pass: chunk faces
        chunks = []
        current_chunk_lines = []
        face_count = 0
        chunk_start = 0

        for line in lines:
            line_stripped = line.strip()

            if line_stripped.startswith('f '):
                current_chunk_lines.append(line)
                face_count += 1

                if face_count >= self.chunk_size:
                    # Include relevant vertices/normals for this chunk
                    chunk_content = (
                        '\n'.join(vertices) + '\n' +
                        '\n'.join(normals) + '\n' +
                        '\n'.join(current_chunk_lines)
                    )

                    chunks.append({
                        'raw_content': chunk_content,
                        'num_vertices': len(vertices),
                        'num_normals': len(normals),
                        'num_faces': len(current_chunk_lines),
                        'start_face': chunk_start,
                        'end_face': chunk_start + face_count,
                        'format': 'obj'
                    })

                    chunk_start += face_count
                    current_chunk_lines = []
                    face_count = 0

        # Add remaining faces
        if current_chunk_lines:
            chunk_content = (
                '\n'.join(vertices) + '\n' +
                '\n'.join(normals) + '\n' +
                '\n'.join(current_chunk_lines)
            )

            chunks.append({
                'raw_content': chunk_content,
                'num_vertices': len(vertices),
                'num_normals': len(normals),
                'num_faces': len(current_chunk_lines),
                'start_face': chunk_start,
                'end_face': chunk_start + face_count,
                'format': 'obj'
            })

        return chunks


class UnifiedCADContentChunker:
    """
    Unified chunker that handles all CAD/Mesh file formats.
    Processes actual file format content, not just geometry.

    Automatically analyzes files to determine optimal chunk size,
    similar to DeepSeek-OCR's dynamic_preprocess.
    """

    # Token budget constraints (similar to vision model limits)
    MAX_TOKENS_PER_CHUNK = 384
    MIN_TOKENS_PER_CHUNK = 64
    MAX_TOTAL_CHUNKS = 64

    def __init__(self, chunk_size: int = None):
        """
        Args:
            chunk_size: Optional fixed chunk size. If None, size is determined dynamically per file.
        """
        self.fixed_chunk_size = chunk_size
        self.stl_chunker = None
        self.step_chunker = None
        self.obj_chunker = None

    def analyze_file(self, file_path: str) -> Dict:
        """
        Analyze file characteristics to determine optimal chunking strategy.
        Mirrors DeepSeek-OCR's image analysis (dimensions, complexity).

        Returns:
            {
                'total_entities': int,
                'file_format': str,
                'chunk_size': int,
                'num_chunks': int,
                'complexity': str,
                'tokens_per_chunk_est': int
            }
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext in ['.stl']:
            # Count facets
            with open(file_path, 'rb') as f:
                header = f.read(80)
                is_ascii = b'solid' in header[:5]

            if is_ascii:
                with open(file_path, 'r') as f:
                    total_entities = f.read().count('facet normal')
            else:
                with open(file_path, 'rb') as f:
                    f.read(80)
                    total_entities = struct.unpack('<I', f.read(4))[0]

            tokens_per_entity = 4  # facet ≈ 4 tokens
            entity_type = 'facet'
            file_format = 'stl'

        elif ext in ['.step', '.stp']:
            # Count entities
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            total_entities = content.count('\n#')

            # Analyze complexity
            complex_count = content.count('BREP') + content.count('NURBS') + content.count('B_SPLINE')
            complexity_ratio = complex_count / max(total_entities, 1)

            tokens_per_entity = 8 if complexity_ratio > 0.3 else 4
            entity_type = 'entity'
            file_format = 'step'

        elif ext in ['.obj']:
            with open(file_path, 'r') as f:
                total_entities = sum(1 for line in f if line.strip().startswith('f '))

            tokens_per_entity = 3
            entity_type = 'face'
            file_format = 'obj'
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        # Calculate optimal chunk size based on token budget
        optimal_chunk_size = self.MAX_TOKENS_PER_CHUNK // tokens_per_entity
        optimal_chunk_size = max(16, min(optimal_chunk_size, 256))

        num_chunks = max(1, min(
            self.MAX_TOTAL_CHUNKS,
            int(np.ceil(total_entities / optimal_chunk_size))
        ))

        # Recalculate chunk size to evenly distribute
        if num_chunks > 1:
            optimal_chunk_size = int(np.ceil(total_entities / num_chunks))

        # Determine complexity
        if num_chunks == 1:
            complexity = 'simple'
        elif num_chunks <= 8:
            complexity = 'moderate'
        else:
            complexity = 'complex'

        return {
            'total_entities': total_entities,
            'file_format': file_format,
            'chunk_size': optimal_chunk_size,
            'num_chunks': num_chunks,
            'complexity': complexity,
            'tokens_per_chunk_est': optimal_chunk_size * tokens_per_entity,
            'entity_type': entity_type
        }

    def chunk_file(self, file_path: str) -> List[Dict]:
        """
        Chunk file based on format.
        Automatically determines optimal chunk size if not fixed.

        Returns list of chunks with:
        - raw_content: Actual file content (text or binary)
        - format: File format
        - Additional format-specific data
        """
        # Determine chunk size
        if self.fixed_chunk_size is not None:
            chunk_size = self.fixed_chunk_size
        else:
            # Analyze file to determine optimal chunk size
            analysis = self.analyze_file(file_path)
            chunk_size = analysis['chunk_size']

        # Create chunkers with determined size
        self.stl_chunker = STLContentChunker(chunk_size)
        self.step_chunker = STEPContentChunker(chunk_size)
        self.obj_chunker = OBJContentChunker(chunk_size)

        path = Path(file_path)
        ext = path.suffix.lower()

        if ext in ['.stl']:
            return self.stl_chunker.chunk_stl(file_path)
        elif ext in ['.step', '.stp']:
            return self.step_chunker.chunk_step(file_path)
        elif ext in ['.obj']:
            return self.obj_chunker.chunk_obj(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def get_chunk_statistics(self, chunks: List[Dict]) -> Dict:
        """Get statistics about chunks."""
        if not chunks:
            return {}

        format_type = chunks[0]['format']
        total_content_size = sum(
            len(c['raw_content']) if isinstance(c['raw_content'], (str, bytes)) else 0
            for c in chunks
        )

        stats = {
            'num_chunks': len(chunks),
            'format': format_type,
            'total_content_size': total_content_size,
            'avg_chunk_size': total_content_size / len(chunks) if chunks else 0
        }

        if 'stl' in format_type:
            stats['total_facets'] = sum(len(c['facets']) for c in chunks)
        elif format_type == 'step':
            stats['total_entities'] = sum(len(c['entities']) for c in chunks)
        elif format_type == 'obj':
            stats['total_faces'] = sum(c['num_faces'] for c in chunks)

        return stats


# Example usage
if __name__ == "__main__":
    chunker = UnifiedCADContentChunker(chunk_size=1000)

    # Chunk a file
    chunks = chunker.chunk_file("example.stl")

    # Get statistics
    stats = chunker.get_chunk_statistics(chunks)
    print(f"Chunked {stats['num_chunks']} chunks")
    print(f"Format: {stats['format']}")

    # Access raw content of first chunk
    print(f"First chunk content length: {len(chunks[0]['raw_content'])}")
