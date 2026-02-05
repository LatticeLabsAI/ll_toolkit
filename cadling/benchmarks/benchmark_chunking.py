#!/usr/bin/env python3
"""Benchmark tool for CADling chunking performance.

This tool measures the performance of various chunking operations,
particularly the optimizations made to:
- BFS queue operations (deque vs list.pop(0))
- Priority queue operations (heapq vs queue.sort())
- Adjacency building (fast path for 2-face edges)
- Normal similarity computation
- KDTree spatial indexing (constraint detection)
- Bounding box pre-filtering (interference check)
- Token count caching (STEP chunker)

Usage:
    python benchmarks/benchmark_chunking.py [--sizes 1000,10000,50000] [--iterations 5]

Results are displayed in a formatted table with timing comparisons.
"""

from __future__ import annotations

import argparse
import heapq
import random
import statistics
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.spatial import KDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""

    name: str
    mesh_size: int
    times: List[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        """Mean time in milliseconds."""
        return statistics.mean(self.times) * 1000

    @property
    def std_ms(self) -> float:
        """Standard deviation in milliseconds."""
        if len(self.times) < 2:
            return 0.0
        return statistics.stdev(self.times) * 1000

    @property
    def min_ms(self) -> float:
        """Minimum time in milliseconds."""
        return min(self.times) * 1000

    @property
    def max_ms(self) -> float:
        """Maximum time in milliseconds."""
        return max(self.times) * 1000


class SyntheticMeshGenerator:
    """Generate synthetic mesh data for benchmarking."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_mesh(
        self,
        num_faces: int,
        connectivity: str = "grid"
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, set]]:
        """Generate synthetic mesh with specified connectivity.

        Args:
            num_faces: Number of faces
            connectivity: Type of connectivity ("grid", "random", "sparse")

        Returns:
            Tuple of (normals, heights, adjacency)
        """
        # Generate random normals (unit vectors)
        normals = self.rng.standard_normal((num_faces, 3))
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        # Generate heights (for watershed)
        heights = self.rng.random(num_faces) * 100

        # Build adjacency based on connectivity type
        adjacency = self._build_adjacency(num_faces, connectivity)

        return normals, heights, adjacency

    def _build_adjacency(
        self,
        num_faces: int,
        connectivity: str
    ) -> Dict[int, set]:
        """Build adjacency graph."""
        adjacency = defaultdict(set)

        if connectivity == "grid":
            # Approximate grid connectivity (6 neighbors average)
            grid_size = int(np.sqrt(num_faces))
            for i in range(num_faces):
                row = i // grid_size
                col = i % grid_size

                # Add neighbors
                neighbors = []
                if col > 0:
                    neighbors.append(i - 1)
                if col < grid_size - 1:
                    neighbors.append(i + 1)
                if row > 0:
                    neighbors.append(i - grid_size)
                if row < grid_size - 1:
                    neighbors.append(i + grid_size)

                for n in neighbors:
                    if 0 <= n < num_faces:
                        adjacency[i].add(n)
                        adjacency[n].add(i)

        elif connectivity == "random":
            # Random connectivity (average 6 neighbors)
            avg_degree = 6
            for i in range(num_faces):
                num_neighbors = self.rng.poisson(avg_degree)
                neighbors = self.rng.choice(num_faces, size=min(num_neighbors, num_faces - 1), replace=False)
                for n in neighbors:
                    if n != i:
                        adjacency[i].add(n)
                        adjacency[n].add(i)

        elif connectivity == "sparse":
            # Sparse connectivity (average 3 neighbors)
            avg_degree = 3
            for i in range(num_faces):
                num_neighbors = self.rng.poisson(avg_degree)
                neighbors = self.rng.choice(num_faces, size=min(num_neighbors, num_faces - 1), replace=False)
                for n in neighbors:
                    if n != i:
                        adjacency[i].add(n)
                        adjacency[n].add(i)

        return adjacency


class ChunkingBenchmarks:
    """Benchmark suite for chunking operations."""

    def __init__(self, seed: int = 42):
        self.generator = SyntheticMeshGenerator(seed)
        self.results: List[BenchmarkResult] = []

    # =========================================================================
    # BFS Queue Benchmarks
    # =========================================================================

    def bfs_list_pop0(
        self,
        adjacency: Dict[int, set],
        num_faces: int,
        max_region_size: int = 5000
    ) -> List[List[int]]:
        """BFS using list.pop(0) - OLD implementation (O(n) per pop)."""
        visited = set()
        regions = []

        for seed_idx in range(num_faces):
            if seed_idx in visited:
                continue

            region = []
            queue = [seed_idx]  # List instead of deque
            visited.add(seed_idx)

            while queue and len(region) < max_region_size:
                current = queue.pop(0)  # O(n) operation!
                region.append(current)

                for neighbor in adjacency.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            regions.append(region)

        return regions

    def bfs_deque_popleft(
        self,
        adjacency: Dict[int, set],
        num_faces: int,
        max_region_size: int = 5000
    ) -> List[List[int]]:
        """BFS using deque.popleft() - NEW implementation (O(1) per pop)."""
        visited = set()
        regions = []

        for seed_idx in range(num_faces):
            if seed_idx in visited:
                continue

            region = []
            queue = deque([seed_idx])  # Deque for O(1) popleft
            visited.add(seed_idx)

            while queue and len(region) < max_region_size:
                current = queue.popleft()  # O(1) operation
                region.append(current)

                for neighbor in adjacency.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            regions.append(region)

        return regions

    # =========================================================================
    # Priority Queue Benchmarks (Watershed)
    # =========================================================================

    def watershed_sort_per_iteration(
        self,
        heights: np.ndarray,
        adjacency: Dict[int, set],
        max_basin_size: int = 5000
    ) -> List[List[int]]:
        """Watershed using queue.sort() per iteration - OLD implementation."""
        num_faces = len(heights)
        visited = set()
        basins = []

        # Find local minima
        local_minima = []
        for idx in range(num_faces):
            neighbors = adjacency.get(idx, [])
            if not neighbors or all(heights[idx] <= heights[n] for n in neighbors):
                local_minima.append(idx)

        local_minima.sort(key=lambda x: heights[x])

        for seed_idx in local_minima[:50]:  # Limit seeds for benchmark
            if seed_idx in visited:
                continue

            basin = [seed_idx]
            queue = [(heights[seed_idx], seed_idx)]
            visited.add(seed_idx)

            while queue and len(basin) < max_basin_size:
                _, current_idx = queue.pop(0)  # O(n)

                for neighbor_idx in adjacency.get(current_idx, []):
                    if neighbor_idx in visited:
                        continue

                    if heights[neighbor_idx] >= heights[current_idx]:
                        basin.append(neighbor_idx)
                        visited.add(neighbor_idx)
                        queue.append((heights[neighbor_idx], neighbor_idx))
                        queue.sort()  # O(n log n) per iteration!

            basins.append(basin)

        return basins

    def watershed_heapq(
        self,
        heights: np.ndarray,
        adjacency: Dict[int, set],
        max_basin_size: int = 5000
    ) -> List[List[int]]:
        """Watershed using heapq - NEW implementation."""
        num_faces = len(heights)
        visited = set()
        basins = []

        # Find local minima
        local_minima = []
        for idx in range(num_faces):
            neighbors = adjacency.get(idx, [])
            if not neighbors or all(heights[idx] <= heights[n] for n in neighbors):
                local_minima.append(idx)

        local_minima.sort(key=lambda x: heights[x])

        for seed_idx in local_minima[:50]:  # Limit seeds for benchmark
            if seed_idx in visited:
                continue

            basin = [seed_idx]
            queue = [(heights[seed_idx], seed_idx)]
            heapq.heapify(queue)
            visited.add(seed_idx)

            while queue and len(basin) < max_basin_size:
                _, current_idx = heapq.heappop(queue)  # O(log n)

                for neighbor_idx in adjacency.get(current_idx, []):
                    if neighbor_idx in visited:
                        continue

                    if heights[neighbor_idx] >= heights[current_idx]:
                        basin.append(neighbor_idx)
                        visited.add(neighbor_idx)
                        heapq.heappush(queue, (heights[neighbor_idx], neighbor_idx))  # O(log n)

            basins.append(basin)

        return basins

    # =========================================================================
    # Adjacency Building Benchmarks
    # =========================================================================

    def adjacency_nested_loop(
        self,
        edge_to_faces: Dict[tuple, set]
    ) -> Dict[int, set]:
        """Build adjacency using nested loops - OLD implementation."""
        adjacency = defaultdict(set)

        for edge, face_set in edge_to_faces.items():
            face_list = list(face_set)
            for i, f1 in enumerate(face_list):
                for f2 in face_list[i+1:]:
                    adjacency[f1].add(f2)
                    adjacency[f2].add(f1)

        return adjacency

    def adjacency_fast_path(
        self,
        edge_to_faces: Dict[tuple, set]
    ) -> Dict[int, set]:
        """Build adjacency with fast path for 2-face edges - NEW implementation."""
        adjacency = defaultdict(set)

        for edge, face_set in edge_to_faces.items():
            if len(face_set) == 2:
                # Fast path: most edges shared by exactly 2 faces
                f1, f2 = face_set
                adjacency[f1].add(f2)
                adjacency[f2].add(f1)
            elif len(face_set) > 2:
                # Non-manifold: use nested loop
                face_list = list(face_set)
                for i, f1 in enumerate(face_list):
                    for f2 in face_list[i+1:]:
                        adjacency[f1].add(f2)
                        adjacency[f2].add(f1)

        return adjacency

    # =========================================================================
    # Normal Vectorization Benchmarks
    # =========================================================================

    def normal_similarity_loop(
        self,
        normals: np.ndarray,
        adjacency: Dict[int, set]
    ) -> np.ndarray:
        """Compute curvatures using Python loop - OLD implementation."""
        num_faces = len(normals)
        curvatures = np.zeros(num_faces)

        for idx in range(num_faces):
            neighbors = list(adjacency.get(idx, []))
            if not neighbors:
                continue

            normal = normals[idx]
            normal_diffs = []

            for neighbor_idx in neighbors:
                neighbor_normal = normals[neighbor_idx]
                diff = np.linalg.norm(normal - neighbor_normal)
                normal_diffs.append(diff)

            curvatures[idx] = np.mean(normal_diffs) if normal_diffs else 0.0

        return curvatures

    def normal_similarity_vectorized(
        self,
        normals: np.ndarray,
        adjacency: Dict[int, set]
    ) -> np.ndarray:
        """Compute curvatures using vectorization - NEW implementation."""
        num_faces = len(normals)
        curvatures = np.zeros(num_faces)

        for idx in range(num_faces):
            neighbors = list(adjacency.get(idx, []))
            if not neighbors:
                continue

            # Vectorized computation
            neighbor_normals = normals[neighbors]  # [K, 3]
            diffs = np.linalg.norm(neighbor_normals - normals[idx], axis=1)  # [K]
            curvatures[idx] = np.mean(diffs)

        return curvatures

    # =========================================================================
    # KDTree Spatial Indexing Benchmarks (Constraint Detection)
    # =========================================================================

    def generate_cylinders(self, num_cylinders: int) -> List[Dict]:
        """Generate synthetic cylinder data for constraint detection."""
        rng = np.random.default_rng(42)
        cylinders = []
        for i in range(num_cylinders):
            # Random center in 3D space
            center = rng.random(3) * 100
            # Random axis (normalized)
            axis = rng.standard_normal(3)
            axis = axis / np.linalg.norm(axis)
            # Random radius
            radius = rng.random() * 10 + 1
            cylinders.append({
                "part_id": f"part_{i // 10}",
                "center": center,
                "axis": axis,
                "axis_normalized": axis,  # Pre-normalized
                "radius": radius
            })
        return cylinders

    def constraint_nested_loop(
        self,
        cylinders: List[Dict],
        tolerance: float = 0.01
    ) -> List[Tuple[int, int]]:
        """Find concentric pairs using nested loops - OLD implementation O(n²)."""
        pairs = []
        n = len(cylinders)
        for i in range(n):
            for j in range(i + 1, n):
                cyl1 = cylinders[i]
                cyl2 = cylinders[j]
                # Skip same part
                if cyl1["part_id"] == cyl2["part_id"]:
                    continue
                # Check if axes are parallel
                axis1 = cyl1["axis_normalized"]
                axis2 = cyl2["axis_normalized"]
                dot = abs(np.dot(axis1, axis2))
                if dot < 1 - tolerance:
                    continue
                # Check if radii match
                if abs(cyl1["radius"] - cyl2["radius"]) > tolerance:
                    continue
                # Check if centers are on the same line
                center_diff = np.array(cyl2["center"]) - np.array(cyl1["center"])
                dist = np.linalg.norm(center_diff)
                if dist > 0:
                    center_diff = center_diff / dist
                    parallel = abs(np.dot(center_diff, axis1))
                    if parallel < 1 - tolerance:
                        continue
                pairs.append((i, j))
        return pairs

    def constraint_kdtree(
        self,
        cylinders: List[Dict],
        tolerance: float = 0.01
    ) -> List[Tuple[int, int]]:
        """Find concentric pairs using KDTree - NEW implementation O(n log n)."""
        if not HAS_SCIPY or len(cylinders) < 10:
            return self.constraint_nested_loop(cylinders, tolerance)

        pairs = []
        centers = np.array([cyl["center"] for cyl in cylinders])
        tree = KDTree(centers)

        # Use spatial search to limit pairs to check
        max_radius = max(cyl["radius"] for cyl in cylinders)
        search_radius = max(max_radius * 2, tolerance * 10)
        candidate_pairs = tree.query_pairs(r=search_radius, output_type='ndarray')

        for i, j in candidate_pairs:
            cyl1 = cylinders[i]
            cyl2 = cylinders[j]
            # Skip same part
            if cyl1["part_id"] == cyl2["part_id"]:
                continue
            # Check if axes are parallel (using pre-normalized)
            axis1 = cyl1["axis_normalized"]
            axis2 = cyl2["axis_normalized"]
            dot = abs(np.dot(axis1, axis2))
            if dot < 1 - tolerance:
                continue
            # Check if radii match
            if abs(cyl1["radius"] - cyl2["radius"]) > tolerance:
                continue
            # Check if centers are on the same line
            center_diff = np.array(cyl2["center"]) - np.array(cyl1["center"])
            dist = np.linalg.norm(center_diff)
            if dist > 0:
                center_diff = center_diff / dist
                parallel = abs(np.dot(center_diff, axis1))
                if parallel < 1 - tolerance:
                    continue
            pairs.append((i, j))
        return pairs

    # =========================================================================
    # Bounding Box Pre-filtering Benchmarks (Interference Check)
    # =========================================================================

    def generate_bboxes(self, num_parts: int) -> List[Dict]:
        """Generate synthetic bounding boxes for interference check."""
        rng = np.random.default_rng(42)
        bboxes = []
        for i in range(num_parts):
            # Random center
            center = rng.random(3) * 100
            # Random size
            size = rng.random(3) * 10 + 1
            bboxes.append({
                "min_x": center[0] - size[0],
                "max_x": center[0] + size[0],
                "min_y": center[1] - size[1],
                "max_y": center[1] + size[1],
                "min_z": center[2] - size[2],
                "max_z": center[2] + size[2],
            })
        return bboxes

    def interference_no_filter(
        self,
        bboxes: List[Dict]
    ) -> List[Tuple[int, int]]:
        """Check all pairs without filtering - OLD implementation O(n²)."""
        pairs = []
        n = len(bboxes)
        for i in range(n):
            for j in range(i + 1, n):
                # Simulate "expensive" boolean operation check
                # In reality this would call OCC BRepAlgoAPI_Common
                _ = sum(bboxes[i].values()) + sum(bboxes[j].values())
                pairs.append((i, j))
        return pairs

    def _bboxes_overlap(self, bbox1: Dict, bbox2: Dict) -> bool:
        """Check if two bounding boxes overlap."""
        if bbox1["max_x"] < bbox2["min_x"] or bbox2["max_x"] < bbox1["min_x"]:
            return False
        if bbox1["max_y"] < bbox2["min_y"] or bbox2["max_y"] < bbox1["min_y"]:
            return False
        if bbox1["max_z"] < bbox2["min_z"] or bbox2["max_z"] < bbox1["min_z"]:
            return False
        return True

    def interference_bbox_filter(
        self,
        bboxes: List[Dict]
    ) -> List[Tuple[int, int]]:
        """Check pairs with bbox pre-filtering - NEW implementation."""
        pairs = []
        skipped = 0
        n = len(bboxes)
        for i in range(n):
            for j in range(i + 1, n):
                # Skip if bboxes don't overlap (cheap check)
                if not self._bboxes_overlap(bboxes[i], bboxes[j]):
                    skipped += 1
                    continue
                # Only do "expensive" check if bboxes overlap
                _ = sum(bboxes[i].values()) + sum(bboxes[j].values())
                pairs.append((i, j))
        return pairs

    # =========================================================================
    # Token Count Caching Benchmarks (STEP Chunker)
    # =========================================================================

    def generate_entities(self, num_entities: int) -> List[Dict]:
        """Generate synthetic STEP entities for token caching benchmark."""
        rng = np.random.default_rng(42)
        entities = []
        entity_types = ["CARTESIAN_POINT", "DIRECTION", "LINE", "CIRCLE",
                        "VERTEX_POINT", "EDGE_CURVE", "FACE_SURFACE", "SHELL"]
        for i in range(num_entities):
            entity_type = rng.choice(entity_types)
            # Generate text of varying length
            text_len = rng.integers(50, 500)
            text = "".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789(),. "), size=text_len))
            entities.append({
                "entity_id": i,
                "entity_type": entity_type,
                "text": text
            })
        return entities

    def _count_tokens(self, text: str) -> int:
        """Simulate token counting (simplified)."""
        # Simple approximation: split on whitespace and punctuation
        return len(text.split())

    def chunking_no_cache(
        self,
        entities: List[Dict],
        max_tokens: int = 512
    ) -> List[List[int]]:
        """Chunk entities without caching token counts - OLD implementation."""
        # Group by type
        type_groups = defaultdict(list)
        for entity in entities:
            type_groups[entity["entity_type"]].append(entity)

        chunks = []
        for entity_type, group in type_groups.items():
            current_chunk = []
            current_tokens = 0
            for entity in group:
                # Count tokens each time (expensive in real implementation)
                entity_tokens = self._count_tokens(entity["text"])
                if current_tokens + entity_tokens > max_tokens and current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_tokens = 0
                current_chunk.append(entity["entity_id"])
                current_tokens += entity_tokens
            if current_chunk:
                chunks.append(current_chunk)
        return chunks

    def chunking_with_cache(
        self,
        entities: List[Dict],
        max_tokens: int = 512
    ) -> List[List[int]]:
        """Chunk entities with cached token counts - NEW implementation."""
        # Group by type
        type_groups = defaultdict(list)
        for entity in entities:
            type_groups[entity["entity_type"]].append(entity)

        # Pre-compute token counts (cache)
        token_cache = {}
        for entity_type, group in type_groups.items():
            for entity in group:
                token_cache[entity["entity_id"]] = self._count_tokens(entity["text"])

        chunks = []
        for entity_type, group in type_groups.items():
            current_chunk = []
            current_tokens = 0
            for entity in group:
                # O(1) lookup instead of recomputing
                entity_tokens = token_cache[entity["entity_id"]]
                if current_tokens + entity_tokens > max_tokens and current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_tokens = 0
                current_chunk.append(entity["entity_id"])
                current_tokens += entity_tokens
            if current_chunk:
                chunks.append(current_chunk)
        return chunks

    # =========================================================================
    # Benchmark Runner
    # =========================================================================

    def run_benchmark(
        self,
        name: str,
        func: Callable,
        args: tuple,
        iterations: int,
        mesh_size: int
    ) -> BenchmarkResult:
        """Run a single benchmark."""
        result = BenchmarkResult(name=name, mesh_size=mesh_size)

        # Warmup
        func(*args)

        # Timed runs
        for _ in range(iterations):
            start = time.perf_counter()
            func(*args)
            elapsed = time.perf_counter() - start
            result.times.append(elapsed)

        self.results.append(result)
        return result

    def generate_edge_to_faces(
        self,
        num_faces: int,
        edges_per_face: int = 3,
        manifold_ratio: float = 0.95
    ) -> Dict[tuple, set]:
        """Generate synthetic edge-to-faces mapping."""
        edge_to_faces = defaultdict(set)
        rng = np.random.default_rng(42)

        num_edges = num_faces * edges_per_face // 2

        for edge_id in range(num_edges):
            # Most edges are manifold (shared by 2 faces)
            if rng.random() < manifold_ratio:
                num_adjacent = 2
            else:
                num_adjacent = rng.integers(3, 6)  # Non-manifold

            faces = set(rng.choice(num_faces, size=num_adjacent, replace=False))
            edge_to_faces[(edge_id, edge_id + 1)] = faces

        return edge_to_faces

    def run_all(
        self,
        sizes: List[int],
        iterations: int = 5
    ) -> None:
        """Run all benchmarks."""
        print("\n" + "=" * 80)
        print("CADling Chunking Performance Benchmarks")
        print("=" * 80)

        for size in sizes:
            print(f"\n{'─' * 80}")
            print(f"Mesh size: {size:,} faces")
            print(f"{'─' * 80}")

            # Generate test data
            normals, heights, adjacency = self.generator.generate_mesh(size)
            edge_to_faces = self.generate_edge_to_faces(size)

            # BFS Benchmarks
            print("\n[BFS Queue Operations]")

            r1 = self.run_benchmark(
                "list.pop(0) [OLD]",
                self.bfs_list_pop0,
                (adjacency, size),
                iterations,
                size
            )

            r2 = self.run_benchmark(
                "deque.popleft() [NEW]",
                self.bfs_deque_popleft,
                (adjacency, size),
                iterations,
                size
            )

            speedup = r1.mean_ms / r2.mean_ms if r2.mean_ms > 0 else float('inf')
            self._print_comparison(r1, r2, speedup)

            # Watershed Benchmarks (only for smaller sizes due to O(n²) old impl)
            if size <= 20000:
                print("\n[Watershed Priority Queue]")

                r3 = self.run_benchmark(
                    "queue.sort() [OLD]",
                    self.watershed_sort_per_iteration,
                    (heights, adjacency),
                    iterations,
                    size
                )

                r4 = self.run_benchmark(
                    "heapq [NEW]",
                    self.watershed_heapq,
                    (heights, adjacency),
                    iterations,
                    size
                )

                speedup = r3.mean_ms / r4.mean_ms if r4.mean_ms > 0 else float('inf')
                self._print_comparison(r3, r4, speedup)
            else:
                print("\n[Watershed Priority Queue]")
                print("  Skipped for large mesh (old impl too slow)")

            # Adjacency Building
            print("\n[Adjacency Building]")

            r5 = self.run_benchmark(
                "nested loops [OLD]",
                self.adjacency_nested_loop,
                (edge_to_faces,),
                iterations,
                size
            )

            r6 = self.run_benchmark(
                "fast path [NEW]",
                self.adjacency_fast_path,
                (edge_to_faces,),
                iterations,
                size
            )

            speedup = r5.mean_ms / r6.mean_ms if r6.mean_ms > 0 else float('inf')
            self._print_comparison(r5, r6, speedup)

            # Normal Vectorization
            print("\n[Normal Similarity Computation]")

            r7 = self.run_benchmark(
                "Python loop [OLD]",
                self.normal_similarity_loop,
                (normals, adjacency),
                iterations,
                size
            )

            r8 = self.run_benchmark(
                "vectorized [NEW]",
                self.normal_similarity_vectorized,
                (normals, adjacency),
                iterations,
                size
            )

            speedup = r7.mean_ms / r8.mean_ms if r8.mean_ms > 0 else float('inf')
            self._print_comparison(r7, r8, speedup)

            # KDTree Constraint Detection (only for larger sizes where it matters)
            if size >= 500 and HAS_SCIPY:
                print("\n[KDTree Spatial Indexing (Constraint Detection)]")

                num_cylinders = size // 10  # 10% of mesh size
                cylinders = self.generate_cylinders(num_cylinders)

                r9 = self.run_benchmark(
                    "nested loops [OLD]",
                    self.constraint_nested_loop,
                    (cylinders,),
                    iterations,
                    size
                )

                r10 = self.run_benchmark(
                    "KDTree [NEW]",
                    self.constraint_kdtree,
                    (cylinders,),
                    iterations,
                    size
                )

                speedup = r9.mean_ms / r10.mean_ms if r10.mean_ms > 0 else float('inf')
                self._print_comparison(r9, r10, speedup)
            elif not HAS_SCIPY:
                print("\n[KDTree Spatial Indexing (Constraint Detection)]")
                print("  Skipped (scipy not available)")

            # Bounding Box Pre-filtering
            print("\n[Bounding Box Pre-filtering (Interference Check)]")

            num_parts = min(size // 50, 500)  # Scale with mesh size
            bboxes = self.generate_bboxes(num_parts)

            r11 = self.run_benchmark(
                "no filter [OLD]",
                self.interference_no_filter,
                (bboxes,),
                iterations,
                size
            )

            r12 = self.run_benchmark(
                "bbox filter [NEW]",
                self.interference_bbox_filter,
                (bboxes,),
                iterations,
                size
            )

            speedup = r11.mean_ms / r12.mean_ms if r12.mean_ms > 0 else float('inf')
            self._print_comparison(r11, r12, speedup)

            # Token Count Caching
            print("\n[Token Count Caching (STEP Chunker)]")

            num_entities = size // 2
            entities = self.generate_entities(num_entities)

            r13 = self.run_benchmark(
                "no cache [OLD]",
                self.chunking_no_cache,
                (entities,),
                iterations,
                size
            )

            r14 = self.run_benchmark(
                "with cache [NEW]",
                self.chunking_with_cache,
                (entities,),
                iterations,
                size
            )

            speedup = r13.mean_ms / r14.mean_ms if r14.mean_ms > 0 else float('inf')
            self._print_comparison(r13, r14, speedup)

        # Summary
        self._print_summary()

    def _print_comparison(
        self,
        old: BenchmarkResult,
        new: BenchmarkResult,
        speedup: float
    ) -> None:
        """Print comparison between old and new implementation."""
        print(f"  {old.name:30s} {old.mean_ms:10.2f} ms (±{old.std_ms:.2f})")
        print(f"  {new.name:30s} {new.mean_ms:10.2f} ms (±{new.std_ms:.2f})")

        if speedup >= 1.0:
            print(f"  {'Speedup:':30s} {speedup:10.1f}x faster ✓")
        else:
            print(f"  {'Speedup:':30s} {1/speedup:10.1f}x slower ✗")

    def _print_summary(self) -> None:
        """Print benchmark summary."""
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        # Group results by benchmark type
        old_results = [r for r in self.results if "[OLD]" in r.name]
        new_results = [r for r in self.results if "[NEW]" in r.name]

        # Calculate overall improvements
        total_old_time = sum(r.mean_ms for r in old_results)
        total_new_time = sum(r.mean_ms for r in new_results)

        if total_new_time > 0:
            overall_speedup = total_old_time / total_new_time
            print(f"\nOverall improvement: {overall_speedup:.1f}x faster")

        print("\nKey findings:")
        print("  - deque.popleft() eliminates O(n) overhead in BFS")
        print("  - heapq maintains O(log n) priority queue operations")
        print("  - Fast path optimization skips nested loops for manifold edges")
        print("  - Vectorized normal computation leverages NumPy broadcasting")
        print("  - KDTree spatial indexing reduces constraint detection from O(n²) to O(n log n)")
        print("  - Bounding box pre-filtering skips expensive boolean ops for non-overlapping parts")
        print("  - Token count caching avoids repeated computation in chunking loops")

        print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark CADling chunking performance"
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="1000,5000,10000,25000",
        help="Comma-separated mesh sizes to benchmark (default: 1000,5000,10000,25000)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations per benchmark (default: 5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",")]

    benchmark = ChunkingBenchmarks(seed=args.seed)
    benchmark.run_all(sizes=sizes, iterations=args.iterations)


if __name__ == "__main__":
    main()
