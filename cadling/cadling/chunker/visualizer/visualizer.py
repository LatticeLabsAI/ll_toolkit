"""Chunk visualization utilities.

This module provides visualization tools for CAD chunks,
including 2D/3D plots, topology graphs, and chunk distribution analysis.

Classes:
    ChunkVisualizer: Main visualization class
    TopologyVisualizer: Visualize chunk topology graphs
    DistributionVisualizer: Visualize chunk distribution and statistics
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union, Tuple

import numpy as np

from cadling.chunker.base_chunker import CADChunk

_log = logging.getLogger(__name__)


class ChunkVisualizer:
    """Main chunk visualizer.

    Provides various visualization methods for CAD chunks.
    """

    def __init__(self):
        """Initialize chunk visualizer."""
        self.matplotlib_available = False
        self.plotly_available = False

        try:
            import matplotlib.pyplot as plt

            self.matplotlib_available = True
            self.plt = plt
        except ImportError:
            _log.warning("matplotlib not available, some visualizations disabled")

        try:
            import plotly.graph_objects as go

            self.plotly_available = True
            self.go = go
        except ImportError:
            _log.debug("plotly not available, interactive visualizations disabled")

    def visualize_chunk_sizes(
        self,
        chunks: List[CADChunk],
        output_path: Optional[Union[str, Path]] = None,
        interactive: bool = False,
    ):
        """Visualize chunk size distribution.

        Args:
            chunks: List of CAD chunks
            output_path: Output file path (optional)
            interactive: Use interactive plotly instead of matplotlib
        """
        if not chunks:
            _log.warning("No chunks to visualize")
            return

        # Compute chunk sizes (token counts)
        sizes = []
        labels = []

        for i, chunk in enumerate(chunks):
            size = len(chunk.text.split())  # Rough word count
            sizes.append(size)
            labels.append(chunk.chunk_id or f"Chunk {i}")

        if interactive and self.plotly_available:
            self._plot_sizes_interactive(sizes, labels, output_path)
        elif self.matplotlib_available:
            self._plot_sizes_matplotlib(sizes, labels, output_path)
        else:
            _log.error("No plotting library available")

    def visualize_chunk_types(
        self,
        chunks: List[CADChunk],
        output_path: Optional[Union[str, Path]] = None,
        interactive: bool = False,
    ):
        """Visualize entity type distribution across chunks.

        Args:
            chunks: List of CAD chunks
            output_path: Output file path (optional)
            interactive: Use interactive plotly instead of matplotlib
        """
        if not chunks:
            _log.warning("No chunks to visualize")
            return

        # Count entity types
        type_counts = {}

        for chunk in chunks:
            if chunk.meta and chunk.meta.entity_types:
                for entity_type in chunk.meta.entity_types:
                    type_counts[entity_type] = type_counts.get(entity_type, 0) + 1

        if not type_counts:
            _log.warning("No entity type data found")
            return

        types = list(type_counts.keys())
        counts = list(type_counts.values())

        if interactive and self.plotly_available:
            self._plot_types_interactive(types, counts, output_path)
        elif self.matplotlib_available:
            self._plot_types_matplotlib(types, counts, output_path)
        else:
            _log.error("No plotting library available")

    def visualize_spatial_chunks(
        self,
        chunks: List[CADChunk],
        output_path: Optional[Union[str, Path]] = None,
        interactive: bool = False,
    ):
        """Visualize spatial distribution of chunks (3D bounds).

        Args:
            chunks: List of CAD chunks
            output_path: Output file path (optional)
            interactive: Use interactive plotly instead of matplotlib
        """
        if not chunks:
            _log.warning("No chunks to visualize")
            return

        # Extract bounds from chunks
        bounds_data = []

        for chunk in chunks:
            if chunk.meta and chunk.meta.properties:
                props = chunk.meta.properties

                if "bounds_min" in props and "bounds_max" in props:
                    bounds_data.append(
                        {
                            "min": props["bounds_min"],
                            "max": props["bounds_max"],
                            "label": chunk.chunk_id,
                        }
                    )

        if not bounds_data:
            _log.warning("No spatial bounds data found")
            return

        if interactive and self.plotly_available:
            self._plot_spatial_interactive(bounds_data, output_path)
        elif self.matplotlib_available:
            self._plot_spatial_matplotlib(bounds_data, output_path)
        else:
            _log.error("No plotting library available")

    def _plot_sizes_matplotlib(
        self,
        sizes: List[int],
        labels: List[str],
        output_path: Optional[Union[str, Path]],
    ):
        """Plot chunk sizes using matplotlib.

        Args:
            sizes: List of chunk sizes
            labels: List of chunk labels
            output_path: Output file path
        """
        fig, (ax1, ax2) = self.plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart
        ax1.bar(range(len(sizes)), sizes, color="steelblue")
        ax1.set_xlabel("Chunk Index")
        ax1.set_ylabel("Size (words)")
        ax1.set_title("Chunk Sizes")
        ax1.grid(axis="y", alpha=0.3)

        # Histogram
        ax2.hist(sizes, bins=20, color="steelblue", edgecolor="black")
        ax2.set_xlabel("Size (words)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Chunk Size Distribution")
        ax2.grid(axis="y", alpha=0.3)

        self.plt.tight_layout()

        if output_path:
            self.plt.savefig(output_path, dpi=300, bbox_inches="tight")
            _log.info(f"Saved visualization to {output_path}")
        else:
            self.plt.show()

        self.plt.close()

    def _plot_sizes_interactive(
        self,
        sizes: List[int],
        labels: List[str],
        output_path: Optional[Union[str, Path]],
    ):
        """Plot chunk sizes using plotly.

        Args:
            sizes: List of chunk sizes
            labels: List of chunk labels
            output_path: Output file path
        """
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Chunk Sizes", "Size Distribution"),
            specs=[[{"type": "bar"}, {"type": "histogram"}]],
        )

        # Bar chart
        fig.add_trace(
            self.go.Bar(
                x=list(range(len(sizes))),
                y=sizes,
                name="Size",
                marker_color="steelblue",
            ),
            row=1,
            col=1,
        )

        # Histogram
        fig.add_trace(
            self.go.Histogram(x=sizes, name="Distribution", marker_color="steelblue"),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="Chunk Index", row=1, col=1)
        fig.update_yaxes(title_text="Size (words)", row=1, col=1)
        fig.update_xaxes(title_text="Size (words)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)

        fig.update_layout(
            height=500, showlegend=False, title_text="Chunk Size Analysis"
        )

        if output_path:
            fig.write_html(str(output_path))
            _log.info(f"Saved interactive visualization to {output_path}")
        else:
            fig.show()

    def _plot_types_matplotlib(
        self,
        types: List[str],
        counts: List[int],
        output_path: Optional[Union[str, Path]],
    ):
        """Plot entity types using matplotlib.

        Args:
            types: Entity types
            counts: Entity counts
            output_path: Output file path
        """
        fig, (ax1, ax2) = self.plt.subplots(1, 2, figsize=(14, 6))

        # Bar chart
        ax1.barh(types, counts, color="coral")
        ax1.set_xlabel("Count")
        ax1.set_ylabel("Entity Type")
        ax1.set_title("Entity Type Distribution")
        ax1.grid(axis="x", alpha=0.3)

        # Pie chart
        ax2.pie(counts, labels=types, autopct="%1.1f%%", startangle=90)
        ax2.set_title("Entity Type Proportions")

        self.plt.tight_layout()

        if output_path:
            self.plt.savefig(output_path, dpi=300, bbox_inches="tight")
            _log.info(f"Saved visualization to {output_path}")
        else:
            self.plt.show()

        self.plt.close()

    def _plot_types_interactive(
        self,
        types: List[str],
        counts: List[int],
        output_path: Optional[Union[str, Path]],
    ):
        """Plot entity types using plotly.

        Args:
            types: Entity types
            counts: Entity counts
            output_path: Output file path
        """
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Entity Type Distribution", "Entity Type Proportions"),
            specs=[[{"type": "bar"}, {"type": "pie"}]],
        )

        # Bar chart
        fig.add_trace(
            self.go.Bar(y=types, x=counts, orientation="h", marker_color="coral"),
            row=1,
            col=1,
        )

        # Pie chart
        fig.add_trace(self.go.Pie(labels=types, values=counts), row=1, col=2)

        fig.update_xaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Entity Type", row=1, col=1)

        fig.update_layout(
            height=500, showlegend=False, title_text="Entity Type Analysis"
        )

        if output_path:
            fig.write_html(str(output_path))
            _log.info(f"Saved interactive visualization to {output_path}")
        else:
            fig.show()

    def _plot_spatial_matplotlib(
        self,
        bounds_data: List[dict],
        output_path: Optional[Union[str, Path]],
    ):
        """Plot spatial bounds using matplotlib.

        Args:
            bounds_data: List of bounds dictionaries
            output_path: Output file path
        """
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = self.plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Plot bounding boxes
        colors = self.plt.cm.tab10(np.linspace(0, 1, len(bounds_data)))

        for i, bound in enumerate(bounds_data):
            min_pt = np.array(bound["min"])
            max_pt = np.array(bound["max"])

            # Create box vertices
            vertices = [
                [min_pt[0], min_pt[1], min_pt[2]],
                [max_pt[0], min_pt[1], min_pt[2]],
                [max_pt[0], max_pt[1], min_pt[2]],
                [min_pt[0], max_pt[1], min_pt[2]],
                [min_pt[0], min_pt[1], max_pt[2]],
                [max_pt[0], min_pt[1], max_pt[2]],
                [max_pt[0], max_pt[1], max_pt[2]],
                [min_pt[0], max_pt[1], max_pt[2]],
            ]

            # Draw edges
            edges = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
                [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
                [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
            ]

            ax.add_collection3d(
                Poly3DCollection(
                    edges,
                    facecolors=colors[i],
                    linewidths=1,
                    edgecolors="black",
                    alpha=0.3,
                )
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Spatial Chunk Distribution")

        if output_path:
            self.plt.savefig(output_path, dpi=300, bbox_inches="tight")
            _log.info(f"Saved visualization to {output_path}")
        else:
            self.plt.show()

        self.plt.close()

    def _plot_spatial_interactive(
        self,
        bounds_data: List[dict],
        output_path: Optional[Union[str, Path]],
    ):
        """Plot spatial bounds using plotly.

        Args:
            bounds_data: List of bounds dictionaries
            output_path: Output file path
        """
        fig = self.go.Figure()

        for i, bound in enumerate(bounds_data):
            min_pt = bound["min"]
            max_pt = bound["max"]

            # Create box mesh
            x = [
                min_pt[0],
                max_pt[0],
                max_pt[0],
                min_pt[0],
                min_pt[0],
                max_pt[0],
                max_pt[0],
                min_pt[0],
            ]
            y = [
                min_pt[1],
                min_pt[1],
                max_pt[1],
                max_pt[1],
                min_pt[1],
                min_pt[1],
                max_pt[1],
                max_pt[1],
            ]
            z = [
                min_pt[2],
                min_pt[2],
                min_pt[2],
                min_pt[2],
                max_pt[2],
                max_pt[2],
                max_pt[2],
                max_pt[2],
            ]

            # Define box faces
            i_faces = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
            j_faces = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
            k_faces = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]

            fig.add_trace(
                self.go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    i=i_faces,
                    j=j_faces,
                    k=k_faces,
                    name=bound.get("label", f"Chunk {i}"),
                    opacity=0.3,
                    showlegend=True,
                )
            )

        fig.update_layout(
            title="Spatial Chunk Distribution",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
            ),
            height=700,
        )

        if output_path:
            fig.write_html(str(output_path))
            _log.info(f"Saved interactive visualization to {output_path}")
        else:
            fig.show()


class TopologyVisualizer:
    """Visualize chunk topology graphs."""

    def __init__(self):
        """Initialize topology visualizer."""
        self.networkx_available = False

        try:
            import networkx as nx

            self.nx = nx
            self.networkx_available = True
        except ImportError:
            _log.warning("networkx not available, topology visualization disabled")

    def visualize_topology(
        self,
        chunks: List[CADChunk],
        output_path: Optional[Union[str, Path]] = None,
    ):
        """Visualize topology graph from chunks.

        Args:
            chunks: List of CAD chunks
            output_path: Output file path (optional)
        """
        if not self.networkx_available:
            _log.error("networkx required for topology visualization")
            return

        # Build graph from chunk topology
        G = self.nx.Graph()

        for chunk in chunks:
            if chunk.meta and chunk.meta.topology_subgraph:
                subgraph = chunk.meta.topology_subgraph

                # Add nodes
                if "nodes" in subgraph:
                    for node in subgraph["nodes"]:
                        G.add_node(node)

                # Add edges
                if "edges" in subgraph:
                    for edge in subgraph["edges"]:
                        if len(edge) >= 2:
                            G.add_edge(edge[0], edge[1])

        if G.number_of_nodes() == 0:
            _log.warning("No topology data found in chunks")
            return

        _log.info(
            f"Topology graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )

        # Draw graph
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 8))
            pos = self.nx.spring_layout(G)
            self.nx.draw(
                G,
                pos,
                with_labels=True,
                node_color="lightblue",
                node_size=500,
                font_size=8,
                font_weight="bold",
                edge_color="gray",
                alpha=0.7,
            )
            plt.title("Chunk Topology Graph")

            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                _log.info(f"Saved topology visualization to {output_path}")
            else:
                plt.show()

            plt.close()

        except ImportError:
            _log.error("matplotlib required for topology visualization")


class DistributionVisualizer:
    """Visualize chunk distribution statistics."""

    def print_statistics(self, chunks: List[CADChunk]):
        """Print chunk statistics.

        Args:
            chunks: List of CAD chunks
        """
        if not chunks:
            print("No chunks to analyze")
            return

        print(f"\n{'=' * 60}")
        print("CHUNK STATISTICS")
        print(f"{'=' * 60}\n")

        print(f"Total chunks: {len(chunks)}")

        # Size statistics
        sizes = [len(chunk.text.split()) for chunk in chunks]
        print(f"\nChunk sizes (words):")
        print(f"  Min: {min(sizes)}")
        print(f"  Max: {max(sizes)}")
        print(f"  Mean: {np.mean(sizes):.1f}")
        print(f"  Median: {np.median(sizes):.1f}")

        # Entity statistics
        entity_counts = [
            len(chunk.meta.entity_ids) if chunk.meta and chunk.meta.entity_ids else 0
            for chunk in chunks
        ]

        if any(entity_counts):
            print(f"\nEntities per chunk:")
            print(f"  Min: {min(entity_counts)}")
            print(f"  Max: {max(entity_counts)}")
            print(f"  Mean: {np.mean(entity_counts):.1f}")

        # Entity type distribution
        all_types = []
        for chunk in chunks:
            if chunk.meta and chunk.meta.entity_types:
                all_types.extend(chunk.meta.entity_types)

        if all_types:
            from collections import Counter

            type_counts = Counter(all_types)
            print(f"\nEntity type distribution:")
            for entity_type, count in type_counts.most_common(10):
                print(f"  {entity_type}: {count}")

        print(f"\n{'=' * 60}\n")
