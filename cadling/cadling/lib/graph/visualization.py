"""Graph visualization utilities for topology graphs.

Provides visualization functions for CAD topology graphs including:
- Adjacency graph visualization
- Entity relationship graphs
- Hierarchical layout visualization
- Node degree distribution plots
- Connected components visualization
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _has_matplotlib = True
except ImportError:
    plt = None
    mpatches = None
    _has_matplotlib = False

_log = logging.getLogger(__name__)


class TopologyGraphVisualizer:
    """Visualizer for CAD topology graphs.

    Creates visual representations of topology graphs showing:
    - Node connections (adjacency)
    - Entity types and relationships
    - Graph structure (components, hierarchy)
    - Statistical properties (degree distribution, connectivity)
    """

    def __init__(self, topology_graph: Any, output_dir: Optional[Path] = None):
        """Initialize topology graph visualizer.

        Args:
            topology_graph: TopologyGraph object with adjacency_list
            output_dir: Optional directory for saving visualizations
        """
        self.topology = topology_graph
        self.output_dir = output_dir
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        # Try to import networkx for graph layout
        try:
            import networkx as nx
            self.nx = nx
            self.has_networkx = True
        except ImportError:
            self.nx = None
            self.has_networkx = False
            _log.warning("networkx not available - limited visualization capabilities")

    def visualize_adjacency_graph(
        self,
        max_nodes: int = 100,
        layout: str = "spring",
        node_size: int = 50,
        figsize: Tuple[int, int] = (12, 10)
    ) -> Optional[Path]:
        """Visualize topology graph as adjacency network.

        Args:
            max_nodes: Maximum nodes to display (for readability)
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai')
            node_size: Size of nodes in visualization
            figsize: Figure size (width, height)

        Returns:
            Path to saved visualization or None
        """
        if not self.has_networkx:
            _log.error("networkx required for graph visualization")
            return None

        # Create networkx graph from adjacency list
        G = self.nx.DiGraph()

        # Add nodes and edges (limit to max_nodes for readability)
        node_subset = list(self.topology.adjacency_list.keys())[:max_nodes]
        for node in node_subset:
            G.add_node(node)
            neighbors = self.topology.adjacency_list.get(node, [])
            for neighbor in neighbors:
                if neighbor in node_subset:
                    G.add_edge(node, neighbor)

        # Calculate layout
        if layout == "spring":
            pos = self.nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        elif layout == "circular":
            pos = self.nx.circular_layout(G)
        elif layout == "kamada_kawai":
            pos = self.nx.kamada_kawai_layout(G)
        else:
            pos = self.nx.spring_layout(G, seed=42)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Draw graph
        self.nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color='lightblue',
            node_size=node_size,
            alpha=0.8
        )

        self.nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color='gray',
            alpha=0.5,
            arrows=True,
            arrowsize=10,
            width=0.5
        )

        # Add labels for subset of nodes (to avoid clutter)
        label_subset = {n: str(n) for i, n in enumerate(node_subset) if i % 5 == 0}
        self.nx.draw_networkx_labels(
            G, pos, label_subset, ax=ax,
            font_size=8,
            font_color='black'
        )

        # Calculate graph statistics
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = self.nx.density(G)
        num_components = self.nx.number_weakly_connected_components(G)

        # Add title and statistics
        ax.set_title(
            f"Topology Graph Visualization\n"
            f"Nodes: {num_nodes}/{self.topology.num_nodes} | "
            f"Edges: {num_edges}/{self.topology.num_edges} | "
            f"Density: {density:.4f} | "
            f"Components: {num_components}",
            fontsize=12,
            fontweight='bold'
        )
        ax.axis('off')

        plt.tight_layout()

        # Save figure
        if self.output_dir:
            output_path = self.output_dir / "topology_adjacency_graph.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            _log.info(f"Saved adjacency graph visualization to {output_path}")
            plt.close()
            return output_path
        else:
            plt.show()
            return None

    def visualize_degree_distribution(
        self,
        figsize: Tuple[int, int] = (12, 5)
    ) -> Optional[Path]:
        """Visualize in-degree and out-degree distributions.

        Args:
            figsize: Figure size (width, height)

        Returns:
            Path to saved visualization or None
        """
        # Calculate degree distributions
        out_degrees = []
        in_degrees = {}

        for node, neighbors in self.topology.adjacency_list.items():
            out_degrees.append(len(neighbors))
            for neighbor in neighbors:
                in_degrees[neighbor] = in_degrees.get(neighbor, 0) + 1

        in_degree_values = [in_degrees.get(n, 0) for n in self.topology.adjacency_list.keys()]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Out-degree distribution
        ax1.hist(out_degrees, bins=max(20, max(out_degrees) if out_degrees else 1),
                 color='skyblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Out-Degree', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.set_title(
            f'Out-Degree Distribution\n'
            f'Mean: {np.mean(out_degrees):.2f} | Max: {max(out_degrees) if out_degrees else 0}',
            fontsize=11,
            fontweight='bold'
        )
        ax1.grid(True, alpha=0.3)

        # In-degree distribution
        ax2.hist(in_degree_values, bins=max(20, max(in_degree_values) if in_degree_values else 1),
                 color='lightcoral', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('In-Degree', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.set_title(
            f'In-Degree Distribution\n'
            f'Mean: {np.mean(in_degree_values):.2f} | Max: {max(in_degree_values) if in_degree_values else 0}',
            fontsize=11,
            fontweight='bold'
        )
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        if self.output_dir:
            output_path = self.output_dir / "topology_degree_distribution.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            _log.info(f"Saved degree distribution to {output_path}")
            plt.close()
            return output_path
        else:
            plt.show()
            return None

    def visualize_connected_components(
        self,
        max_nodes_per_component: int = 50,
        figsize: Tuple[int, int] = (14, 10)
    ) -> Optional[Path]:
        """Visualize connected components of the graph.

        Args:
            max_nodes_per_component: Max nodes to show per component
            figsize: Figure size (width, height)

        Returns:
            Path to saved visualization or None
        """
        if not self.has_networkx:
            _log.error("networkx required for component visualization")
            return None

        # Create networkx graph
        G = self.nx.DiGraph()
        for node, neighbors in self.topology.adjacency_list.items():
            G.add_node(node)
            for neighbor in neighbors:
                G.add_edge(node, neighbor)

        # Find weakly connected components
        components = list(self.nx.weakly_connected_components(G))
        components = sorted(components, key=len, reverse=True)

        # Limit to top 6 components for visualization
        num_components = min(len(components), 6)

        if num_components == 0:
            _log.warning("No connected components found")
            return None

        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink', 'lavender']

        for i in range(num_components):
            component = list(components[i])[:max_nodes_per_component]
            subgraph = G.subgraph(component)

            # Calculate layout
            pos = self.nx.spring_layout(subgraph, k=0.5, iterations=30, seed=42)

            # Draw component
            self.nx.draw_networkx_nodes(
                subgraph, pos, ax=axes[i],
                node_color=colors[i],
                node_size=100,
                alpha=0.8
            )

            self.nx.draw_networkx_edges(
                subgraph, pos, ax=axes[i],
                edge_color='gray',
                alpha=0.5,
                arrows=True,
                arrowsize=8,
                width=0.5
            )

            axes[i].set_title(
                f'Component {i+1}\n'
                f'{len(components[i])} nodes, {subgraph.number_of_edges()} edges',
                fontsize=10,
                fontweight='bold'
            )
            axes[i].axis('off')

        # Hide unused subplots
        for i in range(num_components, 6):
            axes[i].axis('off')

        fig.suptitle(
            f'Connected Components Visualization\n'
            f'Total components: {len(components)} | Showing: {num_components}',
            fontsize=13,
            fontweight='bold'
        )

        plt.tight_layout()

        # Save figure
        if self.output_dir:
            output_path = self.output_dir / "topology_connected_components.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            _log.info(f"Saved connected components visualization to {output_path}")
            plt.close()
            return output_path
        else:
            plt.show()
            return None

    def visualize_adjacency_matrix(
        self,
        max_nodes: int = 200,
        figsize: Tuple[int, int] = (10, 9)
    ) -> Optional[Path]:
        """Visualize graph as adjacency matrix heatmap.

        Args:
            max_nodes: Maximum nodes to include in matrix
            figsize: Figure size (width, height)

        Returns:
            Path to saved visualization or None
        """
        # Build adjacency matrix
        node_list = list(self.topology.adjacency_list.keys())[:max_nodes]
        node_to_idx = {node: i for i, node in enumerate(node_list)}

        adjacency_matrix = np.zeros((len(node_list), len(node_list)))

        for node in node_list:
            if node in self.topology.adjacency_list:
                neighbors = self.topology.adjacency_list[node]
                for neighbor in neighbors:
                    if neighbor in node_to_idx:
                        adjacency_matrix[node_to_idx[node], node_to_idx[neighbor]] = 1

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot matrix
        im = ax.imshow(adjacency_matrix, cmap='Blues', aspect='auto', interpolation='nearest')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Connection (0=None, 1=Edge)', rotation=270, labelpad=20)

        # Calculate sparsity
        num_edges = np.sum(adjacency_matrix)
        sparsity = 1 - (num_edges / (len(node_list) ** 2))

        ax.set_title(
            f'Adjacency Matrix Visualization\n'
            f'{len(node_list)} nodes | '
            f'{int(num_edges)} edges | '
            f'Sparsity: {sparsity:.4f}',
            fontsize=12,
            fontweight='bold'
        )
        ax.set_xlabel('Node ID', fontsize=10)
        ax.set_ylabel('Node ID', fontsize=10)

        plt.tight_layout()

        # Save figure
        if self.output_dir:
            output_path = self.output_dir / "topology_adjacency_matrix.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            _log.info(f"Saved adjacency matrix to {output_path}")
            plt.close()
            return output_path
        else:
            plt.show()
            return None

    def visualize_uv_grid_samples(
        self,
        face_uv_grids: Dict[int, np.ndarray],
        max_faces: int = 6,
        figsize: Tuple[int, int] = (18, 12)
    ) -> Optional[Path]:
        """Visualize UV-grid samples for up to 6 faces.

        Creates 3D scatter plots of UV-grid samples with color-coded normals
        and trimming mask overlay.

        Args:
            face_uv_grids: Dict mapping face index -> UV-grid [10, 10, 7]
            max_faces: Maximum number of faces to visualize (default: 6)
            figsize: Figure size (width, height)

        Returns:
            Path to saved visualization or None
        """
        if not face_uv_grids:
            _log.warning("No UV-grids provided for visualization")
            return None

        try:
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            _log.error("mpl_toolkits.mplot3d required for UV-grid visualization")
            return None

        # Select faces with largest UV-grids (most data)
        face_indices = sorted(face_uv_grids.keys())[:max_faces]
        num_faces = len(face_indices)

        if num_faces == 0:
            _log.warning("No valid UV-grids to visualize")
            return None

        # Create subplot grid (2 rows x 3 columns)
        fig = plt.figure(figsize=figsize)
        rows, cols = 2, 3

        for i, face_idx in enumerate(face_indices):
            if i >= max_faces:
                break

            uv_grid = face_uv_grids[face_idx]
            if uv_grid.shape != (10, 10, 7):
                _log.warning(f"Face {face_idx} has invalid UV-grid shape: {uv_grid.shape}")
                continue

            # Extract channels
            points = uv_grid[:, :, 0:3]  # [10, 10, 3]
            normals = uv_grid[:, :, 3:6]  # [10, 10, 3]
            trimming_mask = uv_grid[:, :, 6]  # [10, 10]

            # Flatten for scatter plot
            points_flat = points.reshape(-1, 3)
            normals_flat = normals.reshape(-1, 3)
            trimming_flat = trimming_mask.reshape(-1)

            # Create color from normals (map [-1, 1] -> [0, 1])
            colors = (normals_flat + 1.0) / 2.0
            colors = np.clip(colors, 0, 1)

            # Apply alpha channel based on trimming mask
            alphas = trimming_flat * 0.8 + 0.2  # [0.2, 1.0]

            # Create 3D subplot
            ax = fig.add_subplot(rows, cols, i + 1, projection='3d')

            # Scatter plot with color from normals
            for j in range(len(points_flat)):
                ax.scatter(
                    points_flat[j, 0],
                    points_flat[j, 1],
                    points_flat[j, 2],
                    c=[colors[j]],
                    s=30,
                    alpha=alphas[j],
                    edgecolors='k',
                    linewidths=0.5
                )

            ax.set_title(f'Face {face_idx}\nUV-Grid Samples', fontsize=10, fontweight='bold')
            ax.set_xlabel('X', fontsize=8)
            ax.set_ylabel('Y', fontsize=8)
            ax.set_zlabel('Z', fontsize=8)
            ax.tick_params(labelsize=7)

        # Hide unused subplots
        for i in range(num_faces, rows * cols):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.axis('off')

        fig.suptitle(
            f'UV-Grid Samples Visualization (10×10×7)\n'
            f'Colors represent normal directions (RGB), Alpha shows trimming mask',
            fontsize=13,
            fontweight='bold'
        )

        plt.tight_layout()

        # Save figure
        if self.output_dir:
            output_path = self.output_dir / "uv_grid_samples.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            _log.info(f"Saved UV-grid samples visualization to {output_path}")
            plt.close()
            return output_path
        else:
            plt.show()
            return None

    def visualize_dihedral_distribution(
        self,
        dihedral_data: Dict[str, Any],
        figsize: Tuple[int, int] = (12, 6)
    ) -> Optional[Path]:
        """Plot dihedral angle distribution histogram.

        Creates histogram of angles with statistics and reference markers
        for mechanical part validation (expected peak around 90°).

        Args:
            dihedral_data: Dictionary with 'angles', 'mean', 'std', 'median' keys
            figsize: Figure size (width, height)

        Returns:
            Path to saved visualization or None
        """
        if not dihedral_data or 'angles' not in dihedral_data:
            _log.warning("No dihedral angle data provided")
            return None

        angles_rad = np.array(dihedral_data['angles'])
        if len(angles_rad) == 0:
            _log.warning("No dihedral angles to visualize")
            return None

        # Convert to degrees for visualization
        angles_deg = np.degrees(angles_rad)
        mean_deg = np.degrees(dihedral_data.get('mean', 0))
        std_deg = np.degrees(dihedral_data.get('std', 0))
        median_deg = np.degrees(dihedral_data.get('median', 0))

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Histogram with 36 bins (5° per bin)
        counts, bins, patches = ax.hist(
            angles_deg,
            bins=36,
            range=(0, 180),
            color='skyblue',
            edgecolor='black',
            alpha=0.7,
            label='Dihedral angles'
        )

        # Add mean and median lines
        ax.axvline(mean_deg, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_deg:.1f}°')
        ax.axvline(median_deg, color='green', linestyle='--', linewidth=2, label=f'Median: {median_deg:.1f}°')

        # Add 90° reference line (expected for mechanical parts)
        ax.axvline(90, color='orange', linestyle=':', linewidth=2, alpha=0.5, label='90° (Reference)')

        # Statistics text box
        stats_text = (
            f'Statistics:\n'
            f'Count: {len(angles_deg)}\n'
            f'Mean: {mean_deg:.1f}°\n'
            f'Std: {std_deg:.1f}°\n'
            f'Median: {median_deg:.1f}°'
        )
        ax.text(
            0.98, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        ax.set_xlabel('Dihedral Angle (degrees)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(
            'Dihedral Angle Distribution\n'
            'Angles between adjacent face normals',
            fontsize=12,
            fontweight='bold'
        )
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        if self.output_dir:
            output_path = self.output_dir / "dihedral_distribution.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            _log.info(f"Saved dihedral distribution to {output_path}")
            plt.close()
            return output_path
        else:
            plt.show()
            return None

    def visualize_curvature_distribution(
        self,
        curvature_data: Dict[str, Any],
        figsize: Tuple[int, int] = (14, 6)
    ) -> Optional[Path]:
        """Plot curvature distributions (Gaussian and mean).

        Creates histograms for Gaussian and mean curvature with annotations
        for expected peaks (planar surfaces at K≈0, cylindrical at K=0, H≠0).

        Args:
            curvature_data: Dictionary with 'gaussian' and 'mean' subdicts
            figsize: Figure size (width, height)

        Returns:
            Path to saved visualization or None
        """
        if not curvature_data:
            _log.warning("No curvature data provided")
            return None

        gaussian_data = curvature_data.get('gaussian', {})
        mean_data = curvature_data.get('mean', {})

        if not gaussian_data and not mean_data:
            _log.warning("No curvature data to visualize")
            return None

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot Gaussian curvature
        if gaussian_data and 'values' in gaussian_data:
            gaussian_values = np.array(gaussian_data['values'])
            if len(gaussian_values) > 0:
                # Clip extreme values for visualization
                gaussian_clipped = np.clip(gaussian_values, -10, 10)

                ax1.hist(
                    gaussian_clipped,
                    bins=50,
                    color='steelblue',
                    edgecolor='black',
                    alpha=0.7
                )
                ax1.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='K=0 (Planar/Cylindrical)')

                # Statistics
                mean_k = gaussian_data.get('mean', 0)
                std_k = gaussian_data.get('std', 0)

                stats_text = (
                    f'Count: {len(gaussian_values)}\n'
                    f'Mean: {mean_k:.4f}\n'
                    f'Std: {std_k:.4f}'
                )
                ax1.text(
                    0.98, 0.97, stats_text,
                    transform=ax1.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                )

                ax1.set_xlabel('Gaussian Curvature (K = k1 × k2)', fontsize=10)
                ax1.set_ylabel('Frequency', fontsize=10)
                ax1.set_title('Gaussian Curvature Distribution', fontsize=11, fontweight='bold')
                ax1.set_yscale('log')
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3)

        # Plot Mean curvature
        if mean_data and 'values' in mean_data:
            mean_values = np.array(mean_data['values'])
            if len(mean_values) > 0:
                # Clip extreme values for visualization
                mean_clipped = np.clip(mean_values, -10, 10)

                ax2.hist(
                    mean_clipped,
                    bins=50,
                    color='coral',
                    edgecolor='black',
                    alpha=0.7
                )
                ax2.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='H=0 (Planar)')

                # Statistics
                mean_h = mean_data.get('mean', 0)
                std_h = mean_data.get('std', 0)

                stats_text = (
                    f'Count: {len(mean_values)}\n'
                    f'Mean: {mean_h:.4f}\n'
                    f'Std: {std_h:.4f}'
                )
                ax2.text(
                    0.98, 0.97, stats_text,
                    transform=ax2.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                )

                ax2.set_xlabel('Mean Curvature (H = (k1 + k2) / 2)', fontsize=10)
                ax2.set_ylabel('Frequency', fontsize=10)
                ax2.set_title('Mean Curvature Distribution', fontsize=11, fontweight='bold')
                ax2.set_yscale('log')
                ax2.legend(fontsize=8)
                ax2.grid(True, alpha=0.3)

        fig.suptitle(
            'Surface Curvature Distributions\n'
            'Expected peaks near 0 for planes, cylinders, and cones',
            fontsize=13,
            fontweight='bold'
        )

        plt.tight_layout()

        # Save figure
        if self.output_dir:
            output_path = self.output_dir / "curvature_distribution.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            _log.info(f"Saved curvature distribution to {output_path}")
            plt.close()
            return output_path
        else:
            plt.show()
            return None

    def visualize_surface_type_distribution(
        self,
        surface_type_data: Dict[str, int],
        figsize: Tuple[int, int] = (10, 6)
    ) -> Optional[Path]:
        """Plot surface type distribution bar chart.

        Creates horizontal bar chart showing frequency of each surface type
        (PLANE, CYLINDRICAL_SURFACE, etc.) with percentage labels.

        Args:
            surface_type_data: Dict mapping surface type name to count
            figsize: Figure size (width, height)

        Returns:
            Path to saved visualization or None
        """
        if not surface_type_data:
            _log.warning("No surface type data provided")
            return None

        # Sort by count (descending)
        sorted_items = sorted(surface_type_data.items(), key=lambda x: x[1], reverse=True)

        surface_types = [item[0] for item in sorted_items]
        counts = [item[1] for item in sorted_items]
        total = sum(counts)

        if total == 0:
            _log.warning("No surfaces to visualize")
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Color palette
        colors = plt.cm.Set3(np.linspace(0, 1, len(surface_types)))

        # Horizontal bar chart
        bars = ax.barh(surface_types, counts, color=colors, edgecolor='black', alpha=0.8)

        # Add percentage labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            percentage = count / total * 100
            ax.text(
                bar.get_width() + total * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{count} ({percentage:.1f}%)',
                va='center',
                fontsize=9
            )

        ax.set_xlabel('Count', fontsize=11)
        ax.set_ylabel('Surface Type', fontsize=11)
        ax.set_title(
            f'Surface Type Distribution\n'
            f'Total surfaces: {total} | Types: {len(surface_types)}',
            fontsize=12,
            fontweight='bold'
        )
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        # Save figure
        if self.output_dir:
            output_path = self.output_dir / "surface_type_distribution.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            _log.info(f"Saved surface type distribution to {output_path}")
            plt.close()
            return output_path
        else:
            plt.show()
            return None

    def visualize_brep_hierarchy(
        self,
        hierarchy_data: Dict[str, Any],
        figsize: Tuple[int, int] = (12, 8)
    ) -> Optional[Path]:
        """Visualize BRep hierarchy as Sankey-style flow diagram.

        Shows the topological hierarchy: Shells → Faces → Edges → Vertices
        with flow widths proportional to counts and Euler characteristic validation.

        Args:
            hierarchy_data: Dictionary with 'num_shells', 'num_faces', 'num_edges',
                          'num_vertices', 'euler_characteristic' keys
            figsize: Figure size (width, height)

        Returns:
            Path to saved visualization or None
        """
        if not hierarchy_data:
            _log.warning("No hierarchy data provided")
            return None

        # Extract counts
        num_shells = hierarchy_data.get('num_shells', 0)
        num_faces = hierarchy_data.get('num_faces', 0)
        num_edges = hierarchy_data.get('num_edges', 0)
        num_vertices = hierarchy_data.get('num_vertices', 0)
        euler_char = hierarchy_data.get('euler_characteristic', 0)
        topology_type = hierarchy_data.get('topology_type', 'unknown')

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')

        # Define levels and positions
        levels = ['Shells', 'Faces', 'Edges', 'Vertices']
        counts = [num_shells, num_faces, num_edges, num_vertices]
        positions = [0.15, 0.4, 0.65, 0.9]

        # Find max count for normalization
        max_count = max(counts) if max(counts) > 0 else 1

        # Draw boxes for each level
        colors = ['#FFB6C1', '#87CEFA', '#90EE90', '#FFD700']

        for i, (level, count, x_pos, color) in enumerate(zip(levels, counts, positions, colors)):
            # Box height proportional to count
            height = 0.3 * (count / max_count) if count > 0 else 0.05
            y_pos = 0.5 - height / 2

            # Draw rectangle
            rect = mpatches.Rectangle(
                (x_pos - 0.08, y_pos),
                0.16,
                height,
                linewidth=2,
                edgecolor='black',
                facecolor=color,
                alpha=0.7
            )
            ax.add_patch(rect)

            # Add label
            ax.text(
                x_pos, 0.8, level,
                ha='center', va='bottom',
                fontsize=12, fontweight='bold'
            )

            # Add count
            ax.text(
                x_pos, y_pos + height / 2, f'{count}',
                ha='center', va='center',
                fontsize=14, fontweight='bold'
            )

            # Draw connecting arrows
            if i < len(levels) - 1:
                arrow = mpatches.FancyArrowPatch(
                    (x_pos + 0.08, y_pos + height / 2),
                    (positions[i + 1] - 0.08, y_pos + height / 2),
                    arrowstyle='->',
                    mutation_scale=30,
                    linewidth=2,
                    color='gray',
                    alpha=0.6
                )
                ax.add_patch(arrow)

        # Add Euler characteristic annotation
        euler_text = (
            f'Euler Characteristic: V - E + F = {euler_char}\n'
            f'({num_vertices} - {num_edges} + {num_faces} = {euler_char})\n'
            f'Expected: 2 for solid, 0 for surface'
        )

        ax.text(
            0.5, 0.15, euler_text,
            ha='center', va='top',
            fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        # Add topology type
        ax.text(
            0.5, 0.95, f'Topology Type: {topology_type.upper()}',
            ha='center', va='top',
            fontsize=13, fontweight='bold'
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        fig.suptitle(
            'BRep Topological Hierarchy\n'
            'Shells → Faces → Edges → Vertices',
            fontsize=14,
            fontweight='bold',
            y=0.98
        )

        plt.tight_layout()

        # Save figure
        if self.output_dir:
            output_path = self.output_dir / "brep_hierarchy.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            _log.info(f"Saved BRep hierarchy visualization to {output_path}")
            plt.close()
            return output_path
        else:
            plt.show()
            return None

    def generate_all_visualizations(
        self,
        max_nodes: int = 100
    ) -> Dict[str, Path]:
        """Generate all available visualizations.

        Args:
            max_nodes: Maximum nodes for graph layouts

        Returns:
            Dictionary mapping visualization name to output path
        """
        visualizations = {}

        _log.info("Generating all topology visualizations...")

        # Adjacency graph
        try:
            path = self.visualize_adjacency_graph(max_nodes=max_nodes)
            if path:
                visualizations['adjacency_graph'] = path
        except Exception as e:
            _log.error(f"Failed to generate adjacency graph: {e}")

        # Degree distribution
        try:
            path = self.visualize_degree_distribution()
            if path:
                visualizations['degree_distribution'] = path
        except Exception as e:
            _log.error(f"Failed to generate degree distribution: {e}")

        # Connected components
        try:
            path = self.visualize_connected_components(max_nodes_per_component=50)
            if path:
                visualizations['connected_components'] = path
        except Exception as e:
            _log.error(f"Failed to generate connected components: {e}")

        # Adjacency matrix
        try:
            path = self.visualize_adjacency_matrix(max_nodes=max_nodes)
            if path:
                visualizations['adjacency_matrix'] = path
        except Exception as e:
            _log.error(f"Failed to generate adjacency matrix: {e}")

        _log.info(f"Generated {len(visualizations)} visualizations")

        return visualizations


    def visualize_cad_entity_relationships(
        self,
        document: Any,
        figsize: Tuple[int, int] = (14, 10)
    ) -> Optional[Path]:
        """Visualize CAD-specific entity relationships and types.

        Shows the actual CAD entity structure with type groupings and
        reference patterns specific to STEP/BRep topology.

        Args:
            document: CADlingDocument with items and topology
            figsize: Figure size (width, height)

        Returns:
            Path to saved visualization or None
        """
        if not self.has_networkx:
            _log.error("networkx required for entity visualization")
            return None

        # Analyze entity types and references
        entity_types = {}
        entity_refs = {}

        for item in document.items:
            entity_type = item.entity_type if hasattr(item, 'entity_type') else type(item).__name__
            entity_id = item.entity_id if hasattr(item, 'entity_id') else id(item)

            entity_types[entity_id] = entity_type

            # Get references from this entity
            if hasattr(item, 'reference_params'):
                entity_refs[entity_id] = item.reference_params

        # Group entities by type
        type_groups = {}
        for eid, etype in entity_types.items():
            if etype not in type_groups:
                type_groups[etype] = []
            type_groups[etype].append(eid)

        # Create figure with multiple subplots
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Entity type distribution pie chart
        ax1 = fig.add_subplot(gs[0, 0])
        type_counts = {t: len(ids) for t, ids in type_groups.items()}
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        if sorted_types:
            labels = [t for t, _ in sorted_types]
            sizes = [c for _, c in sorted_types]
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Entity Type Distribution (Top 10)', fontsize=11, fontweight='bold')

        # 2. Entity reference matrix by type
        ax2 = fig.add_subplot(gs[0, 1])

        # Build type-to-type reference matrix
        top_types = [t for t, _ in sorted_types[:8]] if len(sorted_types) >= 8 else [t for t, _ in sorted_types]
        type_matrix = np.zeros((len(top_types), len(top_types)))

        for eid, refs in entity_refs.items():
            if eid in entity_types:
                from_type = entity_types[eid]
                if from_type in top_types:
                    from_idx = top_types.index(from_type)
                    for ref_id in refs:
                        if ref_id in entity_types:
                            to_type = entity_types[ref_id]
                            if to_type in top_types:
                                to_idx = top_types.index(to_type)
                                type_matrix[from_idx, to_idx] += 1

        im = ax2.imshow(type_matrix, cmap='YlOrRd', aspect='auto')
        ax2.set_xticks(range(len(top_types)))
        ax2.set_yticks(range(len(top_types)))
        ax2.set_xticklabels([t[:15] for t in top_types], rotation=45, ha='right', fontsize=8)
        ax2.set_yticklabels([t[:15] for t in top_types], fontsize=8)
        ax2.set_title('Entity Type Reference Matrix', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Referenced Entity Type', fontsize=9)
        ax2.set_ylabel('Referencing Entity Type', fontsize=9)
        plt.colorbar(im, ax=ax2, label='# References')

        # 3. Reference pattern histogram
        ax3 = fig.add_subplot(gs[1, 0])

        ref_counts = [len(refs) for refs in entity_refs.values()]
        if ref_counts:
            ax3.hist(ref_counts, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
            ax3.set_xlabel('Number of References per Entity', fontsize=9)
            ax3.set_ylabel('Frequency', fontsize=9)
            ax3.set_title(
                f'Entity Reference Distribution\n'
                f'Mean: {np.mean(ref_counts):.1f} | Max: {max(ref_counts)}',
                fontsize=11,
                fontweight='bold'
            )
            ax3.grid(True, alpha=0.3)

        # 4. BRep topology summary (if available)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')

        # Extract BRep topology info
        topology_info = []
        if hasattr(document, 'metadata') and document.metadata:
            topo_type = document.metadata.get('topology_type', {})
            if topo_type:
                topology_info.append("BRep Topology Summary:\n")
                topology_info.append(f"Type: {topo_type.get('representation_type', 'unknown')}\n")

                topo_counts = topo_type.get('topology_counts', {})
                if topo_counts:
                    topology_info.append("\nTopology Elements:")
                    for elem, count in topo_counts.items():
                        topology_info.append(f"  • {elem}: {count}")

                if 'euler_characteristic' in topo_type:
                    topology_info.append(f"\nEuler Characteristic: {topo_type['euler_characteristic']}")
                if 'estimated_genus' in topo_type:
                    topology_info.append(f"Estimated Genus: {topo_type['estimated_genus']}")

        if topology_info:
            ax4.text(0.1, 0.5, '\n'.join(topology_info),
                    fontsize=10, verticalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        else:
            ax4.text(0.5, 0.5, 'No BRep topology metadata available',
                    fontsize=10, ha='center', va='center', style='italic')

        ax4.set_title('BRep Topology Information', fontsize=11, fontweight='bold')

        fig.suptitle(
            f'CAD Entity Relationship Analysis\n'
            f'Total Entities: {len(entity_types)} | Types: {len(type_groups)} | References: {len(entity_refs)}',
            fontsize=13,
            fontweight='bold'
        )

        plt.tight_layout()

        # Save figure
        if self.output_dir:
            output_path = self.output_dir / "cad_entity_relationships.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            _log.info(f"Saved CAD entity relationships to {output_path}")
            plt.close()
            return output_path
        else:
            plt.show()
            return None


def visualize_topology_graph(
    topology_graph: Any,
    output_dir: Path,
    max_nodes: int = 100
) -> Dict[str, Path]:
    """Convenience function to generate all topology visualizations.

    Args:
        topology_graph: TopologyGraph object
        output_dir: Directory for saving visualizations
        max_nodes: Maximum nodes for layouts

    Returns:
        Dictionary mapping visualization name to output path
    """
    visualizer = TopologyGraphVisualizer(topology_graph, output_dir)
    return visualizer.generate_all_visualizations(max_nodes=max_nodes)
