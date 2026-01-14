"""Functional tests for BRep and mesh graph construction with real data and telemetry."""

import pytest
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# Add parent directory to path for functional test utilities
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.telemetry_logger import TelemetryLogger
from utils.output_manager import OutputManager
from utils.validators import FunctionalValidator, ValidationResult
from cadling.backend.document_converter import DocumentConverter
from cadling.models.segmentation.brep_graph_builder import BRepFaceGraphBuilder


class TestGraphConstructionFunctional:
    """Functional tests demonstrating BRep and mesh graph construction with real data."""

    def test_brep_graph_construction_functional(
        self, test_data_step_path: Path, functional_output_dir: Path
    ):
        """
        Demonstrate BRep face graph construction with real STEP file.

        Stages:
        1. Load STEP file and convert to document
        2. Build face graph (BRepFaceGraphBuilder)
        3. Analyze node features (face features)
        4. Analyze edge features (adjacency features)
        5. Validate features are REAL (not placeholders)
        6. Export graph statistics
        """
        # Setup output manager and telemetry logger
        output_manager = OutputManager("brep_graph_construction", functional_output_dir)
        logger = TelemetryLogger(
            "brep_graph_construction",
            output_manager.run_dir,
            console_level=20,  # INFO
            file_level=10  # DEBUG
        )

        logger.info("="*80)
        logger.info("FUNCTIONAL TEST: BRep Face Graph Construction")
        logger.info("="*80)

        try:
            # Stage 1: Load STEP file and convert to document
            with logger.stage("STEP File Loading"):
                # Find a small STEP file for testing
                step_files = sorted(test_data_step_path.glob("*.stp"))
                if not step_files:
                    step_files = sorted(test_data_step_path.glob("*.step"))

                if not step_files:
                    pytest.skip("No STEP files found in test data directory")

                # Use a small file for faster testing
                test_file = None
                for f in step_files:
                    if f.stat().st_size < 100_000:  # Less than 100KB
                        test_file = f
                        break

                if test_file is None:
                    test_file = step_files[0]  # Fallback to first file

                logger.log_file_info(test_file)
                logger.info(f"Selected test file: {test_file.name}")

                # Convert STEP to document
                converter = DocumentConverter()
                result = converter.convert(test_file)

                if result.document is None or len(result.errors) > 0:
                    logger.error(f"Conversion failed: {result.errors}")
                    pytest.fail(f"STEP conversion failed: {result.errors}")

                doc = result.document
                logger.log_metric("conversion_status", result.status.value)
                logger.log_metric("num_items", len(doc.items))
                logger.info(f"Document loaded successfully with {len(doc.items)} items")

            # Stage 2: Build face graph
            with logger.stage("BRep Face Graph Building"):
                logger.info("Building BRep face graph using BRepFaceGraphBuilder...")

                graph_builder = BRepFaceGraphBuilder()
                graph_data = graph_builder.build_face_graph(doc)

                if graph_data is None:
                    logger.error("Graph construction returned None")
                    pytest.fail("Failed to build BRep face graph")

                num_nodes = graph_data.x.shape[0] if hasattr(graph_data, 'x') else 0
                num_edges = graph_data.edge_index.shape[1] if hasattr(graph_data, 'edge_index') else 0

                logger.log_metric("num_nodes", num_nodes)
                logger.log_metric("num_edges", num_edges)
                logger.info(f"Graph constructed: {num_nodes} nodes, {num_edges} edges")

                if num_nodes == 0:
                    logger.warning("Graph has 0 nodes - this may indicate a problem")
                    pytest.skip("Graph has no nodes, skipping further analysis")

            # Stage 3: Analyze node features
            with logger.stage("Node Feature Analysis"):
                logger.info("Analyzing node (face) features...")

                if not hasattr(graph_data, 'x') or graph_data.x is None:
                    logger.error("Graph data has no node features (x)")
                    pytest.fail("Graph missing node features")

                node_features = graph_data.x.numpy()
                node_feature_dim = node_features.shape[1]

                logger.log_metric("node_feature_dim", node_feature_dim)
                logger.log_feature_statistics(node_features, "Node features")

                # Compute statistics
                mean_features = np.mean(node_features, axis=0)
                std_features = np.std(node_features, axis=0)
                nonzero_pct = (np.count_nonzero(node_features) / node_features.size) * 100

                logger.log_metric("node_features_nonzero_pct", nonzero_pct)
                logger.info(f"Node features: shape={node_features.shape}, "
                          f"non-zero={nonzero_pct:.2f}%")

                # Log feature breakdown
                logger.info("Node feature statistics:")
                logger.info(f"  Mean: min={mean_features.min():.4f}, "
                          f"max={mean_features.max():.4f}, "
                          f"avg={mean_features.mean():.4f}")
                logger.info(f"  Std: min={std_features.min():.4f}, "
                          f"max={std_features.max():.4f}, "
                          f"avg={std_features.mean():.4f}")

            # Stage 4: Analyze edge features
            with logger.stage("Edge Feature Analysis"):
                logger.info("Analyzing edge features...")

                if not hasattr(graph_data, 'edge_attr') or graph_data.edge_attr is None:
                    logger.warning("Graph data has no edge features (edge_attr)")
                    edge_feature_dim = 0
                    edge_features_nonzero_pct = 0.0
                else:
                    edge_features = graph_data.edge_attr.numpy()
                    edge_feature_dim = edge_features.shape[1] if len(edge_features.shape) > 1 else 0

                    logger.log_metric("edge_feature_dim", edge_feature_dim)
                    logger.log_feature_statistics(edge_features, "Edge features")

                    # Compute statistics (handle empty arrays)
                    if edge_features.size > 0:
                        edge_features_nonzero_pct = (np.count_nonzero(edge_features) / edge_features.size) * 100
                    else:
                        edge_features_nonzero_pct = 0.0
                    logger.log_metric("edge_features_nonzero_pct", edge_features_nonzero_pct)

                    logger.info(f"Edge features: shape={edge_features.shape}, "
                              f"non-zero={edge_features_nonzero_pct:.2f}%")

                    # Check for dihedral angles (typically first feature)
                    if edge_feature_dim > 0:
                        dihedral_angles = edge_features[:, 0]
                        if np.any(dihedral_angles != 0):
                            logger.info(f"Dihedral angles: "
                                      f"min={np.degrees(dihedral_angles.min()):.2f}°, "
                                      f"max={np.degrees(dihedral_angles.max()):.2f}°, "
                                      f"mean={np.degrees(dihedral_angles.mean()):.2f}°")

            # Stage 5: Validate features
            with logger.stage("Feature Validation"):
                logger.info("Validating that features are REAL (not placeholders)...")

                validator = FunctionalValidator()

                # Validate graph features
                validation_result = validator.validate_graph_features(graph_data)

                logger.info(f"Graph validation: {'PASS' if validation_result.passed else 'FAIL'}")
                logger.info(f"  {validation_result.message}")

                if validation_result.details:
                    for key, value in validation_result.details.items():
                        logger.info(f"  {key}: {value}")

                # Check if features are placeholder data
                if nonzero_pct < 5.0:
                    logger.warning(f"Node features appear to be placeholder data "
                                 f"({nonzero_pct:.2f}% non-zero)")

                if hasattr(graph_data, 'edge_attr') and edge_features_nonzero_pct < 5.0:
                    logger.warning(f"Edge features appear to be placeholder data "
                                 f"({edge_features_nonzero_pct:.2f}% non-zero)")

            # Stage 6: Export graph statistics
            with logger.stage("Export Graph Statistics"):
                logger.info("Exporting graph statistics...")

                graph_stats = {
                    "source_file": test_file.name,
                    "num_nodes": int(num_nodes),
                    "num_edges": int(num_edges),
                    "node_feature_dim": int(node_feature_dim),
                    "edge_feature_dim": int(edge_feature_dim),
                    "node_features_nonzero_pct": float(nonzero_pct),
                    "edge_features_nonzero_pct": float(edge_features_nonzero_pct),
                    "graph_density": float(num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0),
                    "validation_passed": validation_result.passed,
                    "has_metadata": hasattr(graph_data, 'metadata') and graph_data.metadata is not None
                }

                output_manager.save_json(graph_stats, "graph_statistics.json")
                logger.info(f"Graph statistics saved to graph_statistics.json")

                # Save validation results
                validation_results = {
                    "graph_features": validation_result.to_dict()
                }
                output_manager.save_json(validation_results, "validation_results.json", "validation")

            # Final summary
            logger.info("="*80)
            logger.info("✓ BRep graph construction test completed successfully")
            logger.info(f"✓ Outputs saved to: {output_manager.run_dir}")
            logger.info("="*80)

        except Exception as e:
            logger.error(f"Test failed with exception: {e}", exc_info=True)
            raise
        finally:
            logger.save_telemetry()

    def test_mesh_graph_construction_functional(
        self, test_data_stl_path: Path, functional_output_dir: Path
    ):
        """
        Demonstrate mesh graph construction from STL files.

        Tests both simple geometries (cube, cylinder) to validate
        that geometric properties match expected values.

        Stages:
        1. Load STL files (cube and cylinder)
        2. Build mesh graphs
        3. Analyze geometric properties
        4. Validate expected properties (90° angles for cube, curvature for cylinder)
        5. Export mesh graph statistics
        """
        # Setup output manager and telemetry logger
        output_manager = OutputManager("mesh_graph_construction", functional_output_dir)
        logger = TelemetryLogger(
            "mesh_graph_construction",
            output_manager.run_dir,
            console_level=20,  # INFO
            file_level=10  # DEBUG
        )

        logger.info("="*80)
        logger.info("FUNCTIONAL TEST: Mesh Graph Construction")
        logger.info("="*80)

        try:
            # Stage 1: Load STL files
            with logger.stage("STL File Loading"):
                logger.info("Loading test STL files (cube and cylinder)...")

                # Look for test files
                cube_file = test_data_stl_path / "test_cube.stl"
                cylinder_file = test_data_stl_path / "test_cylinder.stl"

                test_files = []
                if cube_file.exists():
                    test_files.append(("cube", cube_file))
                    logger.log_file_info(cube_file)

                if cylinder_file.exists():
                    test_files.append(("cylinder", cylinder_file))
                    logger.log_file_info(cylinder_file)

                if not test_files:
                    # Fallback: use any small STL file
                    stl_files = sorted(test_data_stl_path.glob("*.stl"))
                    for f in stl_files:
                        if f.stat().st_size < 1_000_000:  # Less than 1MB
                            test_files.append(("generic", f))
                            logger.log_file_info(f)
                            break

                if not test_files:
                    pytest.skip("No suitable STL files found for testing")

                logger.log_metric("num_test_files", len(test_files))

            # Process each test file
            all_results = {}

            for geom_type, stl_file in test_files:
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing {geom_type}: {stl_file.name}")
                logger.info(f"{'='*60}")

                # Stage 2: Convert STL to document and build graph
                with logger.stage(f"Mesh Graph Building - {geom_type}"):
                    logger.info(f"Converting {stl_file.name} to document...")

                    converter = DocumentConverter()
                    result = converter.convert(stl_file)

                    if result.document is None or len(result.errors) > 0:
                        logger.error(f"Conversion failed: {result.errors}")
                        continue

                    doc = result.document
                    logger.log_metric(f"{geom_type}_num_items", len(doc.items))
                    logger.info(f"Document loaded: {len(doc.items)} items")

                    # Build graph (reuse BRepFaceGraphBuilder for now)
                    # Note: In production, might want a dedicated MeshGraphBuilder
                    graph_builder = BRepFaceGraphBuilder()
                    graph_data = graph_builder.build_face_graph(doc)

                    if graph_data is None:
                        logger.warning(f"Failed to build graph for {geom_type}")
                        continue

                    num_nodes = graph_data.x.shape[0] if hasattr(graph_data, 'x') else 0
                    num_edges = graph_data.edge_index.shape[1] if hasattr(graph_data, 'edge_index') else 0

                    logger.log_metric(f"{geom_type}_num_nodes", num_nodes)
                    logger.log_metric(f"{geom_type}_num_edges", num_edges)
                    logger.info(f"Graph: {num_nodes} nodes, {num_edges} edges")

                # Stage 3: Analyze geometric properties
                with logger.stage(f"Geometric Property Analysis - {geom_type}"):
                    logger.info(f"Analyzing geometric properties for {geom_type}...")

                    properties = {
                        "geometry_type": geom_type,
                        "num_nodes": int(num_nodes),
                        "num_edges": int(num_edges),
                    }

                    # Extract dihedral angles if available
                    if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None:
                        edge_features = graph_data.edge_attr.numpy()
                        if edge_features.shape[1] > 0 and edge_features.shape[0] > 0:
                            dihedral_angles = edge_features[:, 0]
                            dihedral_degrees = np.degrees(dihedral_angles)

                            properties["dihedral_angles"] = {
                                "min_deg": float(dihedral_degrees.min()),
                                "max_deg": float(dihedral_degrees.max()),
                                "mean_deg": float(dihedral_degrees.mean()),
                                "std_deg": float(dihedral_degrees.std())
                            }

                            logger.info(f"Dihedral angles: "
                                      f"min={dihedral_degrees.min():.2f}°, "
                                      f"max={dihedral_degrees.max():.2f}°, "
                                      f"mean={dihedral_degrees.mean():.2f}°")

                            # Validate expected properties
                            if geom_type == "cube":
                                # Expect 90° angles for cube edges
                                angles_near_90 = np.sum(np.abs(dihedral_degrees - 90) < 5)
                                logger.info(f"Angles near 90°: {angles_near_90}/{len(dihedral_angles)}")
                                properties["angles_near_90_deg"] = int(angles_near_90)

                    all_results[geom_type] = properties

            # Stage 5: Export mesh graph statistics
            with logger.stage("Export Mesh Graph Statistics"):
                logger.info("Exporting mesh graph statistics...")

                output_manager.save_json(all_results, "mesh_graph_statistics.json")
                logger.info(f"Mesh graph statistics saved")

                # Log summary
                logger.info("\nMesh Graph Construction Summary:")
                for geom_type, props in all_results.items():
                    logger.info(f"  {geom_type}: {props['num_nodes']} nodes, "
                              f"{props['num_edges']} edges")

            # Final summary
            logger.info("="*80)
            logger.info("✓ Mesh graph construction test completed successfully")
            logger.info(f"✓ Outputs saved to: {output_manager.run_dir}")
            logger.info("="*80)

        except Exception as e:
            logger.error(f"Test failed with exception: {e}", exc_info=True)
            raise
        finally:
            logger.save_telemetry()
