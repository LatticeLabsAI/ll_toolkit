"""Functional tests for STEP conversion pipeline with real data and telemetry.

These are comprehensive functional runs that demonstrate the complete STEP pipeline
with real CAD files, detailed logging at each stage, and validation of outputs.

Now includes industry-standard CAD AI features:
- UV-grid extraction (10×10×7 for faces, 10×6 for edges)
- Enhanced node features (48-dim) and edge features (16-dim)
- Geometric distribution analysis (dihedral angles, curvature)
- PyTorch Geometric export with UV-grids
- Enhanced visualizations and validations
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

# Add parent directory to path for functional test utilities
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_file_categories import FileCategory, categorize_test_files
from utils.output_manager import OutputManager
from utils.telemetry_logger import TelemetryLogger
from utils.validators import FunctionalValidator, ValidationResult

from cadling.backend.document_converter import DocumentConverter
from cadling.lib.geometry.distribution_analyzer import (
    BRepHierarchyAnalyzer,
    CurvatureAnalyzer,
    DihedralAngleAnalyzer,
    SurfaceTypeAnalyzer,
)

# Import new CAD AI modules
from cadling.lib.geometry.uv_grid_extractor import (
    EdgeUVGridExtractor,
    FaceUVGridExtractor,
)
from cadling.lib.graph.enhanced_features import (
    extract_enhanced_edge_features,
    extract_enhanced_node_features,
    get_curve_type_from_edge,
    get_surface_type_from_face,
)
from cadling.lib.graph.pyg_exporter import (
    export_to_pyg_with_uvgrids,
    save_pyg_data,
    validate_pyg_data,
)
from cadling.lib.graph.visualization import TopologyGraphVisualizer

# Import OCC for topology traversal
try:
    from OCC.Extend.TopologyUtils import TopologyExplorer
    HAS_OCC = True
except ImportError:
    HAS_OCC = False


class TestSTEPPipelineFunctional:
    """Functional tests demonstrating STEP conversion pipeline with real data."""

    def test_step_pipeline_small_file(self, test_data_step_path: Path, functional_output_dir: Path):
        """Demonstrate complete STEP pipeline with small file.

        This test shows the full pipeline from STEP file to document with:
        - File information and metadata
        - STEP conversion using DocumentConverter
        - Document analysis (entity counts, types)
        - Topology analysis (nodes, edges, density)
        - Validation of document structure
        - Export of artifacts and telemetry

        Args:
            test_data_step_path: Path to STEP test data directory
            functional_output_dir: Base output directory for functional test outputs
        """
        # Setup output manager and telemetry logger
        output_manager = OutputManager("step_pipeline_small", functional_output_dir)
        logger = TelemetryLogger(
            "step_pipeline_small",
            output_manager.run_dir,
            console_level=20,  # INFO
            file_level=10  # DEBUG
        )

        logger.info("="*80)
        logger.info("FUNCTIONAL TEST: STEP Pipeline - Small File")
        logger.info("="*80)

        # Get small test files
        categories = categorize_test_files(test_data_step_path, "*.stp")
        small_files = categories[FileCategory.SMALL]

        if not small_files:
            pytest.skip("No small STEP files found for testing")

        # Use first small file
        test_file = small_files[0]

        try:
            # Stage 1: File Information
            with logger.stage("File Information"):
                logger.log_file_info(test_file)

                # Save file info to artifacts
                file_info = {
                    "name": test_file.name,
                    "path": str(test_file),
                    "size_bytes": test_file.stat().st_size,
                    "size_mb": test_file.stat().st_size / (1024 * 1024)
                }
                output_manager.save_json(file_info, "input_file_info.json", "artifacts")

            # Stage 2: STEP Conversion
            with logger.stage("STEP Conversion"):
                logger.info(f"Converting STEP file: {test_file.name}")

                converter = DocumentConverter()
                result = converter.convert(test_file)

                logger.log_metric("conversion_status", result.status.value)
                logger.log_metric("num_items", len(result.document.items))
                logger.log_metric("num_errors", len(result.errors))

                if result.errors:
                    for i, error in enumerate(result.errors[:5]):  # Log first 5 errors
                        logger.warning(f"  Error {i+1}: {error}")

                # Access document
                doc = result.document
                logger.info(f"Document created: {doc.name}")
                logger.info(f"  Format: {doc.format}")
                logger.info(f"  Origin: {doc.origin}")

            # Stage 3: Document Analysis
            with logger.stage("Document Analysis"):
                logger.info(f"Analyzing document structure...")

                # Count entity types
                entity_types: Dict[str, int] = {}
                for item in doc.items:
                    item_type = type(item).__name__
                    entity_types[item_type] = entity_types.get(item_type, 0) + 1

                logger.info(f"Entity types found:")
                for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
                    logger.info(f"  {entity_type}: {count}")

                logger.log_metric("unique_entity_types", len(entity_types))
                logger.log_metric("total_entities", len(doc.items))

                # Save entity analysis
                entity_analysis = {
                    "total_entities": len(doc.items),
                    "unique_types": len(entity_types),
                    "entity_type_distribution": entity_types
                }
                output_manager.save_json(entity_analysis, "entity_analysis.json", "intermediates")

            # Stage 4: Topology Analysis
            with logger.stage("Topology Analysis"):
                logger.info("Analyzing topology structure...")

                # Topology is already built in the document
                if doc.topology:
                    topology = doc.topology

                    logger.info(f"Topology graph:")
                    logger.info(f"  Nodes: {topology.num_nodes}")
                    logger.info(f"  Edges: {topology.num_edges}")
                    logger.info(f"  Adjacency list size: {len(topology.adjacency_list)}")

                    logger.log_metric("topology_num_nodes", topology.num_nodes)
                    logger.log_metric("topology_num_edges", topology.num_edges)
                    logger.log_metric("topology_adjacency_size", len(topology.adjacency_list))

                    # Calculate graph density
                    if topology.num_nodes > 1:
                        max_edges = topology.num_nodes * (topology.num_nodes - 1) / 2
                        density = topology.num_edges / max_edges if max_edges > 0 else 0
                        logger.log_metric("graph_density", round(density, 4))
                        logger.info(f"  Graph density: {density:.4f}")

                    # Get topology statistics from metadata
                    if hasattr(doc, 'metadata') and doc.metadata:
                        topo_stats = doc.metadata.get("topology_statistics", {})
                        if topo_stats:
                            logger.info(f"Topology statistics:")
                            for key, value in topo_stats.items():
                                logger.info(f"  {key}: {value}")

                        # Get topology type info
                        topo_type = doc.metadata.get("topology_type", {})
                        if topo_type:
                            logger.info(f"Topology type: {topo_type.get('representation_type', 'unknown')}")

                    topology_data = {
                        "num_nodes": topology.num_nodes,
                        "num_edges": topology.num_edges,
                        "adjacency_list_size": len(topology.adjacency_list),
                        "density": round(density, 4) if topology.num_nodes > 1 else 0,
                        "statistics": doc.metadata.get("topology_statistics", {}) if hasattr(doc, 'metadata') else {},
                        "topology_type": doc.metadata.get("topology_type", {}) if hasattr(doc, 'metadata') else {}
                    }
                    output_manager.save_json(topology_data, "topology_statistics.json", "intermediates")

                    # Generate CAD-specific entity relationship visualization
                    logger.info("Generating CAD entity relationship visualization...")
                    try:
                        visualizer = TopologyGraphVisualizer(
                            topology,
                            output_manager.get_visualization_dir()
                        )

                        # Generate CAD-specific visualization
                        cad_viz = visualizer.visualize_cad_entity_relationships(doc)
                        if cad_viz:
                            logger.info(f"  ✓ CAD entity relationships: {cad_viz.name}")

                        # Generate degree distribution (useful for understanding graph structure)
                        degree_viz = visualizer.visualize_degree_distribution()
                        if degree_viz:
                            logger.info(f"  ✓ Degree distribution: {degree_viz.name}")

                        logger.log_metric("cad_visualization_generated", 1 if cad_viz else 0)
                    except Exception as e:
                        logger.warning(f"Could not generate visualizations: {e}")
                else:
                    logger.warning("No topology information in document")

            # Stage 5: UV-Grid Extraction
            face_uv_grids = {}
            edge_uv_grids = {}
            occ_faces = []
            occ_edges = []

            if HAS_OCC and hasattr(doc, '_occ_shape') and doc._occ_shape:
                with logger.stage("UV-Grid Extraction"):
                    logger.info("Extracting UV-grids from faces and edges...")

                    try:
                        # Extract OCC faces and edges from document
                        topo = TopologyExplorer(doc._occ_shape)
                        occ_faces = list(topo.faces())
                        occ_edges = list(topo.edges())

                        logger.info(f"Found {len(occ_faces)} faces and {len(occ_edges)} edges")

                        # Extract face UV-grids
                        face_uv_grids = FaceUVGridExtractor.extract_batch_uv_grids(occ_faces)
                        edge_uv_grids = EdgeUVGridExtractor.extract_batch_uv_grids(occ_edges)

                        logger.log_metric("num_face_uv_grids", len(face_uv_grids))
                        logger.log_metric("num_edge_uv_grids", len(edge_uv_grids))

                        if len(occ_faces) > 0:
                            face_coverage = len(face_uv_grids) / len(occ_faces) * 100
                            logger.log_metric("face_uv_grid_coverage_percent", round(face_coverage, 1))
                            logger.info(f"  Face UV-grid coverage: {face_coverage:.1f}%")

                        if len(occ_edges) > 0:
                            edge_coverage = len(edge_uv_grids) / len(occ_edges) * 100
                            logger.log_metric("edge_uv_grid_coverage_percent", round(edge_coverage, 1))
                            logger.info(f"  Edge UV-grid coverage: {edge_coverage:.1f}%")

                        # Save UV-grids as compressed numpy arrays
                        if face_uv_grids:
                            face_grids_path = output_manager.get_output_dir() / "face_uv_grids.npz"
                            np.savez_compressed(
                                face_grids_path,
                                **{str(k): v for k, v in face_uv_grids.items()}
                            )
                            logger.info(f"  ✓ Saved face UV-grids: {face_grids_path.name}")

                        if edge_uv_grids:
                            edge_grids_path = output_manager.get_output_dir() / "edge_uv_grids.npz"
                            np.savez_compressed(
                                edge_grids_path,
                                **{str(k): v for k, v in edge_uv_grids.items()}
                            )
                            logger.info(f"  ✓ Saved edge UV-grids: {edge_grids_path.name}")

                    except Exception as e:
                        logger.warning(f"UV-grid extraction failed: {e}")
            else:
                logger.info("Skipping UV-grid extraction (OCC not available or no _occ_shape)")

            # Stage 6: Geometric Distribution Analysis
            dihedral_data = None
            curvature_data = None
            surface_type_data = None
            hierarchy_data = None

            if HAS_OCC and occ_faces:
                with logger.stage("Geometric Distribution Analysis"):
                    logger.info("Analyzing geometric distributions...")

                    try:
                        # Analyze dihedral angles
                        dihedral_data = DihedralAngleAnalyzer.compute_dihedral_angles(doc)
                        if dihedral_data and 'mean' in dihedral_data:
                            logger.log_metric("mean_dihedral_angle_deg", round(np.degrees(dihedral_data['mean']), 1))
                            logger.log_metric("median_dihedral_angle_deg", round(np.degrees(dihedral_data['median']), 1))
                            logger.info(f"  Dihedral angles: mean={np.degrees(dihedral_data['mean']):.1f}°, median={np.degrees(dihedral_data['median']):.1f}°")

                        # Analyze curvature distributions
                        curvature_data = CurvatureAnalyzer.compute_curvature_distribution(occ_faces)
                        if curvature_data:
                            logger.info(f"  Curvature analysis completed")
                            if 'gaussian' in curvature_data and 'mean' in curvature_data['gaussian']:
                                logger.log_metric("mean_gaussian_curvature", round(curvature_data['gaussian']['mean'], 6))
                            if 'mean' in curvature_data and 'mean' in curvature_data['mean']:
                                logger.log_metric("mean_mean_curvature", round(curvature_data['mean']['mean'], 6))

                        # Analyze surface types
                        surface_type_data = SurfaceTypeAnalyzer.analyze_surface_types(doc)
                        if surface_type_data:
                            logger.log_metric("num_surface_types", len(surface_type_data))
                            logger.info(f"  Surface types found: {len(surface_type_data)}")
                            for surf_type, count in sorted(surface_type_data.items(), key=lambda x: x[1], reverse=True)[:5]:
                                logger.info(f"    {surf_type}: {count}")

                        # Analyze BRep hierarchy
                        hierarchy_data = BRepHierarchyAnalyzer.extract_hierarchy(doc)
                        if hierarchy_data:
                            logger.log_metric("euler_characteristic", hierarchy_data.get('euler_characteristic', 0))
                            logger.info(f"  BRep hierarchy: {hierarchy_data.get('num_faces', 0)} faces, {hierarchy_data.get('num_edges', 0)} edges")
                            logger.info(f"  Euler characteristic: {hierarchy_data.get('euler_characteristic', 0)}")

                        # Save all distribution data
                        distributions = {
                            "dihedral": dihedral_data,
                            "curvature": curvature_data,
                            "surface_types": surface_type_data,
                            "hierarchy": hierarchy_data
                        }
                        output_manager.save_json(distributions, "geometric_distributions.json", "intermediates")
                        logger.info(f"  ✓ Saved geometric distributions")

                    except Exception as e:
                        logger.warning(f"Geometric distribution analysis failed: {e}")
            else:
                logger.info("Skipping geometric distribution analysis (no OCC faces available)")

            # Stage 7: Enhanced Feature Extraction
            node_features = None
            edge_features = None

            if HAS_OCC and occ_faces:
                with logger.stage("Enhanced Feature Extraction"):
                    logger.info("Extracting enhanced node and edge features...")

                    try:
                        # Extract enhanced node features (48-dim)
                        node_features_list = []
                        for idx, face in enumerate(occ_faces):
                            uv_grid = face_uv_grids.get(idx, None)
                            surface_type = get_surface_type_from_face(face)
                            features = extract_enhanced_node_features(face, surface_type, uv_grid)
                            node_features_list.append(features)

                        if node_features_list:
                            node_features = np.vstack(node_features_list)
                            logger.log_metric("node_feature_dim", node_features.shape[1])
                            logger.log_metric("num_nodes", node_features.shape[0])
                            logger.info(f"  Node features: {node_features.shape}")

                            # Log feature statistics
                            non_zero_pct = (node_features != 0).sum() / node_features.size * 100
                            logger.log_metric("node_features_nonzero_percent", round(non_zero_pct, 1))
                            logger.info(f"    Non-zero: {non_zero_pct:.1f}%")

                        # Extract enhanced edge features (16-dim)
                        edge_features_list = []
                        for idx, edge in enumerate(occ_edges):
                            uv_grid = edge_uv_grids.get(idx, None)
                            curve_type = get_curve_type_from_edge(edge)

                            # Get adjacent faces for this edge
                            topo = TopologyExplorer(doc._occ_shape)
                            adjacent_faces = list(topo.faces_from_edge(edge))

                            features = extract_enhanced_edge_features(edge, curve_type, uv_grid, adjacent_faces)
                            edge_features_list.append(features)

                        if edge_features_list:
                            edge_features = np.vstack(edge_features_list)
                            logger.log_metric("edge_feature_dim", edge_features.shape[1])
                            logger.log_metric("num_edges_features", edge_features.shape[0])
                            logger.info(f"  Edge features: {edge_features.shape}")

                            # Log feature statistics
                            non_zero_pct = (edge_features != 0).sum() / edge_features.size * 100
                            logger.log_metric("edge_features_nonzero_percent", round(non_zero_pct, 1))
                            logger.info(f"    Non-zero: {non_zero_pct:.1f}%")

                        # Save feature summary
                        feature_summary = {
                            "node_features": {
                                "shape": list(node_features.shape) if node_features is not None else None,
                                "mean": float(np.mean(node_features)) if node_features is not None else None,
                                "std": float(np.std(node_features)) if node_features is not None else None,
                                "min": float(np.min(node_features)) if node_features is not None else None,
                                "max": float(np.max(node_features)) if node_features is not None else None
                            },
                            "edge_features": {
                                "shape": list(edge_features.shape) if edge_features is not None else None,
                                "mean": float(np.mean(edge_features)) if edge_features is not None else None,
                                "std": float(np.std(edge_features)) if edge_features is not None else None,
                                "min": float(np.min(edge_features)) if edge_features is not None else None,
                                "max": float(np.max(edge_features)) if edge_features is not None else None
                            }
                        }
                        output_manager.save_json(feature_summary, "enhanced_features_summary.json", "intermediates")
                        logger.info(f"  ✓ Saved enhanced features summary")

                    except Exception as e:
                        logger.warning(f"Enhanced feature extraction failed: {e}")
            else:
                logger.info("Skipping enhanced feature extraction (no OCC faces available)")

            # Stage 8: PyTorch Geometric Export
            pyg_data = None

            if node_features is not None and edge_features is not None:
                with logger.stage("PyTorch Geometric Export"):
                    logger.info("Exporting to PyTorch Geometric format...")

                    try:
                        # Build face-to-face edge index from shared edges
                        edge_index_list = []
                        if HAS_OCC and occ_faces and occ_edges:
                            topo = TopologyExplorer(doc._occ_shape)

                            # For each edge, find adjacent faces and create edge connections
                            for edge in occ_edges:
                                adjacent_face_list = list(topo.faces_from_edge(edge))
                                if len(adjacent_face_list) == 2:
                                    # Find indices of these faces in occ_faces
                                    try:
                                        idx1 = occ_faces.index(adjacent_face_list[0])
                                        idx2 = occ_faces.index(adjacent_face_list[1])
                                        # Add bidirectional edges
                                        edge_index_list.append([idx1, idx2])
                                        edge_index_list.append([idx2, idx1])
                                    except (ValueError, IndexError):
                                        # Face not in our list, skip
                                        continue

                        if edge_index_list:
                            edge_index = np.array(edge_index_list, dtype=np.int64).T
                        else:
                            # Create empty edge index if no adjacency
                            edge_index = np.zeros((2, 0), dtype=np.int64)

                        logger.info(f"  Built edge index: {edge_index.shape} from {len(occ_faces)} faces")
                        logger.log_metric("edge_index_num_edges", edge_index.shape[1] if len(edge_index.shape) > 1 else 0)

                        # Export to PyG
                        pyg_data = export_to_pyg_with_uvgrids(
                            node_features=node_features,
                            edge_index=edge_index,
                            edge_features=edge_features,
                            face_uv_grids=face_uv_grids,
                            edge_uv_grids=edge_uv_grids,
                            labels=None,
                            metadata={
                                "source_file": test_file.name,
                                "num_faces": len(occ_faces),
                                "num_edges": len(occ_edges)
                            }
                        )

                        if pyg_data:
                            # Validate PyG data
                            validation_errors = validate_pyg_data(pyg_data)
                            if validation_errors:
                                logger.warning(f"PyG validation found {len(validation_errors)} issues:")
                                for error in validation_errors[:5]:  # Log first 5
                                    logger.warning(f"  - {error}")
                            else:
                                logger.info("  ✓ PyG data validation: PASSED")

                            logger.log_metric("pyg_validation_errors", len(validation_errors))

                            # Save PyG data
                            pyg_path = output_manager.get_output_dir() / "graph_data.pt"
                            save_pyg_data(pyg_data, pyg_path, include_metadata=True)

                            logger.log_metric("pyg_num_nodes", int(pyg_data.x.shape[0]))
                            logger.log_metric("pyg_num_edges", int(pyg_data.edge_index.shape[1]))
                            logger.info(f"  ✓ Saved PyG data: {pyg_path.name}")
                        else:
                            logger.warning("  PyG export returned None")

                    except Exception as e:
                        logger.warning(f"PyG export failed: {e}")
            else:
                logger.info("Skipping PyG export (no enhanced features available)")

            # Stage 9: Enhanced Visualizations
            if HAS_OCC and (face_uv_grids or dihedral_data or curvature_data or surface_type_data or hierarchy_data):
                with logger.stage("Enhanced Visualizations"):
                    logger.info("Generating enhanced visualizations...")

                    try:
                        viz = TopologyGraphVisualizer(
                            doc.topology if doc.topology else None,
                            output_manager.get_visualization_dir()
                        )

                        viz_files = []

                        # UV-grid samples
                        if face_uv_grids:
                            uv_viz = viz.visualize_uv_grid_samples(face_uv_grids, max_faces=6)
                            if uv_viz:
                                viz_files.append(uv_viz)
                                logger.info(f"  ✓ UV-grid samples: {uv_viz.name}")

                        # Dihedral distribution
                        if dihedral_data:
                            dihedral_viz = viz.visualize_dihedral_distribution(dihedral_data)
                            if dihedral_viz:
                                viz_files.append(dihedral_viz)
                                logger.info(f"  ✓ Dihedral distribution: {dihedral_viz.name}")

                        # Curvature distribution
                        if curvature_data:
                            curvature_viz = viz.visualize_curvature_distribution(curvature_data)
                            if curvature_viz:
                                viz_files.append(curvature_viz)
                                logger.info(f"  ✓ Curvature distribution: {curvature_viz.name}")

                        # Surface type distribution
                        if surface_type_data:
                            surface_viz = viz.visualize_surface_type_distribution(surface_type_data)
                            if surface_viz:
                                viz_files.append(surface_viz)
                                logger.info(f"  ✓ Surface type distribution: {surface_viz.name}")

                        # BRep hierarchy
                        if hierarchy_data:
                            hierarchy_viz = viz.visualize_brep_hierarchy(hierarchy_data)
                            if hierarchy_viz:
                                viz_files.append(hierarchy_viz)
                                logger.info(f"  ✓ BRep hierarchy: {hierarchy_viz.name}")

                        logger.log_metric("num_visualizations_generated", len(viz_files))

                    except Exception as e:
                        logger.warning(f"Enhanced visualizations failed: {e}")
            else:
                logger.info("Skipping enhanced visualizations (no data available)")

            # Stage 10: Enhanced Validation
            if face_uv_grids or dihedral_data or surface_type_data or pyg_data:
                with logger.stage("Enhanced Validation"):
                    logger.info("Running enhanced validation checks...")

                    validator = FunctionalValidator()
                    enhanced_validations = {}

                    try:
                        # Validate UV-grids
                        if face_uv_grids or edge_uv_grids:
                            uv_validation = validator.validate_uv_grid_features(face_uv_grids, edge_uv_grids)
                            enhanced_validations["uv_grids"] = uv_validation.to_dict()
                            logger.info(f"  UV-grid validation: {'PASS' if uv_validation.passed else 'FAIL'}")
                            if not uv_validation.passed:
                                logger.info(f"    {uv_validation.message}")

                        # Validate geometric distributions
                        if dihedral_data and curvature_data:
                            dist_validation = validator.validate_geometric_distributions(dihedral_data, curvature_data)
                            enhanced_validations["distributions"] = dist_validation.to_dict()
                            logger.info(f"  Distribution validation: {'PASS' if dist_validation.passed else 'FAIL'}")
                            if not dist_validation.passed:
                                logger.info(f"    {dist_validation.message}")

                        # Validate surface types
                        if surface_type_data:
                            surface_validation = validator.validate_surface_types(surface_type_data)
                            enhanced_validations["surface_types"] = surface_validation.to_dict()
                            logger.info(f"  Surface type validation: {'PASS' if surface_validation.passed else 'FAIL'}")
                            if not surface_validation.passed:
                                logger.info(f"    {surface_validation.message}")

                        # Validate PyG export
                        if pyg_data:
                            pyg_validation = validator.validate_pyg_export(pyg_data)
                            enhanced_validations["pyg_export"] = pyg_validation.to_dict()
                            logger.info(f"  PyG export validation: {'PASS' if pyg_validation.passed else 'FAIL'}")
                            if not pyg_validation.passed:
                                logger.info(f"    {pyg_validation.message}")

                        # Save enhanced validation results
                        if enhanced_validations:
                            output_manager.save_json(enhanced_validations, "enhanced_validation_results.json", "validation")
                            logger.info(f"  ✓ Saved enhanced validation results")

                    except Exception as e:
                        logger.warning(f"Enhanced validation failed: {e}")
            else:
                logger.info("Skipping enhanced validation (no enhanced data available)")

            # Stage 11: Validation
            with logger.stage("Validation"):
                logger.info("Validating document structure...")

                validator = FunctionalValidator()

                # Validate document structure
                doc_validation = validator.validate_document_structure(doc)
                logger.info(f"Document validation: {'PASS' if doc_validation.passed else 'FAIL'}")
                logger.info(f"  {doc_validation.message}")

                # Collect validation results
                validation_results = [doc_validation.to_dict()]

                # Save validation results
                validation_report = {
                    "file": test_file.name,
                    "validations": validation_results,
                    "overall_status": "PASS" if all(v["passed"] for v in validation_results) else "FAIL"
                }
                output_manager.save_json(validation_report, "validation_results.json", "validation")

            # Stage 6: Export Artifacts
            with logger.stage("Export Artifacts"):
                logger.info("Exporting document and summary...")

                # Export document summary
                doc_summary = {
                    "name": doc.name,
                    "format": str(doc.format) if doc.format else None,
                    "origin": str(doc.origin) if doc.origin else None,
                    "num_items": len(doc.items),
                    "entity_types": entity_types,
                    "has_topology": doc.topology is not None,
                    "topology_nodes": doc.topology.num_nodes if doc.topology else 0,
                    "topology_edges": doc.topology.num_edges if doc.topology else 0
                }
                output_manager.save_json(doc_summary, "document_summary.json", "reports")

                # Export markdown summary
                markdown = f"""# STEP Pipeline Functional Test Results

## File Information
- **Name**: {test_file.name}
- **Size**: {file_info['size_mb']:.2f} MB
- **Path**: {test_file}

## Document Summary
- **Name**: {doc.name}
- **Format**: {doc.format}
- **Total Items**: {len(doc.items)}
- **Unique Entity Types**: {len(entity_types)}

## Entity Type Distribution
"""
                for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
                    markdown += f"- {entity_type}: {count}\n"

                # Add topology information if available
                if doc.topology:
                    markdown += f"\n## Topology Graph\n"
                    markdown += f"- **Nodes**: {doc.topology.num_nodes}\n"
                    markdown += f"- **Edges**: {doc.topology.num_edges}\n"
                    markdown += f"- **Graph Density**: {round(density, 4) if topology.num_nodes > 1 else 0}\n"
                    markdown += f"- **Type**: {doc.metadata.get('topology_type', {}).get('representation_type', 'unknown') if hasattr(doc, 'metadata') else 'unknown'}\n"
                    markdown += f"\n### Generated Visualizations\n"
                    markdown += f"- `cad_entity_relationships.png` - CAD entity types, reference patterns, and BRep topology\n"
                    markdown += f"- `topology_degree_distribution.png` - Node in-degree and out-degree distributions\n"

                markdown += f"\n## Validation Results\n"
                markdown += f"- Overall Status: {validation_report['overall_status']}\n"
                for validation in validation_results:
                    markdown += f"- {validation['name']}: {'PASS' if validation['passed'] else 'FAIL'} - {validation['message']}\n"

                output_manager.save_text(markdown, "summary.md", "reports")

                logger.info(f"✓ Document summary exported")
                logger.info(f"✓ Markdown summary exported")

            # Create master summary report
            output_manager.create_summary_report()
            logger.info(f"✓ Master summary report created")

            # Save telemetry
            logger.save_telemetry()

            logger.info("="*80)
            logger.info("✓ Functional test completed successfully")
            logger.info(f"✓ Outputs saved to: {output_manager.run_dir}")
            logger.info("="*80)

        except Exception as e:
            logger.error(f"Functional test failed: {e}")
            logger.save_telemetry()
            raise

    def test_step_pipeline_batch_processing(self, test_data_step_path: Path, functional_output_dir: Path):
        """Demonstrate batch processing of multiple STEP files with different sizes.

        This test processes 2 files from each category (small, medium, large) to show:
        - Batch processing capabilities
        - Performance across different file sizes
        - Success/failure rates
        - Aggregate statistics

        Args:
            test_data_step_path: Path to STEP test data directory
            functional_output_dir: Base output directory for functional test outputs
        """
        # Setup output manager and telemetry logger
        output_manager = OutputManager("step_pipeline_batch", functional_output_dir)
        logger = TelemetryLogger(
            "step_pipeline_batch",
            output_manager.run_dir,
            console_level=20,  # INFO
            file_level=10  # DEBUG
        )

        logger.info("="*80)
        logger.info("FUNCTIONAL TEST: STEP Pipeline - Batch Processing")
        logger.info("="*80)

        try:
            # Stage 1: Collect Test Files
            with logger.stage("Collect Test Files"):
                categories = categorize_test_files(test_data_step_path, "*.stp")

                # Select 2 files from each available category
                test_files: List[Path] = []
                for category_name in [FileCategory.SMALL, FileCategory.MEDIUM, FileCategory.LARGE]:
                    files = categories.get(category_name, [])
                    if files:
                        # Take up to 2 files from this category
                        test_files.extend(files[:2])
                        logger.info(f"  {category_name}: {len(files[:2])} files selected")

                logger.log_metric("total_files_to_process", len(test_files))

                if not test_files:
                    pytest.skip("No test files found for batch processing")

            # Stage 2: Batch Processing
            batch_results: List[Dict[str, Any]] = []
            successful = 0
            failed = 0

            converter = DocumentConverter()

            with logger.stage("Batch Processing"):
                for i, test_file in enumerate(test_files, 1):
                    logger.info(f"Processing file {i}/{len(test_files)}: {test_file.name}")

                    file_result = {
                        "file_name": test_file.name,
                        "file_size_mb": test_file.stat().st_size / (1024 * 1024),
                        "file_path": str(test_file)
                    }

                    try:
                        # Convert file
                        import time
                        start_time = time.time()
                        result = converter.convert(test_file)
                        conversion_time = time.time() - start_time

                        file_result["status"] = result.status.value
                        file_result["conversion_time_seconds"] = round(conversion_time, 3)
                        file_result["num_items"] = len(result.document.items)
                        file_result["num_errors"] = len(result.errors)

                        if result.status.value == "success":
                            successful += 1
                            logger.info(f"  ✓ Success - {len(result.document.items)} items in {conversion_time:.2f}s")
                        else:
                            failed += 1
                            logger.warning(f"  ✗ Failed - {result.status.value}")

                    except Exception as e:
                        failed += 1
                        file_result["status"] = "error"
                        file_result["error"] = str(e)
                        logger.error(f"  ✗ Error: {e}")

                    batch_results.append(file_result)

                logger.log_metric("successful_conversions", successful)
                logger.log_metric("failed_conversions", failed)
                logger.log_metric("success_rate", round(successful / len(test_files) * 100, 1) if test_files else 0)

            # Stage 3: Aggregate Statistics
            with logger.stage("Aggregate Statistics"):
                successful_results = [r for r in batch_results if r.get("status") == "success"]

                if successful_results:
                    # Calculate statistics for successful conversions
                    conversion_times = [r["conversion_time_seconds"] for r in successful_results]
                    item_counts = [r["num_items"] for r in successful_results]
                    file_sizes = [r["file_size_mb"] for r in successful_results]

                    logger.info(f"Statistics for successful conversions:")
                    logger.info(f"  Conversion time: mean={np.mean(conversion_times):.2f}s, std={np.std(conversion_times):.2f}s")
                    logger.info(f"  Items per file: mean={np.mean(item_counts):.1f}, std={np.std(item_counts):.1f}")
                    logger.info(f"  File sizes: mean={np.mean(file_sizes):.2f} MB, std={np.std(file_sizes):.2f} MB")

                    logger.log_metric("mean_conversion_time", round(np.mean(conversion_times), 3))
                    logger.log_metric("mean_items_per_file", round(np.mean(item_counts), 1))

                # Save batch results
                batch_summary = {
                    "total_files": len(test_files),
                    "successful": successful,
                    "failed": failed,
                    "success_rate_percent": round(successful / len(test_files) * 100, 1) if test_files else 0,
                    "results": batch_results
                }
                output_manager.save_json(batch_summary, "batch_results.json", "reports")

            # Create master summary report
            output_manager.create_summary_report()

            # Save telemetry
            logger.save_telemetry()

            logger.info("="*80)
            logger.info("✓ Batch processing completed")
            logger.info(f"✓ Success rate: {successful}/{len(test_files)} ({batch_summary['success_rate_percent']}%)")
            logger.info(f"✓ Outputs saved to: {output_manager.run_dir}")
            logger.info("="*80)

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            logger.save_telemetry()
            raise
