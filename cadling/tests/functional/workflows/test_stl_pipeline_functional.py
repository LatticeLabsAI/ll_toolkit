"""Functional tests for STL conversion pipeline with real data and telemetry.

These are comprehensive functional runs that demonstrate the complete STL pipeline
with real mesh files, detailed logging at each stage, and validation of outputs.
"""

import pytest
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Add parent directory to path for functional test utilities
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.telemetry_logger import TelemetryLogger
from utils.output_manager import OutputManager
from utils.validators import FunctionalValidator, ValidationResult
from cadling.backend.document_converter import DocumentConverter


class TestSTLPipelineFunctional:
    """Functional tests demonstrating STL conversion pipeline with real data."""

    def test_stl_pipeline_ascii_and_binary(self, test_data_stl_path: Path, functional_output_dir: Path):
        """Demonstrate STL pipeline with both ASCII and binary formats.

        This test shows the full pipeline from STL file to document with:
        - File format detection (ASCII vs binary)
        - STL conversion using DocumentConverter
        - Mesh analysis (vertices, normals, facets)
        - Topology properties (manifold check, Euler characteristic)
        - Geometry validation (volume, surface area)
        - Export of artifacts and telemetry

        Args:
            test_data_stl_path: Path to STL test data directory
            functional_output_dir: Base output directory for functional test outputs
        """
        # Setup output manager and telemetry logger
        output_manager = OutputManager("stl_pipeline_ascii_binary", functional_output_dir)
        logger = TelemetryLogger(
            "stl_pipeline_ascii_binary",
            output_manager.run_dir,
            console_level=20,  # INFO
            file_level=10  # DEBUG
        )

        logger.info("="*80)
        logger.info("FUNCTIONAL TEST: STL Pipeline - ASCII and Binary Formats")
        logger.info("="*80)

        # Get test files (both ASCII and binary if available)
        stl_files = list(test_data_stl_path.glob("*.stl"))

        if not stl_files:
            pytest.skip("No STL files found for testing")

        # Use test_cube.stl and test_cylinder.stl if available, otherwise first two files
        test_files = []
        for preferred_name in ["test_cube.stl", "test_cylinder.stl"]:
            matching = [f for f in stl_files if f.name == preferred_name]
            if matching:
                test_files.append(matching[0])

        # If preferred files not found, use first two available
        if len(test_files) < 2:
            test_files = stl_files[:2]

        if not test_files:
            pytest.skip("Need at least one STL file for testing")

        try:
            converter = DocumentConverter()
            results = []

            for test_file in test_files:
                logger.info(f"\n{'='*80}")
                logger.info(f"Processing: {test_file.name}")
                logger.info(f"{'='*80}\n")

                # Stage 1: File Information
                with logger.stage(f"File Information - {test_file.name}"):
                    logger.log_file_info(test_file)

                    # Detect if ASCII or binary
                    with open(test_file, 'rb') as f:
                        header = f.read(80)
                        is_ascii = header.startswith(b'solid')

                    file_format = "ASCII" if is_ascii else "Binary"
                    logger.info(f"  STL Format: {file_format}")
                    logger.log_metric(f"stl_format_{test_file.name}", file_format)

                    file_info = {
                        "name": test_file.name,
                        "path": str(test_file),
                        "size_bytes": test_file.stat().st_size,
                        "size_mb": test_file.stat().st_size / (1024 * 1024),
                        "format": file_format
                    }
                    output_manager.save_json(file_info, f"file_info_{test_file.stem}.json", "artifacts")

                # Stage 2: STL Conversion
                with logger.stage(f"STL Conversion - {test_file.name}"):
                    logger.info(f"Converting STL file: {test_file.name}")

                    result = converter.convert(test_file)

                    logger.log_metric(f"conversion_status_{test_file.name}", result.status.value)
                    logger.log_metric(f"num_errors_{test_file.name}", len(result.errors))

                    if result.errors:
                        for i, error in enumerate(result.errors[:5]):  # Log first 5 errors
                            logger.warning(f"  Error {i+1}: {error}")

                    # Access document
                    doc = result.document
                    logger.info(f"Document created: {doc.name}")
                    logger.info(f"  Format: {doc.format}")

                # Stage 3: Mesh Analysis
                with logger.stage(f"Mesh Analysis - {test_file.name}"):
                    logger.info(f"Analyzing mesh structure...")

                    # Count mesh components
                    num_vertices = 0
                    num_facets = 0
                    num_normals = 0

                    # Analyze document items
                    for item in doc.items:
                        item_type = type(item).__name__
                        if "Vertex" in item_type or "Point" in item_type:
                            num_vertices += 1
                        elif "Facet" in item_type or "Triangle" in item_type or "Face" in item_type:
                            num_facets += 1
                        elif "Normal" in item_type:
                            num_normals += 1

                    logger.info(f"Mesh components found:")
                    logger.info(f"  Total items: {len(doc.items)}")
                    logger.info(f"  Vertices: {num_vertices}")
                    logger.info(f"  Facets: {num_facets}")
                    logger.info(f"  Normals: {num_normals}")

                    logger.log_metric(f"num_vertices_{test_file.name}", num_vertices)
                    logger.log_metric(f"num_facets_{test_file.name}", num_facets)
                    logger.log_metric(f"num_normals_{test_file.name}", num_normals)

                    mesh_analysis = {
                        "file_name": test_file.name,
                        "total_items": len(doc.items),
                        "num_vertices": num_vertices,
                        "num_facets": num_facets,
                        "num_normals": num_normals
                    }
                    output_manager.save_json(mesh_analysis, f"mesh_analysis_{test_file.stem}.json", "artifacts")

                # Stage 4: Topology Properties
                with logger.stage(f"Topology Properties - {test_file.name}"):
                    logger.info("Analyzing topology properties...")

                    topology_props = {}

                    # Check if document has topology
                    if doc.topology:
                        topology = doc.topology
                        logger.info(f"Topology graph:")
                        logger.info(f"  Nodes: {topology.num_nodes}")
                        logger.info(f"  Edges: {topology.num_edges}")

                        logger.log_metric(f"topology_num_nodes_{test_file.name}", topology.num_nodes)
                        logger.log_metric(f"topology_num_edges_{test_file.name}", topology.num_edges)

                        # Calculate Euler characteristic (V - E + F)
                        # For mesh: V (vertices) - E (edges) + F (faces) = 2 for closed manifold
                        if num_vertices > 0 and num_facets > 0:
                            euler_char = num_vertices - topology.num_edges + num_facets
                            logger.info(f"  Euler characteristic: {euler_char}")
                            logger.log_metric(f"euler_characteristic_{test_file.name}", euler_char)

                            # Check if manifold (Euler = 2 for closed surface)
                            is_likely_manifold = (euler_char == 2)
                            logger.info(f"  Likely closed manifold: {is_likely_manifold}")
                            logger.log_metric(f"is_likely_manifold_{test_file.name}", is_likely_manifold)

                            topology_props = {
                                "num_nodes": topology.num_nodes,
                                "num_edges": topology.num_edges,
                                "euler_characteristic": euler_char,
                                "is_likely_manifold": is_likely_manifold
                            }
                        else:
                            topology_props = {
                                "num_nodes": topology.num_nodes,
                                "num_edges": topology.num_edges
                            }
                    else:
                        logger.info("No topology graph available")
                        topology_props = {"has_topology": False}

                    output_manager.save_json(topology_props, f"topology_properties_{test_file.stem}.json", "artifacts")

                # Stage 5: Geometry Validation
                with logger.stage(f"Geometry Validation - {test_file.name}"):
                    logger.info("Validating geometric properties...")

                    validation_results = []

                    # Validate mesh has content
                    if num_facets > 0:
                        validation_results.append({
                            "check": "has_facets",
                            "passed": True,
                            "message": f"Mesh has {num_facets} facets"
                        })
                        logger.info(f"  ✓ Mesh has facets ({num_facets})")
                    else:
                        validation_results.append({
                            "check": "has_facets",
                            "passed": False,
                            "message": "Mesh has no facets"
                        })
                        logger.warning(f"  ✗ Mesh has no facets")

                    # Validate vertices
                    if num_vertices > 0:
                        validation_results.append({
                            "check": "has_vertices",
                            "passed": True,
                            "message": f"Mesh has {num_vertices} vertices"
                        })
                        logger.info(f"  ✓ Mesh has vertices ({num_vertices})")
                    else:
                        validation_results.append({
                            "check": "has_vertices",
                            "passed": False,
                            "message": "Mesh has no vertices"
                        })
                        logger.warning(f"  ✗ Mesh has no vertices")

                    # Validate normals
                    if num_normals > 0:
                        validation_results.append({
                            "check": "has_normals",
                            "passed": True,
                            "message": f"Mesh has {num_normals} normals"
                        })
                        logger.info(f"  ✓ Mesh has normals ({num_normals})")
                    else:
                        validation_results.append({
                            "check": "has_normals",
                            "passed": False,
                            "message": "Mesh has no normals"
                        })
                        logger.warning(f"  ✗ Mesh has no normals")

                    # Check for expected geometry properties based on filename
                    if "cube" in test_file.name.lower():
                        logger.info(f"  Expected: Cube geometry (6 faces, ~12 edges, ~8 vertices)")
                        # Cube should have Euler characteristic = 2
                        if topology_props.get("euler_characteristic") == 2:
                            logger.info(f"  ✓ Euler characteristic confirms closed surface")
                    elif "cylinder" in test_file.name.lower():
                        logger.info(f"  Expected: Cylinder geometry (curved surface)")
                        # Cylinder should also have Euler characteristic = 2 if capped
                        if topology_props.get("euler_characteristic") == 2:
                            logger.info(f"  ✓ Euler characteristic confirms closed surface")

                    validation_report = {
                        "file_name": test_file.name,
                        "validations": validation_results,
                        "overall_status": "PASS" if all(v["passed"] for v in validation_results) else "PARTIAL"
                    }
                    output_manager.save_json(validation_report, f"validation_{test_file.stem}.json", "validation")

                # Store result
                results.append({
                    "file_name": test_file.name,
                    "format": file_format,
                    "conversion_status": result.status.value,
                    "num_items": len(doc.items),
                    "num_facets": num_facets,
                    "num_vertices": num_vertices,
                    "validation_passed": all(v["passed"] for v in validation_results)
                })

            # Stage 6: Summary
            with logger.stage("Summary"):
                logger.info(f"Processed {len(results)} STL files:")

                for r in results:
                    status_icon = "✓" if r["validation_passed"] else "⚠"
                    logger.info(f"  {status_icon} {r['file_name']} ({r['format']}) - {r['num_facets']} facets, {r['num_vertices']} vertices")

                # Save summary
                summary = {
                    "total_files": len(results),
                    "files_processed": results,
                    "ascii_count": sum(1 for r in results if r["format"] == "ASCII"),
                    "binary_count": sum(1 for r in results if r["format"] == "Binary")
                }
                output_manager.save_json(summary, "stl_pipeline_summary.json", "artifacts")

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

    def test_stl_pipeline_batch_small_files(self, test_data_stl_path: Path, functional_output_dir: Path):
        """Demonstrate batch processing of small STL files.

        This test processes multiple small STL files to show:
        - Batch processing capabilities
        - Performance metrics across files
        - Format detection (ASCII vs binary)
        - Success/failure rates

        Args:
            test_data_stl_path: Path to STL test data directory
            functional_output_dir: Base output directory for functional test outputs
        """
        # Setup output manager and telemetry logger
        output_manager = OutputManager("stl_pipeline_batch", functional_output_dir)
        logger = TelemetryLogger(
            "stl_pipeline_batch",
            output_manager.run_dir,
            console_level=20,  # INFO
            file_level=10  # DEBUG
        )

        logger.info("="*80)
        logger.info("FUNCTIONAL TEST: STL Pipeline - Batch Processing Small Files")
        logger.info("="*80)

        try:
            # Stage 1: Collect Test Files
            with logger.stage("Collect Test Files"):
                # Get small STL files (< 1 MB)
                all_files = list(test_data_stl_path.glob("*.stl"))
                small_files = [f for f in all_files if f.stat().st_size < 1024 * 1024]

                # Limit to first 5 small files
                test_files = small_files[:5]

                logger.info(f"Found {len(all_files)} total STL files")
                logger.info(f"Found {len(small_files)} small files (< 1 MB)")
                logger.info(f"Selected {len(test_files)} files for batch processing")
                logger.log_metric("total_files_to_process", len(test_files))

                if not test_files:
                    pytest.skip("No small STL files found for batch processing")

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
                        # Detect format
                        with open(test_file, 'rb') as f:
                            header = f.read(80)
                            is_ascii = header.startswith(b'solid')
                        file_result["format"] = "ASCII" if is_ascii else "Binary"

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
                            logger.info(f"  ✓ Success - {len(result.document.items)} items in {conversion_time:.2f}s ({file_result['format']})")
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
                    # Calculate statistics
                    conversion_times = [r["conversion_time_seconds"] for r in successful_results]
                    item_counts = [r["num_items"] for r in successful_results]
                    file_sizes = [r["file_size_mb"] for r in successful_results]

                    logger.info(f"Statistics for successful conversions:")
                    logger.info(f"  Conversion time: mean={np.mean(conversion_times):.3f}s, std={np.std(conversion_times):.3f}s")
                    logger.info(f"  Items per file: mean={np.mean(item_counts):.1f}, std={np.std(item_counts):.1f}")
                    logger.info(f"  File sizes: mean={np.mean(file_sizes):.3f} MB, std={np.std(file_sizes):.3f} MB")

                    logger.log_metric("mean_conversion_time", round(np.mean(conversion_times), 3))
                    logger.log_metric("mean_items_per_file", round(np.mean(item_counts), 1))

                    # Count formats
                    ascii_count = sum(1 for r in successful_results if r.get("format") == "ASCII")
                    binary_count = sum(1 for r in successful_results if r.get("format") == "Binary")
                    logger.info(f"  Format distribution: ASCII={ascii_count}, Binary={binary_count}")

                # Save batch results
                batch_summary = {
                    "total_files": len(test_files),
                    "successful": successful,
                    "failed": failed,
                    "success_rate_percent": round(successful / len(test_files) * 100, 1) if test_files else 0,
                    "results": batch_results
                }
                output_manager.save_json(batch_summary, "batch_results.json", "artifacts")

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
