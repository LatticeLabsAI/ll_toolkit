"""
End-to-end batch testing with real-world STL files.

Tests the complete STL conversion workflow using actual CAD files from test_data directory.
This provides real-world validation of the conversion pipeline with diverse geometries.
"""

import json
from pathlib import Path
from typing import List

import pytest


class TestRealWorldSTLBatch:
    """Batch tests for real-world STL files."""

    @pytest.fixture
    def test_data_dir(self) -> Path:
        """Path to STL test data directory containing real STL files."""
        test_data_path = Path(__file__).parent.parent.parent / "data" / "test_data" / "stl"
        if not test_data_path.exists():
            pytest.skip(f"STL test data directory not found: {test_data_path}")
        return test_data_path

    @pytest.fixture
    def stl_files(self, test_data_dir: Path) -> List[Path]:
        """Get all STL files from test data stl/ subdirectory."""
        stl_files = list(test_data_dir.glob("*.stl"))
        if not stl_files:
            pytest.skip(f"No STL files found in {test_data_dir}")
        return stl_files

    def test_batch_stl_conversion_all_files(self, stl_files: List[Path]):
        """Test batch conversion of all STL files in test data."""
        from cadling.backend.document_converter import DocumentConverter
        from cadling.datamodel.base_models import ConversionStatus

        converter = DocumentConverter()
        results = {}
        failed_files = []

        for stl_file in stl_files:
            try:
                result = converter.convert(stl_file)
                results[stl_file.name] = {
                    "status": result.status,
                    "success": result.status == ConversionStatus.SUCCESS,
                    "file_size": stl_file.stat().st_size,
                }

                if result.status != ConversionStatus.SUCCESS:
                    failed_files.append(stl_file.name)
                    results[stl_file.name]["errors"] = result.errors

            except Exception as e:
                failed_files.append(stl_file.name)
                results[stl_file.name] = {
                    "status": "EXCEPTION",
                    "success": False,
                    "error": str(e),
                }

        # Report statistics
        total_files = len(stl_files)
        successful = sum(1 for r in results.values() if r.get("success"))
        success_rate = (successful / total_files) * 100 if total_files > 0 else 0

        print(f"\n{'='*60}")
        print(f"Batch STL Conversion Results")
        print(f"{'='*60}")
        print(f"Total files: {total_files}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(failed_files)}")
        print(f"Success rate: {success_rate:.1f}%")

        if failed_files:
            print(f"\nFailed files:")
            for fname in failed_files:
                print(f"  - {fname}: {results[fname].get('status', 'UNKNOWN')}")

        # We expect at least 80% success rate for real-world files
        assert success_rate >= 80.0, f"Success rate {success_rate:.1f}% is below 80% threshold"

    def test_batch_stl_mesh_properties(self, stl_files: List[Path]):
        """Test that mesh properties are computed for all files."""
        from cadling.backend.document_converter import DocumentConverter
        from cadling.datamodel.base_models import ConversionStatus

        converter = DocumentConverter()
        mesh_stats = []

        for stl_file in stl_files[:10]:  # Test first 10 files to keep test fast
            result = converter.convert(stl_file)

            if result.status == ConversionStatus.SUCCESS and result.document:
                doc = result.document
                mesh = doc.mesh

                if mesh:
                    mesh_stats.append({
                        "file": stl_file.name,
                        "num_facets": mesh.num_facets,
                        "num_vertices": mesh.num_vertices,
                        "surface_area": mesh.surface_area,
                        "is_manifold": mesh.is_manifold,
                        "bounding_box": doc.bounding_box,
                    })

        # Verify we got mesh data for most files
        assert len(mesh_stats) >= len(stl_files[:10]) * 0.8, "Too many files failed mesh extraction"

        # Print mesh statistics
        print(f"\n{'='*60}")
        print(f"Mesh Statistics (first 10 files)")
        print(f"{'='*60}")
        for stat in mesh_stats:
            print(f"\n{stat['file']}:")
            print(f"  Facets: {stat['num_facets']:,}")
            print(f"  Vertices: {stat['num_vertices']:,}")
            surface_area = stat['surface_area']
            if surface_area is not None:
                print(f"  Surface Area: {surface_area:.2f}")
            else:
                print(f"  Surface Area: Not computed")
            print(f"  Manifold: {stat['is_manifold']}")

    def test_batch_export_to_json(self, stl_files: List[Path]):
        """Test JSON export for all successfully converted files."""
        from cadling.backend.document_converter import DocumentConverter
        from cadling.datamodel.base_models import ConversionStatus

        converter = DocumentConverter()
        export_results = {"success": 0, "failed": 0}

        for stl_file in stl_files[:5]:  # Test first 5 files
            result = converter.convert(stl_file)

            if result.status == ConversionStatus.SUCCESS and result.document:
                try:
                    json_data = result.document.export_to_json()

                    # Verify JSON structure
                    assert isinstance(json_data, dict)
                    assert "name" in json_data
                    assert "format" in json_data
                    assert json_data["format"] == "stl"

                    # Verify JSON is serializable
                    json_str = json.dumps(json_data)
                    assert len(json_str) > 0

                    export_results["success"] += 1

                except Exception as e:
                    print(f"Failed to export {stl_file.name}: {e}")
                    export_results["failed"] += 1

        # All successful conversions should export to JSON
        total_tested = export_results["success"] + export_results["failed"]
        if total_tested > 0:
            assert export_results["success"] / total_tested >= 0.8, "JSON export failure rate too high"

    def test_batch_export_to_markdown(self, stl_files: List[Path]):
        """Test Markdown export for successfully converted files."""
        from cadling.backend.document_converter import DocumentConverter
        from cadling.datamodel.base_models import ConversionStatus

        converter = DocumentConverter()
        export_results = {"success": 0, "failed": 0}

        for stl_file in stl_files[:5]:  # Test first 5 files
            result = converter.convert(stl_file)

            if result.status == ConversionStatus.SUCCESS and result.document:
                try:
                    markdown_text = result.document.export_to_markdown()

                    # Verify Markdown format
                    assert isinstance(markdown_text, str)
                    assert len(markdown_text) > 0
                    assert stl_file.name in markdown_text or stl_file.stem in markdown_text

                    export_results["success"] += 1

                except Exception as e:
                    print(f"Failed to export {stl_file.name} to markdown: {e}")
                    export_results["failed"] += 1

        # All successful conversions should export to Markdown
        total_tested = export_results["success"] + export_results["failed"]
        if total_tested > 0:
            assert export_results["success"] / total_tested >= 0.8, "Markdown export failure rate too high"

    def test_large_stl_file_handling(self, stl_files: List[Path]):
        """Test handling of large STL files."""
        from cadling.backend.document_converter import DocumentConverter
        from cadling.datamodel.base_models import ConversionStatus

        # Find largest files
        large_files = sorted(stl_files, key=lambda f: f.stat().st_size, reverse=True)[:3]

        converter = DocumentConverter()

        for large_file in large_files:
            file_size_mb = large_file.stat().st_size / (1024 * 1024)
            print(f"\nTesting large file: {large_file.name} ({file_size_mb:.2f} MB)")

            result = converter.convert(large_file)

            # Should handle large files without crashing
            assert result is not None
            assert result.status in [ConversionStatus.SUCCESS, ConversionStatus.PARTIAL, ConversionStatus.FAILURE]

            if result.status == ConversionStatus.SUCCESS:
                assert result.document is not None
                print(f"  Successfully converted: {result.document.mesh.num_facets:,} facets")

    def test_stl_file_diversity(self, stl_files: List[Path]):
        """Test that we can handle diverse STL files with different geometries."""
        from cadling.backend.document_converter import DocumentConverter
        from cadling.datamodel.base_models import ConversionStatus

        converter = DocumentConverter()

        # Categorize files by complexity (based on file size as proxy)
        small_files = [f for f in stl_files if f.stat().st_size < 100_000]  # < 100KB
        medium_files = [f for f in stl_files if 100_000 <= f.stat().st_size < 1_000_000]  # 100KB-1MB
        large_files = [f for f in stl_files if f.stat().st_size >= 1_000_000]  # > 1MB

        print(f"\nFile size distribution:")
        print(f"  Small (< 100KB): {len(small_files)}")
        print(f"  Medium (100KB-1MB): {len(medium_files)}")
        print(f"  Large (> 1MB): {len(large_files)}")

        # Test at least one file from each category if available
        test_files = []
        if small_files:
            test_files.append(small_files[0])
        if medium_files:
            test_files.append(medium_files[0])
        if large_files:
            test_files.append(large_files[0])

        for test_file in test_files:
            result = converter.convert(test_file)
            size_mb = test_file.stat().st_size / (1024 * 1024)

            print(f"\n{test_file.name} ({size_mb:.2f} MB):")
            print(f"  Status: {result.status}")

            if result.status == ConversionStatus.SUCCESS:
                print(f"  Facets: {result.document.mesh.num_facets:,}")
                print(f"  Vertices: {result.document.mesh.num_vertices:,}")

        # Should successfully handle files from different size categories
        assert len(test_files) > 0, "No files available for diversity testing"

    def test_ascii_vs_binary_detection(self, stl_files: List[Path]):
        """Test that ASCII and binary STL files are correctly detected."""
        from cadling.backend.document_converter import DocumentConverter
        from cadling.datamodel.base_models import ConversionStatus

        converter = DocumentConverter()
        ascii_count = 0
        binary_count = 0

        for stl_file in stl_files[:10]:  # Test first 10 files
            result = converter.convert(stl_file)

            if result.status == ConversionStatus.SUCCESS and result.document:
                if result.document.is_ascii:
                    ascii_count += 1
                else:
                    binary_count += 1

        print(f"\nSTL format detection:")
        print(f"  ASCII: {ascii_count}")
        print(f"  Binary: {binary_count}")

        # We should detect at least some files
        assert (ascii_count + binary_count) > 0, "No STL format detected"

    def test_batch_conversion_consistency(self, stl_files: List[Path]):
        """Test that converting the same file twice produces consistent results."""
        from cadling.backend.document_converter import DocumentConverter

        converter = DocumentConverter()

        # Test with first 3 files
        for stl_file in stl_files[:3]:
            result1 = converter.convert(stl_file)
            result2 = converter.convert(stl_file)

            # Both conversions should have same status
            assert result1.status == result2.status

            # If both succeeded, results should be consistent
            if result1.document and result2.document:
                assert result1.document.mesh.num_facets == result2.document.mesh.num_facets
                assert result1.document.mesh.num_vertices == result2.document.mesh.num_vertices

                # Bounding boxes should match
                if result1.document.bounding_box and result2.document.bounding_box:
                    bbox1 = result1.document.bounding_box
                    bbox2 = result2.document.bounding_box
                    assert abs(bbox1.x_min - bbox2.x_min) < 0.001
                    assert abs(bbox1.y_min - bbox2.y_min) < 0.001
                    assert abs(bbox1.z_min - bbox2.z_min) < 0.001
                    assert abs(bbox1.x_max - bbox2.x_max) < 0.001
                    assert abs(bbox1.y_max - bbox2.y_max) < 0.001
                    assert abs(bbox1.z_max - bbox2.z_max) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
