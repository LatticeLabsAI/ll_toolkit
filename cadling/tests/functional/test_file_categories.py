"""Helper to categorize test files by size and complexity."""

from pathlib import Path
from typing import Dict, List


class FileCategory:
    """File size categories for testing."""
    SMALL = "small"      # < 100 KB - quick tests
    MEDIUM = "medium"    # 100 KB - 10 MB - typical files
    LARGE = "large"      # 10 MB - 100 MB - stress tests
    XLARGE = "xlarge"    # > 100 MB - extreme stress tests


def categorize_test_files(test_data_dir: Path, extension: str = "*.stp") -> Dict[str, List[Path]]:
    """Categorize test files by size.

    Args:
        test_data_dir: Directory containing test files
        extension: File extension pattern (e.g., "*.stp", "*.stl")

    Returns:
        Dictionary mapping category -> list of file paths
    """
    categories = {
        FileCategory.SMALL: [],
        FileCategory.MEDIUM: [],
        FileCategory.LARGE: [],
        FileCategory.XLARGE: []
    }

    for file_path in test_data_dir.glob(extension):
        if not file_path.is_file():
            continue

        size_bytes = file_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        if size_mb < 0.1:
            categories[FileCategory.SMALL].append(file_path)
        elif size_mb < 10:
            categories[FileCategory.MEDIUM].append(file_path)
        elif size_mb < 100:
            categories[FileCategory.LARGE].append(file_path)
        else:
            categories[FileCategory.XLARGE].append(file_path)

    return categories
