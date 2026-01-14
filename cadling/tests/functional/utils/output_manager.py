"""Output manager for functional tests.

Manages timestamped output directories and file organization for test artifacts.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import shutil
import json


class OutputManager:
    """Manages output directories and artifacts for functional tests."""

    def __init__(self, base_dir: Optional[Path] = None, test_name: Optional[str] = None):
        """Initialize output manager.

        Args:
            base_dir: Base directory for outputs (defaults to ./test_outputs)
            test_name: Name of the test (used for directory naming)
        """
        if base_dir is None:
            base_dir = Path.cwd() / "test_outputs"
        
        self.base_dir = Path(base_dir)
        self.test_name = test_name or "functional_test"
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_dir / f"{self.test_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.subdirs = {
            'logs': self.run_dir / 'logs',
            'outputs': self.run_dir / 'outputs',
            'intermediates': self.run_dir / 'intermediates',
            'visualizations': self.run_dir / 'visualizations',
            'reports': self.run_dir / 'reports'
        }

        for subdir in self.subdirs.values():
            subdir.mkdir(exist_ok=True)

        # Track artifacts
        self.artifacts: Dict[str, List[Path]] = {
            'logs': [],
            'outputs': [],
            'intermediates': [],
            'visualizations': [],
            'reports': []
        }

    def get_log_dir(self) -> Path:
        """Get the log directory.

        Returns:
            Path to log directory
        """
        return self.subdirs['logs']

    def get_output_dir(self) -> Path:
        """Get the output directory.

        Returns:
            Path to output directory
        """
        return self.subdirs['outputs']

    def get_intermediate_dir(self) -> Path:
        """Get the intermediate directory.

        Returns:
            Path to intermediate directory
        """
        return self.subdirs['intermediates']

    def get_visualization_dir(self) -> Path:
        """Get the visualization directory.

        Returns:
            Path to visualization directory
        """
        return self.subdirs['visualizations']

    def get_report_dir(self) -> Path:
        """Get the report directory.

        Returns:
            Path to report directory
        """
        return self.subdirs['reports']

    def save_artifact(
        self, 
        category: str, 
        filename: str, 
        content: Any,
        format: str = 'text'
    ) -> Path:
        """Save an artifact to the appropriate directory.

        Args:
            category: Category (logs, outputs, intermediates, visualizations, reports)
            filename: Name of the file
            content: Content to save
            format: Format ('text', 'json', 'binary')

        Returns:
            Path to saved artifact
        """
        if category not in self.subdirs:
            raise ValueError(f"Unknown category: {category}")

        file_path = self.subdirs[category] / filename

        if format == 'text':
            with open(file_path, 'w') as f:
                f.write(str(content))
        elif format == 'json':
            with open(file_path, 'w') as f:
                json.dump(content, f, indent=2)
        elif format == 'binary':
            with open(file_path, 'wb') as f:
                f.write(content)
        else:
            raise ValueError(f"Unknown format: {format}")

        self.artifacts[category].append(file_path)
        return file_path

    def copy_artifact(self, source: Path, category: str, new_name: Optional[str] = None) -> Path:
        """Copy a file to the artifact directory.

        Args:
            source: Source file path
            category: Category to copy to
            new_name: Optional new name for the file

        Returns:
            Path to copied artifact
        """
        if category not in self.subdirs:
            raise ValueError(f"Unknown category: {category}")

        dest_name = new_name or source.name
        dest_path = self.subdirs[category] / dest_name

        shutil.copy2(source, dest_path)
        self.artifacts[category].append(dest_path)
        
        return dest_path

    def get_artifacts(self, category: Optional[str] = None) -> List[Path]:
        """Get list of artifacts.

        Args:
            category: Optional category filter

        Returns:
            List of artifact paths
        """
        if category:
            return list(self.artifacts.get(category, []))
        
        # Return all artifacts
        all_artifacts = []
        for artifact_list in self.artifacts.values():
            all_artifacts.extend(artifact_list)
        return all_artifacts

    def create_summary_report(self) -> Path:
        """Create a summary report of all artifacts.

        Returns:
            Path to summary report
        """
        summary = {
            'test_name': self.test_name,
            'run_directory': str(self.run_dir),
            'timestamp': datetime.now().isoformat(),
            'artifacts': {}
        }

        for category, artifact_list in self.artifacts.items():
            summary['artifacts'][category] = [
                {
                    'name': path.name,
                    'path': str(path),
                    'size': path.stat().st_size if path.exists() else 0
                }
                for path in artifact_list
            ]

        report_path = self.get_report_dir() / 'summary.json'
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return report_path

    def cleanup(self, keep_logs: bool = True):
        """Clean up output directory.

        Args:
            keep_logs: Whether to keep log files
        """
        for category, subdir in self.subdirs.items():
            if category == 'logs' and keep_logs:
                continue
            
            if subdir.exists():
                shutil.rmtree(subdir)
                subdir.mkdir(exist_ok=True)

        # Clear artifact tracking
        for category in self.artifacts:
            if category == 'logs' and keep_logs:
                continue
            self.artifacts[category] = []

    def get_run_directory(self) -> Path:
        """Get the run directory path.

        Returns:
            Path to run directory
        """
        return self.run_dir
