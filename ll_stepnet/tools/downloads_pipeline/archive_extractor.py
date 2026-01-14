"""
Archive Extractor for STEP Files.

Recursively processes archives in downloads directory, extracts STEP files,
and organizes them in the unstructured dataset directory.

Usage:
    python -m tools.downloads_pipeline.archive_extractor
    python -m tools.downloads_pipeline.archive_extractor --source data/downloads/odysee
    python -m tools.downloads_pipeline.archive_extractor --dry-run
"""

import os
import sys
import json
import hashlib
import shutil
import zipfile
import tarfile
import tempfile
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set
from datetime import datetime
import uuid

# Optional dependencies with graceful fallback
try:
    import rarfile
    RARFILE_AVAILABLE = True
except ImportError:
    RARFILE_AVAILABLE = False

try:
    import py7zr
    PY7ZR_AVAILABLE = True
except ImportError:
    PY7ZR_AVAILABLE = False


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ArchiveExtractor:
    """
    Extract STEP files from archives in the downloads directory.

    Features:
    - Recursive archive detection
    - One-at-a-time processing
    - Duplicate tracking via MD5 hashes
    - Safe transfer with verification
    - Nested archive handling
    - Progress reporting
    """

    ARCHIVE_EXTENSIONS = {'.zip', '.rar', '.7z', '.tar', '.tar.gz', '.tar.bz2', '.tgz', '.tbz2'}
    STEP_EXTENSIONS = {'.step', '.stp'}
    MAX_RECURSION_DEPTH = 10

    def __init__(
        self,
        downloads_dir: str,
        output_dir: str,
        db_path: Optional[str] = None,
        dry_run: bool = False,
        force: bool = False,
        verbose: bool = False
    ):
        """
        Initialize the archive extractor.

        Args:
            downloads_dir: Path to downloads directory
            output_dir: Path to output directory for STEP files
            db_path: Path to tracking database JSON file
            dry_run: If True, show what would be done without actually doing it
            force: If True, reprocess all archives ignoring tracking database
            verbose: If True, enable verbose logging
        """
        self.downloads_dir = Path(downloads_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.dry_run = dry_run
        self.force = force

        if verbose:
            logger.setLevel(logging.DEBUG)

        # Set up tracking database
        if db_path is None:
            db_path = Path(__file__).parent / 'processed_files.json'
        self.db_path = Path(db_path)

        # Load or initialize database
        self.db = self._load_database()

        # Statistics
        self.stats = {
            'archives_found': 0,
            'archives_processed': 0,
            'archives_skipped': 0,
            'archives_failed': 0,
            'step_files_found': 0,
            'step_files_moved': 0,
            'step_files_failed': 0,
        }

        # Ensure output directory exists
        if not self.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Archive Extractor initialized")
        logger.info(f"  Downloads: {self.downloads_dir}")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Database: {self.db_path}")
        logger.info(f"  Dry run: {self.dry_run}")
        logger.info(f"  Force reprocess: {self.force}")

    def _load_database(self) -> Dict:
        """Load the tracking database or create a new one."""
        if self.db_path.exists() and not self.force:
            try:
                with open(self.db_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load database: {e}")
                logger.warning("Starting with fresh database")

        return {
            'processed_archives': {},
            'processed_step_files': {}
        }

    def _save_database(self):
        """Save the tracking database."""
        if self.dry_run:
            return

        try:
            with open(self.db_path, 'w') as f:
                json.dump(self.db, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save database: {e}")

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute MD5 hash of a file."""
        md5 = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    md5.update(chunk)
            return md5.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to compute hash for {file_path}: {e}")
            return ""

    def find_archives(self) -> List[Path]:
        """
        Recursively find all archive files in downloads directory.

        Returns:
            List of paths to archive files
        """
        archives = []

        for root, dirs, files in os.walk(self.downloads_dir):
            root_path = Path(root)

            # Skip temp directories
            if 'temp' in root_path.parts or 'tmp' in root_path.parts:
                continue

            for file in files:
                file_path = root_path / file

                # Check if file has archive extension
                if any(str(file_path).lower().endswith(ext) for ext in self.ARCHIVE_EXTENSIONS):
                    archives.append(file_path)

        return sorted(archives)

    def is_processed(self, archive_path: Path) -> bool:
        """
        Check if archive has already been processed.

        Args:
            archive_path: Path to archive file

        Returns:
            True if already processed, False otherwise
        """
        if self.force:
            return False

        # Try to find by path first (faster)
        archive_str = str(archive_path)
        if archive_str in self.db['processed_archives']:
            return True

        # Try to find by hash (handles renamed files)
        archive_hash = self._compute_file_hash(archive_path)
        if not archive_hash:
            return False

        for record in self.db['processed_archives'].values():
            if record.get('md5_hash') == archive_hash:
                return True

        return False

    def _get_archive_type(self, archive_path: Path) -> Optional[str]:
        """Determine archive type from extension."""
        path_str = str(archive_path).lower()

        if path_str.endswith('.zip'):
            return 'zip'
        elif path_str.endswith('.rar'):
            return 'rar'
        elif path_str.endswith('.7z'):
            return '7z'
        elif path_str.endswith(('.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2')):
            return 'tar'

        return None

    def extract_archive(self, archive_path: Path, temp_dir: Path) -> bool:
        """
        Extract archive to temporary directory.

        Args:
            archive_path: Path to archive file
            temp_dir: Path to temporary extraction directory

        Returns:
            True if extraction successful, False otherwise
        """
        archive_type = self._get_archive_type(archive_path)

        if not archive_type:
            logger.error(f"Unknown archive type: {archive_path}")
            return False

        try:
            if archive_type == 'zip':
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    zf.extractall(temp_dir)
                return True

            elif archive_type == 'tar':
                with tarfile.open(archive_path, 'r:*') as tf:
                    tf.extractall(temp_dir)
                return True

            elif archive_type == 'rar':
                if not RARFILE_AVAILABLE:
                    logger.error(f"RAR support not available (install rarfile and unrar)")
                    return False
                with rarfile.RarFile(archive_path, 'r') as rf:
                    rf.extractall(temp_dir)
                return True

            elif archive_type == '7z':
                if not PY7ZR_AVAILABLE:
                    logger.error(f"7z support not available (install py7zr)")
                    return False
                with py7zr.SevenZipFile(archive_path, 'r') as szf:
                    szf.extractall(temp_dir)
                return True

        except Exception as e:
            logger.error(f"Failed to extract {archive_path}: {e}")
            return False

    def find_step_files(self, directory: Path) -> List[Path]:
        """
        Recursively find all STEP files in directory.

        Args:
            directory: Directory to search

        Returns:
            List of paths to STEP files
        """
        step_files = []

        for root, dirs, files in os.walk(directory):
            root_path = Path(root)

            for file in files:
                file_path = root_path / file

                # Check for STEP extensions
                if file_path.suffix.lower() in self.STEP_EXTENSIONS:
                    step_files.append(file_path)

        return step_files

    def find_nested_archives(self, directory: Path) -> List[Path]:
        """
        Recursively find nested archives in extracted directory.

        Args:
            directory: Directory to search

        Returns:
            List of paths to nested archive files
        """
        archives = []

        for root, dirs, files in os.walk(directory):
            root_path = Path(root)

            for file in files:
                file_path = root_path / file

                if any(str(file_path).lower().endswith(ext) for ext in self.ARCHIVE_EXTENSIONS):
                    archives.append(file_path)

        return archives

    def verify_transfer(self, source: Path, dest: Path) -> bool:
        """
        Verify file was successfully transferred.

        Args:
            source: Source file path
            dest: Destination file path

        Returns:
            True if transfer verified, False otherwise
        """
        if not dest.exists():
            return False

        # Quick check: file sizes
        if source.stat().st_size != dest.stat().st_size:
            logger.warning(f"Size mismatch: {source} vs {dest}")
            return False

        # Thorough check: MD5 hashes
        source_hash = self._compute_file_hash(source)
        dest_hash = self._compute_file_hash(dest)

        if source_hash != dest_hash:
            logger.warning(f"Hash mismatch: {source} vs {dest}")
            return False

        return True

    def _generate_destination_path(
        self,
        step_file: Path,
        archive_path: Path,
        temp_dir: Path
    ) -> Path:
        """
        Generate destination path with flattened structure.

        Strategy: All STEP files go directly into output_dir root.
        If filename collision occurs, append a counter to make it unique.
        Example: downloads/odysee/FTN.zip/parts/bracket.step
              -> dataset_unstructured/bracket.step
              -> dataset_unstructured/bracket_2.step (if collision)
        """
        # Start with just the filename in output root
        dest_path = self.output_dir / step_file.name

        # Handle filename collisions by appending counter
        if dest_path.exists():
            counter = 2
            stem = step_file.stem
            suffix = step_file.suffix
            while dest_path.exists():
                dest_path = self.output_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        return dest_path

    def _process_step_files(
        self,
        step_files: List[Path],
        archive_path: Path,
        temp_dir: Path,
        max_retries: int = 3
    ) -> tuple[int, int]:
        """
        Process and transfer STEP files.

        Args:
            step_files: List of STEP file paths
            archive_path: Source archive path
            temp_dir: Temporary extraction directory
            max_retries: Maximum number of transfer retries

        Returns:
            Tuple of (successful_transfers, failed_transfers)
        """
        successful = 0
        failed = 0

        for step_file in step_files:
            dest_path = self._generate_destination_path(step_file, archive_path, temp_dir)

            # Check if this file already exists
            step_hash = self._compute_file_hash(step_file)
            if step_hash in self.db['processed_step_files']:
                logger.debug(f"Skipping duplicate: {step_file.name}")
                self.stats['step_files_found'] += 1
                continue

            logger.debug(f"  Moving: {step_file.name} -> {dest_path}")

            if self.dry_run:
                successful += 1
                self.stats['step_files_found'] += 1
                self.stats['step_files_moved'] += 1
                continue

            # Try to transfer with retries
            for attempt in range(max_retries):
                try:
                    # Create parent directory
                    dest_path.parent.mkdir(parents=True, exist_ok=True)

                    # Copy file
                    shutil.copy2(step_file, dest_path)

                    # Verify transfer
                    if self.verify_transfer(step_file, dest_path):
                        # Record in database
                        self.db['processed_step_files'][step_hash] = {
                            'original_name': step_file.name,
                            'destination_path': str(dest_path),
                            'source_archive': str(archive_path),
                            'size_bytes': step_file.stat().st_size,
                            'processed_date': datetime.now().isoformat()
                        }

                        successful += 1
                        self.stats['step_files_found'] += 1
                        self.stats['step_files_moved'] += 1
                        break
                    else:
                        logger.warning(f"Verification failed for {step_file.name}, attempt {attempt + 1}/{max_retries}")
                        if dest_path.exists():
                            dest_path.unlink()

                except Exception as e:
                    logger.warning(f"Transfer failed for {step_file.name}: {e}, attempt {attempt + 1}/{max_retries}")

            else:
                # All retries failed
                logger.error(f"Failed to transfer {step_file.name} after {max_retries} attempts")
                failed += 1
                self.stats['step_files_failed'] += 1

        return successful, failed

    def process_archive(
        self,
        archive_path: Path,
        recursion_depth: int = 0
    ) -> bool:
        """
        Process a single archive file.

        Args:
            archive_path: Path to archive file
            recursion_depth: Current recursion depth (for nested archives)

        Returns:
            True if processing successful, False otherwise
        """
        if recursion_depth > self.MAX_RECURSION_DEPTH:
            logger.error(f"Maximum recursion depth exceeded for {archive_path}")
            return False

        logger.info(f"Processing: {archive_path.name}")

        # Create temporary extraction directory
        temp_dir = Path(tempfile.mkdtemp(prefix=f'stepnet_extract_{uuid.uuid4().hex[:8]}_'))

        try:
            # Extract archive
            if not self.dry_run:
                if not self.extract_archive(archive_path, temp_dir):
                    self.stats['archives_failed'] += 1
                    return False

            # Find STEP files
            step_files = self.find_step_files(temp_dir) if not self.dry_run else []
            logger.info(f"  Found {len(step_files)} STEP files")

            # Find nested archives
            nested_archives = self.find_nested_archives(temp_dir) if not self.dry_run else []
            if nested_archives:
                logger.info(f"  Found {len(nested_archives)} nested archives")

            # Process STEP files
            successful, failed = self._process_step_files(step_files, archive_path, temp_dir)
            logger.info(f"  Transferred: {successful} successful, {failed} failed")

            # Process nested archives recursively
            for nested_archive in nested_archives:
                logger.info(f"  Processing nested archive: {nested_archive.name}")
                self.process_archive(nested_archive, recursion_depth + 1)

            # Update database
            if not self.dry_run:
                archive_hash = self._compute_file_hash(archive_path)
                self.db['processed_archives'][str(archive_path)] = {
                    'original_path': str(archive_path),
                    'processed_date': datetime.now().isoformat(),
                    'step_files_found': len(step_files),
                    'step_files_moved': successful,
                    'size_bytes': archive_path.stat().st_size,
                    'md5_hash': archive_hash
                }
                self._save_database()

            self.stats['archives_processed'] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to process {archive_path}: {e}")
            self.stats['archives_failed'] += 1
            return False

        finally:
            # Clean up temp directory
            if not self.dry_run and temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")

    def process_all(self):
        """Process all archives in downloads directory."""
        logger.info("="*70)
        logger.info("Starting Archive Extraction Pipeline")
        logger.info("="*70)

        # Find all archives
        archives = self.find_archives()
        self.stats['archives_found'] = len(archives)

        logger.info(f"Found {len(archives)} archives in {self.downloads_dir}")

        if not archives:
            logger.info("No archives found to process")
            return

        # Process each archive
        for idx, archive_path in enumerate(archives, 1):
            # Check if already processed
            if self.is_processed(archive_path):
                logger.info(f"[{idx}/{len(archives)}] Skipping (already processed): {archive_path.name}")
                self.stats['archives_skipped'] += 1
                continue

            logger.info(f"[{idx}/{len(archives)}] Processing: {archive_path.name}")
            self.process_archive(archive_path)

        # Print summary
        self._print_summary()

    def _print_summary(self):
        """Print processing summary."""
        logger.info("="*70)
        logger.info("Processing Summary")
        logger.info("="*70)
        logger.info(f"Archives found:     {self.stats['archives_found']}")
        logger.info(f"Archives processed: {self.stats['archives_processed']}")
        logger.info(f"Archives skipped:   {self.stats['archives_skipped']}")
        logger.info(f"Archives failed:    {self.stats['archives_failed']}")
        logger.info(f"")
        logger.info(f"STEP files found:   {self.stats['step_files_found']}")
        logger.info(f"STEP files moved:   {self.stats['step_files_moved']}")
        logger.info(f"STEP files failed:  {self.stats['step_files_failed']}")
        logger.info("="*70)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Extract STEP files from archives in downloads directory'
    )

    parser.add_argument(
        '--source',
        type=str,
        default='data/downloads',
        help='Source downloads directory (default: data/downloads)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/dataset_unstructured',
        help='Output directory for STEP files (default: data/dataset_unstructured)'
    )

    parser.add_argument(
        '--db',
        type=str,
        default=None,
        help='Path to tracking database JSON file'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually doing it'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Reprocess all archives, ignoring tracking database'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Create extractor
    extractor = ArchiveExtractor(
        downloads_dir=args.source,
        output_dir=args.output,
        db_path=args.db,
        dry_run=args.dry_run,
        force=args.force,
        verbose=args.verbose
    )

    # Process all archives
    extractor.process_all()


if __name__ == '__main__':
    main()
