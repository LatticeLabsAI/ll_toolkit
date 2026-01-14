#!/usr/bin/env python3
"""
Flatten dataset structure by moving all STEP files to root directory.

This script finds all .step and .stp files in subdirectories of dataset_unstructured
and moves them to the root level, handling filename collisions by appending counters.
"""

import sys
from pathlib import Path
import shutil
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def flatten_dataset(dataset_dir: Path, dry_run: bool = False):
    """
    Move all STEP files from subdirectories to root of dataset_dir.

    Args:
        dataset_dir: Root dataset directory
        dry_run: If True, only show what would be done
    """
    if not dataset_dir.exists():
        logger.error(f"Dataset directory does not exist: {dataset_dir}")
        return

    # Find all STEP files in subdirectories (not in root)
    step_files = []
    for pattern in ['**/*.step', '**/*.stp', '**/*.STEP', '**/*.STP']:
        for file_path in dataset_dir.glob(pattern):
            # Skip files already in root
            if file_path.parent != dataset_dir:
                step_files.append(file_path)

    logger.info(f"Found {len(step_files)} STEP files in subdirectories")

    if len(step_files) == 0:
        logger.info("No files to move")
        return

    moved = 0
    failed = 0

    for step_file in step_files:
        # Generate destination path in root
        dest_path = dataset_dir / step_file.name

        # Handle filename collisions
        if dest_path.exists():
            counter = 2
            stem = step_file.stem
            suffix = step_file.suffix
            while dest_path.exists():
                dest_path = dataset_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        logger.debug(f"Moving: {step_file.relative_to(dataset_dir)} -> {dest_path.name}")

        if not dry_run:
            try:
                shutil.move(str(step_file), str(dest_path))
                moved += 1
            except Exception as e:
                logger.error(f"Failed to move {step_file.name}: {e}")
                failed += 1
        else:
            moved += 1

    logger.info(f"Moved: {moved}, Failed: {failed}")

    # Clean up empty directories
    if not dry_run:
        logger.info("Cleaning up empty directories...")
        cleanup_empty_dirs(dataset_dir)


def cleanup_empty_dirs(root_dir: Path):
    """
    Remove empty subdirectories recursively.

    Args:
        root_dir: Root directory to clean
    """
    removed = 0

    # Get all subdirectories, sorted by depth (deepest first)
    subdirs = [d for d in root_dir.rglob('*') if d.is_dir()]
    subdirs.sort(key=lambda p: len(p.parts), reverse=True)

    for subdir in subdirs:
        try:
            # Only remove if empty
            if not any(subdir.iterdir()):
                logger.debug(f"Removing empty directory: {subdir.relative_to(root_dir)}")
                subdir.rmdir()
                removed += 1
        except Exception as e:
            logger.warning(f"Could not remove {subdir}: {e}")

    logger.info(f"Removed {removed} empty directories")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Flatten dataset by moving all STEP files to root directory'
    )
    parser.add_argument(
        '--dataset-dir',
        type=Path,
        default=Path('data/dataset_unstructured'),
        help='Dataset directory (default: data/dataset_unstructured)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("="*70)
    logger.info("Flattening Dataset Structure")
    logger.info("="*70)
    logger.info(f"Dataset directory: {args.dataset_dir}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("")

    flatten_dataset(args.dataset_dir, args.dry_run)

    logger.info("="*70)
    logger.info("Done")
    logger.info("="*70)


if __name__ == '__main__':
    main()
