# Archive Extraction Pipeline

Automatically extract STEP files from downloaded archives and organize them for training.

## Features

- **Recursive Archive Search**: Automatically finds all archives in downloads directory
- **Multiple Format Support**: ZIP, TAR, RAR, 7z (with optional dependencies)
- **Nested Archive Handling**: Recursively extracts archives within archives
- **Duplicate Tracking**: Skips already-processed archives and files using MD5 hashes
- **Safe Transfer**: Verifies file integrity before cleanup
- **Progress Reporting**: Real-time progress and detailed summary statistics
- **Dry Run Mode**: Preview what will be extracted without making changes

## Quick Start

### Basic Usage

Extract all archives from downloads directory:
```bash
cd /Users/ryanoboyle/LatticeLabs_toolkit/ll_stepnet
python -m tools.downloads_pipeline.archive_extractor
```

### Command-Line Options

```bash
# Process specific source directory
python -m tools.downloads_pipeline.archive_extractor --source data/downloads/odysee

# Specify custom output directory
python -m tools.downloads_pipeline.archive_extractor --output data/my_dataset

# Dry run (show what would be done)
python -m tools.downloads_pipeline.archive_extractor --dry-run

# Force reprocess all archives (ignore tracking database)
python -m tools.downloads_pipeline.archive_extractor --force

# Verbose logging
python -m tools.downloads_pipeline.archive_extractor --verbose

# Custom tracking database location
python -m tools.downloads_pipeline.archive_extractor --db /path/to/custom_db.json
```

## Installation

### Required Dependencies

The tool uses built-in Python libraries for ZIP and TAR files:
- `zipfile` (built-in)
- `tarfile` (built-in)

### Optional Dependencies

For RAR and 7z support, install additional packages:

```bash
# Install Python packages
pip install rarfile py7zr

# For RAR support, also install unrar binary:
# macOS:
brew install unrar

# Ubuntu/Debian:
sudo apt-get install unrar

# Windows: Download from https://www.rarlab.com/
```

## How It Works

### Processing Flow

1. **Scan**: Recursively searches `data/downloads/` for archive files
2. **Filter**: Skips archives already processed (tracked in `processed_files.json`)
3. **Extract**: Extracts each archive to a temporary directory
4. **Search**: Recursively finds all `.step` and `.stp` files
5. **Nested**: Recursively processes any archives found within archives
6. **Transfer**: Copies STEP files to `data/dataset_unstructured/`
7. **Verify**: Confirms successful transfer using file size and MD5 hash
8. **Track**: Updates tracking database with processed archives and files
9. **Cleanup**: Removes temporary directory ONLY after verification succeeds

### Directory Structure

**Input** (`data/downloads/`):
```
downloads/
├── odysee/
│   ├── FTN.2 Suppressor Pack.zip
│   ├── Bracket Assembly.rar
│   └── Parts Collection.7z
└── thingiverse/
    └── Gun Parts.zip
```

**Output** (`data/dataset_unstructured/`):
```
dataset_unstructured/
├── odysee/
│   ├── FTN.2 Suppressor Pack/
│   │   ├── suppressor_body.step
│   │   └── barrel_mount.step
│   ├── Bracket Assembly/
│   │   └── bracket.step
│   └── Parts Collection/
│       └── part1.step
└── thingiverse/
    └── Gun Parts/
        ├── receiver.step
        └── trigger.stp
```

## Tracking Database

The tool maintains a JSON database (`processed_files.json`) to track:

- **Processed Archives**: MD5 hash, path, processing date, files found/moved
- **Processed STEP Files**: MD5 hash, destination, source archive, size

This prevents:
- Reprocessing the same archive multiple times
- Duplicate STEP files even if re-downloaded with different names

### Database Structure

```json
{
  "processed_archives": {
    "/path/to/archive.zip": {
      "original_path": "/path/to/archive.zip",
      "processed_date": "2026-01-10T02:30:00",
      "step_files_found": 15,
      "step_files_moved": 15,
      "size_bytes": 1048576,
      "md5_hash": "abc123def456..."
    }
  },
  "processed_step_files": {
    "file_hash_xyz": {
      "original_name": "bracket.step",
      "destination_path": "/path/to/dataset/bracket.step",
      "source_archive": "/path/to/archive.zip",
      "size_bytes": 51200,
      "processed_date": "2026-01-10T02:30:15"
    }
  }
}
```

## Safety Features

### Transfer Verification

Every file transfer is verified using:
1. **Size Check**: Quick comparison of source and destination file sizes
2. **Hash Check**: MD5 hash comparison for data integrity
3. **Retry Logic**: Up to 3 attempts per file with automatic retry on failure

### Error Handling

- **Corrupt Archives**: Logged as error, skipped, processing continues
- **Extraction Failures**: Archive preserved for manual review
- **Transfer Failures**: Detailed logging, temp files cleaned up
- **Disk Full**: Processing halts with clear error message

### Cleanup Policy

- Temp directories are only deleted AFTER all files are verified
- Failed transfers keep temp directory for debugging
- Uses `try/finally` blocks to ensure cleanup even on errors

## Examples

### Example 1: Initial Run

```bash
$ python -m tools.downloads_pipeline.archive_extractor

======================================================================
Starting Archive Extraction Pipeline
======================================================================
Found 367 archives in /Users/ryanoboyle/LatticeLabs_toolkit/ll_stepnet/data/downloads
[1/367] Processing: FTN.2 Suppressor Pack.zip
  Found 25 STEP files
  Transferred: 25 successful, 0 failed
[2/367] Processing: Bracket Assembly.rar
  Found 3 STEP files
  Transferred: 3 successful, 0 failed
...

======================================================================
Processing Summary
======================================================================
Archives found:     367
Archives processed: 367
Archives skipped:   0
Archives failed:    2

STEP files found:   4,521
STEP files moved:   4,518
STEP files failed:  3
======================================================================
```

### Example 2: Subsequent Run (Skips Duplicates)

```bash
$ python -m tools.downloads_pipeline.archive_extractor

======================================================================
Starting Archive Extraction Pipeline
======================================================================
Found 367 archives in /Users/ryanoboyle/LatticeLabs_toolkit/ll_stepnet/data/downloads
[1/367] Skipping (already processed): FTN.2 Suppressor Pack.zip
[2/367] Skipping (already processed): Bracket Assembly.rar
...
[367/367] Skipping (already processed): Gun Parts.zip

======================================================================
Processing Summary
======================================================================
Archives found:     367
Archives processed: 0
Archives skipped:   367
Archives failed:    0

STEP files found:   0
STEP files moved:   0
STEP files failed:  0
======================================================================
```

### Example 3: Dry Run

```bash
$ python -m tools.downloads_pipeline.archive_extractor --dry-run

Archive Extractor initialized
  Downloads: /Users/ryanoboyle/LatticeLabs_toolkit/ll_stepnet/data/downloads
  Output: /Users/ryanoboyle/LatticeLabs_toolkit/ll_stepnet/data/dataset_unstructured
  Database: /Users/ryanoboyle/LatticeLabs_toolkit/ll_stepnet/tools/downloads_pipeline/processed_files.json
  Dry run: True
  Force reprocess: False
Found 367 archives in /Users/ryanoboyle/LatticeLabs_toolkit/ll_stepnet/data/downloads
[1/367] Processing: FTN.2 Suppressor Pack.zip
...
```

## Troubleshooting

### RAR Files Not Extracting

**Problem**: `RAR support not available`

**Solution**:
```bash
# Install rarfile Python package
pip install rarfile

# Install unrar binary (macOS)
brew install unrar
```

### 7z Files Not Extracting

**Problem**: `7z support not available`

**Solution**:
```bash
pip install py7zr
```

### Disk Space Issues

**Problem**: `No space left on device`

**Solution**:
- Free up disk space
- Process archives in smaller batches using `--source` flag
- Clean up `data/dataset_unstructured/` if needed

### Reset Tracking Database

To start fresh and reprocess all archives:

```bash
# Option 1: Use --force flag
python -m tools.downloads_pipeline.archive_extractor --force

# Option 2: Delete database file
rm tools/downloads_pipeline/processed_files.json
python -m tools.downloads_pipeline.archive_extractor
```

## Technical Details

### Supported Archive Formats

| Format | Extension | Handler | Dependency |
|--------|-----------|---------|------------|
| ZIP | `.zip` | `zipfile` | Built-in |
| TAR | `.tar`, `.tar.gz`, `.tgz`, `.tar.bz2`, `.tbz2` | `tarfile` | Built-in |
| RAR | `.rar` | `rarfile` | Optional (requires unrar) |
| 7-Zip | `.7z` | `py7zr` | Optional |

### Recursion Limits

- **Maximum recursion depth**: 10 levels (prevents infinite loops from circular references)
- Archives nested deeper than 10 levels will be logged as errors

### Performance

- **Sequential Processing**: One archive at a time to avoid resource exhaustion
- **Estimated Speed**: ~10 seconds per archive (varies by size)
- **367 Archives**: ~1 hour total processing time

## Integration with ll_stepnet

The extracted STEP files can be used with ll_stepnet's data loading:

```python
from stepnet.data import load_dataset_from_directory

# Load extracted dataset
dataset = load_dataset_from_directory(
    'data/dataset_unstructured',
    use_topology=True
)
```

## License

Part of the ll_stepnet project.
