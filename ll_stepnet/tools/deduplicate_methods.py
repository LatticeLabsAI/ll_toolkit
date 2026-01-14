#!/usr/bin/env python3
"""
Method Deduplication Tool

This script:
1. Analyzes Python methods in gatgpt/core module
2. Compares with methods in ll_stepnet
3. Identifies duplicate methods (by name and signature)
4. Copies non-duplicate methods to learning_curve directory

Usage:
    python3 tools/deduplicate_methods.py
"""

import os
import ast
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import shutil


class MethodAnalyzer:
    """Analyzes Python files to extract method information."""

    def __init__(self):
        self.methods = {}  # {method_name: [MethodInfo, ...]}
        self.files_analyzed = 0
        self.total_methods = 0

    def analyze_file(self, file_path: str) -> List[Dict]:
        """
        Extract all function/method definitions from a Python file.

        Returns:
            List of method info dictionaries
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source = f.read()

            tree = ast.parse(source, filename=file_path)
            methods = []

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_info = self._extract_method_info(node, file_path, source)
                    methods.append(method_info)

            self.files_analyzed += 1
            self.total_methods += len(methods)

            return methods

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return []

    def _extract_method_info(self, node: ast.FunctionDef, file_path: str, source: str) -> Dict:
        """Extract detailed information about a method."""
        # Get method signature
        args = []
        for arg in node.args.args:
            args.append(arg.arg)

        # Get method body (first few lines for comparison)
        try:
            body_lines = ast.get_source_segment(source, node)
            if body_lines:
                # Get first 10 lines of body for similarity comparison
                body_preview = '\n'.join(body_lines.split('\n')[:10])
            else:
                body_preview = ""
        except:
            body_preview = ""

        # Create method signature hash
        signature = f"{node.name}({', '.join(args)})"
        sig_hash = hashlib.md5(signature.encode()).hexdigest()[:8]

        # Create body hash for similarity detection
        body_hash = hashlib.md5(body_preview.encode()).hexdigest()[:8]

        return {
            'name': node.name,
            'args': args,
            'signature': signature,
            'sig_hash': sig_hash,
            'body_hash': body_hash,
            'body_preview': body_preview[:200],  # First 200 chars
            'file_path': file_path,
            'lineno': node.lineno,
            'is_async': isinstance(node, ast.AsyncFunctionDef),
            'decorators': [self._get_decorator_name(d) for d in node.decorator_list]
        }

    def _get_decorator_name(self, decorator) -> str:
        """Extract decorator name."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
        return str(decorator)

    def analyze_directory(self, directory: str, pattern: str = "**/*.py") -> Dict:
        """
        Analyze all Python files in a directory.

        Returns:
            Dictionary mapping method names to list of occurrences
        """
        dir_path = Path(directory)
        methods_dict = defaultdict(list)

        print(f"\nAnalyzing directory: {directory}")

        for py_file in dir_path.glob(pattern):
            if '__pycache__' in str(py_file):
                continue

            methods = self.analyze_file(str(py_file))

            for method in methods:
                methods_dict[method['name']].append(method)

        print(f"  Files analyzed: {self.files_analyzed}")
        print(f"  Total methods found: {self.total_methods}")
        print(f"  Unique method names: {len(methods_dict)}")

        return dict(methods_dict)


class MethodDeduplicator:
    """Identifies and handles duplicate methods."""

    def __init__(self):
        self.duplicates = []
        self.unique_methods = []

    def find_duplicates(
        self,
        source_methods: Dict,
        target_methods: Dict,
        similarity_threshold: float = 0.8
    ) -> Tuple[List, List]:
        """
        Find duplicate methods between source and target.

        Args:
            source_methods: Methods from source (e.g., gatgpt)
            target_methods: Methods from target (e.g., ll_stepnet)
            similarity_threshold: Threshold for considering methods similar

        Returns:
            (duplicate_methods, unique_methods)
        """
        duplicates = []
        unique = []

        for method_name, source_instances in source_methods.items():
            if method_name in target_methods:
                # Method name exists in target
                target_instances = target_methods[method_name]

                for src_method in source_instances:
                    is_duplicate = False

                    for tgt_method in target_instances:
                        # Check if signatures match
                        if src_method['sig_hash'] == tgt_method['sig_hash']:
                            duplicates.append({
                                'method_name': method_name,
                                'source': src_method,
                                'target': tgt_method,
                                'reason': 'exact_signature_match'
                            })
                            is_duplicate = True
                            break

                        # Check if body is very similar
                        if src_method['body_hash'] == tgt_method['body_hash']:
                            duplicates.append({
                                'method_name': method_name,
                                'source': src_method,
                                'target': tgt_method,
                                'reason': 'similar_body'
                            })
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        unique.append(src_method)
            else:
                # Method name doesn't exist in target - it's unique
                unique.extend(source_instances)

        self.duplicates = duplicates
        self.unique_methods = unique

        return duplicates, unique

    def generate_report(self, output_file: str = None):
        """Generate a detailed deduplication report."""
        report = []
        report.append("=" * 80)
        report.append("METHOD DEDUPLICATION REPORT")
        report.append("=" * 80)

        report.append(f"\nTotal duplicate methods found: {len(self.duplicates)}")
        report.append(f"Total unique methods found: {len(self.unique_methods)}")

        if self.duplicates:
            report.append("\n" + "=" * 80)
            report.append("DUPLICATE METHODS (Not Copied)")
            report.append("=" * 80)

            for dup in self.duplicates:
                report.append(f"\n  Method: {dup['method_name']}")
                report.append(f"  Signature: {dup['source']['signature']}")
                report.append(f"  Reason: {dup['reason']}")
                report.append(f"  Source: {dup['source']['file_path']}:{dup['source']['lineno']}")
                report.append(f"  Target: {dup['target']['file_path']}:{dup['target']['lineno']}")

        if self.unique_methods:
            report.append("\n" + "=" * 80)
            report.append("UNIQUE METHODS (Available for Copy)")
            report.append("=" * 80)

            # Group by file
            methods_by_file = defaultdict(list)
            for method in self.unique_methods:
                methods_by_file[method['file_path']].append(method)

            for file_path, methods in sorted(methods_by_file.items()):
                report.append(f"\n  File: {file_path}")
                for method in methods:
                    report.append(f"    - {method['signature']} (line {method['lineno']})")

        report_text = '\n'.join(report)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"\nReport saved to: {output_file}")

        return report_text


def copy_unique_methods(unique_methods: List[Dict], output_dir: str):
    """
    Copy files containing unique methods to output directory.

    Args:
        unique_methods: List of unique method info
        output_dir: Destination directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Group methods by source file
    files_to_copy = set()
    for method in unique_methods:
        files_to_copy.add(method['file_path'])

    copied_count = 0
    for file_path in files_to_copy:
        src = Path(file_path)
        # Create relative path structure
        rel_path = src.name  # Just use filename for simplicity

        dst = output_path / rel_path

        # If destination exists, append number
        if dst.exists():
            base_name = dst.stem
            suffix = dst.suffix
            counter = 1
            while dst.exists():
                dst = output_path / f"{base_name}_{counter}{suffix}"
                counter += 1

        try:
            shutil.copy2(src, dst)
            copied_count += 1
            print(f"  Copied: {src.name} -> {dst.name}")
        except Exception as e:
            print(f"  Error copying {src}: {e}")

    return copied_count


def main():
    """Main execution function."""
    print("=" * 80)
    print("PYTHON METHOD DEDUPLICATION TOOL")
    print("=" * 80)

    # Paths
    gatgpt_core = "/Users/ryanoboyle/gatgpt/core"
    ll_stepnet_base = "/Users/ryanoboyle/LatticeLabs_toolkit/ll_stepnet"
    output_dir = f"{ll_stepnet_base}/data/learning_curve/methods"
    report_file = f"{ll_stepnet_base}/data/learning_curve/deduplication_report.txt"

    # Step 1: Analyze gatgpt/core
    print("\n[Step 1] Analyzing gatgpt/core module...")
    gatgpt_analyzer = MethodAnalyzer()
    gatgpt_methods = gatgpt_analyzer.analyze_directory(gatgpt_core)

    # Step 2: Analyze ll_stepnet
    print("\n[Step 2] Analyzing ll_stepnet module...")
    stepnet_analyzer = MethodAnalyzer()
    stepnet_methods = stepnet_analyzer.analyze_directory(f"{ll_stepnet_base}/stepnet")

    # Also analyze data/learning_curve if it has Python files
    learning_curve_path = f"{ll_stepnet_base}/data/learning_curve"
    if Path(learning_curve_path).exists():
        print(f"\n[Step 3] Analyzing {learning_curve_path}...")
        lc_analyzer = MethodAnalyzer()
        lc_methods = lc_analyzer.analyze_directory(learning_curve_path)
        # Merge with stepnet_methods
        for name, methods in lc_methods.items():
            if name in stepnet_methods:
                stepnet_methods[name].extend(methods)
            else:
                stepnet_methods[name] = methods

    # Step 4: Find duplicates
    print("\n[Step 4] Finding duplicates...")
    deduplicator = MethodDeduplicator()
    duplicates, unique = deduplicator.find_duplicates(gatgpt_methods, stepnet_methods)

    print(f"  Duplicate methods: {len(duplicates)}")
    print(f"  Unique methods: {len(unique)}")

    # Step 5: Generate report
    print("\n[Step 5] Generating report...")
    report = deduplicator.generate_report(report_file)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  gatgpt/core methods: {gatgpt_analyzer.total_methods}")
    print(f"  ll_stepnet methods: {stepnet_analyzer.total_methods}")
    print(f"  Duplicate methods: {len(duplicates)}")
    print(f"  Unique methods: {len(unique)}")

    # Step 6: Copy unique method files
    if unique:
        print("\n[Step 6] Copying files with unique methods...")
        copied = copy_unique_methods(unique, output_dir)
        print(f"  Files copied: {copied}")
    else:
        print("\n[Step 6] No unique methods to copy")

    print("\n" + "=" * 80)
    print("DEDUPLICATION COMPLETE")
    print("=" * 80)
    print(f"  Report: {report_file}")
    print(f"  Unique methods copied to: {output_dir}")


if __name__ == '__main__':
    main()
