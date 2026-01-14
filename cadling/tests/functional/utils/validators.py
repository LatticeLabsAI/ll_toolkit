"""Validation utilities for functional tests.

Provides utilities for validating outputs from real CAD processing workflows.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json


class Validators:
    """Validation utilities for functional test outputs."""

    @staticmethod
    def validate_file_exists(file_path: Path) -> Tuple[bool, Optional[str]]:
        """Validate that a file exists.

        Args:
            file_path: Path to file

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not file_path.exists():
            return False, f"File does not exist: {file_path}"
        
        if not file_path.is_file():
            return False, f"Path is not a file: {file_path}"
        
        return True, None

    @staticmethod
    def validate_file_not_empty(file_path: Path) -> Tuple[bool, Optional[str]]:
        """Validate that a file is not empty.

        Args:
            file_path: Path to file

        Returns:
            Tuple of (is_valid, error_message)
        """
        is_valid, error = Validators.validate_file_exists(file_path)
        if not is_valid:
            return is_valid, error

        size = file_path.stat().st_size
        if size == 0:
            return False, f"File is empty: {file_path}"
        
        return True, None

    @staticmethod
    def validate_json_file(file_path: Path, required_keys: Optional[List[str]] = None) -> Tuple[bool, Optional[str]]:
        """Validate a JSON file.

        Args:
            file_path: Path to JSON file
            required_keys: Optional list of required keys

        Returns:
            Tuple of (is_valid, error_message)
        """
        is_valid, error = Validators.validate_file_not_empty(file_path)
        if not is_valid:
            return is_valid, error

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"

        if required_keys:
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                return False, f"Missing required keys: {missing_keys}"

        return True, None

    @staticmethod
    def validate_dict_structure(
        data: Dict[str, Any],
        required_keys: List[str],
        optional_keys: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate dictionary structure.

        Args:
            data: Dictionary to validate
            required_keys: List of required keys
            optional_keys: Optional list of allowed optional keys

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required keys
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            return False, f"Missing required keys: {missing_keys}"

        # Check for unexpected keys
        if optional_keys is not None:
            allowed_keys = set(required_keys) | set(optional_keys)
            unexpected_keys = [key for key in data.keys() if key not in allowed_keys]
            if unexpected_keys:
                return False, f"Unexpected keys: {unexpected_keys}"

        return True, None

    @staticmethod
    def validate_positive_number(value: Any, name: str = "value") -> Tuple[bool, Optional[str]]:
        """Validate that a value is a positive number.

        Args:
            value: Value to validate
            name: Name of the value for error messages

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(value, (int, float)):
            return False, f"{name} must be a number, got {type(value).__name__}"

        if value <= 0:
            return False, f"{name} must be positive, got {value}"

        return True, None

    @staticmethod
    def validate_non_negative_number(value: Any, name: str = "value") -> Tuple[bool, Optional[str]]:
        """Validate that a value is a non-negative number.

        Args:
            value: Value to validate
            name: Name of the value for error messages

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(value, (int, float)):
            return False, f"{name} must be a number, got {type(value).__name__}"

        if value < 0:
            return False, f"{name} must be non-negative, got {value}"

        return True, None

    @staticmethod
    def validate_list_not_empty(value: Any, name: str = "list") -> Tuple[bool, Optional[str]]:
        """Validate that a value is a non-empty list.

        Args:
            value: Value to validate
            name: Name of the value for error messages

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(value, list):
            return False, f"{name} must be a list, got {type(value).__name__}"

        if len(value) == 0:
            return False, f"{name} must not be empty"

        return True, None

    @staticmethod
    def validate_string_not_empty(value: Any, name: str = "string") -> Tuple[bool, Optional[str]]:
        """Validate that a value is a non-empty string.

        Args:
            value: Value to validate
            name: Name of the value for error messages

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(value, str):
            return False, f"{name} must be a string, got {type(value).__name__}"

        if len(value.strip()) == 0:
            return False, f"{name} must not be empty"

        return True, None

    @staticmethod
    def validate_step_result(result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate STEP parsing result.

        Args:
            result: STEP parsing result dictionary

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required keys
        required_keys = ['header', 'entities']
        is_valid, error = Validators.validate_dict_structure(result, required_keys)
        if not is_valid:
            return is_valid, error

        # Validate entities is a dict
        if not isinstance(result['entities'], dict):
            return False, "entities must be a dictionary"

        # Validate entities is not empty
        if len(result['entities']) == 0:
            return False, "entities must not be empty"

        # Validate header is a dict
        if not isinstance(result['header'], dict):
            return False, "header must be a dictionary"

        return True, None

    @staticmethod
    def validate_graph_structure(graph_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate graph structure.

        Args:
            graph_data: Graph data dictionary

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for nodes and edges
        if 'nodes' not in graph_data and 'vertices' not in graph_data:
            return False, "Graph must have 'nodes' or 'vertices'"

        if 'edges' not in graph_data and 'links' not in graph_data:
            return False, "Graph must have 'edges' or 'links'"

        # Get node/edge lists
        nodes = graph_data.get('nodes') or graph_data.get('vertices', [])
        edges = graph_data.get('edges') or graph_data.get('links', [])

        # Validate they are lists
        if not isinstance(nodes, list):
            return False, "Nodes must be a list"

        if not isinstance(edges, list):
            return False, "Edges must be a list"

        # Validate not empty
        if len(nodes) == 0:
            return False, "Graph must have at least one node"

        return True, None

    @staticmethod
    def validate_geometry_data(geometry: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate geometry data structure.

        Args:
            geometry: Geometry data dictionary

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Common geometry fields
        expected_fields = ['vertices', 'faces', 'normals', 'bounds']
        
        # Check at least one expected field exists
        has_field = any(field in geometry for field in expected_fields)
        if not has_field:
            return False, f"Geometry must have at least one of: {expected_fields}"

        # If vertices exist, validate they're a list
        if 'vertices' in geometry:
            is_valid, error = Validators.validate_list_not_empty(
                geometry['vertices'], 
                'vertices'
            )
            if not is_valid:
                return is_valid, error

        return True, None

    @staticmethod
    def validate_pipeline_result(result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate pipeline processing result.

        Args:
            result: Pipeline result dictionary

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check success flag
        if 'success' in result:
            if not isinstance(result['success'], bool):
                return False, "success must be a boolean"
            
            if not result['success']:
                # If not successful, should have error message
                if 'error' not in result:
                    return False, "Failed result must have error message"

        # If successful, check for output data
        if result.get('success', True):
            expected_keys = ['data', 'output', 'result', 'entities', 'geometry']
            has_output = any(key in result for key in expected_keys)
            if not has_output:
                return False, f"Successful result must have one of: {expected_keys}"

        return True, None


class ValidationResult:
    """Result of a validation check."""

    def __init__(self, is_valid: bool, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize validation result.

        Args:
            is_valid: Whether validation passed
            message: Optional message describing the result
            details: Optional dictionary of additional details
        """
        self.is_valid = is_valid
        self.message = message or ("Validation passed" if is_valid else "Validation failed")
        self.details = details or {}

    def __bool__(self) -> bool:
        """Boolean conversion returns is_valid."""
        return self.is_valid

    def __str__(self) -> str:
        """String representation."""
        status = "PASS" if self.is_valid else "FAIL"
        return f"[{status}] {self.message}"


class FunctionalValidator:
    """High-level validator for functional test workflows."""

    def __init__(self, logger: Optional[Any] = None):
        """Initialize functional validator.

        Args:
            logger: Optional logger for validation messages
        """
        self.logger = logger
        self.validation_results: List[ValidationResult] = []

    def validate(self, check_name: str, is_valid: bool, message: Optional[str] = None, **details) -> ValidationResult:
        """Perform a validation check and log the result.

        Args:
            check_name: Name of the validation check
            is_valid: Whether the check passed
            message: Optional custom message
            **details: Additional details to include

        Returns:
            ValidationResult object
        """
        result = ValidationResult(is_valid, message or check_name, details)
        self.validation_results.append(result)

        if self.logger:
            # Support both standard logger and TelemetryLogger
            log_info = getattr(self.logger, 'info', None) or getattr(self.logger, 'log_info', None)
            log_debug = getattr(self.logger, 'debug', None) or getattr(self.logger, 'log_debug', None)
            log_error = getattr(self.logger, 'error', None) or getattr(self.logger, 'log_error', None)

            if is_valid:
                if log_info:
                    log_info(f"✓ {check_name}")
                if details and log_debug:
                    for key, value in details.items():
                        log_debug(f"  {key}: {value}")
            else:
                if log_error:
                    log_error(f"✗ {check_name}: {result.message}")
                if details and log_error:
                    for key, value in details.items():
                        log_error(f"  {key}: {value}")

        return result

    def validate_file_exists(self, file_path: Path, check_name: Optional[str] = None) -> ValidationResult:
        """Validate file exists.

        Args:
            file_path: Path to file
            check_name: Optional custom check name

        Returns:
            ValidationResult
        """
        is_valid, error = Validators.validate_file_exists(file_path)
        return self.validate(
            check_name or f"File exists: {file_path.name}",
            is_valid,
            error,
            file_path=str(file_path)
        )

    def validate_file_not_empty(self, file_path: Path, check_name: Optional[str] = None) -> ValidationResult:
        """Validate file is not empty.

        Args:
            file_path: Path to file
            check_name: Optional custom check name

        Returns:
            ValidationResult
        """
        is_valid, error = Validators.validate_file_not_empty(file_path)
        size = file_path.stat().st_size if file_path.exists() else 0
        return self.validate(
            check_name or f"File not empty: {file_path.name}",
            is_valid,
            error,
            file_path=str(file_path),
            size_bytes=size
        )

    def validate_dict_has_keys(
        self,
        data: Dict[str, Any],
        required_keys: List[str],
        check_name: Optional[str] = None
    ) -> ValidationResult:
        """Validate dictionary has required keys.

        Args:
            data: Dictionary to validate
            required_keys: List of required keys
            check_name: Optional custom check name

        Returns:
            ValidationResult
        """
        is_valid, error = Validators.validate_dict_structure(data, required_keys)
        return self.validate(
            check_name or "Dictionary structure",
            is_valid,
            error,
            required_keys=required_keys,
            present_keys=list(data.keys())
        )

    def validate_positive(self, value: Any, name: str, check_name: Optional[str] = None) -> ValidationResult:
        """Validate value is positive.

        Args:
            value: Value to validate
            name: Name of the value
            check_name: Optional custom check name

        Returns:
            ValidationResult
        """
        is_valid, error = Validators.validate_positive_number(value, name)
        return self.validate(
            check_name or f"{name} is positive",
            is_valid,
            error,
            value=value
        )

    def validate_list_not_empty(self, value: List, name: str, check_name: Optional[str] = None) -> ValidationResult:
        """Validate list is not empty.

        Args:
            value: List to validate
            name: Name of the list
            check_name: Optional custom check name

        Returns:
            ValidationResult
        """
        is_valid, error = Validators.validate_list_not_empty(value, name)
        length = len(value) if isinstance(value, list) else 0
        return self.validate(
            check_name or f"{name} not empty",
            is_valid,
            error,
            length=length
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results.

        Returns:
            Summary dictionary
        """
        total = len(self.validation_results)
        passed = sum(1 for r in self.validation_results if r.is_valid)
        failed = total - passed

        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / total if total > 0 else 0,
            'all_passed': failed == 0
        }

    def assert_all_passed(self):
        """Assert that all validations passed.

        Raises:
            AssertionError: If any validation failed
        """
        summary = self.get_summary()
        if not summary['all_passed']:
            failed_checks = [r for r in self.validation_results if not r.is_valid]
            error_msg = f"{summary['failed']} validation(s) failed:\n"
            for result in failed_checks:
                error_msg += f"  - {result.message}\n"
            raise AssertionError(error_msg)
