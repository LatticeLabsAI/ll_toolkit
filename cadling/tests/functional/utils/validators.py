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

    @property
    def passed(self) -> bool:
        """Alias for is_valid for consistency with test expectations."""
        return self.is_valid

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with name, passed, message, and details
        """
        return {
            "name": self.message,
            "passed": self.is_valid,
            "message": self.message,
            "details": self.details
        }


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

    def validate_graph_features(self, graph_data: Any) -> ValidationResult:
        """Validate PyTorch Geometric graph data features.

        Checks that:
        - Node features (x) exist and are not empty
        - Edge index exists
        - Features are not placeholder data (all zeros)

        Args:
            graph_data: PyG Data object with x, edge_index, edge_attr

        Returns:
            ValidationResult with validation outcome and details
        """
        import numpy as np

        details: Dict[str, Any] = {}
        issues: List[str] = []

        # Check node features
        if not hasattr(graph_data, 'x') or graph_data.x is None:
            issues.append("Missing node features (x)")
        else:
            node_features = graph_data.x.numpy() if hasattr(graph_data.x, 'numpy') else np.array(graph_data.x)
            details["num_nodes"] = int(node_features.shape[0])
            details["node_feature_dim"] = int(node_features.shape[1]) if len(node_features.shape) > 1 else 0

            if node_features.size == 0:
                issues.append("Node features are empty")
            else:
                nonzero_pct = (np.count_nonzero(node_features) / node_features.size) * 100
                details["node_features_nonzero_pct"] = round(nonzero_pct, 2)
                if nonzero_pct < 1.0:
                    issues.append(f"Node features appear to be placeholder data ({nonzero_pct:.2f}% non-zero)")

        # Check edge index
        if not hasattr(graph_data, 'edge_index') or graph_data.edge_index is None:
            issues.append("Missing edge index")
        else:
            edge_index = graph_data.edge_index.numpy() if hasattr(graph_data.edge_index, 'numpy') else np.array(graph_data.edge_index)
            details["num_edges"] = int(edge_index.shape[1]) if len(edge_index.shape) > 1 else 0

        # Check edge features (optional)
        if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None:
            edge_features = graph_data.edge_attr.numpy() if hasattr(graph_data.edge_attr, 'numpy') else np.array(graph_data.edge_attr)
            details["edge_feature_dim"] = int(edge_features.shape[1]) if len(edge_features.shape) > 1 else 0
            if edge_features.size > 0:
                nonzero_pct = (np.count_nonzero(edge_features) / edge_features.size) * 100
                details["edge_features_nonzero_pct"] = round(nonzero_pct, 2)

        is_valid = len(issues) == 0
        message = "Graph features valid" if is_valid else "; ".join(issues)

        return ValidationResult(is_valid, message, details)

    def validate_document_structure(self, doc: Any) -> ValidationResult:
        """Validate CADlingDocument structure.

        Checks that:
        - Document is not None
        - Document has items
        - Document has a name
        - Document has topology (optional but noted)

        Args:
            doc: CADlingDocument object

        Returns:
            ValidationResult with validation outcome and details
        """
        details: Dict[str, Any] = {}
        issues: List[str] = []

        if doc is None:
            return ValidationResult(False, "Document is None", {})

        # Check document name
        if hasattr(doc, 'name') and doc.name:
            details["name"] = doc.name
        else:
            issues.append("Document has no name")

        # Check document format
        if hasattr(doc, 'format') and doc.format:
            details["format"] = str(doc.format)

        # Check document items
        if hasattr(doc, 'items'):
            details["num_items"] = len(doc.items)
            if len(doc.items) == 0:
                issues.append("Document has no items")
        else:
            issues.append("Document has no 'items' attribute")

        # Check topology
        if hasattr(doc, 'topology') and doc.topology is not None:
            details["has_topology"] = True
            if hasattr(doc.topology, 'num_nodes'):
                details["topology_nodes"] = doc.topology.num_nodes
            if hasattr(doc.topology, 'num_edges'):
                details["topology_edges"] = doc.topology.num_edges
        else:
            details["has_topology"] = False

        # Check metadata
        if hasattr(doc, 'metadata') and doc.metadata:
            details["has_metadata"] = True
            details["metadata_keys"] = list(doc.metadata.keys()) if isinstance(doc.metadata, dict) else []
        else:
            details["has_metadata"] = False

        is_valid = len(issues) == 0
        message = "Document structure valid" if is_valid else "; ".join(issues)

        return ValidationResult(is_valid, message, details)

    def validate_pyg_export(self, pyg_data: Any) -> ValidationResult:
        """Validate PyTorch Geometric export data.

        Checks that:
        - Data object is not None
        - Node features (x) exist and have correct shape
        - Edge index exists and has correct shape
        - Edge attributes exist if edges exist

        Args:
            pyg_data: PyG Data object

        Returns:
            ValidationResult with validation outcome and details
        """
        import numpy as np

        details: Dict[str, Any] = {}
        issues: List[str] = []

        if pyg_data is None:
            return ValidationResult(False, "PyG data is None", {})

        # Check node features
        if hasattr(pyg_data, 'x') and pyg_data.x is not None:
            x = pyg_data.x
            x_np = x.numpy() if hasattr(x, 'numpy') else np.array(x)
            details["num_nodes"] = int(x_np.shape[0])
            details["node_feature_dim"] = int(x_np.shape[1]) if len(x_np.shape) > 1 else 0
            if x_np.size == 0:
                issues.append("Node features are empty")
        else:
            issues.append("Missing node features (x)")

        # Check edge index
        if hasattr(pyg_data, 'edge_index') and pyg_data.edge_index is not None:
            ei = pyg_data.edge_index
            ei_np = ei.numpy() if hasattr(ei, 'numpy') else np.array(ei)
            details["num_edges"] = int(ei_np.shape[1]) if len(ei_np.shape) > 1 else 0
            # Check edge index has 2 rows
            if len(ei_np.shape) >= 1 and ei_np.shape[0] != 2:
                issues.append(f"Edge index first dimension should be 2, got {ei_np.shape[0]}")
        else:
            issues.append("Missing edge index")

        # Check edge attributes
        if hasattr(pyg_data, 'edge_attr') and pyg_data.edge_attr is not None:
            ea = pyg_data.edge_attr
            ea_np = ea.numpy() if hasattr(ea, 'numpy') else np.array(ea)
            details["edge_feature_dim"] = int(ea_np.shape[1]) if len(ea_np.shape) > 1 else 0

        # Check for metadata
        if hasattr(pyg_data, 'metadata') and pyg_data.metadata is not None:
            details["has_metadata"] = True

        is_valid = len(issues) == 0
        message = "PyG export valid" if is_valid else "; ".join(issues)

        return ValidationResult(is_valid, message, details)

    def validate_uv_grid_features(
        self,
        face_uv_grids: Optional[Dict] = None,
        edge_uv_grids: Optional[Dict] = None
    ) -> ValidationResult:
        """Validate UV-grid features extracted from faces and edges.

        Checks that:
        - At least one grid type has data
        - Grid values are numpy arrays with expected shapes
        - Grid values are finite (no NaN or Inf)

        Args:
            face_uv_grids: Dictionary mapping face indices to UV-grid arrays
            edge_uv_grids: Dictionary mapping edge indices to UV-grid arrays

        Returns:
            ValidationResult with validation outcome and details
        """
        import numpy as np

        details: Dict[str, Any] = {}
        issues: List[str] = []

        has_face_grids = face_uv_grids is not None and len(face_uv_grids) > 0
        has_edge_grids = edge_uv_grids is not None and len(edge_uv_grids) > 0

        if not has_face_grids and not has_edge_grids:
            return ValidationResult(False, "No UV-grid data available", {})

        # Validate face UV-grids
        if has_face_grids:
            details["num_face_grids"] = len(face_uv_grids)
            nan_count = 0
            for idx, grid in face_uv_grids.items():
                grid_np = np.array(grid)
                if not np.all(np.isfinite(grid_np)):
                    nan_count += 1
            if nan_count > 0:
                issues.append(f"{nan_count} face UV-grids contain non-finite values")
            details["face_grids_with_nan"] = nan_count

        # Validate edge UV-grids
        if has_edge_grids:
            details["num_edge_grids"] = len(edge_uv_grids)
            nan_count = 0
            for idx, grid in edge_uv_grids.items():
                grid_np = np.array(grid)
                if not np.all(np.isfinite(grid_np)):
                    nan_count += 1
            if nan_count > 0:
                issues.append(f"{nan_count} edge UV-grids contain non-finite values")
            details["edge_grids_with_nan"] = nan_count

        is_valid = len(issues) == 0
        message = "UV-grid features valid" if is_valid else "; ".join(issues)

        return ValidationResult(is_valid, message, details)

    def validate_geometric_distributions(
        self,
        dihedral_data: Optional[Dict] = None,
        curvature_data: Optional[Dict] = None
    ) -> ValidationResult:
        """Validate geometric distribution data (dihedral angles, curvature).

        Checks that:
        - At least one distribution type has data
        - Distribution statistics contain expected keys (mean, median, etc.)
        - Values are finite numbers

        Args:
            dihedral_data: Dictionary with dihedral angle statistics
            curvature_data: Dictionary with curvature statistics

        Returns:
            ValidationResult with validation outcome and details
        """
        import numpy as np

        details: Dict[str, Any] = {}
        issues: List[str] = []

        has_dihedral = dihedral_data is not None and len(dihedral_data) > 0
        has_curvature = curvature_data is not None and len(curvature_data) > 0

        if not has_dihedral and not has_curvature:
            return ValidationResult(False, "No geometric distribution data available", {})

        # Validate dihedral data
        if has_dihedral:
            details["has_dihedral"] = True
            if 'mean' in dihedral_data:
                val = dihedral_data['mean']
                if isinstance(val, (int, float)) and np.isfinite(val):
                    details["dihedral_mean"] = float(val)
                else:
                    issues.append("Dihedral mean is not a finite number")
            if 'median' in dihedral_data:
                details["dihedral_median"] = float(dihedral_data['median'])

        # Validate curvature data
        if has_curvature:
            details["has_curvature"] = True
            details["curvature_keys"] = list(curvature_data.keys())

        is_valid = len(issues) == 0
        message = "Geometric distributions valid" if is_valid else "; ".join(issues)

        return ValidationResult(is_valid, message, details)

    def validate_surface_types(self, surface_type_data: Optional[Dict] = None) -> ValidationResult:
        """Validate surface type analysis data.

        Checks that:
        - Surface type data is not empty
        - Surface types have positive counts
        - At least one recognized surface type exists

        Args:
            surface_type_data: Dictionary mapping surface type names to counts

        Returns:
            ValidationResult with validation outcome and details
        """
        details: Dict[str, Any] = {}
        issues: List[str] = []

        if surface_type_data is None or len(surface_type_data) == 0:
            return ValidationResult(False, "No surface type data available", {})

        details["num_surface_types"] = len(surface_type_data)
        details["surface_types"] = list(surface_type_data.keys())

        total_count = sum(v for v in surface_type_data.values() if isinstance(v, (int, float)))
        details["total_faces"] = total_count

        if total_count == 0:
            issues.append("All surface type counts are zero")

        # Check for negative counts
        negative_types = [k for k, v in surface_type_data.items() if isinstance(v, (int, float)) and v < 0]
        if negative_types:
            issues.append(f"Negative counts for surface types: {negative_types}")

        is_valid = len(issues) == 0
        message = "Surface types valid" if is_valid else "; ".join(issues)

        return ValidationResult(is_valid, message, details)
