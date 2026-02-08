"""
Unit tests for ManufacturabilityAssessmentModel and DFM components.

Tests cover:
- DFMIssue, ManufacturabilityReport models
- DFM rule evaluation
- VLM-based assessment
- Report generation
"""

import pytest
from unittest.mock import Mock, patch

from cadling.experimental.models import (
    ManufacturabilityAssessmentModel,
    DFMIssue,
    DFMRule,
    IssueSeverity,
    ManufacturabilityReport,
    ManufacturingProcess,
)
from cadling.experimental.datamodel import CADAnnotationOptions


@pytest.fixture
def mock_doc():
    """Create a mock CADlingDocument."""
    doc = Mock()
    doc.topology = {"num_faces": 10, "num_edges": 20}
    return doc


@pytest.fixture
def mock_item_with_features():
    """Create a mock CADItem with detected features."""
    item = Mock()
    item.self_ref = "test_item"
    item.properties = {
        "rendered_images": {"isometric": Mock()},
        "machining_features": [
            {
                "feature_type": "hole",
                "parameters": {"diameter": 2.0, "depth": 50.0},
                "location": [10, 10],
            },
            {
                "feature_type": "pocket",
                "subtype": "rectangular_pocket",
                "parameters": {"length": 100.0, "width": 10.0, "depth": 80.0},
                "location": [50, 50],
            },
        ],
        "bounding_box": {"x": 200, "y": 100, "z": 50},
    }
    return item


class TestDFMIssue:
    """Test DFMIssue pydantic model."""

    def test_initialization(self):
        """Test DFMIssue initialization."""
        issue = DFMIssue(
            issue_type="thin_wall",
            severity=IssueSeverity.HIGH,
            description="Wall thickness below minimum",
            recommendation="Increase wall thickness to 0.5mm",
            estimated_impact="High risk of deflection",
            affected_processes=[ManufacturingProcess.CNC_MILLING],
        )

        assert issue.issue_type == "thin_wall"
        assert issue.severity == IssueSeverity.HIGH
        assert len(issue.affected_processes) == 1

    def test_optional_fields(self):
        """Test issue with optional fields."""
        issue = DFMIssue(
            issue_type="test", severity=IssueSeverity.LOW, description="Test issue"
        )

        assert issue.location is None
        assert issue.recommendation == ""
        assert issue.estimated_impact is None


class TestManufacturabilityReport:
    """Test ManufacturabilityReport pydantic model."""

    def test_initialization(self):
        """Test report initialization."""
        report = ManufacturabilityReport(
            overall_score=75.0,
            complexity_score=50.0,
            issues=[],
            suggested_processes=[ManufacturingProcess.CNC_MILLING],
            estimated_difficulty="Moderate difficulty",
            cost_drivers=["Material", "Setup time"],
            recommendations=["Add fillets to internal corners"],
        )

        assert report.overall_score == 75.0
        assert report.complexity_score == 50.0
        assert len(report.suggested_processes) == 1

    def test_score_validation(self):
        """Test score validation (0-100)."""
        # Valid scores
        report = ManufacturabilityReport(overall_score=0.0, complexity_score=0.0)
        assert report.overall_score == 0.0

        report = ManufacturabilityReport(overall_score=100.0, complexity_score=100.0)
        assert report.overall_score == 100.0

        # Out of bounds
        with pytest.raises(Exception):  # Pydantic ValidationError
            ManufacturabilityReport(overall_score=-1.0, complexity_score=50.0)

        with pytest.raises(Exception):
            ManufacturabilityReport(overall_score=101.0, complexity_score=50.0)


class TestIssueSeverity:
    """Test IssueSeverity enumeration."""

    def test_severity_levels(self):
        """Test all severity levels."""
        assert IssueSeverity.CRITICAL == "critical"
        assert IssueSeverity.HIGH == "high"
        assert IssueSeverity.MEDIUM == "medium"
        assert IssueSeverity.LOW == "low"
        assert IssueSeverity.INFO == "info"


class TestManufacturingProcess:
    """Test ManufacturingProcess enumeration."""

    def test_process_types(self):
        """Test all process types."""
        assert ManufacturingProcess.CNC_MILLING == "cnc_milling"
        assert ManufacturingProcess.CNC_TURNING == "cnc_turning"
        assert ManufacturingProcess.ADDITIVE == "additive_manufacturing"


class TestManufacturabilityAssessmentModel:
    """Test ManufacturabilityAssessmentModel."""

    @patch("cadling.experimental.models.manufacturability_assessment_model.ApiVlmModel")
    def test_initialization(self, mock_vlm_class):
        """Test model initialization."""
        options = CADAnnotationOptions(vlm_model="gpt-4-vision-preview")

        mock_vlm_instance = Mock()
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = ManufacturabilityAssessmentModel(options)

        assert model.options == options
        assert model.vlm is not None
        assert len(model.dfm_rules) > 0

    @patch("cadling.experimental.models.manufacturability_assessment_model.ApiVlmModel")
    def test_dfm_rules_initialized(self, mock_vlm_class):
        """Test that DFM rules are initialized."""
        options = CADAnnotationOptions()

        mock_vlm_instance = Mock()
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = ManufacturabilityAssessmentModel(options)

        # Should have multiple rules
        assert len(model.dfm_rules) >= 4
        # Check rule types
        rule_types = [type(rule).__name__ for rule in model.dfm_rules]
        assert "ThinWallRule" in rule_types
        assert "DeepPocketRule" in rule_types
        assert "SharpCornerRule" in rule_types
        assert "SmallHoleRule" in rule_types

    @patch("cadling.experimental.models.manufacturability_assessment_model.ApiVlmModel")
    def test_apply_dfm_rules(
        self, mock_vlm_class, mock_doc, mock_item_with_features
    ):
        """Test applying geometric DFM rules."""
        options = CADAnnotationOptions()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = ManufacturabilityAssessmentModel(options)

            # Apply DFM rules
            issues = model._apply_dfm_rules(mock_doc, mock_item_with_features)

        # Should detect issues (e.g., deep pocket, small hole)
        assert isinstance(issues, list)

    @patch("cadling.experimental.models.manufacturability_assessment_model.ApiVlmModel")
    def test_thin_wall_rule(self, mock_vlm_class, mock_doc):
        """Test thin wall detection rule."""
        from cadling.experimental.models.manufacturability_assessment_model import (
            ThinWallRule,
        )

        rule = ThinWallRule(min_thickness=0.5)

        # Item with thin wall feature
        item = Mock()
        item.properties = {
            "machining_features": [
                {
                    "feature_type": "rib",
                    "parameters": {"thickness": 0.3},  # Below minimum
                }
            ]
        }

        issues = rule.evaluate(mock_doc, item)

        assert len(issues) > 0
        assert issues[0].issue_type == "thin_wall"
        assert issues[0].severity == IssueSeverity.HIGH

    @patch("cadling.experimental.models.manufacturability_assessment_model.ApiVlmModel")
    def test_deep_pocket_rule(self, mock_vlm_class, mock_doc):
        """Test deep pocket detection rule."""
        from cadling.experimental.models.manufacturability_assessment_model import (
            DeepPocketRule,
        )

        rule = DeepPocketRule(max_depth_to_width_ratio=4.0)

        # Item with deep pocket
        item = Mock()
        item.properties = {
            "machining_features": [
                {
                    "feature_type": "pocket",
                    "parameters": {"depth": 50.0, "width": 10.0},  # Ratio = 5
                }
            ]
        }

        issues = rule.evaluate(mock_doc, item)

        assert len(issues) > 0
        assert issues[0].issue_type == "deep_pocket"

    @patch("cadling.experimental.models.manufacturability_assessment_model.ApiVlmModel")
    def test_sharp_corner_rule(self, mock_vlm_class, mock_doc):
        """Test sharp corner detection rule."""
        from cadling.experimental.models.manufacturability_assessment_model import (
            SharpCornerRule,
        )

        rule = SharpCornerRule()

        # Item with pockets but no fillets
        item = Mock()
        item.properties = {
            "machining_features": [
                {"feature_type": "pocket", "parameters": {}}
                # No fillets
            ]
        }

        issues = rule.evaluate(mock_doc, item)

        assert len(issues) > 0
        assert issues[0].issue_type == "sharp_internal_corners"

    @patch("cadling.experimental.models.manufacturability_assessment_model.ApiVlmModel")
    def test_small_hole_rule(self, mock_vlm_class, mock_doc):
        """Test small hole detection rule."""
        from cadling.experimental.models.manufacturability_assessment_model import (
            SmallHoleRule,
        )

        rule = SmallHoleRule(min_diameter=1.0)

        # Item with small hole
        item = Mock()
        item.properties = {
            "machining_features": [
                {"feature_type": "hole", "parameters": {"diameter": 0.5}}
            ]
        }

        issues = rule.evaluate(mock_doc, item)

        assert len(issues) > 0
        assert issues[0].issue_type == "small_hole"

    @patch("cadling.experimental.models.manufacturability_assessment_model.ApiVlmModel")
    def test_vlm_assessment(self, mock_vlm_class, mock_doc, mock_item_with_features):
        """Test VLM-based assessment."""
        options = CADAnnotationOptions()

        # Mock VLM response
        mock_vlm_instance = Mock()
        mock_response = Mock()
        mock_response.raw_text = """
        {
            "issues": [
                {
                    "issue_type": "complex_surface",
                    "severity": "medium",
                    "description": "Complex curved surface",
                    "recommendation": "Consider simplifying"
                }
            ],
            "suggested_processes": ["cnc_milling"],
            "complexity": "medium",
            "cost_drivers": ["Setup time"],
            "recommendations": ["Use 5-axis machining"]
        }
        """
        mock_vlm_instance.predict.return_value = mock_response
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = ManufacturabilityAssessmentModel(options)

            issues, insights = model._vlm_assessment(mock_doc, mock_item_with_features)

        assert len(issues) > 0
        assert "complexity" in insights

    @patch("cadling.experimental.models.manufacturability_assessment_model.ApiVlmModel")
    def test_generate_report(self, mock_vlm_class, mock_doc, mock_item_with_features):
        """Test report generation."""
        options = CADAnnotationOptions()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = ManufacturabilityAssessmentModel(options)

            issues = [
                DFMIssue(
                    issue_type="thin_wall",
                    severity=IssueSeverity.HIGH,
                    description="Test issue",
                )
            ]
            insights = {
                "complexity": "medium",
                "suggested_processes": ["cnc_milling"],
                "cost_drivers": ["Material"],
                "recommendations": ["Add fillets"],
            }

            report = model._generate_report(
                mock_doc, mock_item_with_features, issues, insights
            )

        assert isinstance(report, ManufacturabilityReport)
        assert report.overall_score <= 100.0
        assert report.overall_score >= 0.0
        assert len(report.issues) == len(issues)

    @patch("cadling.experimental.models.manufacturability_assessment_model.ApiVlmModel")
    def test_call_complete_workflow(
        self, mock_vlm_class, mock_doc, mock_item_with_features
    ):
        """Test complete assessment workflow."""
        options = CADAnnotationOptions()

        # Mock VLM
        mock_vlm_instance = Mock()
        mock_response = Mock()
        mock_response.raw_text = '{"issues": [], "complexity": "low", "suggested_processes": [], "cost_drivers": [], "recommendations": []}'
        mock_vlm_instance.predict.return_value = mock_response
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = ManufacturabilityAssessmentModel(options)
            model(mock_doc, [mock_item_with_features])

        # Check report was generated
        assert "manufacturability_report" in mock_item_with_features.properties
        report_dict = mock_item_with_features.properties["manufacturability_report"]
        assert "overall_score" in report_dict

    @patch("cadling.experimental.models.manufacturability_assessment_model.ApiVlmModel")
    def test_supports_batch_processing(self, mock_vlm_class):
        """Test batch processing support."""
        options = CADAnnotationOptions()

        mock_vlm_instance = Mock()
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = ManufacturabilityAssessmentModel(options)

        assert model.supports_batch_processing() is False

    @patch("cadling.experimental.models.manufacturability_assessment_model.ApiVlmModel")
    def test_requires_gpu(self, mock_vlm_class):
        """Test GPU requirements."""
        # API model - no GPU
        options = CADAnnotationOptions(vlm_model="gpt-4-vision-preview")

        mock_vlm_instance = Mock()
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = ManufacturabilityAssessmentModel(options)

        assert model.requires_gpu() is False

    @patch("cadling.experimental.models.manufacturability_assessment_model.ApiVlmModel")
    def test_get_model_info(self, mock_vlm_class):
        """Test model info retrieval."""
        options = CADAnnotationOptions()

        mock_vlm_instance = Mock()
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = ManufacturabilityAssessmentModel(options)

        info = model.get_model_info()

        assert "vlm_model" in info
        assert "dfm_rules" in info

    @patch("cadling.experimental.models.manufacturability_assessment_model.ApiVlmModel")
    def test_error_handling(self, mock_vlm_class, mock_doc, mock_item_with_features):
        """Test error handling during assessment."""
        options = CADAnnotationOptions()

        # Mock VLM to raise error
        mock_vlm_instance = Mock()
        mock_vlm_instance.predict.side_effect = Exception("VLM error")
        mock_vlm_class.return_value = mock_vlm_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = ManufacturabilityAssessmentModel(options)

            # Should not crash
            model(mock_doc, [mock_item_with_features])

        # Should still have manufacturability report generated (from DFM rules at least)
        assert "manufacturability_report" in mock_item_with_features.properties
        # May also have error recorded from VLM failure
        report = mock_item_with_features.properties["manufacturability_report"]
        assert "overall_score" in report
