"""Manufacturability assessment model using DFM rules and VLM.

This module provides an enrichment model for assessing the manufacturability
of CAD parts, identifying potential manufacturing issues, and providing
recommendations for design improvements.

Classes:
    DFMRule: Base class for Design for Manufacturing rules
    ManufacturabilityAssessmentModel: Enrichment model for DFM assessment
    ManufacturabilityReport: Structured DFM assessment report

Example:
    from cadling.experimental.models import ManufacturabilityAssessmentModel
    from cadling.experimental.datamodel import CADAnnotationOptions

    options = CADAnnotationOptions(vlm_model="gpt-4-vision")
    model = ManufacturabilityAssessmentModel(options)
    model(doc, item_batch)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

from cadling.models.base_model import EnrichmentModel
from cadling.models.vlm_model import (
    ApiVlmModel,
    ApiVlmOptions,
    InlineVlmModel,
    InlineVlmOptions,
    VlmModel,
)

if TYPE_CHECKING:
    from cadling.datamodel.base_models import CADItem, CADlingDocument
    from cadling.experimental.datamodel import CADAnnotationOptions

_log = logging.getLogger(__name__)


class IssueSeverity(str, Enum):
    """Severity level for manufacturing issues."""

    CRITICAL = "critical"  # Will prevent manufacturing
    HIGH = "high"  # Significant cost/time impact
    MEDIUM = "medium"  # Moderate impact
    LOW = "low"  # Minor impact
    INFO = "info"  # Informational only


class ManufacturingProcess(str, Enum):
    """Common manufacturing processes."""

    CNC_MILLING = "cnc_milling"
    CNC_TURNING = "cnc_turning"
    DRILLING = "drilling"
    GRINDING = "grinding"
    EDM = "edm"
    CASTING = "casting"
    FORGING = "forging"
    STAMPING = "stamping"
    ADDITIVE = "additive_manufacturing"
    SHEET_METAL = "sheet_metal"


class DFMIssue(BaseModel):
    """Represents a Design for Manufacturing issue.

    Attributes:
        issue_type: Type of issue (thin_wall, deep_pocket, sharp_corner, etc.)
        severity: Severity level of the issue
        description: Human-readable description
        location: Location in model (if identifiable)
        recommendation: Suggested fix or improvement
        estimated_impact: Estimated impact on cost/time
        affected_processes: Manufacturing processes affected
    """

    issue_type: str
    severity: IssueSeverity
    description: str
    location: Optional[Dict[str, Any]] = None
    recommendation: str = ""
    estimated_impact: Optional[str] = None
    affected_processes: List[ManufacturingProcess] = Field(default_factory=list)


class ManufacturabilityReport(BaseModel):
    """Structured manufacturability assessment report.

    Attributes:
        overall_score: Overall manufacturability score (0-100)
        complexity_score: Manufacturing complexity score (0-100)
        issues: List of identified DFM issues
        suggested_processes: Recommended manufacturing processes
        estimated_difficulty: Overall difficulty assessment
        cost_drivers: Main factors affecting manufacturing cost
        recommendations: General manufacturing recommendations
    """

    overall_score: float = Field(ge=0.0, le=100.0)
    complexity_score: float = Field(ge=0.0, le=100.0)
    issues: List[DFMIssue] = Field(default_factory=list)
    suggested_processes: List[ManufacturingProcess] = Field(default_factory=list)
    estimated_difficulty: str = ""
    cost_drivers: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class DFMRule(ABC):
    """Base class for Design for Manufacturing rules."""

    @abstractmethod
    def evaluate(
        self, doc: "CADlingDocument", item: "CADItem"
    ) -> List[DFMIssue]:
        """Evaluate DFM rule against a CAD item.

        Args:
            doc: The CAD document
            item: The CAD item to evaluate

        Returns:
            List of DFM issues found
        """
        pass


class ThinWallRule(DFMRule):
    """Check for thin walls that may be difficult to machine."""

    def __init__(self, min_thickness: float = 0.5):
        self.min_thickness = min_thickness

    def evaluate(
        self, doc: "CADlingDocument", item: "CADItem"
    ) -> List[DFMIssue]:
        issues = []
        # Check for thin features in detected features
        features = item.properties.get("machining_features", [])
        for feature in features:
            params = feature.get("parameters", {})
            thickness = params.get("thickness") or params.get("width")
            if thickness and thickness < self.min_thickness:
                issues.append(
                    DFMIssue(
                        issue_type="thin_wall",
                        severity=IssueSeverity.HIGH,
                        description=f"Thin wall detected: {thickness}mm (min recommended: {self.min_thickness}mm)",
                        location=feature.get("location"),
                        recommendation=f"Increase wall thickness to at least {self.min_thickness}mm for reliable machining",
                        estimated_impact="High risk of vibration, deflection, and tool breakage",
                        affected_processes=[
                            ManufacturingProcess.CNC_MILLING,
                            ManufacturingProcess.CNC_TURNING,
                        ],
                    )
                )
        return issues


class DeepPocketRule(DFMRule):
    """Check for deep pockets that may be difficult to machine."""

    def __init__(self, max_depth_to_width_ratio: float = 4.0):
        self.max_ratio = max_depth_to_width_ratio

    def evaluate(
        self, doc: "CADlingDocument", item: "CADItem"
    ) -> List[DFMIssue]:
        issues = []
        features = item.properties.get("machining_features", [])
        for feature in features:
            if feature.get("feature_type") in ["pocket", "slot"]:
                params = feature.get("parameters", {})
                depth = params.get("depth", 0)
                width = params.get("width") or params.get("diameter", 1)

                if width > 0 and depth / width > self.max_ratio:
                    issues.append(
                        DFMIssue(
                            issue_type="deep_pocket",
                            severity=IssueSeverity.MEDIUM,
                            description=f"Deep pocket detected: depth/width ratio = {depth/width:.1f} (max recommended: {self.max_ratio})",
                            location=feature.get("location"),
                            recommendation="Consider redesigning to reduce depth or increase width, or plan for specialized tooling",
                            estimated_impact="May require special long-reach tooling and multiple operations",
                            affected_processes=[ManufacturingProcess.CNC_MILLING],
                        )
                    )
        return issues


class SharpCornerRule(DFMRule):
    """Check for sharp internal corners that cannot be machined."""

    def evaluate(
        self, doc: "CADlingDocument", item: "CADItem"
    ) -> List[DFMIssue]:
        issues = []
        features = item.properties.get("machining_features", [])

        # Check for pockets without fillets
        pockets = [f for f in features if f.get("feature_type") == "pocket"]
        fillets = [f for f in features if f.get("feature_type") == "fillet"]

        if pockets and not fillets:
            issues.append(
                DFMIssue(
                    issue_type="sharp_internal_corners",
                    severity=IssueSeverity.HIGH,
                    description="Pockets detected without internal fillets",
                    recommendation="Add fillets to internal corners (min radius = tool radius). Sharp internal corners cannot be machined with rotating tools.",
                    estimated_impact="Sharp corners are impossible to machine; part will have radiused corners equal to the smallest tool radius",
                    affected_processes=[
                        ManufacturingProcess.CNC_MILLING,
                        ManufacturingProcess.EDM,
                    ],
                )
            )

        return issues


class SmallHoleRule(DFMRule):
    """Check for very small holes that are difficult to machine."""

    def __init__(self, min_diameter: float = 1.0):
        self.min_diameter = min_diameter

    def evaluate(
        self, doc: "CADlingDocument", item: "CADItem"
    ) -> List[DFMIssue]:
        issues = []
        features = item.properties.get("machining_features", [])

        for feature in features:
            if feature.get("feature_type") == "hole":
                params = feature.get("parameters", {})
                diameter = params.get("diameter", 0)

                if 0 < diameter < self.min_diameter:
                    issues.append(
                        DFMIssue(
                            issue_type="small_hole",
                            severity=IssueSeverity.MEDIUM,
                            description=f"Small hole detected: {diameter}mm diameter (min recommended: {self.min_diameter}mm)",
                            location=feature.get("location"),
                            recommendation=f"Increase hole diameter to at least {self.min_diameter}mm if possible",
                            estimated_impact="Small holes are more expensive to drill and have higher breakage risk",
                            affected_processes=[ManufacturingProcess.DRILLING],
                        )
                    )

        return issues


class ManufacturabilityAssessmentModel(EnrichmentModel):
    """Enrichment model for assessing CAD part manufacturability.

    This model combines geometric analysis with VLM-based visual assessment
    to evaluate manufacturability and provide DFM feedback. It:

    1. **Geometric Rule Checking**: Applies DFM rules to extracted features
       - Thin wall detection
       - Deep pocket analysis
       - Sharp corner identification
       - Small hole detection
       - Tolerance analysis

    2. **VLM-Based Assessment**: Uses vision to identify:
       - Hard-to-access features
       - Complex surface geometries
       - Undercuts and difficult orientations
       - Tool access limitations

    3. **Manufacturing Process Selection**: Recommends appropriate processes

    4. **Cost and Complexity Estimation**: Provides difficulty scoring

    The model generates a comprehensive ManufacturabilityReport with
    actionable recommendations.

    Attributes:
        options: Configuration options
        vlm: Vision-language model for visual assessment
        dfm_rules: List of DFM rules to apply

    Example:
        options = CADAnnotationOptions(vlm_model="gpt-4-vision")
        model = ManufacturabilityAssessmentModel(options)
        model(doc, [item])

        # Access manufacturability report
        report = item.properties.get("manufacturability_report")
        print(f"Overall Score: {report['overall_score']}/100")
        for issue in report["issues"]:
            print(f"- {issue['description']}")
    """

    def __init__(self, options: CADAnnotationOptions):
        """Initialize manufacturability assessment model.

        Args:
            options: Configuration options

        Raises:
            ValueError: If VLM model not supported
        """
        super().__init__()
        self.options = options
        self.vlm = self._initialize_vlm()
        self.dfm_rules = self._initialize_dfm_rules()

        _log.info(
            f"Initialized ManufacturabilityAssessmentModel with {len(self.dfm_rules)} DFM rules"
        )

    def _initialize_vlm(self) -> VlmModel:
        """Initialize the VLM based on model name."""
        model_lower = self.options.vlm_model.lower()

        # API-based models
        if any(
            name in model_lower for name in ["gpt", "claude", "vision", "opus", "sonnet"]
        ):
            import os

            api_key = os.environ.get("OPENAI_API_KEY", "")
            if "claude" in model_lower:
                api_key = os.environ.get("ANTHROPIC_API_KEY", "")

            vlm_options = ApiVlmOptions(
                api_key=api_key,
                model_name=self.options.vlm_model,
                temperature=0.2,  # Slight creativity for suggestions
                max_tokens=4096,
                use_ocr=False,
            )
            return ApiVlmModel(vlm_options)

        # Local models
        else:
            vlm_options = InlineVlmOptions(
                model_path=self.options.vlm_model,
                device="cuda" if self.requires_gpu() else "cpu",
                temperature=0.2,
                max_tokens=2048,
                use_ocr=False,
            )
            return InlineVlmModel(vlm_options)

    def _initialize_dfm_rules(self) -> List[DFMRule]:
        """Initialize DFM rule checkers.

        Returns:
            List of DFM rule instances
        """
        return [
            ThinWallRule(min_thickness=0.5),
            DeepPocketRule(max_depth_to_width_ratio=4.0),
            SharpCornerRule(),
            SmallHoleRule(min_diameter=1.0),
        ]

    def __call__(
        self,
        doc: CADlingDocument,
        item_batch: List[CADItem],
    ) -> None:
        """Assess manufacturability of CAD items.

        Args:
            doc: The CADlingDocument being enriched
            item_batch: List of CADItem objects to process

        Note:
            Manufacturability report is added to item.properties["manufacturability_report"]
        """
        _log.info(f"Processing {len(item_batch)} items for manufacturability assessment")

        for item in item_batch:
            try:
                # Step 1: Apply geometric DFM rules
                rule_issues = self._apply_dfm_rules(doc, item)

                # Step 2: VLM-based visual assessment
                vlm_issues, vlm_insights = self._vlm_assessment(doc, item)

                # Step 3: Combine and generate report
                all_issues = rule_issues + vlm_issues
                report = self._generate_report(doc, item, all_issues, vlm_insights)

                # Add to item properties
                item.properties["manufacturability_report"] = report.model_dump()
                item.properties["manufacturability_score"] = report.overall_score
                item.properties["manufacturability_assessment_model"] = (
                    self.__class__.__name__
                )

                # Add provenance
                if hasattr(item, "add_provenance"):
                    item.add_provenance(
                        component_type="enrichment_model",
                        component_name=self.__class__.__name__,
                    )

                _log.info(
                    f"Manufacturability assessment for item {item.self_ref}: "
                    f"Score={report.overall_score:.1f}/100, "
                    f"Issues={len(all_issues)}"
                )

            except Exception as e:
                _log.error(
                    f"Manufacturability assessment failed for item {item.self_ref}: {e}"
                )
                item.properties["manufacturability_report"] = None
                item.properties["manufacturability_error"] = str(e)

    def _apply_dfm_rules(
        self, doc: CADlingDocument, item: CADItem
    ) -> List[DFMIssue]:
        """Apply geometric DFM rules to detect issues.

        Args:
            doc: The document
            item: The item to check

        Returns:
            List of detected DFM issues
        """
        all_issues = []

        for rule in self.dfm_rules:
            try:
                issues = rule.evaluate(doc, item)
                all_issues.extend(issues)
                _log.debug(
                    f"{rule.__class__.__name__} found {len(issues)} issues"
                )
            except Exception as e:
                _log.error(f"DFM rule {rule.__class__.__name__} failed: {e}")

        return all_issues

    def _vlm_assessment(
        self, doc: CADlingDocument, item: CADItem
    ) -> tuple[List[DFMIssue], Dict[str, Any]]:
        """Perform VLM-based visual manufacturability assessment.

        Args:
            doc: The document
            item: The item to assess

        Returns:
            Tuple of (issues found, assessment insights)
        """
        rendered_images = item.properties.get("rendered_images", {})
        if not rendered_images:
            return [], {}

        prompt = """
You are a manufacturing engineer reviewing a CAD part design. Analyze this part for manufacturability and identify:

1. **Hard-to-machine features**: Complex surfaces, undercuts, difficult-to-access areas
2. **Tool access issues**: Features that would be difficult to reach with standard tooling
3. **Setup complexity**: Features requiring multiple setups or special fixturing
4. **Tight tolerances**: Areas where achieving specified tolerances would be difficult
5. **Surface finish challenges**: Complex surfaces requiring special finishing operations

For each issue identified, provide:
- Issue type and severity (critical, high, medium, low, info)
- Description of the issue
- Manufacturing impact
- Recommended solution or alternative approach

Also provide:
- Recommended manufacturing processes (CNC milling, turning, EDM, casting, etc.)
- Estimated manufacturing complexity (low, medium, high, very high)
- Main cost drivers
- General DFM recommendations

Return your assessment as JSON:
{
  "issues": [{"issue_type": "...", "severity": "...", "description": "...", "recommendation": "..."}],
  "suggested_processes": ["..."],
  "complexity": "medium",
  "cost_drivers": ["..."],
  "recommendations": ["..."]
}
"""

        issues = []
        insights = {}

        try:
            # Use isometric view if available, otherwise first available view
            view_name = "isometric" if "isometric" in rendered_images else list(rendered_images.keys())[0]
            image = rendered_images[view_name]

            # Run VLM assessment
            response = self.vlm.predict(image, prompt)

            # Parse response
            import json

            json_start = response.raw_text.find("{")
            json_end = response.raw_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response.raw_text[json_start:json_end]
                parsed = json.loads(json_str)

                # Extract issues
                for issue_data in parsed.get("issues", []):
                    issues.append(
                        DFMIssue(
                            issue_type=issue_data.get("issue_type", "unknown"),
                            severity=IssueSeverity(
                                issue_data.get("severity", "medium")
                            ),
                            description=issue_data.get("description", ""),
                            recommendation=issue_data.get("recommendation", ""),
                            estimated_impact=issue_data.get("impact"),
                        )
                    )

                # Extract insights
                insights = {
                    "suggested_processes": parsed.get("suggested_processes", []),
                    "complexity": parsed.get("complexity", "medium"),
                    "cost_drivers": parsed.get("cost_drivers", []),
                    "recommendations": parsed.get("recommendations", []),
                }

            _log.debug(f"VLM assessment found {len(issues)} issues")

        except Exception as e:
            _log.error(f"VLM assessment failed: {e}")

        return issues, insights

    def _generate_report(
        self,
        doc: CADlingDocument,
        item: CADItem,
        issues: List[DFMIssue],
        vlm_insights: Dict[str, Any],
    ) -> ManufacturabilityReport:
        """Generate comprehensive manufacturability report.

        Args:
            doc: The document
            item: The item
            issues: All detected issues
            vlm_insights: Insights from VLM assessment

        Returns:
            ManufacturabilityReport
        """
        # Calculate overall score (100 - penalties for issues)
        score = 100.0
        severity_penalties = {
            IssueSeverity.CRITICAL: 20,
            IssueSeverity.HIGH: 10,
            IssueSeverity.MEDIUM: 5,
            IssueSeverity.LOW: 2,
            IssueSeverity.INFO: 0,
        }

        for issue in issues:
            score -= severity_penalties.get(issue.severity, 0)

        score = max(0.0, score)

        # Calculate complexity score
        complexity_map = {"low": 20, "medium": 50, "high": 75, "very high": 90}
        complexity_score = complexity_map.get(
            vlm_insights.get("complexity", "medium"), 50
        )

        # Determine difficulty
        if score >= 80:
            difficulty = "Easy to manufacture"
        elif score >= 60:
            difficulty = "Moderate difficulty"
        elif score >= 40:
            difficulty = "Challenging to manufacture"
        else:
            difficulty = "Very difficult to manufacture"

        # Extract suggested processes
        suggested_processes = []
        for proc_name in vlm_insights.get("suggested_processes", []):
            try:
                proc = ManufacturingProcess(proc_name.lower().replace(" ", "_"))
                suggested_processes.append(proc)
            except ValueError:
                _log.debug(f"Unknown manufacturing process: {proc_name}")

        return ManufacturabilityReport(
            overall_score=score,
            complexity_score=complexity_score,
            issues=issues,
            suggested_processes=suggested_processes,
            estimated_difficulty=difficulty,
            cost_drivers=vlm_insights.get("cost_drivers", []),
            recommendations=vlm_insights.get("recommendations", []),
        )

    def supports_batch_processing(self) -> bool:
        """Whether this model supports batch processing."""
        return False

    def requires_gpu(self) -> bool:
        """Whether this model requires GPU acceleration."""
        model_lower = self.options.vlm_model.lower()
        return not any(
            name in model_lower for name in ["gpt", "claude", "vision", "opus", "sonnet"]
        )

    def get_model_info(self) -> Dict[str, str]:
        """Get information about this model."""
        info = super().get_model_info()
        info.update(
            {
                "vlm_model": self.options.vlm_model,
                "dfm_rules": str(len(self.dfm_rules)),
            }
        )
        return info
