"""Comprehensive test suite for ll_gen.config module.

Tests all configuration dataclasses, enums, and the get_ll_gen_config helper:
- Enum values and string representations
- Dataclass defaults and custom initialization
- Nested configuration with dotted key overrides
- Type validation and field factories
"""
from __future__ import annotations

from dataclasses import fields

import pytest

from ll_gen.config import (
    CodegenConfig,
    CodeLanguage,
    DatasetConfig,
    DisposalConfig,
    ErrorCategory,
    ErrorSeverity,
    ExportConfig,
    FeedbackConfig,
    GenerationRoute,
    LLGenConfig,
    RoutingConfig,
    StepSchema,
    get_ll_gen_config,
)


# ============================================================================
# SECTION 1: Enum Tests
# ============================================================================


class TestGenerationRouteEnum:
    """Test GenerationRoute enum values and properties."""

    def test_has_all_code_routes(self) -> None:
        """Test that all code generation routes are defined."""
        assert GenerationRoute.CODE_CADQUERY == "code_cadquery"
        assert GenerationRoute.CODE_OPENSCAD == "code_openscad"
        assert GenerationRoute.CODE_PYTHONOCC == "code_pythonocc"

    def test_has_all_neural_routes(self) -> None:
        """Test that all neural generation routes are defined."""
        assert GenerationRoute.NEURAL_VAE == "neural_vae"
        assert GenerationRoute.NEURAL_DIFFUSION == "neural_diffusion"
        assert GenerationRoute.NEURAL_VQVAE == "neural_vqvae"

    def test_route_count(self) -> None:
        """Test that there are exactly 6 routes."""
        assert len(GenerationRoute) == 6

    def test_route_is_str_subclass(self) -> None:
        """Test that GenerationRoute values are strings."""
        for route in GenerationRoute:
            assert isinstance(route, str)
            assert isinstance(route.value, str)

    def test_route_from_string(self) -> None:
        """Test that routes can be created from string values."""
        route = GenerationRoute("code_cadquery")
        assert route == GenerationRoute.CODE_CADQUERY


class TestCodeLanguageEnum:
    """Test CodeLanguage enum values."""

    def test_all_languages_defined(self) -> None:
        """Test that all code languages are defined."""
        assert CodeLanguage.CADQUERY == "cadquery"
        assert CodeLanguage.OPENSCAD == "openscad"
        assert CodeLanguage.PYTHONOCC == "pythonocc"

    def test_language_count(self) -> None:
        """Test that there are exactly 3 languages."""
        assert len(CodeLanguage) == 3

    def test_language_is_str_subclass(self) -> None:
        """Test that CodeLanguage values are strings."""
        for lang in CodeLanguage:
            assert isinstance(lang, str)


class TestErrorCategoryEnum:
    """Test ErrorCategory enum values."""

    def test_all_categories_defined(self) -> None:
        """Test that all 6 error categories are defined."""
        expected_categories = [
            "invalid_params",
            "topology_error",
            "boolean_failure",
            "self_intersection",
            "degenerate_shape",
            "tolerance_violation",
        ]
        actual_values = [e.value for e in ErrorCategory]
        assert sorted(actual_values) == sorted(expected_categories)

    def test_category_count(self) -> None:
        """Test that there are exactly 6 categories."""
        assert len(ErrorCategory) == 6

    def test_category_is_str_subclass(self) -> None:
        """Test that ErrorCategory values are strings."""
        for cat in ErrorCategory:
            assert isinstance(cat, str)


class TestStepSchemaEnum:
    """Test StepSchema enum values."""

    def test_all_schemas_defined(self) -> None:
        """Test that all STEP schemas are defined."""
        assert StepSchema.AP203 == "AP203"
        assert StepSchema.AP214 == "AP214"
        assert StepSchema.AP242 == "AP242"

    def test_schema_count(self) -> None:
        """Test that there are exactly 3 schemas."""
        assert len(StepSchema) == 3


class TestErrorSeverityEnum:
    """Test ErrorSeverity enum values."""

    def test_all_severities_defined(self) -> None:
        """Test that all severity levels are defined."""
        assert ErrorSeverity.CRITICAL == "critical"
        assert ErrorSeverity.WARNING == "warning"
        assert ErrorSeverity.INFO == "info"

    def test_severity_count(self) -> None:
        """Test that there are exactly 3 severity levels."""
        assert len(ErrorSeverity) == 3

    def test_severity_ordering(self) -> None:
        """Test that severity names are lowercase."""
        for sev in ErrorSeverity:
            assert sev.value == sev.value.lower()


# ============================================================================
# SECTION 2: Dataclass Default Tests
# ============================================================================


class TestRoutingConfigDefaults:
    """Test RoutingConfig default values."""

    def test_default_initialization(self) -> None:
        """Test RoutingConfig initializes with all defaults."""
        config = RoutingConfig()
        assert isinstance(config.mechanical_keywords, list)
        assert isinstance(config.openscad_keywords, list)
        assert isinstance(config.freeform_keywords, list)
        assert isinstance(config.exploration_keywords, list)
        assert isinstance(config.codebook_keywords, list)
        assert config.confidence_threshold == 0.3
        assert config.default_route == GenerationRoute.CODE_CADQUERY

    def test_mechanical_keywords_include_common_terms(self) -> None:
        """Test that mechanical keywords include common CAD terms."""
        config = RoutingConfig()
        common_terms = ["extrude", "fillet", "chamfer", "hole", "sweep"]
        for term in common_terms:
            assert term in config.mechanical_keywords

    def test_openscad_keywords_include_csg_terms(self) -> None:
        """Test that OpenSCAD keywords include CSG terms."""
        config = RoutingConfig()
        csg_terms = ["union", "difference", "intersection", "hull"]
        for term in csg_terms:
            assert term in config.openscad_keywords

    def test_freeform_keywords_include_organic_terms(self) -> None:
        """Test that freeform keywords include organic shape terms."""
        config = RoutingConfig()
        organic_terms = ["smooth", "flowing", "organic", "curved"]
        for term in organic_terms:
            assert term in config.freeform_keywords

    def test_custom_confidence_threshold(self) -> None:
        """Test RoutingConfig with custom confidence threshold."""
        config = RoutingConfig(confidence_threshold=0.5)
        assert config.confidence_threshold == 0.5

    def test_custom_default_route(self) -> None:
        """Test RoutingConfig with custom default route."""
        config = RoutingConfig(default_route=GenerationRoute.NEURAL_VAE)
        assert config.default_route == GenerationRoute.NEURAL_VAE

    def test_custom_keyword_lists(self) -> None:
        """Test RoutingConfig with custom keyword lists."""
        config = RoutingConfig(
            mechanical_keywords=["custom", "terms"],
            freeform_keywords=["organic"],
        )
        assert config.mechanical_keywords == ["custom", "terms"]
        assert config.freeform_keywords == ["organic"]


class TestCodegenConfigDefaults:
    """Test CodegenConfig default values."""

    def test_default_initialization(self) -> None:
        """Test CodegenConfig initializes with all defaults."""
        config = CodegenConfig()
        assert config.model_name == "claude-sonnet-4-20250514"
        assert config.api_provider == "anthropic"
        assert config.max_tokens == 4096
        assert config.temperature == 0.2
        assert config.execution_timeout == 30
        assert config.max_retries == 3
        assert config.default_backend == CodeLanguage.CADQUERY
        assert config.include_examples is True
        assert config.max_example_tokens == 2000

    def test_allowed_modules_default(self) -> None:
        """Test that allowed_modules includes safe defaults."""
        config = CodegenConfig()
        assert "cadquery" in config.allowed_modules
        assert "math" in config.allowed_modules
        assert "numpy" in config.allowed_modules

    def test_custom_model_settings(self) -> None:
        """Test CodegenConfig with custom model settings."""
        config = CodegenConfig(
            model_name="gpt-4",
            api_provider="openai",
            temperature=0.7,
            max_tokens=8192,
        )
        assert config.model_name == "gpt-4"
        assert config.api_provider == "openai"
        assert config.temperature == 0.7
        assert config.max_tokens == 8192

    def test_custom_execution_settings(self) -> None:
        """Test CodegenConfig with custom execution settings."""
        config = CodegenConfig(
            execution_timeout=60,
            max_retries=5,
        )
        assert config.execution_timeout == 60
        assert config.max_retries == 5

    def test_custom_backend(self) -> None:
        """Test CodegenConfig with custom default backend."""
        config = CodegenConfig(default_backend=CodeLanguage.OPENSCAD)
        assert config.default_backend == CodeLanguage.OPENSCAD


class TestDisposalConfigDefaults:
    """Test DisposalConfig default values."""

    def test_default_initialization(self) -> None:
        """Test DisposalConfig initializes with all defaults."""
        config = DisposalConfig()
        assert config.tolerance == 1e-7
        assert config.angular_tolerance == 1e-5
        assert config.enable_auto_repair is True
        assert config.max_repair_passes == 3
        assert config.check_manifoldness is True
        assert config.check_euler is True
        assert config.check_watertightness is True
        assert config.check_self_intersection is True
        assert config.always_introspect is True

    def test_shapefix_defaults(self) -> None:
        """Test ShapeFix configuration defaults."""
        config = DisposalConfig()
        assert config.shapefix_precision == 1e-7
        assert config.shapefix_max_tolerance == 1e-3
        assert config.shapefix_min_tolerance == 1e-7

    def test_fuzzy_tolerance_steps_default(self) -> None:
        """Test fuzzy tolerance escalation steps."""
        config = DisposalConfig()
        expected = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
        assert config.fuzzy_tolerance_steps == expected

    def test_custom_tolerance_settings(self) -> None:
        """Test DisposalConfig with custom tolerance settings."""
        config = DisposalConfig(
            tolerance=1e-6,
            angular_tolerance=1e-4,
        )
        assert config.tolerance == 1e-6
        assert config.angular_tolerance == 1e-4

    def test_custom_repair_settings(self) -> None:
        """Test DisposalConfig with custom repair settings."""
        config = DisposalConfig(
            enable_auto_repair=False,
            max_repair_passes=5,
        )
        assert config.enable_auto_repair is False
        assert config.max_repair_passes == 5

    def test_disable_validation_checks(self) -> None:
        """Test DisposalConfig with validation checks disabled."""
        config = DisposalConfig(
            check_manifoldness=False,
            check_euler=False,
            check_watertightness=False,
            check_self_intersection=False,
        )
        assert config.check_manifoldness is False
        assert config.check_euler is False
        assert config.check_watertightness is False
        assert config.check_self_intersection is False


class TestExportConfigDefaults:
    """Test ExportConfig default values."""

    def test_default_initialization(self) -> None:
        """Test ExportConfig initializes with all defaults."""
        config = ExportConfig()
        assert config.step_schema == StepSchema.AP214
        assert config.stl_linear_deflection == 0.1
        assert config.stl_angular_deflection == 0.5
        assert config.stl_ascii is False
        assert config.render_resolution == 512

    def test_render_views_default(self) -> None:
        """Test default render views include standard views."""
        config = ExportConfig()
        expected_views = ["front", "top", "right", "isometric"]
        assert config.render_views == expected_views

    def test_custom_step_schema(self) -> None:
        """Test ExportConfig with custom STEP schema."""
        config = ExportConfig(step_schema=StepSchema.AP242)
        assert config.step_schema == StepSchema.AP242

    def test_custom_stl_settings(self) -> None:
        """Test ExportConfig with custom STL settings."""
        config = ExportConfig(
            stl_linear_deflection=0.01,
            stl_angular_deflection=0.1,
            stl_ascii=True,
        )
        assert config.stl_linear_deflection == 0.01
        assert config.stl_angular_deflection == 0.1
        assert config.stl_ascii is True

    def test_custom_render_settings(self) -> None:
        """Test ExportConfig with custom render settings."""
        config = ExportConfig(
            render_views=["front", "back"],
            render_resolution=1024,
        )
        assert config.render_views == ["front", "back"]
        assert config.render_resolution == 1024


class TestFeedbackConfigDefaults:
    """Test FeedbackConfig default values."""

    def test_default_initialization(self) -> None:
        """Test FeedbackConfig initializes with all defaults."""
        config = FeedbackConfig()
        assert config.validity_reward == 1.0
        assert config.shape_constructed_reward == 0.3
        assert config.repairable_reward == 0.2
        assert config.per_tier_reward == 0.1
        assert config.semantic_match_reward == 0.2
        assert config.critical_error_penalty == -0.1
        assert config.dimension_tolerance_pct == 0.10

    def test_custom_reward_weights(self) -> None:
        """Test FeedbackConfig with custom reward weights."""
        config = FeedbackConfig(
            validity_reward=2.0,
            shape_constructed_reward=0.5,
            critical_error_penalty=-0.5,
        )
        assert config.validity_reward == 2.0
        assert config.shape_constructed_reward == 0.5
        assert config.critical_error_penalty == -0.5

    def test_custom_dimension_tolerance(self) -> None:
        """Test FeedbackConfig with custom dimension tolerance."""
        config = FeedbackConfig(dimension_tolerance_pct=0.05)
        assert config.dimension_tolerance_pct == 0.05


class TestDatasetConfigDefaults:
    """Test DatasetConfig default values."""

    def test_default_initialization(self) -> None:
        """Test DatasetConfig initializes with all defaults."""
        config = DatasetConfig()
        assert config.deepcad_path == "latticelabs/deepcad"
        assert config.abc_path == "latticelabs/abc"
        assert config.text2cad_path == "latticelabs/text2cad"
        assert config.sketchgraphs_path == "latticelabs/sketchgraphs"
        assert config.streaming is True
        assert config.shuffle is True
        assert config.shuffle_buffer_size == 10000
        assert config.max_samples is None

    def test_tokenization_defaults(self) -> None:
        """Test tokenization setting defaults."""
        config = DatasetConfig()
        assert config.max_commands == 60
        assert config.quantization_bits == 8
        assert config.normalization_range == 2.0

    def test_split_defaults(self) -> None:
        """Test dataset split defaults."""
        config = DatasetConfig()
        assert config.train_split == "train"
        assert config.val_split == "validation"
        assert config.test_split == "test"

    def test_custom_dataset_paths(self) -> None:
        """Test DatasetConfig with custom dataset paths."""
        config = DatasetConfig(
            deepcad_path="/local/deepcad",
            abc_path="/local/abc",
        )
        assert config.deepcad_path == "/local/deepcad"
        assert config.abc_path == "/local/abc"

    def test_custom_loading_settings(self) -> None:
        """Test DatasetConfig with custom loading settings."""
        config = DatasetConfig(
            streaming=False,
            shuffle=False,
            max_samples=1000,
        )
        assert config.streaming is False
        assert config.shuffle is False
        assert config.max_samples == 1000


# ============================================================================
# SECTION 3: LLGenConfig Tests
# ============================================================================


class TestLLGenConfigDefaults:
    """Test LLGenConfig top-level configuration."""

    def test_default_initialization(self) -> None:
        """Test LLGenConfig initializes with all default sub-configs."""
        config = LLGenConfig()
        assert isinstance(config.routing, RoutingConfig)
        assert isinstance(config.codegen, CodegenConfig)
        assert isinstance(config.disposal, DisposalConfig)
        assert isinstance(config.export, ExportConfig)
        assert isinstance(config.feedback, FeedbackConfig)
        assert isinstance(config.datasets, DatasetConfig)

    def test_global_settings_defaults(self) -> None:
        """Test LLGenConfig global settings have correct defaults."""
        config = LLGenConfig()
        assert config.max_retries == 3
        assert config.output_dir == "output"
        assert config.log_level == "INFO"
        assert config.device == "cpu"

    def test_custom_global_settings(self) -> None:
        """Test LLGenConfig with custom global settings."""
        config = LLGenConfig(
            max_retries=5,
            output_dir="/custom/output",
            log_level="DEBUG",
            device="cuda",
        )
        assert config.max_retries == 5
        assert config.output_dir == "/custom/output"
        assert config.log_level == "DEBUG"
        assert config.device == "cuda"

    def test_custom_sub_configs(self) -> None:
        """Test LLGenConfig with custom sub-configurations."""
        routing = RoutingConfig(confidence_threshold=0.5)
        codegen = CodegenConfig(temperature=0.5)

        config = LLGenConfig(routing=routing, codegen=codegen)

        assert config.routing.confidence_threshold == 0.5
        assert config.codegen.temperature == 0.5

    def test_sub_configs_are_independent_instances(self) -> None:
        """Test that each LLGenConfig instance has independent sub-configs."""
        config1 = LLGenConfig()
        config2 = LLGenConfig()

        config1.codegen.temperature = 0.9
        assert config2.codegen.temperature == 0.2  # Original default


# ============================================================================
# SECTION 4: get_ll_gen_config Tests
# ============================================================================


class TestGetLLGenConfigHelper:
    """Test the get_ll_gen_config helper function."""

    def test_no_overrides(self) -> None:
        """Test get_ll_gen_config with no overrides returns defaults."""
        config = get_ll_gen_config()
        assert isinstance(config, LLGenConfig)
        assert config.max_retries == 3
        assert config.codegen.temperature == 0.2

    def test_top_level_override(self) -> None:
        """Test get_ll_gen_config with top-level key override."""
        config = get_ll_gen_config(max_retries=5, output_dir="/test/output")
        assert config.max_retries == 5
        assert config.output_dir == "/test/output"

    def test_dotted_key_override(self) -> None:
        """Test get_ll_gen_config with dotted key override."""
        config = get_ll_gen_config(**{"codegen.temperature": 0.7})
        assert config.codegen.temperature == 0.7

    def test_multiple_dotted_overrides(self) -> None:
        """Test get_ll_gen_config with multiple dotted overrides."""
        config = get_ll_gen_config(
            **{
                "codegen.temperature": 0.5,
                "codegen.max_tokens": 8192,
                "disposal.tolerance": 1e-6,
                "feedback.validity_reward": 2.0,
            }
        )
        assert config.codegen.temperature == 0.5
        assert config.codegen.max_tokens == 8192
        assert config.disposal.tolerance == 1e-6
        assert config.feedback.validity_reward == 2.0

    def test_mixed_top_level_and_dotted_overrides(self) -> None:
        """Test get_ll_gen_config with both top-level and dotted overrides."""
        config = get_ll_gen_config(
            max_retries=10,
            device="cuda",
            **{
                "codegen.temperature": 0.3,
                "export.render_resolution": 1024,
            }
        )
        assert config.max_retries == 10
        assert config.device == "cuda"
        assert config.codegen.temperature == 0.3
        assert config.export.render_resolution == 1024

    def test_invalid_top_level_key_ignored(self) -> None:
        """Test that invalid top-level keys are silently ignored."""
        config = get_ll_gen_config(nonexistent_key="value")
        # Should not raise, just ignore
        assert not hasattr(config, "nonexistent_key")

    def test_invalid_nested_key_ignored(self) -> None:
        """Test that invalid nested keys are silently ignored."""
        config = get_ll_gen_config(**{"codegen.nonexistent": "value"})
        # Should not raise, just ignore
        assert not hasattr(config.codegen, "nonexistent")

    def test_invalid_section_ignored(self) -> None:
        """Test that invalid section names in dotted keys are ignored."""
        config = get_ll_gen_config(**{"nonexistent.temperature": 0.5})
        # Should not raise, just ignore
        assert isinstance(config, LLGenConfig)

    def test_deeply_nested_key_handled(self) -> None:
        """Test that keys with multiple dots use first dot as split."""
        # "a.b.c" splits as section="a", attr="b.c"
        # Since "b.c" is not a valid attr, it should be ignored
        config = get_ll_gen_config(**{"codegen.a.b.c": "value"})
        # Should not raise
        assert isinstance(config, LLGenConfig)


class TestGetLLGenConfigWithEnums:
    """Test get_ll_gen_config with enum value overrides."""

    def test_override_step_schema(self) -> None:
        """Test overriding export.step_schema with enum value."""
        config = get_ll_gen_config(**{"export.step_schema": StepSchema.AP242})
        assert config.export.step_schema == StepSchema.AP242

    def test_override_default_route(self) -> None:
        """Test overriding routing.default_route with enum value."""
        config = get_ll_gen_config(
            **{"routing.default_route": GenerationRoute.NEURAL_VAE}
        )
        assert config.routing.default_route == GenerationRoute.NEURAL_VAE

    def test_override_default_backend(self) -> None:
        """Test overriding codegen.default_backend with enum value."""
        config = get_ll_gen_config(
            **{"codegen.default_backend": CodeLanguage.OPENSCAD}
        )
        assert config.codegen.default_backend == CodeLanguage.OPENSCAD


# ============================================================================
# SECTION 5: Dataclass Structure Tests
# ============================================================================


class TestDataclassStructure:
    """Test dataclass field structure and types."""

    def test_routing_config_field_count(self) -> None:
        """Test RoutingConfig has expected number of fields."""
        config_fields = fields(RoutingConfig)
        assert len(config_fields) == 7

    def test_codegen_config_field_count(self) -> None:
        """Test CodegenConfig has expected number of fields."""
        config_fields = fields(CodegenConfig)
        assert len(config_fields) == 10

    def test_disposal_config_field_count(self) -> None:
        """Test DisposalConfig has expected number of fields."""
        config_fields = fields(DisposalConfig)
        assert len(config_fields) == 13

    def test_export_config_field_count(self) -> None:
        """Test ExportConfig has expected number of fields."""
        config_fields = fields(ExportConfig)
        assert len(config_fields) == 6

    def test_feedback_config_field_count(self) -> None:
        """Test FeedbackConfig has expected number of fields."""
        config_fields = fields(FeedbackConfig)
        assert len(config_fields) == 7

    def test_dataset_config_field_count(self) -> None:
        """Test DatasetConfig has expected number of fields."""
        config_fields = fields(DatasetConfig)
        assert len(config_fields) == 14

    def test_ll_gen_config_field_count(self) -> None:
        """Test LLGenConfig has expected number of fields."""
        config_fields = fields(LLGenConfig)
        assert len(config_fields) == 13


class TestConfigImmutability:
    """Test that config modifications don't affect defaults."""

    def test_list_field_independence(self) -> None:
        """Test that list fields are independent between instances."""
        config1 = RoutingConfig()
        config2 = RoutingConfig()

        config1.mechanical_keywords.append("custom_term")

        assert "custom_term" in config1.mechanical_keywords
        assert "custom_term" not in config2.mechanical_keywords

    def test_sub_config_independence(self) -> None:
        """Test that sub-configs are independent between instances."""
        config1 = LLGenConfig()
        config2 = LLGenConfig()

        config1.codegen.temperature = 0.9

        assert config1.codegen.temperature == 0.9
        assert config2.codegen.temperature == 0.2
