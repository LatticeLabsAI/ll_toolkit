"""Unit tests for CAD property prediction model."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch


class TestCADPropertyPredictor:
    """Test CADPropertyPredictor model initialization and configuration."""

    def test_model_initialization_without_artifacts(self):
        """Test model initializes without artifacts path."""
        from cadling.models.property_prediction import CADPropertyPredictor

        model = CADPropertyPredictor()

        assert model.model is None
        assert model.artifacts_path is None
        assert model.vocab_size == 50000
        assert model.output_dim == 1024
        assert model.material_density == 7850.0  # Steel default
        assert len(model.property_names) == 6
        assert "volume_mm3" in model.property_names
        assert "surface_area_mm2" in model.property_names
        assert "mass_g" in model.property_names

    def test_model_initialization_with_custom_properties(self):
        """Test model initializes with custom property names."""
        from cadling.models.property_prediction import CADPropertyPredictor

        custom_props = ["volume", "area", "perimeter"]
        model = CADPropertyPredictor(property_names=custom_props)

        assert model.property_names == custom_props
        assert len(model.property_names) == 3

    def test_model_initialization_with_custom_density(self):
        """Test model initializes with custom material density."""
        from cadling.models.property_prediction import CADPropertyPredictor

        # Aluminum density
        model = CADPropertyPredictor(material_density=2700.0)

        assert model.material_density == 2700.0

    def test_model_has_required_attributes(self):
        """Test model has all required attributes."""
        from cadling.models.property_prediction import CADPropertyPredictor

        model = CADPropertyPredictor()

        assert hasattr(model, "tokenizer")
        assert hasattr(model, "feature_extractor")
        assert hasattr(model, "topology_builder")
        assert hasattr(model, "property_names")
        assert hasattr(model, "material_density")
        assert hasattr(model, "vocab_size")
        assert hasattr(model, "output_dim")

    def test_model_has_required_methods(self):
        """Test model has all required methods."""
        from cadling.models.property_prediction import CADPropertyPredictor

        model = CADPropertyPredictor()

        assert callable(model.__call__)
        assert callable(model._predict_properties)
        assert callable(model.supports_batch_processing)
        assert callable(model.get_batch_size)
        assert callable(model.requires_gpu)


class TestCADPropertyPredictorMethods:
    """Test CADPropertyPredictor helper methods."""

    def test_supports_batch_processing(self):
        """Test that model supports batch processing."""
        from cadling.models.property_prediction import CADPropertyPredictor

        model = CADPropertyPredictor()

        assert model.supports_batch_processing() is True

    def test_get_batch_size(self):
        """Test that model returns correct batch size."""
        from cadling.models.property_prediction import CADPropertyPredictor

        model = CADPropertyPredictor()

        assert model.get_batch_size() == 16

    def test_requires_gpu(self):
        """Test that model indicates GPU benefit."""
        from cadling.models.property_prediction import CADPropertyPredictor

        model = CADPropertyPredictor()

        assert model.requires_gpu() is True


class TestCADPropertyPredictorCall:
    """Test CADPropertyPredictor __call__ method."""

    def test_call_without_model_loaded(self):
        """Test that __call__ skips when model not loaded."""
        from cadling.models.property_prediction import CADPropertyPredictor
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = CADPropertyPredictor()  # No artifacts_path

        # Create mock document and items
        doc = STEPDocument(
            name="test.step",
            origin=CADDocumentOrigin(
                filename="test.step",
                format=InputFormat.STEP,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        item = STEPEntityItem(
            label=CADItemLabel(text="Test Entity"),
            entity_id=1,
            entity_type="CYLINDRICAL_SURFACE",
            text="#1=CYLINDRICAL_SURFACE('',#2,0.5);",
        )

        # Call should return without error and without adding properties
        model(doc, [item])

        assert "predicted_volume_mm3" not in item.properties
        assert "predicted_surface_area_mm2" not in item.properties

    def test_call_with_non_step_entity(self):
        """Test that __call__ skips non-STEP entities."""
        from cadling.models.property_prediction import CADPropertyPredictor
        from cadling.datamodel.step import STEPDocument
        from cadling.datamodel.base_models import CADItem, CADItemLabel, CADDocumentOrigin, InputFormat

        model = CADPropertyPredictor()

        doc = STEPDocument(
            name="test.step",
            origin=CADDocumentOrigin(
                filename="test.step",
                format=InputFormat.STEP,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        # Create non-STEP item
        item = CADItem(
            item_type="mesh",
            label=CADItemLabel(text="Mesh Item"),
        )

        # Mock the model to be loaded
        model.model = Mock()

        # Call should skip this item
        model(doc, [item])

        # Model should not have been called
        model.model.assert_not_called()

    @patch("cadling.models.property_prediction.torch")
    def test_call_with_mocked_model(self, mock_torch):
        """Test property prediction with mocked model inference."""
        from cadling.models.property_prediction import CADPropertyPredictor
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = CADPropertyPredictor()

        # Mock the STEPForPropertyPrediction model
        mock_model = Mock()
        # Return 6 property values: volume, area, mass, bbox_x, bbox_y, bbox_z
        mock_predictions = torch.tensor([[1000.0, 500.0, 50.0, 10.0, 20.0, 30.0]])
        mock_model.return_value = mock_predictions
        model.model = mock_model

        # Mock torch operations
        mock_torch.tensor.return_value = Mock()
        mock_torch.no_grad.return_value.__enter__ = Mock()
        mock_torch.no_grad.return_value.__exit__ = Mock()

        # Create document and item
        doc = STEPDocument(
            name="test.step",
            origin=CADDocumentOrigin(
                filename="test.step",
                format=InputFormat.STEP,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        item = STEPEntityItem(
            label=CADItemLabel(text="Test Entity"),
            entity_id=1,
            entity_type="CYLINDRICAL_SURFACE",
            text="#1=CYLINDRICAL_SURFACE('',#2,0.5);",
        )

        # Run prediction
        with torch.no_grad():
            model(doc, [item])

        # Check that properties were added
        assert "predicted_volume_mm3" in item.properties
        assert item.properties["predicted_volume_mm3"] == 1000.0
        assert "predicted_surface_area_mm2" in item.properties
        assert item.properties["predicted_surface_area_mm2"] == 500.0

    def test_mass_calculation_from_volume(self):
        """Test that mass is calculated from volume and density."""
        from cadling.models.property_prediction import CADPropertyPredictor
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        # Use aluminum density
        model = CADPropertyPredictor(material_density=2700.0)

        # Mock the model
        mock_model = Mock()
        # Volume: 1,000,000 mm³ = 0.001 m³
        mock_predictions = torch.tensor([[1000000.0, 500.0, 50.0, 10.0, 20.0, 30.0]])
        mock_model.return_value = mock_predictions
        model.model = mock_model

        doc = STEPDocument(
            name="test.step",
            origin=CADDocumentOrigin(
                filename="test.step",
                format=InputFormat.STEP,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        item = STEPEntityItem(
            label=CADItemLabel(text="Test Entity"),
            entity_id=1,
            entity_type="ADVANCED_BREP_SHAPE_REPRESENTATION",
            text="#1=ADVANCED_BREP_SHAPE_REPRESENTATION('Part',(#2),#3);",
        )

        with patch("cadling.models.property_prediction.torch") as mock_torch:
            mock_torch.tensor.return_value = Mock()
            mock_torch.no_grad.return_value.__enter__ = Mock()
            mock_torch.no_grad.return_value.__exit__ = Mock()

            model(doc, [item])

        # Check mass calculation: 0.001 m³ * 2700 kg/m³ = 2.7 kg
        assert "predicted_mass_kg" in item.properties
        assert abs(item.properties["predicted_mass_kg"] - 2.7) < 0.01

    def test_provenance_tracking(self):
        """Test that provenance is added to items."""
        from cadling.models.property_prediction import CADPropertyPredictor
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = CADPropertyPredictor()

        # Mock the model
        mock_model = Mock()
        mock_model.return_value = torch.tensor([[1000.0, 500.0, 50.0, 10.0, 20.0, 30.0]])
        model.model = mock_model

        doc = STEPDocument(
            name="test.step",
            origin=CADDocumentOrigin(
                filename="test.step",
                format=InputFormat.STEP,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        item = STEPEntityItem(
            label=CADItemLabel(text="Test Entity"),
            entity_id=1,
            entity_type="PLANE",
            text="#1=PLANE('',#2);",
        )

        with patch("cadling.models.property_prediction.torch") as mock_torch:
            mock_torch.tensor.return_value = Mock()
            mock_torch.no_grad.return_value.__enter__ = Mock()
            mock_torch.no_grad.return_value.__exit__ = Mock()

            model(doc, [item])

        # Verify provenance was added
        assert len(item.prov) > 0
        assert any(
            prov.component_type == "enrichment_model" and
            prov.component_name == "CADPropertyPredictor"
            for prov in item.prov
        )


class TestCADPropertyPredictorBatchProcessing:
    """Test CADPropertyPredictor batch processing."""

    def test_batch_processing_multiple_items(self):
        """Test predicting properties for multiple items in a batch."""
        from cadling.models.property_prediction import CADPropertyPredictor
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = CADPropertyPredictor()

        # Mock the model
        mock_model = Mock()
        mock_model.return_value = torch.tensor([[1000.0, 500.0, 50.0, 10.0, 20.0, 30.0]])
        model.model = mock_model

        doc = STEPDocument(
            name="test.step",
            origin=CADDocumentOrigin(
                filename="test.step",
                format=InputFormat.STEP,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        # Create multiple items
        items = [
            STEPEntityItem(
                label=CADItemLabel(text=f"Entity {i}"),
                entity_id=i,
                entity_type="CARTESIAN_POINT",
                text=f"#{i}=CARTESIAN_POINT('',({i},{i},{i}));",
            )
            for i in range(5)
        ]

        with patch("cadling.models.property_prediction.torch") as mock_torch:
            mock_torch.tensor.return_value = Mock()
            mock_torch.no_grad.return_value.__enter__ = Mock()
            mock_torch.no_grad.return_value.__exit__ = Mock()

            model(doc, items)

        # All items should have predictions
        for item in items:
            assert "predicted_volume_mm3" in item.properties


class TestCADPropertyPredictorPredictions:
    """Test property prediction functionality."""

    def test_prediction_all_properties(self):
        """Test that all configured properties are predicted."""
        from cadling.models.property_prediction import CADPropertyPredictor
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = CADPropertyPredictor()

        # Mock the model
        mock_model = Mock()
        mock_model.return_value = torch.tensor([[1000.0, 500.0, 50.0, 10.0, 20.0, 30.0]])
        model.model = mock_model

        doc = STEPDocument(
            name="test.step",
            origin=CADDocumentOrigin(
                filename="test.step",
                format=InputFormat.STEP,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        item = STEPEntityItem(
            label=CADItemLabel(text="Test Entity"),
            entity_id=1,
            entity_type="CYLINDRICAL_SURFACE",
            text="#1=CYLINDRICAL_SURFACE('',#2,0.5);",
        )

        with patch("cadling.models.property_prediction.torch") as mock_torch:
            mock_torch.tensor.return_value = Mock()
            mock_torch.no_grad.return_value.__enter__ = Mock()
            mock_torch.no_grad.return_value.__exit__ = Mock()

            model(doc, [item])

        # Check all default properties are present
        assert "predicted_volume_mm3" in item.properties
        assert "predicted_surface_area_mm2" in item.properties
        assert "predicted_mass_g" in item.properties
        assert "predicted_bbox_x_mm" in item.properties
        assert "predicted_bbox_y_mm" in item.properties
        assert "predicted_bbox_z_mm" in item.properties
        assert "predicted_mass_kg" in item.properties  # Derived property

    def test_prediction_custom_properties(self):
        """Test prediction with custom property names."""
        from cadling.models.property_prediction import CADPropertyPredictor
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        custom_props = ["prop_a", "prop_b", "prop_c"]
        model = CADPropertyPredictor(property_names=custom_props)

        # Mock the model
        mock_model = Mock()
        mock_model.return_value = torch.tensor([[10.0, 20.0, 30.0]])
        model.model = mock_model

        doc = STEPDocument(
            name="test.step",
            origin=CADDocumentOrigin(
                filename="test.step",
                format=InputFormat.STEP,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        item = STEPEntityItem(
            label=CADItemLabel(text="Test Entity"),
            entity_id=1,
            entity_type="PLANE",
            text="#1=PLANE('',#2);",
        )

        with patch("cadling.models.property_prediction.torch") as mock_torch:
            mock_torch.tensor.return_value = Mock()
            mock_torch.no_grad.return_value.__enter__ = Mock()
            mock_torch.no_grad.return_value.__exit__ = Mock()

            model(doc, [item])

        # Check custom properties
        assert "predicted_prop_a" in item.properties
        assert item.properties["predicted_prop_a"] == 10.0
        assert "predicted_prop_b" in item.properties
        assert item.properties["predicted_prop_b"] == 20.0
        assert "predicted_prop_c" in item.properties
        assert item.properties["predicted_prop_c"] == 30.0


class TestIntegration:
    """Integration tests for CADPropertyPredictor."""

    def test_predictor_with_different_materials(self):
        """Test predictor with various material densities."""
        from cadling.models.property_prediction import CADPropertyPredictor

        # Test common materials
        steel = CADPropertyPredictor(material_density=7850.0)
        aluminum = CADPropertyPredictor(material_density=2700.0)
        titanium = CADPropertyPredictor(material_density=4500.0)
        plastic = CADPropertyPredictor(material_density=1200.0)

        assert steel.material_density == 7850.0
        assert aluminum.material_density == 2700.0
        assert titanium.material_density == 4500.0
        assert plastic.material_density == 1200.0

    def test_predictor_with_large_entity(self):
        """Test predictor handles large entity text correctly."""
        from cadling.models.property_prediction import CADPropertyPredictor
        from cadling.datamodel.step import STEPDocument, STEPEntityItem
        from cadling.datamodel.base_models import CADItemLabel, CADDocumentOrigin, InputFormat

        model = CADPropertyPredictor()

        # Create very long entity text
        long_text = "#1=ADVANCED_BREP_SHAPE_REPRESENTATION('Part',(" + ",".join([f"#{i}" for i in range(2, 1000)]) + "),#1000);"

        # Mock model
        mock_model = Mock()
        mock_model.return_value = torch.tensor([[1000.0, 500.0, 50.0, 10.0, 20.0, 30.0]])
        model.model = mock_model

        doc = STEPDocument(
            name="test.step",
            origin=CADDocumentOrigin(
                filename="test.step",
                format=InputFormat.STEP,
                binary_hash="test_hash",
            ),
            hash="test_hash",
        )

        item = STEPEntityItem(
            label=CADItemLabel(text="Large Entity"),
            entity_id=1,
            entity_type="ADVANCED_BREP_SHAPE_REPRESENTATION",
            text=long_text,
        )

        with patch("cadling.models.property_prediction.torch") as mock_torch:
            mock_torch.tensor.return_value = Mock()
            mock_torch.no_grad.return_value.__enter__ = Mock()
            mock_torch.no_grad.return_value.__exit__ = Mock()

            # Should handle long text without error
            model(doc, [item])

        assert "predicted_volume_mm3" in item.properties
