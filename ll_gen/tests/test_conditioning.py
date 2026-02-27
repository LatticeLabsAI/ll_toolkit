"""Comprehensive test suite for ll_gen conditioning module.

Tests cover all 5 classes in the conditioning package:
1. ConditioningEmbeddings - Dataclass with validation and tensor conversion
2. TextConditioningEncoder - Text prompt encoding with fallback
3. ImageConditioningEncoder - Image encoding with fallback
4. MultiModalConditioner - Fusion of text and image embeddings
5. ConstraintPredictor - Extraction of geometric constraints from prompts/embeddings

All tests work without optional dependencies (torch, ll_stepnet, PIL).
Fallback modes are comprehensively tested.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ll_gen.conditioning.constraint_predictor import (
    ConstraintPrediction,
    ConstraintPredictor,
    ConstraintType,
)
from ll_gen.conditioning.embeddings import ConditioningEmbeddings
from ll_gen.conditioning.image_encoder import ImageConditioningEncoder
from ll_gen.conditioning.multimodal import MultiModalConditioner
from ll_gen.conditioning.text_encoder import TextConditioningEncoder

# =============================================================================
# ConditioningEmbeddings Tests
# =============================================================================


class TestConditioningEmbeddings:
    """Test ConditioningEmbeddings dataclass with validation and conversion."""

    @pytest.mark.unit
    def test_init_default_values(self):
        """Test ConditioningEmbeddings initialization with defaults."""
        emb = ConditioningEmbeddings()

        assert emb.token_embeddings is None
        assert emb.pooled_embedding is None
        assert emb.source_type == "text"
        assert emb.source_model == "unknown"
        assert emb.embed_dim == 768
        assert emb.metadata == {}

    @pytest.mark.unit
    def test_init_with_embeddings(self):
        """Test ConditioningEmbeddings with actual embeddings."""
        token_emb = np.random.randn(10, 768).astype(np.float32)
        pooled = np.random.randn(768).astype(np.float32)

        emb = ConditioningEmbeddings(
            token_embeddings=token_emb,
            pooled_embedding=pooled,
            source_type="text",
            source_model="bert-base-uncased",
            embed_dim=768,
            metadata={"prompt": "test"},
        )

        assert emb.token_embeddings.shape == (10, 768)
        assert emb.pooled_embedding.shape == (768,)
        assert emb.source_type == "text"
        assert emb.source_model == "bert-base-uncased"
        assert emb.metadata["prompt"] == "test"

    @pytest.mark.unit
    def test_validate_valid_pooled_only(self):
        """Test validate() passes with valid pooled embedding only."""
        pooled = np.random.randn(768).astype(np.float32)
        emb = ConditioningEmbeddings(
            pooled_embedding=pooled,
            embed_dim=768,
        )
        assert emb.validate() is True

    @pytest.mark.unit
    def test_validate_valid_token_only(self):
        """Test validate() passes with valid token embeddings only."""
        tokens = np.random.randn(20, 768).astype(np.float32)
        emb = ConditioningEmbeddings(
            token_embeddings=tokens,
            embed_dim=768,
        )
        assert emb.validate() is True

    @pytest.mark.unit
    def test_validate_valid_both(self):
        """Test validate() passes with both valid embeddings."""
        tokens = np.random.randn(15, 768).astype(np.float32)
        pooled = np.random.randn(768).astype(np.float32)
        emb = ConditioningEmbeddings(
            token_embeddings=tokens,
            pooled_embedding=pooled,
            embed_dim=768,
        )
        assert emb.validate() is True

    @pytest.mark.unit
    def test_validate_fails_both_none(self):
        """Test validate() fails when both embeddings are None."""
        emb = ConditioningEmbeddings(
            token_embeddings=None,
            pooled_embedding=None,
        )
        assert emb.validate() is False

    @pytest.mark.unit
    def test_validate_fails_pooled_wrong_dim(self):
        """Test validate() fails when pooled_embedding has wrong ndim."""
        pooled = np.random.randn(768, 1).astype(np.float32)
        emb = ConditioningEmbeddings(
            pooled_embedding=pooled,
            embed_dim=768,
        )
        assert emb.validate() is False

    @pytest.mark.unit
    def test_validate_fails_pooled_wrong_shape(self):
        """Test validate() fails when pooled_embedding shape != embed_dim."""
        pooled = np.random.randn(512).astype(np.float32)
        emb = ConditioningEmbeddings(
            pooled_embedding=pooled,
            embed_dim=768,
        )
        assert emb.validate() is False

    @pytest.mark.unit
    def test_validate_fails_token_wrong_dim(self):
        """Test validate() fails when token_embeddings has wrong ndim."""
        tokens = np.random.randn(10).astype(np.float32)
        emb = ConditioningEmbeddings(
            token_embeddings=tokens,
            embed_dim=768,
        )
        assert emb.validate() is False

    @pytest.mark.unit
    def test_validate_fails_token_wrong_shape(self):
        """Test validate() fails when token_embeddings shape[1] != embed_dim."""
        tokens = np.random.randn(10, 512).astype(np.float32)
        emb = ConditioningEmbeddings(
            token_embeddings=tokens,
            embed_dim=768,
        )
        assert emb.validate() is False

    @pytest.mark.unit
    def test_summary(self):
        """Test summary() returns correct dictionary."""
        tokens = np.random.randn(10, 768).astype(np.float32)
        pooled = np.random.randn(768).astype(np.float32)
        emb = ConditioningEmbeddings(
            token_embeddings=tokens,
            pooled_embedding=pooled,
            source_type="image",
            source_model="dino_vits16",
            embed_dim=768,
            metadata={"image_size": 224},
        )

        summary = emb.summary()

        assert summary["source_type"] == "image"
        assert summary["source_model"] == "dino_vits16"
        assert summary["embed_dim"] == 768
        assert summary["has_pooled_embedding"] is True
        assert summary["has_token_embeddings"] is True
        assert summary["token_seq_len"] == 10
        assert summary["metadata"] == {"image_size": 224}

    @pytest.mark.unit
    def test_summary_partial(self):
        """Test summary() with only pooled embedding."""
        pooled = np.random.randn(768).astype(np.float32)
        emb = ConditioningEmbeddings(
            pooled_embedding=pooled,
            source_type="text",
        )

        summary = emb.summary()

        assert summary["has_pooled_embedding"] is True
        assert summary["has_token_embeddings"] is False
        assert summary["token_seq_len"] is None

    @pytest.mark.unit
    def test_to_tensor_with_pooled(self):
        """Test to_tensor() with valid pooled embedding."""
        pooled = np.ones(768, dtype=np.float32)
        emb = ConditioningEmbeddings(pooled_embedding=pooled)

        tensor = emb.to_tensor("cpu")

        # Should return None if torch is not available, or a tensor if available
        if tensor is not None:
            assert tensor.shape == (768,)
            assert "float" in str(tensor.dtype)

    @pytest.mark.unit
    def test_to_tensor_no_pooled(self):
        """Test to_tensor() returns None when no pooled embedding."""
        emb = ConditioningEmbeddings(pooled_embedding=None)
        tensor = emb.to_tensor("cpu")
        assert tensor is None

    @pytest.mark.unit
    def test_to_token_tensor_with_tokens(self):
        """Test to_token_tensor() with valid token embeddings."""
        tokens = np.ones((10, 768), dtype=np.float32)
        emb = ConditioningEmbeddings(token_embeddings=tokens)

        tensor = emb.to_token_tensor("cpu")

        if tensor is not None:
            assert tensor.shape == (10, 768)
            assert "float" in str(tensor.dtype)

    @pytest.mark.unit
    def test_to_token_tensor_no_tokens(self):
        """Test to_token_tensor() returns None when no token embeddings."""
        emb = ConditioningEmbeddings(token_embeddings=None)
        tensor = emb.to_token_tensor("cpu")
        assert tensor is None

    @pytest.mark.unit
    def test_from_tensor_with_torch(self):
        """Test from_tensor() classmethod (fallback when torch unavailable)."""
        pooled_np = np.random.randn(768).astype(np.float32)

        emb = ConditioningEmbeddings.from_tensor(
            tensor=pooled_np,
            source_type="text",
            source_model="bert-base-uncased",
            metadata={"prompt": "test"},
        )

        assert emb.source_type == "text"
        assert emb.source_model == "bert-base-uncased"
        assert emb.metadata["prompt"] == "test"
        if emb.pooled_embedding is not None:
            assert emb.pooled_embedding.shape == (768,)

    @pytest.mark.unit
    def test_from_tensor_with_token_tensor(self):
        """Test from_tensor() with both pooled and token tensors."""
        pooled_np = np.random.randn(768).astype(np.float32)
        token_np = np.random.randn(10, 768).astype(np.float32)

        emb = ConditioningEmbeddings.from_tensor(
            tensor=pooled_np,
            source_type="multimodal",
            source_model="hybrid",
            token_tensor=token_np,
        )

        assert emb.source_type == "multimodal"
        if emb.pooled_embedding is not None:
            assert emb.pooled_embedding.shape == (768,)
        if emb.token_embeddings is not None:
            assert emb.token_embeddings.shape == (10, 768)


# =============================================================================
# TextConditioningEncoder Tests
# =============================================================================


class TestTextConditioningEncoder:
    """Test TextConditioningEncoder with hash fallback."""

    @pytest.mark.unit
    def test_init_default(self):
        """Test TextConditioningEncoder initialization with defaults."""
        encoder = TextConditioningEncoder()

        assert encoder.model_name == "bert-base-uncased"
        assert encoder.conditioning_dim == 768
        assert encoder.freeze_encoder is True
        assert encoder.device == "cpu"

    @pytest.mark.unit
    def test_init_custom(self):
        """Test TextConditioningEncoder initialization with custom values."""
        encoder = TextConditioningEncoder(
            model_name="roberta-base",
            conditioning_dim=512,
            freeze_encoder=False,
            device="cuda:0",
        )

        assert encoder.model_name == "roberta-base"
        assert encoder.conditioning_dim == 512
        assert encoder.freeze_encoder is False
        assert encoder.device == "cuda:0"

    @pytest.mark.unit
    def test_encode_fallback_simple(self):
        """Test encode() uses hash fallback without ll_stepnet."""
        encoder = TextConditioningEncoder(conditioning_dim=768)
        emb = encoder.encode("Hello world test")

        assert emb.source_type == "text"
        assert emb.source_model == "hash_fallback"
        assert emb.embed_dim == 768
        assert emb.token_embeddings is not None
        assert emb.pooled_embedding is not None
        assert emb.validate() is True

    @pytest.mark.unit
    def test_encode_fallback_deterministic(self):
        """Test encode() fallback produces same embedding for same prompt."""
        encoder = TextConditioningEncoder(conditioning_dim=768)

        emb1 = encoder.encode("test prompt")
        emb2 = encoder.encode("test prompt")

        # Same prompt should produce same embeddings
        np.testing.assert_array_almost_equal(
            emb1.pooled_embedding, emb2.pooled_embedding
        )
        np.testing.assert_array_almost_equal(
            emb1.token_embeddings, emb2.token_embeddings
        )

    @pytest.mark.unit
    def test_encode_fallback_different_prompts(self):
        """Test encode() fallback produces different embeddings for different prompts."""
        encoder = TextConditioningEncoder(conditioning_dim=768)

        emb1 = encoder.encode("prompt one")
        emb2 = encoder.encode("prompt two")

        # Different prompts should produce different embeddings
        assert not np.allclose(
            emb1.pooled_embedding, emb2.pooled_embedding
        )

    @pytest.mark.unit
    def test_encode_fallback_seq_len_bounds(self):
        """Test encode() fallback respects sequence length bounds."""
        encoder = TextConditioningEncoder(conditioning_dim=768)

        # Short prompt (1 word) -> min 4
        emb_short = encoder.encode("test")
        assert emb_short.token_embeddings.shape[0] >= 4

        # Long prompt -> max 64
        long_prompt = " ".join(["word"] * 100)
        emb_long = encoder.encode(long_prompt)
        assert emb_long.token_embeddings.shape[0] <= 64

    @pytest.mark.unit
    def test_encode_fallback_l2_normalized(self):
        """Test encode() fallback token embeddings are L2 normalized."""
        encoder = TextConditioningEncoder(conditioning_dim=768)
        emb = encoder.encode("test prompt")

        # Check L2 normalization along feature dimension
        norms = np.linalg.norm(emb.token_embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones_like(norms), decimal=5)

    @pytest.mark.unit
    def test_encode_fallback_metadata(self):
        """Test encode() fallback includes correct metadata."""
        encoder = TextConditioningEncoder(conditioning_dim=768)
        prompt = "Create a box with dimensions"
        emb = encoder.encode(prompt)

        assert emb.metadata["prompt"] == prompt
        assert "seq_len" in emb.metadata
        assert "seed" in emb.metadata
        assert "word_count" in emb.metadata
        assert emb.metadata["word_count"] == 5

    @pytest.mark.unit
    def test_encode_batch(self):
        """Test encode_batch() encodes multiple prompts."""
        encoder = TextConditioningEncoder(conditioning_dim=768)
        prompts = ["test one", "test two", "test three"]

        embeddings = encoder.encode_batch(prompts)

        assert len(embeddings) == 3
        for i, emb in enumerate(embeddings):
            assert emb.source_type == "text"
            assert emb.metadata["prompt"] == prompts[i]

    @pytest.mark.unit
    def test_encode_empty_string(self):
        """Test encode() handles empty string gracefully."""
        encoder = TextConditioningEncoder(conditioning_dim=768)
        emb = encoder.encode("")

        assert emb.token_embeddings is not None
        assert emb.pooled_embedding is not None
        assert emb.validate() is True

    @pytest.mark.unit
    def test_encode_special_characters(self):
        """Test encode() handles special characters."""
        encoder = TextConditioningEncoder(conditioning_dim=768)
        prompt = "Box 100mm × 50mm @ 45°"

        emb = encoder.encode(prompt)

        assert emb.pooled_embedding is not None
        assert emb.token_embeddings is not None


# =============================================================================
# ImageConditioningEncoder Tests
# =============================================================================


class TestImageConditioningEncoder:
    """Test ImageConditioningEncoder with hash fallback."""

    @pytest.mark.unit
    def test_init_default(self):
        """Test ImageConditioningEncoder initialization with defaults."""
        encoder = ImageConditioningEncoder()

        assert encoder.model_name == "dino_vits16"
        assert encoder.conditioning_dim == 768
        assert encoder.freeze_encoder is True
        assert encoder.device == "cpu"
        assert encoder.image_size == 224

    @pytest.mark.unit
    def test_init_custom(self):
        """Test ImageConditioningEncoder initialization with custom values."""
        encoder = ImageConditioningEncoder(
            model_name="dino_vitb16",
            conditioning_dim=512,
            freeze_encoder=False,
            device="cuda:0",
            image_size=512,
        )

        assert encoder.model_name == "dino_vitb16"
        assert encoder.conditioning_dim == 512
        assert encoder.freeze_encoder is False
        assert encoder.device == "cuda:0"
        assert encoder.image_size == 512

    @pytest.mark.unit
    def test_encode_fallback_with_temp_file(self):
        """Test encode() fallback with temporary image file."""
        encoder = ImageConditioningEncoder(conditioning_dim=768)

        # Create a temporary dummy file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = Path(f.name)
            f.write(b"dummy_image_data")

        try:
            emb = encoder.encode(temp_path)

            assert emb.source_type == "image"
            assert emb.source_model == "hash_fallback"
            assert emb.embed_dim == 768
            assert emb.token_embeddings is not None
            assert emb.token_embeddings.shape[0] == 196  # 14×14 patch grid
            assert emb.pooled_embedding is not None
            assert emb.validate() is True
        finally:
            if temp_path.exists():
                temp_path.unlink()

    @pytest.mark.unit
    def test_encode_fallback_deterministic(self):
        """Test encode() fallback produces same embedding for same image path."""
        encoder = ImageConditioningEncoder(conditioning_dim=768)

        emb1 = encoder.encode(Path("/tmp/test_image.png"))
        emb2 = encoder.encode(Path("/tmp/test_image.png"))

        np.testing.assert_array_almost_equal(
            emb1.pooled_embedding, emb2.pooled_embedding
        )
        np.testing.assert_array_almost_equal(
            emb1.token_embeddings, emb2.token_embeddings
        )

    @pytest.mark.unit
    def test_encode_fallback_patch_count(self):
        """Test encode() fallback produces correct patch count."""
        encoder = ImageConditioningEncoder(conditioning_dim=768)
        emb = encoder.encode(Path("/tmp/test.png"))

        # ViT-like 14×14 patch grid = 196 patches
        assert emb.token_embeddings.shape == (196, 768)

    @pytest.mark.unit
    def test_encode_fallback_metadata(self):
        """Test encode() fallback includes correct metadata."""
        encoder = ImageConditioningEncoder(
            conditioning_dim=768,
            image_size=256,
        )
        image_path = Path("/tmp/test_image.png")
        emb = encoder.encode(image_path)

        assert emb.metadata["image_path"] == str(image_path)
        assert emb.metadata["image_size"] == 256
        assert emb.metadata["patch_count"] == 196

    @pytest.mark.unit
    def test_encode_from_array_with_uint8(self):
        """Test encode_from_array() with uint8 array."""
        encoder = ImageConditioningEncoder(conditioning_dim=768)

        # Create dummy image array (H, W, 3) with values 0-255
        image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

        emb = encoder.encode_from_array(image)

        assert emb.source_type == "image"
        assert emb.token_embeddings is not None
        assert emb.pooled_embedding is not None
        assert emb.validate() is True

    @pytest.mark.unit
    def test_encode_from_array_with_float(self):
        """Test encode_from_array() with float array [0, 1]."""
        encoder = ImageConditioningEncoder(conditioning_dim=768)

        # Create dummy image array with values 0.0-1.0
        image = np.random.rand(224, 224, 3).astype(np.float32)

        emb = encoder.encode_from_array(image)

        assert emb.source_type == "image"
        assert emb.token_embeddings is not None
        assert emb.validate() is True

    @pytest.mark.unit
    def test_encode_path_type_conversion(self):
        """Test encode() accepts both Path and string."""
        encoder = ImageConditioningEncoder(conditioning_dim=768)

        # Test with Path
        emb1 = encoder.encode(Path("/tmp/test1.png"))
        assert emb1.source_type == "image"

        # Test with string
        emb2 = encoder.encode("/tmp/test2.png")
        assert emb2.source_type == "image"


# =============================================================================
# MultiModalConditioner Tests
# =============================================================================


class TestMultiModalConditioner:
    """Test MultiModalConditioner with fusion strategies."""

    @pytest.mark.unit
    def test_init_default(self):
        """Test MultiModalConditioner initialization with defaults."""
        conditioner = MultiModalConditioner()

        assert conditioner.text_encoder.model_name == "bert-base-uncased"
        assert conditioner.image_encoder.model_name == "dino_vits16"
        assert conditioner.conditioning_dim == 768
        assert conditioner.fusion_method == "concat"
        assert conditioner.device == "cpu"

    @pytest.mark.unit
    def test_init_custom(self):
        """Test MultiModalConditioner initialization with custom values."""
        conditioner = MultiModalConditioner(
            text_model="roberta-base",
            image_model="dino_vitb16",
            conditioning_dim=512,
            fusion_method="average",
            device="cuda:0",
        )

        assert conditioner.text_encoder.model_name == "roberta-base"
        assert conditioner.image_encoder.model_name == "dino_vitb16"
        assert conditioner.conditioning_dim == 512
        assert conditioner.fusion_method == "average"

    @pytest.mark.unit
    def test_init_invalid_fusion_method(self):
        """Test MultiModalConditioner initialization fails with invalid fusion method."""
        with pytest.raises(ValueError, match="fusion_method must be one of"):
            MultiModalConditioner(fusion_method="invalid_method")

    @pytest.mark.unit
    def test_encode_text_only(self):
        """Test encode() with text only (no image_path)."""
        conditioner = MultiModalConditioner(fusion_method="concat")
        emb = conditioner.encode("test prompt")

        # When image_path is None, returns text embedding directly
        assert emb.source_type == "text"
        assert emb.source_model == "hash_fallback"
        assert emb.token_embeddings is not None
        assert emb.pooled_embedding is not None

    @pytest.mark.unit
    def test_encode_with_image_concat(self):
        """Test encode() with image using concat fusion."""
        conditioner = MultiModalConditioner(
            conditioning_dim=768,
            fusion_method="concat",
        )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = Path(f.name)
            f.write(b"dummy_image_data")

        try:
            emb = conditioner.encode("test prompt", str(temp_path))

            assert emb.source_type == "multimodal"
            assert emb.metadata["fusion_method"] == "concat"
            # Concat doubles the pooled embedding dimension
            if emb.pooled_embedding is not None:
                assert emb.pooled_embedding.shape[0] == 768 * 2
        finally:
            if temp_path.exists():
                temp_path.unlink()

    @pytest.mark.unit
    def test_encode_with_image_average(self):
        """Test encode() with image using average fusion."""
        conditioner = MultiModalConditioner(
            conditioning_dim=768,
            fusion_method="average",
        )

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = Path(f.name)
            f.write(b"dummy_image_data")

        try:
            emb = conditioner.encode("test prompt", str(temp_path))

            assert emb.source_type == "multimodal"
            assert emb.metadata["fusion_method"] == "average"
            # Average keeps same embedding dimension
            if emb.pooled_embedding is not None:
                assert emb.pooled_embedding.shape[0] == 768
        finally:
            if temp_path.exists():
                temp_path.unlink()

    @pytest.mark.unit
    def test_encode_with_image_text_only(self):
        """Test encode() with image but text_only fusion."""
        conditioner = MultiModalConditioner(fusion_method="text_only")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = Path(f.name)
            f.write(b"dummy_image_data")

        try:
            emb = conditioner.encode("test prompt", str(temp_path))

            # text_only fusion still returns multimodal with text embeddings
            assert emb.source_type == "multimodal"
            assert emb.metadata["fusion_method"] == "text_only"
        finally:
            if temp_path.exists():
                temp_path.unlink()

    @pytest.mark.unit
    def test_encode_with_image_image_only(self):
        """Test encode() with image but image_only fusion."""
        conditioner = MultiModalConditioner(fusion_method="image_only")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = Path(f.name)
            f.write(b"dummy_image_data")

        try:
            emb = conditioner.encode("test prompt", str(temp_path))

            # image_only fusion still returns multimodal with image embeddings
            assert emb.source_type == "multimodal"
            assert emb.metadata["fusion_method"] == "image_only"
        finally:
            if temp_path.exists():
                temp_path.unlink()

    @pytest.mark.unit
    def test_encode_batch(self):
        """Test encode_batch() with multiple prompts."""
        conditioner = MultiModalConditioner(fusion_method="concat")
        prompts = ["test one", "test two", "test three"]

        embeddings = conditioner.encode_batch(prompts)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert emb.source_type == "text"

    @pytest.mark.unit
    def test_encode_batch_with_images_mismatch(self):
        """Test encode_batch() raises error when image paths count mismatches."""
        conditioner = MultiModalConditioner()
        prompts = ["test one", "test two"]
        image_paths = ["/tmp/img1.png"]  # Only 1 image for 2 prompts

        with pytest.raises(ValueError, match="Number of prompts"):
            conditioner.encode_batch(prompts, image_paths)

    @pytest.mark.unit
    def test_encode_batch_with_images(self):
        """Test encode_batch() with multiple prompts and images."""
        conditioner = MultiModalConditioner(fusion_method="concat")
        prompts = ["test one", "test two"]
        image_paths = ["/tmp/img1.png", "/tmp/img2.png"]

        embeddings = conditioner.encode_batch(prompts, image_paths)

        assert len(embeddings) == 2
        for emb in embeddings:
            assert emb.source_type == "multimodal"


# =============================================================================
# ConstraintPredictor Tests
# =============================================================================


class TestConstraintType:
    """Test ConstraintType enum."""

    @pytest.mark.unit
    def test_constraint_type_members(self):
        """Test all constraint types are defined."""
        assert ConstraintType.BOUNDING_BOX.value == "bounding_box"
        assert ConstraintType.SYMMETRY.value == "symmetry"
        assert ConstraintType.PLANARITY.value == "planarity"
        assert ConstraintType.SMOOTHNESS.value == "smoothness"
        assert ConstraintType.CONNECTIVITY.value == "connectivity"
        assert ConstraintType.MANIFOLD.value == "manifold"
        assert ConstraintType.REGULARITY.value == "regularity"
        assert ConstraintType.WATERTIGHT.value == "watertight"

    @pytest.mark.unit
    def test_constraint_type_count(self):
        """Test that all 8 constraint types exist."""
        assert len(list(ConstraintType)) == 8


class TestConstraintPrediction:
    """Test ConstraintPrediction dataclass."""

    @pytest.mark.unit
    def test_init_valid(self):
        """Test ConstraintPrediction initialization with valid confidence."""
        pred = ConstraintPrediction(
            constraint_type=ConstraintType.BOUNDING_BOX,
            confidence=0.75,
            parameters={"dimensions": [100, 50, 20]},
            source="dimension_regex",
        )

        assert pred.constraint_type == ConstraintType.BOUNDING_BOX
        assert pred.confidence == 0.75
        assert pred.parameters["dimensions"] == [100, 50, 20]
        assert pred.source == "dimension_regex"

    @pytest.mark.unit
    def test_init_confidence_0(self):
        """Test ConstraintPrediction with confidence 0.0."""
        pred = ConstraintPrediction(
            constraint_type=ConstraintType.SYMMETRY,
            confidence=0.0,
            parameters={},
            source="test",
        )
        assert pred.confidence == 0.0

    @pytest.mark.unit
    def test_init_confidence_1(self):
        """Test ConstraintPrediction with confidence 1.0."""
        pred = ConstraintPrediction(
            constraint_type=ConstraintType.SYMMETRY,
            confidence=1.0,
            parameters={},
            source="test",
        )
        assert pred.confidence == 1.0

    @pytest.mark.unit
    def test_init_invalid_confidence_too_low(self):
        """Test ConstraintPrediction fails with confidence < 0."""
        with pytest.raises(ValueError, match="confidence must be in"):
            ConstraintPrediction(
                constraint_type=ConstraintType.SYMMETRY,
                confidence=-0.1,
                parameters={},
                source="test",
            )

    @pytest.mark.unit
    def test_init_invalid_confidence_too_high(self):
        """Test ConstraintPrediction fails with confidence > 1."""
        with pytest.raises(ValueError, match="confidence must be in"):
            ConstraintPrediction(
                constraint_type=ConstraintType.SYMMETRY,
                confidence=1.5,
                parameters={},
                source="test",
            )


class TestConstraintPredictor:
    """Test ConstraintPredictor."""

    @pytest.mark.unit
    def test_init_default(self):
        """Test ConstraintPredictor initialization with defaults."""
        predictor = ConstraintPredictor()

        assert predictor.device == "cpu"
        assert predictor.embedding_dim == 768
        assert predictor._learned_model is None

    @pytest.mark.unit
    def test_init_custom(self):
        """Test ConstraintPredictor initialization with custom values."""
        predictor = ConstraintPredictor(device="cuda:0", embedding_dim=512)

        assert predictor.device == "cuda:0"
        assert predictor.embedding_dim == 512

    @pytest.mark.unit
    def test_predict_from_prompt_no_constraints(self):
        """Test predict_from_prompt() with prompt without keywords."""
        predictor = ConstraintPredictor()
        predictions = predictor.predict_from_prompt("just a simple shape")

        # Should always have MANIFOLD constraint
        assert len(predictions) >= 1
        assert any(p.constraint_type == ConstraintType.MANIFOLD for p in predictions)

    @pytest.mark.unit
    def test_predict_from_prompt_bounding_box(self):
        """Test predict_from_prompt() detects dimensional constraints."""
        predictor = ConstraintPredictor()
        predictions = predictor.predict_from_prompt("A box 100mm × 50mm × 20mm")

        bbox_preds = [p for p in predictions if p.constraint_type == ConstraintType.BOUNDING_BOX]
        assert len(bbox_preds) > 0
        assert bbox_preds[0].source == "dimension_regex"
        assert bbox_preds[0].parameters["count"] >= 3

    @pytest.mark.unit
    def test_predict_from_prompt_symmetry(self):
        """Test predict_from_prompt() detects symmetry."""
        predictor = ConstraintPredictor()
        predictions = predictor.predict_from_prompt("A symmetric mirror shape")

        symmetry_preds = [p for p in predictions if p.constraint_type == ConstraintType.SYMMETRY]
        assert len(symmetry_preds) > 0
        assert symmetry_preds[0].confidence == 0.85

    @pytest.mark.unit
    def test_predict_from_prompt_smoothness(self):
        """Test predict_from_prompt() detects smoothness."""
        predictor = ConstraintPredictor()
        predictions = predictor.predict_from_prompt("A smooth rounded edge with fillets")

        smoothness_preds = [p for p in predictions if p.constraint_type == ConstraintType.SMOOTHNESS]
        assert len(smoothness_preds) > 0
        assert smoothness_preds[0].source == "keyword"

    @pytest.mark.unit
    def test_predict_from_prompt_regularity(self):
        """Test predict_from_prompt() detects regularity."""
        predictor = ConstraintPredictor()
        predictions = predictor.predict_from_prompt("A grid pattern of evenly spaced holes")

        regularity_preds = [p for p in predictions if p.constraint_type == ConstraintType.REGULARITY]
        assert len(regularity_preds) > 0
        assert regularity_preds[0].confidence == 0.80

    @pytest.mark.unit
    def test_predict_from_prompt_connectivity(self):
        """Test predict_from_prompt() detects connectivity."""
        predictor = ConstraintPredictor()
        predictions = predictor.predict_from_prompt("Two parts joined together")

        connectivity_preds = [p for p in predictions if p.constraint_type == ConstraintType.CONNECTIVITY]
        assert len(connectivity_preds) > 0

    @pytest.mark.unit
    def test_predict_from_prompt_watertight(self):
        """Test predict_from_prompt() detects watertight."""
        predictor = ConstraintPredictor()
        predictions = predictor.predict_from_prompt("A closed solid printable shape")

        watertight_preds = [p for p in predictions if p.constraint_type == ConstraintType.WATERTIGHT]
        assert len(watertight_preds) > 0
        assert watertight_preds[0].confidence == 0.85

    @pytest.mark.unit
    def test_predict_from_prompt_planarity(self):
        """Test predict_from_prompt() detects planarity."""
        predictor = ConstraintPredictor()
        predictions = predictor.predict_from_prompt("A flat planar surface")

        planarity_preds = [p for p in predictions if p.constraint_type == ConstraintType.PLANARITY]
        assert len(planarity_preds) > 0
        assert planarity_preds[0].confidence == 0.80

    @pytest.mark.unit
    def test_predict_from_prompt_multiple_constraints(self):
        """Test predict_from_prompt() detects multiple constraints."""
        predictor = ConstraintPredictor()
        predictions = predictor.predict_from_prompt(
            "A 100mm × 50mm symmetric flat box with smooth edges and grid pattern"
        )

        # Should detect multiple constraint types
        constraint_types = {p.constraint_type for p in predictions}
        assert len(constraint_types) >= 3

    @pytest.mark.unit
    def test_predict_from_prompt_always_includes_manifold(self):
        """Test predict_from_prompt() always includes MANIFOLD."""
        predictor = ConstraintPredictor()
        predictions = predictor.predict_from_prompt("any prompt")

        manifold_preds = [p for p in predictions if p.constraint_type == ConstraintType.MANIFOLD]
        assert len(manifold_preds) > 0
        assert manifold_preds[0].confidence == 0.5
        assert manifold_preds[0].parameters.get("default") is True

    @pytest.mark.unit
    def test_predict_from_embeddings_no_model(self):
        """Test predict_from_embeddings() returns empty when no learned model."""
        predictor = ConstraintPredictor()
        embeddings = ConditioningEmbeddings(
            pooled_embedding=np.random.randn(768).astype(np.float32)
        )

        predictions = predictor.predict_from_embeddings(embeddings)

        assert predictions == []

    @pytest.mark.unit
    def test_predict_from_embeddings_no_pooled(self):
        """Test predict_from_embeddings() returns empty when no pooled embedding."""
        predictor = ConstraintPredictor()
        embeddings = ConditioningEmbeddings(
            token_embeddings=np.random.randn(10, 768).astype(np.float32)
        )

        predictions = predictor.predict_from_embeddings(embeddings)

        assert predictions == []

    @pytest.mark.unit
    def test_set_get_learned_model(self):
        """Test set_learned_model() and get_learned_model()."""
        predictor = ConstraintPredictor()
        mock_model = MagicMock()

        predictor.set_learned_model(mock_model)

        retrieved = predictor.get_learned_model()
        assert retrieved is not None

    @pytest.mark.unit
    def test_set_learned_model_none(self):
        """Test set_learned_model() with None."""
        predictor = ConstraintPredictor()
        predictor._learned_model = MagicMock()

        predictor.set_learned_model(None)

        assert predictor.get_learned_model() is None

    @pytest.mark.unit
    def test_to_loss_weights_empty(self):
        """Test to_loss_weights() with empty predictions."""
        predictor = ConstraintPredictor()
        weights = predictor.to_loss_weights([])

        assert weights == {}

    @pytest.mark.unit
    def test_to_loss_weights_single_prediction(self):
        """Test to_loss_weights() with single prediction."""
        predictor = ConstraintPredictor()
        predictions = [
            ConstraintPrediction(
                constraint_type=ConstraintType.BOUNDING_BOX,
                confidence=0.8,
                parameters={},
                source="test",
            )
        ]

        weights = predictor.to_loss_weights(predictions)

        assert "bounding_box" in weights
        assert weights["bounding_box"] == 0.8

    @pytest.mark.unit
    def test_to_loss_weights_multiple_predictions(self):
        """Test to_loss_weights() with multiple predictions."""
        predictor = ConstraintPredictor()
        predictions = [
            ConstraintPrediction(
                constraint_type=ConstraintType.BOUNDING_BOX,
                confidence=0.8,
                parameters={},
                source="test",
            ),
            ConstraintPrediction(
                constraint_type=ConstraintType.SYMMETRY,
                confidence=0.9,
                parameters={},
                source="test",
            ),
        ]

        weights = predictor.to_loss_weights(predictions)

        assert len(weights) == 2
        assert weights["bounding_box"] == 0.8
        assert weights["symmetry"] == 0.9

    @pytest.mark.unit
    def test_predict_from_prompt_dimension_regex_variants(self):
        """Test predict_from_prompt() dimension regex matches variants."""
        predictor = ConstraintPredictor()

        # Test mm
        preds_mm = predictor.predict_from_prompt("100mm wide")
        assert any(p.constraint_type == ConstraintType.BOUNDING_BOX for p in preds_mm)

        # Test cm
        preds_cm = predictor.predict_from_prompt("50cm tall")
        assert any(p.constraint_type == ConstraintType.BOUNDING_BOX for p in preds_cm)

        # Test inches
        preds_in = predictor.predict_from_prompt("10 inches long")
        assert any(p.constraint_type == ConstraintType.BOUNDING_BOX for p in preds_in)

        # Test decimal
        preds_dec = predictor.predict_from_prompt("3.5mm thick")
        assert any(p.constraint_type == ConstraintType.BOUNDING_BOX for p in preds_dec)

    @pytest.mark.unit
    def test_predict_from_prompt_case_insensitive(self):
        """Test predict_from_prompt() keywords are case-insensitive."""
        predictor = ConstraintPredictor()

        # Lowercase
        preds_lower = predictor.predict_from_prompt("smooth edges")
        smoothness_lower = [p for p in preds_lower if p.constraint_type == ConstraintType.SMOOTHNESS]

        # Uppercase
        preds_upper = predictor.predict_from_prompt("SMOOTH EDGES")
        smoothness_upper = [p for p in preds_upper if p.constraint_type == ConstraintType.SMOOTHNESS]

        # Both should find smoothness
        assert len(smoothness_lower) > 0
        assert len(smoothness_upper) > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestConditioningIntegration:
    """Integration tests for conditioning module."""

    @pytest.mark.unit
    def test_text_encoder_to_multimodal(self):
        """Test text encoder output can be used in multimodal conditioner."""
        text_encoder = TextConditioningEncoder()
        emb = text_encoder.encode("test prompt")

        # Verify it's valid for multimodal use
        assert emb.source_type == "text"
        assert emb.validate() is True

    @pytest.mark.unit
    def test_constraint_predictor_with_embeddings(self):
        """Test constraint predictor with encoder output."""
        encoder = TextConditioningEncoder()
        predictor = ConstraintPredictor()

        prompt = "A 100mm box with symmetric smooth edges"
        encoder.encode(prompt)

        # Get constraints from prompt
        prompt_constraints = predictor.predict_from_prompt(prompt)
        assert len(prompt_constraints) > 0

        # Get loss weights
        weights = predictor.to_loss_weights(prompt_constraints)
        assert len(weights) > 0

    @pytest.mark.unit
    def test_multimodal_metadata_preservation(self):
        """Test multimodal conditioner preserves metadata when image provided."""
        conditioner = MultiModalConditioner(fusion_method="concat")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = Path(f.name)
            f.write(b"dummy_image_data")

        try:
            prompt = "test prompt"
            emb = conditioner.encode(prompt, str(temp_path))

            # Check metadata - fusion_method only present when image provided
            assert emb.metadata["prompt"] == prompt
            assert "fusion_method" in emb.metadata
            assert emb.metadata["fusion_method"] == "concat"
        finally:
            if temp_path.exists():
                temp_path.unlink()

    @pytest.mark.unit
    def test_embeddings_roundtrip_numpy(self):
        """Test embeddings can roundtrip through numpy."""
        token_emb = np.random.randn(10, 768).astype(np.float32)
        pooled = np.random.randn(768).astype(np.float32)

        original = ConditioningEmbeddings(
            token_embeddings=token_emb,
            pooled_embedding=pooled,
            source_type="test",
            source_model="test_model",
        )

        # Simulate roundtrip
        reconstructed = ConditioningEmbeddings(
            token_embeddings=original.token_embeddings.copy(),
            pooled_embedding=original.pooled_embedding.copy(),
            source_type=original.source_type,
            source_model=original.source_model,
            embed_dim=original.embed_dim,
        )

        assert reconstructed.validate() is True
        np.testing.assert_array_equal(
            original.token_embeddings, reconstructed.token_embeddings
        )
        np.testing.assert_array_equal(
            original.pooled_embedding, reconstructed.pooled_embedding
        )
