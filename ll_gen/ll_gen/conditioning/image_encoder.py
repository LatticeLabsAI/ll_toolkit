"""Image conditioning encoder — encodes images into embeddings.

Wraps ``ll_stepnet``'s ``ImageConditioner`` when available. Falls back to
deterministic hash-based embeddings for reproducibility in testing without
vision models installed.
"""
from __future__ import annotations

import hashlib
import logging
import tempfile
from pathlib import Path

import numpy as np

from ll_gen.conditioning.embeddings import ConditioningEmbeddings

_log = logging.getLogger(__name__)

_STEPNET_AVAILABLE = False
try:
    from ll_stepnet.stepnet.conditioning import ImageConditioner

    _STEPNET_AVAILABLE = True
except ImportError:
    _log.debug("ll_stepnet not available; image encoder will use hash fallback")

_PIL_AVAILABLE = False
try:
    from PIL import Image

    _PIL_AVAILABLE = True
except ImportError:
    _log.debug("PIL not available; image encoder requires fallback")


class ImageConditioningEncoder:
    """Encodes images into ConditioningEmbeddings.

    Uses ll_stepnet's ImageConditioner if available, otherwise falls back
    to deterministic hash-based embeddings.

    Attributes:
        model_name: Vision model identifier (e.g., "dino_vits16").
        conditioning_dim: Embedding dimension.
        freeze_encoder: Whether to freeze encoder parameters.
        device: Torch device ("cpu" or "cuda:*").
        image_size: Image size for preprocessing.
    """

    def __init__(
        self,
        model_name: str = "dino_vits16",
        conditioning_dim: int = 768,
        freeze_encoder: bool = True,
        device: str = "cpu",
        image_size: int = 224,
    ) -> None:
        """Initialize ImageConditioningEncoder.

        Args:
            model_name: Vision model identifier.
            conditioning_dim: Embedding dimension.
            freeze_encoder: Whether to freeze encoder parameters.
            device: Torch device ("cpu" or "cuda:*").
            image_size: Size to resize images to (square).
        """
        self.model_name = model_name
        self.conditioning_dim = conditioning_dim
        self.freeze_encoder = freeze_encoder
        self.device = device
        self.image_size = image_size
        self._conditioner = None

    def encode(self, image_path: str | Path) -> ConditioningEmbeddings:
        """Encode an image from file path.

        Args:
            image_path: Path to image file.

        Returns:
            ConditioningEmbeddings with patch/region embeddings.
        """
        image_path = Path(image_path)

        if not _STEPNET_AVAILABLE:
            return self._encode_fallback(image_path)

        try:
            if self._conditioner is None:
                self._init_conditioner()

            if not _PIL_AVAILABLE:
                _log.warning("PIL not available; using fallback")
                return self._encode_fallback(image_path)

            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

            # Convert to tensor and encode
            image_tensor = self._image_to_tensor(image)
            image_tensor = image_tensor.to(self.device)

            with self._safe_no_grad():
                patch_emb = self._conditioner.encode_image(image_tensor)

            # Extract embeddings
            patch_emb_np = patch_emb.detach().cpu().numpy()

            # Pool patch embeddings to single vector
            pooled = patch_emb_np.mean(axis=0).astype(np.float32)

            return ConditioningEmbeddings(
                token_embeddings=patch_emb_np,
                pooled_embedding=pooled,
                source_type="image",
                source_model=self.model_name,
                embed_dim=self.conditioning_dim,
                metadata={
                    "image_path": str(image_path),
                    "image_size": self.image_size,
                    "patch_count": patch_emb_np.shape[0],
                },
            )
        except Exception as e:
            _log.warning(f"Error encoding image with ll_stepnet: {e}; using fallback")
            return self._encode_fallback(image_path)

    def encode_from_array(self, image: np.ndarray) -> ConditioningEmbeddings:
        """Encode an image from numpy array.

        Args:
            image: Image array (H, W, 3) with values in [0, 255] or [0, 1].

        Returns:
            ConditioningEmbeddings.
        """
        try:
            if not _PIL_AVAILABLE:
                raise ImportError("PIL not available")

            # Normalize if needed
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

            # Convert to PIL and save to temp file
            pil_image = Image.fromarray(image)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                temp_path = Path(f.name)
                pil_image.save(temp_path)

            try:
                result = self.encode(temp_path)
            finally:
                if temp_path.exists():
                    temp_path.unlink()

            return result
        except Exception as e:
            _log.warning(f"Error encoding array: {e}; generating fallback")
            return self._encode_fallback(Path(str(hash(image.tobytes()))))

    def _init_conditioner(self) -> None:
        """Lazy-initialize the ImageConditioner.

        Instantiates the model and moves it to the target device.
        """
        if not _STEPNET_AVAILABLE:
            raise RuntimeError("ll_stepnet not available")

        try:
            self._conditioner = ImageConditioner(
                model_name=self.model_name,
                device=self.device,
            )
            if self.freeze_encoder:
                for param in self._conditioner.model.parameters():
                    param.requires_grad = False
            _log.info(f"Initialized ImageConditioner with {self.model_name}")
        except Exception as e:
            _log.error(f"Failed to initialize ImageConditioner: {e}")
            raise

    def _encode_fallback(self, image_path: Path) -> ConditioningEmbeddings:
        """Deterministic hash-based embedding fallback.

        Produces reproducible embeddings without vision models:
        - Hash the image file path to a deterministic seed
        - Use numpy.random with that seed to generate embeddings
        - Simulate patch embeddings

        Args:
            image_path: Path to image.

        Returns:
            ConditioningEmbeddings with hash-based embeddings.
        """
        # Generate seed from path hash
        path_str = str(image_path)
        path_hash = hashlib.sha256(path_str.encode()).digest()
        seed = int.from_bytes(path_hash[:4], byteorder="big") % (2**31)

        # Create random state with deterministic seed
        rng = np.random.RandomState(seed)

        # Simulate patch grid (14x14 for ViT-like models)
        patch_count = 196
        token_emb = rng.randn(patch_count, self.conditioning_dim).astype(np.float32)
        token_emb = token_emb / (
            np.linalg.norm(token_emb, axis=1, keepdims=True) + 1e-9
        )

        pooled = token_emb.mean(axis=0).astype(np.float32)

        return ConditioningEmbeddings(
            token_embeddings=token_emb,
            pooled_embedding=pooled,
            source_type="image",
            source_model="hash_fallback",
            embed_dim=self.conditioning_dim,
            metadata={
                "image_path": str(image_path),
                "image_size": self.image_size,
                "patch_count": patch_count,
                "seed": int(seed),
            },
        )

    @staticmethod
    def _image_to_tensor(image):
        """Convert PIL Image to torch tensor.

        Args:
            image: PIL Image object.

        Returns:
            torch.Tensor of shape (1, 3, H, W) with normalized values.
        """
        try:
            import torchvision.transforms as transforms

            # Normalize for ImageNet
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
            tensor = transform(image)
            return tensor.unsqueeze(0)
        except ImportError:
            _log.error("torch or torchvision not available")
            raise

    @staticmethod
    def _safe_no_grad():
        """Context manager for torch.no_grad() if torch is available.

        Returns:
            torch.no_grad() context manager or a no-op context manager.
        """
        try:
            import torch

            return torch.no_grad()
        except ImportError:

            class NoOp:
                """No-op context manager when torch unavailable."""

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    """Exit context, doing nothing."""
                    return False

            return NoOp()
