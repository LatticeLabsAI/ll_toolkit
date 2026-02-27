"""Text conditioning encoder — encodes text prompts into embeddings.

Wraps ``ll_stepnet``'s ``TextConditioner`` when available. Falls back to
deterministic hash-based embeddings for reproducibility in testing and
pipeline development without transformers installed.
"""
from __future__ import annotations

import hashlib
import logging

import numpy as np

from ll_gen.conditioning._utils import safe_no_grad as _safe_no_grad
from ll_gen.conditioning.embeddings import ConditioningEmbeddings

_log = logging.getLogger(__name__)

_STEPNET_AVAILABLE = False
try:
    from ll_stepnet.stepnet.conditioning import TextConditioner

    _STEPNET_AVAILABLE = True
except ImportError:
    _log.debug("ll_stepnet not available; text encoder will use hash fallback")


class TextConditioningEncoder:
    """Encodes text prompts into ConditioningEmbeddings.

    Uses ll_stepnet's TextConditioner if available, otherwise falls back
    to deterministic hash-based embeddings.

    Attributes:
        model_name: Hugging Face model identifier.
        conditioning_dim: Embedding dimension.
        freeze_encoder: Whether to freeze encoder parameters.
        device: Torch device ("cpu" or "cuda:*").
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        conditioning_dim: int = 768,
        freeze_encoder: bool = True,
        device: str = "cpu",
    ) -> None:
        """Initialize TextConditioningEncoder.

        Args:
            model_name: Hugging Face model identifier.
            conditioning_dim: Embedding dimension.
            freeze_encoder: Whether to freeze encoder parameters.
            device: Torch device ("cpu" or "cuda:*").
        """
        self.model_name = model_name
        self.conditioning_dim = conditioning_dim
        self.freeze_encoder = freeze_encoder
        self.device = device
        self._conditioner = None

    def encode(self, prompt: str) -> ConditioningEmbeddings:
        """Encode a single text prompt.

        Args:
            prompt: Text prompt to encode.

        Returns:
            ConditioningEmbeddings with token and pooled embeddings.
        """
        if not _STEPNET_AVAILABLE:
            return self._encode_fallback(prompt)

        try:
            if self._conditioner is None:
                self._init_conditioner()

            # Tokenize and encode
            tokens = self._conditioner.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            with _safe_no_grad():
                outputs = self._conditioner.model(**tokens, output_hidden_states=True)

            # Extract last hidden state for token embeddings
            token_emb = outputs.last_hidden_state.detach().cpu().numpy()

            # Pool via mean (excluding special tokens)
            attention_mask = tokens.get("attention_mask")
            if attention_mask is not None:
                mask = attention_mask.float().unsqueeze(-1)
                pooled = (token_emb * mask.cpu().numpy()).sum(axis=1) / (
                    mask.cpu().numpy().sum(axis=1) + 1e-9
                )
            else:
                pooled = token_emb.mean(axis=1)

            return ConditioningEmbeddings(
                token_embeddings=token_emb[0],
                pooled_embedding=pooled[0],
                source_type="text",
                source_model=self.model_name,
                embed_dim=self.conditioning_dim,
                metadata={
                    "prompt": prompt,
                    "seq_len": token_emb.shape[1],
                },
            )
        except Exception as e:
            _log.warning(f"Error encoding text with ll_stepnet: {e}; using fallback")
            return self._encode_fallback(prompt)

    def encode_batch(self, prompts: list[str]) -> list[ConditioningEmbeddings]:
        """Encode multiple text prompts.

        Args:
            prompts: List of text prompts.

        Returns:
            List of ConditioningEmbeddings.
        """
        if not _STEPNET_AVAILABLE:
            return [self.encode(prompt) for prompt in prompts]

        try:
            if self._conditioner is None:
                self._init_conditioner()

            # Tokenize all prompts at once
            tokens = self._conditioner.tokenizer(
                prompts,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}

            with _safe_no_grad():
                outputs = self._conditioner.model(**tokens, output_hidden_states=True)

            token_emb_all = outputs.last_hidden_state.detach().cpu().numpy()
            attention_mask = tokens.get("attention_mask")

            results = []
            for i in range(len(prompts)):
                token_emb_i = token_emb_all[i]
                if attention_mask is not None:
                    mask_i = attention_mask[i].float().unsqueeze(-1).cpu().numpy()
                    pooled_i = (token_emb_i * mask_i).sum(axis=0) / (
                        mask_i.sum(axis=0) + 1e-9
                    )
                else:
                    pooled_i = token_emb_i.mean(axis=0)

                results.append(
                    ConditioningEmbeddings(
                        token_embeddings=token_emb_i,
                        pooled_embedding=pooled_i,
                        source_type="text",
                        source_model=self.model_name,
                        embed_dim=self.conditioning_dim,
                        metadata={
                            "prompt": prompts[i],
                            "seq_len": token_emb_i.shape[0],
                        },
                    )
                )
            return results
        except Exception as e:
            _log.warning(f"Error batch encoding with ll_stepnet: {e}; using fallback loop")
            return [self.encode(prompt) for prompt in prompts]

    def _init_conditioner(self) -> None:
        """Lazy-initialize the TextConditioner.

        Instantiates the model and moves it to the target device.
        """
        if not _STEPNET_AVAILABLE:
            raise RuntimeError("ll_stepnet not available")

        try:

            self._conditioner = TextConditioner(
                model_name=self.model_name,
                device=self.device,
            )
            if self.freeze_encoder:
                for param in self._conditioner.model.parameters():
                    param.requires_grad = False
            _log.info(f"Initialized TextConditioner with {self.model_name}")
        except Exception as e:
            _log.error(f"Failed to initialize TextConditioner: {e}")
            raise

    def _encode_fallback(self, prompt: str) -> ConditioningEmbeddings:
        """Deterministic hash-based embedding fallback.

        Produces reproducible embeddings without transformers:
        - Hash the prompt to a deterministic seed
        - Use numpy.random with that seed to generate embeddings
        - Scale sequence length based on word count

        Args:
            prompt: Text prompt.

        Returns:
            ConditioningEmbeddings with hash-based embeddings.
        """
        # Generate seed from prompt hash
        prompt_hash = hashlib.sha256(prompt.encode()).digest()
        seed = int.from_bytes(prompt_hash[:4], byteorder="big") % (2**31)

        # Create random state with deterministic seed
        rng = np.random.RandomState(seed)

        # Estimate sequence length from word count
        word_count = len(prompt.split())
        seq_len = min(max(word_count, 4), 64)

        # Generate embeddings
        token_emb = rng.randn(seq_len, self.conditioning_dim).astype(np.float32)
        token_emb = token_emb / (np.linalg.norm(token_emb, axis=1, keepdims=True) + 1e-9)

        pooled = token_emb.mean(axis=0).astype(np.float32)

        return ConditioningEmbeddings(
            token_embeddings=token_emb,
            pooled_embedding=pooled,
            source_type="text",
            source_model="hash_fallback",
            embed_dim=self.conditioning_dim,
            metadata={
                "prompt": prompt,
                "seq_len": seq_len,
                "seed": int(seed),
                "word_count": word_count,
            },
        )

    @staticmethod
    def _safe_no_grad():
        """Context manager for torch.no_grad() if torch is available.

        Returns:
            torch.no_grad() context manager or a no-op context manager.
        """
        return _safe_no_grad()
