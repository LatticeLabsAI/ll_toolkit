"""Vision-Language Model (VLM) integration for CAD annotation extraction.

This module provides integration with VLMs for extracting annotations
(dimensions, tolerances, notes) from rendered CAD images or screenshots.

Classes:
    VlmModel: Abstract base class for all VLMs
    ApiVlmModel: API-based VLMs (GPT-4V, Claude)
    InlineVlmModel: Local VLMs (LLaVA, Qwen-VL)
    VlmResponse: Response from VLM prediction
    VlmOptions: Base options for VLM models
"""

from __future__ import annotations

import base64
import io
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from PIL import Image

_log = logging.getLogger(__name__)

# Try to import optional dependencies
_TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoModelForVision2Seq, AutoProcessor

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _log.debug("transformers not available for InlineVlmModel")

_OPENAI_AVAILABLE = False
try:
    import openai

    _OPENAI_AVAILABLE = True
except ImportError:
    _log.debug("openai not available for ApiVlmModel")

_ANTHROPIC_AVAILABLE = False
try:
    import anthropic

    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _log.debug("anthropic not available for ApiVlmModel")

_EASYOCR_AVAILABLE = False
try:
    import easyocr

    _EASYOCR_AVAILABLE = True
except ImportError:
    _log.debug("easyocr not available for OCR enhancement")


class VlmAnnotation(BaseModel):
    """Single annotation extracted by VLM.

    Attributes:
        annotation_type: Type of annotation (dimension, tolerance, note, label)
        text: Extracted text content
        value: Numeric value (for dimensions)
        unit: Unit (for dimensions)
        confidence: Confidence score (0-1)
        bbox: Bounding box in image [x, y, width, height]
    """

    annotation_type: str  # dimension, tolerance, note, label
    text: str
    value: Optional[float] = None
    unit: Optional[str] = None
    confidence: float = 1.0
    bbox: Optional[list[float]] = None


class VlmResponse(BaseModel):
    """Response from VLM prediction.

    Attributes:
        annotations: List of extracted annotations
        raw_text: Raw text response from model
        metadata: Additional metadata from model
    """

    annotations: list[VlmAnnotation] = Field(default_factory=list)
    raw_text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class VlmOptions(BaseModel):
    """Base options for VLM models.

    Attributes:
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum tokens in response
        use_ocr: Whether to enhance with OCR
    """

    temperature: float = 0.0
    max_tokens: int = 4096
    use_ocr: bool = True


class ApiVlmOptions(VlmOptions):
    """Options for API-based VLM models.

    Attributes:
        api_key: API key for the service
        model_name: Model identifier (gpt-4-vision-preview, claude-3-opus-20240229)
        api_base: Optional custom API base URL
        timeout: Request timeout in seconds
    """

    api_key: str
    model_name: str = "gpt-4-vision-preview"
    api_base: Optional[str] = None
    timeout: int = 60


class InlineVlmOptions(VlmOptions):
    """Options for local/inline VLM models.

    Attributes:
        model_path: Path to model or HuggingFace model ID
        device: Device to run on (cpu, cuda, mps)
        precision: Model precision (fp32, fp16, int8)
    """

    model_path: str = "llava-hf/llava-1.5-7b-hf"
    device: str = "cpu"
    precision: str = "fp32"


class VlmModel(ABC):
    """Base class for vision-language models.

    VLMs analyze rendered CAD images or screenshots to extract:
    - Dimensions and measurements
    - Tolerances and specifications
    - Notes and labels
    - Other textual annotations

    Subclasses must implement:
    - predict(image, prompt): Run inference on image

    Example:
        vlm = ApiVlmModel(ApiVlmOptions(api_key="..."))
        response = vlm.predict(cad_image, "Extract dimensions")
        for annotation in response.annotations:
            print(f"{annotation.annotation_type}: {annotation.text}")
    """

    def __init__(self, options: VlmOptions):
        """Initialize VLM model.

        Args:
            options: VLM configuration options
        """
        self.options = options
        self.ocr_reader = None

        # Initialize OCR if requested
        if options.use_ocr and _EASYOCR_AVAILABLE:
            try:
                self.ocr_reader = easyocr.Reader(["en"])
                _log.info("Initialized EasyOCR reader")
            except Exception as e:
                _log.warning(f"Failed to initialize EasyOCR: {e}")

    @abstractmethod
    def predict(self, image: Image.Image, prompt: str) -> VlmResponse:
        """Run VLM prediction on image.

        Args:
            image: PIL Image to analyze
            prompt: Text prompt for the model

        Returns:
            VlmResponse with extracted annotations
        """
        pass

    def _enhance_with_ocr(
        self, image: Image.Image, vlm_response: VlmResponse
    ) -> VlmResponse:
        """Enhance VLM response with OCR results.

        Args:
            image: Original image
            vlm_response: Response from VLM

        Returns:
            Enhanced VlmResponse
        """
        if not self.ocr_reader:
            return vlm_response

        try:
            # Run OCR
            import numpy as np

            ocr_results = self.ocr_reader.readtext(np.array(image))

            # Add OCR results as annotations
            for bbox, text, confidence in ocr_results:
                # Check if this text is not already in VLM annotations
                if not any(ann.text == text for ann in vlm_response.annotations):
                    # Convert bbox to [x, y, width, height]
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    vlm_response.annotations.append(
                        VlmAnnotation(
                            annotation_type="ocr_text",
                            text=text,
                            confidence=confidence,
                            bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                        )
                    )

            _log.info(f"Enhanced with {len(ocr_results)} OCR results")

        except Exception as e:
            _log.error(f"OCR enhancement failed: {e}")

        return vlm_response

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string.

        Args:
            image: PIL Image

        Returns:
            Base64-encoded image string
        """
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


class ApiVlmModel(VlmModel):
    """API-based VLM for CAD annotation extraction.

    Supports:
    - OpenAI GPT-4V (gpt-4-vision-preview, gpt-4-turbo-2024-04-09)
    - Anthropic Claude (claude-3-opus-20240229, claude-3-sonnet-20240229)

    The model sends rendered CAD images to the API and receives
    structured annotations in response.

    Attributes:
        options: API configuration options
        client: API client instance

    Example:
        vlm = ApiVlmModel(ApiVlmOptions(
            api_key="sk-...",
            model_name="gpt-4-vision-preview"
        ))
        response = vlm.predict(rendered_image, "Extract all dimensions")
    """

    def __init__(self, options: ApiVlmOptions):
        """Initialize API-based VLM.

        Args:
            options: API configuration options
        """
        super().__init__(options)
        self.options: ApiVlmOptions = options
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        """Initialize API client based on model name.

        Returns:
            API client instance

        Raises:
            ImportError: If required API library not installed
            ValueError: If model name not recognized
        """
        model_lower = self.options.model_name.lower()

        # OpenAI models
        if "gpt" in model_lower or "vision" in model_lower:
            if not _OPENAI_AVAILABLE:
                raise ImportError("openai package required for GPT-4V models")

            client = openai.OpenAI(
                api_key=self.options.api_key,
                base_url=self.options.api_base,
                timeout=self.options.timeout,
            )
            _log.info(f"Initialized OpenAI client for {self.options.model_name}")
            return client

        # Anthropic Claude models
        elif "claude" in model_lower:
            if not _ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package required for Claude models")

            client = anthropic.Anthropic(
                api_key=self.options.api_key,
                base_url=self.options.api_base,
                timeout=self.options.timeout,
            )
            _log.info(f"Initialized Anthropic client for {self.options.model_name}")
            return client

        else:
            raise ValueError(
                f"Unsupported model: {self.options.model_name}. "
                "Supported: gpt-4-vision-preview, claude-3-*"
            )

    def predict(self, image: Image.Image, prompt: str) -> VlmResponse:
        """Run VLM prediction using API.

        Args:
            image: PIL Image to analyze
            prompt: Text prompt for the model

        Returns:
            VlmResponse with extracted annotations
        """
        model_lower = self.options.model_name.lower()

        try:
            if "gpt" in model_lower or "vision" in model_lower:
                response = self._predict_openai(image, prompt)
            elif "claude" in model_lower:
                response = self._predict_anthropic(image, prompt)
            else:
                raise ValueError(f"Unsupported model: {self.options.model_name}")

            # Enhance with OCR if enabled
            if self.options.use_ocr:
                response = self._enhance_with_ocr(image, response)

            return response

        except Exception as e:
            _log.error(f"VLM prediction failed: {e}")
            return VlmResponse(
                annotations=[],
                raw_text="",
                metadata={"error": str(e)},
            )

    def _predict_openai(self, image: Image.Image, prompt: str) -> VlmResponse:
        """Predict using OpenAI GPT-4V.

        Args:
            image: PIL Image
            prompt: Text prompt

        Returns:
            VlmResponse
        """
        # Convert image to base64
        image_b64 = self._image_to_base64(image)

        # Build structured prompt
        full_prompt = f"""{prompt}

Please extract all CAD annotations and return them as a JSON array with the following structure:
[
  {{
    "annotation_type": "dimension|tolerance|note|label",
    "text": "extracted text",
    "value": numeric_value_or_null,
    "unit": "unit_string_or_null",
    "confidence": 0.0-1.0
  }}
]
"""

        # Call API
        response = self.client.chat.completions.create(
            model=self.options.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            temperature=self.options.temperature,
            max_tokens=self.options.max_tokens,
        )

        raw_text = response.choices[0].message.content

        # Parse JSON response
        annotations = self._parse_json_annotations(raw_text)

        return VlmResponse(
            annotations=annotations,
            raw_text=raw_text,
            metadata={
                "model": self.options.model_name,
                "usage": response.usage.model_dump() if response.usage else {},
            },
        )

    def _predict_anthropic(self, image: Image.Image, prompt: str) -> VlmResponse:
        """Predict using Anthropic Claude.

        Args:
            image: PIL Image
            prompt: Text prompt

        Returns:
            VlmResponse
        """
        # Convert image to base64
        image_b64 = self._image_to_base64(image)

        # Build structured prompt
        full_prompt = f"""{prompt}

Please extract all CAD annotations and return them as a JSON array with the following structure:
[
  {{
    "annotation_type": "dimension|tolerance|note|label",
    "text": "extracted text",
    "value": numeric_value_or_null,
    "unit": "unit_string_or_null",
    "confidence": 0.0-1.0
  }}
]
"""

        # Call API
        response = self.client.messages.create(
            model=self.options.model_name,
            max_tokens=self.options.max_tokens,
            temperature=self.options.temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": full_prompt},
                    ],
                }
            ],
        )

        raw_text = response.content[0].text

        # Parse JSON response
        annotations = self._parse_json_annotations(raw_text)

        return VlmResponse(
            annotations=annotations,
            raw_text=raw_text,
            metadata={
                "model": self.options.model_name,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            },
        )

    def _parse_json_annotations(self, text: str) -> list[VlmAnnotation]:
        """Parse JSON annotations from model response.

        Args:
            text: Raw text response

        Returns:
            List of VlmAnnotation objects
        """
        annotations = []

        try:
            # Try to find JSON array in response
            json_start = text.find("[")
            json_end = text.rfind("]") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                parsed = json.loads(json_str)

                for item in parsed:
                    annotations.append(VlmAnnotation(**item))

            _log.info(f"Parsed {len(annotations)} annotations from JSON")

        except Exception as e:
            _log.error(f"Failed to parse JSON annotations: {e}")

        return annotations


class InlineVlmModel(VlmModel):
    """Local/inline VLM for CAD annotation extraction.

    Supports local vision-language models via HuggingFace transformers:
    - LLaVA (llava-hf/llava-1.5-7b-hf, llava-hf/llava-1.5-13b-hf)
    - Qwen-VL (Qwen/Qwen-VL-Chat)
    - BLIP-2 (Salesforce/blip2-opt-2.7b)

    These models run locally without API calls, providing privacy
    and offline capabilities.

    Attributes:
        options: Model configuration options
        processor: HuggingFace processor
        model: HuggingFace model

    Example:
        vlm = InlineVlmModel(InlineVlmOptions(
            model_path="llava-hf/llava-1.5-7b-hf",
            device="cuda"
        ))
        response = vlm.predict(rendered_image, "Extract dimensions")
    """

    def __init__(self, options: InlineVlmOptions):
        """Initialize local VLM.

        Args:
            options: Model configuration options

        Raises:
            ImportError: If transformers not available
        """
        super().__init__(options)
        self.options: InlineVlmOptions = options

        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers package required for InlineVlmModel. "
                "Install with: pip install transformers"
            )

        # Load model and processor
        self.processor, self.model = self._load_model()

    def _load_model(self) -> tuple[Any, Any]:
        """Load HuggingFace model and processor.

        Returns:
            Tuple of (processor, model)
        """
        try:
            _log.info(f"Loading model from {self.options.model_path}...")

            # Load processor
            processor = AutoProcessor.from_pretrained(self.options.model_path)

            # Load model with appropriate precision
            import torch

            dtype = torch.float32
            if self.options.precision == "fp16":
                dtype = torch.float16
            elif self.options.precision == "int8":
                # Requires bitsandbytes
                model = AutoModelForVision2Seq.from_pretrained(
                    self.options.model_path,
                    load_in_8bit=True,
                    device_map="auto",
                )
                return processor, model

            model = AutoModelForVision2Seq.from_pretrained(
                self.options.model_path,
                torch_dtype=dtype,
            )

            # Move to device
            model = model.to(self.options.device)
            model.eval()

            _log.info(
                f"Loaded {self.options.model_path} on {self.options.device} "
                f"({self.options.precision})"
            )

            return processor, model

        except Exception as e:
            _log.error(f"Failed to load model: {e}")
            raise

    def predict(self, image: Image.Image, prompt: str) -> VlmResponse:
        """Run VLM prediction locally.

        Args:
            image: PIL Image to analyze
            prompt: Text prompt for the model

        Returns:
            VlmResponse with extracted annotations
        """
        try:
            # Build structured prompt
            full_prompt = f"""{prompt}

Please extract all CAD annotations and return them as a JSON array:
[{{"annotation_type": "dimension|tolerance|note", "text": "...", "value": null, "unit": null}}]
"""

            # Prepare inputs
            inputs = self.processor(
                text=full_prompt,
                images=image,
                return_tensors="pt",
            )

            # Move to device
            import torch

            inputs = {k: v.to(self.options.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.options.max_tokens,
                    temperature=self.options.temperature if self.options.temperature > 0 else None,
                    do_sample=self.options.temperature > 0,
                )

            # Decode
            raw_text = self.processor.batch_decode(
                outputs, skip_special_tokens=True
            )[0]

            # Parse annotations
            annotations = self._parse_json_annotations(raw_text)

            response = VlmResponse(
                annotations=annotations,
                raw_text=raw_text,
                metadata={
                    "model": self.options.model_path,
                    "device": self.options.device,
                },
            )

            # Enhance with OCR if enabled
            if self.options.use_ocr:
                response = self._enhance_with_ocr(image, response)

            return response

        except Exception as e:
            _log.error(f"VLM prediction failed: {e}")
            return VlmResponse(
                annotations=[],
                raw_text="",
                metadata={"error": str(e)},
            )

    def _parse_json_annotations(self, text: str) -> list[VlmAnnotation]:
        """Parse JSON annotations from model response.

        Args:
            text: Raw text response

        Returns:
            List of VlmAnnotation objects
        """
        annotations = []

        try:
            # Try to find JSON array in response
            json_start = text.find("[")
            json_end = text.rfind("]") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                parsed = json.loads(json_str)

                for item in parsed:
                    annotations.append(VlmAnnotation(**item))

            _log.info(f"Parsed {len(annotations)} annotations from JSON")

        except Exception as e:
            _log.error(f"Failed to parse JSON annotations: {e}")

        return annotations
