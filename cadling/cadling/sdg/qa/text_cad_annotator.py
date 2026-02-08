"""Text-CAD paired data annotator for multi-level shape descriptions.

This module provides the TextCADAnnotator class that generates Text2CAD-style
multi-level annotations from STEP files by rendering multi-view images,
describing them with a VLM, and producing four annotation levels:

- abstract: High-level shape description (e.g., "two concentric cylinders")
- intermediate: Feature-level description with approximate dimensions
- detailed: Full dimensional specification with tolerances and features
- expert: Complete parametric/coordinate specification

Classes:
    TextCADAnnotator: Generate multi-level text annotations from CAD files

Example:
    from cadling.sdg.qa.text_cad_annotator import TextCADAnnotator

    annotator = TextCADAnnotator(
        vlm_model_name="gpt-4-vision-preview",
        llm_model_name="gpt-4",
        api_provider="openai",
    )
    result = annotator.annotate("part.step", num_views=4)
    print(result["abstract"])
    print(result["expert"])
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from cadling.sdg.qa.base import AnnotationLevel, LlmOptions, LlmProvider
from cadling.sdg.qa.utils import ChatAgent, initialize_llm

_log = logging.getLogger(__name__)

# Standard view configurations for multi-view rendering.
# Each entry is (view_name, azimuth, elevation).
_DEFAULT_VIEWS = [
    ("front", 0.0, 0.0),
    ("right", 90.0, 0.0),
    ("top", 0.0, 90.0),
    ("isometric", 45.0, 35.264),
]

# Additional views when more than 4 are requested
_EXTENDED_VIEWS = [
    ("back", 180.0, 0.0),
    ("left", 270.0, 0.0),
    ("bottom", 0.0, -90.0),
    ("isometric_rear", 225.0, 35.264),
]

# Annotation level descriptions for prompt construction
_LEVEL_DESCRIPTIONS = {
    AnnotationLevel.ABSTRACT: (
        "Provide an ABSTRACT, high-level description of the overall shape. "
        "Use plain language a non-engineer could understand. "
        "Describe the general form, purpose, and visual appearance. "
        "Example: 'two concentric cylinders' or 'L-shaped bracket with holes'."
    ),
    AnnotationLevel.INTERMEDIATE: (
        "Provide an INTERMEDIATE description naming specific CAD features "
        "and approximate dimensions. Reference feature types (fillet, chamfer, "
        "pocket, boss, hole) and relative proportions. "
        "Example: 'a hollow tube with outer diameter ~50mm and four "
        "equally-spaced mounting holes on the top flange'."
    ),
    AnnotationLevel.DETAILED: (
        "Provide a DETAILED engineering description with precise dimensions, "
        "tolerances, material callouts, and feature specifications. Include "
        "all measurable quantities with units and reasonable precision. "
        "Example: 'cylindrical shell, OD=50.0mm, ID=40.0mm, height=100.0mm, "
        "4x M6 through-holes on PCD 35mm, surface finish Ra 1.6'."
    ),
    AnnotationLevel.EXPERT: (
        "Provide an EXPERT-LEVEL description with full parametric specifications, "
        "exact coordinates, B-Rep topology details, construction sequence, "
        "and geometric constraint definitions. Include coordinate systems, "
        "datum references, and all numerical precision available. "
        "Example: 'Manifold solid BRep: outer cylinder axis along Z "
        "(origin 0,0,0), R_outer=25.0mm, R_inner=20.0mm, extruded "
        "0-100mm in +Z. 4x through-holes at (17.5cos(n*pi/2), "
        "17.5sin(n*pi/2), 0) for n=0..3, R=3.0mm, "
        "axis parallel to Z. 12 ADVANCED_FACEs, 24 ORIENTED_EDGEs'."
    ),
}


class TextCADAnnotator:
    """Generate Text2CAD-style multi-level annotations from STEP files.

    Renders multi-view images of a STEP file, feeds them to a Vision-Language
    Model to obtain a shape description, then uses an LLM to generate four
    annotation levels (abstract, intermediate, detailed, expert).

    Attributes:
        vlm_model_name: Model name for VLM inference
        llm_model_name: Model name for text LLM inference
        api_provider: API provider string (maps to LlmProvider)
        llm_agent: ChatAgent for LLM calls
        vlm_client: VLM client instance

    Example:
        annotator = TextCADAnnotator(
            vlm_model_name="gpt-4-vision-preview",
            llm_model_name="gpt-4",
            api_provider="openai",
        )

        # Single file
        result = annotator.annotate("bracket.step", num_views=4)
        for level in ["abstract", "intermediate", "detailed", "expert"]:
            print(f"{level}: {result[level]}")

        # Batch
        results = annotator.annotate_batch(["part1.step", "part2.step"])
    """

    def __init__(
        self,
        vlm_model_name: str | None = None,
        llm_model_name: str = "gpt-4",
        api_provider: str = "openai",
    ):
        """Initialize TextCADAnnotator.

        Args:
            vlm_model_name: VLM model identifier for image analysis.
                Defaults to 'gpt-4-vision-preview' for OpenAI or
                'claude-3-opus-20240229' for Anthropic. Set to None
                to auto-select based on provider.
            llm_model_name: LLM model identifier for text generation.
            api_provider: Provider string ('openai', 'anthropic', etc.).
        """
        self.vlm_model_name = vlm_model_name
        self.llm_model_name = llm_model_name
        self.api_provider = api_provider

        # Map string provider to enum
        self._provider = self._resolve_provider(api_provider)

        # Auto-select VLM model if not specified
        if self.vlm_model_name is None:
            self.vlm_model_name = self._default_vlm_model(self._provider)

        # Initialize LLM agent for text generation
        llm_options = LlmOptions(
            provider=self._provider,
            model_id=self.llm_model_name,
            temperature=0.3,
            max_tokens=2048,
        )
        self.llm_agent = ChatAgent.from_options(llm_options)

        # Initialize VLM client (lazy -- constructed on first use)
        self._vlm_client: Any = None
        self._temp_dir: str | None = None

        _log.info(
            f"Initialized TextCADAnnotator (vlm={self.vlm_model_name}, "
            f"llm={self.llm_model_name}, provider={self.api_provider})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate(
        self,
        step_file_path: str,
        num_views: int = 4,
    ) -> dict:
        """Generate multi-level annotations for a STEP file.

        Pipeline:
          1. Render *num_views* images of the model.
          2. Send images to VLM to obtain a visual description.
          3. Use LLM to expand the description into four annotation levels.
          4. Return a dictionary with all levels and metadata.

        Args:
            step_file_path: Path to the STEP file to annotate.
            num_views: Number of views to render (1-8, default 4).

        Returns:
            Dictionary with keys:
                - 'abstract': High-level shape description
                - 'intermediate': Feature-level description
                - 'detailed': Full dimensional specification
                - 'expert': Parametric/coordinate specification
                - 'vlm_description': Raw VLM description
                - 'metadata': Dict with source file, model info, timing

        Raises:
            FileNotFoundError: If STEP file does not exist.
            RuntimeError: If rendering or VLM inference fails.
        """
        step_path = Path(step_file_path)
        if not step_path.exists():
            raise FileNotFoundError(f"STEP file not found: {step_path}")

        start_time = time.time()
        annotation_id = str(uuid.uuid4())

        _log.info(f"Annotating '{step_path.name}' with {num_views} views")

        # Step 1: Render multi-view images
        image_paths = self._render_views(str(step_path), num_views)
        _log.info(f"Rendered {len(image_paths)} views for '{step_path.name}'")

        # Step 2: Get VLM description from rendered images
        vlm_description = self._describe_with_vlm(image_paths)
        _log.info(
            f"VLM description ({len(vlm_description)} chars): "
            f"{vlm_description[:120]}..."
        )

        # Step 3: Generate four annotation levels via LLM
        annotations = self._generate_annotations(vlm_description)

        elapsed = time.time() - start_time

        # Step 4: Compose result with metadata
        result = {
            AnnotationLevel.ABSTRACT.value: annotations.get(
                AnnotationLevel.ABSTRACT.value, ""
            ),
            AnnotationLevel.INTERMEDIATE.value: annotations.get(
                AnnotationLevel.INTERMEDIATE.value, ""
            ),
            AnnotationLevel.DETAILED.value: annotations.get(
                AnnotationLevel.DETAILED.value, ""
            ),
            AnnotationLevel.EXPERT.value: annotations.get(
                AnnotationLevel.EXPERT.value, ""
            ),
            "vlm_description": vlm_description,
            "metadata": {
                "annotation_id": annotation_id,
                "source_file": str(step_path),
                "source_name": step_path.name,
                "num_views": len(image_paths),
                "view_names": [
                    Path(p).stem.split("_", 1)[-1]
                    if "_" in Path(p).stem
                    else Path(p).stem
                    for p in image_paths
                ],
                "vlm_model": self.vlm_model_name,
                "llm_model": self.llm_model_name,
                "provider": self.api_provider,
                "time_seconds": round(elapsed, 3),
            },
        }

        # Clean up temporary rendered images
        self._cleanup_temp_images(image_paths)

        _log.info(
            f"Annotation complete for '{step_path.name}' in {elapsed:.2f}s"
        )

        return result

    def annotate_batch(
        self,
        step_files: list[str],
        num_views: int = 4,
    ) -> list[dict]:
        """Batch-annotate multiple STEP files.

        Processes each file sequentially, collecting results. Errors on
        individual files are logged and captured in the result metadata
        rather than raising.

        Args:
            step_files: List of paths to STEP files.
            num_views: Number of views per file (1-8, default 4).

        Returns:
            List of annotation dictionaries (same structure as annotate()).
            Failed files produce a dict with empty annotation levels and
            an 'error' key in metadata.
        """
        results: list[dict] = []
        total = len(step_files)

        _log.info(f"Starting batch annotation of {total} STEP files")
        batch_start = time.time()

        for idx, step_file in enumerate(step_files):
            _log.info(f"[{idx + 1}/{total}] Annotating '{step_file}'")

            try:
                result = self.annotate(step_file, num_views=num_views)
                results.append(result)

            except Exception as e:
                error_msg = f"Failed to annotate '{step_file}': {e}"
                _log.error(error_msg)
                results.append({
                    AnnotationLevel.ABSTRACT.value: "",
                    AnnotationLevel.INTERMEDIATE.value: "",
                    AnnotationLevel.DETAILED.value: "",
                    AnnotationLevel.EXPERT.value: "",
                    "vlm_description": "",
                    "metadata": {
                        "annotation_id": str(uuid.uuid4()),
                        "source_file": step_file,
                        "source_name": Path(step_file).name,
                        "error": str(e),
                        "vlm_model": self.vlm_model_name,
                        "llm_model": self.llm_model_name,
                        "provider": self.api_provider,
                    },
                })

        batch_elapsed = time.time() - batch_start
        num_ok = sum(
            1 for r in results if "error" not in r.get("metadata", {})
        )

        _log.info(
            f"Batch annotation complete: {num_ok}/{total} succeeded, "
            f"{batch_elapsed:.2f}s total"
        )

        return results

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_views(
        self,
        step_path: str,
        num_views: int,
    ) -> list[str]:
        """Render multiple views of a STEP file to temporary image files.

        Uses the cadling STEP backend and pythonocc-core for rendering.
        Falls back to trimesh-based rendering if pythonocc is unavailable.

        Args:
            step_path: Absolute path to the STEP file.
            num_views: Number of views to render (clamped to 1-8).

        Returns:
            List of absolute paths to rendered PNG images.

        Raises:
            RuntimeError: If no rendering backend is available.
        """
        max_views = len(_DEFAULT_VIEWS) + len(_EXTENDED_VIEWS)
        num_views = max(1, min(num_views, max_views))

        # Select view configs
        view_configs = list(_DEFAULT_VIEWS[:num_views])
        if num_views > len(_DEFAULT_VIEWS):
            extra = num_views - len(_DEFAULT_VIEWS)
            view_configs.extend(_EXTENDED_VIEWS[:extra])

        # Create temp directory for rendered images
        self._temp_dir = tempfile.mkdtemp(prefix="textcad_views_")
        rendered_paths: list[str] = []

        # Try pythonocc-based rendering via STEP backend
        try:
            rendered_paths = self._render_with_step_backend(
                step_path, view_configs
            )
            if rendered_paths:
                return rendered_paths
        except Exception as e:
            _log.warning(f"pythonocc rendering failed: {e}")

        # Fallback: trimesh-based rendering
        try:
            rendered_paths = self._render_with_trimesh(
                step_path, view_configs
            )
            if rendered_paths:
                return rendered_paths
        except Exception as e:
            _log.warning(f"trimesh rendering failed: {e}")

        # Fallback: generate placeholder images with shape info
        _log.warning(
            "No rendering backend available, generating placeholder views"
        )
        rendered_paths = self._render_placeholder_views(
            step_path, view_configs
        )

        return rendered_paths

    def _render_with_step_backend(
        self,
        step_path: str,
        view_configs: list[tuple[str, float, float]],
    ) -> list[str]:
        """Render views using the cadling STEP backend.

        Args:
            step_path: Path to STEP file.
            view_configs: List of (name, azimuth, elevation) tuples.

        Returns:
            List of rendered image paths.
        """
        import hashlib

        from cadling.backend.step.step_backend import STEPBackend
        from cadling.datamodel.base_models import CADInputDocument, InputFormat

        # Build the input document descriptor required by STEPBackend
        step_file = Path(step_path)
        with open(step_file, "rb") as fh:
            doc_hash = hashlib.sha256(fh.read()).hexdigest()
        in_doc = CADInputDocument(
            file=step_file,
            format=InputFormat.STEP,
            document_hash=doc_hash,
        )
        backend = STEPBackend(in_doc=in_doc, path_or_stream=step_path)

        rendered_paths: list[str] = []
        for view_name, _azimuth, _elevation in view_configs:
            try:
                image = backend.render_view(view_name, resolution=1024)
                assert self._temp_dir is not None
                out_path = os.path.join(
                    self._temp_dir, f"view_{view_name}.png"
                )
                image.save(out_path)
                rendered_paths.append(out_path)
                _log.debug(f"Rendered view '{view_name}' via STEP backend")
            except Exception as e:
                _log.warning(
                    f"Failed to render view '{view_name}' via STEP backend: {e}"
                )

        return rendered_paths

    def _render_with_trimesh(
        self,
        step_path: str,
        view_configs: list[tuple[str, float, float]],
    ) -> list[str]:
        """Render views using trimesh (requires STEP -> mesh conversion).

        Args:
            step_path: Path to STEP file.
            view_configs: List of (name, azimuth, elevation) tuples.

        Returns:
            List of rendered image paths.
        """
        try:
            import numpy as np
            import trimesh
        except ImportError as e:
            raise RuntimeError(
                "trimesh not available. Install with: pip install trimesh"
            ) from e

        # Load STEP via trimesh (uses gmsh/assimp underneath)
        mesh = trimesh.load(step_path)

        # Ensure it is a single Trimesh (merge if scene)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if not isinstance(mesh, trimesh.Trimesh):
            raise RuntimeError(
                f"Expected trimesh.Trimesh, got {type(mesh).__name__}"
            )

        rendered_paths: list[str] = []
        for view_name, azimuth, elevation in view_configs:
            try:
                # Compute camera transform from azimuth/elevation
                az_rad = np.radians(azimuth)
                el_rad = np.radians(elevation)

                distance = mesh.bounding_sphere.primitive.radius * 3.0
                cam_x = distance * np.cos(el_rad) * np.sin(az_rad)
                cam_y = distance * np.cos(el_rad) * np.cos(az_rad)
                cam_z = distance * np.sin(el_rad)

                scene = trimesh.Scene(mesh)
                scene.camera.resolution = [1024, 1024]

                # Set camera transform
                camera_target = mesh.centroid
                camera_position = camera_target + np.array(
                    [cam_x, cam_y, cam_z]
                )

                camera_transform = np.eye(4)
                camera_transform[:3, 3] = camera_position

                # Look-at direction
                forward = camera_target - camera_position
                forward = forward / (np.linalg.norm(forward) + 1e-8)

                # Choose an up vector that is not parallel to forward
                up_candidate = np.array([0.0, 0.0, 1.0])
                if abs(np.dot(forward, up_candidate)) > 0.99:
                    up_candidate = np.array([0.0, 1.0, 0.0])

                right = np.cross(forward, up_candidate)
                right = right / (np.linalg.norm(right) + 1e-8)
                up = np.cross(right, forward)

                rotation = np.eye(4)
                rotation[:3, 0] = right
                rotation[:3, 1] = up
                rotation[:3, 2] = -forward
                rotation[:3, 3] = camera_position

                scene.camera_transform = rotation

                # Render to bytes then save
                png_data = scene.save_image(resolution=[1024, 1024])
                assert self._temp_dir is not None
                out_path = os.path.join(
                    self._temp_dir, f"view_{view_name}.png"
                )
                with open(out_path, "wb") as f:
                    f.write(png_data)
                rendered_paths.append(out_path)

                _log.debug(f"Rendered view '{view_name}' via trimesh")

            except Exception as e:
                _log.warning(
                    f"Failed to render view '{view_name}' via trimesh: {e}"
                )

        return rendered_paths

    def _render_placeholder_views(
        self,
        step_path: str,
        view_configs: list[tuple[str, float, float]],
    ) -> list[str]:
        """Generate placeholder images annotated with view info.

        Used when no rendering backend is available. Creates simple
        text images so the VLM still receives valid image inputs
        (though quality of annotation will be limited).

        Args:
            step_path: Path to STEP file.
            view_configs: List of (name, azimuth, elevation) tuples.

        Returns:
            List of placeholder image paths.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as e:
            raise RuntimeError(
                "Pillow not available. Install with: pip install Pillow"
            ) from e

        # Extract basic STEP info for the placeholder
        step_info = self._extract_step_text_info(step_path)

        rendered_paths: list[str] = []
        for view_name, azimuth, elevation in view_configs:
            img = Image.new("RGB", (1024, 1024), color=(240, 240, 240))
            draw = ImageDraw.Draw(img)

            # Draw view label and file info
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24
                )
                font_small = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16
                )
            except OSError:
                font = ImageFont.load_default()
                font_small = font

            draw.text(
                (50, 50),
                f"View: {view_name}",
                fill=(0, 0, 0),
                font=font,
            )
            draw.text(
                (50, 90),
                f"Azimuth: {azimuth:.1f}, Elevation: {elevation:.1f}",
                fill=(80, 80, 80),
                font=font_small,
            )
            draw.text(
                (50, 130),
                f"File: {Path(step_path).name}",
                fill=(80, 80, 80),
                font=font_small,
            )

            # Add STEP entity summary
            y_offset = 180
            for line in step_info[:15]:
                draw.text(
                    (50, y_offset),
                    line[:90],
                    fill=(60, 60, 60),
                    font=font_small,
                )
                y_offset += 22

            draw.text(
                (50, 900),
                "[Placeholder - no rendering backend available]",
                fill=(180, 0, 0),
                font=font_small,
            )

            assert self._temp_dir is not None
            out_path = os.path.join(
                self._temp_dir, f"view_{view_name}.png"
            )
            img.save(out_path)
            rendered_paths.append(out_path)

        return rendered_paths

    def _extract_step_text_info(self, step_path: str) -> list[str]:
        """Extract basic text information from a STEP file for placeholders.

        Reads the STEP file header and extracts entity type counts
        to give the VLM some context even without rendered images.

        Args:
            step_path: Path to the STEP file.

        Returns:
            List of descriptive lines about the STEP file contents.
        """
        info_lines: list[str] = []

        try:
            with open(step_path, errors="ignore") as f:
                content = f.read(50000)  # Read first 50KB

            # Count entity types
            entity_pattern = re.compile(r"#\d+\s*=\s*([A-Z_]+)\(")
            entity_counts: dict[str, int] = {}
            for match in entity_pattern.finditer(content):
                etype = match.group(1)
                entity_counts[etype] = entity_counts.get(etype, 0) + 1

            # Build summary lines
            info_lines.append(
                f"STEP entities found: {sum(entity_counts.values())}"
            )

            # Sort by count descending
            sorted_entities = sorted(
                entity_counts.items(), key=lambda x: x[1], reverse=True
            )
            for etype, count in sorted_entities[:12]:
                info_lines.append(f"  {etype}: {count}")

            # Extract description from header if present
            desc_match = re.search(
                r"FILE_DESCRIPTION\s*\(\s*\(\s*'([^']*)'\s*\)", content
            )
            if desc_match:
                info_lines.insert(0, f"Description: {desc_match.group(1)}")

            name_match = re.search(
                r"FILE_NAME\s*\(\s*'([^']*)'", content
            )
            if name_match:
                info_lines.insert(0, f"Name: {name_match.group(1)}")

        except Exception as e:
            info_lines.append(f"Could not parse STEP file: {e}")

        return info_lines

    # ------------------------------------------------------------------
    # VLM description
    # ------------------------------------------------------------------

    def _describe_with_vlm(self, image_paths: list[str]) -> str:
        """Send rendered view images to VLM and get a shape description.

        Constructs a multi-image prompt asking the VLM to describe the
        CAD model shown from multiple angles.

        Args:
            image_paths: List of paths to rendered view images.

        Returns:
            Textual description of the CAD model from the VLM.

        Raises:
            RuntimeError: If VLM inference fails.
        """
        if not image_paths:
            return "No views available for VLM description."

        vlm_client = self._get_vlm_client()

        prompt = (
            "You are an expert CAD/mechanical engineer. The following images "
            "show multiple views of the same 3D CAD model rendered from "
            "different angles.\n\n"
            "Provide a comprehensive description of this part including:\n"
            "1. Overall shape and general form\n"
            "2. Specific geometric features (holes, fillets, chamfers, "
            "pockets, bosses, ribs)\n"
            "3. Approximate dimensions and proportions\n"
            "4. Topology (number of faces, edges, symmetry)\n"
            "5. Likely manufacturing method and material\n"
            "6. Functional purpose (if apparent)\n\n"
            "Be as specific and technical as possible. Reference features "
            "visible in specific views when helpful."
        )

        assert self.vlm_model_name is not None  # Set in __init__
        model_lower = self.vlm_model_name.lower()

        try:
            if (
                "gpt" in model_lower
                or "vision" in model_lower
                or "o1" in model_lower
            ):
                description = self._vlm_call_openai(
                    vlm_client, prompt, image_paths
                )
            elif "claude" in model_lower:
                description = self._vlm_call_anthropic(
                    vlm_client, prompt, image_paths
                )
            else:
                # Default to OpenAI-compatible API
                description = self._vlm_call_openai(
                    vlm_client, prompt, image_paths
                )

            return description.strip()

        except Exception as e:
            _log.error(f"VLM description failed: {e}")
            raise RuntimeError(f"VLM inference failed: {e}") from e

    def _vlm_call_openai(
        self,
        client: Any,
        prompt: str,
        image_paths: list[str],
    ) -> str:
        """Make VLM call via OpenAI API (GPT-4V or compatible).

        Args:
            client: OpenAI client instance.
            prompt: Text prompt.
            image_paths: List of image file paths.

        Returns:
            Model response text.
        """
        import base64

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        for img_path in image_paths:
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_b64}",
                    "detail": "high",
                },
            })

        response = client.chat.completions.create(
            model=self.vlm_model_name,
            messages=[{"role": "user", "content": content}],
            temperature=0.2,
            max_tokens=2048,
        )

        return response.choices[0].message.content or ""

    def _vlm_call_anthropic(
        self,
        client: Any,
        prompt: str,
        image_paths: list[str],
    ) -> str:
        """Make VLM call via Anthropic API (Claude 3+).

        Args:
            client: Anthropic client instance.
            prompt: Text prompt.
            image_paths: List of image file paths.

        Returns:
            Model response text.
        """
        import base64

        content: list[dict[str, Any]] = []

        for img_path in image_paths:
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_b64,
                },
            })

        content.append({"type": "text", "text": prompt})

        response = client.messages.create(
            model=self.vlm_model_name,
            max_tokens=2048,
            temperature=0.2,
            messages=[{"role": "user", "content": content}],
        )

        return response.content[0].text

    # ------------------------------------------------------------------
    # Annotation generation
    # ------------------------------------------------------------------

    def _generate_annotations(self, description: str) -> dict:
        """Generate four annotation levels from a VLM description.

        Uses the LLM to transform the raw VLM description into
        structured annotations at each level of detail.

        Args:
            description: VLM-generated description of the CAD model.

        Returns:
            Dictionary mapping level names to annotation strings.
        """
        annotations: dict[str, str] = {}

        for level in AnnotationLevel:
            try:
                prompt = self._build_annotation_prompt(description, level)
                raw_response = self.llm_agent.ask(prompt, max_tokens=1024)
                annotation_text = self._postprocess_annotation(
                    raw_response, level
                )
                annotations[level.value] = annotation_text

                _log.debug(
                    f"Generated {level.value} annotation "
                    f"({len(annotation_text)} chars)"
                )

            except Exception as e:
                _log.error(
                    f"Failed to generate {level.value} annotation: {e}"
                )
                annotations[level.value] = ""

        return annotations

    def _build_annotation_prompt(
        self,
        description: str,
        level: AnnotationLevel,
    ) -> str:
        """Build the LLM prompt for generating a specific annotation level.

        Args:
            description: Raw VLM description of the model.
            level: Annotation level to generate.

        Returns:
            Formatted prompt string.
        """
        level_instruction = _LEVEL_DESCRIPTIONS.get(level, "")

        prompt = (
            "You are an expert CAD engineer creating structured text "
            "annotations for a Text2CAD training dataset. Given the "
            "following description of a 3D CAD model (generated from "
            "multi-view analysis), produce a single text annotation at "
            "the specified level.\n\n"
            f"=== Model Description ===\n{description}\n\n"
            f"=== Annotation Level: {level.value.upper()} ===\n"
            f"{level_instruction}\n\n"
            "IMPORTANT RULES:\n"
            "- Output ONLY the annotation text, no preamble or labels.\n"
            "- Keep the annotation as a single coherent paragraph or "
            "short set of specification lines.\n"
            "- Be factually consistent with the description provided.\n"
            "- For dimensions or values not explicitly stated in the "
            "description, infer reasonable engineering values and note "
            "them as approximate (~) where appropriate.\n\n"
            "Annotation:"
        )

        return prompt

    def _postprocess_annotation(
        self,
        raw_text: str,
        level: AnnotationLevel,
    ) -> str:
        """Clean and validate a generated annotation.

        Args:
            raw_text: Raw LLM response.
            level: Annotation level.

        Returns:
            Cleaned annotation string.
        """
        text = raw_text.strip()

        # Remove common LLM preambles
        prefixes_to_remove = [
            "Annotation:",
            f"{level.value.upper()}:",
            f"{level.value.capitalize()}:",
            "Here is the annotation:",
            "Here's the annotation:",
            f"Here is the {level.value} annotation:",
        ]
        for prefix in prefixes_to_remove:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        # Remove surrounding quotes if present
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1].strip()

        return text

    # ------------------------------------------------------------------
    # Client initialization helpers
    # ------------------------------------------------------------------

    def _get_vlm_client(self) -> Any:
        """Get or lazily initialize the VLM API client.

        Returns:
            API client instance suitable for VLM calls.
        """
        if self._vlm_client is not None:
            return self._vlm_client

        assert self.vlm_model_name is not None  # Set in __init__
        vlm_options = LlmOptions(
            provider=self._provider,
            model_id=self.vlm_model_name,
            temperature=0.2,
            max_tokens=2048,
        )
        self._vlm_client = initialize_llm(vlm_options)

        _log.info(f"Initialized VLM client: {self.vlm_model_name}")
        return self._vlm_client

    @staticmethod
    def _resolve_provider(api_provider: str) -> LlmProvider:
        """Resolve a provider string to LlmProvider enum.

        Args:
            api_provider: Provider name string.

        Returns:
            LlmProvider enum value.

        Raises:
            ValueError: If provider string is not recognized.
        """
        provider_map = {
            "openai": LlmProvider.OPENAI,
            "anthropic": LlmProvider.ANTHROPIC,
            "vllm": LlmProvider.VLLM,
            "ollama": LlmProvider.OLLAMA,
            "openai_compatible": LlmProvider.OPENAI_COMPATIBLE,
        }

        provider = provider_map.get(api_provider.lower())
        if provider is None:
            raise ValueError(
                f"Unknown API provider: '{api_provider}'. "
                f"Supported: {list(provider_map.keys())}"
            )
        return provider

    @staticmethod
    def _default_vlm_model(provider: LlmProvider) -> str:
        """Return the default VLM model name for a given provider.

        Args:
            provider: LLM provider.

        Returns:
            Default model identifier string.
        """
        defaults = {
            LlmProvider.OPENAI: "gpt-4-vision-preview",
            LlmProvider.ANTHROPIC: "claude-3-opus-20240229",
            LlmProvider.VLLM: "llava-hf/llava-1.5-13b-hf",
            LlmProvider.OLLAMA: "llava:13b",
            LlmProvider.OPENAI_COMPATIBLE: "gpt-4-vision-preview",
        }
        return defaults.get(provider, "gpt-4-vision-preview")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup_temp_images(self, image_paths: list[str]) -> None:
        """Remove temporary rendered image files.

        Args:
            image_paths: List of temporary image file paths to remove.
        """
        for path in image_paths:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError as e:
                _log.debug(f"Failed to remove temp image {path}: {e}")

        # Remove temp directory if empty
        if self._temp_dir and os.path.isdir(self._temp_dir):
            try:
                os.rmdir(self._temp_dir)
            except OSError:
                pass  # Directory not empty or already removed
