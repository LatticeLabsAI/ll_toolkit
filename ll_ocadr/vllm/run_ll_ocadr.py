"""
Run LL-OCADR inference on CAD/Mesh files.
Supports STEP, STL, OBJ, and PLY formats.

Example usage:
    python run_ll_ocadr.py --mesh-file example.step \
        --prompt "<mesh>\nDescribe this CAD model."
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

try:
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available. Install with: pip install vllm")

from latticelabs_ocadr import (
    LatticelabsOCADRForCausalLM,
    LLOCADRMultiModalProcessor
)
from config import LLOCADRConfig, get_config_for_model
from process.mesh_process import MeshLoader


class LLOCADRInference:
    """
    Inference wrapper for LL-OCADR model.
    Handles model loading, preprocessing, and generation.
    """

    def __init__(
        self,
        model_path: str,
        model_size: str = "7b",
        device: str = "cuda",
        use_vllm: bool = True
    ):
        """
        Initialize LL-OCADR inference engine.

        Args:
            model_path: Path to model checkpoint or HuggingFace model ID
            model_size: Model size ("1.8b", "7b", or "14b")
            device: Device to run on ("cuda", "mps", or "cpu")
            use_vllm: Whether to use vLLM for inference
        """
        self.model_path = model_path
        self.model_size = model_size
        self.device = device
        self.use_vllm = use_vllm and VLLM_AVAILABLE

        # Load configuration
        self.config = get_config_for_model(model_size)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.language_model_name,
            trust_remote_code=True
        )

        # Add mesh token if not present
        if self.config.mesh_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([self.config.mesh_token])
            self.config.mesh_token_id = self.tokenizer.convert_tokens_to_ids(
                self.config.mesh_token
            )
        else:
            self.config.mesh_token_id = self.tokenizer.convert_tokens_to_ids(
                self.config.mesh_token
            )

        # Initialize model
        if self.use_vllm:
            self.engine = None  # Lazy initialization
        else:
            self.model = self._load_model()
            self.model.to(self.device)
            self.model.eval()

        # Initialize mesh loader
        self.mesh_loader = MeshLoader()

        print(f"✓ Initialized LL-OCADR ({model_size}) on {device}")

    def _load_model(self) -> LatticelabsOCADRForCausalLM:
        """Load model from checkpoint."""
        model = LatticelabsOCADRForCausalLM(self.config)

        # Load checkpoint if provided
        if Path(self.model_path).exists():
            checkpoint = torch.load(
                self.model_path,
                map_location=self.device
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"✓ Loaded checkpoint from {self.model_path}")
        else:
            print(f"⚠ No checkpoint found at {self.model_path}, using random weights")

        return model

    async def _init_vllm_engine(self):
        """Initialize vLLM engine (lazy)."""
        if self.engine is None:
            engine_args = AsyncEngineArgs(
                model=self.model_path,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
                trust_remote_code=True,
                dtype="float16" if self.device == "cuda" else "float32"
            )
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            print("✓ Initialized vLLM engine")

    def validate_mesh_file(self, mesh_file: str) -> bool:
        """
        Validate mesh file exists and has supported extension.

        Args:
            mesh_file: Path to mesh file

        Returns:
            True if valid, False otherwise
        """
        path = Path(mesh_file)

        if not path.exists():
            print(f"✗ File not found: {mesh_file}")
            return False

        supported_extensions = {'.step', '.stp', '.stl', '.obj', '.ply'}
        if path.suffix.lower() not in supported_extensions:
            print(f"✗ Unsupported file format: {path.suffix}")
            print(f"  Supported: {', '.join(supported_extensions)}")
            return False

        return True

    async def generate_async(
        self,
        mesh_file: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text asynchronously using vLLM.

        Args:
            mesh_file: Path to mesh file
            prompt: Text prompt with <mesh> placeholder
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter

        Returns:
            Generated text
        """
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM not available. Install with: pip install vllm")

        # Initialize engine
        await self._init_vllm_engine()

        # Validate mesh file
        if not self.validate_mesh_file(mesh_file):
            return ""

        # Prepare sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )

        # Generate
        print(f"\n🔄 Processing mesh: {Path(mesh_file).name}")
        print(f"📝 Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"📝 Prompt: {prompt}")

        outputs = []
        async for output in self.engine.generate(
            prompt=prompt,
            multi_modal_data={"mesh": [mesh_file]},
            sampling_params=sampling_params
        ):
            outputs.append(output)

        result = outputs[-1].text if outputs else ""
        return result

    @torch.no_grad()
    def generate(
        self,
        mesh_file: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text using native PyTorch model (non-vLLM).

        Args:
            mesh_file: Path to mesh file
            prompt: Text prompt with <mesh> placeholder
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter

        Returns:
            Generated text
        """
        # Validate mesh file
        if not self.validate_mesh_file(mesh_file):
            return ""

        print(f"\n🔄 Processing mesh: {Path(mesh_file).name}")
        print(f"📝 Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"📝 Prompt: {prompt}")

        # Preprocess mesh
        from process.mesh_process import LLOCADRProcessor

        processor = LLOCADRProcessor(
            tokenizer=self.tokenizer,
            mesh_token_id=self.config.mesh_token_id,
            min_chunk_size=self.config.min_chunk_size,
            max_chunks=self.config.max_chunks,
            target_global_faces=self.config.target_global_faces
        )

        inputs = processor.tokenize_with_meshes(
            mesh_files=[mesh_file],
            conversation=prompt,
            cropping=True
        )

        # Move to device
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device)

        # Generate
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0
        )

        # Decode
        output_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        # Remove prompt from output
        if prompt in output_text:
            output_text = output_text.replace(prompt, "").strip()

        return output_text


async def main_async(args):
    """Main async entry point for vLLM inference."""
    # Initialize inference engine
    inference = LLOCADRInference(
        model_path=args.model_path,
        model_size=args.model_size,
        device=args.device,
        use_vllm=True
    )

    # Prepare prompt
    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompt = f.read().strip()
    else:
        prompt = args.prompt

    # Ensure prompt has mesh placeholder
    if "<mesh>" not in prompt:
        prompt = f"<mesh>\n{prompt}"

    # Generate
    result = await inference.generate_async(
        mesh_file=args.mesh_file,
        prompt=prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p
    )

    # Display result
    print("\n" + "="*60)
    print("RESULT:")
    print("="*60)
    print(result)
    print("="*60 + "\n")

    # Save to file if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(result)
        print(f"✓ Saved to {args.output_file}")


def main_sync(args):
    """Main synchronous entry point for native PyTorch inference."""
    # Initialize inference engine
    inference = LLOCADRInference(
        model_path=args.model_path,
        model_size=args.model_size,
        device=args.device,
        use_vllm=False
    )

    # Prepare prompt
    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompt = f.read().strip()
    else:
        prompt = args.prompt

    # Ensure prompt has mesh placeholder
    if "<mesh>" not in prompt:
        prompt = f"<mesh>\n{prompt}"

    # Generate
    result = inference.generate(
        mesh_file=args.mesh_file,
        prompt=prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p
    )

    # Display result
    print("\n" + "="*60)
    print("RESULT:")
    print("="*60)
    print(result)
    print("="*60 + "\n")

    # Save to file if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(result)
        print(f"✓ Saved to {args.output_file}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run LL-OCADR inference on CAD/Mesh files"
    )

    # Required arguments
    parser.add_argument(
        "--mesh-file",
        type=str,
        required=True,
        help="Path to mesh file (STEP, STL, OBJ, PLY)"
    )

    # Model arguments
    parser.add_argument(
        "--model-path",
        type=str,
        default="latticelabs/ll-ocadr-7b",
        help="Path to model checkpoint or HuggingFace model ID"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="7b",
        choices=["1.8b", "7b", "14b"],
        help="Model size"
    )

    # Prompt arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default="<mesh>\nDescribe this CAD model and list its key features.",
        help="Text prompt (use <mesh> as placeholder)"
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        help="Read prompt from file (overrides --prompt)"
    )

    # Generation arguments
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0 = greedy)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter"
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        choices=["cuda", "mps", "cpu"],
        help="Device to run on"
    )
    parser.add_argument(
        "--no-vllm",
        action="store_true",
        help="Disable vLLM (use native PyTorch)"
    )

    # Output arguments
    parser.add_argument(
        "--output-file",
        type=str,
        help="Save result to file"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Check if vLLM should be used
    use_vllm = VLLM_AVAILABLE and not args.no_vllm

    if use_vllm:
        asyncio.run(main_async(args))
    else:
        if args.no_vllm:
            print("⚠ Running with native PyTorch (--no-vllm specified)")
        else:
            print("⚠ vLLM not available, falling back to native PyTorch")
        main_sync(args)
