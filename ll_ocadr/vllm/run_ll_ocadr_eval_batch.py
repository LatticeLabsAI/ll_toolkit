"""
Batch evaluation script for LL-OCADR.
Process multiple CAD/Mesh files and evaluate results.

Example usage:
    python run_ll_ocadr_eval_batch.py \
        --data-dir /path/to/cad/files \
        --output-file results.jsonl
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

import torch
from tqdm import tqdm

from run_ll_ocadr import LLOCADRInference


@dataclass
class EvaluationResult:
    """Single evaluation result."""
    mesh_file: str
    prompt: str
    generated_text: str
    reference_text: Optional[str] = None
    processing_time: float = 0.0
    num_tokens: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class BatchEvaluator:
    """
    Batch evaluator for LL-OCADR.
    Processes multiple mesh files and collects results.
    """

    def __init__(
        self,
        model_path: str,
        model_size: str = "7b",
        device: str = "cuda",
        use_vllm: bool = True
    ):
        """
        Initialize batch evaluator.

        Args:
            model_path: Path to model checkpoint or HuggingFace model ID
            model_size: Model size ("1.8b", "7b", or "14b")
            device: Device to run on
            use_vllm: Whether to use vLLM
        """
        self.inference = LLOCADRInference(
            model_path=model_path,
            model_size=model_size,
            device=device,
            use_vllm=use_vllm
        )
        self.use_vllm = use_vllm
        self.results: List[EvaluationResult] = []

    def find_mesh_files(
        self,
        data_dir: str,
        extensions: Optional[List[str]] = None
    ) -> List[Path]:
        """
        Find all mesh files in directory.

        Args:
            data_dir: Directory to search
            extensions: File extensions to include (default: all supported)

        Returns:
            List of mesh file paths
        """
        if extensions is None:
            extensions = ['.step', '.stp', '.stl', '.obj', '.ply']

        data_path = Path(data_dir)
        mesh_files = []

        for ext in extensions:
            mesh_files.extend(data_path.glob(f"**/*{ext}"))

        return sorted(mesh_files)

    def load_prompts(
        self,
        prompt_file: Optional[str] = None,
        default_prompt: str = "<mesh>\nDescribe this CAD model."
    ) -> Dict[str, str]:
        """
        Load prompts for evaluation.

        Args:
            prompt_file: JSON file mapping filenames to prompts
            default_prompt: Default prompt if no file provided

        Returns:
            Dictionary mapping filename to prompt
        """
        if prompt_file is None:
            return {}

        prompt_path = Path(prompt_file)
        if not prompt_path.exists():
            print(f"⚠ Prompt file not found: {prompt_file}")
            return {}

        with open(prompt_path, 'r') as f:
            prompts = json.load(f)

        return prompts

    def load_references(
        self,
        reference_file: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Load reference texts for evaluation.

        Args:
            reference_file: JSON file mapping filenames to reference texts

        Returns:
            Dictionary mapping filename to reference text
        """
        if reference_file is None:
            return {}

        ref_path = Path(reference_file)
        if not ref_path.exists():
            print(f"⚠ Reference file not found: {reference_file}")
            return {}

        with open(ref_path, 'r') as f:
            references = json.load(f)

        return references

    async def evaluate_async(
        self,
        mesh_files: List[Path],
        prompts: Dict[str, str],
        references: Dict[str, str],
        default_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9
    ):
        """
        Evaluate batch asynchronously using vLLM.

        Args:
            mesh_files: List of mesh file paths
            prompts: Dictionary of per-file prompts
            references: Dictionary of reference texts
            default_prompt: Default prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
        """
        print(f"\n🚀 Starting batch evaluation ({len(mesh_files)} files)")
        print(f"   Using vLLM async inference")

        for mesh_file in tqdm(mesh_files, desc="Processing"):
            filename = mesh_file.name

            # Get prompt for this file
            prompt = prompts.get(filename, default_prompt)
            if "<mesh>" not in prompt:
                prompt = f"<mesh>\n{prompt}"

            # Get reference if available
            reference = references.get(filename)

            # Process file
            start_time = time.time()
            error = None
            generated_text = ""

            try:
                generated_text = await self.inference.generate_async(
                    mesh_file=str(mesh_file),
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p
                )
            except Exception as e:
                error = str(e)
                print(f"\n✗ Error processing {filename}: {error}")

            processing_time = time.time() - start_time

            # Count tokens (approximate)
            num_tokens = len(generated_text.split()) if generated_text else 0

            # Store result
            result = EvaluationResult(
                mesh_file=str(mesh_file),
                prompt=prompt,
                generated_text=generated_text,
                reference_text=reference,
                processing_time=processing_time,
                num_tokens=num_tokens,
                error=error
            )
            self.results.append(result)

    def evaluate_sync(
        self,
        mesh_files: List[Path],
        prompts: Dict[str, str],
        references: Dict[str, str],
        default_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9
    ):
        """
        Evaluate batch synchronously using native PyTorch.

        Args:
            mesh_files: List of mesh file paths
            prompts: Dictionary of per-file prompts
            references: Dictionary of reference texts
            default_prompt: Default prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
        """
        print(f"\n🚀 Starting batch evaluation ({len(mesh_files)} files)")
        print(f"   Using native PyTorch inference")

        for mesh_file in tqdm(mesh_files, desc="Processing"):
            filename = mesh_file.name

            # Get prompt for this file
            prompt = prompts.get(filename, default_prompt)
            if "<mesh>" not in prompt:
                prompt = f"<mesh>\n{prompt}"

            # Get reference if available
            reference = references.get(filename)

            # Process file
            start_time = time.time()
            error = None
            generated_text = ""

            try:
                generated_text = self.inference.generate(
                    mesh_file=str(mesh_file),
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p
                )
            except Exception as e:
                error = str(e)
                print(f"\n✗ Error processing {filename}: {error}")

            processing_time = time.time() - start_time

            # Count tokens (approximate)
            num_tokens = len(generated_text.split()) if generated_text else 0

            # Store result
            result = EvaluationResult(
                mesh_file=str(mesh_file),
                prompt=prompt,
                generated_text=generated_text,
                reference_text=reference,
                processing_time=processing_time,
                num_tokens=num_tokens,
                error=error
            )
            self.results.append(result)

    def save_results(self, output_file: str):
        """
        Save evaluation results to JSONL file.

        Args:
            output_file: Output file path
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for result in self.results:
                f.write(json.dumps(result.to_dict()) + '\n')

        print(f"\n✓ Saved {len(self.results)} results to {output_file}")

    def print_summary(self):
        """Print evaluation summary statistics."""
        if not self.results:
            print("\n⚠ No results to summarize")
            return

        total = len(self.results)
        successful = sum(1 for r in self.results if r.error is None)
        failed = total - successful

        total_time = sum(r.processing_time for r in self.results)
        avg_time = total_time / total if total > 0 else 0

        total_tokens = sum(r.num_tokens for r in self.results)
        avg_tokens = total_tokens / successful if successful > 0 else 0

        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total files:        {total}")
        print(f"Successful:         {successful}")
        print(f"Failed:             {failed}")
        print(f"Total time:         {total_time:.2f}s")
        print(f"Avg time per file:  {avg_time:.2f}s")
        print(f"Total tokens:       {total_tokens}")
        print(f"Avg tokens per file:{avg_tokens:.1f}")

        if failed > 0:
            print("\nFailed files:")
            for result in self.results:
                if result.error:
                    print(f"  ✗ {Path(result.mesh_file).name}: {result.error}")

        print("="*60 + "\n")


async def main_async(args):
    """Main async entry point."""
    # Initialize evaluator
    evaluator = BatchEvaluator(
        model_path=args.model_path,
        model_size=args.model_size,
        device=args.device,
        use_vllm=True
    )

    # Find mesh files
    mesh_files = evaluator.find_mesh_files(
        data_dir=args.data_dir,
        extensions=args.extensions.split(',') if args.extensions else None
    )

    if not mesh_files:
        print(f"✗ No mesh files found in {args.data_dir}")
        return

    print(f"✓ Found {len(mesh_files)} mesh files")

    # Load prompts and references
    prompts = evaluator.load_prompts(args.prompt_file, args.default_prompt)
    references = evaluator.load_references(args.reference_file)

    # Limit number of files if specified
    if args.max_files:
        mesh_files = mesh_files[:args.max_files]
        print(f"✓ Limited to {len(mesh_files)} files (--max-files {args.max_files})")

    # Evaluate
    await evaluator.evaluate_async(
        mesh_files=mesh_files,
        prompts=prompts,
        references=references,
        default_prompt=args.default_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p
    )

    # Save results
    evaluator.save_results(args.output_file)

    # Print summary
    evaluator.print_summary()


def main_sync(args):
    """Main synchronous entry point."""
    # Initialize evaluator
    evaluator = BatchEvaluator(
        model_path=args.model_path,
        model_size=args.model_size,
        device=args.device,
        use_vllm=False
    )

    # Find mesh files
    mesh_files = evaluator.find_mesh_files(
        data_dir=args.data_dir,
        extensions=args.extensions.split(',') if args.extensions else None
    )

    if not mesh_files:
        print(f"✗ No mesh files found in {args.data_dir}")
        return

    print(f"✓ Found {len(mesh_files)} mesh files")

    # Load prompts and references
    prompts = evaluator.load_prompts(args.prompt_file, args.default_prompt)
    references = evaluator.load_references(args.reference_file)

    # Limit number of files if specified
    if args.max_files:
        mesh_files = mesh_files[:args.max_files]
        print(f"✓ Limited to {len(mesh_files)} files (--max-files {args.max_files})")

    # Evaluate
    evaluator.evaluate_sync(
        mesh_files=mesh_files,
        prompts=prompts,
        references=references,
        default_prompt=args.default_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p
    )

    # Save results
    evaluator.save_results(args.output_file)

    # Print summary
    evaluator.print_summary()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch evaluation for LL-OCADR"
    )

    # Required arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing mesh files"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output JSONL file for results"
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
        "--default-prompt",
        type=str,
        default="<mesh>\nDescribe this CAD model and list its key features.",
        help="Default prompt for all files"
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        help="JSON file mapping filenames to prompts"
    )
    parser.add_argument(
        "--reference-file",
        type=str,
        help="JSON file mapping filenames to reference texts"
    )

    # Generation arguments
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
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

    # File filtering
    parser.add_argument(
        "--extensions",
        type=str,
        help="Comma-separated list of extensions (e.g., '.step,.stl')"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to process"
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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Check if vLLM should be used
    from run_ll_ocadr import VLLM_AVAILABLE
    use_vllm = VLLM_AVAILABLE and not args.no_vllm

    if use_vllm:
        asyncio.run(main_async(args))
    else:
        if args.no_vllm:
            print("⚠ Running with native PyTorch (--no-vllm specified)")
        else:
            print("⚠ vLLM not available, falling back to native PyTorch")
        main_sync(args)
