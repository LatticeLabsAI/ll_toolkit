"""
Test script for LL-OCADR pipeline.
Demonstrates file content chunking and processing.
"""

import sys
from pathlib import Path

# Add vllm directory to path
sys.path.insert(0, str(Path(__file__).parent / 'vllm' / 'process'))

from file_content_chunker import UnifiedCADContentChunker


def test_file_chunker(file_path: str):
    """Test the file content chunker."""
    print(f"\n{'='*60}")
    print(f"Testing LL-OCADR File Content Chunker")
    print(f"{'='*60}\n")

    # Check if file exists
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File not found: {file_path}")
        print(f"\nPlease provide a CAD/Mesh file (STEP, STL, or OBJ)")
        return False

    print(f"📁 File: {path.name}")
    print(f"📊 Size: {path.stat().st_size / 1024:.2f} KB")
    print(f"🔧 Format: {path.suffix.upper()}\n")

    try:
        # Initialize chunker with dynamic chunking
        chunker = UnifiedCADContentChunker()  # chunk_size=None enables dynamic analysis

        print("⚙️  Analyzing file...")
        analysis = chunker.analyze_file(file_path)

        print(f"\n📊 File Analysis:")
        print(f"   Total {analysis['entity_type']}s: {analysis['total_entities']}")
        print(f"   Complexity: {analysis['complexity']}")
        print(f"   Optimal chunk size: {analysis['chunk_size']} {analysis['entity_type']}s")
        print(f"   Estimated chunks: {analysis['num_chunks']}")
        print(f"   Est. tokens/chunk: ~{analysis['tokens_per_chunk_est']}")

        print("\n⚙️  Chunking file content...")
        chunks = chunker.chunk_file(file_path)

        # Get statistics
        stats = chunker.get_chunk_statistics(chunks)

        print(f"\n✅ Chunking complete!\n")
        print(f"{'='*60}")
        print(f"CHUNK STATISTICS")
        print(f"{'='*60}")
        print(f"Number of chunks:     {stats['num_chunks']}")
        print(f"Format:               {stats['format']}")
        print(f"Total content size:   {stats['total_content_size']:,} bytes")
        print(f"Avg chunk size:       {stats['avg_chunk_size']:.2f} bytes")

        if 'total_facets' in stats:
            print(f"Total facets:         {stats['total_facets']:,}")
        elif 'total_entities' in stats:
            print(f"Total entities:       {stats['total_entities']:,}")
        elif 'total_faces' in stats:
            print(f"Total faces:          {stats['total_faces']:,}")

        print(f"{'='*60}\n")

        # Show first chunk details
        if chunks:
            first_chunk = chunks[0]
            print(f"{'='*60}")
            print(f"FIRST CHUNK PREVIEW")
            print(f"{'='*60}")
            print(f"Format:               {first_chunk['format']}")

            if 'start_facet' in first_chunk:
                print(f"Facet range:          {first_chunk['start_facet']} - {first_chunk['end_facet']}")
                print(f"Number of facets:     {len(first_chunk['facets'])}")
            elif 'start_entity' in first_chunk:
                print(f"Entity range:         {first_chunk['start_entity']} - {first_chunk['end_entity']}")
                print(f"Number of entities:   {len(first_chunk['entities'])}")
            elif 'num_faces' in first_chunk:
                print(f"Face range:           {first_chunk['start_face']} - {first_chunk['end_face']}")
                print(f"Number of faces:      {first_chunk['num_faces']}")

            # Show raw content preview
            raw_content = first_chunk['raw_content']
            if isinstance(raw_content, bytes):
                print(f"\nRaw content:          Binary data ({len(raw_content)} bytes)")
            else:
                preview = raw_content[:500]
                if len(raw_content) > 500:
                    preview += "\n... (truncated)"
                print(f"\nRaw content preview:\n{preview}")

            print(f"{'='*60}\n")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline(file_path: str):
    """Test the full LL-OCADR preprocessing pipeline."""
    print(f"\n{'='*60}")
    print(f"Testing Full LL-OCADR Pipeline")
    print(f"{'='*60}\n")

    try:
        from transformers import AutoTokenizer
        sys.path.insert(0, str(Path(__file__).parent / 'vllm'))
        from config import LLOCADRConfig
        from process.mesh_process import LLOCADRProcessor

        print("⚙️  Initializing components...")

        # Initialize config
        config = LLOCADRConfig()

        # Initialize tokenizer
        print(f"📝 Loading tokenizer: {config.language_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            config.language_model_name,
            trust_remote_code=True
        )

        # Add mesh token
        if config.mesh_token not in tokenizer.get_vocab():
            tokenizer.add_tokens([config.mesh_token])

        mesh_token_id = tokenizer.convert_tokens_to_ids(config.mesh_token)

        # Initialize processor with dynamic chunking
        processor = LLOCADRProcessor(
            tokenizer=tokenizer,
            mesh_token_id=mesh_token_id
            # chunk_size=None by default - uses dynamic analysis
        )

        print(f"✅ Components initialized\n")

        # Process mesh
        print(f"🔄 Processing mesh file: {Path(file_path).name}")
        conversation = f"{config.mesh_token}\nDescribe this CAD model."

        result = processor.tokenize_with_meshes(
            mesh_files=[file_path],
            conversation=conversation,
            cropping=True
        )

        print(f"\n✅ Processing complete!\n")
        print(f"{'='*60}")
        print(f"PIPELINE OUTPUT")
        print(f"{'='*60}")
        print(f"Input IDs shape:         {result['input_ids'].shape}")
        print(f"Vertex coords shape:     {result['vertex_coords'].shape}")
        print(f"Vertex normals shape:    {result['vertex_normals'].shape}")
        print(f"Chunks coords shape:     {result['chunks_coords'].shape}")
        print(f"Chunks normals shape:    {result['chunks_normals'].shape}")
        print(f"Spatial partition:       {result['mesh_spatial_partition']}")
        print(f"Num mesh tokens:         {result['num_mesh_tokens']}")
        print(f"{'='*60}\n")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Test LL-OCADR pipeline")
    parser.add_argument(
        "--file",
        type=str,
        help="Path to CAD/Mesh file (STEP, STL, OBJ)"
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["chunker", "pipeline", "all"],
        default="all",
        help="Which test to run"
    )

    args = parser.parse_args()

    if not args.file:
        print("❌ Please provide a file with --file <path>")
        print("\nExample:")
        print("  python test_ll_ocadr.py --file model.stl")
        print("  python test_ll_ocadr.py --file model.step --test chunker")
        return

    print("\n" + "="*60)
    print("LL-OCADR TEST SUITE")
    print("="*60)

    results = []

    if args.test in ["chunker", "all"]:
        results.append(("File Content Chunker", test_file_chunker(args.file)))

    if args.test in ["pipeline", "all"]:
        results.append(("Full Pipeline", test_full_pipeline(args.file)))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name:.<40} {status}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
