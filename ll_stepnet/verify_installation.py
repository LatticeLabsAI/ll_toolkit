"""
Verification script to check LL-STEPNet installation.
Run this after installing to ensure everything works.
"""

import sys


def check_imports():
    """Check if all modules can be imported."""
    print("Checking imports...")

    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ PyTorch not found: {e}")
        return False

    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy not found: {e}")
        return False

    try:
        from stepnet import (
            STEPTokenizer,
            STEPFeatureExtractor,
            STEPTopologyBuilder,
            STEPEncoder,
            STEPForClassification,
            STEPForPropertyPrediction,
            STEPForSimilarity,
            STEPForCaptioning,
            STEPForQA,
            STEPDataset,
            STEPTrainer,
            create_dataloader
        )
        print("  ✓ All LL-STEPNet modules imported successfully")
    except ImportError as e:
        print(f"  ✗ LL-STEPNet import failed: {e}")
        return False

    return True


def check_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")

    try:
        from stepnet import STEPTokenizer

        tokenizer = STEPTokenizer()
        test_text = "#10=CARTESIAN_POINT('',(0.,0.,0.));"

        tokens = tokenizer.tokenize(test_text)
        token_ids = tokenizer.encode(test_text)

        assert len(tokens) > 0
        assert len(token_ids) > 0

        print("  ✓ Tokenizer works")
    except Exception as e:
        print(f"  ✗ Tokenizer test failed: {e}")
        return False

    try:
        from stepnet import STEPFeatureExtractor

        extractor = STEPFeatureExtractor()
        features = extractor.extract_geometric_features(test_text)

        assert 'entity_id' in features
        assert 'entity_type' in features

        print("  ✓ Feature extractor works")
    except Exception as e:
        print(f"  ✗ Feature extractor test failed: {e}")
        return False

    try:
        from stepnet import STEPTopologyBuilder

        chunk_text = """
        #10=CARTESIAN_POINT('',(0.,0.,0.));
        #11=DIRECTION('',(0.,0.,1.));
        #12=AXIS2_PLACEMENT_3D('',#10,#11,#13);
        """

        features_list = extractor.extract_features_from_chunk(chunk_text)
        builder = STEPTopologyBuilder()
        topology = builder.build_complete_topology(features_list)

        assert topology['num_nodes'] > 0
        assert 'adjacency_matrix' in topology

        print("  ✓ Topology builder works")
    except Exception as e:
        print(f"  ✗ Topology builder test failed: {e}")
        return False

    try:
        import torch
        from stepnet import STEPEncoder

        encoder = STEPEncoder(vocab_size=1000, output_dim=512)
        token_ids = torch.randint(0, 1000, (1, 128))

        output = encoder(token_ids)

        assert output.shape == (1, 512)

        print("  ✓ Encoder works")
    except Exception as e:
        print(f"  ✗ Encoder test failed: {e}")
        return False

    return True


def check_models():
    """Test task-specific models."""
    print("\nTesting task-specific models...")

    import torch

    token_ids = torch.randint(0, 1000, (1, 128))

    try:
        from stepnet import STEPForClassification

        model = STEPForClassification(vocab_size=1000, num_classes=5, output_dim=512)
        logits = model(token_ids)

        assert logits.shape == (1, 5)
        print("  ✓ Classification model works")
    except Exception as e:
        print(f"  ✗ Classification model test failed: {e}")
        return False

    try:
        from stepnet import STEPForPropertyPrediction

        model = STEPForPropertyPrediction(vocab_size=1000, num_properties=6, output_dim=512)
        properties = model(token_ids)

        assert properties.shape == (1, 6)
        print("  ✓ Property prediction model works")
    except Exception as e:
        print(f"  ✗ Property prediction model test failed: {e}")
        return False

    try:
        from stepnet import STEPForSimilarity

        model = STEPForSimilarity(vocab_size=1000, embedding_dim=256)
        embedding = model(token_ids)

        assert embedding.shape == (1, 256)
        print("  ✓ Similarity model works")
    except Exception as e:
        print(f"  ✗ Similarity model test failed: {e}")
        return False

    return True


def check_test_data():
    """Check if test data exists."""
    print("\nChecking test data...")

    from pathlib import Path

    data_dir = Path(__file__).parent / 'data' / 'test_files'

    if not data_dir.exists():
        print(f"  ⚠ Test data directory not found: {data_dir}")
        print("    Run tests with: python tests/test_with_step_files.py")
        return False

    step_files = list(data_dir.glob('*.step'))

    if len(step_files) == 0:
        print(f"  ⚠ No STEP files found in {data_dir}")
        return False

    print(f"  ✓ Found {len(step_files)} STEP test files:")
    for fp in step_files:
        print(f"      - {fp.name}")

    return True


def main():
    print("=" * 60)
    print("LL-STEPNet Installation Verification")
    print("=" * 60)

    all_passed = True

    # Check imports
    if not check_imports():
        all_passed = False
        print("\n⚠ Some imports failed. Run: pip install -e .")
    else:
        # Only check functionality if imports worked
        if not check_basic_functionality():
            all_passed = False

        if not check_models():
            all_passed = False

    # Check test data (optional)
    check_test_data()

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ Installation verification PASSED!")
        print("\nNext steps:")
        print("  - Run tests: python tests/test_with_step_files.py")
        print("  - See examples: ls examples/")
        print("  - Read docs: cat README.md")
    else:
        print("✗ Some checks FAILED!")
        print("\nTroubleshooting:")
        print("  1. Install package: pip install -e .")
        print("  2. Install dependencies: pip install torch numpy tqdm")
        print("  3. Check Python version (requires >=3.8)")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
