"""
Tests for LL-STEPNet using actual STEP files.
"""
import sys
from pathlib import Path
from datetime import datetime

import pytest

# Skip entire module if torch is not installed
torch = pytest.importorskip("torch")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stepnet import (
    STEPTokenizer,
    STEPFeatureExtractor,
    STEPTopologyBuilder,
    STEPEncoder,
    STEPForClassification,
    STEPForPropertyPrediction,
    STEPForSimilarity
)


# Output file for logging
LOG_FILE = None
LOG_PATH = None


def log_print(message="", end="\n"):
    """Print to both console and log file."""
    print(message, end=end)
    if LOG_FILE:
        LOG_FILE.write(message + end)
        LOG_FILE.flush()


# STEP test files
TEST_DIR = Path(__file__).resolve().parent.parent / "data" / "test_files"
TEST_FILES = [
    str(TEST_DIR / "4in_rod.step"),
    str(TEST_DIR / "Button.step"),
    str(TEST_DIR / "Pic adapter.step"),
    str(TEST_DIR / "Stock_Hinge.step"),
    str(TEST_DIR / "Butt pad.step"),
]


def read_step_file(file_path: str) -> str:
    """Read STEP file and extract DATA section."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Extract DATA section
    if 'DATA;' in content and 'ENDSEC;' in content:
        data_start = content.index('DATA;') + 5
        data_end = content.index('ENDSEC;', data_start)
        return content[data_start:data_end].strip()
    else:
        return content


def test_tokenizer_on_files():
    """Test tokenizer on actual STEP files."""
    log_print("\n" + "=" * 60)
    log_print("Testing STEPTokenizer on actual STEP files")
    log_print("=" * 60)

    tokenizer = STEPTokenizer()

    for file_path in TEST_FILES:
        log_print(f"\nProcessing: {Path(file_path).name}")

        # Read file
        step_content = read_step_file(file_path)
        lines = [line for line in step_content.split('\n') if line.strip()]
        log_print(f"  Lines in DATA section: {len(lines)}")

        # Tokenize first 10 entities
        sample = '\n'.join(lines[:10])
        tokens = tokenizer.tokenize(sample)
        token_ids = tokenizer.encode(sample)

        log_print(f"  Sample tokens (first 20): {tokens[:20]}")
        log_print(f"  Token IDs (first 20): {token_ids[:20]}")
        log_print(f"  Total tokens in sample: {len(tokens)}")
        log_print(f"  Vocab coverage: {len(set(token_ids))} unique tokens")

        # Verify no errors
        assert len(tokens) > 0, "Tokenization failed"
        assert len(token_ids) > 0, "Encoding failed"
        log_print("  ✓ PASSED")


def test_feature_extraction_on_files():
    """Test feature extraction on actual STEP files."""
    log_print("\n" + "=" * 60)
    log_print("Testing STEPFeatureExtractor on actual STEP files")
    log_print("=" * 60)

    extractor = STEPFeatureExtractor()

    for file_path in TEST_FILES:
        log_print(f"\nProcessing: {Path(file_path).name}")

        # Read file
        step_content = read_step_file(file_path)

        # Extract features from chunk
        features_list = extractor.extract_features_from_chunk(step_content)

        log_print(f"  Total entities extracted: {len(features_list)}")

        if len(features_list) > 0:
            # Show sample features
            sample_feature = features_list[0]
            log_print(f"  Sample entity:")
            log_print(f"    Entity ID: {sample_feature['entity_id']}")
            log_print(f"    Entity Type: {sample_feature['entity_type']}")
            log_print(f"    Numeric params: {sample_feature['numeric_params'][:5]}...")
            log_print(f"    References: {sample_feature['references'][:5]}...")

        # Count entity types
        entity_types = {}
        for feat in features_list:
            etype = feat['entity_type']
            entity_types[etype] = entity_types.get(etype, 0) + 1

        log_print(f"  Unique entity types: {len(entity_types)}")
        log_print(f"  Top 5 entity types:")
        for etype, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True)[:5]:
            log_print(f"    {etype}: {count}")

        assert len(features_list) > 0, "Feature extraction failed"
        log_print("  ✓ PASSED")


def test_topology_building_on_files():
    """Test topology building on actual STEP files."""
    log_print("\n" + "=" * 60)
    log_print("Testing STEPTopologyBuilder on actual STEP files")
    log_print("=" * 60)

    extractor = STEPFeatureExtractor()
    builder = STEPTopologyBuilder()

    for file_path in TEST_FILES:
        log_print(f"\nProcessing: {Path(file_path).name}")

        # Read file
        step_content = read_step_file(file_path)

        # Extract features
        features_list = extractor.extract_features_from_chunk(step_content)

        # Build topology
        topology = builder.build_complete_topology(features_list)

        log_print(f"  Topology statistics:")
        log_print(f"    Nodes: {topology['num_nodes']}")
        log_print(f"    Edges: {topology['num_edges']}")
        log_print(f"    Adjacency matrix shape: {topology['adjacency_matrix'].shape}")
        log_print(f"    Edge index shape: {topology['edge_index'].shape}")
        log_print(f"    Node features shape: {topology['node_features'].shape}")

        # Calculate graph density
        max_edges = topology['num_nodes'] * (topology['num_nodes'] - 1)
        if max_edges > 0:
            density = topology['num_edges'] / max_edges
            log_print(f"    Graph density: {density:.4f}")

        assert topology['num_nodes'] > 0, "No nodes in topology"
        assert 'adjacency_matrix' in topology, "Missing adjacency matrix"
        log_print("  ✓ PASSED")


def test_encoder_on_files():
    """Test encoder on actual STEP files."""
    log_print("\n" + "=" * 60)
    log_print("Testing STEPEncoder on actual STEP files")
    log_print("=" * 60)

    tokenizer = STEPTokenizer()
    extractor = STEPFeatureExtractor()
    builder = STEPTopologyBuilder()
    encoder = STEPEncoder(vocab_size=50000, output_dim=1024)
    encoder.eval()

    for file_path in TEST_FILES[:3]:  # Test on first 3 files for speed
        log_print(f"\nProcessing: {Path(file_path).name}")

        # Read file
        step_content = read_step_file(file_path)

        # Tokenize (truncate to max length)
        max_length = 2048
        token_ids = tokenizer.encode(step_content)
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            # Pad
            token_ids.extend([0] * (max_length - len(token_ids)))

        token_ids_tensor = torch.tensor([token_ids], dtype=torch.long)

        # Build topology
        features_list = extractor.extract_features_from_chunk(step_content)
        topology_data = builder.build_complete_topology(features_list)

        # Encode
        with torch.no_grad():
            output = encoder(token_ids_tensor, topology_data=topology_data)

        log_print(f"  Input shape: {token_ids_tensor.shape}")
        log_print(f"  Output shape: {output.shape}")
        log_print(f"  Output mean: {output.mean().item():.4f}")
        log_print(f"  Output std: {output.std().item():.4f}")

        assert output.shape == (1, 1024), f"Wrong output shape: {output.shape}"
        log_print("  ✓ PASSED")


def test_classification_on_files():
    """Test classification model on actual STEP files."""
    log_print("\n" + "=" * 60)
    log_print("Testing STEPForClassification on actual STEP files")
    log_print("=" * 60)

    tokenizer = STEPTokenizer()
    extractor = STEPFeatureExtractor()
    builder = STEPTopologyBuilder()

    num_classes = 5
    class_names = ['rod', 'button', 'adapter', 'hinge', 'pad']
    model = STEPForClassification(vocab_size=50000, num_classes=num_classes, output_dim=1024)
    model.eval()

    for file_path in TEST_FILES:
        log_print(f"\nProcessing: {Path(file_path).name}")

        # Read file
        step_content = read_step_file(file_path)

        # Tokenize
        max_length = 2048
        token_ids = tokenizer.encode(step_content)
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids.extend([0] * (max_length - len(token_ids)))

        token_ids_tensor = torch.tensor([token_ids], dtype=torch.long)

        # Build topology
        features_list = extractor.extract_features_from_chunk(step_content)
        topology_data = builder.build_complete_topology(features_list)

        # Classify
        with torch.no_grad():
            logits = model(token_ids_tensor, topology_data=topology_data)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

        log_print(f"  Predicted class: {class_names[predicted_class]} ({predicted_class})")
        log_print(f"  Confidence: {probabilities[0, predicted_class].item():.2%}")
        log_print(f"  All probabilities: {probabilities[0].tolist()}")

        assert logits.shape == (1, num_classes), f"Wrong output shape: {logits.shape}"
        assert torch.allclose(probabilities.sum(), torch.tensor(1.0)), "Probabilities don't sum to 1"
        log_print("  ✓ PASSED")


def test_property_prediction_on_files():
    """Test property prediction model on actual STEP files."""
    log_print("\n" + "=" * 60)
    log_print("Testing STEPForPropertyPrediction on actual STEP files")
    log_print("=" * 60)

    tokenizer = STEPTokenizer()
    extractor = STEPFeatureExtractor()
    builder = STEPTopologyBuilder()

    num_properties = 6
    property_names = ['volume', 'surface_area', 'mass', 'bbox_x', 'bbox_y', 'bbox_z']
    model = STEPForPropertyPrediction(vocab_size=50000, num_properties=num_properties, output_dim=1024)
    model.eval()

    for file_path in TEST_FILES[:3]:  # Test on first 3 for speed
        log_print(f"\nProcessing: {Path(file_path).name}")

        # Read file
        step_content = read_step_file(file_path)

        # Tokenize
        max_length = 2048
        token_ids = tokenizer.encode(step_content)
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids.extend([0] * (max_length - len(token_ids)))

        token_ids_tensor = torch.tensor([token_ids], dtype=torch.long)

        # Build topology
        features_list = extractor.extract_features_from_chunk(step_content)
        topology_data = builder.build_complete_topology(features_list)

        # Predict properties
        with torch.no_grad():
            properties = model(token_ids_tensor, topology_data=topology_data)

        log_print(f"  Predicted properties:")
        for name, value in zip(property_names, properties[0]):
            log_print(f"    {name}: {value.item():.4f}")

        assert properties.shape == (1, num_properties), f"Wrong output shape: {properties.shape}"
        log_print("  ✓ PASSED")


def test_similarity_on_files():
    """Test similarity model on actual STEP files."""
    log_print("\n" + "=" * 60)
    log_print("Testing STEPForSimilarity on actual STEP files")
    log_print("=" * 60)

    tokenizer = STEPTokenizer()
    extractor = STEPFeatureExtractor()
    builder = STEPTopologyBuilder()

    embedding_dim = 512
    model = STEPForSimilarity(vocab_size=50000, embedding_dim=embedding_dim)
    model.eval()

    embeddings_list = []

    # Get embeddings for all files
    for file_path in TEST_FILES:
        log_print(f"\nProcessing: {Path(file_path).name}")

        # Read file
        step_content = read_step_file(file_path)

        # Tokenize
        max_length = 2048
        token_ids = tokenizer.encode(step_content)
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids.extend([0] * (max_length - len(token_ids)))

        token_ids_tensor = torch.tensor([token_ids], dtype=torch.long)

        # Build topology
        features_list = extractor.extract_features_from_chunk(step_content)
        topology_data = builder.build_complete_topology(features_list)

        # Get embedding
        with torch.no_grad():
            embedding = model(token_ids_tensor, topology_data=topology_data)

        log_print(f"  Embedding shape: {embedding.shape}")
        log_print(f"  L2 norm: {torch.norm(embedding, p=2).item():.4f}")

        embeddings_list.append(embedding)

        assert embedding.shape == (1, embedding_dim), f"Wrong shape: {embedding.shape}"
        assert torch.allclose(torch.norm(embedding, p=2, dim=1), torch.ones(1)), "Not L2 normalized"
        log_print("  ✓ Embedding generated")

    # Compute similarity matrix
    log_print("\nComputing pairwise similarities...")
    all_embeddings = torch.cat(embeddings_list, dim=0)  # [5, 512]
    similarity_matrix = torch.matmul(all_embeddings, all_embeddings.T)  # [5, 5]

    log_print("\nSimilarity matrix:")
    log_print("         ", end="")
    for fp in TEST_FILES:
        log_print(f"{Path(fp).stem[:8]:>8s} ", end="")
    log_print()

    for i, fp in enumerate(TEST_FILES):
        log_print(f"{Path(fp).stem[:8]:>8s} ", end="")
        for j in range(len(TEST_FILES)):
            log_print(f"{similarity_matrix[i, j].item():8.4f} ", end="")
        log_print()

    # Verify diagonal is 1.0 (self-similarity)
    for i in range(len(TEST_FILES)):
        assert torch.allclose(similarity_matrix[i, i], torch.tensor(1.0), atol=1e-4), \
            f"Self-similarity not 1.0 at index {i}: {similarity_matrix[i, i].item()}"

    log_print("\n✓ All self-similarities are 1.0")
    log_print("✓ PASSED")


def main():
    global LOG_FILE, LOG_PATH

    # Create output log file
    output_dir = Path(__file__).parent.parent / 'test_results'
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_PATH = output_dir / f'test_results_{timestamp}.txt'
    LOG_FILE = open(LOG_PATH, 'w', encoding='utf-8')

    try:
        log_print("\n" + "=" * 60)
        log_print("LL-STEPNet Tests with Actual STEP Files")
        log_print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_print(f"Output file: {LOG_PATH}")
        log_print("=" * 60)
        log_print(f"\nTest files:")
        for fp in TEST_FILES:
            log_print(f"  - {Path(fp).name}")

        test_tokenizer_on_files()
        test_feature_extraction_on_files()
        test_topology_building_on_files()
        test_encoder_on_files()
        test_classification_on_files()
        test_property_prediction_on_files()
        test_similarity_on_files()

        log_print("\n" + "=" * 60)
        log_print("ALL TESTS PASSED! ✓")
        log_print("=" * 60)
        log_print(f"\nFull results saved to: {LOG_PATH}")

    finally:
        if LOG_FILE:
            LOG_FILE.close()
            print(f"\n✓ Test results saved to: {LOG_PATH}")


if __name__ == '__main__':
    main()
