"""
Example: Run inference on STEP files.
Demonstrates how to use trained models for prediction.
"""

import sys
sys.path.insert(0, '../')

import torch
from stepnet import STEPTokenizer, STEPFeatureExtractor, STEPTopologyBuilder
from stepnet.tasks import STEPForClassification, STEPForPropertyPrediction


def load_step_file(file_path: str, max_length: int = 2048):
    """Load and preprocess a single STEP file."""
    # Read file
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Extract DATA section
    if 'DATA;' in content and 'ENDSEC;' in content:
        data_start = content.index('DATA;') + 5
        data_end = content.index('ENDSEC;', data_start)
        chunk_text = content[data_start:data_end].strip()
    else:
        chunk_text = content

    # Tokenize
    tokenizer = STEPTokenizer()
    token_ids = tokenizer.encode(chunk_text)

    # Truncate/pad
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
    else:
        token_ids.extend([0] * (max_length - len(token_ids)))

    # Extract features and build topology
    extractor = STEPFeatureExtractor()
    features_list = extractor.extract_features_from_chunk(chunk_text)

    builder = STEPTopologyBuilder()
    topology_data = builder.build_complete_topology(features_list)

    return torch.tensor([token_ids], dtype=torch.long), topology_data


def classify_part(model_path: str, step_file: str, class_names: list):
    """
    Classify a STEP file into part category.

    Args:
        model_path: Path to trained classification model
        step_file: Path to STEP file
        class_names: List of class names
    """
    print(f"\nClassifying: {step_file}")

    # Load model
    model = STEPForClassification(num_classes=len(class_names))
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load and process STEP file
    token_ids, topology_data = load_step_file(step_file)

    # Run inference
    with torch.no_grad():
        logits = model(token_ids, topology_data=topology_data)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    # Print results
    print(f"Predicted class: {class_names[predicted_class]}")
    print(f"Confidence: {probabilities[0, predicted_class].item():.2%}")
    print("\nTop 3 predictions:")
    top3_probs, top3_indices = torch.topk(probabilities[0], k=3)
    for prob, idx in zip(top3_probs, top3_indices):
        print(f"  {class_names[idx]}: {prob.item():.2%}")


def predict_properties(model_path: str, step_file: str, property_names: list):
    """
    Predict physical properties from STEP file.

    Args:
        model_path: Path to trained property prediction model
        step_file: Path to STEP file
        property_names: List of property names
    """
    print(f"\nPredicting properties: {step_file}")

    # Load model
    model = STEPForPropertyPrediction(num_properties=len(property_names))
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load and process STEP file
    token_ids, topology_data = load_step_file(step_file)

    # Run inference
    with torch.no_grad():
        properties = model(token_ids, topology_data=topology_data)

    # Print results
    print("Predicted properties:")
    for name, value in zip(property_names, properties[0]):
        print(f"  {name}: {value.item():.4f}")


def main():
    # Example part categories
    class_names = [
        'bracket', 'housing', 'shaft', 'gear', 'plate',
        'connector', 'fastener', 'bearing', 'cover', 'frame'
    ]

    # Example properties
    property_names = [
        'volume_mm3', 'surface_area_mm2', 'mass_g',
        'bbox_x_mm', 'bbox_y_mm', 'bbox_z_mm'
    ]

    # Example STEP file
    step_file = 'examples/sample_part.step'

    print("=" * 60)
    print("STEP Inference Examples")
    print("=" * 60)

    # Classification example
    print("\n--- Part Classification ---")
    classify_part(
        model_path='checkpoints/classification/best_model.pt',
        step_file=step_file,
        class_names=class_names
    )

    # Property prediction example
    print("\n--- Property Prediction ---")
    predict_properties(
        model_path='checkpoints/property_prediction/best_model.pt',
        step_file=step_file,
        property_names=property_names
    )


if __name__ == '__main__':
    main()
