"""
Pre-train on completely raw, unsorted STEP files.
NO LABELS NEEDED - fully unsupervised!

Just point to a directory of STEP files and train.
"""

import sys
sys.path.insert(0, '../')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import json

from stepnet.tokenizer import STEPTokenizer
from stepnet.features import STEPFeatureExtractor
from stepnet.topology import STEPTopologyBuilder
from stepnet.pretrain import STEPForCausalLM, STEPForMaskedLM, mask_tokens


class RawSTEPDataset(Dataset):
    """
    Dataset for raw, unsorted STEP files with STEP-aware topology extraction.
    NO LABELS - just loads STEP files, tokenizes them, and extracts topology.
    """

    def __init__(self, step_files: list, tokenizer: STEPTokenizer, max_length: int = 2048,
                 use_topology: bool = True):
        self.step_files = step_files
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_topology = use_topology

        if use_topology:
            self.feature_extractor = STEPFeatureExtractor()
            self.topology_builder = STEPTopologyBuilder()

    def __len__(self):
        return len(self.step_files)

    def __getitem__(self, idx):
        step_file = self.step_files[idx]

        # Read STEP file
        with open(step_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Extract DATA section
        if 'DATA;' in content and 'ENDSEC;' in content:
            data_start = content.index('DATA;') + 5
            data_end = content.index('ENDSEC;', data_start)
            step_text = content[data_start:data_end].strip()
        else:
            step_text = content

        # Tokenize
        token_ids = self.tokenizer.encode(step_text)

        # Truncate or pad
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids.extend([0] * (self.max_length - len(token_ids)))

        result = {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'file_path': step_file
        }

        # Extract topology if enabled (STEP-aware!)
        if self.use_topology:
            try:
                # Extract features
                features_list = self.feature_extractor.extract_features_from_chunk(step_text)

                # Build topology
                topology_data = self.topology_builder.build_complete_topology(features_list)

                result['topology_data'] = topology_data
            except Exception as e:
                # If topology extraction fails, continue without it
                print(f"Warning: Failed to extract topology from {step_file}: {e}")
                result['topology_data'] = None

        return result


def collate_fn(batch):
    """
    Collate batch for pre-training with STEP topology.

    Note: Topology data is per-sample (different sizes), so we return a list of topology dicts.
    The model will process each topology individually during the forward pass.
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])

    # Collect topology data (can't batch different-sized graphs)
    topology_data_list = [item.get('topology_data') for item in batch]

    return {
        'input_ids': input_ids,
        'topology_data_list': topology_data_list
    }


def train_causal_lm(data_dir: str, output_dir: str, num_epochs: int = 3):
    """
    Train autoregressive (GPT-style) model on raw STEP files.

    Args:
        data_dir: Directory containing raw STEP files (no organization needed!)
        output_dir: Where to save checkpoints
        num_epochs: Number of training epochs
    """
    print("=" * 60)
    print("Pre-training: Causal Language Model (GPT-style)")
    print("Predicting next token in STEP files")
    print("=" * 60)

    # Find all STEP files recursively
    print(f"\nScanning {data_dir} for STEP files...")
    step_files = []
    for ext in ['*.step', '*.stp']:
        step_files.extend(list(Path(data_dir).rglob(ext)))

    step_files = [str(f) for f in step_files]
    print(f"Found {len(step_files)} STEP files")

    if len(step_files) == 0:
        print("ERROR: No STEP files found!")
        return

    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = STEPTokenizer()

    # Create dataset (NO LABELS!)
    print("Creating dataset...")
    dataset = RawSTEPDataset(step_files, tokenizer, max_length=2048)

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Initialize model
    print("Initializing model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = STEPForCausalLM(
        vocab_size=50000,
        embed_dim=512,
        num_layers=6,
        num_heads=8,
        max_length=2048,
        dropout=0.1
    )
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Training loop
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            topology_data_list = batch.get('topology_data_list', [None] * len(input_ids))

            # Labels are just the input shifted by 1
            labels = input_ids.clone()

            # Process each sample in the batch (topology data is per-sample)
            batch_loss = 0.0
            for i in range(len(input_ids)):
                sample_input = input_ids[i:i+1]
                sample_labels = labels[i:i+1]
                sample_topology = topology_data_list[i]

                # Forward pass with STEP topology
                outputs = model(sample_input, topology_data=sample_topology, labels=sample_labels)
                loss = outputs['loss']

                batch_loss += loss

            # Average loss over batch
            batch_loss = batch_loss / len(input_ids)

            # Backward pass
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # Metrics
            total_loss += batch_loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': batch_loss.item()})

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }
        torch.save(checkpoint, output_path / f'checkpoint_epoch_{epoch+1}.pt')

    print(f"\nTraining complete! Model saved to {output_dir}")


def train_masked_lm(data_dir: str, output_dir: str, num_epochs: int = 3):
    """
    Train masked language model (BERT-style) on raw STEP files.

    Args:
        data_dir: Directory containing raw STEP files
        output_dir: Where to save checkpoints
        num_epochs: Number of training epochs
    """
    print("=" * 60)
    print("Pre-training: Masked Language Model (BERT-style)")
    print("Predicting masked tokens from context")
    print("=" * 60)

    # Find all STEP files
    print(f"\nScanning {data_dir} for STEP files...")
    step_files = []
    for ext in ['*.step', '*.stp']:
        step_files.extend(list(Path(data_dir).rglob(ext)))

    step_files = [str(f) for f in step_files]
    print(f"Found {len(step_files)} STEP files")

    # Initialize tokenizer
    tokenizer = STEPTokenizer()

    # Create dataset
    dataset = RawSTEPDataset(step_files, tokenizer, max_length=2048)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = STEPForMaskedLM(
        vocab_size=50000,
        embed_dim=512,
        num_layers=6,
        num_heads=8,
        max_length=2048
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Training loop
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            topology_data_list = batch.get('topology_data_list', [None] * len(input_ids))

            # Mask tokens (15% random masking)
            masked_input, labels = mask_tokens(
                input_ids.clone(),
                mask_token_id=49999,  # vocab_size - 1
                vocab_size=50000,
                mask_prob=0.15
            )

            # Process each sample in the batch (topology data is per-sample)
            batch_loss = 0.0
            for i in range(len(input_ids)):
                sample_input = masked_input[i:i+1]
                sample_labels = labels[i:i+1]
                sample_topology = topology_data_list[i]

                # Forward pass with STEP topology
                outputs = model(sample_input, topology_data=sample_topology, labels=sample_labels)
                loss = outputs['loss']

                batch_loss += loss

            # Average loss over batch
            batch_loss = batch_loss / len(input_ids)

            # Backward pass
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': batch_loss.item()})

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, output_path / f'checkpoint_epoch_{epoch+1}.pt')

    print(f"\nTraining complete! Model saved to {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Pre-train on raw STEP files')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing STEP files (will search recursively)')
    parser.add_argument('--output_dir', type=str, default='checkpoints/pretrain',
                        help='Output directory for checkpoints')
    parser.add_argument('--task', type=str, choices=['causal', 'masked'], default='causal',
                        help='Pre-training task: causal (GPT) or masked (BERT)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')

    args = parser.parse_args()

    if args.task == 'causal':
        train_causal_lm(args.data_dir, args.output_dir, args.epochs)
    else:
        train_masked_lm(args.data_dir, args.output_dir, args.epochs)


if __name__ == '__main__':
    # Example usage:
    # python pretrain_unsupervised.py --data_dir /path/to/raw_step_files --task causal --epochs 5

    # For quick testing with existing test data:
    train_causal_lm(
        data_dir='../data/test_files',
        output_dir='../checkpoints/pretrain_test',
        num_epochs=2
    )
